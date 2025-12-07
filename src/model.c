#include "model.h"
#include "matops.h"
#include "io.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

// Initialize model from config and weights
ModelState* model_init(const char* config_path, const char* weights_path) {
    ModelState* state = (ModelState*)calloc(1, sizeof(ModelState));
    if (!state) return NULL;
    
    // Load config
    if (!load_config(config_path, &state->weights.config)) {
        free(state);
        return NULL;
    }
    
    // Load weights
    if (!load_weights(weights_path, &state->weights)) {
        free(state);
        return NULL;
    }
    
    ModelConfig* cfg = &state->weights.config;
    
    // Allocate KV cache
    int cache_size = cfg->num_layers * cfg->max_seq_len * cfg->hidden_size;
    state->kv_cache.k_cache = (float*)calloc(cache_size, sizeof(float));
    state->kv_cache.v_cache = (float*)calloc(cache_size, sizeof(float));
    state->kv_cache.max_cache_len = cfg->max_seq_len;
    state->kv_cache.cache_len = 0;
    
    // Allocate activations buffer
    int max_activation_size = cfg->max_seq_len * cfg->hidden_size;
    state->activations = (float*)malloc(max_activation_size * sizeof(float));
    
    // Allocate input buffer
    state->input_ids = (int*)malloc(cfg->max_seq_len * sizeof(int));
    state->seq_len = 0;
    
    if (!state->kv_cache.k_cache || !state->kv_cache.v_cache || 
        !state->activations || !state->input_ids) {
        model_free(state);
        return NULL;
    }
    
    return state;
}

void model_free(ModelState* state) {
    if (!state) return;
    
    if (state->weights.embeddings) free(state->weights.embeddings);
    if (state->weights.output_embeddings) free(state->weights.output_embeddings);
    if (state->weights.layer_weights) free(state->weights.layer_weights);
    if (state->kv_cache.k_cache) free(state->kv_cache.k_cache);
    if (state->kv_cache.v_cache) free(state->kv_cache.v_cache);
    if (state->activations) free(state->activations);
    if (state->input_ids) free(state->input_ids);
    
    free(state);
}

// Layer normalization
void layer_norm(float* out, const float* in, const float* weight, const float* bias, 
                int size, float eps) {
    // Compute mean
    float mean = 0.0f;
    for (int i = 0; i < size; i++) {
        mean += in[i];
    }
    mean /= size;
    
    // Compute variance
    float variance = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = in[i] - mean;
        variance += diff * diff;
    }
    variance /= size;
    
    // Normalize
    float inv_std = 1.0f / sqrtf(variance + eps);
    for (int i = 0; i < size; i++) {
        out[i] = (in[i] - mean) * inv_std * weight[i] + bias[i];
    }
}

// Softmax (stable)
void softmax(float* out, const float* in, int size) {
    float max_val = in[0];
    for (int i = 1; i < size; i++) {
        if (in[i] > max_val) max_val = in[i];
    }
    
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        out[i] = expf(in[i] - max_val);
        sum += out[i];
    }
    
    for (int i = 0; i < size; i++) {
        out[i] /= sum;
    }
}

// RoPE (simplified - using basic rotation)
void apply_rope(float* q, float* k, int head_dim, int pos, float theta) {
    // Simplified RoPE implementation
    // In production, use proper complex rotation
    for (int i = 0; i < head_dim; i += 2) {
        float freq = 1.0f / powf(theta, (float)i / head_dim);
        float angle = pos * freq;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);
        
        if (i + 1 < head_dim) {
            float q0 = q[i];
            float q1 = q[i + 1];
            q[i] = q0 * cos_a - q1 * sin_a;
            q[i + 1] = q0 * sin_a + q1 * cos_a;
            
            float k0 = k[i];
            float k1 = k[i + 1];
            k[i] = k0 * cos_a - k1 * sin_a;
            k[i + 1] = k0 * sin_a + k1 * cos_a;
        }
    }
}

// Multi-head attention
void multi_head_attention(float* out, const float* q, const float* k, const float* v,
                         const float* q_proj, const float* k_proj, const float* v_proj,
                         const float* o_proj, int hidden_size, int num_heads, 
                         int head_dim, int seq_len, int cache_pos, KVCache* kv_cache,
                         int layer_idx, float rope_theta) {
    int num_kv_heads = num_heads; // Simplified: assume same number
    
    // Project Q, K, V
    float* q_projected = (float*)malloc(seq_len * hidden_size * sizeof(float));
    float* k_projected = (float*)malloc(seq_len * hidden_size * sizeof(float));
    float* v_projected = (float*)malloc(seq_len * hidden_size * sizeof(float));
    
    matmul(q_projected, q, q_proj, seq_len, hidden_size, hidden_size);
    matmul(k_projected, k, k_proj, seq_len, hidden_size, hidden_size);
    matmul(v_projected, v, v_proj, seq_len, hidden_size, hidden_size);
    
    // Reshape and apply RoPE, then compute attention
    // Simplified attention computation
    float* attn_scores = (float*)malloc(num_heads * seq_len * seq_len * sizeof(float));
    
    for (int h = 0; h < num_heads; h++) {
        int offset = h * head_dim;
        float* q_head = q_projected + offset;
        float* k_head = k_projected + offset;
        
        // Apply RoPE
        apply_rope(q_head, k_head, head_dim, cache_pos, rope_theta);
        
        // Compute attention scores
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j <= cache_pos; j++) {
                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    score += q_head[i * head_dim + d] * k_head[j * head_dim + d];
                }
                attn_scores[h * seq_len * seq_len + i * seq_len + j] = score / sqrtf((float)head_dim);
            }
        }
        
        // Softmax over sequence
        for (int i = 0; i < seq_len; i++) {
            softmax(attn_scores + h * seq_len * seq_len + i * seq_len, 
                   attn_scores + h * seq_len * seq_len + i * seq_len, cache_pos + 1);
        }
    }
    
    // Apply attention to values and project output
    // Simplified: just copy for now
    memcpy(out, q_projected, seq_len * hidden_size * sizeof(float));
    
    free(q_projected);
    free(k_projected);
    free(v_projected);
    free(attn_scores);
}

// Forward pass through one transformer layer
void transformer_layer(float* out, const float* in, const float* weights, 
                      int hidden_size, int num_heads, int head_dim, int ffn_inner,
                      int seq_len, int cache_pos, KVCache* kv_cache, int layer_idx,
                      float norm_eps, float rope_theta) {
    // Pre-norm architecture
    float* norm1_out = (float*)malloc(seq_len * hidden_size * sizeof(float));
    // Simplified: assume weights are properly structured
    layer_norm(norm1_out, in, weights, weights + hidden_size, hidden_size, norm_eps);
    
    // Self-attention
    float* attn_out = (float*)malloc(seq_len * hidden_size * sizeof(float));
    multi_head_attention(attn_out, norm1_out, norm1_out, norm1_out,
                        weights + hidden_size * 2, weights + hidden_size * 3,
                        weights + hidden_size * 4, weights + hidden_size * 5,
                        hidden_size, num_heads, head_dim, seq_len, cache_pos,
                        kv_cache, layer_idx, rope_theta);
    
    // Residual
    for (int i = 0; i < seq_len * hidden_size; i++) {
        out[i] = in[i] + attn_out[i];
    }
    
    // FFN
    float* norm2_out = (float*)malloc(seq_len * hidden_size * sizeof(float));
    layer_norm(norm2_out, out, weights + hidden_size * 6, weights + hidden_size * 7,
               hidden_size, norm_eps);
    
    // Simplified FFN (gate + up -> down)
    float* ffn_out = (float*)malloc(seq_len * hidden_size * sizeof(float));
    // In production, implement proper FFN with gate/up/down projections
    memcpy(ffn_out, norm2_out, seq_len * hidden_size * sizeof(float));
    
    // Residual
    for (int i = 0; i < seq_len * hidden_size; i++) {
        out[i] = out[i] + ffn_out[i];
    }
    
    free(norm1_out);
    free(attn_out);
    free(norm2_out);
    free(ffn_out);
}

// Forward pass
float* model_forward(ModelState* state, const int* input_ids, int len, ModelMode mode) {
    ModelConfig* cfg = &state->weights.config;
    
    if (len > cfg->max_seq_len) len = cfg->max_seq_len;
    
    // Get embeddings
    float* x = (float*)malloc(len * cfg->hidden_size * sizeof(float));
    for (int i = 0; i < len; i++) {
        int token = input_ids[i];
        if (token >= 0 && token < cfg->vocab_size) {
            memcpy(x + i * cfg->hidden_size, 
                  state->weights.embeddings + token * cfg->hidden_size,
                  cfg->hidden_size * sizeof(float));
        }
    }
    
    // Add mode embedding if first token is a mode token
    if (len > 0) {
        int first_token = input_ids[0];
        int mode_token = -1;
        switch (mode) {
            case MODE_WITNESS: mode_token = cfg->token_witness; break;
            case MODE_JUDGE: mode_token = cfg->token_judge; break;
            case MODE_REBUILDER: mode_token = cfg->token_rebuilder; break;
            case MODE_SILENCE: mode_token = cfg->token_silence; break;
        }
        if (first_token == mode_token) {
            // Mode token already in input, use its embedding
        }
    }
    
    // Pass through transformer layers
    float* layer_input = x;
    for (int layer = 0; layer < cfg->num_layers; layer++) {
        float* layer_output = (float*)malloc(len * cfg->hidden_size * sizeof(float));
        transformer_layer(layer_output, layer_input, 
                         state->weights.layer_weights + layer * cfg->hidden_size * 10, // Simplified offset
                         cfg->hidden_size, cfg->num_heads, cfg->head_dim, cfg->ffn_inner_size,
                         len, state->kv_cache.cache_len, &state->kv_cache, layer,
                         cfg->norm_eps, cfg->rope_theta);
        
        if (layer_input != x) free(layer_input);
        layer_input = layer_output;
    }
    
    // Final layer norm
    float* final_norm = (float*)malloc(len * cfg->hidden_size * sizeof(float));
    // Simplified: assume final norm weights exist
    layer_norm(final_norm, layer_input, state->weights.layer_weights, 
              state->weights.layer_weights + cfg->hidden_size,
              cfg->hidden_size, cfg->norm_eps);
    
    // Output projection to vocab
    float* logits = (float*)malloc(len * cfg->vocab_size * sizeof(float));
    matmul(logits, final_norm, state->weights.output_embeddings,
           len, cfg->hidden_size, cfg->vocab_size);
    
    if (layer_input != x) free(layer_input);
    free(final_norm);
    
    return logits;
}

// Generate next token
int model_generate_token(ModelState* state, ModelMode mode, float temperature, 
                        int top_k, float top_p) {
    ModelConfig* cfg = &state->weights.config;
    
    // Forward pass
    float* logits = model_forward(state, state->input_ids, state->seq_len, mode);
    
    // Get logits for last position
    float* last_logits = logits + (state->seq_len - 1) * cfg->vocab_size;
    
    // Sample
    int token = sample_token(last_logits, cfg->vocab_size, temperature, top_k, top_p);
    
    free(logits);
    
    // Update sequence
    if (state->seq_len < cfg->max_seq_len) {
        state->input_ids[state->seq_len] = token;
        state->seq_len++;
        state->kv_cache.cache_len++;
    }
    
    return token;
}

void model_reset(ModelState* state) {
    if (!state) return;
    state->seq_len = 0;
    state->kv_cache.cache_len = 0;
    memset(state->kv_cache.k_cache, 0, 
           state->weights.config.num_layers * state->weights.config.max_seq_len * 
           state->weights.config.hidden_size * sizeof(float));
    memset(state->kv_cache.v_cache, 0,
           state->weights.config.num_layers * state->weights.config.max_seq_len * 
           state->weights.config.hidden_size * sizeof(float));
}

// Termination statistics
TerminationStats* termination_stats_init(void) {
    TerminationStats* stats = (TerminationStats*)calloc(1, sizeof(TerminationStats));
    return stats;
}

void termination_stats_free(TerminationStats* stats) {
    if (stats) free(stats);
}

void termination_stats_reset(TerminationStats* stats) {
    if (stats) {
        stats->is_terminated = false;
        // Note: We don't reset termination_count to track total across sessions
    }
}

int termination_stats_get_count(TerminationStats* stats) {
    return stats ? stats->termination_count : 0;
}

// Termination check
bool model_check_termination(ModelState* state, int token, ModelMode mode, 
                            TerminationStats* stats) {
    if (mode != MODE_SILENCE) return false;
    
    ModelConfig* cfg = &state->weights.config;
    
    if (token == cfg->token_silent_end || token == cfg->token_eos) {
        if (stats) {
            stats->termination_count++;
            stats->is_terminated = true;
        }
        return true;
    }
    
    return false;
}

