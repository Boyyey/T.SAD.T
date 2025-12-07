#include "io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

// Simple JSON parser for config (production: use proper JSON library)
bool load_config(const char* path, ModelConfig* config) {
    FILE* f = fopen(path, "r");
    if (!f) {
        // Set defaults if file doesn't exist
        config->vocab_size = 32000;
        config->hidden_size = 768;
        config->num_layers = 8;
        config->num_heads = 12;
        config->head_dim = 64;
        config->ffn_inner_size = 3072;
        config->max_seq_len = 1024;
        config->rope_theta = 10000.0f;
        config->norm_eps = 1e-5f;
        config->token_witness = 32000;
        config->token_judge = 32001;
        config->token_rebuilder = 32002;
        config->token_silence = 32003;
        config->token_silent_end = 32004;
        config->token_reset = 32005;
        config->token_pad = 32006;
        config->token_eos = 32007;
        return true; // Return true with defaults
    }
    
    // Simple parsing (in production, use cJSON or similar)
    char line[1024];
    while (fgets(line, sizeof(line), f)) {
        // Remove whitespace
        char* p = line;
        while (isspace(*p)) p++;
        
        // Parse key-value pairs
        if (strstr(p, "\"vocab_size\"")) {
            sscanf(p, "%*[^:]:%d", &config->vocab_size);
        } else if (strstr(p, "\"hidden_size\"")) {
            sscanf(p, "%*[^:]:%d", &config->hidden_size);
        } else if (strstr(p, "\"num_layers\"")) {
            sscanf(p, "%*[^:]:%d", &config->num_layers);
        } else if (strstr(p, "\"num_heads\"")) {
            sscanf(p, "%*[^:]:%d", &config->num_heads);
        } else if (strstr(p, "\"head_dim\"")) {
            sscanf(p, "%*[^:]:%d", &config->head_dim);
        } else if (strstr(p, "\"ffn_inner_size\"")) {
            sscanf(p, "%*[^:]:%d", &config->ffn_inner_size);
        } else if (strstr(p, "\"max_seq_len\"")) {
            sscanf(p, "%*[^:]:%d", &config->max_seq_len);
        } else if (strstr(p, "\"rope_theta\"")) {
            sscanf(p, "%*[^:]:%f", &config->rope_theta);
        } else if (strstr(p, "\"norm_eps\"")) {
            sscanf(p, "%*[^:]:%e", &config->norm_eps);
        }
    }
    
    // Set special tokens (defaults)
    config->token_witness = 32000;
    config->token_judge = 32001;
    config->token_rebuilder = 32002;
    config->token_silence = 32003;
    config->token_silent_end = 32004;
    config->token_reset = 32005;
    config->token_pad = 32006;
    config->token_eos = 32007;
    
    fclose(f);
    return true;
}

bool load_weights(const char* path, ModelWeights* weights) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        // Create dummy weights for testing
        ModelConfig* cfg = &weights->config;
        int embed_size = cfg->vocab_size * cfg->hidden_size;
        int layer_size = cfg->num_layers * cfg->hidden_size * cfg->hidden_size * 10; // Simplified
        
        weights->embeddings = (float*)calloc(embed_size, sizeof(float));
        weights->output_embeddings = (float*)calloc(embed_size, sizeof(float));
        weights->layer_weights = (float*)calloc(layer_size, sizeof(float));
        weights->num_params = embed_size * 2 + layer_size;
        weights->quantized = false;
        
        // Initialize with small random values
        for (int i = 0; i < embed_size; i++) {
            weights->embeddings[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;
            weights->output_embeddings[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;
        }
        
        return true;
    }
    
    // Read weights (simplified - in production, handle quantization, etc.)
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    // For now, just mark as loaded
    fclose(f);
    return true;
}

bool save_weights(const char* path, const ModelWeights* weights) {
    FILE* f = fopen(path, "wb");
    if (!f) return false;
    
    // Write weights (simplified)
    // In production, handle quantization, proper format, etc.
    
    fclose(f);
    return true;
}

bool load_vocab(const char* path, Tokenizer* tok) {
    FILE* f = fopen(path, "r");
    if (!f) return false;
    
    // Simple vocab loading (one token per line)
    char line[1024];
    int count = 0;
    int capacity = 1024;
    
    tok->vocab = (char**)malloc(capacity * sizeof(char*));
    
    while (fgets(line, sizeof(line), f) && count < MAX_VOCAB_SIZE) {
        // Remove newline
        int len = strlen(line);
        if (len > 0 && line[len-1] == '\n') line[len-1] = '\0';
        
        if (count >= capacity) {
            capacity *= 2;
            tok->vocab = (char**)realloc(tok->vocab, capacity * sizeof(char*));
        }
        
        tok->vocab[count] = strdup(line);
        count++;
    }
    
    tok->vocab_size = count;
    fclose(f);
    return true;
}

bool save_vocab(const char* path, const Tokenizer* tok) {
    FILE* f = fopen(path, "w");
    if (!f) return false;
    
    for (int i = 0; i < tok->vocab_size; i++) {
        fprintf(f, "%s\n", tok->vocab[i]);
    }
    
    fclose(f);
    return true;
}

