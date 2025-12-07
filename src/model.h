#ifndef MODEL_H
#define MODEL_H

#include <stdint.h>
#include <stdbool.h>

#define MAX_SEQ_LEN 1024
#define MAX_VOCAB_SIZE 32768

// Model configuration
typedef struct {
    int vocab_size;
    int hidden_size;
    int num_layers;
    int num_heads;
    int head_dim;
    int ffn_inner_size;
    int max_seq_len;
    float norm_eps;
    float rope_theta;
    
    // Special tokens
    int token_witness;
    int token_judge;
    int token_rebuilder;
    int token_silence;
    int token_silent_end;
    int token_reset;
    int token_pad;
    int token_eos;
} ModelConfig;

// Model weights (quantized)
typedef struct {
    ModelConfig config;
    float* embeddings;           // [vocab_size, hidden_size]
    float* output_embeddings;    // [vocab_size, hidden_size]
    float* layer_weights;        // All layer weights (packed)
    int num_params;
    bool quantized;
} ModelWeights;

// KV Cache for efficient generation
typedef struct {
    float* k_cache;  // [num_layers, max_seq_len, hidden_size]
    float* v_cache;  // [num_layers, max_seq_len, hidden_size]
    int cache_len;
    int max_cache_len;
} KVCache;

// Model state
typedef struct {
    ModelWeights weights;
    KVCache kv_cache;
    float* activations;  // Temporary activations
    int* input_ids;
    int seq_len;
} ModelState;

// Mode enum
typedef enum {
    MODE_WITNESS = 0,
    MODE_JUDGE = 1,
    MODE_REBUILDER = 2,
    MODE_SILENCE = 3
} ModelMode;

// Self-termination statistics
typedef struct {
    int termination_count;
    int total_silence_attempts;
    bool is_terminated;
} TerminationStats;

// Initialize model
ModelState* model_init(const char* config_path, const char* weights_path);
void model_free(ModelState* state);

// Forward pass
float* model_forward(ModelState* state, const int* input_ids, int len, ModelMode mode);

// Generation
int model_generate_token(ModelState* state, ModelMode mode, float temperature, int top_k, float top_p);
void model_reset(ModelState* state);

// Mode control
void model_set_mode(ModelState* state, ModelMode mode);
ModelMode model_get_mode(ModelState* state);

// Termination tracking
TerminationStats* termination_stats_init(void);
void termination_stats_free(TerminationStats* stats);
void termination_stats_reset(TerminationStats* stats);
int termination_stats_get_count(TerminationStats* stats);
bool model_check_termination(ModelState* state, int token, ModelMode mode, TerminationStats* stats);

#endif // MODEL_H

