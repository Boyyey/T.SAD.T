#include "matops.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

// Naive matrix multiplication (can be optimized with SIMD/blocking)
void matmul(float* C, const float* A, const float* B, int m, int k, int n) {
    // Initialize output
    memset(C, 0, m * n * sizeof(float));
    
    // Naive implementation
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                sum += A[i * k + l] * B[l * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

void vec_add(float* out, const float* a, const float* b, int len) {
    for (int i = 0; i < len; i++) {
        out[i] = a[i] + b[i];
    }
}

void vec_scale(float* out, const float* in, float scale, int len) {
    for (int i = 0; i < len; i++) {
        out[i] = in[i] * scale;
    }
}

float vec_dot(const float* a, const float* b, int len) {
    float sum = 0.0f;
    for (int i = 0; i < len; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

// Top-k sampling
static int compare_floats(const void* a, const void* b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    if (fa < fb) return 1;
    if (fa > fb) return -1;
    return 0;
}

int sample_token(float* logits, int vocab_size, float temperature, int top_k, float top_p) {
    // Apply temperature
    if (temperature > 0.0f && temperature != 1.0f) {
        for (int i = 0; i < vocab_size; i++) {
            logits[i] /= temperature;
        }
    }
    
    // Softmax
    float max_logit = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }
    
    float* probs = (float*)malloc(vocab_size * sizeof(float));
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        probs[i] = expf(logits[i] - max_logit);
        sum += probs[i];
    }
    for (int i = 0; i < vocab_size; i++) {
        probs[i] /= sum;
    }
    
    // Top-k filtering
    if (top_k > 0 && top_k < vocab_size) {
        // Create indices array
        int* indices = (int*)malloc(vocab_size * sizeof(int));
        for (int i = 0; i < vocab_size; i++) indices[i] = i;
        
        // Sort by probability
        qsort(indices, vocab_size, sizeof(int), 
              (int (*)(const void*, const void*))compare_floats);
        
        // Zero out probabilities outside top-k
        for (int i = top_k; i < vocab_size; i++) {
            probs[indices[i]] = 0.0f;
        }
        
        // Renormalize
        sum = 0.0f;
        for (int i = 0; i < vocab_size; i++) sum += probs[i];
        if (sum > 0.0f) {
            for (int i = 0; i < vocab_size; i++) probs[i] /= sum;
        }
        
        free(indices);
    }
    
    // Top-p (nucleus) filtering
    if (top_p > 0.0f && top_p < 1.0f) {
        // Sort probabilities
        int* indices = (int*)malloc(vocab_size * sizeof(int));
        for (int i = 0; i < vocab_size; i++) indices[i] = i;
        qsort(indices, vocab_size, sizeof(int),
              (int (*)(const void*, const void*))compare_floats);
        
        float cumsum = 0.0f;
        int cutoff = vocab_size;
        for (int i = 0; i < vocab_size; i++) {
            cumsum += probs[indices[i]];
            if (cumsum >= top_p) {
                cutoff = i + 1;
                break;
            }
        }
        
        // Zero out probabilities outside top-p
        for (int i = cutoff; i < vocab_size; i++) {
            probs[indices[i]] = 0.0f;
        }
        
        // Renormalize
        sum = 0.0f;
        for (int i = 0; i < vocab_size; i++) sum += probs[i];
        if (sum > 0.0f) {
            for (int i = 0; i < vocab_size; i++) probs[i] /= sum;
        }
        
        free(indices);
    }
    
    // Sample from distribution
    float r = (float)rand() / (float)RAND_MAX;
    float cumsum = 0.0f;
    int selected = 0;
    
    for (int i = 0; i < vocab_size; i++) {
        cumsum += probs[i];
        if (r <= cumsum) {
            selected = i;
            break;
        }
    }
    
    free(probs);
    return selected;
}

