#ifndef MATOPS_H
#define MATOPS_H

#include <stdint.h>

// Matrix multiplication: C = A * B
// A: [m, k], B: [k, n], C: [m, n]
void matmul(float* C, const float* A, const float* B, int m, int k, int n);

// Vector operations
void vec_add(float* out, const float* a, const float* b, int len);
void vec_scale(float* out, const float* in, float scale, int len);
float vec_dot(const float* a, const float* b, int len);

// Sampling
int sample_token(float* logits, int vocab_size, float temperature, int top_k, float top_p);

#endif // MATOPS_H

