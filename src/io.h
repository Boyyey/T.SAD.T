#ifndef IO_H
#define IO_H

#include "model.h"
#include "tokenizer.h"
#include <stdbool.h>

// Load model configuration from JSON
bool load_config(const char* path, ModelConfig* config);

// Load model weights from binary file
bool load_weights(const char* path, ModelWeights* weights);

// Save model weights to binary file
bool save_weights(const char* path, const ModelWeights* weights);

// Load vocabulary file
bool load_vocab(const char* path, Tokenizer* tok);

// Save vocabulary file
bool save_vocab(const char* path, const Tokenizer* tok);

#endif // IO_H

