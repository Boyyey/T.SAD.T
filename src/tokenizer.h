#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stdint.h>
#include <stdbool.h>

#define MAX_TOKEN_LEN 256
#define MAX_VOCAB_SIZE 32768

typedef struct {
    char** vocab;
    int vocab_size;
    int* token_ids;
    bool loaded;
} Tokenizer;

// Initialize tokenizer
Tokenizer* tokenizer_init(const char* vocab_path);
void tokenizer_free(Tokenizer* tok);

// Encode text to token IDs
int* tokenizer_encode(Tokenizer* tok, const char* text, int* out_len);
int* tokenizer_encode_with_mode(Tokenizer* tok, const char* text, int mode_token, int* out_len);

// Decode token IDs to text
char* tokenizer_decode(Tokenizer* tok, const int* token_ids, int len);

// Get special token ID
int tokenizer_get_special_token(Tokenizer* tok, const char* token_name);

// Simple BPE-like tokenization (fallback if no vocab file)
int* simple_tokenize(const char* text, int* out_len);

#endif // TOKENIZER_H

