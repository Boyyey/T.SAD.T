#include "tokenizer.h"
#include "io.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

Tokenizer* tokenizer_init(const char* vocab_path) {
    Tokenizer* tok = (Tokenizer*)calloc(1, sizeof(Tokenizer));
    if (!tok) return NULL;
    
    // Try to load vocab file
    if (vocab_path && load_vocab(vocab_path, tok)) {
        tok->loaded = true;
        return tok;
    }
    
    // Fallback: create simple tokenizer
    tok->vocab_size = 256; // Byte-level
    tok->vocab = (char**)malloc(tok->vocab_size * sizeof(char*));
    for (int i = 0; i < 256; i++) {
        tok->vocab[i] = (char*)malloc(2);
        tok->vocab[i][0] = (char)i;
        tok->vocab[i][1] = '\0';
    }
    tok->loaded = true;
    
    return tok;
}

void tokenizer_free(Tokenizer* tok) {
    if (!tok) return;
    
    if (tok->vocab) {
        for (int i = 0; i < tok->vocab_size; i++) {
            if (tok->vocab[i]) free(tok->vocab[i]);
        }
        free(tok->vocab);
    }
    
    if (tok->token_ids) free(tok->token_ids);
    free(tok);
}

int* tokenizer_encode(Tokenizer* tok, const char* text, int* out_len) {
    if (!tok || !text) {
        *out_len = 0;
        return NULL;
    }
    
    // Simple byte-level encoding (fallback)
    int len = strlen(text);
    int* tokens = (int*)malloc(len * sizeof(int));
    
    for (int i = 0; i < len; i++) {
        tokens[i] = (unsigned char)text[i];
    }
    
    *out_len = len;
    return tokens;
}

int* tokenizer_encode_with_mode(Tokenizer* tok, const char* text, int mode_token, int* out_len) {
    int text_len;
    int* text_tokens = tokenizer_encode(tok, text, &text_len);
    
    if (!text_tokens) {
        *out_len = 0;
        return NULL;
    }
    
    // Prepend mode token
    int* tokens = (int*)malloc((text_len + 1) * sizeof(int));
    tokens[0] = mode_token;
    memcpy(tokens + 1, text_tokens, text_len * sizeof(int));
    
    free(text_tokens);
    *out_len = text_len + 1;
    return tokens;
}

char* tokenizer_decode(Tokenizer* tok, const int* token_ids, int len) {
    if (!tok || !token_ids || len == 0) {
        return strdup("");
    }
    
    // Simple byte-level decoding
    char* text = (char*)malloc(len + 1);
    for (int i = 0; i < len; i++) {
        if (token_ids[i] >= 0 && token_ids[i] < 256) {
            text[i] = (char)token_ids[i];
        } else {
            text[i] = '?';
        }
    }
    text[len] = '\0';
    
    return text;
}

int tokenizer_get_special_token(Tokenizer* tok, const char* token_name) {
    // Map special token names to IDs
    // These should match model_config.json
    if (strcmp(token_name, "witness") == 0) return 32000;
    if (strcmp(token_name, "judge") == 0) return 32001;
    if (strcmp(token_name, "rebuilder") == 0) return 32002;
    if (strcmp(token_name, "silence") == 0) return 32003;
    if (strcmp(token_name, "silent_end") == 0) return 32004;
    if (strcmp(token_name, "reset") == 0) return 32005;
    if (strcmp(token_name, "pad") == 0) return 32006;
    if (strcmp(token_name, "eos") == 0) return 32007;
    
    return -1;
}

int* simple_tokenize(const char* text, int* out_len) {
    int len = strlen(text);
    int* tokens = (int*)malloc(len * sizeof(int));
    
    for (int i = 0; i < len; i++) {
        tokens[i] = (unsigned char)text[i];
    }
    
    *out_len = len;
    return tokens;
}

