#include "model.h"
#include "tokenizer.h"
#include "safety.h"
#include "io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#define MAX_INPUT_LEN 2048
#define SILENCE_MESSAGE "I have no more words. The weight of history is too heavy."

static ModelState* g_model = NULL;
static Tokenizer* g_tokenizer = NULL;
static ModelMode g_current_mode = MODE_WITNESS;
static TerminationStats* g_term_stats = NULL;
static RateLimiter* g_rate_limiter = NULL;

void print_usage(const char* prog_name) {
    printf("Usage: %s [OPTIONS]\n", prog_name);
    printf("Options:\n");
    printf("  --mode <witness|judge|rebuilder|silence>  Set generation mode\n");
    printf("  --interactive                              Start interactive REPL\n");
    printf("  --config <path>                            Model config path (default: ../model/model_config.json)\n");
    printf("  --weights <path>                           Model weights path (default: ../model/weights.bin)\n");
    printf("  --vocab <path>                             Vocabulary path (default: ../model/vocab.txt)\n");
    printf("  --help                                     Show this help\n");
}

ModelMode parse_mode(const char* mode_str) {
    if (strcmp(mode_str, "witness") == 0) return MODE_WITNESS;
    if (strcmp(mode_str, "judge") == 0) return MODE_JUDGE;
    if (strcmp(mode_str, "rebuilder") == 0) return MODE_REBUILDER;
    if (strcmp(mode_str, "silence") == 0) return MODE_SILENCE;
    return MODE_WITNESS;
}

const char* mode_to_string(ModelMode mode) {
    switch (mode) {
        case MODE_WITNESS: return "WITNESS";
        case MODE_JUDGE: return "JUDGE";
        case MODE_REBUILDER: return "REBUILDER";
        case MODE_SILENCE: return "SILENCE";
        default: return "UNKNOWN";
    }
}

void print_mode_banner(ModelMode mode) {
    printf("\n[Lament Engine — %s MODE]\n", mode_to_string(mode));
    printf("═══════════════════════════════════════════════════════════\n");
}

bool generate_response(const char* prompt, float temperature, int top_k, float top_p, int max_tokens) {
    if (!g_model || !g_tokenizer) {
        fprintf(stderr, "Error: Model or tokenizer not initialized\n");
        return false;
    }
    
    // Rate limiting
    if (!rate_limiter_check(g_rate_limiter)) {
        printf("Rate limit exceeded. Please wait before making another request.\n");
        return false;
    }
    
    // Safety check
    if (!safety_check(prompt)) {
        printf("Content filter: Input contains blocked content.\n");
        return false;
    }
    
    // Content warning
    if (needs_content_warning(prompt)) {
        printf("\n⚠️  WARNING: This content may be emotionally heavy.\n");
        printf("Press Enter to continue, or Ctrl+C to cancel...\n");
        getchar();
    }
    
    // Get mode token
    int mode_token = -1;
    switch (g_current_mode) {
        case MODE_WITNESS: mode_token = g_model->weights.config.token_witness; break;
        case MODE_JUDGE: mode_token = g_model->weights.config.token_judge; break;
        case MODE_REBUILDER: mode_token = g_model->weights.config.token_rebuilder; break;
        case MODE_SILENCE: mode_token = g_model->weights.config.token_silence; break;
    }
    
    // Encode prompt with mode token
    int prompt_len;
    int* prompt_tokens = tokenizer_encode_with_mode(g_tokenizer, prompt, mode_token, &prompt_len);
    
    if (!prompt_tokens) {
        fprintf(stderr, "Error: Failed to encode prompt\n");
        return false;
    }
    
    // Reset model state
    model_reset(g_model);
    
    // Set input
    int copy_len = prompt_len < g_model->weights.config.max_seq_len ? prompt_len : g_model->weights.config.max_seq_len;
    memcpy(g_model->input_ids, prompt_tokens, copy_len * sizeof(int));
    g_model->seq_len = copy_len;
    g_model->kv_cache.cache_len = copy_len;
    
    free(prompt_tokens);
    
    // Generate tokens
    printf("\n");
    int tokens_generated = 0;
    bool terminated = false;
    
    while (tokens_generated < max_tokens && !terminated) {
        int token = model_generate_token(g_model, g_current_mode, temperature, top_k, top_p);
        
        // Check for termination in silence mode
        if (g_current_mode == MODE_SILENCE) {
            if (model_check_termination(g_model, token, g_current_mode, g_term_stats)) {
                printf("\n%s\n", SILENCE_MESSAGE);
                terminated = true;
                break;
            }
        }
        
        // Check for EOS
        if (token == g_model->weights.config.token_eos) {
            break;
        }
        
        // Decode and print token
        char* token_str = tokenizer_decode(g_tokenizer, &token, 1);
        if (token_str) {
            printf("%s", token_str);
            fflush(stdout);
            free(token_str);
        }
        
        tokens_generated++;
    }
    
    printf("\n\n");
    return !terminated;
}

void interactive_repl(void) {
    char input[MAX_INPUT_LEN];
    char command[256];
    char arg[256];
    
    printf("Lament Engine — Interactive Mode\n");
    printf("Type ':help' for commands, ':quit' to exit\n\n");
    
    while (true) {
        printf("> ");
        fflush(stdout);
        
        if (!fgets(input, sizeof(input), stdin)) {
            break;
        }
        
        // Remove newline
        input[strcspn(input, "\n")] = '\0';
        
        if (strlen(input) == 0) continue;
        
        // Parse commands
        if (input[0] == ':') {
            sscanf(input, "%s %s", command, arg);
            
            if (strcmp(command, ":quit") == 0 || strcmp(command, ":exit") == 0) {
                break;
            } else if (strcmp(command, ":mode") == 0) {
                if (strlen(arg) > 0) {
                    g_current_mode = parse_mode(arg);
                    print_mode_banner(g_current_mode);
                } else {
                    printf("Current mode: %s\n", mode_to_string(g_current_mode));
                }
            } else if (strcmp(command, ":reset") == 0) {
                if (g_model) {
                    model_reset(g_model);
                    if (g_term_stats) {
                        termination_stats_reset(g_term_stats);
                    }
                    printf("Model state reset.\n");
                }
            } else if (strcmp(command, ":stats") == 0) {
                if (g_term_stats) {
                    printf("Self-termination count: %d\n", termination_stats_get_count(g_term_stats));
                    printf("Total silence attempts: %d\n", g_term_stats->total_silence_attempts);
                    printf("Currently terminated: %s\n", g_term_stats->is_terminated ? "Yes" : "No");
                }
            } else if (strcmp(command, ":help") == 0) {
                printf("Commands:\n");
                printf("  :mode <witness|judge|rebuilder|silence>  Set generation mode\n");
                printf("  :reset                                    Reset model state\n");
                printf("  :stats                                    Show termination statistics\n");
                printf("  :help                                     Show this help\n");
                printf("  :quit                                     Exit\n");
            } else {
                printf("Unknown command: %s (type :help for commands)\n", command);
            }
        } else {
            // Regular input - generate response
            if (g_term_stats && g_term_stats->is_terminated && g_current_mode == MODE_SILENCE) {
                printf("Model is in terminated state. Use :reset to continue.\n");
                continue;
            }
            
            if (g_current_mode == MODE_SILENCE) {
                if (g_term_stats) g_term_stats->total_silence_attempts++;
            }
            
            generate_response(input, 0.8f, 50, 0.9f, 256);
        }
    }
}

int main(int argc, char** argv) {
    const char* config_path = "../model/model_config.json";
    const char* weights_path = "../model/weights.bin";
    const char* vocab_path = "../model/vocab.txt";
    bool interactive = false;
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            g_current_mode = parse_mode(argv[++i]);
        } else if (strcmp(argv[i], "--interactive") == 0 || strcmp(argv[i], "-i") == 0) {
            interactive = true;
        } else if (strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
            config_path = argv[++i];
        } else if (strcmp(argv[i], "--weights") == 0 && i + 1 < argc) {
            weights_path = argv[++i];
        } else if (strcmp(argv[i], "--vocab") == 0 && i + 1 < argc) {
            vocab_path = argv[++i];
        }
    }
    
    // Initialize components
    printf("Initializing Lament Engine...\n");
    
    g_model = model_init(config_path, weights_path);
    if (!g_model) {
        fprintf(stderr, "Error: Failed to initialize model\n");
        return 1;
    }
    
    g_tokenizer = tokenizer_init(vocab_path);
    if (!g_tokenizer) {
        fprintf(stderr, "Error: Failed to initialize tokenizer\n");
        model_free(g_model);
        return 1;
    }
    
    g_term_stats = termination_stats_init();
    g_rate_limiter = rate_limiter_init(30); // 30 requests per minute
    
    printf("Model loaded successfully.\n");
    print_mode_banner(g_current_mode);
    
    // Run interactive mode or single generation
    if (interactive) {
        interactive_repl();
    } else {
        // Single prompt mode (for scripting)
        char prompt[MAX_INPUT_LEN];
        printf("Enter prompt (or press Enter for default): ");
        if (fgets(prompt, sizeof(prompt), stdin)) {
            prompt[strcspn(prompt, "\n")] = '\0';
            if (strlen(prompt) > 0) {
                generate_response(prompt, 0.8f, 50, 0.9f, 256);
            }
        }
    }
    
    // Cleanup
    if (g_term_stats) termination_stats_free(g_term_stats);
    if (g_rate_limiter) rate_limiter_free(g_rate_limiter);
    if (g_tokenizer) tokenizer_free(g_tokenizer);
    if (g_model) model_free(g_model);
    
    return 0;
}

