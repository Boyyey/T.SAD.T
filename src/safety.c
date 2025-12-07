#include "safety.h"
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <ctype.h>

// Blocked phrases (simplified - in production, use more sophisticated filtering)
static const char* blocked_phrases[] = {
    // Add specific blocked phrases here
    NULL
};

bool contains_blocked_phrase(const char* text) {
    if (!text) return false;
    
    // Convert to lowercase for comparison
    char* lower = (char*)malloc(strlen(text) + 1);
    for (int i = 0; text[i]; i++) {
        lower[i] = tolower(text[i]);
    }
    lower[strlen(text)] = '\0';
    
    // Check against blocked phrases
    for (int i = 0; blocked_phrases[i]; i++) {
        if (strstr(lower, blocked_phrases[i])) {
            free(lower);
            return true;
        }
    }
    
    free(lower);
    return false;
}

bool needs_content_warning(const char* text) {
    // Check for sensitive keywords
    const char* sensitive_keywords[] = {
        "atrocity", "genocide", "massacre", "torture", "suffering",
        NULL
    };
    
    if (!text) return false;
    
    char* lower = (char*)malloc(strlen(text) + 1);
    for (int i = 0; text[i]; i++) {
        lower[i] = tolower(text[i]);
    }
    lower[strlen(text)] = '\0';
    
    for (int i = 0; sensitive_keywords[i]; i++) {
        if (strstr(lower, sensitive_keywords[i])) {
            free(lower);
            return true;
        }
    }
    
    free(lower);
    return false;
}

bool safety_check(const char* text) {
    // Block if contains blocked phrases
    if (contains_blocked_phrase(text)) {
        return false;
    }
    
    return true;
}

RateLimiter* rate_limiter_init(int max_per_minute) {
    RateLimiter* limiter = (RateLimiter*)calloc(1, sizeof(RateLimiter));
    if (!limiter) return NULL;
    
    limiter->max_requests_per_minute = max_per_minute;
    limiter->last_request_time = time(NULL);
    limiter->request_count = 0;
    
    return limiter;
}

void rate_limiter_free(RateLimiter* limiter) {
    if (limiter) free(limiter);
}

bool rate_limiter_check(RateLimiter* limiter) {
    if (!limiter) return true;
    
    long current_time = time(NULL);
    
    // Reset if a minute has passed
    if (current_time - limiter->last_request_time >= 60) {
        limiter->request_count = 0;
        limiter->last_request_time = current_time;
    }
    
    // Check limit
    if (limiter->request_count >= limiter->max_requests_per_minute) {
        return false;
    }
    
    limiter->request_count++;
    return true;
}

