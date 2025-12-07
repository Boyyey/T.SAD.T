#ifndef SAFETY_H
#define SAFETY_H

#include <stdbool.h>

// Content filter: check if text should be blocked
bool safety_check(const char* text);

// Check for blocked phrases
bool contains_blocked_phrase(const char* text);

// Add content warning if needed
bool needs_content_warning(const char* text);

// Rate limiting
typedef struct {
    int request_count;
    long last_request_time;
    int max_requests_per_minute;
} RateLimiter;

RateLimiter* rate_limiter_init(int max_per_minute);
void rate_limiter_free(RateLimiter* limiter);
bool rate_limiter_check(RateLimiter* limiter);

#endif // SAFETY_H

