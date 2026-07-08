#ifndef LLAMA_SPECULATIVE_SAGUARO_H
#define LLAMA_SPECULATIVE_SAGUARO_H

#include "llama.h"
#include <cstdint>
#include <vector>
#include <unordered_map>

#define SAGUARO_MAX_ENTRIES 1024
#define SAGUARO_CONTEXT_WINDOW 64

struct saguaro_cache_entry {
    std::vector<llama_token> context;
    std::vector<llama_token> draft_tokens;
    float acceptance_rate;
    int32_t lru_age;
};

struct llama_speculative_saguaro {
    std::unordered_map<uint64_t, saguaro_cache_entry> cache;
    saguaro_cache_entry * lru_order[SAGUARO_MAX_ENTRIES];
    int32_t lru_head = 0;
    int32_t n_entries;

    bool lookup(const llama_token * context, int32_t n_ctx,
                 llama_token * draft_out, int32_t * draft_len);
    void insert(const llama_token * context, int32_t n_ctx,
                const llama_token * draft, int32_t n_draft,
                float acceptance);
    void evict_oldest();
};

#endif
