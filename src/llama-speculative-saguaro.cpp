#include "llama-speculative-saguaro.h"
#include <cstring>
#include <algorithm>
#include <limits>

static uint64_t saguaro_hash(const llama_token * ctx, int32_t n) {
    int32_t start = n - SAGUARO_CONTEXT_WINDOW;
    if (start < 0) start = 0;
    uint64_t h = 2166136261ull;
    for (int32_t i = start; i < n; i++) {
        h ^= (uint64_t)ctx[i];
        h *= 16777619ull;
    }
    return h;
}

bool llama_speculative_saguaro::lookup(const llama_token * context, int32_t n_ctx,
                                       llama_token * draft_out, int32_t * draft_len) {
    uint64_t key = saguaro_hash(context, n_ctx);
    auto it = cache.find(key);
    if (it == cache.end()) {
        *draft_len = 0;
        return false;
    }
    saguaro_cache_entry & e = it->second;
    int32_t out = std::min((int32_t)e.draft_tokens.size(), *draft_len);
    memcpy(draft_out, e.draft_tokens.data(), out * sizeof(llama_token));
    *draft_len = out;
    e.lru_age = --lru_head;
    return true;
}

void llama_speculative_saguaro::insert(const llama_token * context, int32_t n_ctx,
                                        const llama_token * draft, int32_t n_draft,
                                        float acceptance) {
    uint64_t key = saguaro_hash(context, n_ctx);
    if (n_entries >= SAGUARO_MAX_ENTRIES) {
        evict_oldest();
    }
    saguaro_cache_entry e;
    int32_t start = n_ctx - SAGUARO_CONTEXT_WINDOW;
    if (start < 0) start = 0;
    e.context.assign(context + start, context + n_ctx);
    e.draft_tokens.assign(draft, draft + n_draft);
    e.acceptance_rate = acceptance;
    e.lru_age = --lru_head;
    auto [it, inserted] = cache.emplace(key, std::move(e));
    if (!inserted) {
        it->second.draft_tokens.assign(draft, draft + n_draft);
        it->second.acceptance_rate = acceptance;
        it->second.lru_age = --lru_head;
    } else {
        n_entries++;
    }
}

void llama_speculative_saguaro::evict_oldest() {
    int32_t oldest_age = 0;
    uint64_t oldest_key = 0;
    for (auto & [k, v] : cache) {
        if (v.lru_age < oldest_age) {
            oldest_age = v.lru_age;
            oldest_key = k;
        }
    }
    cache.erase(oldest_key);
    n_entries--;
}
