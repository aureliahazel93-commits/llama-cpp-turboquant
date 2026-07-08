#include "router-shadow-tree.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <queue>

static uint64_t hash_tokens(const llama_token * tokens, int32_t n) {
    uint64_t h = 14695981039346656037ULL;
    for (int32_t i = 0; i < n; i++) {
        h ^= (uint64_t)tokens[i];
        h *= 1099511628211ULL;
    }
    return h;
}

void shadow_tree_update(struct shadow_tree * tree,
                        const uint64_t * hashes,
                        const int32_t * lengths,
                        int32_t count) {
    for (int32_t i = 0; i < count; i++) {
        tree->prefix_hashes[hashes[i]] = lengths[i];
    }
    tree->n_entries = (int)tree->prefix_hashes.size();
    auto now = std::chrono::steady_clock::now();
    tree->last_sync_us = std::chrono::duration_cast<std::chrono::microseconds>(
        now.time_since_epoch()).count();
}

int32_t shadow_tree_score(const struct shadow_tree * tree,
                           const llama_token * tokens,
                           int32_t n_tokens) {
    int32_t best = 0;
    for (int32_t len = 1; len <= n_tokens; len = len < 4 ? len + 1 : len * 2) {
        if (len > n_tokens) len = n_tokens;
        uint64_t h = hash_tokens(tokens, len);
        auto it = tree->prefix_hashes.find(h);
        if (it != tree->prefix_hashes.end()) {
            best = it->second;
        }
    }
    return best;
}

int32_t route_cache_aware(const struct shadow_tree * trees,
                          int32_t n_workers,
                          const llama_token * prompt_tokens,
                          int32_t n_prompt) {
    int32_t best_worker = 0;
    int32_t best_score = -1;
    int32_t best_load = INT32_MAX;

    for (int32_t i = 0; i < n_workers; i++) {
        int32_t score = shadow_tree_score(&trees[i], prompt_tokens, n_prompt);
        int32_t load = trees[i].n_entries;
        if (score > best_score || (score == best_score && load < best_load)) {
            best_score = score;
            best_load = load;
            best_worker = i;
        }
    }

    return best_worker;
}
