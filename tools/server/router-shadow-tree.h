#ifndef ROUTER_SHADOW_TREE_H
#define ROUTER_SHADOW_TREE_H

#include "llama.h"
#include <cstdint>
#include <vector>
#include <unordered_map>

struct shadow_tree {
    int worker_id;
    int n_entries;
    uint64_t last_sync_us;
    std::unordered_map<uint64_t, int32_t> prefix_hashes;
};

void shadow_tree_update(struct shadow_tree * tree,
                        const uint64_t * hashes,
                        const int32_t * lengths,
                        int32_t count);

int32_t shadow_tree_score(const struct shadow_tree * tree,
                           const llama_token * tokens,
                           int32_t n_tokens);

int32_t route_cache_aware(const struct shadow_tree * trees,
                          int32_t n_workers,
                          const llama_token * prompt_tokens,
                          int32_t n_prompt);

#endif
