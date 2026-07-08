// cat-first: tranche 1

#pragma once

#include "llama.h"

#include <cstdint>
#include <cstring>
#include <cstdlib>

//
// prefix_node — radix tree node mapping token-sequence prefix → KV block IDs
//

struct prefix_node {
    llama_token   token;          // token at this position (root has token = -1)
    prefix_node * children[256];  // child pointers (token value → next node)
    int32_t     * block_ids;      // physical KV block IDs for this prefix
    int           n_blocks;       // number of blocks
    int           ref_count;      // sequences sharing this node's blocks
    uint64_t      hash;           // xxhash of this node's token + parent hash
    uint64_t      parent_hash;    // parent's hash for chain verification
    prefix_node * parent;
    prefix_node * lru_prev;       // LRU doubly-linked list
    prefix_node * lru_next;
};

//
// prefix_match — result of a radix tree longest-prefix lookup
//

struct prefix_match {
    prefix_node * node;           // matched node (longest prefix)
    int           match_len;      // length of matched prefix (tokens)
    int32_t     * block_ids;      // KV block IDs for the matched prefix
    int           n_blocks;       // number of blocks
    bool          exact;          // true if exact match
};

//
// llama_prefix_cache — SGLang-style radix tree for prefix caching
//

class llama_prefix_cache {
public:
    explicit llama_prefix_cache(int n_blocks_total);
    ~llama_prefix_cache();

    // Insert a token sequence → block_ids mapping
    void insert(const llama_token * tokens, int n_tokens,
                const int32_t * block_ids, int n_blocks);

    // Find longest prefix match for a token sequence
    prefix_match lookup(const llama_token * tokens, int n_tokens) const;

    // Split a node when a sequence diverges from a shared prefix
    void split(const llama_token * seq_a, int len_a,
               const llama_token * seq_b, int len_b,
               const int32_t * blocks_b, int n_blocks_b);

    // Increment ref_count on a node
    void inc_ref(prefix_node * node);

    // Decrement ref_count, free blocks if ref_count reaches 0
    void dec_ref(prefix_node * node);

    // Evict LRU leaf nodes until n_blocks_free >= needed; returns blocks freed
    int evict(int n_blocks_needed);

    // Stats
    int  total_blocks() const;
    int  used_blocks() const;
    int  hit_count() const;
    void reset_stats();

private:
    prefix_node * root;
    int n_blocks_total;
    int n_blocks_used;

    // LRU list (head = most recent, tail = least recent)
    prefix_node * lru_head;
    prefix_node * lru_tail;

    // Free block pool (mirrors Phase 21 kv_block_pool interface)
    int * free_blocks;
    int   n_free_blocks;

    // Stats
    mutable int hits;
    mutable int misses;

    // Hash utilities (xxhash-inspired: PRNG-style mixing)
    uint64_t compute_hash(uint64_t parent_hash, llama_token token) const;

    // Internal node operations
    prefix_node * find_child(prefix_node * parent, llama_token token) const;
    prefix_node * create_child(prefix_node * parent, llama_token token);
    void remove_from_lru(prefix_node * node);
    void touch_lru(prefix_node * node);

    // Internal allocation
    prefix_node * alloc_node();
    void free_node(prefix_node * node);
};
