// cat-first: tranche 1

#include "llama-prefix-cache.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <vector>

// cat-first: tranche 2

static constexpr uint64_t FNV_OFFSET = 0xcbf29ce484222325ULL;
static constexpr uint64_t FNV_PRIME  = 0x100000001b3ULL;

llama_prefix_cache::llama_prefix_cache(int n_blocks_total)
    : root(new prefix_node()),
      n_blocks_total(n_blocks_total),
      n_blocks_used(0),
      lru_head(nullptr),
      lru_tail(nullptr),
      free_blocks(new int[n_blocks_total]),
      n_free_blocks(n_blocks_total),
      hits(0),
      misses(0) {
    assert(n_blocks_total > 0);

    std::memset(root, 0, sizeof(prefix_node));
    root->token = -1;

    for (int i = 0; i < n_blocks_total; i++) {
        free_blocks[i] = i;
    }
}

llama_prefix_cache::~llama_prefix_cache() {
    free_node(root);
    delete[] free_blocks;
}

uint64_t llama_prefix_cache::compute_hash(uint64_t parent_hash, llama_token token) const {
    uint64_t h = parent_hash ^ FNV_OFFSET;
    h *= FNV_PRIME;
    h ^= static_cast<uint64_t>(token);
    return h;
}

// cat-first: tranche 3

prefix_node * llama_prefix_cache::alloc_node() {
    auto * node = new prefix_node();
    std::memset(node, 0, sizeof(prefix_node));
    node->token = -1;
    return node;
}

void llama_prefix_cache::free_node(prefix_node * node) {
    if (node == nullptr) return;
    for (int i = 0; i < 256; i++) {
        if (node->children[i] != nullptr) {
            free_node(node->children[i]);
        }
    }
    delete[] node->block_ids;
    delete node;
}

prefix_node * llama_prefix_cache::find_child(prefix_node * parent, llama_token token) const {
    return parent->children[(uint8_t)token];
}

prefix_node * llama_prefix_cache::create_child(prefix_node * parent, llama_token token) {
    auto * node = alloc_node();
    node->token = token;
    node->parent = parent;
    node->parent_hash = parent->hash;
    parent->children[(uint8_t)token] = node;
    return node;
}

// cat-first: tranche 4

prefix_match llama_prefix_cache::lookup(const llama_token * tokens, int n_tokens) const {
    prefix_node * cur = root;
    int match_len = 0;
    uint64_t h = 0;

    for (int i = 0; i < n_tokens; i++) {
        prefix_node * child = find_child(cur, tokens[i]);
        if (child == nullptr) {
            misses++;
            if (match_len > 0) {
                hits++;
            }
            return { cur, match_len, cur->block_ids, cur->n_blocks, false };
        }
        h = compute_hash(h, tokens[i]);
        child->hash = h;
        cur = child;
        match_len++;
    }

    hits++;
    return { cur, match_len, cur->block_ids, cur->n_blocks, true };
}

// cat-first: tranche 5

void llama_prefix_cache::insert(const llama_token * tokens, int n_tokens,
                                const int32_t * block_ids, int n_blocks) {
    prefix_node * cur = root;
    uint64_t h = 0;

    for (int i = 0; i < n_tokens; i++) {
        prefix_node * child = find_child(cur, tokens[i]);
        if (child == nullptr) {
            child = create_child(cur, tokens[i]);
        }
        h = compute_hash(h, tokens[i]);
        child->hash = h;
        cur = child;
    }

    if (cur->block_ids != nullptr && cur->n_blocks > 0) {
        if (cur->n_blocks == n_blocks &&
            std::memcmp(cur->block_ids, block_ids, n_blocks * sizeof(int32_t)) == 0) {
            cur->ref_count++;
            touch_lru(cur);
            return;
        }
        delete[] cur->block_ids;
    }

    cur->block_ids = new int32_t[n_blocks];
    std::memcpy(cur->block_ids, block_ids, n_blocks * sizeof(int32_t));
    cur->n_blocks  = n_blocks;
    cur->ref_count = 1;

    n_blocks_used += n_blocks;
    n_free_blocks -= n_blocks;

    touch_lru(cur);
}

// cat-first: tranche 6

void llama_prefix_cache::split(
        const llama_token * seq_a, int len_a,
        const llama_token * seq_b, int len_b,
        const int32_t * blocks_b, int n_blocks_b) {
    int L = 0;
    int min_len = std::min(len_a, len_b);
    for (; L < min_len; L++) {
        if (seq_a[L] != seq_b[L]) break;
    }

    prefix_node * cur = root;
    uint64_t h = 0;
    for (int i = 0; i < L; i++) {
        cur = find_child(cur, seq_a[i]);
        if (cur == nullptr) return;
        h = compute_hash(h, seq_a[i]);
    }

    if (cur->block_ids == nullptr || cur->n_blocks == 0) return;

    prefix_node * child_a = create_child(cur, seq_a[L]);
    child_a->block_ids = cur->block_ids;
    child_a->n_blocks  = cur->n_blocks;
    child_a->ref_count = cur->ref_count;
    child_a->hash = compute_hash(h, seq_a[L]);

    cur->block_ids = nullptr;
    cur->n_blocks  = 0;
    cur->ref_count = 0;

    prefix_node * child_b = create_child(cur, seq_b[L]);
    child_b->block_ids = new int32_t[n_blocks_b];
    std::memcpy(child_b->block_ids, blocks_b, n_blocks_b * sizeof(int32_t));
    child_b->n_blocks  = n_blocks_b;
    child_b->ref_count = 1;
    child_b->hash = compute_hash(h, seq_b[L]);

    n_blocks_used += n_blocks_b;
    n_free_blocks -= n_blocks_b;

    touch_lru(child_a);
    touch_lru(child_b);
}

// cat-first: tranche 7

void llama_prefix_cache::inc_ref(prefix_node * node) {
    if (node == nullptr || node == root) return;
    node->ref_count++;
}

void llama_prefix_cache::dec_ref(prefix_node * node) {
    if (node == nullptr || node == root) return;
    if (--node->ref_count <= 0) {
        n_blocks_used -= node->n_blocks;
        n_free_blocks += node->n_blocks;
        delete[] node->block_ids;
        node->block_ids = nullptr;
        node->n_blocks = 0;
    }
}

int llama_prefix_cache::evict(int n_blocks_needed) {
    int freed = 0;
    int remaining = n_blocks_needed;

    while (remaining > 0 && lru_tail != nullptr) {
        prefix_node * victim = lru_tail;
        if (victim->ref_count > 0) {
            remove_from_lru(victim);
            touch_lru(victim);
            continue;
        }
        if (victim->block_ids == nullptr || victim->n_blocks == 0) {
            remove_from_lru(victim);
            continue;
        }

        int to_free = victim->n_blocks;
        freed += to_free;
        remaining -= to_free;
        n_blocks_used -= to_free;
        n_free_blocks += to_free;

        delete[] victim->block_ids;
        victim->block_ids = nullptr;
        victim->n_blocks = 0;
        victim->ref_count = 0;

        remove_from_lru(victim);
    }

    return freed;
}

void llama_prefix_cache::remove_from_lru(prefix_node * node) {
    if (node->lru_prev) {
        node->lru_prev->lru_next = node->lru_next;
    } else if (lru_head == node) {
        lru_head = node->lru_next;
    }
    if (node->lru_next) {
        node->lru_next->lru_prev = node->lru_prev;
    } else if (lru_tail == node) {
        lru_tail = node->lru_prev;
    }
    node->lru_prev = nullptr;
    node->lru_next = nullptr;
}

void llama_prefix_cache::touch_lru(prefix_node * node) {
    if (node == root) return;
    remove_from_lru(node);
    node->lru_next = lru_head;
    node->lru_prev = nullptr;
    if (lru_head) {
        lru_head->lru_prev = node;
    }
    lru_head = node;
    if (lru_tail == nullptr) {
        lru_tail = node;
    }
}

int llama_prefix_cache::total_blocks() const {
    return n_blocks_total;
}

int llama_prefix_cache::used_blocks() const {
    return n_blocks_used;
}

int llama_prefix_cache::hit_count() const {
    return hits;
}

void llama_prefix_cache::reset_stats() {
    hits = 0;
    misses = 0;
}
