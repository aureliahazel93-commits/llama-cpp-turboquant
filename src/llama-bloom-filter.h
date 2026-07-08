#ifndef LLAMA_BLOOM_FILTER_H
#define LLAMA_BLOOM_FILTER_H

#include "llama.h"
#include <cstdint>
#include <cstring>
#include <vector>

#define BLOOM_BITS (1 << 20)
#define BLOOM_HASH_SEED1 0x9E3779B97F4A7C15ULL
#define BLOOM_HASH_SEED2 0xBF58476D1CE4E5B9ULL

class llama_bloom_filter {
public:
    llama_bloom_filter() { bits_.resize(BLOOM_BITS / 64, 0); }

    void mark_rejected(llama_token prev, llama_token next) {
        uint64_t h1 = hash(prev, next, BLOOM_HASH_SEED1);
        uint64_t h2 = hash(prev, next, BLOOM_HASH_SEED2);
        bits_[h1 % bits_.size()] |= (1ULL << (h1 % 64));
        bits_[h2 % bits_.size()] |= (1ULL << (h2 % 64));
    }

    bool is_rejected(llama_token prev, llama_token next) const {
        uint64_t h1 = hash(prev, next, BLOOM_HASH_SEED1);
        uint64_t h2 = hash(prev, next, BLOOM_HASH_SEED2);
        return ((bits_[h1 % bits_.size()] >> (h1 % 64)) & 1) ||
               ((bits_[h2 % bits_.size()] >> (h2 % 64)) & 1);
    }

    void clear() { std::memset(bits_.data(), 0, bits_.size() * 8); }

private:
    std::vector<uint64_t> bits_;

    static uint64_t hash(llama_token a, llama_token b, uint64_t seed) {
        uint64_t h = seed;
        h ^= (uint64_t)a;
        h *= 1099511628211ULL;
        h ^= (uint64_t)b;
        h *= 1099511628211ULL;
        return h;
    }
};

#endif
