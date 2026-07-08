#ifndef LLAMA_PHANTOM_BUFFER_H
#define LLAMA_PHANTOM_BUFFER_H

#include "llama.h"
#include <atomic>
#include <cstdint>
#include <cstring>

#define PHANTOM_MAX_DRAFT 16
#define PHANTOM_MAX_SLOTS 4

struct phantom_draft_slot {
    llama_token tokens[PHANTOM_MAX_DRAFT];
    int32_t n_tokens;
    uint64_t req_id;
    std::atomic<bool> ready{false};
    std::atomic<bool> consumed{false};
};

class llama_phantom_buffer {
public:
    llama_phantom_buffer() {}

    bool produce(const llama_token * tokens, int32_t n, uint64_t req_id) {
        int32_t t = tail_.load(std::memory_order_relaxed);
        int32_t slot = t % PHANTOM_MAX_SLOTS;
        if (slots_[slot].ready.load(std::memory_order_acquire)) return false;
        int32_t count = (n < PHANTOM_MAX_DRAFT) ? n : PHANTOM_MAX_DRAFT;
        std::memcpy(slots_[slot].tokens, tokens, count * sizeof(llama_token));
        slots_[slot].n_tokens = count;
        slots_[slot].req_id = req_id;
        slots_[slot].consumed.store(false, std::memory_order_release);
        slots_[slot].ready.store(true, std::memory_order_release);
        tail_.store(t + 1, std::memory_order_release);
        return true;
    }

    const phantom_draft_slot * consume() {
        int32_t h = head_.load(std::memory_order_relaxed);
        int32_t slot = h % PHANTOM_MAX_SLOTS;
        if (!slots_[slot].ready.load(std::memory_order_acquire)) return nullptr;
        slots_[slot].consumed.store(true, std::memory_order_release);
        slots_[slot].ready.store(false, std::memory_order_release);
        head_.store(h + 1, std::memory_order_release);
        return &slots_[slot];
    }

    int32_t n_ready() const {
        int32_t t = tail_.load(std::memory_order_acquire);
        int32_t h = head_.load(std::memory_order_acquire);
        int32_t count = t - h;
        return (count > PHANTOM_MAX_SLOTS) ? PHANTOM_MAX_SLOTS : count;
    }

    void clear() {
        for (int32_t i = 0; i < PHANTOM_MAX_SLOTS; i++) {
            slots_[i].n_tokens = 0;
            slots_[i].req_id = 0;
            slots_[i].ready.store(false, std::memory_order_relaxed);
            slots_[i].consumed.store(false, std::memory_order_relaxed);
        }
        head_.store(0, std::memory_order_relaxed);
        tail_.store(0, std::memory_order_relaxed);
    }

private:
    phantom_draft_slot slots_[PHANTOM_MAX_SLOTS];
    std::atomic<int32_t> head_{0};
    std::atomic<int32_t> tail_{0};
};

#endif
