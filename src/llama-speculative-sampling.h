#ifndef LLAMA_SPECULATIVE_SAMPLING_H
#define LLAMA_SPECULATIVE_SAMPLING_H

#include "llama.h"
#include <cstdint>
#include <vector>

enum sampling_tier {
    SAMPLING_NATIVE_CPP = 0,
    SAMPLING_VULKAN_GPU = 1,
    SAMPLING_SCALAR     = 2,
};

struct sampling_result {
    std::vector<llama_token> accepted;
    llama_token bonus_token;
    int32_t n_accepted;
    bool all_accepted;
};

struct spec_sampling_config {
    sampling_tier preferred_tier;
    float temperature;
    int32_t top_k;
};

bool spec_sampling_verify(const float * target_logits, int32_t n_positions, int32_t n_vocab,
                           const llama_token * draft_tokens, int32_t n_draft,
                           const struct spec_sampling_config * config,
                           struct sampling_result * result);

sampling_tier spec_sampling_detect_best();

#endif
