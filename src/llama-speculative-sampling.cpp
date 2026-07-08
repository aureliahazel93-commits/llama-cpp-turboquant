#include "llama-speculative-sampling.h"
#include <limits>

bool spec_sampling_verify(const float * target_logits, int32_t n_positions, int32_t n_vocab,
                           const llama_token * draft_tokens, int32_t n_draft,
                           const struct spec_sampling_config * config,
                           struct sampling_result * result) {
    (void)config;
    result->accepted.clear();
    result->n_accepted = 0;
    result->all_accepted = true;
    result->bonus_token = -1;

    for (int32_t i = 0; i < n_draft; i++) {
        if (i >= n_positions) {
            result->all_accepted = false;
            break;
        }
        int32_t base = i * n_vocab;
        float best_logit = -1e30f;
        llama_token best_idx = 0;
        for (int32_t v = 0; v < n_vocab; v++) {
            if (target_logits[base + v] > best_logit) {
                best_logit = target_logits[base + v];
                best_idx = (llama_token)v;
            }
        }
        if (best_idx == draft_tokens[i]) {
            result->accepted.push_back(best_idx);
            result->n_accepted++;
        } else {
            result->bonus_token = best_idx;
            result->all_accepted = false;
            break;
        }
    }
    return result->all_accepted;
}

sampling_tier spec_sampling_detect_best() {
    return SAMPLING_NATIVE_CPP;
}
