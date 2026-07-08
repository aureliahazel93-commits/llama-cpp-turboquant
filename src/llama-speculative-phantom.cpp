#include "llama-ngram-corpus.h"
#include "llama-bloom-filter.h"
#include "llama-phantom-buffer.h"

#include "llama.h"
#include <algorithm>
#include <vector>

#define PHANTOM_DEFAULT_DEPTH 4
#define PHANTOM_DEFAULT_MAX_DRAFT 8

struct phantom_draft_result {
    std::vector<llama_token> tokens;
    int32_t n_tokens;
};

struct llama_speculative_phantom {
    llama_ngram_corpus corpus;
    llama_bloom_filter bloom;
    int32_t draft_depth;
    int32_t max_draft;

    llama_speculative_phantom(int32_t depth = PHANTOM_DEFAULT_DEPTH,
                               int32_t max_draft = PHANTOM_DEFAULT_MAX_DRAFT)
        : corpus(depth), draft_depth(depth), max_draft(max_draft) {}

    void on_rejected(llama_token prev, llama_token next) {
        bloom.mark_rejected(prev, next);
    }

    phantom_draft_result generate_draft(const llama_token * context,
                                         int32_t n_context,
                                         llama_token last_token) {
        phantom_draft_result result;
        result.n_tokens = 0;
        result.tokens.reserve(max_draft);

        llama_token window[NGRAM_MAX_DEPTH];
        int32_t wlen = 0;

        int32_t start = n_context - draft_depth + 1;
        if (start < 0) start = 0;
        for (int32_t i = start; i < n_context; i++) {
            if (wlen < NGRAM_MAX_DEPTH) {
                window[wlen++] = context[i];
            }
        }
        if (wlen < NGRAM_MAX_DEPTH) {
            window[wlen++] = last_token;
        }

        for (int32_t i = 0; i < max_draft; i++) {
            llama_token draft_token;
            int32_t n = corpus.lookup(window, wlen, &draft_token, 1);
            if (n == 0) break;
            if (bloom.is_rejected(last_token, draft_token)) break;

            result.tokens.push_back(draft_token);
            result.n_tokens++;

            if (wlen >= draft_depth) {
                for (int32_t j = 0; j < draft_depth - 1; j++) {
                    window[j] = window[j + 1];
                }
                window[draft_depth - 1] = draft_token;
            } else {
                if (wlen < NGRAM_MAX_DEPTH) {
                    window[wlen++] = draft_token;
                }
            }
            last_token = draft_token;
        }
        return result;
    }

    void on_generated(const llama_token * tokens, int32_t n) {
        corpus.insert(tokens, n);
    }
};
