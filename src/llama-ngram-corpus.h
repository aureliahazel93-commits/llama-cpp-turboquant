#ifndef LLAMA_NGRAM_CORPUS_H
#define LLAMA_NGRAM_CORPUS_H

#include "llama.h"
#include <cstdint>
#include <vector>
#include <unordered_map>

#define NGRAM_MAX_DEPTH 8

struct ngram_node {
    std::unordered_map<llama_token, int32_t> children;
    int32_t count;
};

class llama_ngram_corpus {
public:
    llama_ngram_corpus(int depth = 4);
    void insert(const llama_token * tokens, int32_t n);
    int32_t lookup(const llama_token * tokens, int32_t n, llama_token * out, int32_t max_out) const;
    void clear();
    int32_t size() const;
private:
    int32_t depth_;
    std::vector<ngram_node> nodes_;
};

#endif
