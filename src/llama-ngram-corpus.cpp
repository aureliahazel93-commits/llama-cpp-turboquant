#include "llama-ngram-corpus.h"

llama_ngram_corpus::llama_ngram_corpus(int depth) : depth_(depth) {
    nodes_.emplace_back();
}

void llama_ngram_corpus::insert(const llama_token * tokens, int32_t n) {
    if (n <= 0) return;
    int32_t idx = 0;
    int32_t limit = (n < depth_) ? n : depth_;
    for (int32_t i = 0; i < limit; i++) {
        auto & children = nodes_[idx].children;
        auto it = children.find(tokens[i]);
        if (it == children.end()) {
            int32_t next = (int32_t)nodes_.size();
            nodes_.emplace_back();
            children[tokens[i]] = next;
            idx = next;
        } else {
            idx = it->second;
        }
        nodes_[idx].count++;
    }
}

int32_t llama_ngram_corpus::lookup(const llama_token * tokens, int32_t n, llama_token * out, int32_t max_out) const {
    if (n <= 0 || max_out <= 0) return 0;
    int32_t idx = 0;
    int32_t limit = (n < depth_) ? n : depth_;
    for (int32_t i = 0; i < limit; i++) {
        auto it = nodes_[idx].children.find(tokens[i]);
        if (it == nodes_[idx].children.end()) return 0;
        idx = it->second;
    }
    int32_t produced = 0;
    auto & children = nodes_[idx].children;
    for (auto it = children.begin(); it != children.end() && produced < max_out; ++it) {
        out[produced++] = it->first;
    }
    return produced;
}

void llama_ngram_corpus::clear() {
    nodes_.clear();
    nodes_.emplace_back();
}

int32_t llama_ngram_corpus::size() const {
    return (int32_t)nodes_.size() - 1;
}
