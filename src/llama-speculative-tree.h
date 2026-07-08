#ifndef LLAMA_SPECULATIVE_TREE_H
#define LLAMA_SPECULATIVE_TREE_H

#include "llama.h"
#include <cstdint>
#include <vector>

#define SPEC_TREE_MAX_DEPTH   5
#define SPEC_TREE_MAX_BRANCH  4
#define SPEC_TREE_MAX_TOKENS  (SPEC_TREE_MAX_DEPTH * SPEC_TREE_MAX_BRANCH)

struct spec_tree_node {
    int32_t  parent;
    int32_t  depth;
    int32_t  token;
    float    prob;
    int32_t  n_children;
    int32_t  children[SPEC_TREE_MAX_BRANCH];
};

struct spec_tree {
    spec_tree_node nodes[SPEC_TREE_MAX_TOKENS];
    int32_t        n_nodes;
    int32_t        root;
    int32_t        max_depth;
    int32_t        n_leaves;
    int32_t        leaf_indices[SPEC_TREE_MAX_TOKENS];
};

struct spec_tree_result {
    int32_t  n_accepted;
    int32_t  reject_at;
    llama_token bonus_token;
    bool     all_accepted;
};

struct spec_tree_params {
    int32_t max_depth;
    int32_t max_branch;
    int32_t top_k;
    float   min_prob;
};

void spec_tree_init(struct spec_tree * tree, const struct spec_tree_params * params);

int32_t spec_tree_insert(struct spec_tree * tree, int32_t parent, llama_token token, float prob);

void spec_tree_build_mask(const struct spec_tree * tree,
                          int32_t n_context_tokens,
                          std::vector<std::vector<bool>> * mask);

void spec_tree_flatten(const struct spec_tree * tree,
                       std::vector<llama_token> * tokens,
                       std::vector<int32_t> * parent_indices);

struct spec_tree_result spec_tree_verify(const struct spec_tree * tree,
                                        const float * target_logits,
                                        int32_t n_vocab,
                                        llama_token bonus_token);

#endif // LLAMA_SPECULATIVE_TREE_H
