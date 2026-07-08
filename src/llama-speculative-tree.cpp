#include "llama-speculative-tree.h"

#include <algorithm>
#include <cstring>
#include <queue>
#include <limits>

void spec_tree_init(struct spec_tree * tree, const struct spec_tree_params * params) {
    std::memset(tree, 0, sizeof(struct spec_tree));

    tree->root = 0;
    tree->n_nodes = 1;
    tree->max_depth = 0;
    tree->n_leaves = 1;

    struct spec_tree_node * root = &tree->nodes[0];
    root->parent = -1;
    root->depth = 0;
    root->token = 0;
    root->prob = 1.0f;
    root->n_children = 0;
    for (int i = 0; i < SPEC_TREE_MAX_BRANCH; i++) {
        root->children[i] = -1;
    }

    if (params) {
        tree->max_depth = params->max_depth;
    }
    tree->leaf_indices[0] = 0;
}

int32_t spec_tree_insert(struct spec_tree * tree, int32_t parent, llama_token token, float prob) {
    if (tree->n_nodes >= SPEC_TREE_MAX_TOKENS) return -1;
    if (parent < 0 || parent >= tree->n_nodes) return -1;
    if (tree->nodes[parent].n_children >= SPEC_TREE_MAX_BRANCH) return -1;

    int32_t idx = tree->n_nodes;
    struct spec_tree_node * node = &tree->nodes[idx];
    node->parent = parent;
    node->depth = tree->nodes[parent].depth + 1;
    node->token = token;
    node->prob = prob;
    node->n_children = 0;
    for (int i = 0; i < SPEC_TREE_MAX_BRANCH; i++) {
        node->children[i] = -1;
    }

    tree->nodes[parent].children[tree->nodes[parent].n_children] = idx;
    tree->nodes[parent].n_children++;
    tree->n_nodes++;

    if (node->depth > tree->max_depth) {
        tree->max_depth = node->depth;
    }

    if (tree->nodes[parent].n_children == 1) {
        for (int i = 0; i < tree->n_leaves; i++) {
            if (tree->leaf_indices[i] == parent) {
                tree->leaf_indices[i] = idx;
                break;
            }
        }
    } else {
        tree->leaf_indices[tree->n_leaves] = idx;
        tree->n_leaves++;
    }

    return idx;
}

void spec_tree_build_mask(const struct spec_tree * tree,
                          int32_t n_context_tokens,
                          std::vector<std::vector<bool>> * mask) {
    int32_t n_tree = tree->n_nodes;
    int32_t n_total = n_context_tokens + n_tree;

    mask->resize(n_total, std::vector<bool>(n_total, false));

    for (int i = 0; i < n_context_tokens; i++) {
        for (int j = 0; j <= i; j++) {
            (*mask)[i][j] = true;
        }
    }

    for (int q = 0; q < n_tree; q++) {
        int pos_q = n_context_tokens + q;
        for (int k = 0; k < n_context_tokens; k++) {
            (*mask)[pos_q][k] = true;
        }
        int node = q;
        while (node != -1) {
            (*mask)[pos_q][n_context_tokens + node] = true;
            node = tree->nodes[node].parent;
        }
    }
}

void spec_tree_flatten(const struct spec_tree * tree,
                       std::vector<llama_token> * tokens,
                       std::vector<int32_t> * parent_indices) {
    tokens->clear();
    parent_indices->clear();
    tokens->reserve(tree->n_nodes);
    parent_indices->reserve(tree->n_nodes);

    for (int i = 0; i < tree->n_nodes; i++) {
        tokens->push_back(tree->nodes[i].token);
        parent_indices->push_back(tree->nodes[i].parent);
    }
}

struct spec_tree_result spec_tree_verify(const struct spec_tree * tree,
                                        const float * target_logits,
                                        int32_t n_vocab,
                                        llama_token bonus_token) {
    struct spec_tree_result result;
    result.n_accepted = 0;
    result.reject_at = -1;
    result.bonus_token = 0;
    result.all_accepted = false;

    int32_t n_tree = tree->n_nodes;

    for (int i = 0; i < n_tree; i++) {
        const float * logits = target_logits + i * n_vocab;
        int32_t argmax = 0;
        float max_logit = logits[0];
        for (int v = 1; v < n_vocab; v++) {
            if (logits[v] > max_logit) {
                max_logit = logits[v];
                argmax = v;
            }
        }

        if (argmax != (int32_t)tree->nodes[i].token) {
            result.n_accepted = i;
            result.reject_at = i;
            result.bonus_token = (llama_token)argmax;
            result.all_accepted = false;
            return result;
        }
    }

    result.n_accepted = n_tree;
    result.reject_at = -1;
    result.bonus_token = bonus_token;
    result.all_accepted = true;
    return result;
}
