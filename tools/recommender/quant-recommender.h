#ifndef QUANT_RECOMMENDER_H
#define QUANT_RECOMMENDER_H

#include "ggml.h"
#include <string>
#include <vector>

struct prober_output {
    int n_layers;
    int n_embd;
    int n_head;
    int n_head_kv;
    size_t param_bytes;
    bool has_attention;
};

struct model_info {
    std::string name;
    std::string arch;
    int n_params_billion;
    size_t total_vram;
};

struct quant_recommendation {
    enum ggml_type weight_type;
    enum ggml_type kv_type_k;
    enum ggml_type kv_type_v;
    std::string backend;
    float vram_mb;
    float quality_pct;
};

quant_recommendation recommend(const prober_output & probe, const model_info & info);

#endif
