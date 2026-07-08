#include "quant-recommender.h"
#include <algorithm>
#include <cmath>

struct quant_candidate {
    enum ggml_type weight;
    enum ggml_type kv;
    float bpw;
    bool has_gpu_kernel;
};

static float type_bpw(enum ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32:  return 32.0f;
        case GGML_TYPE_F16:  return 16.0f;
        case GGML_TYPE_Q4_0: return 4.5f;
        case GGML_TYPE_Q4_1: return 5.0f;
        case GGML_TYPE_Q5_0: return 5.5f;
        case GGML_TYPE_Q5_1: return 6.0f;
        case GGML_TYPE_Q8_0: return 8.5f;
        case GGML_TYPE_Q2_K: return 2.75f;
        case GGML_TYPE_Q3_K: return 3.5f;
        case GGML_TYPE_Q4_K: return 4.5f;
        case GGML_TYPE_Q5_K: return 5.5f;
        case GGML_TYPE_Q6_K: return 6.5f;
        default: return 32.0f;
    }
}

static bool has_gpu_kernel(enum ggml_type t) {
    switch (t) {
        case GGML_TYPE_F16:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
            return true;
        default:
            return false;
    }
}

quant_recommendation recommend(const prober_output & probe, const model_info & info) {
    quant_candidate candidates[] = {
        { GGML_TYPE_F32, GGML_TYPE_F16, 32.0f, false },
        { GGML_TYPE_F16, GGML_TYPE_F16, 16.0f, true },
        { GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 8.5f, true },
        { GGML_TYPE_Q6_K, GGML_TYPE_Q6_K, 6.5f, false },
        { GGML_TYPE_Q5_K, GGML_TYPE_Q5_K, 5.5f, false },
        { GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, 4.5f, false },
        { GGML_TYPE_Q5_1, GGML_TYPE_Q5_1, 6.0f, true },
        { GGML_TYPE_Q5_0, GGML_TYPE_Q5_0, 5.5f, true },
        { GGML_TYPE_Q4_1, GGML_TYPE_Q4_1, 5.0f, true },
        { GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, 4.5f, true },
    };
    int n = sizeof(candidates) / sizeof(candidates[0]);

    std::sort(candidates, candidates + n,
              [](const quant_candidate & a, const quant_candidate & b) {
                  return a.bpw > b.bpw;
              });

    size_t vram_bytes = (size_t)((double)info.total_vram * 1024 * 1024 * 0.9);
    quant_recommendation best = { GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, "cpu", 0.0f, 0.0f };

    for (int i = 0; i < n; i++) {
        float est = (candidates[i].bpw * probe.param_bytes) / 8.0f;
        if (est > vram_bytes) continue;

        float quality = 100.0f * (1.0f - (candidates[i].bpw - 4.0f) / 28.0f);
        quality = std::max(0.0f, std::min(100.0f, quality));

        if (quality > best.quality_pct || best.quality_pct == 0.0f) {
            best.weight_type = candidates[i].weight;
            best.kv_type_k = candidates[i].kv;
            best.kv_type_v = candidates[i].kv;
            best.vram_mb = est / (1024.0f * 1024.0f);
            best.quality_pct = quality;
            best.backend = has_gpu_kernel(candidates[i].weight) ? "cuda" : "cpu";
        }
    }
    return best;
}
