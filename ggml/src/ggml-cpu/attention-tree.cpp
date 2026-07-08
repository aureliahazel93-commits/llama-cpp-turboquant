#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <vector>
#include <limits>

void ggml_compute_attention_tree(
        const float * q_data,
        const float * k_data,
        const float * v_data,
        const float * mask_data,
        float * out_data,
        int n_tree,
        int n_ctx_total,
        int n_heads,
        int head_dim) {
    std::vector<float> scores(n_ctx_total);
    std::vector<float> attn_weights(n_ctx_total);

    for (int q = 0; q < n_tree; q++) {
        for (int h = 0; h < n_heads; h++) {
            const float * q_vec = q_data + (q * n_heads + h) * head_dim;
            float * out_vec = out_data + (q * n_heads + h) * head_dim;

            float max_score = -std::numeric_limits<float>::infinity();

            for (int k = 0; k < n_ctx_total; k++) {
                if (mask_data[q * n_ctx_total + k] < 0.5f) continue;

                const float * k_vec = k_data + (k * n_heads + h) * head_dim;
                float dot = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    dot += q_vec[d] * k_vec[d];
                }
                scores[k] = dot;
                if (dot > max_score) max_score = dot;
            }

            float attn_sum = 0.0f;
            for (int k = 0; k < n_ctx_total; k++) {
                if (mask_data[q * n_ctx_total + k] < 0.5f) {
                    attn_weights[k] = 0.0f;
                    continue;
                }
                float exp_score = std::exp(scores[k] - max_score);
                attn_weights[k] = exp_score;
                attn_sum += exp_score;
            }

            for (int d = 0; d < head_dim; d++) {
                out_vec[d] = 0.0f;
            }

            if (attn_sum > 0.0f) {
                for (int k = 0; k < n_ctx_total; k++) {
                    if (attn_weights[k] == 0.0f) continue;
                    float w = attn_weights[k] / attn_sum;
                    const float * v_vec = v_data + (k * n_heads + h) * head_dim;
                    for (int d = 0; d < head_dim; d++) {
                        out_vec[d] += w * v_vec[d];
                    }
                }
            }
        }
    }
}
