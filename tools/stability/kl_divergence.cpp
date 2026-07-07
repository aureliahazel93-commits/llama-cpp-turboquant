#include "kl_divergence.h"
#include <cmath>
#include <algorithm>

static void softmax(float * probs, const float * logits, int n) {
    float max_val = *std::max_element(logits, logits + n);
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        probs[i] = expf(logits[i] - max_val);
        sum += probs[i];
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; i++) probs[i] *= inv_sum;
}

double kl_divergence(const float * ref, const float * quant, int n) {
    std::vector<float> p(n), q(n);
    softmax(p.data(), ref, n);
    softmax(q.data(), quant, n);
    double kl = 0.0;
    for (int i = 0; i < n; i++) {
        if (p[i] > 1e-12f && q[i] > 1e-12f) {
            kl += p[i] * log(p[i] / q[i]);
        }
    }
    return kl;
}

static float percentile(std::vector<double> & v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t idx = (size_t)((v.size() - 1) * (p / 100.0));
    return (float)v[idx];
}

struct kl_result kl_divergence_batch(const float * ref, const float * quant, int n_tokens, int n_vocab) {
    struct kl_result result = {0.0, 0.0, 0.0, n_tokens};
    std::vector<double> kl_per_token(n_tokens);
    for (int t = 0; t < n_tokens; t++) {
        kl_per_token[t] = kl_divergence(
            ref + t * n_vocab,
            quant + t * n_vocab,
            n_vocab);
        result.mean_kl += kl_per_token[t];
        result.max_kl = std::max(result.max_kl, kl_per_token[t]);
    }
    result.mean_kl /= n_tokens;
    result.p95_kl = percentile(kl_per_token, 95.0);
    return result;
}
