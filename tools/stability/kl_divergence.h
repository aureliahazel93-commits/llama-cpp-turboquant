#pragma once
#include <vector>
#include <cstdint>

// Compute KL divergence D_KL(P || Q) between two logit distributions.
// P = reference (typically F16 model), Q = quantized model.
// Returns divergence in nats. Lower = better quantization quality.
double kl_divergence(
    const float * logits_ref,   // reference model logits [n_vocab]
    const float * logits_quant, // quantized model logits [n_vocab]
    int n_vocab);

// Batch KL divergence: compute average KL across a sequence of token positions.
struct kl_result {
    double mean_kl;
    double max_kl;
    double p95_kl;
    int    n_tokens;
};

struct kl_result kl_divergence_batch(
    const float * logits_ref,   // [n_tokens, n_vocab]
    const float * logits_quant, // [n_tokens, n_vocab]
    int n_tokens,
    int n_vocab);
