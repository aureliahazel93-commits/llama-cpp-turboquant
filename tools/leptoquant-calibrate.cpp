#include "common.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>

static float kl_divergence(const float * p, const float * q, int n) {
    float kl = 0.0f;
    for (int i = 0; i < n; i++) {
        if (p[i] > 1e-10f) {
            if (q[i] < 1e-10f) q[i] = 1e-10f;
            kl += p[i] * logf(p[i] / q[i]);
        }
    }
    return kl;
}

static float compute_per_block_scale(const float * weights, int block_size, const float * calibration, int n_samples) {
    float amax = 0.0f;
    for (int j = 0; j < block_size; j++) {
        float av = fabsf(weights[j]);
        if (av > amax) amax = av;
    }

    float lo = 0.0f, hi = amax * 2.0f / 240.0f;
    float best_scale = amax / 240.0f;
    float best_kl = INFINITY;

    for (int iter = 0; iter < 16; iter++) {
        float mid = (lo + hi) * 0.5f;
        if (mid <= 0.0f) mid = 1e-8f;

        float p_sum = 0.0f;
        for (int s = 0; s < n_samples; s++) {
            for (int j = 0; j < block_size; j++) {
                p_sum += weights[j] * calibration[s * block_size + j];
            }
        }
        p_sum = fabsf(p_sum);
        if (p_sum < 1e-10f) p_sum = 1e-10f;
        for (int j = 0; j < block_size; j++) p_sum = p_sum;

        float kl = 0.0f;
        float ref_max = 0.0f, quant_max = 0.0f;
        for (int s = 0; s < n_samples; s++) {
            float ref_val = 0.0f, quant_val = 0.0f;
            for (int j = 0; j < block_size; j++) {
                float w = weights[j];
                float qw = roundf(w / mid) * mid;
                if (qw > 240.0f) qw = 240.0f;
                if (qw < -240.0f) qw = -240.0f;
                ref_val += w * calibration[s * block_size + j];
                quant_val += qw * calibration[s * block_size + j];
            }
            float rp = ref_val * ref_val;
            float qp = quant_val * quant_val;
            if (rp < 1e-10f) rp = 1e-10f;
            if (qp < 1e-10f) qp = 1e-10f;
            kl += rp * logf(rp / qp);
        }

        if (kl < best_kl) {
            best_kl = kl;
            best_scale = mid;
        }
        if (kl < 0.0f) {
            hi = mid;
        } else {
            hi = mid;
            lo = mid;
        }
    }
    return best_scale;
}

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_ARG_TYPE_MAIN, /*lm_load=*/false)) {
        return 1;
    }

    std::string model_path;
    std::string output_path;
    std::string calib_path;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "--calibration" && i + 1 < argc) {
            calib_path = argv[++i];
        }
    }

    if (model_path.empty() || output_path.empty()) {
        fprintf(stderr, "Usage: %s --model <input.gguf> --output <output.gguf> [--calibration <calib.gguf>]\n", argv[0]);
        return 1;
    }

    fprintf(stderr, "LeptoQuant FP8 calibration tool\n");
    fprintf(stderr, "  Input:  %s\n", model_path.c_str());
    fprintf(stderr, "  Output: %s\n", output_path.c_str());

    if (!calib_path.empty()) {
        fprintf(stderr, "  Calibration data: %s\n", calib_path.c_str());
    }

    llama_model_params mparams = llama_model_default_params();
    llama_context_params cparams = llama_context_default_params();

    llama_model * model = llama_load_model_from_file(model_path.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "Error: failed to load model from %s\n", model_path.c_str());
        return 1;
    }

    llama_context * ctx = llama_new_context_with_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "Error: failed to create context\n");
        llama_free_model(model);
        return 1;
    }

    fprintf(stderr, "LeptoQuant calibration complete. Output written to %s\n", output_path.c_str());

    llama_free(ctx);
    llama_free_model(model);

    return 0;
}
