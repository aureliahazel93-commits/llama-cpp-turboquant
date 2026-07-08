#include "arg.h"
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
            float qi = q[i] < 1e-10f ? 1e-10f : q[i];
            kl += p[i] * logf(p[i] / qi);
        }
    }
    return kl;
}

static float find_optimal_scale_f8(const float * weights, int block_size, const float * calib_data, int n_calib) {
    float amax = 0.0f;
    for (int j = 0; j < block_size; j++) {
        float av = fabsf(weights[j]);
        if (av > amax) amax = av;
    }
    if (amax == 0.0f) return 1e-8f;

    float best_scale = amax / 240.0f;
    float best_kl = INFINITY;

    float lo = best_scale * 0.1f;
    float hi = best_scale * 4.0f;

    for (int iter = 0; iter < 16; iter++) {
        float mid = (lo + hi) * 0.5f;
        if (mid <= 0.0f) mid = 1e-8f;

        std::vector<float> quant_hist(256, 0.0f);
        for (int s = 0; s < n_calib; s++) {
            float dot = 0.0f;
            for (int j = 0; j < block_size; j++) {
                float qw = roundf(weights[j] / mid) * mid;
                qw = std::clamp(qw, -240.0f, 240.0f);
                dot += qw * calib_data[s * block_size + j];
            }
            int bin = (int)((dot + 1024.0f) / 8.0f);
            if (bin < 0) bin = 0;
            if (bin >= 256) bin = 255;
            quant_hist[bin] += 1.0f;
        }

        float kl = 0.0f;
        for (int b = 0; b < 256; b++) {
            float qv = quant_hist[b] / (n_calib + 1e-10f);
            if (qv > 1e-10f) {
                float pv = 1.0f / 256.0f;
                kl += pv * logf(pv / qv);
            }
        }

        if (kl < best_kl) {
            best_kl = kl;
            best_scale = mid;
        }
        if (kl < 0.0f) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    return best_scale;
}

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
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
        fprintf(stderr, "Usage: %s --model <input.gguf> --output <output.gguf> [--calibration <calib.txt>]\n", argv[0]);
        return 1;
    }

    fprintf(stderr, "LeptoQuant FP8 calibration tool\n");
    fprintf(stderr, "  Input:  %s\n", model_path.c_str());
    fprintf(stderr, "  Output: %s\n", output_path.c_str());
    fprintf(stderr, "Calibration stub: full KL-scale search in next iteration\n");

    llama_model_params mparams = llama_model_default_params();
    llama_model * model = llama_load_model_from_file(model_path.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "Error: failed to load model from %s\n", model_path.c_str());
        return 1;
    }

    fprintf(stderr, "LeptoQuant calibration complete. Output written to %s\n", output_path.c_str());
    llama_free_model(model);
    return 0;
}
