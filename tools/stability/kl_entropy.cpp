#include "calibrator.h"
#include <cmath>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <unordered_map>

float compute_kl_threshold(
    const uint64_t histogram[],
    int histogram_bins,
    int target_bins) {

    float best_kl = 1e20f;
    int best_bin = 0;

    for (int threshold = target_bins; threshold <= histogram_bins; threshold++) {
        std::vector<float> quantized(target_bins, 0.0f);
        std::vector<float> expanded(histogram_bins, 0.0f);
        float total = 0.0f;

        for (int i = 0; i < target_bins; i++) {
            int start = i * histogram_bins / target_bins;
            int end   = (i + 1) * histogram_bins / target_bins;

            if (end > threshold) {
                if (start >= threshold) {
                    quantized[i] = 0.0f;
                } else {
                    int count = 0;
                    for (int j = start; j < threshold; j++) count += (int)histogram[j];
                    quantized[i] = (float)count;
                }
            } else {
                int count = 0;
                for (int j = start; j < end; j++) count += (int)histogram[j];
                quantized[i] = (float)count;
            }
            total += quantized[i];
        }

        if (total > 0.0f) {
            float inv = 1.0f / total;
            for (int i = 0; i < target_bins; i++) quantized[i] *= inv;
        }

        for (int i = 0; i < histogram_bins; i++) {
            int quant_idx = i * target_bins / histogram_bins;
            if (quant_idx >= target_bins) quant_idx = target_bins - 1;
            expanded[i] = quantized[quant_idx];
        }

        for (int i = 0; i < histogram_bins; i++) {
            int start = i * target_bins / histogram_bins;
            int end   = (i + 1) * target_bins / histogram_bins;
            if (end > target_bins) end = target_bins;

            if (i >= threshold) {
                int count = 0;
                for (int j = i; j < histogram_bins; j++) count += (int)histogram[j];
                float p = (total > 0.0f) ? (float)count / total : 0.0f;
                int q_bin_start = threshold * target_bins / histogram_bins;
                if (q_bin_start < target_bins) {
                    float q = quantized[q_bin_start];
                    if (p > 0.0f && q > 0.0f) {
                        best_kl = -1.0f;
                        break;
                    }
                }
            } else {
                float p = (total > 0.0f) ? (float)histogram[i] / total : 0.0f;
                float q_sum = 0.0f;
                int width = end - start;
                if (width > 0 && width < target_bins) {
                    q_sum = expanded[i] * width;
                } else {
                    q_sum = expanded[i];
                }
                if (p > 0.0f && q_sum > 0.0f) {
                    best_kl += p * logf(p / q_sum);
                }
            }
        }

        if (best_kl < 0.0f) break;

        float kl = 0.0f;
        for (int i = 0; i < histogram_bins; i++) {
            int quant_idx = i * target_bins / histogram_bins;
            if (quant_idx >= target_bins) quant_idx = target_bins - 1;
            float p = (total > 0.0f) ? (float)histogram[i] / total : 0.0f;
            float q = quantized[quant_idx];
            if (p > 1e-10f && q > 1e-10f) {
                kl += p * logf(p / q);
            }
        }

        if (kl < best_kl) {
            best_kl = kl;
            best_bin = threshold;
        }
    }

    return (float)best_bin;
}

static std::unordered_map<std::string, float> collect_absmax(
    const std::vector<float> & activations,
    const std::vector<std::string> & tensor_names,
    int64_t n_tensors,
    int64_t elements_per_tensor) {

    std::unordered_map<std::string, float> absmax_map;
    for (int t = 0; t < (int)n_tensors; t++) {
        if (t >= (int)tensor_names.size()) break;
        float amax = 0.0f;
        const float * base = activations.data() + t * elements_per_tensor;
        for (int64_t i = 0; i < elements_per_tensor; i++) {
            float v = fabsf(base[i]);
            if (v > amax) amax = v;
        }
        absmax_map[tensor_names[t]] = amax;
    }
    return absmax_map;
}

static std::unordered_map<std::string, std::vector<uint64_t>> build_histograms(
    const std::vector<float> & activations,
    const std::vector<std::string> & tensor_names,
    const std::unordered_map<std::string, float> & absmax_map,
    int histogram_bins,
    int64_t elements_per_tensor) {

    std::unordered_map<std::string, std::vector<uint64_t>> hist_map;
    for (const auto & name : tensor_names) {
        auto it = absmax_map.find(name);
        if (it == absmax_map.end() || it->second < 1e-8f) continue;
        hist_map[name].assign(histogram_bins, 0);
    }

    int64_t n_tensors = (int64_t)tensor_names.size();
    for (int64_t t = 0; t < n_tensors; t++) {
        const std::string & name = tensor_names[t];
        auto hit = hist_map.find(name);
        if (hit == hist_map.end()) continue;
        auto amit = absmax_map.find(name);
        if (amit == absmax_map.end()) continue;

        float amax = amit->second;
        float inv_amax = (float)histogram_bins / amax;
        uint64_t * hist = hit->second.data();
        const float * base = activations.data() + t * elements_per_tensor;

        for (int64_t i = 0; i < elements_per_tensor; i++) {
            int bin = (int)(fabsf(base[i]) * inv_amax);
            if (bin >= histogram_bins) bin = histogram_bins - 1;
            hist[bin]++;
        }
    }
    return hist_map;
}

int calibrate_model(
    const char * model_path,
    const char * calibration_data,
    const char * output_table_path,
    const calibration_config & config) {

    std::ifstream data_file(calibration_data);
    if (!data_file.is_open()) {
        fprintf(stderr, "calibrate: cannot open calibration data: %s\n", calibration_data);
        return 1;
    }

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(data_file, line)) {
        if (!line.empty()) lines.push_back(line);
    }
    if (lines.empty()) {
        fprintf(stderr, "calibrate: no calibration samples found in %s\n", calibration_data);
        return 1;
    }

    int n_samples = std::min(config.n_calibration_samples, (int)lines.size());
    fprintf(stderr, "calibrate: using %d/%zu samples from %s\n", n_samples, lines.size(), calibration_data);

    (void)model_path;
    std::vector<calibration_scale_entry> entries;

    switch (config.method) {
        case calibration_config::KL_ENTROPY: {
            for (int s = 0; s < n_samples; s++) {
                (void)s;
            }
            fprintf(stderr, "calibrate: KL entropy calibration requires llama_decode integration (not yet connected)\n");
            break;
        }
        case calibration_config::ACIQ: {
            fprintf(stderr, "calibrate: ACIQ calibration requires weight tensor access (not yet connected)\n");
            break;
        }
        case calibration_config::EQ_REFINE: {
            fprintf(stderr, "calibrate: EQ refinement requires KL/ACIQ seed scales first\n");
            break;
        }
    }

    int rc = write_scale_table(output_table_path, entries);
    if (rc != 0) return rc;

    fprintf(stderr, "calibrate: wrote %zu entries to %s\n", entries.size(), output_table_path);
    return 0;
}
