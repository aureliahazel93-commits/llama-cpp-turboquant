#pragma once
#include "ggml.h"
#include <string>
#include <vector>
#include <cstdint>

struct calibration_scale_entry {
    std::string tensor_name;
    std::string scale_type;
    std::vector<float> scales;
};

struct calibration_config {
    enum method { KL_ENTROPY, ACIQ, EQ_REFINE } method = KL_ENTROPY;
    int n_calibration_samples = 500;
    int histogram_bins = 4096;
    int target_bins = 128;
    float eq_search_min = 0.5f;
    float eq_search_max = 2.0f;
    int eq_grid_steps = 100;
    int eq_max_samples = 50;
    bool calibrate_kv_cache = true;
    bool fuse_requantize = true;
};

float compute_kl_threshold(
    const uint64_t histogram[],
    int histogram_bins,
    int target_bins);

float compute_aciq_scale(float absmax, int64_t n_elements, int n_bits);

float refine_layer_scale(
    const float * ref_output,
    const float * quant_output,
    int n_elements,
    float current_scale,
    float search_min = 0.5f,
    float search_max = 2.0f,
    int grid_steps = 100);

int write_scale_table(
    const char * path,
    const std::vector<calibration_scale_entry> & entries);

std::vector<calibration_scale_entry> read_scale_table(
    const char * path);

int calibrate_model(
    const char * model_path,
    const char * calibration_data,
    const char * output_table_path,
    const calibration_config & config);

int quantize_with_scales(
    const char * model_path,
    const char * table_path,
    const char * output_path,
    enum ggml_type target_type);
