#pragma once
#include "ggml.h"

struct round_trip_result {
    double max_error;    // max absolute error between original and round-tripped
    double mean_error;   // mean absolute error
    double rms_error;    // root-mean-square error
    double cos_sim;      // cosine similarity (1.0 = perfect)
};

struct round_trip_result round_trip_test(
    const float * weights,       // original F32 weights
    int64_t n_elements,
    enum ggml_type target_type); // quant type to test
