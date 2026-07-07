#include "round_trip.h"
#include "ggml-common.h"
#include "ggml-quants.h"
#include <cmath>
#include <algorithm>
#include <vector>

static void dequantize_to_f32(const void * data, enum ggml_type type, int64_t n, float * dst) {
    const auto * traits = ggml_get_type_traits(type);
    if (traits && traits->to_float) {
        traits->to_float(data, dst, n);
    }
}

static void quantize_from_f32(const float * src, void * dst, enum ggml_type type, int64_t n) {
    const auto * traits = ggml_get_type_traits(type);
    if (traits && traits->from_float_ref) {
        int64_t n_per_row = ggml_blck_size(type) > 0 ? n / ggml_blck_size(type) : n;
        if (n_per_row == 0) n_per_row = n;
        traits->from_float_ref(src, dst, n_per_row);
    }
}

struct round_trip_result round_trip_test(const float * weights, int64_t n_elements, enum ggml_type target_type) {
    struct round_trip_result result = {0.0, 0.0, 0.0, 0.0};

    if (n_elements <= 0 || target_type == GGML_TYPE_F32 || target_type == GGML_TYPE_F16) {
        return result;
    }

    int64_t blck_size = ggml_blck_size(target_type);
    if (blck_size <= 0) return result;

    int64_t n_rows = n_elements / blck_size;
    size_t quant_size = ggml_row_size(target_type, blck_size);

    std::vector<char> quant_buf(n_rows * quant_size);
    std::vector<float> dequant_buf(n_elements);

    // Quantize F32 → target type
    for (int64_t r = 0; r < n_rows; r++) {
        quantize_from_f32(weights + r * blck_size, quant_buf.data() + r * quant_size, target_type, blck_size);
    }

    // Dequantize target type → F32
    for (int64_t r = 0; r < n_rows; r++) {
        dequantize_to_f32(quant_buf.data() + r * quant_size, target_type, blck_size, dequant_buf.data() + r * blck_size);
    }

    // Compute error metrics
    double sum_sq_err = 0.0;
    double dot = 0.0, mag_orig = 0.0, mag_deq = 0.0;

    for (int64_t i = 0; i < n_elements; i++) {
        double err = std::abs((double)weights[i] - (double)dequant_buf[i]);
        result.mean_error += err;
        result.max_error = std::max(result.max_error, err);
        sum_sq_err += err * err;

        dot      += weights[i] * dequant_buf[i];
        mag_orig += weights[i] * weights[i];
        mag_deq  += dequant_buf[i] * dequant_buf[i];
    }

    result.mean_error /= n_elements;
    result.rms_error  = sqrt(sum_sq_err / n_elements);
    result.cos_sim    = (mag_orig > 0.0 && mag_deq > 0.0) ? dot / (sqrt(mag_orig) * sqrt(mag_deq)) : 0.0;

    return result;
}
