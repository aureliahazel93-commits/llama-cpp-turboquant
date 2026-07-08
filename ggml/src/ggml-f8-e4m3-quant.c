#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
#include "ggml-quants.h"

#include <stdint.h>
#include <string.h>
#include <math.h>

static float fp8_e4m3_to_f32(uint8_t v) {
    uint32_t sign = (v >> 7) & 1;
    uint32_t exp  = (v >> 3) & 0xF;
    uint32_t mant = v & 0x7;

    if (exp == 0) {
        float val = ldexpf((float)mant / 8.0f, 1 - 7);
        return sign ? -val : val;
    }

    if (exp == 15) {
        if (mant == 0) {
            return sign ? -INFINITY : INFINITY;
        }
        return NAN;
    }

    float val = ldexpf(1.0f + (float)mant / 8.0f, (int)exp - 7);
    return sign ? -val : val;
}

static uint8_t f32_to_fp8_e4m3(float v) {
    if (isnan(v) || isinf(v)) {
        uint32_t bits;
        memcpy(&bits, &v, 4);
        uint32_t sign = (bits >> 31) & 1;
        if (isinf(v)) return (uint8_t)((sign << 7) | 0x78);
        return (uint8_t)((sign << 7) | 0x7F);
    }
    if (v == 0.0f) return 0;
    if (v == -0.0f) return 0x80;

    uint32_t bits;
    memcpy(&bits, &v, 4);
    uint32_t fsign = (bits >> 31) & 1;

    bool negative = fsign != 0;
    float av = negative ? -v : v;

    if (av >= 240.0f) {
        return (uint8_t)((fsign << 7) | 0x7F);
    }

    float val = av;
    if (val < 0.015625f) {
        uint8_t result = 0;
        for (int i = 7; i >= 0; i--) {
            float threshold = ldexpf(1.0f / 8.0f, i - 6);
            if (val >= threshold) {
                result |= (1 << i);
                val -= threshold;
            }
        }
        return (uint8_t)((fsign << 7) | result);
    }

    int exp_val;
    frexpf(val, &exp_val);
    exp_val--;

    if (exp_val < -6) exp_val = -6;
    if (exp_val > 7) exp_val = 7;

    float norm_val = ldexpf(val, -(exp_val));
    float mant_f = (norm_val - 1.0f) * 8.0f;
    uint32_t mant_i = (uint32_t)(mant_f + 0.5f);
    if (mant_i > 7) {
        mant_i = 7;
        exp_val++;
        if (exp_val > 7) {
            exp_val = 7;
            mant_i = 7;
        }
    }

    uint8_t result = (uint8_t)((fsign << 7) | ((exp_val + 7) << 3) | mant_i);
    return result;
}

void quantize_row_f8_e4m3_ref(const float * GGML_RESTRICT x, block_f8_e4m3 * GGML_RESTRICT y, int64_t k) {
    static_assert(QK8_F8_E4M3 == QK_K, "QK8_F8_E4M3 must equal QK_K");

    const int nb = k / QK8_F8_E4M3;

    for (int i = 0; i < nb; i++) {
        const float * xb = x + i * QK8_F8_E4M3;
        block_f8_e4m3 * yb = y + i;

        float amax = 0.0f;
        for (int j = 0; j < QK8_F8_E4M3; j++) {
            float av = fabsf(xb[j]);
            if (av > amax) amax = av;
        }

        float d = amax / 240.0f;
        if (amax == 0.0f) d = 0.0f;
        yb->d = d;

        for (int j = 0; j < QK8_F8_E4M3; j++) {
            if (d != 0.0f) {
                yb->q[j] = f32_to_fp8_e4m3(xb[j] / d);
            } else {
                yb->q[j] = 0;
            }
        }
    }
}

void dequantize_row_f8_e4m3(const block_f8_e4m3 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    const int nb = k / QK8_F8_E4M3;

    for (int i = 0; i < nb; i++) {
        const block_f8_e4m3 * xb = x + i;
        float * yb = y + i * QK8_F8_E4M3;

        float d = xb->d;
        for (int j = 0; j < QK8_F8_E4M3; j++) {
            yb[j] = fp8_e4m3_to_f32(xb->q[j]) * d;
        }
    }
}

size_t quantize_f8_e4m3(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    const int64_t k = nrows * n_per_row;
    quantize_row_f8_e4m3_ref(src, (block_f8_e4m3 *)dst, k);
    return k;
}
