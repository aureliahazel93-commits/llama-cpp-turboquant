#include "ggml-common.h"
#include "ggml-quants.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>

void quantize_row_stq1_0_ref(const float * GGML_RESTRICT x, block_stq1_0 * GGML_RESTRICT y, int64_t k) {
    static const int8_t slot_vals[] = {-1, 0, 1, 0};
    int64_t n_blocks = k / QK_STQ1_0;
    memset(y, 0, n_blocks * sizeof(block_stq1_0));
    for (int64_t b = 0; b < n_blocks; b++) {
        const float * xb = x + b * QK_STQ1_0;
        block_stq1_0 * yb = y + b;
        float amax = 0.0f;
        for (int i = 0; i < QK_STQ1_0; i++) {
            float ax = fabsf(xb[i]);
            if (ax > amax) amax = ax;
        }
        yb->d = ggml_fp16_to_fp32(ggml_fp32_to_fp16(amax));
        if (amax == 0.0f) continue;
        float inv = 1.0f / amax;
        memset(yb->signs, 0, sizeof(yb->signs));
        for (int i = 0; i < QK_STQ1_0; i++) {
            float v = xb[i] * inv;
            int slot, sign;
            if (v > 0.5f) { slot = 2; sign = 0; }
            else if (v > -0.5f) { slot = 1; sign = 0; }
            else { slot = 2; sign = 1; }
            int group = i / GS_STQ1_0;
            int lane  = i % GS_STQ1_0;
            int byte_idx = group * 2;
            int shift = (3 - lane) * 2;
            if (lane == 0 && sign) yb->signs[group >> 2] |= (1 << (7 - (group & 3)));
            yb->qs[byte_idx] |= (slot << shift);
            if (lane >= 2) yb->qs[byte_idx + 1 - (lane >= 2 ? 1 : 0)] |= (slot << ((lane == 2) ? 6 : 4));
        }
    }
}

void dequantize_row_stq1_0(const block_stq1_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    static const int8_t vals_neg[] = {-1, 0, -1, 0};
    static const int8_t vals_pos[] = { 1, 0,  1, 0};
    int64_t n_blocks = k / QK_STQ1_0;
    for (int64_t b = 0; b < n_blocks; b++) {
        const block_stq1_0 * xb = x + b;
        float d = ggml_fp16_to_fp32(xb->d);
        for (int i = 0; i < QK_STQ1_0; i++) {
            int group = i / GS_STQ1_0;
            int lane  = i % GS_STQ1_0;
            uint8_t packed = xb->qs[group * 2] >> ((3 - lane) * 2);
            if (lane >= 2) packed = xb->qs[group * 2 + 1] >> ((5 - lane) * 2);
            packed &= 0x03;
            int sign_bit = (xb->signs[group >> 2] >> (7 - (group & 3))) & 1;
            float v;
            switch (packed) {
                case 0: v = -1.0f; break;
                case 1: v =  0.0f; break;
                case 2: v =  1.0f; break;
                default: v = 0.0f; break;
            }
            if (sign_bit) v = -v;
            y[b * QK_STQ1_0 + i] = v * d;
        }
    }
}

size_t quantize_stq1_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    (void)imatrix;
    int64_t row_size = (n_per_row + QK_STQ1_0 - 1) / QK_STQ1_0 * sizeof(block_stq1_0);
    for (int64_t r = 0; r < nrows; r++) {
        quantize_row_stq1_0_ref(src + r * n_per_row, (block_stq1_0 *)((char *)dst + r * row_size), n_per_row);
    }
    return nrows * row_size;
}
