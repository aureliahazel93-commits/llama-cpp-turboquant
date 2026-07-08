#include "ggml-quants.h"
#include "ggml-common.h"
#include "ggml-impl.h"

#include <math.h>
#include <string.h>

#define Q1_0_G128_BLOCK_SIZE 128
#define Q1_0_G128_BYTES     18

void dequantize_row_q1_0_g128(const block_q1_0_g128 * GGML_RESTRICT blocks,
                                float * GGML_RESTRICT out,
                                int64_t n) {
    static const int block_size = Q1_0_G128_BLOCK_SIZE;
    const int64_t nrows = n / block_size;
    for (int64_t r = 0; r < nrows; r++) {
        const block_q1_0_g128 * b = &blocks[r];
        float d = GGML_FP16_TO_FP32(b->d);
        const uint8_t * qs = b->qs;
        for (int i = 0; i < block_size; i++) {
            int byte_idx = i / 8;
            int bit_idx = i % 8;
            float bit = (float)((qs[byte_idx] >> bit_idx) & 1);
            out[r * block_size + i] = d * (2.0f * bit - 1.0f);
        }
    }
}

void quantize_row_q1_0_g128_ref(const float * GGML_RESTRICT x,
                                 block_q1_0_g128 * GGML_RESTRICT y,
                                 int64_t n) {
    static const int block_size = Q1_0_G128_BLOCK_SIZE;
    const int64_t nrows = n / block_size;
    for (int64_t r = 0; r < nrows; r++) {
        float amax = 0.0f;
        for (int i = 0; i < block_size; i++) {
            float a = fabsf(x[r * block_size + i]);
            if (a > amax) amax = a;
        }
        float d = amax;
        if (d == 0.0f) d = 1.0f;
        y[r].d = GGML_FP32_TO_FP16(d);
        memset(y[r].qs, 0, sizeof(y[r].qs));
        for (int i = 0; i < block_size; i++) {
            int byte_idx = i / 8;
            int bit_idx = i % 8;
            if (x[r * block_size + i] >= 0.0f) {
                y[r].qs[byte_idx] |= (1 << bit_idx);
            }
        }
    }
}

int64_t quantize_q1_0_g128(const float * src, void * dst, int64_t n, int k, int64_t * hist) {
    (void)k;
    (void)hist;
    quantize_row_q1_0_g128_ref(src, (block_q1_0_g128 *)dst, n);
    return n / Q1_0_G128_BLOCK_SIZE * Q1_0_G128_BYTES;
}
