#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
#include "ggml-quants.h"

#include <stdint.h>
#include <string.h>
#include <math.h>
#include <float.h>

void quantize_row_tequila_ref(const float * GGML_RESTRICT x, block_tequila * GGML_RESTRICT y, int64_t k) {
    (void)x; (void)y; (void)k;
}

void dequantize_row_tequila(const block_tequila * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    (void)x; (void)y; (void)k;
}
