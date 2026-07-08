#include "ggml-fork-types.h"
#include "ggml-quants.h"

static const struct ggml_type_traits fork_type_traits[] = {
#define GGML_FORK_QUANT_TYPE(enum_suffix, id, block, qk, name, fn) \
    [id - GGML_TYPE_FORK_BASE] = { \
        .type_name      = name, \
        .blck_size      = qk, \
        .type_size      = sizeof(block), \
        .is_quantized   = true, \
        .to_float       = (ggml_to_float_t) dequantize_row_##fn, \
        .from_float_ref = (ggml_from_float_t) quantize_row_##fn##_ref, \
    },
#include "ggml-fork-types.def"
#undef GGML_FORK_QUANT_TYPE
};

const struct ggml_type_traits * ggml_get_fork_type_traits(enum ggml_type type) {
    if (!GGML_IS_FORK_TYPE(type)) {
        return NULL;
    }
    return &fork_type_traits[type - GGML_TYPE_FORK_BASE];
}

size_t ggml_fork_dequant_to_f32(
    enum ggml_type type,
    const void * src,
    float * dst,
    int64_t nrows,
    int64_t ncols) {
    const struct ggml_type_traits * traits = ggml_get_type_traits(type);
    if (!traits || !traits->to_float) {
        return 0;
    }
    const int64_t blck_size = traits->blck_size;
    if (blck_size <= 0) {
        return 0;
    }
    const size_t type_size = traits->type_size;
    const ggml_to_float_t dequant_row = traits->to_float;
    const int64_t nrow_blocks = (ncols + blck_size - 1) / blck_size;
    const size_t row_bytes = nrow_blocks * type_size;

    for (int64_t r = 0; r < nrows; r++) {
        dequant_row(
            (const void *)((const char *)src + r * row_bytes),
            dst + r * ncols,
            ncols);
    }
    return nrows * ncols * sizeof(float);
}
