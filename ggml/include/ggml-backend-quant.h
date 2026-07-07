#ifndef GGML_BACKEND_QUANT_H
#define GGML_BACKEND_QUANT_H

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// Per-backend quant operations. NULL fields = not supported natively.
// If all fields are NULL, the type has no GPU kernel and falls back to CPU.
struct ggml_backend_quant_ops {
    int (*fattn_vec)(struct ggml_context * ctx, struct ggml_tensor * q,
                     struct ggml_tensor * k, struct ggml_tensor * v,
                     struct ggml_tensor * dst, int32_t n_head, int32_t n_kv, float scale,
                     float max_alibi_bias, int32_t n_logit_src);
    void (*dequantize_row)(const void * x, float * y, int64_t k);
    void (*quantize_row)(const float * x, void * y, int64_t nrows, int64_t n_per_row);
};

GGML_API void ggml_backend_register_quant_ops(
    ggml_backend_dev_t           dev,
    enum ggml_type               type,
    const struct ggml_backend_quant_ops * ops);

GGML_API bool ggml_backend_supports_quant(
    ggml_backend_dev_t dev,
    enum ggml_type     type);

GGML_API const struct ggml_backend_quant_ops * ggml_backend_get_quant_ops(
    ggml_backend_dev_t dev,
    enum ggml_type     type);

GGML_API void ggml_backend_register_standard_quants(
    ggml_backend_dev_t dev,
    bool is_cpu);

#ifdef __cplusplus
}
#endif
#endif // GGML_BACKEND_QUANT_H
