#ifndef GGML_QUANT_CAPS_H
#define GGML_QUANT_CAPS_H

#include "ggml.h"
// Note: enum ggml_type is declared in ggml.h before this header is included.

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_quant_caps {
    bool     needs_wht;
    bool     turbo_family;
    bool     needs_deferred;
    bool     k_cache_capable;
    bool     v_cache_capable;
    bool     weight_capable;
    bool     cpu_fallback_ok;
    uint32_t head_align;
};

GGML_API const struct ggml_quant_caps * ggml_get_quant_caps(enum ggml_type type);

#ifdef __cplusplus
}
#endif

static inline bool ggml_type_needs_wht(enum ggml_type type) {
    return ggml_get_quant_caps(type)->needs_wht;
}

static inline bool ggml_type_needs_deferred(enum ggml_type type) {
    return ggml_get_quant_caps(type)->needs_deferred;
}

static inline bool ggml_type_is_turbo_family(enum ggml_type type) {
    return ggml_get_quant_caps(type)->turbo_family;
}

static inline uint32_t ggml_type_head_align(enum ggml_type type) {
    return ggml_get_quant_caps(type)->head_align;
}

#endif // GGML_QUANT_CAPS_H
