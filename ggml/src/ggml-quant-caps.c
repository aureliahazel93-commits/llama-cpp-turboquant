#include "ggml-quants.h"
#include "ggml-common.h"
#include "ggml-impl.h"

#include "ggml-quant-caps.h"

static const struct ggml_quant_caps quant_caps_table[GGML_TYPE_COUNT] = {
    [GGML_TYPE_TURBO3_0] = {
        .needs_wht       = true,
        .turbo_family    = true,
        .k_cache_capable = true,
        .v_cache_capable = true,
        .cpu_fallback_ok = true,
        .head_align      = 128,
    },
    [GGML_TYPE_TURBO4_0] = {
        .needs_wht       = true,
        .turbo_family    = true,
        .k_cache_capable = true,
        .v_cache_capable = true,
        .cpu_fallback_ok = true,
        .head_align      = 128,
    },
    [GGML_TYPE_TURBO2_0] = {
        .needs_wht       = true,
        .turbo_family    = true,
        .k_cache_capable = true,
        .v_cache_capable = true,
        .cpu_fallback_ok = true,
        .head_align      = 128,
    },
    [GGML_TYPE_PLANAR3_0] = {
        .needs_wht       = true,
        .needs_deferred  = true,
        .k_cache_capable = true,
        .v_cache_capable = true,
        .cpu_fallback_ok = true,
        .head_align      = 128,
    },
    [GGML_TYPE_ISO3_0] = {
        .needs_wht       = true,
        .needs_deferred  = true,
        .k_cache_capable = true,
        .v_cache_capable = true,
        .cpu_fallback_ok = true,
        .head_align      = 128,
    },
    [GGML_TYPE_PLANAR4_0] = {
        .needs_wht       = true,
        .needs_deferred  = true,
        .k_cache_capable = true,
        .v_cache_capable = true,
        .cpu_fallback_ok = true,
        .head_align      = 128,
    },
    [GGML_TYPE_ISO4_0] = {
        .needs_wht       = true,
        .needs_deferred  = true,
        .k_cache_capable = true,
        .v_cache_capable = true,
        .cpu_fallback_ok = true,
        .head_align      = 128,
    },
    [GGML_TYPE_TQ3_1S] = {
        .needs_wht       = true,
        .weight_capable  = true,
        .cpu_fallback_ok = true,
    },
    [GGML_TYPE_TQ4_1S] = {
        .needs_wht       = true,
        .weight_capable  = true,
        .cpu_fallback_ok = true,
    },
    [GGML_TYPE_STQ1_0] = {
        .weight_capable  = true,
        .cpu_fallback_ok = true,
    },
    [GGML_TYPE_TEQUILA] = {
        .weight_capable  = true,
        .cpu_fallback_ok = true,
    },
};
const struct ggml_quant_caps * ggml_get_quant_caps(enum ggml_type type) {
    GGML_ASSERT(type >= 0 && type < GGML_TYPE_COUNT);
    return &quant_caps_table[type];
}
