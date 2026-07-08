#include "ggml-ops-fork-dispatch.h"
#include "ggml-fork-types.h"
#include "ggml-quants.h"
#include "ops.h"
#include <string.h>

// =====================================================================
// Fork-op handlers (referenced by ggml-ops-fork.def)
// Implemented incrementally — stub returns error until implemented.
// =====================================================================

void compute_forward_dequant_fork(
    struct ggml_compute_params * params,
    struct ggml_tensor * dst) {
    GGML_ASSERT(false &&
        "compute_forward_dequant_fork: not yet implemented.");
    (void)params;
    (void)dst;
}

void compute_forward_quant_fork(
    struct ggml_compute_params * params,
    struct ggml_tensor * dst) {
    GGML_ASSERT(false &&
        "compute_forward_quant_fork: not yet implemented.");
    (void)params;
    (void)dst;
}

void compute_forward_mul_mat_fork(
    struct ggml_compute_params * params,
    struct ggml_tensor * dst) {
    GGML_ASSERT(false &&
        "compute_forward_mul_mat_fork: not yet implemented. "
        "Dequant A[Q4_0_G128], gemm with B[F32], quantize C -> dst.");
    (void)params;
    (void)dst;
}

void compute_forward_weight_transform(
    struct ggml_compute_params * params,
    struct ggml_tensor * dst) {
    GGML_ASSERT(false &&
        "compute_forward_weight_transform: not yet implemented.");
    (void)params;
    (void)dst;
}

// =====================================================================
// Fork-type-specific fast-path kernels (referenced by ggml-ops-fork-types.def)
// Implemented incrementally — stub returns error until implemented.
// =====================================================================

void compute_forward_mul_mat_turbo2_0(
    struct ggml_compute_params * params,
    struct ggml_tensor * dst) {
    GGML_ASSERT(false &&
        "compute_forward_mul_mat_turbo2_0: not yet implemented. "
        "Priority 1: this is 90%+ of inference compute.");
    (void)params;
    (void)dst;
}

void compute_forward_mul_mat_tq4_1s(
    struct ggml_compute_params * params,
    struct ggml_tensor * dst) {
    GGML_ASSERT(false &&
        "compute_forward_mul_mat_tq4_1s: not yet implemented.");
    (void)params;
    (void)dst;
}

void compute_forward_add_turbo2_0(
    struct ggml_compute_params * params,
    struct ggml_tensor * dst) {
    GGML_ASSERT(false &&
        "compute_forward_add_turbo2_0: not yet implemented.");
    (void)params;
    (void)dst;
}

void compute_forward_mul_turbo2_0(
    struct ggml_compute_params * params,
    struct ggml_tensor * dst) {
    GGML_ASSERT(false &&
        "compute_forward_mul_turbo2_0: not yet implemented.");
    (void)params;
    (void)dst;
}
