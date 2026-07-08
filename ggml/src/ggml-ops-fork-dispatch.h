#pragma once

#include "ggml.h"
#include "ggml-fork-types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Upstream's dispatch function, renamed at definition site (ggml-cpu.c L1750).
// Externally visible so fork dispatch layer can call it.
extern void ggml_compute_forward_upstream(
    struct ggml_compute_params * params,
    struct ggml_tensor * tensor);

// Check whether an op ID is a fork op (>= 128, outside upstream range).
bool ggml_is_fork_op(int op);

// Check whether any source tensor of `tensor` uses a fork quant type.
bool ggml_has_fork_type(const struct ggml_tensor * tensor);

// Three-tier fork dispatch. Returns true if the call was handled.
// Returns false if the op is a standard upstream op — call ggml_compute_forward_upstream.
bool ggml_fork_compute_forward(
    struct ggml_compute_params * params,
    struct ggml_tensor * tensor);

// Startup validation. Returns 0 on success, -1 on failure.
int ggml_fork_ops_validate(void);

#ifdef __cplusplus
}
#endif
