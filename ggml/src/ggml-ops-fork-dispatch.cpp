#include "ggml-ops-fork-dispatch.h"
#include "ggml-fork-types.h"
#include "ggml-quants.h"
#include "ops.h"
#include <string.h>
#include <stdio.h>

// Handler stubs defined in ggml-ops-fork-kernels.cpp
extern void compute_forward_dequant_fork(struct ggml_compute_params *, struct ggml_tensor *);
extern void compute_forward_quant_fork(struct ggml_compute_params *, struct ggml_tensor *);
extern void compute_forward_mul_mat_fork(struct ggml_compute_params *, struct ggml_tensor *);
extern void compute_forward_weight_transform(struct ggml_compute_params *, struct ggml_tensor *);
extern void compute_forward_mul_mat_turbo2_0(struct ggml_compute_params *, struct ggml_tensor *);
extern void compute_forward_mul_mat_tq4_1s(struct ggml_compute_params *, struct ggml_tensor *);
extern void compute_forward_add_turbo2_0(struct ggml_compute_params *, struct ggml_tensor *);
extern void compute_forward_mul_turbo2_0(struct ggml_compute_params *, struct ggml_tensor *);

// =====================================================================
// Tier 1 table — auto-generated from ggml-ops-fork.def
// Maps fork op ID → handler.
// =====================================================================

typedef void (*fork_op_handler_t)(
    struct ggml_compute_params * params,
    struct ggml_tensor * tensor);

#define GGML_FORK_OP(name, id, cat, handler) { id, handler },

static const struct {
    int               op_id;
    fork_op_handler_t handler;
} fork_op_table[] = {
    #include "ggml-ops-fork.def"
};

#undef GGML_FORK_OP

static const int fork_op_count =
    (int)(sizeof(fork_op_table) / sizeof(fork_op_table[0]));

// =====================================================================
// Tier 2 table — auto-generated from ggml-ops-fork-types.def
// Maps (upstream op × fork quant type) → specialized handler.
// =====================================================================

#define GGML_FORK_TYPE_OP(op, type_id, handler) { op, type_id, handler },

static const struct {
    int               op;
    int               type_id;
    fork_op_handler_t handler;
} fork_type_op_table[] = {
    #include "ggml-ops-fork-types.def"
};

#undef GGML_FORK_TYPE_OP

static const int fork_type_op_count =
    (int)(sizeof(fork_type_op_table) / sizeof(fork_type_op_table[0]));

// =====================================================================
// Tier 3: dequant-to-F32 fallback
//
// Called when no specialized Tier 2 handler is registered.
// The upstream per-op handler reads src[i]->type to choose a sub-handler.
// If the sub-handler's case list already covers the fork type (ops.cpp patch),
// it runs correctly. Otherwise it hits GGML_ABORT.
//
// Day-one strategy: Tier 3 provides the CORRECTNESS path only for ops that
// already have fork types in their case list in ops.cpp. Tier 2 handlers are
// added incrementally for ops that need performance (MUL_MAT first).
// =====================================================================

static void fork_tier3_fallback(
    struct ggml_compute_params * params,
    struct ggml_tensor * tensor) {

    GGML_ASSERT(false &&
        "fork Tier 3 fallback reached for an op without a registered "
        "(op, fork_type) handler and without ops.cpp fork-type patches. "
        "Add a GGML_FORK_TYPE_OP entry in ggml-ops-fork-types.def, "
        "or patch ops.cpp to add fork types to the op's case list.");
    (void)params;
    (void)tensor;
}

// =====================================================================
// Internal helpers
// =====================================================================

static bool ggml_is_fork_op_internal(int op) {
    for (int i = 0; i < fork_op_count; i++) {
        if (fork_op_table[i].op_id == op) return true;
    }
    return false;
}

static bool ggml_has_fork_type_internal(const struct ggml_tensor * tensor) {
    if (GGML_IS_FORK_TYPE(tensor->type)) return true;
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (tensor->src[i] && GGML_IS_FORK_TYPE(tensor->src[i]->type))
            return true;
    }
    return false;
}

static bool ggml_try_fork_type_handler(
    struct ggml_compute_params * params,
    struct ggml_tensor * tensor) {

    for (int i = 0; i < fork_type_op_count; i++) {
        if (tensor->op != fork_type_op_table[i].op) continue;

        const int wanted_type = fork_type_op_table[i].type_id;
        bool type_matches = false;

        if (GGML_IS_FORK_TYPE(tensor->type) && tensor->type == wanted_type)
            type_matches = true;

        for (int j = 0; j < GGML_MAX_SRC; j++) {
            const ggml_tensor * src = tensor->src[j];
            if (src && GGML_IS_FORK_TYPE(src->type) && src->type == wanted_type) {
                type_matches = true;
                break;
            }
        }

        if (type_matches) {
            fork_type_op_table[i].handler(params, tensor);
            return true;
        }
    }
    return false;
}

// =====================================================================
// Public API
// =====================================================================

bool ggml_is_fork_op(int op) {
    return ggml_is_fork_op_internal(op);
}

bool ggml_has_fork_type(const struct ggml_tensor * tensor) {
    return ggml_has_fork_type_internal(tensor);
}

bool ggml_fork_compute_forward(
    struct ggml_compute_params * params,
    struct ggml_tensor * tensor) {

    // ---- Tier 1: pure fork op ----
    if (ggml_is_fork_op_internal(tensor->op)) {
        for (int i = 0; i < fork_op_count; i++) {
            if (tensor->op == fork_op_table[i].op_id) {
                fork_op_table[i].handler(params, tensor);
                return true;
            }
        }
    }

    // ---- Tier 2: upstream op with fork-typed source ----
    if (ggml_has_fork_type_internal(tensor)) {
        if (ggml_try_fork_type_handler(params, tensor)) return true;

        // ---- Tier 3: fallback (correctness, not performance) ----
        fork_tier3_fallback(params, tensor);
        return true;
    }

    // ---- Not a fork concern — let upstream handle it ----
    return false;
}

// =====================================================================
// Startup validation (call once from ggml_backend_cpu_init)
// =====================================================================

int ggml_fork_ops_validate(void) {
    int failures = 0;

    for (int i = 0; i < fork_op_count; i++) {
        if (fork_op_table[i].handler == NULL) {
            fprintf(stderr,
                "[fork-validate] ERROR: fork_op_table[%d] op_id=%d has NULL handler\n",
                i, fork_op_table[i].op_id);
            failures++;
        }
    }

    for (int i = 0; i < fork_type_op_count; i++) {
        if (fork_type_op_table[i].handler == NULL) {
            fprintf(stderr,
                "[fork-validate] ERROR: fork_type_op_table[%d] op=%d type=%d has NULL handler\n",
                i, fork_type_op_table[i].op, fork_type_op_table[i].type_id);
            failures++;
        }
    }

    for (int i = 0; i < fork_op_count; i++) {
        if (fork_op_table[i].op_id < GGML_TYPE_FORK_BASE) {
            fprintf(stderr,
                "[fork-validate] ERROR: fork_op_table[%d] op_id=%d < FORK_BASE=%d\n",
                i, fork_op_table[i].op_id, GGML_TYPE_FORK_BASE);
            failures++;
        }
    }

    if (failures > 0) {
        fprintf(stderr, "[fork-validate] %d check(s) FAILED.\n", failures);
        return -1;
    }

    fprintf(stdout,
        "[fork-validate] %d fork ops + %d fork-type ops registered. OK.\n",
        fork_op_count, fork_type_op_count);
    return 0;
}
