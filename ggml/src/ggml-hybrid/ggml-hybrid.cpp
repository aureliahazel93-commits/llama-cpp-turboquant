#include "ggml-hybrid.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-quant-caps.h"
#include <cstring>
#include <cstdio>

extern void hybrid_mul_mat_dispatch(
    ggml_tensor * dst,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    void * vk_queue);

extern void hybrid_thermal_update(float cpu_time_ms, float igpu_time_ms);

static struct {
    bool initialized;
    bool uma_capable;
    struct ggml_hybrid_thermal_state thermal;
    void * vk_queue;
} hybrid_g_state = {false, false, {0.0f, 0.0f, 0.0f, 0.0f, 0.5f}, nullptr};

static const char * hybrid_get_name(ggml_backend_t backend) {
    (void)backend;
    return "Hybrid-CPU+iGPU";
}

static void hybrid_free(ggml_backend_t backend) {
    hybrid_g_state.initialized = false;
    delete (ggml_backend *)backend;
}

static bool hybrid_supports_op(ggml_backend_t backend, const struct ggml_tensor * tensor) {
    (void)backend;
    if (tensor->type == GGML_TYPE_COUNT) return true;
    return true;
}

static enum ggml_status hybrid_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    (void)backend;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        if (node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID) {
            if (node->src[0] && node->src[1] && node->src[0]->data && node->src[1]->data) {
                hybrid_mul_mat_dispatch(node, node->src[0], node->src[1], hybrid_g_state.vk_queue);
                continue;
            }
        }
    }

    return GGML_STATUS_SUCCESS;
}

ggml_backend_t ggml_backend_hybrid_init(void) {
    hybrid_g_state.initialized = true;
    hybrid_g_state.uma_capable = false;
    hybrid_g_state.vk_queue = nullptr;

    ggml_backend * backend = new ggml_backend;
    memset(backend->guid, 0, sizeof(ggml_guid));
    backend->iface.get_name = hybrid_get_name;
    backend->iface.free = hybrid_free;
    backend->iface.graph_compute = hybrid_graph_compute;
    backend->device = nullptr;
    backend->context = nullptr;

    return backend;
}

ggml_backend_buffer_type_t ggml_backend_hybrid_buffer_type(void) {
    return nullptr;
}

bool ggml_backend_hybrid_is_uma(void) {
    return hybrid_g_state.uma_capable;
}

struct ggml_hybrid_thermal_state ggml_hybrid_get_thermal(void) {
    return hybrid_g_state.thermal;
}
