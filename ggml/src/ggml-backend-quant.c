#include "ggml-backend-quant.h"
#include <string.h>

#define MAX_BACKENDS 16
#define MAX_QUANT_OPS 64

struct backend_quant_entry {
    ggml_backend_dev_t dev;
    enum ggml_type type;
    struct ggml_backend_quant_ops ops;
};

static struct backend_quant_entry registry[MAX_BACKENDS * MAX_QUANT_OPS];
static int registry_count = 0;

void ggml_backend_register_quant_ops(
    ggml_backend_dev_t dev,
    enum ggml_type type,
    const struct ggml_backend_quant_ops * ops) {
    if (registry_count >= MAX_BACKENDS * MAX_QUANT_OPS) return;
    for (int i = 0; i < registry_count; i++) {
        if (registry[i].dev == dev && registry[i].type == type) {
            registry[i].ops = *ops;
            return;
        }
    }
    registry[registry_count].dev = dev;
    registry[registry_count].type = type;
    registry[registry_count].ops = ops ? *ops : (struct ggml_backend_quant_ops){0};
    registry_count++;
}

bool ggml_backend_supports_quant(ggml_backend_dev_t dev, enum ggml_type type) {
    for (int i = 0; i < registry_count; i++) {
        if (registry[i].dev == dev && registry[i].type == type) {
            return true;
        }
    }
    return false;
}

const struct ggml_backend_quant_ops * ggml_backend_get_quant_ops(
    ggml_backend_dev_t dev, enum ggml_type type) {
    for (int i = 0; i < registry_count; i++) {
        if (registry[i].dev == dev && registry[i].type == type) {
            return &registry[i].ops;
        }
    }
    return NULL;
}

void ggml_backend_register_standard_quants(ggml_backend_dev_t dev, bool is_cpu) {
    if (is_cpu) {
        for (int i = 0; i < GGML_TYPE_COUNT; i++) {
            ggml_backend_register_quant_ops(dev, (enum ggml_type)i, NULL);
        }
    } else {
        static const enum ggml_type gpu_types[] = {
            GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16,
            GGML_TYPE_Q4_0, GGML_TYPE_Q4_1,
            GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
            GGML_TYPE_Q8_0,
            GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K,
            GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ3_XXS,
            GGML_TYPE_IQ1_S, GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_NL,
            GGML_TYPE_IQ1_M, GGML_TYPE_IQ4_XS,
            GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0,
            GGML_TYPE_COUNT
        };
        for (int i = 0; gpu_types[i] != GGML_TYPE_COUNT; i++) {
            ggml_backend_register_quant_ops(dev, gpu_types[i], NULL);
        }
    }
}
