#ifndef GGML_BACKEND_PLUGIN_H
#define GGML_BACKEND_PLUGIN_H

#include "ggml.h"
#include "ggml-backend.h"
#include <stddef.h>

#define GGML_MAX_DEVICES 8

struct ggml_backend_plugin_info {
    const char * name;
    int n_devices;
    size_t device_vram[GGML_MAX_DEVICES];
    int supported_types[256];
    int n_supported_types;
    float compute_score;
    bool is_uma;
    ggml_backend_t (*create_backend)(int device_idx);
    void (*register_quant_ops)(int device_idx);
};

typedef int (*ggml_backend_plugin_init_fn)(struct ggml_backend_plugin_info *);

#endif
