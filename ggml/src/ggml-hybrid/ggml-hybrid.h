#pragma once
#include "ggml-backend.h"
#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

GGML_API ggml_backend_t ggml_backend_hybrid_init(void);

GGML_API ggml_backend_buffer_type_t ggml_backend_hybrid_buffer_type(void);

GGML_API bool ggml_backend_hybrid_is_uma(void);

struct ggml_hybrid_thermal_state {
    float cpu_temp_c;
    float igpu_temp_c;
    float cpu_utilization;
    float igpu_utilization;
    float cpu_to_igpu_ratio;
};

GGML_API struct ggml_hybrid_thermal_state ggml_hybrid_get_thermal(void);

#ifdef __cplusplus
}
#endif
