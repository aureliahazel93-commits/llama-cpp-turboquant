#ifndef GGML_FORK_TYPES_H
#define GGML_FORK_TYPES_H

#include "ggml.h"
#include "ggml-common.h"

#define GGML_TYPE_FORK_BASE  244
#define GGML_TYPE_FORK_MAX   255
#define GGML_IS_FORK_TYPE(t) ((t) >= GGML_TYPE_FORK_BASE && (t) <= GGML_TYPE_FORK_MAX)

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_type_traits;
const struct ggml_type_traits * ggml_get_fork_type_traits(enum ggml_type type);

size_t ggml_fork_dequant_to_f32(
    enum ggml_type type,
    const void * src,
    float * dst,
    int64_t nrows,
    int64_t ncols);

#ifdef __cplusplus
}
#endif

#endif
