#include "ggml.h"

void uma_buffer_init(void) {
}

void uma_buffer_free(void) {
}

void * uma_buffer_alloc(size_t size, size_t alignment) {
    (void)size;
    (void)alignment;
    return nullptr;
}
