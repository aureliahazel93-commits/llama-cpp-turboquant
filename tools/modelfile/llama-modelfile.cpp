#include "common/modelfile.h"
#include <cstdio>
#include <cstring>

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: llama-modelfile <modelfile>\n");
        return 1;
    }

    modelfile_config cfg;
    if (!modelfile_parse(argv[1], &cfg)) {
        fprintf(stderr, "error: failed to parse %s\n", argv[1]);
        return 1;
    }

    printf("[model]\n");
    printf("  name = %s\n", cfg.name.c_str());
    printf("  base = %s\n", cfg.base.c_str());
    printf("[quantization]\n");
    printf("  weight_type  = %s\n", cfg.quantization.weight_type.c_str());
    printf("  cache_type_k = %s\n", cfg.quantization.cache_type_k.c_str());
    printf("  cache_type_v = %s\n", cfg.quantization.cache_type_v.c_str());
    printf("[inference]\n");
    printf("  context_length    = %d\n", cfg.inference.context_length);
    printf("  n_gpu_layers      = %d\n", cfg.inference.n_gpu_layers);
    printf("  backend_preference = %s\n", cfg.inference.backend_preference.c_str());
    printf("[speculative]\n");
    printf("  draft_model = %s\n", cfg.speculative.draft_model.c_str());
    printf("  draft_ratio = %.2f\n", cfg.speculative.draft_ratio);
    printf("[server]\n");
    printf("  port           = %d\n", cfg.server.port);
    printf("  max_sequences  = %d\n", cfg.server.max_sequences);
    return 0;
}
