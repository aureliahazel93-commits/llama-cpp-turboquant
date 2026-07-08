#ifndef MODELFILE_H
#define MODELFILE_H

#include <string>

struct modelfile_quant {
    std::string weight_type;
    std::string cache_type_k;
    std::string cache_type_v;
};

struct modelfile_inference {
    int context_length;
    int n_gpu_layers;
    std::string backend_preference;
};

struct modelfile_speculative {
    std::string draft_model;
    float draft_ratio;
};

struct modelfile_server {
    int port;
    int max_sequences;
};

struct modelfile_config {
    std::string name;
    std::string base;
    modelfile_quant quantization;
    modelfile_inference inference;
    modelfile_speculative speculative;
    modelfile_server server;
};

bool modelfile_parse(const char * path, modelfile_config * out);

#endif
