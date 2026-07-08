#include "quant-recommender.h"
#include <cstdio>
#include <cstring>

int main(int argc, char ** argv) {
    fprintf(stderr, "Quant recommender: not yet connected to prober\n");

    prober_output probe = {};
    probe.n_layers = 32;
    probe.n_embd = 4096;
    probe.n_head = 32;
    probe.n_head_kv = 8;
    probe.param_bytes = 7 * 1024ULL * 1024ULL * 1024ULL;
    probe.has_attention = true;

    model_info info = {};
    info.name = "example-7b";
    info.arch = "llama";
    info.n_params_billion = 7;
    info.total_vram = 8 * 1024ULL * 1024ULL * 1024ULL;

    quant_recommendation rec = recommend(probe, info);
    printf("weight_type: %d\n", (int)rec.weight_type);
    printf("kv_type_k:    %d\n", (int)rec.kv_type_k);
    printf("kv_type_v:    %d\n", (int)rec.kv_type_v);
    printf("backend:      %s\n", rec.backend.c_str());
    printf("vram_mb:      %.1f\n", rec.vram_mb);
    printf("quality_pct:  %.1f\n", rec.quality_pct);
    return 0;
}
