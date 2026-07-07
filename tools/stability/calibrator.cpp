// llama-calibrate: calibration tool for quantization scale tables
// Usage: llama-calibrate --model <model.gguf> --calibrate <data.txt> --output <table> [--method kl|aciq|eq]
//
// Reads calibration text data, forwards through the F16 model, collects
// activation statistics, and produces a scale table for use with llama-quantize.
//
// Methods:
//   kl     - KL entropy (default): 3-pass histogram + threshold sweep, highest accuracy
//   aciq   - ACIQ: single-pass Gaussian clip, fast
//   eq     - EQ refinement: grid-search cosine similarity, highest accuracy but expensive

#include "calibrator.h"
#include "common.h"
#include "llama.h"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char ** argv) {
    common_params params;

    const char * model_path    = nullptr;
    const char * calib_data    = nullptr;
    const char * output_table  = nullptr;
    const char * method_str    = "kl";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--calibrate" && i + 1 < argc) {
            calib_data = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_table = argv[++i];
        } else if (arg == "--method" && i + 1 < argc) {
            method_str = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            fprintf(stderr, "Usage: llama-calibrate --model <model.gguf> --calibrate <data.txt> --output <table> [--method kl|aciq|eq]\n");
            return 0;
        }
    }

    if (!model_path || !calib_data || !output_table) {
        fprintf(stderr, "Usage: llama-calibrate --model <model.gguf> --calibrate <data.txt> --output <table> [--method kl|aciq|eq]\n");
        return 1;
    }

    calibration_config config;

    if (strcmp(method_str, "kl") == 0) {
        config.method = calibration_config::KL_ENTROPY;
    } else if (strcmp(method_str, "aciq") == 0) {
        config.method = calibration_config::ACIQ;
    } else if (strcmp(method_str, "eq") == 0) {
        config.method = calibration_config::EQ_REFINE;
    } else {
        fprintf(stderr, "Unknown method '%s'. Use: kl, aciq, or eq.\n", method_str);
        return 1;
    }

    fprintf(stderr, "Calibrating model: %s\n", model_path);
    fprintf(stderr, "Calibration data: %s\n", calib_data);
    fprintf(stderr, "Output table:     %s\n", output_table);
    fprintf(stderr, "Method:           %s\n", method_str);

    int result = calibrate_model(model_path, calib_data, output_table, config);

    if (result == 0) {
        fprintf(stderr, "Calibration complete: %s\n", output_table);
    } else {
        fprintf(stderr, "Calibration failed with code %d\n", result);
    }

    return result;
}
