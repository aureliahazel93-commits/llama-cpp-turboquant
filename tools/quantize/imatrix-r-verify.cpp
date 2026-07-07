#include "llama-impl.h"
#include "llama-model.h"
#include "llama-model-loader.h"
#include "llama-ext.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

// Dequantize a ggml tensor to f32
static void dequant_to_f32(const ggml_tensor * tensor, float * output) {
    const size_t n = ggml_nelements(tensor);
    if (tensor->type == GGML_TYPE_F32) {
        memcpy(output, tensor->data, n * sizeof(float));
    } else if (tensor->type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row((const ggml_fp16_t *)tensor->data, output, n);
    } else if (tensor->type == GGML_TYPE_BF16) {
        ggml_bf16_to_fp32_row((const ggml_bf16_t *)tensor->data, output, n);
    } else {
        const auto * qtype = ggml_get_type_traits(tensor->type);
        if (qtype && qtype->to_float) {
            qtype->to_float(tensor->data, output, n);
        } else {
            fprintf(stderr, "Error: cannot dequantize type %s\n", ggml_type_name(tensor->type));
            exit(1);
        }
    }
}

// Dequantize a raw buffer to f32
static void dequant_buf_to_f32(const uint8_t * data, size_t n, ggml_type type, float * output) {
    if (type == GGML_TYPE_F32) {
        memcpy(output, data, n * sizeof(float));
    } else if (type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row((const ggml_fp16_t *)data, output, n);
    } else if (type == GGML_TYPE_BF16) {
        ggml_bf16_to_fp32_row((const ggml_bf16_t *)data, output, n);
    } else {
        const auto * qtype = ggml_get_type_traits(type);
        if (qtype && qtype->to_float) {
            qtype->to_float((void *)data, output, n);
        } else {
            fprintf(stderr, "Error: cannot dequantize buf type %s\n", ggml_type_name(type));
            exit(1);
        }
    }
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <original.gguf> <converted.gguf> [tensor_name]\n", argv[0]);
        fprintf(stderr, "  tensor_name defaults to blk.0.ffn_gate_up_exps.weight\n");
        return 1;
    }

    const char * fname_orig  = argv[1];
    const char * fname_conv  = argv[2];
    const char * tensor_name = argc > 3 ? argv[3] : "blk.0.ffn_gate_up_exps.weight";

    // Load original model
    struct ggml_context * ctx_orig = nullptr;
    struct gguf_init_params params = { true, &ctx_orig };
    struct gguf_context * gguf_orig = gguf_init_from_file(fname_orig, params);
    if (!gguf_orig) {
        fprintf(stderr, "Error: cannot open original model '%s'\n", fname_orig);
        return 1;
    }

    // Load converted model
    struct ggml_context * ctx_conv = nullptr;
    struct gguf_init_params params2 = { true, &ctx_conv };
    struct gguf_context * gguf_conv = gguf_init_from_file(fname_conv, params2);
    if (!gguf_conv) {
        fprintf(stderr, "Error: cannot open converted model '%s'\n", fname_conv);
        return 1;
    }

    // Find original tensor
    struct ggml_tensor * orig_tensor = nullptr;
    for (struct ggml_tensor * t = ggml_get_first_tensor(ctx_orig); t; t = ggml_get_next_tensor(ctx_orig, t)) {
        if (strcmp(t->name, tensor_name) == 0) {
            orig_tensor = t;
            break;
        }
    }
    if (!orig_tensor) {
        fprintf(stderr, "Error: tensor '%s' not found in original model\n", tensor_name);
        return 1;
    }

    // Load original tensor data
    const int tensor_idx_orig = gguf_find_tensor(gguf_orig, tensor_name);
    if (tensor_idx_orig < 0) {
        fprintf(stderr, "Error: tensor '%s' not found in original GGUF\n", tensor_name);
        return 1;
    }
    const size_t orig_offset = gguf_get_data_offset(gguf_orig) + gguf_get_tensor_offset(gguf_orig, tensor_idx_orig);
    FILE * f_orig = fopen(fname_orig, "rb");
    fseek(f_orig, orig_offset, SEEK_SET);
    const size_t orig_nbytes = ggml_nbytes(orig_tensor);
    std::vector<uint8_t> orig_data(orig_nbytes);
    fread(orig_data.data(), 1, orig_nbytes, f_orig);
    fclose(f_orig);
    orig_tensor->data = orig_data.data();

    const int64_t ne0 = orig_tensor->ne[0];
    const int64_t ne1 = orig_tensor->ne[1];
    const int64_t ne2 = orig_tensor->ne[2];
    const int64_t n_expert = ne2;
    const int64_t elem_per_expert = ne0 * ne1;

    fprintf(stderr, "=== imatrix-r verify ===\n");
    fprintf(stderr, "Tensor: %s  type=%s  shape={%ld, %ld, %ld}\n",
            tensor_name, ggml_type_name(orig_tensor->type),
            (long)ne0, (long)ne1, (long)ne2);
    fprintf(stderr, "n_expert=%ld  elem_per_expert=%ld\n\n", (long)n_expert, (long)elem_per_expert);

    // Dequantize original to f32
    std::vector<float> orig_f32(ggml_nelements(orig_tensor));
    dequant_to_f32(orig_tensor, orig_f32.data());

    // Find base tensor in converted model
    char base_name[256];
    snprintf(base_name, sizeof(base_name), "%s_base", tensor_name);

    struct ggml_tensor * base_tensor = nullptr;
    int base_gguf_idx = -1;
    for (struct ggml_tensor * t = ggml_get_first_tensor(ctx_conv); t; t = ggml_get_next_tensor(ctx_conv, t)) {
        if (strcmp(t->name, base_name) == 0) {
            base_tensor = t;
            break;
        }
    }
    if (!base_tensor) {
        fprintf(stderr, "Error: base tensor '%s' not found in converted model\n", base_name);
        return 1;
    }
    base_gguf_idx = gguf_find_tensor(gguf_conv, base_name);
    const size_t base_offset = gguf_get_data_offset(gguf_conv) + gguf_get_tensor_offset(gguf_conv, base_gguf_idx);
    FILE * f_conv = fopen(fname_conv, "rb");
    fseek(f_conv, base_offset, SEEK_SET);
    const size_t base_nbytes = ggml_nbytes(base_tensor);
    std::vector<uint8_t> base_data(base_nbytes);
    fread(base_data.data(), 1, base_nbytes, f_conv);
    fclose(f_conv);
    base_tensor->data = base_data.data();

    fprintf(stderr, "Base: %s  type=%s  shape={%ld, %ld, %ld}\n",
            base_name, ggml_type_name(base_tensor->type),
            (long)base_tensor->ne[0], (long)base_tensor->ne[1], (long)base_tensor->ne[2]);

    // Find tier tensors and remapping tables
    struct tier_info {
        std::string res_name;
        std::string idx_name;
        std::string mask_name;
        ggml_tensor * res_tensor = nullptr;
        ggml_tensor * idx_tensor = nullptr;
        ggml_tensor * mask_tensor = nullptr;
        std::vector<uint8_t> res_data;
        std::vector<uint8_t> idx_data;
        std::vector<uint8_t> mask_data;
        int gguf_res_idx = -1;
        int gguf_idx_idx = -1;
        int gguf_mask_idx = -1;
    };

    std::vector<tier_info> tiers(3);
    for (int t = 0; t < 3; ++t) {
        tiers[t].res_name = std::string(tensor_name) + "_res_t" + std::to_string(t + 1);
        tiers[t].idx_name = std::string(tensor_name) + "_res_t" + std::to_string(t + 1) + "_idx";
        tiers[t].mask_name = std::string(tensor_name) + "_res_t" + std::to_string(t + 1) + "_mask";
    }

    f_conv = fopen(fname_conv, "rb");
    for (int t = 0; t < 3; ++t) {
        // Find tier tensors in context
        for (struct ggml_tensor * gt = ggml_get_first_tensor(ctx_conv); gt; gt = ggml_get_next_tensor(ctx_conv, gt)) {
            if (strcmp(gt->name, tiers[t].res_name.c_str()) == 0) tiers[t].res_tensor = gt;
            if (strcmp(gt->name, tiers[t].idx_name.c_str()) == 0) tiers[t].idx_tensor = gt;
            if (strcmp(gt->name, tiers[t].mask_name.c_str()) == 0) tiers[t].mask_tensor = gt;
        }

        if (!tiers[t].res_tensor) {
            fprintf(stderr, "Warning: tier %d residual tensor not found\n", t + 1);
            continue;
        }

        // Load data for each tensor
        auto load_tensor_data = [&](ggml_tensor * gt, const char * name, int & gguf_idx, std::vector<uint8_t> & buf) {
            gguf_idx = gguf_find_tensor(gguf_conv, name);
            if (gguf_idx < 0) return;
            size_t off = gguf_get_data_offset(gguf_conv) + gguf_get_tensor_offset(gguf_conv, gguf_idx);
            size_t nb = ggml_nbytes(gt);
            buf.resize(nb);
            fseek(f_conv, off, SEEK_SET);
            fread(buf.data(), 1, nb, f_conv);
            gt->data = buf.data();
        };

        load_tensor_data(tiers[t].res_tensor, tiers[t].res_name.c_str(), tiers[t].gguf_res_idx, tiers[t].res_data);
        load_tensor_data(tiers[t].idx_tensor, tiers[t].idx_name.c_str(), tiers[t].gguf_idx_idx, tiers[t].idx_data);
        load_tensor_data(tiers[t].mask_tensor, tiers[t].mask_name.c_str(), tiers[t].gguf_mask_idx, tiers[t].mask_data);

        fprintf(stderr, "Tier %d: res type=%s shape={%ld,%ld,%ld}  idx type=%s  mask type=%s\n",
                t + 1,
                ggml_type_name(tiers[t].res_tensor->type),
                (long)tiers[t].res_tensor->ne[0], (long)tiers[t].res_tensor->ne[1], (long)tiers[t].res_tensor->ne[2],
                tiers[t].idx_tensor ? ggml_type_name(tiers[t].idx_tensor->type) : "N/A",
                tiers[t].mask_tensor ? ggml_type_name(tiers[t].mask_tensor->type) : "N/A");
    }
    fclose(f_conv);

    // Read remapping tables
    auto read_idx_table = [&](int t) -> std::vector<int32_t> {
        if (!tiers[t].idx_tensor) return {};
        std::vector<int32_t> result(n_expert, 0);
        memcpy(result.data(), tiers[t].idx_data.data(), n_expert * sizeof(int32_t));
        return result;
    };
    auto read_mask_table = [&](int t) -> std::vector<float> {
        if (!tiers[t].mask_tensor) return {};
        std::vector<float> result(n_expert, 0.0f);
        memcpy(result.data(), tiers[t].mask_data.data(), n_expert * sizeof(float));
        return result;
    };

    fprintf(stderr, "\n--- Remapping tables ---\n");
    std::vector<std::vector<int32_t>> idx_tables(3);
    std::vector<std::vector<float>>   mask_tables(3);
    for (int t = 0; t < 3; ++t) {
        idx_tables[t] = read_idx_table(t);
        mask_tables[t] = read_mask_table(t);
        fprintf(stderr, "Tier %d idx: ", t + 1);
        for (int e = 0; e < (int)n_expert; ++e) fprintf(stderr, "%d ", idx_tables[t][e]);
        fprintf(stderr, "\nTier %d mask: ", t + 1);
        for (int e = 0; e < (int)n_expert; ++e) fprintf(stderr, "%.0f ", mask_tables[t][e]);
        fprintf(stderr, "\n");
    }

    // For each expert, verify: dequant(base[e]) + dequant(residual[tier][local_idx[e]]) ≈ original[e]
    fprintf(stderr, "\n--- Per-expert error analysis ---\n");
    fprintf(stderr, "%-6s  %-8s  %-10s  %-12s  %-12s  %-12s  %-12s\n",
            "Exp", "Tier", "LocalIdx", "BaseMaxErr", "CombMaxErr", "BaseMeanErr", "CombMeanErr");

    double total_base_max = 0, total_comb_max = 0;
    double total_base_mean = 0, total_comb_mean = 0;

    std::vector<float> base_dequant(elem_per_expert);
    std::vector<float> res_dequant(elem_per_expert);
    std::vector<float> combined(elem_per_expert);

    for (int64_t e = 0; e < n_expert; ++e) {
        // Find which tier this expert belongs to
        int tier = -1;
        int local_idx = -1;
        for (int t = 0; t < 3; ++t) {
            if (mask_tables[t][e] > 0.5f) {
                tier = t;
                local_idx = idx_tables[t][e];
                break;
            }
        }

        // Dequantize base for this expert
        const ggml_type base_type = base_tensor->type;
        const size_t base_row_size = ggml_row_size(base_type, ne0);
        const size_t base_off = e * ne1 * base_row_size;
        dequant_buf_to_f32(base_data.data() + base_off, elem_per_expert, base_type, base_dequant.data());

        // Original weights for this expert
        const float * orig_expert = orig_f32.data() + e * elem_per_expert;

        // Compute base-only error
        double base_max_err = 0, base_mean_err = 0;
        for (int64_t j = 0; j < elem_per_expert; ++j) {
            double err = fabs((double)orig_expert[j] - (double)base_dequant[j]);
            if (err > base_max_err) base_max_err = err;
            base_mean_err += err;
        }
        base_mean_err /= elem_per_expert;

        // Compute combined (base + residual) error
        double comb_max_err = 0, comb_mean_err = 0;
        if (tier >= 0 && tiers[tier].res_tensor) {
            // Dequantize residual for this expert
            const ggml_type res_type = tiers[tier].res_tensor->type;
            const size_t res_row_size = ggml_row_size(res_type, ne0);
            const size_t res_off = (size_t)local_idx * ne1 * res_row_size;
            dequant_buf_to_f32(tiers[tier].res_data.data() + res_off, elem_per_expert, res_type, res_dequant.data());

            for (int64_t j = 0; j < elem_per_expert; ++j) {
                combined[j] = base_dequant[j] + res_dequant[j];
                double err = fabs((double)orig_expert[j] - (double)combined[j]);
                if (err > comb_max_err) comb_max_err = err;
                comb_mean_err += err;
            }
            comb_mean_err /= elem_per_expert;
        } else {
            // No tier — base only
            comb_max_err = base_max_err;
            comb_mean_err = base_mean_err;
        }

        fprintf(stderr, "%-6ld  %-8d  %-10d  %-12.6f  %-12.6f  %-12.6f  %-12.6f\n",
                (long)e, tier, local_idx,
                base_max_err, comb_max_err,
                base_mean_err, comb_mean_err);

        total_base_max += base_max_err;
        total_comb_max += comb_max_err;
        total_base_mean += base_mean_err;
        total_comb_mean += comb_mean_err;
    }

    fprintf(stderr, "\n--- Summary ---\n");
    fprintf(stderr, "Avg base-only max error:  %.6f\n", total_base_max / n_expert);
    fprintf(stderr, "Avg combined max error:   %.6f\n", total_comb_max / n_expert);
    fprintf(stderr, "Avg base-only mean error: %.6f\n", total_base_mean / n_expert);
    fprintf(stderr, "Avg combined mean error:  %.6f\n", total_comb_mean / n_expert);
    fprintf(stderr, "\nIf combined error < base error → residuals HELP\n");
    fprintf(stderr, "If combined error > base error → residuals HURT (bug!)\n");

    gguf_free(gguf_orig);
    ggml_free(ctx_orig);
    gguf_free(gguf_conv);
    ggml_free(ctx_conv);

    return 0;
}
