#include "llama-impl.h"
#include "llama-model.h"
#include "llama-model-loader.h"
#include "llama-ext.h"

#include "common.h"

#include <algorithm>
#include <cinttypes>
#include <cstring>
#include <fstream>
#include <map>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

static void zeros(std::ofstream & file, size_t n) {
    char zbuf[4096];
    memset(zbuf, 0, sizeof(zbuf));
    while (n > 0) {
        size_t s = std::min(n, (size_t)sizeof(zbuf));
        file.write(zbuf, s);
        n -= s;
    }
}

struct convert_args {
    int ceiling = 8;
    std::string imatrix_file;
    std::string base_type_str;
    std::string fname_inp;
    std::string fname_out;
    bool gate_up_flat = false;
    bool debug_dual_gu = false;
    bool routing_freq = false;
};

static convert_args parse_args(int argc, char ** argv) {
    convert_args args;
    int i = 1;
    while (i < argc && argv[i][0] == '-') {
        if (strcmp(argv[i], "--imatrix-r") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --imatrix-r requires a value\n");
                exit(1);
            }
            args.ceiling = atoi(argv[i + 1]);
            i += 2;
        } else if (strcmp(argv[i], "--imatrix") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --imatrix requires a path\n");
                exit(1);
            }
            args.imatrix_file = argv[i + 1];
            i += 2;
        } else if (strcmp(argv[i], "--base-type") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --base-type requires a type string\n");
                exit(1);
            }
            args.base_type_str = argv[i + 1];
            i += 2;
        } else if (strcmp(argv[i], "--gate-up-flat") == 0) {
            args.gate_up_flat = true;
            i += 1;
        } else if (strcmp(argv[i], "--debug-dual-gu") == 0) {
            args.debug_dual_gu = true;
            i += 1;
        } else if (strcmp(argv[i], "--routing-freq") == 0) {
            args.routing_freq = true;
            i += 1;
        } else {
            fprintf(stderr, "Error: unknown option '%s'\n", argv[i]);
            exit(1);
        }
    }
    static const int valid_ceilings[] = {16, 8, 6, 4, 3, 2};
    bool valid = false;
    for (int v : valid_ceilings) {
        if (args.ceiling == v) { valid = true; break; }
    }
    if (!valid) {
        fprintf(stderr, "Error: --imatrix-r must be one of: 16, 8, 6, 4, 3, 2 (got %d)\n", args.ceiling);
        exit(1);
    }
    if (i + 1 >= argc) {
        fprintf(stderr, "Usage: %s [--imatrix-r N] [--imatrix <file>] <input.gguf> <output.gguf>\n", argv[0]);
        exit(1);
    }
    args.fname_inp = argv[i++];
    args.fname_out = argv[i++];
    return args;
}

static ggml_type ceiling_to_type(int ceiling) {
    switch (ceiling) {
        case 16: return GGML_TYPE_F16;
        case  8: return GGML_TYPE_Q8_0;
        case  6: return GGML_TYPE_Q6_K;
        case  4: return GGML_TYPE_Q4_K;
        case  3: return GGML_TYPE_Q3_K;
        case  2: return GGML_TYPE_Q2_K;
        default: return GGML_TYPE_Q2_K;
    }
}

static ggml_type parse_type_str(const std::string & s) {
    if (s == "q2_k" || s == "Q2_K") return GGML_TYPE_Q2_K;
    if (s == "q3_k" || s == "Q3_K") return GGML_TYPE_Q3_K;
    if (s == "q4_k" || s == "Q4_K") return GGML_TYPE_Q4_K;
    if (s == "q5_k" || s == "Q5_K") return GGML_TYPE_Q5_K;
    if (s == "q6_k" || s == "Q6_K") return GGML_TYPE_Q6_K;
    if (s == "q8_0" || s == "Q8_0") return GGML_TYPE_Q8_0;
    if (s == "f16"  || s == "F16")  return GGML_TYPE_F16;
    fprintf(stderr, "Error: unknown type '%s' (supported: q2_k, q3_k, q4_k, q5_k, q6_k, q8_0, f16)\n", s.c_str());
    exit(1);
}

static ggml_type tier_to_type(int tier, int ceiling) {
    int types[6][3] = {
        {GGML_TYPE_Q6_K, GGML_TYPE_Q6_K, GGML_TYPE_Q5_K},   // ceiling 16 (all K-quants → Q8_K companion)
        {GGML_TYPE_Q6_K, GGML_TYPE_Q5_K, GGML_TYPE_Q4_K},   // ceiling 8  (was Q8_0 tier1 — companion mismatch!)
        {GGML_TYPE_Q6_K, GGML_TYPE_Q4_K, GGML_TYPE_Q3_K},   // ceiling 6
        {GGML_TYPE_Q4_K, GGML_TYPE_Q3_K, GGML_TYPE_Q2_K},   // ceiling 4
        {GGML_TYPE_Q3_K, GGML_TYPE_Q2_K, GGML_TYPE_Q2_K},   // ceiling 3
        {GGML_TYPE_Q2_K, GGML_TYPE_Q2_K, GGML_TYPE_Q2_K},   // ceiling 2
    };
    int ci = -1;
    switch (ceiling) {
        case 16: ci = 0; break;
        case  8: ci = 1; break;
        case  6: ci = 2; break;
        case  4: ci = 3; break;
        case  3: ci = 4; break;
        case  2: ci = 5; break;
    }
    return (ggml_type)types[ci][tier];
}

static int tier_count(int tier) {
    static const int counts[] = {8, 24, 32};
    return counts[tier];
}

static std::unordered_map<std::string, std::vector<float>> load_imatrix(
        const std::string & fname,
        std::unordered_map<std::string, std::vector<float>> * counts_out = nullptr) {
    struct ggml_context * ctx = nullptr;
    struct gguf_init_params p = { false, &ctx };
    struct gguf_context * ctx_gguf = gguf_init_from_file(fname.c_str(), p);
    if (!ctx_gguf) {
        fprintf(stderr, "Error: cannot open imatrix file '%s'\n", fname.c_str());
        exit(1);
    }

    const std::string sums_suffix = ".in_sum2";
    const std::string counts_suffix = ".counts";
    std::map<std::string, std::pair<struct ggml_tensor *, struct ggml_tensor *>> sums_counts;

    for (struct ggml_tensor * cur = ggml_get_first_tensor(ctx); cur; cur = ggml_get_next_tensor(ctx, cur)) {
        std::string name = cur->name;
        if (name.empty()) continue;
        if (string_remove_suffix(name, sums_suffix)) {
            sums_counts[name].first = cur;
        } else if (string_remove_suffix(name, counts_suffix)) {
            sums_counts[name].second = cur;
        }
    }

    std::unordered_map<std::string, std::vector<float>> result;
    for (const auto & sc : sums_counts) {
        const auto & name = sc.first;
        const auto * sums = sc.second.first;
        const auto * counts = sc.second.second;
        if (!sums || !counts) continue;
        const int64_t ne0 = sums->ne[0];
        const int64_t ne1 = sums->ne[1];
        auto & e = result[name];
        e.resize(ne0 * ne1);
        if (counts_out) {
            (*counts_out)[name].resize(ne1);
        }
        for (int64_t j = 0; j < ne1; ++j) {
            float count = ((const float *)counts->data)[counts->ne[0] == 1 ? 0 : j];
            if (counts_out) {
                (*counts_out)[name][j] = count;
            }
            if (count > 0.0f) {
                for (int64_t i = 0; i < ne0; ++i) {
                    e[j * ne0 + i] = ((const float *)sums->data)[j * ne0 + i] / count;
                }
            } else {
                for (int64_t i = 0; i < ne0; ++i) {
                    e[j * ne0 + i] = 1.0f;
                }
            }
        }
    }

    gguf_free(ctx_gguf);
    ggml_free(ctx);
    return result;
}

struct tier_assignment {
    std::vector<int32_t> tier_map;
    std::vector<int32_t> local_idx;
    tier_assignment(int64_t n) : tier_map(n, 0), local_idx(n, 0) {}
};

static tier_assignment compute_tier_assignment(
        const std::string & tensor_name,
        int64_t n_expert,
        const std::unordered_map<std::string, std::vector<float>> & imatrix_data,
        const std::unordered_map<std::string, std::vector<float>> * routing_counts = nullptr) {
    tier_assignment ta(n_expert);

    GGML_ASSERT(tier_count(0) + tier_count(1) + tier_count(2) == n_expert);

    std::vector<float> scores(n_expert, 1.0f);

    if (routing_counts) {
        auto rc_it = routing_counts->find(tensor_name);
        if (rc_it != routing_counts->end()) {
            const auto & rc = rc_it->second;
            for (int64_t e = 0; e < n_expert && e < (int64_t)rc.size(); ++e) {
                scores[e] = rc[e];
            }
            fprintf(stderr, "  routing-freq: using selection counts for %s\n", tensor_name.c_str());
            auto sorted = scores;
            std::sort(sorted.begin(), sorted.end(), std::greater<float>());
            fprintf(stderr, "    top-4: %.0f %.0f %.0f %.0f  bot-4: %.0f %.0f %.0f %.0f\n",
                    sorted[0], sorted[1], sorted[2], sorted[3],
                    sorted[n_expert-4], sorted[n_expert-3], sorted[n_expert-2], sorted[n_expert-1]);
        }
    } else {
        auto it = imatrix_data.find(tensor_name);
        if (it != imatrix_data.end()) {
            const auto & im = it->second;
            int64_t n_per_row = im.size() / n_expert;
            for (int64_t e = 0; e < n_expert; ++e) {
                const float * expert_im = im.data() + e * n_per_row;
                float sum = 0.0f;
                for (int64_t i = 0; i < n_per_row; ++i) {
                    sum += expert_im[i];
                }
                scores[e] = sum / (float)n_per_row;
            }
        }
    }

    std::vector<int> experts(n_expert);
    std::iota(experts.begin(), experts.end(), 0);
    std::sort(experts.begin(), experts.end(), [&](int a, int b) {
        return scores[a] > scores[b];
    });

    int offsets[3] = {0, tier_count(0), tier_count(0) + tier_count(1)};
    int counts[3] = {tier_count(0), tier_count(1), tier_count(2)};

    // Assign tier_map by score order (highest-scoring experts → tier 0)
    for (int t = 0; t < 3; ++t) {
        for (int k = 0; k < counts[t]; ++k) {
            int e = experts[offsets[t] + k];
            ta.tier_map[e] = t;
        }
    }
    // Assign local_idx by global ID order (matches Pass 2 data write order)
    for (int t = 0; t < 3; ++t) {
        int local = 0;
        for (int64_t e = 0; e < n_expert; ++e) {
            if (ta.tier_map[e] == t) {
                ta.local_idx[e] = local++;
            }
        }
    }

    return ta;
}

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

static void dequant_buf_to_f32(const uint8_t * data, size_t n, ggml_type type, float * output) {
    if (type == GGML_TYPE_F16) {
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

static bool is_expert_tensor(const char * name) {
    return strstr(name, "ffn_gate_up_exps") != nullptr ||
           strstr(name, "ffn_down_exps") != nullptr ||
           strstr(name, "ffn_gate_exps") != nullptr ||
           strstr(name, "ffn_up_exps") != nullptr;
}

static bool is_gate_up_tensor(const char * name) {
    return strstr(name, "ffn_gate_up_exps") != nullptr ||
           strstr(name, "ffn_gate_exps") != nullptr ||
           strstr(name, "ffn_up_exps") != nullptr;
}

static ggml_type get_non_moe_type(const char * name, int64_t ne1, int64_t ne2, ggml_type base_type) {
    if (ne1 <= 1 && ne2 <= 1) return GGML_TYPE_F32;
    if (base_type == GGML_TYPE_Q4_K && strstr(name, "attn_qkv.weight") != nullptr) {
        return GGML_TYPE_Q5_K;
    }
    return base_type;
}

struct out_desc {
    std::string in_name;
    std::string out_name;
    ggml_type   out_type;
    int64_t     ne[4];
    int         n_dims;
    int         expert_idx;
    bool        is_expert_tier;
};

int main(int argc, char ** argv) {
    const convert_args args = parse_args(argc, argv);
    const int ceiling = args.ceiling;
    const ggml_type base_type = args.base_type_str.empty()
        ? ceiling_to_type(ceiling)
        : parse_type_str(args.base_type_str);
    const size_t align = GGUF_DEFAULT_ALIGNMENT;

    fprintf(stderr, "imatrix-r-convert: ceiling=%d, base_type=%s\n", ceiling, ggml_type_name(base_type));

    std::unordered_map<std::string, std::vector<float>> imatrix_data;
    std::unordered_map<std::string, std::vector<float>> routing_counts;
    if (!args.imatrix_file.empty()) {
        fprintf(stderr, "imatrix-r-convert: loading imatrix from %s\n", args.imatrix_file.c_str());
        imatrix_data = load_imatrix(args.imatrix_file, args.routing_freq ? &routing_counts : nullptr);
        fprintf(stderr, "imatrix-r-convert: loaded imatrix for %zu tensors\n", imatrix_data.size());
        if (args.routing_freq) {
            fprintf(stderr, "imatrix-r-convert: routing-freq mode: %zu routing count tensors\n", routing_counts.size());
        }
    }

    fprintf(stderr, "imatrix-r-convert: loading %s\n", args.fname_inp.c_str());

    std::vector<std::string> splits;
    llama_model_loader ml(nullptr, nullptr, nullptr, args.fname_inp.c_str(), splits, nullptr,
        true, false, false, true, nullptr, nullptr);
    ml.init_mappings(false);

    gguf_context_ptr ctx_out{gguf_init_empty()};
    gguf_set_kv(ctx_out.get(), ml.metadata);
    gguf_set_val_u32(ctx_out.get(), "general.quantization_version", GGML_QNT_VERSION);
    gguf_set_val_u32(ctx_out.get(), "hrm_text_moe.residual_ceiling", (uint32_t)ceiling);

    std::vector<const llama_model_loader::llama_tensor_weight *> tensors;
    tensors.reserve(ml.weights_map.size());
    for (const auto & it : ml.weights_map) {
        tensors.push_back(&it.second);
    }
    std::sort(tensors.begin(), tensors.end(), [](const auto * a, const auto * b) {
        return strcmp(ggml_get_name(a->tensor), ggml_get_name(b->tensor)) < 0;
    });

    size_t ctx_buf_size = 256uLL * 1024 * 1024;
    std::vector<uint8_t> ctx_buf(ctx_buf_size);
    struct ggml_init_params gparams = { ctx_buf_size, ctx_buf.data(), true };
    struct ggml_context * gctx = ggml_init(gparams);

    ggml_quantize_init(base_type);
    for (int t = 0; t < 3; ++t) {
        ggml_quantize_init(tier_to_type(t, ceiling));
    }
    ggml_quantize_init(GGML_TYPE_Q5_K);

    std::vector<out_desc> out_descs;

    std::vector<std::pair<std::string, tier_assignment>> tier_assignments;

    for (size_t i = 0; i < tensors.size(); ++i) {
        ggml_tensor * tensor = tensors[i]->tensor;
        const char * name = ggml_get_name(tensor);
        const int64_t ne0 = tensor->ne[0];
        const int64_t ne1 = tensor->ne[1];
        const int64_t ne2 = tensor->ne[2];
        const int64_t n_expert = (ne2 > 1) ? ne2 : 1;

        if (is_expert_tensor(name) && ne2 > 1) {
            tier_assignment ta = compute_tier_assignment(name, n_expert, imatrix_data, args.routing_freq ? &routing_counts : nullptr);
            std::string key_prefix;
            if (strstr(name, "ffn_gate_up_exps")) key_prefix = "gate_up";
            else if (strstr(name, "ffn_down_exps")) key_prefix = "down";
            else if (strstr(name, "ffn_gate_exps")) key_prefix = "gate";
            else if (strstr(name, "ffn_up_exps")) key_prefix = "up";

            std::string blk_key;
            const char * blk_start = strstr(name, "blk.");
            if (blk_start) {
                const char * end = strstr(blk_start + 4, ".");
                if (end) {
                    blk_key.assign(blk_start, end - blk_start);
                }
            }

            if (!key_prefix.empty() && !blk_key.empty()) {
                tier_assignments.push_back({key_prefix + "." + blk_key, ta});
            }

            const bool is_gu = is_gate_up_tensor(name);
            const bool write_flat_gu = (args.gate_up_flat || args.debug_dual_gu) && is_gu;
            const bool skip_tier_gu = args.gate_up_flat && !args.debug_dual_gu && is_gu;

            if (write_flat_gu) {
                // Write gate_up as single flat Q4_K tensor
                out_desc d;
                d.in_name = name;
                d.out_name = name;
                d.out_type = base_type;
                d.ne[0] = ne0; d.ne[1] = ne1; d.ne[2] = ne2; d.ne[3] = 1;
                d.n_dims = 3;
                d.expert_idx = -1;
                d.is_expert_tier = false;
                out_descs.push_back(d);

                ggml_tensor * meta = ggml_new_tensor(gctx, base_type, 3, d.ne);
                ggml_set_name(meta, d.out_name.c_str());
                gguf_add_tensor(ctx_out.get(), meta);
            }
            if (!skip_tier_gu) {
                // Register tier descriptors (full weights per tier, no base)
                for (int t = 0; t < 3; ++t) {
                    ggml_type tt = is_gate_up_tensor(name) ? GGML_TYPE_Q4_K : tier_to_type(t, ceiling);
                    out_desc d;
                    d.in_name = name;
                    d.out_name = std::string(name) + "_res_t" + std::to_string(t + 1);
                    d.out_type = tt;
                    d.ne[0] = ne0; d.ne[1] = ne1; d.ne[2] = tier_count(t); d.ne[3] = 1;
                    d.n_dims = 3;
                    d.expert_idx = -1;
                    d.is_expert_tier = true;
                    out_descs.push_back(d);

                    ggml_tensor * meta = ggml_new_tensor(gctx, tt, 3, d.ne);
                    ggml_set_name(meta, d.out_name.c_str());
                    gguf_add_tensor(ctx_out.get(), meta);
                }
                // Register per-tier remapping tensors (idx + mask) as GGUF tensors
                for (int rt = 0; rt < 3; ++rt) {
                    std::string idx_name = std::string(name) + "_res_t" + std::to_string(rt + 1) + "_idx";
                    ggml_tensor * idx_meta = ggml_new_tensor_2d(gctx, GGML_TYPE_I32, 1, n_expert);
                    ggml_set_name(idx_meta, idx_name.c_str());
                    gguf_add_tensor(ctx_out.get(), idx_meta);

                    std::string mask_name = std::string(name) + "_res_t" + std::to_string(rt + 1) + "_mask";
                    ggml_tensor * mask_meta = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, 1, n_expert);
                    ggml_set_name(mask_meta, mask_name.c_str());
                    gguf_add_tensor(ctx_out.get(), mask_meta);
                }
            }
        } else if (tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_F16 || tensor->type == GGML_TYPE_BF16) {
            out_desc d;
            d.in_name = name;
            d.out_name = name;
            d.out_type = get_non_moe_type(name, ne1, ne2, base_type);
            int64_t total_rows = ne1 * std::max(ne2, (int64_t)1);
            d.ne[0] = ne0; d.ne[1] = total_rows; d.ne[2] = 1; d.ne[3] = 1;
            d.n_dims = (ne2 > 1) ? 3 : 2;
            d.expert_idx = -1;
            d.is_expert_tier = false;
            out_descs.push_back(d);

            ggml_tensor * meta = ggml_new_tensor(gctx, d.out_type, d.n_dims, d.ne);
            ggml_set_name(meta, d.out_name.c_str());
            gguf_add_tensor(ctx_out.get(), meta);

        } else {
            out_desc d;
            d.in_name = name;
            d.out_name = name;
            d.out_type = tensor->type;
            memcpy(d.ne, tensor->ne, sizeof(d.ne));
            d.n_dims = (tensor->ne[3] > 1) ? 4 : (tensor->ne[2] > 1) ? 3 : 2;
            d.expert_idx = -1;
            d.is_expert_tier = false;
            out_descs.push_back(d);

            ggml_tensor * meta = ggml_new_tensor(gctx, d.out_type, d.n_dims, d.ne);
            ggml_set_name(meta, d.out_name.c_str());
            gguf_add_tensor(ctx_out.get(), meta);
        }
    }


    fprintf(stderr, "imatrix-r-convert: %zu input tensors -> %zu output tensors\n",
            tensors.size(), out_descs.size());

    std::ofstream fout(args.fname_out, std::ios::binary);
    fout.exceptions(std::ofstream::failbit);

    const size_t meta_size = gguf_get_meta_size(ctx_out.get());
    zeros(fout, meta_size);

    size_t total_size_new = 0;
    size_t total_size_org = 0;
    int idx = 0;

    for (size_t i = 0; i < tensors.size(); ++i) {
        ggml_tensor * tensor = tensors[i]->tensor;
        const char * name = ggml_get_name(tensor);
        ml.load_data_for(tensor);

        const size_t tensor_size = ggml_nbytes(tensor);
        total_size_org += tensor_size;

        const int64_t ne0 = tensor->ne[0];
        const int64_t ne1 = tensor->ne[1];
        const int64_t ne2 = tensor->ne[2];
        int64_t n_expert = (ne2 > 1) ? ne2 : 1;
        int64_t elem_per_expert = ne0 * ne1;

        idx++;
        fprintf(stderr, "[%4d/%4zu] %-50s %s -> ", idx, tensors.size(), name,
                ggml_type_name(tensor->type));

        if (is_expert_tensor(name) && ne2 > 1) {
            size_t nelements = ggml_nelements(tensor);
            std::vector<float> f32_buf(nelements);
            dequant_to_f32(tensor, f32_buf.data());

            tier_assignment ta(n_expert);
            {
                std::string key_prefix;
                if (strstr(name, "ffn_gate_up_exps")) key_prefix = "gate_up";
                else if (strstr(name, "ffn_down_exps")) key_prefix = "down";
                else if (strstr(name, "ffn_gate_exps")) key_prefix = "gate";
                else if (strstr(name, "ffn_up_exps")) key_prefix = "up";

                std::string blk_key;
                const char * blk_start = strstr(name, "blk.");
                if (blk_start) {
                    const char * end = strstr(blk_start + 4, ".");
                    if (end) {
                        blk_key.assign(blk_start, end - blk_start);
                    }
                }

                if (!key_prefix.empty() && !blk_key.empty()) {
                    std::string search_key = key_prefix + "." + blk_key;
                    for (const auto & kv : tier_assignments) {
                        if (kv.first == search_key) {
                            ta = kv.second;
                            break;
                        }
                    }
                }
            }

            const float * expert_imatrix = nullptr;
            auto im_it = imatrix_data.find(name);
            if (im_it != imatrix_data.end()) {
                expert_imatrix = im_it->second.data();
            }
            if (is_expert_tensor(name)) {
                fprintf(stderr, "  imatrix: %s — %s (%zu values)\n",
                    name, expert_imatrix ? "FOUND" : "MISSING",
                    expert_imatrix ? im_it->second.size() : 0);
                if (expert_imatrix && ne2 > 1) {
                    int64_t n_per_expert = im_it->second.size() / n_expert;
                    for (int e = 0; e < std::min((int64_t)3, n_expert); ++e) {
                        const float * im_e = expert_imatrix + e * n_per_expert;
                        float mn = im_e[0], mx = im_e[0], sm = 0;
                        for (int64_t i = 0; i < n_per_expert; ++i) {
                            mn = std::min(mn, im_e[i]);
                            mx = std::max(mx, im_e[i]);
                            sm += im_e[i];
                        }
                        fprintf(stderr, "    expert %d: mean=%.4f min=%.4f max=%.4f\n", e, sm/n_per_expert, mn, mx);
                    }
                }
            }

            {
                const bool is_gu = is_gate_up_tensor(name);
                const bool write_flat_gu = (args.gate_up_flat || args.debug_dual_gu) && is_gu;
                const bool skip_tier_gu = args.gate_up_flat && !args.debug_dual_gu && is_gu;
            if (write_flat_gu) {
                size_t total_rows = ne1 * ne2;
                size_t row_size = ggml_row_size(base_type, ne0);
                size_t qsize = row_size * total_rows;
                std::vector<uint8_t> qbuf(qsize);
                ggml_quantize_chunk(base_type, f32_buf.data(), qbuf.data(), 0, total_rows, ne0, expert_imatrix);
                fout.write((const char *)qbuf.data(), qbuf.size());
                zeros(fout, GGML_PAD(qbuf.size(), align) - qbuf.size());
                total_size_new += qbuf.size();
                fprintf(stderr, "%s (%.1f MiB, flat)\n", ggml_type_name(base_type), qbuf.size() / 1024.0 / 1024.0);
            }
            if (!skip_tier_gu) {
            for (int t = 0; t < 3; ++t) {
                ggml_type tt = is_gate_up_tensor(name) ? GGML_TYPE_Q4_K : tier_to_type(t, ceiling);
                size_t tc = tier_count(t);
                size_t row_size = ggml_row_size(tt, ne0);
                size_t tier_total = row_size * ne1 * tc;
                std::vector<uint8_t> tier_data(tier_total, 0);

                int local = 0;
                for (int64_t e = 0; e < n_expert; ++e) {
                    if (ta.tier_map[e] != t) continue;

                    const float * w_expert = f32_buf.data() + e * elem_per_expert;

                    const float * e_imatrix = nullptr;
                    if (expert_imatrix) {
                        if (tt == GGML_TYPE_Q4_K) {
                            e_imatrix = expert_imatrix;
                        } else {
                            e_imatrix = expert_imatrix + e * ne0;
                        }
                    }
                    size_t tier_off = (size_t)local * ne1 * row_size;
                    ggml_quantize_chunk(tt, w_expert, tier_data.data() + tier_off,
                                         0, ne1, ne0, e_imatrix);
                    local++;
                }

                fout.write((const char *)tier_data.data(), tier_total);
                zeros(fout, GGML_PAD(tier_total, align) - tier_total);
                total_size_new += tier_total;
            }
            // Write per-tier remapping tensor data (idx + mask)
            for (int rt = 0; rt < 3; ++rt) {
                // idx: I32 [n_expert]
                std::vector<int32_t> idx_buf(n_expert, 0);
                for (int e = 0; e < n_expert; ++e) {
                    if (ta.tier_map[e] == rt) idx_buf[e] = ta.local_idx[e];
                }
                size_t idx_sz = n_expert * sizeof(int32_t);
                fout.write((const char *)idx_buf.data(), idx_sz);
                zeros(fout, GGML_PAD(idx_sz, align) - idx_sz);
                total_size_new += idx_sz;

                // mask: F32 [n_expert]
                std::vector<float> mask_buf(n_expert, 0.0f);
                for (int e = 0; e < n_expert; ++e) {
                    mask_buf[e] = (ta.tier_map[e] == rt) ? 1.0f : 0.0f;
                }
                size_t mask_sz = n_expert * sizeof(float);
                fout.write((const char *)mask_buf.data(), mask_sz);
                zeros(fout, GGML_PAD(mask_sz, align) - mask_sz);
                total_size_new += mask_sz;
            }
            fprintf(stderr, "t1(%s) + t2(%s) + t3(%s)\n",
                    ggml_type_name(tier_to_type(0, ceiling)),
                    ggml_type_name(tier_to_type(1, ceiling)),
                    ggml_type_name(tier_to_type(2, ceiling)));
            }
            }

        } else if (tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_F16 || tensor->type == GGML_TYPE_BF16) {
            ggml_type ct = get_non_moe_type(name, ne1, ne2, base_type);
            size_t nelements = ggml_nelements(tensor);
            std::vector<float> f32_buf(nelements);
            dequant_to_f32(tensor, f32_buf.data());

            int64_t total_rows = ne1 * std::max(ne2, (int64_t)1);
            size_t qsize = ggml_row_size(ct, ne0) * total_rows;
            std::vector<uint8_t> qbuf(qsize);

            const float * im = nullptr;
            auto im_it = imatrix_data.find(name);
            if (im_it != imatrix_data.end()) {
                im = im_it->second.data();
            }
            ggml_quantize_chunk(ct, f32_buf.data(), qbuf.data(), 0, total_rows, ne0, im);

            fout.write((const char *)qbuf.data(), qbuf.size());
            zeros(fout, GGML_PAD(qbuf.size(), align) - qbuf.size());
            total_size_new += qbuf.size();

            fprintf(stderr, "%s (%.1f MiB)\n", ggml_type_name(ct), qbuf.size() / 1024.0 / 1024.0);

        } else {
            fout.write((const char *)tensor->data, tensor_size);
            zeros(fout, GGML_PAD(tensor_size, align) - tensor_size);
            total_size_new += tensor_size;

            fprintf(stderr, "copy (%s, %.1f MiB)\n", ggml_type_name(tensor->type),
                    tensor_size / 1024.0 / 1024.0);
        }
    }

    fout.seekp(0);
    {
        std::vector<uint8_t> meta_data(meta_size);
        gguf_get_meta_data(ctx_out.get(), meta_data.data());
        fout.write((const char *)meta_data.data(), meta_data.size());
    }
    fout.close();

    ggml_quantize_free();
    ggml_free(gctx);

    fprintf(stderr, "\nimatrix-r-convert: model size = %.2f MiB\n", total_size_org / 1024.0 / 1024.0);
    fprintf(stderr, "imatrix-r-convert: quant size = %.2f MiB (%.2f BPW)\n",
            total_size_new / 1024.0 / 1024.0, total_size_new * 8.0 / ml.n_elements);
    fprintf(stderr, "imatrix-r-convert: done\n");
    return 0;
}
