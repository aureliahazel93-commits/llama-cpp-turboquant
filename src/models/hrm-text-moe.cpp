#include "models.h"

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

void llama_model_hrm_text_moe::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_EMBEDDING_SCALE,             hparams.f_embedding_scale);
    ml.get_key(LLM_KV_HRM_LAYERS_PER_STACK,        hparams.n_hrm_layer_per_stack);
    ml.get_key(LLM_KV_HRM_H_CYCLES,                hparams.n_hrm_h_cycles);
    ml.get_key(LLM_KV_HRM_L_CYCLES,                hparams.n_hrm_l_cycles);
    ml.get_key(LLM_KV_HRM_PREFIX_LM,               hparams.hrm_prefix_lm, false);
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp, false);

    switch (hparams.n_embd) {
        case 1536: type = LLM_TYPE_1B; break;
        default:   type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_hrm_text_moe::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_stack = hparams.n_hrm_layer_per_stack;
    GGML_ASSERT(n_stack > 0);
    const int64_t n_cycle_slots = n_stack * (hparams.n_hrm_l_cycles + 1);

    GGML_ASSERT(n_expert_used > 0);
    const int64_t n_ff_exp = hparams.n_ff_exp ? hparams.n_ff_exp : hparams.n_ff() / n_expert_used;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);
    output   = create_tensor(tn(LLM_TENSOR_OUTPUT,     "weight"), {n_embd, n_vocab}, 0);

    hrm_z_l_init = create_tensor(tn(LLM_TENSOR_HRM_Z_L_INIT), {n_embd}, 0);

    hrm_a_bar_l = create_tensor(tn(LLM_TENSOR_HRM_A_BAR_L), {n_embd}, TENSOR_NOT_REQUIRED);
    hrm_b_bar_l = create_tensor(tn(LLM_TENSOR_HRM_B_BAR_L), {n_embd}, TENSOR_NOT_REQUIRED);
    hrm_a_bar_h = create_tensor(tn(LLM_TENSOR_HRM_A_BAR_H), {n_embd}, TENSOR_NOT_REQUIRED);
    hrm_b_bar_h = create_tensor(tn(LLM_TENSOR_HRM_B_BAR_H), {n_embd}, TENSOR_NOT_REQUIRED);

    std::vector<bool> loaded_physical(2 * n_stack, false);

    for (int il = 0; il < n_layer; ++il) {
        auto & layer = layers[il];

        const int64_t layer_in_stack = il % n_stack;
        const int64_t phase = (il % n_cycle_slots) / n_stack;
        const bool is_h_stack = phase == int64_t(hparams.n_hrm_l_cycles);
        const int physical_bid = int((is_h_stack ? n_stack : 0) + layer_in_stack);

        const int flags = loaded_physical[physical_bid] ? TENSOR_DUPLICATED : 0;
        loaded_physical[physical_bid] = true;

        create_tensor_qkv(layer, physical_bid,
                n_embd,
                n_embd_head_k * n_head,
                n_embd_k_gqa,
                n_embd_v_gqa,
                flags);

        layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", physical_bid), {n_embd, n_embd_head_k * n_head}, flags);
        layer.wo        = create_tensor(tn(LLM_TENSOR_ATTN_OUT,  "weight", physical_bid), {n_embd_head_k * n_head, n_embd}, flags);

        layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", physical_bid), {n_embd, n_expert}, flags);

        layer.ffn_gate_up_exps_res_t1 = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "weight_res_t1", physical_bid), {n_embd, n_ff_exp * 2, 8}, TENSOR_NOT_REQUIRED | flags);
        layer.ffn_gate_up_exps_res_t2 = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "weight_res_t2", physical_bid), {n_embd, n_ff_exp * 2, 24}, TENSOR_NOT_REQUIRED | flags);
        layer.ffn_gate_up_exps_res_t3 = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "weight_res_t3", physical_bid), {n_embd, n_ff_exp * 2, 32}, TENSOR_NOT_REQUIRED | flags);
        layer.ffn_down_exps_res_t1 = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight_res_t1", physical_bid), {n_ff_exp, n_embd, 8}, TENSOR_NOT_REQUIRED | flags);
        layer.ffn_down_exps_res_t2 = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight_res_t2", physical_bid), {n_ff_exp, n_embd, 24}, TENSOR_NOT_REQUIRED | flags);
        layer.ffn_down_exps_res_t3 = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight_res_t3", physical_bid), {n_ff_exp, n_embd, 32}, TENSOR_NOT_REQUIRED | flags);

        const bool use_tiered = (layer.ffn_down_exps_res_t1 != nullptr);
        if (!use_tiered) {
            layer.ffn_gate_up_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "weight", physical_bid), {n_embd, n_ff_exp * 2, n_expert}, TENSOR_NOT_REQUIRED | flags);
            if (layer.ffn_gate_up_exps == nullptr) {
                layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", physical_bid), {n_embd, n_ff_exp, n_expert}, flags);
                layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", physical_bid), {n_embd, n_ff_exp, n_expert}, flags);
            }

            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", physical_bid), {n_ff_exp, n_embd, n_expert}, TENSOR_NOT_REQUIRED | flags);
        } else {
            // Tiered down — also load flat gate_up for debug comparison or flat mode
            layer.ffn_gate_up_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "weight", physical_bid), {n_embd, n_ff_exp * 2, n_expert}, TENSOR_NOT_REQUIRED | flags);
        }

        layer.ffn_gate_up_exps_res_t1_idx  = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "weight_res_t1_idx",  physical_bid), {1, n_expert}, TENSOR_NOT_REQUIRED | flags);
        layer.ffn_gate_up_exps_res_t1_mask = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "weight_res_t1_mask", physical_bid), {1, n_expert}, TENSOR_NOT_REQUIRED | flags);
        layer.ffn_gate_up_exps_res_t2_idx  = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "weight_res_t2_idx",  physical_bid), {1, n_expert}, TENSOR_NOT_REQUIRED | flags);
        layer.ffn_gate_up_exps_res_t2_mask = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "weight_res_t2_mask", physical_bid), {1, n_expert}, TENSOR_NOT_REQUIRED | flags);
        layer.ffn_gate_up_exps_res_t3_idx  = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "weight_res_t3_idx",  physical_bid), {1, n_expert}, TENSOR_NOT_REQUIRED | flags);
        layer.ffn_gate_up_exps_res_t3_mask = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "weight_res_t3_mask", physical_bid), {1, n_expert}, TENSOR_NOT_REQUIRED | flags);
        layer.ffn_down_exps_res_t1_idx     = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight_res_t1_idx",  physical_bid), {1, n_expert}, TENSOR_NOT_REQUIRED | flags);
        layer.ffn_down_exps_res_t1_mask    = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight_res_t1_mask", physical_bid), {1, n_expert}, TENSOR_NOT_REQUIRED | flags);
        layer.ffn_down_exps_res_t2_idx     = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight_res_t2_idx",  physical_bid), {1, n_expert}, TENSOR_NOT_REQUIRED | flags);
        layer.ffn_down_exps_res_t2_mask    = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight_res_t2_mask", physical_bid), {1, n_expert}, TENSOR_NOT_REQUIRED | flags);
        layer.ffn_down_exps_res_t3_idx     = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight_res_t3_idx",  physical_bid), {1, n_expert}, TENSOR_NOT_REQUIRED | flags);
        layer.ffn_down_exps_res_t3_mask    = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight_res_t3_mask", physical_bid), {1, n_expert}, TENSOR_NOT_REQUIRED | flags);
    }
}

std::unique_ptr<llm_graph_context> llama_model_hrm_text_moe::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_hrm_text_moe::graph::graph(const llama_model & model_, const llm_graph_params & params) : llm_graph_context(params) {
    const auto & model = static_cast<const llama_model_hrm_text_moe &>(model_);

    GGML_ASSERT(model.tok_embd != nullptr);
    GGML_ASSERT(model.output != nullptr);

    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);
    GGML_ASSERT(n_embd_head > 0);

    const int64_t n_stack = hparams.n_hrm_layer_per_stack;
    const int64_t h_cycles = hparams.n_hrm_h_cycles;
    const int64_t l_cycles = hparams.n_hrm_l_cycles;
    GGML_ASSERT(h_cycles > 0);
    GGML_ASSERT(n_layer == h_cycles * n_stack * (l_cycles + 1));

    ggml_tensor * inp_pos = build_inp_pos();
    auto * inp_attn = build_attn_inp_kv();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    ggml_tensor * hidden_high = build_inp_embd(model.tok_embd);
    ggml_tensor * hidden_low = ggml_repeat(ctx0, model.hrm_z_l_init, hidden_high);
    cb(hidden_low, "hrm_z_l_init", -1);

    const float kq_scale = 1.0f / std::sqrt(float(n_embd_head));

    static float rbias_data[512];
    static bool rbias_loaded = false;
    static struct ggml_context * rbias_ctx = nullptr;
    ggml_tensor * gate_bias = nullptr;
    if (!rbias_loaded) {
        rbias_loaded = true;
        memset(rbias_data, 0, sizeof(rbias_data));
        const char * rbias_file = getenv("HRM_ROUTER_BIAS_FILE");
        if (rbias_file) {
            FILE * f = fopen(rbias_file, "rb");
            if (f) {
                size_t nrd = fread(rbias_data, sizeof(float), n_expert, f);
                fclose(f);
                float bs = 0.0f;
                for (size_t i = 0; i < nrd; i++) bs += fabsf(rbias_data[i]);
                if (bs > 0.0f) {
                    fprintf(stderr, "hrm_text_moe: loaded router bias from %s (%zu experts)\n", rbias_file, nrd);
                }
            }
        }
        size_t sz = ggml_tensor_overhead() + 512*sizeof(float) + 64;
        struct ggml_init_params gip; gip.mem_size = sz; gip.mem_buffer = nullptr; gip.no_alloc = false;
        rbias_ctx = ggml_init(gip);
    }
    float bs = 0.0f;
    for (int i = 0; i < n_expert; i++) bs += fabsf(rbias_data[i]);
    if (bs > 0.0f) {
        gate_bias = ggml_new_tensor_1d(rbias_ctx, GGML_TYPE_F32, n_expert);
        memcpy(gate_bias->data, rbias_data, sizeof(float)*n_expert);
    }

    auto build_stack = [&](ggml_tensor * stack_inp, int slot_offset) -> ggml_tensor * {
        ggml_tensor * stack_cur = stack_inp;

        for (int layer_idx = 0; layer_idx < n_stack; ++layer_idx) {
            const int il = slot_offset + layer_idx;
            const auto & layer = model.layers[il];

            ggml_tensor * inpSA = stack_cur;
            ggml_tensor * cur = build_norm(stack_cur, nullptr, nullptr, LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            {
                ggml_tensor * attn_inp = cur;
                auto [Qcur, Kcur, Vcur] = build_qkv(layer, cur, n_embd_head, n_head, n_head_kv, il);

                ggml_tensor * gate = build_lora_mm(layer.wqkv_gate, attn_inp, layer.wqkv_gate_s);
                cb(gate, "attn_gate_proj", il);

                Qcur = ggml_rope_ext(
                        ctx0, Qcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);
                cb(Qcur, "Qcur_rope", il);

                Kcur = ggml_rope_ext(
                        ctx0, Kcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);
                cb(Kcur, "Kcur_rope", il);

                cur = build_attn(inp_attn,
                        nullptr, nullptr, nullptr,
                        Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
                cb(cur, "attn_out", il);

                gate = ggml_sigmoid(ctx0, gate);
                cb(gate, "attn_gate_sig", il);

                cur = ggml_mul(ctx0, cur, gate);
                cb(cur, "attn_gated", il);

                cur = build_lora_mm(layer.wo, cur, layer.wo_s);
                cb(cur, "attn_o_proj", il);
            }

            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            cur = build_norm(ffn_inp, nullptr, nullptr, LLM_NORM_RMS, il);

            cur = build_moe_ffn(cur,
                    layer.ffn_gate_inp,
                    gate_bias,
                    layer.ffn_up_exps,
                    nullptr,
                    layer.ffn_gate_exps,
                    nullptr,
                    layer.ffn_down_exps,
                    nullptr,
                    nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, true,
                    hparams.expert_weights_scale,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                    il,
                    nullptr,
                    layer.ffn_gate_up_exps,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr);
            cb(cur, "ffn_moe_out", il);

            cur = ggml_add(ctx0, cur, ffn_inp);
            cur = build_cvec(cur, il);
            cb(cur, "hrm_layer_out", il);

            stack_cur = cur;
        }

        stack_cur = build_norm(stack_cur, nullptr, nullptr, LLM_NORM_RMS, slot_offset);
        cb(stack_cur, "stack_final_norm", slot_offset);
        res->t_h_pre_norm = stack_cur;
        return stack_cur;
    };

    for (int h = 0; h < h_cycles; ++h) {
        for (int l = 0; l < l_cycles; ++l) {
            const int slot_offset = int((h * (l_cycles + 1) + l) * n_stack);
            if (model.hrm_a_bar_l) {
                hidden_low = build_stack(
                    ggml_add(ctx0,
                        ggml_mul(ctx0, ggml_repeat(ctx0, model.hrm_a_bar_l, hidden_low), hidden_low),
                        ggml_mul(ctx0, ggml_repeat(ctx0, model.hrm_b_bar_l, hidden_high), hidden_high)),
                    slot_offset);
            } else {
                hidden_low = build_stack(ggml_add(ctx0, hidden_low, hidden_high), slot_offset);
            }
        }

        const int slot_offset = int((h * (l_cycles + 1) + l_cycles) * n_stack);
        if (model.hrm_a_bar_h) {
            hidden_high = build_stack(
                ggml_add(ctx0,
                    ggml_mul(ctx0, ggml_repeat(ctx0, model.hrm_a_bar_h, hidden_high), hidden_high),
                    ggml_mul(ctx0, ggml_repeat(ctx0, model.hrm_b_bar_h, hidden_low), hidden_low)),
                slot_offset);
        } else {
            hidden_high = build_stack(ggml_add(ctx0, hidden_high, hidden_low), slot_offset);
        }
    }

    ggml_tensor * cur = hidden_high;

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);
    cb(cur, "result_output", -1);

    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
}
