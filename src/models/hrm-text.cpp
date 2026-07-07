#include "models.h"

#include <cmath>
#include <vector>

void llama_model_hrm_text::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_EMBEDDING_SCALE,             hparams.f_embedding_scale);
    ml.get_key(LLM_KV_HRM_LAYERS_PER_STACK,        hparams.n_hrm_layer_per_stack);
    ml.get_key(LLM_KV_HRM_H_CYCLES,                hparams.n_hrm_h_cycles);
    ml.get_key(LLM_KV_HRM_L_CYCLES,                hparams.n_hrm_l_cycles);
    ml.get_key(LLM_KV_HRM_PREFIX_LM,               hparams.hrm_prefix_lm, false);

    switch (hparams.n_embd) {
        case 1536: type = LLM_TYPE_1B; break;
        default:   type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_hrm_text::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_stack = hparams.n_hrm_layer_per_stack;
    const int64_t n_cycle_slots = n_stack * (hparams.n_hrm_l_cycles + 1);

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

        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", physical_bid), {n_embd, n_ff}, flags);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", physical_bid), {n_ff, n_embd}, flags);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", physical_bid), {n_embd, n_ff}, flags);
    }
}

std::unique_ptr<llm_graph_context> llama_model_hrm_text::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_hrm_text::graph::graph(const llama_model & model_, const llm_graph_params & params) : llm_graph_context(params) {
    const auto & model = static_cast<const llama_model_hrm_text &>(model_);

    GGML_ASSERT(model.tok_embd != nullptr);
    GGML_ASSERT(model.output != nullptr);
    GGML_ASSERT(model.hrm_z_l_init != nullptr);

    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    const int64_t n_stack = hparams.n_hrm_layer_per_stack;
    const int64_t h_cycles = hparams.n_hrm_h_cycles;
    const int64_t l_cycles = hparams.n_hrm_l_cycles;

    ggml_tensor * inp_pos = build_inp_pos();
    auto * inp_attn = build_attn_inp_kv();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    ggml_tensor * hidden_high = build_inp_embd(model.tok_embd);
    ggml_tensor * hidden_low = ggml_repeat(ctx0, model.hrm_z_l_init, hidden_high);
    cb(hidden_low, "hrm_z_l_init", -1);

    const float kq_scale = 1.0f / std::sqrt(float(n_embd_head));

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
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                    layer.ffn_up,   nullptr, layer.ffn_up_s,
                    layer.ffn_gate, nullptr, layer.ffn_gate_s,
                    layer.ffn_down, nullptr, layer.ffn_down_s,
                    nullptr,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);

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
