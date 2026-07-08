#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init_tgt = common_init_from_params(params);

    llama_model * model_tgt = llama_init_tgt->model();
    llama_context * ctx_tgt = llama_init_tgt->context();

    if (model_tgt == nullptr || ctx_tgt == nullptr) {
        LOG_ERR("%s: failed to load target model\n", __func__);
        return 1;
    }

    llama_model * model_dft = nullptr;
    llama_context * ctx_dft = nullptr;

    {
        const auto & params_spec = params.speculative.draft;

        auto params_dft = params;
        params_dft.devices      = params_spec.devices;
        params_dft.model        = params_spec.mparams;
        params_dft.n_gpu_layers = params_spec.n_gpu_layers;

        if (params_spec.cpuparams.n_threads > 0) {
            params_dft.cpuparams.n_threads       = params_spec.cpuparams.n_threads;
            params_dft.cpuparams_batch.n_threads = params_spec.cpuparams_batch.n_threads;
        }

        auto mparams = common_model_params_to_llama(params_dft);
        model_dft = llama_model_load_from_file(params_dft.model.path.c_str(), mparams);
        if (model_dft == nullptr) {
            LOG_ERR("%s: failed to load drafter model\n", __func__);
            return 1;
        }

        auto cparams = common_context_params_to_llama(params_dft);
        ctx_dft = llama_init_from_model(model_dft, cparams);
        if (ctx_dft == nullptr) {
            LOG_ERR("%s: failed to create drafter context\n", __func__);
            return 1;
        }
    }

    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);

    std::vector<llama_token> inp = common_tokenize(ctx_tgt, params.prompt, true, true);

    if ((int32_t)inp.size() > llama_n_ctx(ctx_tgt)) {
        LOG_ERR("%s: prompt exceeds target context size\n", __func__);
        return 1;
    }

    llama_set_embeddings(ctx_tgt, true);

    llama_batch batch = llama_batch_get_one(inp.data(), (int32_t)inp.size());
    if (llama_decode(ctx_tgt, batch) != 0) {
        LOG_ERR("%s: failed to decode prompt on target\n", __func__);
        return 1;
    }

    struct llama_hidden_state * state = llama_get_hidden_state(ctx_tgt);
    if (state == nullptr) {
        LOG_ERR("%s: failed to get hidden state from target\n", __func__);
        return 1;
    }

    int32_t n_feed = llama_feed_hidden_state(ctx_dft, state, state->n_tokens);
    if (n_feed < 0) {
        LOG_ERR("%s: failed to feed hidden state to drafter (error %d)\n", __func__, n_feed);
        llama_hidden_state_free(state);
        return 1;
    }

    llama_hidden_state_free(state);

    const float * logits_dft = llama_get_logits(ctx_dft);
    if (logits_dft == nullptr) {
        LOG_ERR("%s: drafter produced no logits\n", __func__);
        return 1;
    }

    llama_token id = llama_sampler_sample(nullptr, ctx_dft, -1);

    LOG("\nDraft token: %s\n", common_token_to_piece(ctx_tgt, id).c_str());

    llama_batch batch_verify = llama_batch_get_one(&id, 1);
    if (llama_decode(ctx_tgt, batch_verify) != 0) {
        LOG_ERR("%s: failed to verify draft token on target\n", __func__);
        return 1;
    }

    llama_token id_verify = llama_sampler_sample(nullptr, ctx_tgt, -1);
    bool accepted = (id_verify == id);

    LOG("Verification: %s (token %s)\n",
            accepted ? "accepted" : "rejected",
            common_token_to_piece(ctx_tgt, id_verify).c_str());
    llama_free(ctx_dft);
    llama_model_free(model_dft);
    llama_free(ctx_tgt);
    llama_backend_free();

    return 0;
}
