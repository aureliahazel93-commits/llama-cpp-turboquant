// Usage: stability-runner --ref model-f16.gguf --quant model-q4.gguf --prompt-file test.txt
//
// Output:
//   Reference PPL:  8.234
//   Quantized PPL:  8.291 (+0.69%)
//   Verdict: PASS (PPL delta < 5%)

#include "arg.h"
#include "common.h"
#include "log.h"
#include "kl_divergence.h"
#include "round_trip.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>

static double compute_ppl(llama_context * ctx, const common_params & params) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::string prompt;
    if (!params.prompt_file.empty()) {
        std::ifstream f(params.prompt_file);
        if (f) {
            std::ostringstream ss;
            ss << f.rdbuf();
            prompt = ss.str();
        }
    }
    if (prompt.empty()) {
        LOG_ERR("stability-runner: provide --prompt-file with evaluation text\n");
        return -1.0;
    }

    std::vector<llama_token> tokens(prompt.size() + 1);
    int n_tokens = llama_tokenize(vocab, prompt.c_str(), (int)prompt.size(), tokens.data(), tokens.size(), true, true);
    tokens.resize(n_tokens);
    if (tokens.empty()) {
        LOG_ERR("stability-runner: failed to tokenize prompt\n");
        return -1.0;
    }

    const int n_ctx = llama_n_ctx(ctx);
    const int n_batch = std::min(params.n_batch, n_ctx);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    double total_nll = 0.0;
    int    n_eval    = 0;

    llama_batch batch = llama_batch_init(n_batch, 0, 1);

    for (int i = 0; i < n_tokens; i++) {
        llama_batch_free(batch);
        batch = llama_batch_init(n_batch, 0, 1);
        batch.token[0] = tokens[i];
        batch.pos[0]   = i;
        batch.n_tokens = 1;

        if (llama_decode(ctx, batch)) {
            LOG_ERR("stability-runner: decode failed at token %d\n", i);
            llama_batch_free(batch);
            return -1.0;
        }

        if (i + 1 < n_tokens) {
            const float * logits = llama_get_logits(ctx);
            float max_logit = *std::max_element(logits, logits + n_vocab);
            double logsumexp = 0.0;
            for (int j = 0; j < n_vocab; j++) logsumexp += expf(logits[j] - max_logit);
            logsumexp = max_logit + logf(logsumexp);

            int next = tokens[i + 1];
            if (next >= 0 && next < n_vocab) {
                total_nll += -(logits[next] - logsumexp);
                n_eval++;
            }
        }
    }

    llama_batch_free(batch);
    return (n_eval > 0) ? exp(total_nll / n_eval) : -1.0;
}

int main(int argc, char ** argv) {
    common_params params;
    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_STABILITY)) {
        return 1;
    }

    if (params.kl_reference.empty()) {
        LOG_ERR("stability-runner: --kl-reference <model> is required\n");
        return 1;
    }

    LOG_INF("Stability runner: ref=%s, quant=%s\n", params.kl_reference.c_str(), params.model.path.c_str());

    // Load reference model
    common_params ref_params = params;
    ref_params.model.path = params.kl_reference;
    auto ref_init = common_init_from_params(ref_params);
    if (!ref_init) {
        LOG_ERR("stability-runner: failed to load reference model %s\n", params.kl_reference.c_str());
        return 1;
    }
    auto * ref_ctx = ref_init->context();
    if (!ref_ctx) {
        LOG_ERR("stability-runner: failed to create context for reference model\n");
        return 1;
    }

    // Load quantized model
    auto quant_init = common_init_from_params(params);
    if (!quant_init) {
        LOG_ERR("stability-runner: failed to load quantized model %s\n", params.model.path.c_str());
        return 1;
    }
    auto * quant_ctx = quant_init->context();
    if (!quant_ctx) {
        LOG_ERR("stability-runner: failed to create context for quantized model\n");
        return 1;
    }

    // Compute PPL for both
    double ref_ppl   = compute_ppl(ref_ctx, params);
    double quant_ppl = compute_ppl(quant_ctx, params);

    if (ref_ppl > 0 && quant_ppl > 0) {
        double ppl_delta = (quant_ppl - ref_ppl) / ref_ppl * 100.0;
        LOG_INF("Reference PPL:  %.4f\n", ref_ppl);
        LOG_INF("Quantized PPL:  %.4f (%+.2f%%)\n", quant_ppl, ppl_delta);
        LOG_INF("Verdict: %s (PPL delta %.2f%% < 5%%)\n",
                (fabs(ppl_delta) < 5.0) ? "PASS" : "FAIL", ppl_delta);
    } else {
        LOG_ERR("stability-runner: failed to compute PPL\n");
    }

    llama_backend_free();
    return 0;
}
