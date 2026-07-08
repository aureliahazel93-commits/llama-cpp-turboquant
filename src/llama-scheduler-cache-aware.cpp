#include "llama-prefix-cache.h"

#include <algorithm>
#include <numeric>
#include <vector>

std::vector<int> cache_aware_order(
        const llama_prefix_cache & cache,
        const std::vector<std::vector<llama_token>> & requests) {
    const int n = requests.size();
    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    std::vector<int> scores(n);
    for (int i = 0; i < n; i++) {
        const auto & req = requests[i];
        prefix_match m = cache.lookup(req.data(), (int) req.size());
        scores[i] = m.match_len;
    }

    std::stable_sort(indices.begin(), indices.end(),
            [&scores](int a, int b) { return scores[a] > scores[b]; });

    return indices;
}

double cache_hit_rate(
        const llama_prefix_cache & cache,
        const std::vector<std::vector<llama_token>> & requests) {
    if (requests.empty()) return 0.0;

    int total_tokens = 0;
    int cached_tokens = 0;

    for (const auto & req : requests) {
        int n_tokens = (int) req.size();
        total_tokens += n_tokens;
        if (n_tokens > 0) {
            prefix_match m = cache.lookup(req.data(), n_tokens);
            cached_tokens += m.match_len;
        }
    }

    return total_tokens > 0 ? (double) cached_tokens / total_tokens : 0.0;
}

int estimate_cache_savings(
        const llama_prefix_cache & cache,
        const std::vector<std::vector<llama_token>> & requests) {
    int blocks_saved = 0;

    for (const auto & req : requests) {
        int n_tokens = (int) req.size();
        if (n_tokens > 0) {
            prefix_match m = cache.lookup(req.data(), n_tokens);
            blocks_saved += m.n_blocks;
        }
    }

    return blocks_saved;
}
