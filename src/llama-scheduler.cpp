#include "llama-scheduler.h"
#include "llama-kv-cache-paged.h"
#include "llama-prefix-cache.h"

#include <chrono>
#include <algorithm>
#include <cassert>

static int32_t blocks_for_tokens(int n_tokens, int block_size) {
    return (n_tokens + block_size - 1) / block_size;
}

static int64_t now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct llama_scheduler::impl {
    llama_scheduler_params    params;
    llama_kv_cache_paged   * kv_cache     = nullptr;
    llama_prefix_cache     * prefix_cache = nullptr;
    std::deque<sched_request> all_requests;
    uint64_t                 next_id      = 1;
    int32_t                  next_seq_id  = 255;
    std::vector<int32_t>     free_seq_ids;
    int32_t                  n_blocks_used = 0;

    int32_t allocate_seq_id() {
        if (!free_seq_ids.empty()) {
            int32_t id = free_seq_ids.back();
            free_seq_ids.pop_back();
            return id;
        }
        int32_t id = next_seq_id--;
        assert(id >= 128);
        return id;
    }

    void free_seq_id(int32_t id) {
        free_seq_ids.push_back(id);
    }

    static int64_t effective_priority(const sched_request & req) {
        return (int64_t)req.priority - (int64_t)req.preemption_count * 1000;
    }

    int32_t count_free_blocks() const {
        if (kv_cache) {
            return kv_cache->n_blocks_free();
        }
        return params.n_blocks_total - n_blocks_used;
    }

    void do_alloc_blocks(int n) {
        if (kv_cache) {
            for (int i = 0; i < n; i++) {
                kv_cache->alloc_block();
            }
        }
        n_blocks_used += n;
    }

    void do_free_block_ids(const std::vector<int> & ids) {
        if (kv_cache) {
            for (int id : ids) {
                kv_cache->free_block(id);
            }
        }
        n_blocks_used -= (int32_t)ids.size();
    }
};

llama_scheduler::llama_scheduler(const llama_scheduler_params & params)
    : pimpl(new impl()) {
    pimpl->params = params;
    pimpl->params.max_num_batched_tokens = params.max_num_batched_tokens > 0
        ? params.max_num_batched_tokens
        : params.n_max_batch;
}

llama_scheduler::~llama_scheduler() {
    delete pimpl;
}

void llama_scheduler::set_kv_cache(llama_kv_cache_paged * kv_cache) {
    pimpl->kv_cache = kv_cache;
}

void llama_scheduler::set_prefix_cache(llama_prefix_cache * prefix_cache) {
    pimpl->prefix_cache = prefix_cache;
}

uint64_t llama_scheduler::add_request(const sched_request_params & params) {
    sched_request req;
    req.id                  = pimpl->next_id++;
    req.state               = WAITING;
    req.priority            = params.priority;
    req.preemption_count    = 0;
    req.n_prompt            = params.n_prompt;
    req.n_prompt_processed  = 0;
    req.seq_id              = pimpl->allocate_seq_id();
    req.n_predict           = params.n_predict;
    req.n_predict_remaining = params.n_predict;
    req.stop_reason         = STOP_NONE;
    req.is_prefill          = true;
    req.arrival_time_us     = now_us();

    if (params.prompt_tokens && params.n_prompt > 0) {
        req.prompt_tokens.assign(params.prompt_tokens, params.prompt_tokens + params.n_prompt);
    }

    pimpl->all_requests.push_back(std::move(req));
    return req.id;
}

bool llama_scheduler::abort_request(uint64_t id) {
    for (auto & req : pimpl->all_requests) {
        if (req.id == id && req.state != FINISHED) {
            req.stop_reason = STOP_ABORTED;
            return true;
        }
    }
    return false;
}

int llama_scheduler::notify_token(uint64_t id, llama_token token, sched_stop_reason stop) {
    for (auto & req : pimpl->all_requests) {
        if (req.id == id && req.state == RUNNING) {
            if (stop != STOP_NONE) {
                req.stop_reason = stop;
            } else {
                req.generated_tokens.push_back(token);
                if (req.n_predict_remaining > 0) {
                    req.n_predict_remaining--;
                    if (req.n_predict_remaining <= 0) {
                        req.stop_reason = STOP_LIMIT;
                    }
                }
            }
            return 0;
        }
    }
    return -1;
}

int llama_scheduler::free_blocks(uint64_t id) {
    for (auto & req : pimpl->all_requests) {
        if (req.id == id && req.state != FINISHED) {
            pimpl->do_free_block_ids(req.block_ids);
            req.block_ids.clear();
            req.n_prompt_processed = 0;
            return 0;
        }
    }
    return -1;
}

struct sched_result llama_scheduler::step() {
    sched_result result = {};
    result.action = SCHED_NOTHING;

    auto & all = pimpl->all_requests;

    // 1. process_completions: finalize RUNNING requests with stop_reason set
    {
        auto it = all.begin();
        while (it != all.end()) {
            if (it->state == RUNNING && it->stop_reason != STOP_NONE) {
                sched_completion comp;
                comp.id          = it->id;
                comp.tokens      = std::move(it->generated_tokens);
                comp.stop_reason = it->stop_reason;
                comp.n_prompt    = it->n_prompt;
                comp.n_generated = (int32_t)comp.tokens.size();
                result.completions.push_back(std::move(comp));

                pimpl->do_free_block_ids(it->block_ids);
                pimpl->free_seq_id(it->seq_id);
                it = all.erase(it);
            } else {
                ++it;
            }
        }
    }

    // 2. preempt_if_needed: evict lowest effective_priority RUNNING if no free blocks
    while (pimpl->count_free_blocks() < 1) {
        auto worst = all.end();
        int64_t worst_pri = INT64_MAX;
        int64_t worst_time = INT64_MAX;

        for (auto it = all.begin(); it != all.end(); ++it) {
            if (it->state == RUNNING) {
                int64_t ep = impl::effective_priority(*it);
                if (ep < worst_pri || (ep == worst_pri && it->arrival_time_us < worst_time)) {
                    worst_pri = ep;
                    worst_time = it->arrival_time_us;
                    worst = it;
                }
            }
        }

        if (worst == all.end()) break;

        worst->state = PREEMPTED;
        worst->preemption_count++;
        worst->n_prompt_processed = 0;
        pimpl->do_free_block_ids(worst->block_ids);
        worst->block_ids.clear();
    }

    // 3. admit_waiting: promote WAITING/PREEMPTED to RUNNING
    int n_running = 0;
    for (auto & r : all) {
        if (r.state == RUNNING) n_running++;
    }

    for (auto & req : all) {
        if (n_running >= pimpl->params.n_max_seq) break;
        if (req.state != WAITING && req.state != PREEMPTED) continue;
        if (pimpl->count_free_blocks() < 1) break;

        int tokens_total = req.n_prompt + (int)req.generated_tokens.size();
        int blocks_needed = blocks_for_tokens(tokens_total, pimpl->params.block_size);
        int prefix_blocks = 0;

        if (pimpl->prefix_cache && !req.prompt_tokens.empty()) {
            int start = 0;
            if (req.state == PREEMPTED && req.n_prompt_processed > 0) {
                start = req.n_prompt_processed;
            }
            int remaining = req.n_prompt - start;
            if (remaining > 0) {
                prefix_match m = pimpl->prefix_cache->lookup(
                    req.prompt_tokens.data() + start, remaining);
                if (m.n_blocks > 0) {
                    prefix_blocks = m.n_blocks;
                }
            }
        }

        int new_blocks = blocks_needed - prefix_blocks;
        if (new_blocks < 0) new_blocks = 0;

        if (new_blocks > pimpl->count_free_blocks()) continue;

        pimpl->do_alloc_blocks(new_blocks);
        req.block_ids.resize(blocks_needed);
        req.state = RUNNING;
        n_running++;
    }

    // 4. build_batch
    n_running = 0;
    int32_t decode_count = 0;
    for (auto & req : all) {
        if (req.state == RUNNING) {
            n_running++;
            if (req.n_prompt_processed < req.n_prompt) {
                // prefill — counted separately
            } else {
                decode_count++;
            }
        }
    }

    if (n_running == 0) {
        return result;
    }

    int32_t budget = std::min(pimpl->params.max_num_batched_tokens, pimpl->params.n_max_batch);
    int32_t prefill_budget = budget - decode_count;

    auto & bd = result.batch_data;
    bd.tokens.reserve(budget);
    bd.positions.reserve(budget);
    bd.logits_flags.reserve(budget);
    bd.seq_id_counts.reserve(budget);
    bd.seq_id_ptrs.clear();

    // prefill pass: add chunks of prefill requests
    for (auto & req : all) {
        if (req.state != RUNNING) continue;
        if (!req.is_prefill) continue;
        if (prefill_budget <= 0) break;

        int remaining = req.n_prompt - req.n_prompt_processed;
        int chunk = std::min(remaining, prefill_budget);
        for (int i = 0; i < chunk; i++) {
            bd.tokens.push_back(req.prompt_tokens[req.n_prompt_processed + i]);
            bd.positions.push_back(req.n_prompt_processed + i);
            bd.logits_flags.push_back(0);
            bd.seq_id_counts.push_back(1);
            bd.seq_ids_flat.push_back(req.seq_id);
            bd.seq_id_ptrs.push_back(&bd.seq_ids_flat.back());
        }
        req.n_prompt_processed += chunk;
        prefill_budget -= chunk;
        if (req.n_prompt_processed == req.n_prompt) {
            req.is_prefill = false;
        }
    }

    // decode pass: add 1 token per decoding request
    for (auto & req : all) {
        if (req.state != RUNNING) continue;
        if (req.is_prefill) continue;
        bd.tokens.push_back(req.generated_tokens.back());
        bd.positions.push_back(req.n_prompt_processed + (int)req.generated_tokens.size() - 1);
        bd.logits_flags.push_back(1);
        bd.seq_id_counts.push_back(1);
        bd.seq_ids_flat.push_back(req.seq_id);
        bd.seq_id_ptrs.push_back(&bd.seq_ids_flat.back());
    }

    if (bd.tokens.empty()) {
        return result;
    }

    result.batch.n_tokens = (int32_t)bd.tokens.size();
    result.batch.token    = bd.tokens.data();
    result.batch.pos      = bd.positions.data();
    result.batch.logits   = bd.logits_flags.data();
    result.batch.n_seq_id = bd.seq_id_counts.data();
    result.batch.seq_id   = bd.seq_id_ptrs.empty() ? nullptr : bd.seq_id_ptrs.data();

    result.n_sequences = n_running;
    result.action = SCHED_DECODE;
    return result;
}

bool llama_scheduler::has_pending() const {
    for (auto & req : pimpl->all_requests) {
        if (req.state == WAITING || req.state == PREEMPTED || req.state == RUNNING) {
            return true;
        }
    }
    return false;
}

int llama_scheduler::n_waiting() const {
    int count = 0;
    for (auto & req : pimpl->all_requests) {
        if (req.state == WAITING) count++;
    }
    return count;
}

int llama_scheduler::n_running() const {
    int count = 0;
    for (auto & req : pimpl->all_requests) {
        if (req.state == RUNNING) count++;
    }
    return count;
}

int llama_scheduler::n_free_blocks() const {
    return pimpl->count_free_blocks();
}
