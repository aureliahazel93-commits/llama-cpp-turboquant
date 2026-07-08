#ifndef LLAMA_SCHEDULER_H
#define LLAMA_SCHEDULER_H

#include "llama.h"
#include <cstdint>
#include <vector>
#include <deque>

struct llama_kv_cache_paged;
struct llama_prefix_cache;

enum sched_request_state {
    WAITING    = 0,
    RUNNING    = 1,
    PREEMPTED  = 2,
    FINISHED   = 3,
};

enum sched_stop_reason {
    STOP_NONE      = 0,
    STOP_EOS       = 1,
    STOP_LIMIT     = 2,
    STOP_WORD      = 3,
    STOP_ABORTED   = 4,
};

enum sched_step_action {
    SCHED_DECODE  = 0,
    SCHED_NOTHING = 1,
    SCHED_ERROR   = 2,
};

struct sched_request_params {
    const llama_token * prompt_tokens = nullptr;
    int                  n_prompt     = 0;
    int32_t              priority     = 0;
    int32_t              n_predict    = -1;
};

struct sched_request {
    uint64_t                 id;
    sched_request_state      state;
    int32_t                  priority;
    int32_t                  preemption_count;
    std::vector<llama_token> prompt_tokens;
    int                      n_prompt;
    int                      n_prompt_processed;
    std::vector<llama_token> generated_tokens;
    std::vector<int>         block_ids;
    int32_t                  seq_id;
    int32_t                  n_predict;
    int32_t                  n_predict_remaining;
    sched_stop_reason        stop_reason;
    bool                     is_prefill;
    int64_t                  arrival_time_us;
};

struct sched_completion {
    uint64_t                 id;
    std::vector<llama_token> tokens;
    sched_stop_reason        stop_reason;
    int32_t                  n_prompt;
    int32_t                  n_generated;
};

struct sched_batch_data {
    std::vector<llama_token>   tokens;
    std::vector<llama_pos>     positions;
    std::vector<int8_t>        logits_flags;
    std::vector<int32_t>       seq_id_counts;
    std::vector<llama_seq_id>  seq_ids_flat;
    std::vector<llama_seq_id *> seq_id_ptrs;
};

struct sched_result {
    sched_step_action              action;
    struct llama_batch             batch;
    int32_t                        n_sequences;
    std::vector<sched_completion>  completions;
    sched_batch_data               batch_data;
};

struct llama_scheduler_params {
    int32_t n_max_batch;
    int32_t n_max_seq;
    int32_t block_size;
    int32_t n_blocks_total;
    bool    cache_aware;
    int32_t max_num_batched_tokens;
};

class llama_scheduler {
public:
    explicit llama_scheduler(const struct llama_scheduler_params & params);
    ~llama_scheduler();

    void set_kv_cache(struct llama_kv_cache_paged * kv_cache);
    void set_prefix_cache(struct llama_prefix_cache * prefix_cache);

    uint64_t add_request(const struct sched_request_params & params);
    bool     abort_request(uint64_t id);
    int      notify_token(uint64_t id, llama_token token, sched_stop_reason stop);
    int      free_blocks(uint64_t id);

    struct sched_result step();

    bool has_pending() const;
    int  n_waiting() const;
    int  n_running() const;
    int  n_free_blocks() const;

private:
    struct impl;
    impl * pimpl;
};

#endif // LLAMA_SCHEDULER_H
