#include "llama-scheduler.h"
#include "llama.h"

#include <atomic>
#include <mutex>
#include <thread>

struct overlap_batch_slot {
    struct sched_result  result;
    std::atomic<bool>    ready{false};
    std::atomic<bool>    consumed{false};
    std::atomic<bool>    needs_rebuild{false};

    void reset() {
        ready = false;
        consumed = false;
        needs_rebuild = false;
    }
};

class llama_scheduler_overlap {
public:
    explicit llama_scheduler_overlap(const llama_scheduler_params & params);
    ~llama_scheduler_overlap();

    void start();
    void stop();

    struct sched_result next_batch();

    void submit_result(const struct sched_result & result,
                       const std::vector<uint64_t> & req_ids,
                       const std::vector<llama_token> & tokens,
                       const std::vector<sched_stop_reason> & stops);

    void set_kv_cache(llama_kv_cache_paged * kv_cache);
    void set_prefix_cache(llama_prefix_cache * prefix_cache);
    uint64_t add_request(const sched_request_params & params);
    bool abort_request(uint64_t id);
    bool has_pending() const;

private:
    void planner_loop();

    llama_scheduler scheduler;
    std::thread planner_thread;
    std::atomic<bool> running{false};

    overlap_batch_slot slots[2];
    std::atomic<int> current_slot{0};
    std::atomic<int> planned_slot{1};

    mutable std::mutex submit_mutex;


    struct pending_update {
        std::vector<uint64_t> req_ids;
        std::vector<llama_token> tokens;
        std::vector<sched_stop_reason> stops;
        std::atomic<bool> has_data{false};
    };
    pending_update pending_buf;
};

llama_scheduler_overlap::llama_scheduler_overlap(const llama_scheduler_params & params)
    : scheduler(params) {}

llama_scheduler_overlap::~llama_scheduler_overlap() {
    stop();
}

void llama_scheduler_overlap::start() {
    running.store(true);
    planner_thread = std::thread(&llama_scheduler_overlap::planner_loop, this);
}

void llama_scheduler_overlap::stop() {
    bool expected = true;
    if (!running.compare_exchange_strong(expected, false)) return;
    if (planner_thread.joinable()) {
        planner_thread.join();
    }
}

void llama_scheduler_overlap::planner_loop() {
    while (running.load()) {
        if (pending_buf.has_data.load()) {
            std::vector<uint64_t> req_ids;
            std::vector<llama_token> tokens;
            std::vector<sched_stop_reason> stops;
            {
                std::lock_guard<std::mutex> lock(submit_mutex);
                req_ids = std::move(pending_buf.req_ids);
                tokens = std::move(pending_buf.tokens);
                stops = std::move(pending_buf.stops);
                pending_buf.has_data.store(false);
            }
            for (size_t i = 0; i < req_ids.size(); i++) {
                scheduler.notify_token(req_ids[i], tokens[i], stops[i]);
            }
        }

        int pslot = planned_slot.load();
        auto & slot = slots[pslot];
        slot.result = scheduler.step();
        slot.ready.store(true);
        slot.consumed.store(false);

        while (slot.ready.load() && running.load()) {
            if (slot.consumed.load() || slot.needs_rebuild.load()) {
                slot.reset();
                break;
            }
            std::this_thread::yield();
        }
    }
}

struct sched_result llama_scheduler_overlap::next_batch() {
    int pslot = planned_slot.load();
    auto & slot = slots[pslot];

    while (!slot.ready.load() && running.load()) {
        std::this_thread::yield();
    }

    int old_current = current_slot.exchange(pslot);
    planned_slot.store(1 - old_current);

    slot.consumed.store(true);
    return slot.result;
}

void llama_scheduler_overlap::submit_result(const struct sched_result & /*result*/,
                                            const std::vector<uint64_t> & req_ids,
                                            const std::vector<llama_token> & tokens,
                                            const std::vector<sched_stop_reason> & stops) {
    bool any_stop = false;
    for (size_t i = 0; i < stops.size(); i++) {
        if (stops[i] != STOP_NONE) {
            any_stop = true;
            break;
        }
    }

    {
        std::lock_guard<std::mutex> lock(submit_mutex);
        pending_buf.req_ids = req_ids;
        pending_buf.tokens = tokens;
        pending_buf.stops = stops;
        pending_buf.has_data.store(true);
    }

    if (any_stop) {
        int pslot = planned_slot.load();
        slots[pslot].needs_rebuild.store(true);
    }
}

void llama_scheduler_overlap::set_kv_cache(llama_kv_cache_paged * kv_cache) {
    scheduler.set_kv_cache(kv_cache);
}

void llama_scheduler_overlap::set_prefix_cache(llama_prefix_cache * prefix_cache) {
    scheduler.set_prefix_cache(prefix_cache);
}

uint64_t llama_scheduler_overlap::add_request(const sched_request_params & params) {
    return scheduler.add_request(params);
}

bool llama_scheduler_overlap::abort_request(uint64_t id) {
    return scheduler.abort_request(id);
}

bool llama_scheduler_overlap::has_pending() const {
    return scheduler.has_pending();
}
