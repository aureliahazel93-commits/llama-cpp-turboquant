#include "llama.h"
#include <algorithm>
#include <cmath>
#include <cstdint>

#define PHANTOM_MAX_WORKERS 8
#define PHANTOM_MEASURE_WINDOW 100
#define PHANTOM_IMPROVEMENT_THRESHOLD 0.02f
#define PHANTOM_COOLDOWN_ROUNDS 200

struct phantom_scaler {
    int32_t n_workers;
    int32_t rounds_since_last_change;
    int32_t cooldown_remaining;
    float throughput_history[PHANTOM_MAX_WORKERS][PHANTOM_MEASURE_WINDOW];
    int32_t history_idx[PHANTOM_MAX_WORKERS];
    int32_t history_count[PHANTOM_MAX_WORKERS];
    float last_throughput;
};

void phantom_scaler_init(struct phantom_scaler * s, int32_t initial_workers) {
    s->n_workers = initial_workers;
    s->rounds_since_last_change = 0;
    s->cooldown_remaining = 0;
    s->last_throughput = 0.0f;
    for (int i = 0; i < PHANTOM_MAX_WORKERS; i++) {
        s->history_idx[i] = 0;
        s->history_count[i] = 0;
    }
}

void phantom_scaler_record(struct phantom_scaler * s, int32_t worker, float tok_per_sec) {
    if (worker < 0 || worker >= PHANTOM_MAX_WORKERS) return;
    int32_t idx = s->history_idx[worker] % PHANTOM_MEASURE_WINDOW;
    s->throughput_history[worker][idx] = tok_per_sec;
    s->history_idx[worker]++;
    s->history_count[worker] = std::min(s->history_count[worker] + 1, PHANTOM_MEASURE_WINDOW);
}

bool phantom_scaler_step(struct phantom_scaler * s, float current_throughput) {
    if (s->cooldown_remaining > 0) {
        s->cooldown_remaining--;
        return false;
    }
    s->rounds_since_last_change++;
    if (s->rounds_since_last_change < PHANTOM_MEASURE_WINDOW) return false;

    bool changed = false;

    if (s->n_workers < PHANTOM_MAX_WORKERS && s->last_throughput > 0.0f) {
        float improvement = (current_throughput - s->last_throughput) / s->last_throughput;
        if (improvement >= PHANTOM_IMPROVEMENT_THRESHOLD) {
            s->n_workers++;
            changed = true;
        }
    }

    if (!changed && s->n_workers > 1 && s->last_throughput > 0.0f) {
        float decline = (s->last_throughput - current_throughput) / s->last_throughput;
        if (decline >= PHANTOM_IMPROVEMENT_THRESHOLD) {
            s->n_workers--;
            changed = true;
        }
    }

    if (changed) {
        s->rounds_since_last_change = 0;
        s->cooldown_remaining = PHANTOM_COOLDOWN_ROUNDS;
    }
    s->last_throughput = current_throughput;
    return changed;
}
