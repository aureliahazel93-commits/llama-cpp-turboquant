#pragma once

#include "llama.h"

#include <cstdint>
#include <vector>

#define FSM_MAX_JUMP_LENGTH 64

struct fsm_state {
    int id;
    int n_transitions;
};

struct fsm_transition {
    int from_state;
    int to_state;
    uint32_t character;
    uint32_t char_mask;
    uint32_t char_lower;
    uint32_t char_upper;
    bool is_forced;
};

struct jump_path {
    int start_state;
    int end_state;
    std::vector<uint32_t> forced_chars;
    std::vector<llama_token> forced_tokens;
    int total_chars;
};

struct grammar_fsm {
    std::vector<fsm_state> states;
    std::vector<fsm_transition> transitions;
    int initial_state;
    int n_states;
    int n_transitions;

    std::vector<jump_path> jump_paths;
};

int fsm_compress_jumps(grammar_fsm & fsm);

const jump_path * fsm_find_jump(const grammar_fsm & fsm, int current_state);

bool fsm_build_from_string(grammar_fsm & fsm, const char * grammar_str);

void fsm_free(grammar_fsm & fsm);

// Returns forced tokens for batch prefill, or empty vector if no jump possible
std::vector<llama_token> grammar_jump_forward(const grammar_fsm & fsm, int current_state, const struct llama_vocab & vocab);
