#include "llama-grammar-fsm.h"

#include <cstdint>
#include <string>

static std::string codepoint_to_utf8(uint32_t cp) {
    std::string s;
    if (cp < 0x80) {
        s += (char)cp;
    } else if (cp < 0x800) {
        s += (char)(0xC0 | (cp >> 6));
        s += (char)(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        s += (char)(0xE0 | (cp >> 12));
        s += (char)(0x80 | ((cp >> 6) & 0x3F));
        s += (char)(0x80 | (cp & 0x3F));
    } else if (cp < 0x110000) {
        s += (char)(0xF0 | (cp >> 18));
        s += (char)(0x80 | ((cp >> 12) & 0x3F));
        s += (char)(0x80 | ((cp >> 6) & 0x3F));
        s += (char)(0x80 | (cp & 0x3F));
    }
    return s;
}

static int find_transition_from(const grammar_fsm & fsm, int state_id) {
    for (int i = 0; i < fsm.n_transitions; i++) {
        if (fsm.transitions[i].from_state == state_id) {
            return i;
        }
    }
    return -1;
}

static int count_transitions_from(const grammar_fsm & fsm, int state_id) {
    int count = 0;
    for (int i = 0; i < fsm.n_transitions; i++) {
        if (fsm.transitions[i].from_state == state_id) {
            count++;
        }
    }
    return count;
}

int fsm_compress_jumps(grammar_fsm & fsm) {
    fsm.jump_paths.clear();

    for (int s = 0; s < fsm.n_states; s++) {
        if (count_transitions_from(fsm, s) != 1) {
            continue;
        }

        int first_trans = find_transition_from(fsm, s);
        if (first_trans < 0) {
            continue;
        }

        jump_path jp;
        jp.start_state = s;
        jp.end_state = s;
        jp.total_chars = 0;

        int cur = s;
        while (jp.total_chars < FSM_MAX_JUMP_LENGTH) {
            int n_out = count_transitions_from(fsm, cur);
            if (n_out != 1) {
                break;
            }

            int ti = find_transition_from(fsm, cur);
            if (ti < 0) {
                break;
            }

            const auto & t = fsm.transitions[ti];
            jp.forced_chars.push_back(t.character);
            jp.total_chars++;

            int next = t.to_state;
            jp.end_state = next;

            int n_next = count_transitions_from(fsm, next);
            if (n_next != 1) {
                break;
            }

            cur = next;
        }

        if (jp.total_chars >= 2 && jp.end_state != jp.start_state) {
            std::string utf8;
            for (uint32_t cp : jp.forced_chars) {
                utf8 += codepoint_to_utf8(cp);
            }

            for (size_t i = 0; i < utf8.size(); i++) {
                jp.forced_tokens.push_back((llama_token)(unsigned char)utf8[i]);
            }

            fsm.jump_paths.push_back(std::move(jp));
        }
    }

    return (int)fsm.jump_paths.size();
}

const jump_path * fsm_find_jump(const grammar_fsm & fsm, int current_state) {
    for (size_t i = 0; i < fsm.jump_paths.size(); i++) {
        if (fsm.jump_paths[i].start_state == current_state) {
            return &fsm.jump_paths[i];
        }
    }
    return nullptr;
}

std::vector<llama_token> grammar_jump_forward(const grammar_fsm & fsm, int current_state, const struct llama_vocab & vocab) {
    (void)vocab;

    const jump_path * jp = fsm_find_jump(fsm, current_state);
    if (jp == nullptr) {
        return {};
    }

    return jp->forced_tokens;
}

bool fsm_build_from_string(grammar_fsm & fsm, const char * grammar_str) {
    fsm_free(fsm);
    if (!grammar_str) return false;

    int cur_state = 0;
    fsm.states.push_back({0, 0});
    fsm.initial_state = 0;
    fsm.n_states = 1;
    fsm.n_transitions = 0;

    const char * p = grammar_str;
    while (*p) {
        if (*p == ' ') { p++; continue; }
        if (*p == '"') {
            p++;
            while (*p && *p != '"') {
                uint32_t cp = (unsigned char)*p;
                if ((cp & 0x80) == 0) { p++; }
                else if ((cp & 0xE0) == 0xC0) { cp = (cp & 0x1F) << 6; cp |= ((unsigned char)*++p & 0x3F); p++; }
                else if ((cp & 0xF0) == 0xE0) { cp = (cp & 0x0F) << 12; cp |= ((unsigned char)*++p & 0x3F) << 6; cp |= ((unsigned char)*++p & 0x3F); p++; }
                else if ((cp & 0xF8) == 0xF0) { cp = (cp & 0x07) << 18; cp |= ((unsigned char)*++p & 0x3F) << 12; cp |= ((unsigned char)*++p & 0x3F) << 6; cp |= ((unsigned char)*++p & 0x3F); p++; }
                else { p++; continue; }

                int next_state = fsm.n_states++;
                fsm.states.push_back({next_state, 0});
                fsm.transitions.push_back({cur_state, next_state, cp, 0, cp, cp, false});
                fsm.n_transitions++;
                cur_state = next_state;
            }
            if (*p == '"') p++;
        } else if (*p == '[') {
            p++;
            uint32_t range_start = 0, range_end = 0;
            while (*p && *p != ']') {
                if (*p == '-' && range_start != 0) {
                    p++;
                    range_end = (unsigned char)*p++;
                    for (uint32_t c = range_start; c <= range_end; c++) {
                        int next_state = fsm.n_states++;
                        fsm.states.push_back({next_state, 0});
                        fsm.transitions.push_back({cur_state, next_state, c, 0, c, c, false});
                        fsm.n_transitions++;
                    }
                    range_start = 0;
                } else {
                    range_start = (unsigned char)*p++;
                }
            }
            if (*p == ']') p++;
        } else {
            p++;
        }
    }

    if (fsm.n_states > 0) {
        fsm.states.back().n_transitions = -1;
    }

    return fsm.n_transitions > 0;
}

void fsm_free(grammar_fsm & fsm) {
    fsm.states.clear();
    fsm.transitions.clear();
    fsm.jump_paths.clear();
    fsm.initial_state = 0;
    fsm.n_states = 0;
    fsm.n_transitions = 0;
}
