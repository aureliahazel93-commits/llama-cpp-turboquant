#include "llama-grammar.h"
#include "llama.h"
#include <cstring>
#include "llama-grammar-fsm.h"

bool xgrammar_build_fsm(grammar_fsm & fsm, const char * grammar_str) {
    (void)fsm;
    (void)grammar_str;
    return false;
}

bool xgrammar_accept_token(const grammar_fsm & fsm, int current_state, llama_token token) {
    (void)fsm;
    (void)current_state;
    (void)token;
    return true;
}

bool xgrammar_is_available() {
    return false;
}
