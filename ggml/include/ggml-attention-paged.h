#pragma once

#include "ggml.h"

#ifdef  __cplusplus
extern "C" {
#endif

// GGML_OP_ATTN_PAGED = 579
// Paged KV-cache attention op (multi-query / grouped-query friendly)
//
// Inputs:
//   q            [n_head, n_token_q,   1,          head_dim]
//   k_blocks     [n_blocks, n_head_kv, block_size, head_dim]
//   v_blocks     [n_blocks, n_head_kv, block_size, head_dim]
//   block_table  [n_seq,      max_blocks_per_seq]
//   seq_lens     [n_seq]
//   mask         [1, n_token_q, n_token_q, 1]   (optional)
//
// Output: [head_dim, n_tokens_out, 1, n_seq]

GGML_API struct ggml_tensor * ggml_attention_paged(
        struct ggml_context * ctx,
        struct ggml_tensor  * q,           // [n_head, n_token_q, 1,          head_dim]
        struct ggml_tensor  * k_blocks,     // [n_blocks, n_head_kv, block_size, head_dim]
        struct ggml_tensor  * v_blocks,     // [n_blocks, n_head_kv, block_size, head_dim]
        struct ggml_tensor  * block_table,  // [n_seq,      max_blocks_per_seq]
        struct ggml_tensor  * seq_lens,     // [n_seq]
        struct ggml_tensor  * mask,         // optional attention mask
        float                 scale,
        float                 max_bias,
        float                 logit_softcap);

GGML_API void ggml_attention_paged_set_params(
        struct ggml_tensor * a,
        struct ggml_tensor * k_blocks,
        struct ggml_tensor * v_blocks,
        struct ggml_tensor * block_table,
        struct ggml_tensor * seq_lens);

#ifdef  __cplusplus
}
#endif
