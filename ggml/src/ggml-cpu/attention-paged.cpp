#include "ops.h"

#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "common.h"
#include "ggml.h"

#include <cstring>
#include <cmath>

// T2: Gather K/V blocks for one (seq, head_kv) pair into contiguous scratch buffers.
//
// k_blocks:  [n_blocks, n_head_kv, block_size, head_dim]
// v_blocks:  [n_blocks, n_head_kv, block_size, head_dim]
// block_table: [n_seq, max_blocks_per_seq]  (int32)
// seq_lens: [n_seq]                          (int32)
//
// After gather: K_gathered[0..seq_len-1] and V_gathered[0..seq_len-1]
// hold the actual K/V tokens for this sequence (truncated to seq_len).

static void gather_kv_blocks_f16(
        const ggml_fp16_t * k_blocks_data,
        const ggml_fp16_t * v_blocks_data,
        const int32_t     * block_table_data,
        const int32_t     * seq_lens_data,
        int n_blocks, int block_size, int head_dim,
        int n_head_kv,
        int n_seq, int max_blocks_per_seq,
        int iseq, int ihead_kv,
        ggml_fp16_t * K_gathered,
        ggml_fp16_t * V_gathered) {

    const int seq_len = seq_lens_data[iseq];
    const int n_blocks_needed = (seq_len + block_size - 1) / block_size;

    int gathered_tokens = 0;

    for (int b = 0; b < n_blocks_needed; b++) {
        // block_table is [n_seq, max_blocks_per_seq], row-major
        const int block_id = block_table_data[iseq * max_blocks_per_seq + b];

        // Number of valid tokens in this block (last block may be partial)
        const int tokens_in_block = (b == n_blocks_needed - 1)
            ? (seq_len - gathered_tokens)
            : block_size;
        if (tokens_in_block <= 0) break;

        // K_blocks: [n_blocks, n_head_kv, block_size, head_dim]
        const int elems_per_block = n_head_kv * block_size * head_dim;
        const int k_off = block_id * elems_per_block
                       + ihead_kv * block_size * head_dim;
        memcpy(K_gathered + gathered_tokens * head_dim,
               k_blocks_data + k_off,
               tokens_in_block * head_dim * sizeof(ggml_fp16_t));

        // V_blocks: same layout
        const int v_off = block_id * elems_per_block
                       + ihead_kv * block_size * head_dim;
        memcpy(V_gathered + gathered_tokens * head_dim,
               v_blocks_data + v_off,
               tokens_in_block * head_dim * sizeof(ggml_fp16_t));

        gathered_tokens += tokens_in_block;
    }
}

// T3: Main compute function — gather K/V per sequence, then softmax-attention per head.

void ggml_compute_forward_attn_paged(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * q          = dst->src[0];  // [n_head, n_token_q, 1, head_dim]
    const ggml_tensor * k_blocks   = dst->src[1];  // [n_blocks, n_head_kv, block_size, head_dim]
    const ggml_tensor * v_blocks   = dst->src[2];  // [n_blocks, n_head_kv, block_size, head_dim]
    const ggml_tensor * block_table = dst->src[3]; // [n_seq, max_blocks_per_seq]
    const ggml_tensor * seq_lens   = dst->src[4];  // [n_seq]
    const ggml_tensor * mask       = dst->src[5];  // [1, n_token_q, n_token_q, 1] optional

    GGML_TENSOR_LOCALS(int64_t, neq, q,         ne)
    GGML_TENSOR_LOCALS(size_t,  nbq, q,         nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k_blocks,  ne)
    GGML_TENSOR_LOCALS(size_t,  nbk, k_blocks,  nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v_blocks,  ne)
    GGML_TENSOR_LOCALS(size_t,  nbv, v_blocks,  nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,       ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,       nb)

    const int ith = params->ith;
    const int nth = params->nth;

    // Tensor dimensions (from header comment)
    const int head_dim       = (int) neq0;         // per-head dimension
    const int n_token_q      = (int) neq1;         // query token count (= n_tokens_out)
    const int n_seq          = (int) seq_lens->ne[0];
    const int n_head         = (int) neq2;         // number of query heads
    const int n_head_kv      = (int) nek1;         // number of KV heads
    const int block_size     = (int) nek2;         // tokens per KV block
    const int n_blocks       = (int) nek0;         // total number of KV blocks
    const int max_blocks_per_seq = (int) block_table->ne[1];

    // op_params: scale=0, max_bias=1, logit_softcap=2
    float scale = 1.0f;
    memcpy(&scale,         (float *) dst->op_params + 0, sizeof(float));
    GGML_ASSERT(scale != 0.0f && "attn_paged: scale must not be zero");

    const float sqrt_d = sqrtf((float) head_dim);
    scale /= sqrt_d;

    // --- scratch buffer layout ---
    // Per-head, per-sequence workspace:
    //   K_gathered : [max_seq_len, head_dim]  F16
    //   V_gathered : [max_seq_len, head_dim]  F16
    //   scores     : [n_token_q, max_seq_len]  F32
    //   maxes      : [n_token_q]               F32
    //   sums       : [n_token_q]               F32
    // We allocate max_seq_len = sum of all seq_lens (worst case) but
    // in practice only need max(seq_lens).  Use the largest seq_len.

    // Compute max_seq_len for scratch sizing
    int max_seq_len = 0;
    {
        const int32_t * sl = (const int32_t *) seq_lens->data;
        for (int i = 0; i < n_seq; i++) {
            if (sl[i] > max_seq_len) max_seq_len = sl[i];
        }
    }

    const size_t kv_bytes  = (size_t) max_seq_len * head_dim * sizeof(ggml_fp16_t);
    const size_t sc_bytes  = (size_t) n_token_q  * max_seq_len * sizeof(float);
    const size_t mx_bytes  = (size_t) n_token_q  * sizeof(float);
    const size_t sm_bytes  = (size_t) n_token_q  * sizeof(float);
    const size_t per_head_bytes = kv_bytes * 2 + sc_bytes + mx_bytes + sm_bytes;

    // --- dispatch: parallelise over (head * seq) pairs ---
    const int total_work = n_head * n_seq;
    const int dr = (total_work + nth - 1) / nth;
    const int w0 = dr * ith;
    const int w1 = MIN(w0 + dr, total_work);

    if (ith == 0) {
        memset(dst->data, 0, nb0 * ne0 * ne1 * ne2 * ne3);
    }
    ggml_barrier(params->threadpool);

    // raw data pointers
    const ggml_fp16_t * q_data      = (const ggml_fp16_t *) q->data;
    const ggml_fp16_t * k_data      = (const ggml_fp16_t *) k_blocks->data;
    const ggml_fp16_t * v_data      = (const ggml_fp16_t *) v_blocks->data;
    const int32_t     * bt_data     = (const int32_t *) block_table->data;
    const int32_t     * sl_data     = (const int32_t *) seq_lens->data;
    const ggml_fp16_t * mask_data   = mask ? (const ggml_fp16_t *) mask->data : nullptr;

    // wdata layout per work-item:
    //   K_gathered [max_seq_len * head_dim]
    //   V_gathered [max_seq_len * head_dim]
    //   scores     [n_token_q * max_seq_len]
    //   maxes      [n_token_q]
    //   sums       [n_token_q]

    for (int work = w0; work < w1; work++) {
        const int ihead  = work / n_seq;   // query head index
        const int iseq   = work % n_seq;   // sequence index

        // Per-head scratch within wdata
        char * wbase = (char *) params->wdata + (size_t) work * per_head_bytes;

        ggml_fp16_t * K_gathered = (ggml_fp16_t *) wbase;
        ggml_fp16_t * V_gathered = K_gathered + max_seq_len * head_dim;
        float       * scores     = (float       *) (V_gathered + max_seq_len * head_dim);
        float       * maxes      = scores + n_token_q * max_seq_len;
        float       * sums       = maxes  + n_token_q;

        // --- T2: gather K/V for this sequence ---
        gather_kv_blocks_f16(
            k_data, v_data, bt_data, sl_data,
            n_blocks, block_size, head_dim, n_head_kv,
            n_seq, max_blocks_per_seq,
            iseq, ihead % n_head_kv,
            K_gathered, V_gathered);

        const int seq_len = sl_data[iseq];

        // --- Build per-seq mask slice (if mask present) ---
        // mask: [1, n_token_q, n_token_q, 1] in header comment
        // We treat it as [n_token_q, seq_len] after truncation.
        const ggml_fp16_t * mask_seq = nullptr;
        if (mask_data) {
            // mask stored as [1, n_token_q, n_token_q, 1] with strides:
            //   nb[0]=n_token_q*n_token_q, nb[1]=n_token_q, nb[2]=1, nb[3]=0
            // Truncate columns to seq_len
            // mask[iq, ik] = mask_data[iq * n_token_q + ik]   (row-major in dims 1..2)
            mask_seq = mask_data;  // use directly; truncate in the loop
        }

        // --- Build Q pointer for this head ---
        // Q: [n_head, n_token_q, 1, head_dim]
        // q_row[iq] = q_data + iq*nbq1 + ihead*nbq2
        const ggml_fp16_t * Q_head = q_data + ihead * nbq2;

        // --- T4: standard softmax-attention ---
        // Use per-row scratch from scores/maxes/sums already allocated above.
        for (int iq = 0; iq < n_token_q; iq++) {
            const ggml_fp16_t * q_row = Q_head + iq * head_dim;
            float q_max = -INFINITY;

            for (int ik = 0; ik < seq_len; ik++) {
                const ggml_fp16_t * k_row = K_gathered + ik * head_dim;
                float s = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    s += GGML_CPU_FP16_TO_FP32(q_row[d])
                       * GGML_CPU_FP16_TO_FP32(k_row[d]);
                }
                s *= scale;

                if (mask_seq) {
                    const float mv = GGML_CPU_FP16_TO_FP32(mask_seq[iq * n_token_q + ik]);
                    if (mv == -INFINITY) {
                        s = -INFINITY;
                    } else {
                        s += mv;
                    }
                }

                scores[iq * max_seq_len + ik] = s;
                if (s > q_max) q_max = s;
            }
            maxes[iq] = q_max;
        }

        // softmax numerator
        for (int iq = 0; iq < n_token_q; iq++) {
            const float q_max = maxes[iq];
            float q_sum = 0.0f;
            for (int ik = 0; ik < seq_len; ik++) {
                const float s = scores[iq * max_seq_len + ik];
                if (s == -INFINITY) {
                    scores[iq * max_seq_len + ik] = 0.0f;
                } else {
                    const float e = expf(s - q_max);
                    scores[iq * max_seq_len + ik] = e;
                    q_sum += e;
                }
            }
            sums[iq] = q_sum;
        }

        // weighted V sum + write output
        // Output: [head_dim, n_token_q, 1, n_seq]
        // dst[ihead, iq, 0, iseq] -> byte offset = ihead*nb1 + iq*nb2 + iseq*nb3
        for (int iq = 0; iq < n_token_q; iq++) {
            const float inv_sum = sums[iq] > 0.0f ? 1.0f / sums[iq] : 0.0f;
            ggml_fp16_t * dst_row = (ggml_fp16_t *)
                ((char *) dst->data + ihead * nb1 + iq * nb2 + iseq * nb3);

            for (int d = 0; d < head_dim; d++) {
                float acc = 0.0f;
                for (int ik = 0; ik < seq_len; ik++) {
                    const float w = scores[iq * max_seq_len + ik] * inv_sum;
                    const float v = GGML_CPU_FP16_TO_FP32(V_gathered[ik * head_dim + d]);
                    acc += w * v;
                }
                dst_row[d] = GGML_CPU_FP32_TO_FP16(acc);
            }
        }
    }
}
