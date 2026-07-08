// cat-first: tranche 1

#pragma once

#include "llama-kv-cache.h"

#include <stdexcept>
#include <unordered_map>
#include <vector>

//
// llama_kv_cache_paged_context (forward declaration)
//

struct llama_kv_cache_paged_context;

//
// llama_kv_cache_paged
//

struct kv_block_pool {
    ggml_tensor * k_blocks;   // [n_blocks, n_head_kv, block_size, head_dim]
    ggml_tensor * v_blocks;   // [n_blocks, n_head_kv, block_size, head_dim]
    int         * free_list;  // linked list of free block IDs; -1 terminator
    int           n_free;
    int           n_total;
};

struct seq_block_table {
    int32_t * block_ids;      // [n_blocks] physical block IDs for this sequence
    int       n_blocks;       // number of blocks in table
    int       n_allocated;    // number of blocks currently used
};

struct block_hash_entry {
    uint64_t hash;
    int      block_id;
    int      ref_count;       // 0=free, 1=sole owner, >1=shared (CoW)
    int      prev_free;       // intrusive doubly-linked LRU free list
    int      next_free;
};

class llama_kv_cache_paged : public llama_memory_i {
public:
    llama_kv_cache_paged(
            const llama_model & model,
                    ggml_type   type_k,
                    ggml_type   type_v,
                         bool   v_trans,
                         bool   offload,
                         bool   unified,
                     uint32_t   n_total_blocks,
                     uint32_t   block_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_pad,
               llama_swa_type   swa_type,
        const layer_filter_cb & filter,
        const  layer_reuse_cb & reuse,
        const  layer_share_cb & share);

    ~llama_kv_cache_paged();

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    //
    // paged KV cache specific API
    //

    llama_kv_cache::slot_info_vec_t prepare(const std::vector<llama_ubatch> & ubatches);

    void update(llama_context * lctx, bool do_shift);

    //
    // block allocator
    //

    int  alloc_block();
    void free_block(int block_id);

    //
    // copy-on-write
    //

    void  inc_ref(int block_id);
    void  dec_ref(int block_id);

    int n_blocks_total() const { return pool.n_total; }
    int n_blocks_free()  const { return pool.n_free; }

    void enable_kv_shard(int tp_rank, int tp_size);
    void disable_kv_shard();
    bool is_sharded() const;
    int  shard_rank() const;
    int  shard_size() const;

private:
    friend struct llama_kv_cache_paged_context;

    const llama_model & model;

    const uint32_t block_size = 16;
    const uint32_t n_seq_max  = 1;
    const uint32_t n_pad      = 1;
    const llama_swa_type swa_type = LLAMA_SWA_TYPE_NONE;

    kv_block_pool pool;

    // seq_id -> seq_block_table
    std::unordered_map<llama_seq_id, seq_block_table> seq_tables;

    // block_id -> block_hash_entry (for CoW tracking)
    std::unordered_map<int, block_hash_entry> block_entries;

    int head_free;          // head of free-list (-1 if empty)
    int tail_free;          // tail of free-list (for eviction order)

    int kv_shard_rank = -1;
    int kv_shard_size = 1;
};
