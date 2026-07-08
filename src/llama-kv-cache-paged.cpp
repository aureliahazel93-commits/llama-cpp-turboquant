// cat-first: tranche 1

#include "llama-kv-cache-paged.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-model.h"
#include "llama-context.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>

//
// llama_kv_cache_paged_context
//

struct llama_kv_cache_paged_context : public llama_memory_context_i {
    const llama_kv_cache_paged * cache;
    std::vector<llama_ubatch> ubatches;
    size_t cur = 0;
    llama_memory_status status;

    std::vector<llama_kv_cache::slot_info> sinfos;

    llama_kv_cache_paged_context(const llama_kv_cache_paged * cache, llama_memory_status status)
        : cache(cache), status(status) {}

    llama_kv_cache_paged_context(
            const llama_kv_cache_paged * cache,
            std::vector<llama_kv_cache::slot_info> sinfos,
            std::vector<llama_ubatch> ubatches)
        : cache(cache), status(LLAMA_MEMORY_STATUS_SUCCESS),
          sinfos(std::move(sinfos)), ubatches(std::move(ubatches)) {}

    bool next() override {
        assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
        if (++cur >= ubatches.size()) {
            return false;
        }
        return true;
    }

    bool apply() override {
        assert(!llama_memory_status_is_fail(status));
        if (ubatches.empty()) {
            return true; // update context: nothing to do
        }
        // block table already committed in prepare()
        return true;
    }

    llama_memory_status get_status() const override { return status; }
    const llama_ubatch & get_ubatch() const override {
        assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
        return ubatches[cur];
    }

    uint32_t get_n_kv() const {
        uint32_t result = 0;
        for (uint32_t s = 0; s < sinfos[cur].n_stream(); ++s) {
            result = std::max(result, (uint32_t) sinfos[cur].idxs[s].size());
        }
        const uint32_t n_pad_cur = std::max(cache->n_pad, 256u);
        return std::max(n_pad_cur, GGML_PAD(result, n_pad_cur));
    }

    ggml_type type_k() const { return cache->pool.k_blocks->type; }
    ggml_type type_v() const { return cache->pool.v_blocks->type; }

    ggml_tensor * get_k(ggml_context * ctx, int32_t il) const { GGML_UNUSED(ctx); GGML_UNUSED(il); return cache->pool.k_blocks; }
    ggml_tensor * get_v(ggml_context * ctx, int32_t il) const { GGML_UNUSED(ctx); GGML_UNUSED(il); return cache->pool.v_blocks; }

    ggml_tensor * get_turbo_rotation()     const { return nullptr; }
    ggml_tensor * get_turbo_rotation_inv() const { return nullptr; }
    ggml_tensor * get_turbo_rot_forward()  const override { return nullptr; }
    ggml_tensor * get_turbo_rot_inverse()  const override { return nullptr; }
    ggml_tensor * get_turbo_innerq_scale_inv() const override { return nullptr; }

    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il) const;

    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    ggml_tensor * build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;

    void set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const;
    void set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const;

    void set_input_k_shift(ggml_tensor * dst) const;
    void set_input_kq_mask(ggml_tensor * dst, const llama_ubatch & ubatch, bool causal_attn) const;
    void set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch & ubatch) const;

    void set_input_k_rot(ggml_tensor * dst) const;
    void set_input_v_rot(ggml_tensor * dst) const;
};

//
// llama_kv_cache_paged
//

llama_kv_cache_paged::llama_kv_cache_paged(
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
      const  layer_share_cb & share)
    : model(model), block_size(block_size), n_seq_max(n_seq_max), n_pad(n_pad),
      swa_type(swa_type) {
    GGML_UNUSED(offload);
    GGML_UNUSED(unified);
    GGML_UNUSED(filter);
    GGML_UNUSED(reuse);
    GGML_UNUSED(share);
    GGML_UNUSED(v_trans);

    const uint32_t n_head_kv  = model.hparams.n_head_kv(0);
    const uint32_t n_head_kv_actual = is_sharded() ? n_head_kv / shard_size() : n_head_kv;
    const uint32_t head_dim_k = model.hparams.n_embd_head_k(0);
    const uint32_t head_dim_v = model.hparams.n_embd_head_v(0);

    LLAMA_LOG_INFO("%s: n_total_blocks=%u block_size=%u n_head_kv=%u head_dim_k=%u head_dim_v=%u\n",
                   __func__, n_total_blocks, block_size, n_head_kv_actual, head_dim_k, head_dim_v);

    ggml_init_params params = {
        /*.mem_size   =*/ 2 * ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx{ ggml_init(params) };

    pool.k_blocks = ggml_new_tensor_4d(ctx.get(), type_k, head_dim_k, block_size, n_head_kv_actual, n_total_blocks);
    pool.v_blocks = ggml_new_tensor_4d(ctx.get(), type_v, head_dim_v, block_size, n_head_kv_actual, n_total_blocks);
    ggml_format_name(pool.k_blocks, "paged_k_blocks");
    ggml_format_name(pool.v_blocks, "paged_v_blocks");

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_cpu_buffer_type());
    if (!buf) {
        throw std::runtime_error("failed to allocate buffer for paged KV cache blocks");
    }
    ggml_backend_buffer_clear(buf, 0);

    pool.n_total = n_total_blocks;
    pool.n_free  = n_total_blocks;
    pool.free_list = new int[n_total_blocks];

    for (int i = 0; i < (int) n_total_blocks; i++) {
        pool.free_list[i] = (i + 1 < (int) n_total_blocks) ? (i + 1) : -1;
    }
    head_free = 0;
    tail_free = (int) n_total_blocks - 1;
}

llama_kv_cache_paged::~llama_kv_cache_paged() {
    delete[] pool.free_list;
    for (auto & [seq_id, table] : seq_tables) {
        delete[] table.block_ids;
    }
    seq_tables.clear();
}
// cat-first: tranche 2

//
// block allocator
//

int llama_kv_cache_paged::alloc_block() {
    GGML_ASSERT(pool.n_free > 0 && "out of blocks");
    int bid = head_free;
    head_free = pool.free_list[bid];
    if (head_free == -1) tail_free = -1;
    pool.free_list[bid] = -1;
    pool.n_free--;
    block_entries[bid] = { 0, bid, 1, -1, -1 };
    return bid;
}

void llama_kv_cache_paged::free_block(int block_id) {
    pool.free_list[block_id] = -1;
    if (tail_free != -1) pool.free_list[tail_free] = block_id;
    tail_free = block_id;
    if (head_free == -1) head_free = block_id;
    pool.n_free++;
    block_entries.erase(block_id);
}

//
// copy-on-write
//

void llama_kv_cache_paged::inc_ref(int block_id) {
    block_entries[block_id].ref_count++;
}

void llama_kv_cache_paged::dec_ref(int block_id) {
    auto it = block_entries.find(block_id);
    if (it == block_entries.end()) return;
    if (--it->second.ref_count == 0) {
        free_block(block_id);
    }
}

//
// llama_memory_i
//

bool llama_kv_cache_paged::get_can_shift() const {
    return false;
}

void llama_kv_cache_paged::clear(bool data) {
    GGML_UNUSED(data);
    pool.n_free = pool.n_total;
    for (int i = 0; i < pool.n_total; i++) {
        pool.free_list[i] = (i + 1 < pool.n_total) ? (i + 1) : -1;
    }
    head_free = 0;
    tail_free = (int) pool.n_total - 1;
    seq_tables.clear();
    block_entries.clear();
}
// cat-first: tranche 3

//
// seq ops
//

bool llama_kv_cache_paged::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    auto it = seq_tables.find(seq_id);
    if (it == seq_tables.end()) return true;
    auto & table = it->second;
    if (table.n_blocks == 0) return true;
    if (p0 < 0) p0 = 0;
    if (p1 < 0) p1 = std::numeric_limits<llama_pos>::max();
    int64_t first_block = p0 / (int64_t) block_size;
    int64_t last_block  = (p1 - 1) / (int64_t) block_size;
    if (p0 >= p1) return true;
    int32_t write_idx = 0;
    for (int32_t i = 0; i < table.n_blocks; i++) {
        if (i >= (int32_t) first_block && i <= (int32_t) last_block) {
            dec_ref(table.block_ids[i]);
            continue;
        }
        if (write_idx != i) table.block_ids[write_idx] = table.block_ids[i];
        write_idx++;
    }
    table.n_blocks = write_idx;
    table.n_allocated = write_idx;
    if (table.n_blocks == 0) seq_tables.erase(it);
    return true;
}

void llama_kv_cache_paged::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    GGML_UNUSED(p0);
    GGML_UNUSED(p1);
    auto it_src = seq_tables.find(seq_id_src);
    if (it_src == seq_tables.end()) return;
    auto & src = it_src->second;
    seq_tables[seq_id_dst] = src;
    auto & dst = seq_tables[seq_id_dst];
    // duplicate block table, inc refs
    if (src.n_blocks > 0) {
        dst.block_ids = new int32_t[src.n_blocks];
        std::copy(src.block_ids, src.block_ids + src.n_blocks, dst.block_ids);
        dst.n_blocks   = src.n_blocks;
        dst.n_allocated = src.n_allocated;
        for (int i = 0; i < dst.n_blocks; i++) {
            inc_ref(dst.block_ids[i]);
        }
    }
}

void llama_kv_cache_paged::seq_keep(llama_seq_id seq_id) {
    std::vector<llama_seq_id> to_remove;
    for (auto & kv : seq_tables) {
        if (kv.first != seq_id) to_remove.push_back(kv.first);
    }
    for (auto sid : to_remove) {
        seq_tables.erase(sid);
    }
}

void llama_kv_cache_paged::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    GGML_UNUSED(seq_id); GGML_UNUSED(p0); GGML_UNUSED(p1); GGML_UNUSED(shift);
}

void llama_kv_cache_paged::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    GGML_UNUSED(seq_id); GGML_UNUSED(p0); GGML_UNUSED(p1); GGML_UNUSED(d);
}

llama_pos llama_kv_cache_paged::seq_pos_min(llama_seq_id seq_id) const {
    auto it = seq_tables.find(seq_id);
    if (it == seq_tables.end()) return -1;
    const auto & table = it->second;
    if (table.n_blocks == 0) return -1;
    return 0;
}

llama_pos llama_kv_cache_paged::seq_pos_max(llama_seq_id seq_id) const {
    auto it = seq_tables.find(seq_id);
    if (it == seq_tables.end()) return -1;
    const auto & table = it->second;
    if (table.n_blocks == 0) return -1;
    return (llama_pos) (table.n_blocks - 1) * (int64_t) block_size + block_size - 1;
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache_paged::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> ret;
    if (pool.k_blocks && pool.k_blocks->buffer) {
        ret[ggml_backend_buffer_get_type(pool.k_blocks->buffer)]
            += ggml_backend_buffer_get_size(pool.k_blocks->buffer);
    }
    if (pool.v_blocks && pool.v_blocks->buffer) {
        ret[ggml_backend_buffer_get_type(pool.v_blocks->buffer)]
            += ggml_backend_buffer_get_size(pool.v_blocks->buffer);
    }
    return ret;
}
// cat-first: tranche 4

//
// init_batch / init_full / init_update
//

llama_memory_context_ptr llama_kv_cache_paged::init_batch(
        llama_batch_allocr & balloc,
        uint32_t n_ubatch,
        bool embd_all) {
    GGML_UNUSED(embd_all);
    do {
        balloc.split_reset();
        std::vector<llama_ubatch> ubatches;
        while (true) {
            auto ubatch = balloc.split_simple(n_ubatch);
            if (ubatch.n_tokens == 0) break;
            ubatches.push_back(std::move(ubatch));
        }
        if (balloc.get_n_used() < balloc.get_n_tokens()) break;
        auto sinfos = prepare(ubatches);
        if (sinfos.empty()) break;
        return std::make_unique<llama_kv_cache_paged_context>(this, std::move(sinfos), std::move(ubatches));
    } while (false);
    return std::make_unique<llama_kv_cache_paged_context>(this, LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_kv_cache_paged::init_full() {
    std::vector<llama_kv_cache::slot_info> sinfos;
    sinfos.resize(1);
    sinfos[0].s0 = 0;
    sinfos[0].s1 = 0;
    sinfos[0].strm = { 0 };
    sinfos[0].idxs.resize(1, { 0 });
    return std::make_unique<llama_kv_cache_paged_context>(this, std::move(sinfos), std::vector<llama_ubatch>());
}

llama_memory_context_ptr llama_kv_cache_paged::init_update(llama_context * lctx, bool optimize) {
    GGML_UNUSED(lctx);
    GGML_UNUSED(optimize);
    return std::make_unique<llama_kv_cache_paged_context>(this, LLAMA_MEMORY_STATUS_NO_UPDATE);
}
// cat-first: tranche 5

//
// prepare / apply_ubatch
//

llama_kv_cache::slot_info_vec_t llama_kv_cache_paged::prepare(const std::vector<llama_ubatch> & ubatches) {
    llama_kv_cache::slot_info_vec_t res;

    for (const auto & ubatch : ubatches) {
        uint32_t n_tokens = ubatch.n_tokens;
        if (n_tokens == 0) return {};

        llama_kv_cache::slot_info sinfo;
        sinfo.s0 = LLAMA_MAX_SEQ;
        sinfo.s1 = 0;
        sinfo.resize(1);
        sinfo.strm[0] = 0;
        sinfo.idxs[0].reserve(n_tokens);

        for (uint32_t i = 0; i < n_tokens; i++) {
            llama_pos pos = ubatch.pos[i];
            int32_t block_id = (int32_t)(pos / (int64_t) block_size);

            // get or create block table for this seq
            auto it_table = seq_tables.find(ubatch.seq_id[i][0]);
            if (it_table == seq_tables.end()) {
                seq_tables[ubatch.seq_id[i][0]] = { nullptr, 0, 0 };
                it_table = seq_tables.find(ubatch.seq_id[i][0]);
            }
            auto & table = it_table->second;

            // grow block table
            if (block_id >= table.n_allocated) {
                int old_n = table.n_allocated;
                table.n_allocated = block_id + 1;
                int32_t * new_ids = new int32_t[table.n_allocated];
                if (old_n > 0 && table.block_ids) {
                    std::copy(table.block_ids, table.block_ids + old_n, new_ids);
                }
                for (int j = old_n; j < table.n_allocated; j++) {
                    new_ids[j] = -1;
                }
                delete[] table.block_ids;
                table.block_ids = new_ids;
            }

            if (table.block_ids[block_id] == -1) {
                table.block_ids[block_id] = alloc_block();
            }

            int32_t bid = table.block_ids[block_id];
            int32_t intra_off = (int32_t)(pos % (int64_t) block_size);
            sinfo.idxs[0].push_back((uint32_t)(bid * (int64_t) block_size + intra_off));
            sinfo.s0 = std::min(sinfo.s0, 0u);
            sinfo.s1 = std::max(sinfo.s1, 0u);
        }
        res.push_back(sinfo);
    }
    return res;
}
// cat-first: tranche 6

//
// update
//

void llama_kv_cache_paged::update(llama_context * lctx, bool do_shift) {
    GGML_UNUSED(lctx);
    GGML_UNUSED(do_shift);
}

// cat-first: tranche 7

//
// cpy_k / cpy_v
//

ggml_tensor * llama_kv_cache_paged_context::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const {
    GGML_UNUSED(ctx); GGML_UNUSED(il);
    auto * blocks = cache->pool.k_blocks;
    const int64_t n_tokens = k_cur->ne[2];
    const int64_t head_dim  = k_cur->ne[0];
    const int64_t n_head_kv = blocks->ne[1];
    const int64_t head_offset = cache->is_sharded() ? cache->shard_rank() * n_head_kv : 0;
    const int64_t blk_size  = cache->block_size;

    if (blocks->buffer) {
        const int64_t row_bytes = head_dim * n_head_kv * ggml_type_size(blocks->type);
        const int64_t src_row_stride = head_dim * k_cur->ne[1] * ggml_type_size(blocks->type);
        char * dst_data = (char *) blocks->data;
        const char * src_data = (const char *) k_cur->data;
        const int64_t * idxs_data = (const int64_t *) k_idxs->data;

        for (int64_t i = 0; i < n_tokens; i++) {
            int64_t flat  = idxs_data[i];
            int64_t bid  = flat / blk_size;
            int64_t offs = (flat % blk_size) * head_dim * n_head_kv
                         + bid * blk_size * head_dim * n_head_kv;
            memcpy(dst_data + offs, src_data + i * src_row_stride + head_offset * head_dim * ggml_type_size(blocks->type), row_bytes);
        }
    }
    return blocks;
}

ggml_tensor * llama_kv_cache_paged_context::cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il) const {
    GGML_UNUSED(ctx); GGML_UNUSED(il);
    auto * blocks = cache->pool.v_blocks;
    const int64_t n_tokens = v_cur->ne[2];
    const int64_t head_dim  = v_cur->ne[0];
    const int64_t n_head_kv = blocks->ne[1];
    const int64_t head_offset = cache->is_sharded() ? cache->shard_rank() * n_head_kv : 0;
    const int64_t blk_size  = cache->block_size;

    if (blocks->buffer) {
        const int64_t row_bytes = head_dim * n_head_kv * ggml_type_size(blocks->type);
        const int64_t src_row_stride = head_dim * v_cur->ne[1] * ggml_type_size(blocks->type);
        char * dst_data = (char *) blocks->data;
        const char * src_data = (const char *) v_cur->data;
        const int64_t * idxs_data = (const int64_t *) v_idxs->data;

        for (int64_t i = 0; i < n_tokens; i++) {
            int64_t flat  = idxs_data[i];
            int64_t bid  = flat / blk_size;
            int64_t offs = (flat % blk_size) * head_dim * n_head_kv
                         + bid * blk_size * head_dim * n_head_kv;
            memcpy(dst_data + offs, src_data + i * src_row_stride + head_offset * head_dim * ggml_type_size(blocks->type), row_bytes);
        }
    }
    return blocks;
}
// cat-first: tranche 8

//
// build_input / set_input
//

ggml_tensor * llama_kv_cache_paged_context::build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    const uint32_t n_tokens = ubatch.n_tokens;
    ggml_tensor * k_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
    ggml_set_input(k_idxs);
    return k_idxs;
}

ggml_tensor * llama_kv_cache_paged_context::build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    const uint32_t n_tokens = ubatch.n_tokens;
    ggml_tensor * v_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
    ggml_set_input(v_idxs);
    return v_idxs;
}

void llama_kv_cache_paged_context::set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    int64_t * data = (int64_t *) dst->data;
    const uint32_t n_tokens = ubatch->n_tokens;
    const int64_t blk_size = cache->block_size;
    for (uint32_t i = 0; i < n_tokens; i++) {
        llama_pos pos = ubatch->pos[i];
        int64_t bid = pos / (int64_t) blk_size;
        int64_t intra = pos % (int64_t) blk_size;
        data[i] = bid * blk_size + intra;
    }
}

void llama_kv_cache_paged_context::set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    set_input_k_idxs(dst, ubatch);
}

void llama_kv_cache_paged_context::set_input_k_shift(ggml_tensor * dst) const {
    GGML_UNUSED(dst);
}

void llama_kv_cache_paged_context::set_input_kq_mask(ggml_tensor * dst, const llama_ubatch & ubatch, bool causal_attn) const {
    GGML_UNUSED(dst); GGML_UNUSED(ubatch); GGML_UNUSED(causal_attn);
}

void llama_kv_cache_paged_context::set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch & ubatch) const {
    GGML_UNUSED(dst); GGML_UNUSED(ubatch);
}

void llama_kv_cache_paged_context::set_input_k_rot(ggml_tensor * dst) const { GGML_UNUSED(dst); }
void llama_kv_cache_paged_context::set_input_v_rot(ggml_tensor * dst) const { GGML_UNUSED(dst); }

void llama_kv_cache_paged::enable_kv_shard(int tp_rank, int tp_size) {
    kv_shard_rank = tp_rank;
    kv_shard_size = tp_size;
}

void llama_kv_cache_paged::disable_kv_shard() {
    kv_shard_rank = -1;
    kv_shard_size = 1;
}

bool llama_kv_cache_paged::is_sharded() const {
    return kv_shard_rank >= 0;
}

int llama_kv_cache_paged::shard_rank() const { return kv_shard_rank; }
int llama_kv_cache_paged::shard_size() const { return kv_shard_size; }
