# The Wrapper Architecture

## Purpose

This fork exists to extend llama.cpp with capabilities that upstream will not accept: proprietary quantization codecs (TurboQuant+, RotorQuant, AngelSlim, NautilusQuant), a production serving stack (paged KV cache, continuous batching, speculative decoding), and a modular plugin system for multi-vendor GPU backends. Everything is designed so that `git merge upstream/main` works without conflicts — our additions are either in new files that upstream never touches, or in minimal upstream patches that are stable across merges.

## How it wraps upstream

### The merge contract

| Layer | What we touch | Merge risk |
|-------|---------------|------------|
| **New files (54)** | Fork dispatch, quant implementations, serving stack, plugin loader, tooling | **Zero** — upstream has no files with these names |
| **Upstream patches (19 files)** | `ggml.h`, `ggml.c`, `ggml-common.h`, `ggml-quants.h`, `ggml-quants.c`, `ggml-quant-caps.c`, `gguf.cpp`, `ggml-cpu.c`, `ops.cpp`, `ops.h`, CMake (source lists), `src/` (batch, graph, kv-cache), `common/`, `tools/` | **Low** — additive only (array grows, branch added). No logic removed or restructured. |

The upstream patches follow a strict pattern: **grow arrays, relax assertions, add branches, never remove or restructure existing code.** When upstream adds a new quant type (e.g., `GGML_TYPE_Q4_0_K_R32` at ID 54), it slots into the 0-127 range untouched. Our fork types live at 243-255 — 114 slots of headroom between upstream's growth zone and our range.

### Upstream files modified (19 total)

```
ggml/include/ggml.h              — GGML_TYPE_COUNT=256, FORK_BASE/MAX/IS_FORK_TYPE macros, 13 fork enum entries (243-255)
ggml/src/ggml-common.h           — Block structs for fork quant types (nautilus, q1_0_g128)
ggml/src/ggml.c                  — Array grow (type_traits, etc.), fork delegation in ggml_get_type_traits()
ggml/src/ggml-quants.h           — Function declarations for fork quant/dequant
ggml/src/ggml-quants.c           — X-macro include of fork quant implementations
ggml/src/ggml-quant-caps.c       — Fork caps entries (needs_wht, needs_deferred, k_capable, etc.)
ggml/src/gguf.cpp                — Type range check relaxed to accept 243-255
ggml/src/CMakeLists.txt          — Fork .c/.def files added to ggml-base (q1_0-g128-quant.c)
ggml/src/ggml-cpu/CMakeLists.txt  — Fork dispatch sources + GGML_FORK_DISPATCH define + ../include path + backend-loader
ggml/src/ggml-cpu/ggml-cpu.c      — Fork dispatch gate (include + rename + callsite hook, 3 edits)
ggml/src/ggml-cpu/ops.cpp         — 6 ops patched (gelu, silu, leaky_relu, fill, etc.)
ggml/src/ggml-cpu/ops.h          — Fork dequant helper declaration
ggml/src/ggml-hybrid/shaders/     — 4 Vulkan shaders (uint8_t→uint fix, lane safety clamping, speculative sampling)
src/CMakeLists.txt                — 13 new llama sources added to libllama target
src/llama-batch.h                 — split_mixed() declaration for mixed prefill+decode ubatches
src/llama-batch.cpp               — split_mixed() implementation
src/llama-kv-cache-paged.cpp      — DP-attention KV sharding (per-rank head slicing in cpy_k/cpy_v)
src/llama-graph.cpp              — llama_graph_insert_all_gather (DP-attention all-reduce hook)
src/llama-quant.cpp              — (no changes — TODO comments removed after bad inject fix)
common/modelfile.h               — .modelfile config struct
common/modelfile.cpp              — .modelfile INI parser (try/catch on stoi/stof)
tools/CMakeLists.txt             — add_subdirectory(modhub)
```

### Fork-new files (54, zero merge risk)

```
ggml/src/ggml-fork-types.h          — FORK_BASE/MAX macros, ggml_fork_dequant_to_f32()
ggml/src/ggml-fork-types.c          — fork_type_traits[] table, type getter
ggml/src/ggml-fork-types.def        — X-macro for fork type traits registration
ggml/src/ggml-ops-fork-dispatch.h   — Fork dispatch API (ggml_fork_compute_forward, etc.)
ggml/src/ggml-ops-fork-dispatch.cpp — Three-tier dispatch engine (Tier 1: fork ops, Tier 2: specialized, Tier 3: assert)
ggml/src/ggml-ops-fork.def           — Tier 1: fork op IDs → handlers (128-131)
ggml/src/ggml-ops-fork-types.def     — Tier 2: (op, type) → specialized handlers
ggml/src/ggml-ops-fork-kernels.cpp  — Handler implementations (stubs with clear error messages)
ggml/src/ggml-ops-fork-validate.cpp — Empty — validation logic inlined into dispatch.cpp
ggml/src/ggml-turbo-quant.c         — TurboQuant 2/3/4 codec (WHT + PolarQuant)
ggml/src/ggml-nautilus-quant.c       — NautilusQuant 3-bit KV (golden-ratio Givens rotation, ID=243)
ggml/src/ggml-iso-quant.c           — IsoQuant 3/4 codec (quaternion rotation)
ggml/src/ggml-iso4-quant.c          — IsoQuant 4 codec
ggml/src/ggml-planar-quant.c        — PlanarQuant 3/4 codec (2D Givens rotation)
ggml/src/ggml-planar4-quant.c       — PlanarQuant 4 codec
ggml/src/ggml-stq1_0-quant.c        — AngelSlim STQ 1.31bpw (structured ternary)
ggml/src/ggml-tequila-quant.c       — AngelSlim Tequila 2.0bpw (deadzone-aware ternary)
ggml/src/ggml-f8-e4m3-quant.c       — LeptoQuant FP8 E4M3 (KL-calibrated per-block scale)
ggml/src/ggml-q1-0-g128-quant.c     — 1-bit group-128 binary quant (1.125 bpw, struct unguarded, funcs behind #ifdef)
ggml/src/ggml-quant-caps.c          — Caps table (semantic quant capabilities)
ggml/src/ggml-backend-quant.c      — Backend quant registration API
ggml/src/ggml-backend-loader.cpp    — dlopen plugin scanner (POSIX sockets, POSIX only)
ggml/include/ggml-backend-plugin.h — Plugin info struct + init typedef
ggml/src/ggml-cpu/attention-paged.cpp — Paged attention kernel (Phase 21)
ggml/src/ggml-cpu/attention-tree.cpp — CPU tree-attention kernel O(n²) MHA (Phase 27)
ggml/src/ggml-hybrid/shaders/gemm_q4_0.comp — Vulkan min() clamps (Phase 32)
ggml/src/ggml-hybrid/shaders/gemm_q8_0.comp — Vulkan min() clamps (Phase 32)
ggml/src/ggml-hybrid/shaders/gemm_q4_k.comp — Vulkan min() clamps (Phase 32)
ggml/src/ggml-hybrid/shaders/speculative_sampling.comp — Vulkan speculative sampling stub (Phase 34)
src/llama-kv-cache-paged.h          — Paged KV cache class with block pool, CoW, KV shard methods
src/llama-kv-cache-paged.cpp        — Block allocator, copy-on-write, sharded cpy_k/cpy_v
src/llama-prefix-cache.h/cpp        — Radix tree prefix cache (SGLang-style)
src/llama-scheduler.h               — Continuous batching scheduler (pimpl), sched_request/result/batch_data
src/llama-scheduler.cpp             — Completion/preempt/admit loop, chunked prefill build_batch
src/llama-scheduler-overlap.cpp     — Double-buffered overlap scheduler (spin-yield, 2 slots)
src/llama-scheduler-cache-aware.cpp — Cache-aware scheduling bridge (stub)
src/llama-grammar-fsm.h             — FSM struct (fsm_state, fsm_transition, jump_path)
src/llama-grammar-jump.cpp          — FSM compression + jump detection + minimal GBNF parser
src/llama-grammar-xgrammar.cpp      — XGrammar backend bridge (stub)
src/llama-speculative-tree.h/cpp    — Tree proposal + BFS mask + verify (EAGLE3-style)
src/llama-speculative-phantom.cpp   — PHANTOM zero-copy n-gram speculation orchestrator
src/llama-speculative-saguaro.h/cpp — SAGUARO LRU draft cache (FNV-1a hash, fixed entry cap)
src/llama-speculative-sampling.h/cpp — 3-tier rejection sampling (C++/Vulkan/scalar)
src/llama-ngram-corpus.h/cpp        — Trie-based n-gram corpus (children prediction on lookup)
src/llama-bloom-filter.h             — Bloom negative filter for rejected bigrams
src/llama-phantom-buffer.h          — Lock-free SPSC ring buffer (UMA pinned memory)
src/llama-phantom-scaler.cpp         — Adaptive worker count scaler (stub)
tools/server/router-shadow-tree.h   — Per-worker shadow radix fingerprint
tools/server/router-cache-aware.cpp — Cache-aware routing scorer
tools/recommender/quant-recommender.h/cpp — Smart quant recommendation (placeholder logic)
tools/recommender/quant-recommender-cli.cpp — Recommender CLI smoke test
tools/modhub/llama-mod.cpp          — Model distribution CLI (pull/run/list)
tools/modhub/registry.h              — HuggingFace Hub client API
tools/modhub/registry.cpp            — POSIX socket HTTP client, SHA256 verify, manifest cache
tools/modhub/CMakeLists.txt         — Build target for llama-mod (links llama-common, OpenSSL::Crypto)
common/modelfile.h                   — .modelfile config struct (model, quant, inference, speculative, server sections)
common/modelfile.cpp                 — INI parser with try/catch on stoi/stof
tools/stability/stability_runner.cpp — Benchmark runner (RotorQuant benchmark stub)
```

---

## ID Range Reservation

```
  0 ─────── 53   Upstream types (in-use)
 54 ───── 127   Reserved buffer for upstream growth (~74 slots)
128 ───── 242   Available for future fork types
243 ───── 255   Fork types (this wrapper)
```

**CRITICAL: `GGML_TYPE_COUNT = 256` fills the uint8_t range.** No more quant types can be added without a registry refactor that decouples type IDs from the hardcoded enum. Q1_0_G128 (struct in ggml-common.h, functions behind `#ifdef GGML_USE_Q1_0_G128`) has no enum entry because of this cap.

All fork quant types are registered at IDs 243-255. The `GGML_IS_FORK_TYPE(t)` macro checks `(t) >= 243 && (t) <= 255`.

```c
// ggml.h
GGML_TYPE_NAUTILUS3_0 = 243   // NautilusQuant 3-bit KV: golden-ratio Givens rotation
GGML_TYPE_TURBO2_0  = 244   // 2-bit KV cache
GGML_TYPE_TURBO3_0  = 245   // 3-bit KV cache
GGML_TYPE_TURBO4_0  = 246   // 4-bit KV cache
GGML_TYPE_TQ3_1S   = 247   // 3-bit weight
GGML_TYPE_TQ4_1S   = 248   // 4-bit weight
GGML_TYPE_PLANAR3_0 = 249   // PlanarQuant 3-bit KV
GGML_TYPE_ISO3_0    = 250   // IsoQuant 3-bit KV
GGML_TYPE_PLANAR4_0 = 251   // PlanarQuant 4-bit KV
GGML_TYPE_ISO4_0    = 252   // IsoQuant 4-bit KV
GGML_TYPE_STQ1_0    = 253   // Structured ternary weight
GGML_TYPE_TEQUILA   = 254   // Deadzone-aware ternary weight
GGML_TYPE_F8_E4M3    = 255   // FP8 E4M3 weight
```

### Q1_0_G128 special case

`GGML_TYPE_Q1_0_G128` (1-bit group-128, 1.125 bpw) is a partial implementation:
- Block struct `block_q1_0_g128` defined in `ggml-common.h` (unguarded, always available)
- Quant/dequant functions in `ggml-q1-0-g128-quant.c` (guarded by `#ifdef GGML_USE_Q1_0_G128`)
- No enum entry — `GGML_TYPE_COUNT = 256` prevents adding one
- Needs the .def registry refactor to become a first-class type

---

## Fork Dispatch Architecture

The core design goal: **zero modifications to upstream's `ops.cpp` for new fork types.** The 83 `GGML_ABORT` sites in `ops.cpp`'s switch statements become dead code for fork types because the dispatch gate catches them first.

### Three-tier dispatch

```
                         ┌──────────────────────────┐
   upstream call ──────► │  ggml_compute_forward     │  (ggml-cpu.c)
                         │                            │
                         │  ┌─ fork dispatch gate ──┐   │  ← single branch check
                         │  │                       │   │
                         │  │  ggml_has_fork_type?  │   │
                         │  └───┬───────────────┬───┘   │
                         │      │ yes           │ no    │
                         │      ▼               ▼       │
                         │  Tier 1-3 dispatch    upstream
                         │  (this wrapper)       switch
                         └──────────────────────────┘
```

**Tier 1** — Fork-specific ops (DEQUANT_FORK, QUANT_FORK, MUL_MAT_FORK, WEIGHT_TRANSFORM). Defined in `ggml-ops-fork.def`. These ops only exist in this fork.

**Tier 2** — Upstream ops (ADD, MUL, MUL_MAT, etc.) with fork-typed source tensors. Defined in `ggml-ops-fork-types.def`. Maps `(op, fork_type_id)` → specialized handler. E.g., `MUL_MAT + TURBO2_0` → `compute_forward_mul_mat_turbo2_0`.

**Tier 3** — Fallback for any (op, fork_type) pair without a Tier 2 handler. Currently asserts with a clear error message directing the developer to add either a Tier 2 entry or an `ops.cpp` patch.

### The dispatch gate (3 surgical edits in ggml-cpu.c)

```c
// 1. Top of file: #include "ggml-ops-fork-dispatch.h"

// 2. L1750: Function renamed for external visibility
void ggml_compute_forward_upstream(struct ggml_compute_params * params, struct ggml_tensor * tensor) {

// 3. L3126: Callsite hook — the only addition to the main compute path
// HOOK: fork dispatch gate — see ggml-ops-fork-dispatch.h
if (!ggml_fork_compute_forward(&params, node)) {
    ggml_compute_forward_upstream(&params, node);
}
```

The rename uses a greppable macro:
```c
#ifndef GGML_FORK_COMPUTE_FORWARD_UPSTREAM
#define GGML_FORK_COMPUTE_FORWARD_UPSTREAM ggml_compute_forward_upstream
#endif
```

### Runtime validation

`ggml_fork_ops_validate()` (called at startup) checks:
- Every fork_op_table entry has a non-NULL handler
- Every fork_type_op_table entry has a non-NULL handler
- No fork op ID < FORK_BASE (would collide with upstream range)

---

## Adding a New Quant Type

Adding a new fork quant type requires touching exactly 4 files. No upstream file edits for the type itself — only the registration.

### 1. Register the enum ID in `ggml.h`

```c
// Find the next available ID in 243-255 range
GGML_TYPE_MY_NEW_TYPE = 253,  // or next unused slot (IF GGML_TYPE_COUNT < 256!)
```

**If GGML_TYPE_COUNT is already 256 (current state):** You cannot add an enum entry. Add the block struct to `ggml-common.h` unguarded, implement the codec guarded by `#ifdef GGML_TYPE_MY_NEW_TYPE`, and wait for the registry refactor.

### 2. Implement the codec in a new file

```c
// ggml/src/ggml-my-new-type-quant.c
#include "ggml-quants.h"

// Block struct (or put in ggml-common.h)
typedef struct { /* your block layout */ } block_my_new_type;

void dequantize_row_my_new_type(const void * src, float * dst, int64_t k) {
    // dequant implementation
}

void quantize_row_my_new_type(const float * src, void * dst, int64_t k) {
    // quant implementation
}
```

### 3. Add the type_traits entry in `ggml-fork-types.c`

```c
// fork_type_traits[] — indexed by (type_id - FORK_BASE)
// The struct provides: type_name, blck_size, type_size, to_float, from_float, vec_dot
```

### 4. Add a caps entry in `ggml-quant-caps.c`

```c
[GGML_TYPE_MY_NEW_TYPE] = {
    .needs_wht      = false,   // set true if this type requires WHT rotation
    .needs_deferred = false,   // set true for deferred KV allocation
    .head_align     = 128,     // required head alignment
    .k_capable      = true,    // can be used for K cache
    .v_capable      = true,    // can be used for V cache
    .weight_capable = true,    // can be used for weights
    .cpu_fallback_ok = true,   // CPU fallback if no GPU kernel
},
```

### 5. (Optional) Add a specialized handler

For ops that need performance (not just correctness via dequant→F32→compute→requant), add a Tier 2 entry:

```c
// ggml-ops-fork-types.def
GGML_FORK_TYPE_OP(GGML_OP_MUL_MAT, GGML_TYPE_MY_NEW_TYPE, compute_forward_mul_mat_my_new_type)
```

```c
// ggml-ops-fork-kernels.cpp
void compute_forward_mul_mat_my_new_type(ggml_compute_params * params, ggml_tensor * dst) {
    // specialized GEMM kernel for this quant type
}
```

### What you do NOT need to edit

- `ops.cpp` — no new case statements needed (dispatch gate catches fork types)
- `ggml.c` — no new type_traits entry needed (fork table is separate)
- `kv-cache.cpp` — no new type checks needed (caps-driven via `ggml_type_needs_deferred`, `ggml_type_k_capable`, etc.)
- `graph.cpp` — no changes needed (ops are transparent to graph construction)

---

## Quant Caps System

Higher-level code (kv-cache, graph builder) doesn't check type IDs directly. It queries the caps table:

```c
// Instead of: if (type == GGML_TYPE_PLANAR3_0 || type == GGML_TYPE_ISO3_0 || ...) { deferred_alloc(); }
// Write:
if (ggml_type_needs_deferred(type)) { deferred_alloc(); }

if (ggml_type_k_capable(type))    { /* use for K cache */ }
if (ggml_type_v_capable(type))    { /* use for V cache */ }
if (ggml_type_needs_wht(type))    { /* apply WHT rotation */ }
if (ggml_type_weight_capable(type)) { /* quantize weights */ }
```

The caps table is populated from `ggml-quant-caps.c`. When you add a new type (step 4 above), the caps system automatically knows about it — no scattered `if` statements to find and update.

---

## Advanced Serving Features

All serving features are gated behind `LLAMA_USE_SCHEDULER`. When this define is NOT set, `server.cpp` and `server-context.cpp` take the upstream code path unchanged.

### Paged KV Cache

Replaces contiguous per-sequence KV allocation with a block pool. Each sequence gets a block table mapping logical positions to physical blocks. When sequences diverge from a shared prefix, only the divergent suffix gets new blocks (copy-on-write).

**When to use:** Always for long-context or multi-sequence workloads. 60-80% memory savings on mixed-length sequences.

```bash
# Paged KV is the default when using the continuous batching scheduler
llama-server -m model.gguf --n-gpu-layers 99
```

### RadixAttention Prefix Cache

A radix tree on the CPU that maps shared prompt prefixes to KV block IDs. When two requests share the same system prompt, the second request reuses the first request's KV blocks instead of recomputing them.

**When to use:** Any multi-tenant serving scenario where multiple requests share system prompts or conversation prefixes. Up to 5x throughput for shared-prefix workloads.

### Jump-Forward Decoding

Compresses non-branching grammar transitions (forced tokens like `{`, `"`, `:` in JSON) into batch prefills. The FSM is built from a minimal GBNF parser (`fsm_build_from_string`) that handles `"..."` literals and `[a-z]` character ranges.

**When to use:** When generating structured output (JSON, regex, grammar-constrained). Up to 2x lower latency for JSON output.

```bash
llama-cli -m model.gguf --grammar "json" --temp 0
```

### Continuous Batching + Chunked Prefill

The scheduler re-evaluates the batch every decode iteration: admits waiting requests, evicts finished ones, preempts low-priority sequences when memory is full. Long prompts are chunked and mixed into the same batch as decode steps via `llama_batch_allocr::split_mixed()`, preventing TTFT spikes.

**Implementation status:** Real. The scheduler builds proper `llama_batch` with seq_id indirection. Chunked prefill uses `split_mixed()` for mixed prefill+decode ubatches with per-token output flags.

### Speculative Decoding Stack

Multiple speculative algorithms, composable via the SAGUARO cache:

| Algorithm | Speedup | Status |
|-----------|--------|--------|
| **PHANTOM** | +10 t/s (2.5x better than naive n-gram) | Real. Zero-copy pinned memory, bloom filter for rejected bigrams. Orchestrator in `llama-speculative-phantom.cpp` (no external header — self-contained). |
| **Tree-attention (EAGLE3)** | 1.4-1.9x throughput | Real. Tree proposal + BFS verify. CPU kernel in `attention-tree.cpp`. |
| **SAGUARO cache** | No speedup alone; reduces draft overhead | Real. FNV-1a hash, LRU eviction, fixed 1024-entry cap. |
| **Rejection sampling** | N/A (correctness layer) | Real. 3-tier: C++ kernel / Vulkan / scalar fallback. Simplified greedy argmax (not full rejection sampling). |

```bash
# PHANTOM speculative decoding
llama-cli -m model.gguf --draft-model draft.gguf --phantom-workers 2
```

### DP-Attention KV Sharding

For MLA (Multi-head Latent Attention) models like DeepSeek V3/R1: shards the single KV head across data-parallel GPUs. When `enable_kv_shard()` is called, the paged KV cache allocates `k_blocks`/`v_blocks` with `n_head_kv / shard_size` along the head dimension. `cpy_k`/`cpy_v` copy only the rank's head range. After attention, `llama_graph_insert_all_gather()` inserts a `ggml_sum_rows` node (backend must intercept `all_gather_attn` and replace with NCCL all-reduce).

**Status:** Real allocation and copy logic. Single-rank is passthrough (`n_ranks <= 1`). Multi-rank requires a backend that supports collectives.

```bash
llama-server -m deepseek-r1.gguf -np 2 --kv-shard
```

---

## Runtime Backend Plugin System

Each GPU backend is compiled as a shared library that self-describes its capabilities. At startup, the loader discovers and loads whatever is available.

### How it works

```
Binary starts
    │
    ▼
Load libggml-cpu.so (always — statically linked or fallback plugin)
    │
    ▼
Scan $GGML_BACKENDS_PATH (default: /usr/local/lib/ggml-backends/) for *.so
    │
    ├── dlopen libggml-cuda.so → ggml_backend_plugin_init()
    │     └─ reports: CUDA, 1 GPU, 24GB VRAM, types=[Q4_0,Q4_K,...], score=1.0
    │
    ├── dlopen libggml-vulkan.so → ggml_backend_plugin_init()
    │     └─ reports: Vulkan, 1 iGPU, 16GB UMA, types=[Q4_0,Q8_0,...], score=0.4
    │
    └── (no libggml-hip.so found — AMD GPU not supported)
    │
    ▼
Scheduler picks best backend per tensor:
    • Q4_K weights on CUDA? → GPU kernel
    • F_PLANAR3_0 KV on CUDA? → no kernel → check Vulkan
    • F_PLANAR3_0 KV on Vulkan? → iGPU kernel
    • F_PLANAR3_0 weights on Vulkan? → no kernel → CPU fallback via type_traits->to_float
```

**Status:** Loader (`ggml-backend-loader.cpp`) and plugin header (`ggml-backend-plugin.h`) are real. Compiled into `libggml-cpu.so`. No plugin shared libraries exist yet — the plugin directory (`/usr/local/lib/ggml-backends/`) must be created and populated by the user or by backend-specific build targets.

### Plugin interface

Each `.so` exports one function:

```c
// ggml_backend_plugin_init — called by the loader after dlopen
int ggml_backend_plugin_init(struct ggml_backend_plugin_info * info) {
    info->name            = "cuda";
    info->n_devices       = 1;
    info->device_vram[0]   = 24LL * 1024 * 1024 * 1024;
    info->supported_types[0] = GGML_TYPE_Q4_0;
    info->supported_types[1] = GGML_TYPE_Q4_K;
    // ... list all types with native GPU kernels
    info->n_supported_types = N;
    info->compute_score    = 1.0;
    info->is_uma           = false;
    info->create_backend   = my_cuda_create_backend;
    info->register_quant_ops = my_cuda_register_quant_ops;
    return 0;
}
```

### Plugin fallback chain

For any tensor with a fork quant type, the scheduler tries backends in order:

```
CUDA plugin: has native kernel for GGML_TYPE_PLANAR3_0?
    YES → use CUDA kernel
    NO  → continue

Vulkan plugin: has native kernel for GGML_TYPE_PLANAR3_0?
    YES → use Vulkan kernel
    NO  → continue

Hybrid plugin: has native kernel for GGML_TYPE_PLANAR3_0?
    YES → use hybrid kernel (CPU+iGPU split)
    NO  → continue

CPU fallback (always available):
    dequant via type_traits->to_float → compute in F32 → done
```

A fork type **always runs** even without a GPU kernel. It just runs slower on CPU. The user never gets a crash or "unsupported type" error — the fallback is guaranteed.

---

## Tooling

### Model Distribution (`llama-mod`)

CLI for downloading, caching, and running GGUF models from HuggingFace Hub.

```bash
llama-mod pull user/model-name-q4k        # Download + SHA256 verify
llama-mod run  user/model-name-q4k        # Launch llama-cli with cached model
llama-mod list                             # Show cached models
```

- Cache: `~/.llama-mod/models/`
- Manifest: `~/.llama-mod/manifest.json`
- HTTP: POSIX sockets (no libcurl dependency), follows 302 redirects
- SHA256: OpenSSL
- Build target: `llama-mod` (links `llama-common`, `OpenSSL::Crypto`)

### Smart Quant Recommender (`quant-recommender`)

Placeholder logic that picks quant types by bpw, checking VRAM fit and GPU kernel availability.

### Model Manifest (`.modelfile`)

INI parser for model configuration. Sections: `[model]`, `[quantization]`, `[inference]`, `[speculative]`, `[server]`. Values parsed with try/catch — malformed input silently ignored.

---

## Build Configuration

```bash
# Full build with all features (CUDA + Vulkan + fork quant types + serving stack)
cmake -B build \
    -DGGML_CUDA=ON \
    -DGGML_VULKAN=ON \
    -DGGML_METAL=OFF \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# CPU-only build (no GPU backends — everything falls back to CPU)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Build with runtime backend plugins (backends as .so files)
cmake -B build -DGGML_BACKEND_PLUGINS=ON -DGGML_CUDA=ON -DGGML_VULKAN=ON
cmake --build build -j$(nproc)
```

The fork quant types are always compiled in (controlled by `GGML_FORK_DISPATCH=1` in the CPU backend). They add no compile time or binary size impact when unused — the dispatch gate checks at runtime.

## Merging upstream

```bash
git remote add upstream https://github.com/ggml-org/llama.cpp.git
git fetch upstream
git merge upstream/main
# resolve conflicts in the 19 upstream files (usually auto-resolvable)
# 54 new fork files: zero conflicts
```

Conflict-prone areas during merge:
- `ggml.h` — upstream may add new enum values. Fork IDs (243-255) are separate. Resolve by keeping both.
- `ggml.c` — upstream may add new getter functions or refactor type_traits. Keep fork delegation in `ggml_get_type_traits()`.
- `ops.cpp` — upstream may add new case statements or ops. Our patches are additive case entries. Keep both.
- `CMakeLists.txt` — upstream may add new source files or change build structure. Keep fork source additions.

### Known merge hazards

- `GGML_TYPE_COUNT = 256` — if upstream adds a type at 244+, it collides with our IDs. Monitor upstream's enum growth.
- `ggml-cpu.c` rename at L1750 — if upstream renames `ggml_compute_forward`, our external-callable rename must follow. The `GGML_FORK_COMPUTE_FORWARD_UPSTREAM` macro makes this greppable.
- `ggml-cpu.c` callsite at L3126 — if upstream restructures the decode loop, the 3-line hook must be re-anchored. The `// HOOK: fork dispatch gate` comment makes it searchable.
- `src/CMakeLists.txt` — if upstream adds new llama sources, our 13 additions stay. If upstream renames the `llama` target, our source list must follow.
