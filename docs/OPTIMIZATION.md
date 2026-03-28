# RSR Multiplier Optimization Guide

How the RSR multipliers in this repository are optimized and why they are fast.

**Scope:** RSR implementations under `multiplier/` and `kernels/`.

---

## The Core Idea

Every RSR multiplier does this:

1. **Preprocess** the weight matrix once.
2. **Group** columns with identical `k`-row block patterns.
3. At inference, **aggregate** the input values for each group once.
4. **Scatter** that sum to the affected output rows.

Naive low-bit matvec touches every matrix entry. RSR touches every *unique column pattern per block* instead. Everything else in this document is about reducing the overhead around that algorithmic win.

## Preprocessing

For a binary block, each column becomes a `k`-bit integer. For a ternary block, each column becomes two `k`-bit integers (positive mask, negative mask), combined into a `2k`-bit ternary code. Columns with the same code share one aggregate at inference time.

The fastest preprocessing uses **counting sort** over the discrete pattern space (`2^k` buckets for binary, `4^k` for ternary), giving `O(n + buckets)` per block instead of `O(n log n)`.

## Why `k` Limits Exist

- Binary counting sort: `2^k` buckets — impractical for very large `k`
- Ternary counting sort: `4^k` buckets — grows much faster
- Bitmask-scatter variants store row membership in `uint16` → requires `k ≤ 16`
- 16-bit permutation indices require column count ≤ 65535

These are not arbitrary guardrails — they are what enable compact metadata and cheap inner loops.

---

## Binary CPU: `multiplier/bit_1`

### `RSRPythonMultiplier`

*Files: `multiplier/bit_1/rsr_py.py`*

Pure Python/PyTorch reference. Encodes columns into integers, comparison-sorts by pattern, finds unique patterns, aggregates with `scatter_add_`, and scatters via `unique_bits.T @ aggregated`. Proves the algorithm; not the fastest due to PyTorch dispatch overhead and general-purpose sorting.

### `RSRCppMultiplier`

*Files: `multiplier/bit_1/cpu/rsr_cpp.py`, `kernels/bit_1/cpu/rsr_prep.c`, `kernels/bit_1/cpu/rsr.c`*

Moves the hot path to C. Preprocessing uses counting sort (parallelized with OpenMP). Inference is a single fused kernel for gather, aggregate, and scatter. Long groups use AVX2 `_mm256_i32gather_ps` with 4× unrolling.

### `RSRCppV2_4Multiplier`

*Files: `multiplier/bit_1/cpu/rsr_cpp_v2_4.py`, `kernels/bit_1/cpu/rsr_v2_4.c`*

Adds `schedule(static)` OpenMP and pre-allocates one 64-byte-aligned `v_perm` buffer per thread. The kernel first gathers `v[perm]` into the contiguous buffer, then aggregates over contiguous slices. Trades extra buffer traffic for a cleaner aggregation phase — works well when contiguous summation outweighs the write cost.

### `RSRCppV4_2Multiplier`

*Files: `multiplier/bit_1/cpu/rsr_cpp_v4_2.py`, `kernels/bit_1/cpu/rsr_v4_2.c`*

Removes the `v_perm` buffer entirely. Gathers and aggregates in one pass directly from `v`. Uses scalar-unrolled loads (8-way switch) with `_mm_prefetch` at 64-element distance instead of AVX2 gather. Less memory traffic, less temporary storage, better cache behavior on the actual bottleneck (random reads from `v`).

**This is the key binary CPU kernel.**

### `RSRCppNonSquareMultiplier`

*Files: `multiplier/bit_1/cpu/rsr_cpp_nonsquare.py`, `kernels/bit_1/cpu/rsr_prep_nonsquare.c`*

Supports `n_rows × n_cols` matrices by padding rows to a multiple of `k`, running non-square preprocessing, and reusing the v4.2 inference kernel with `n_cols` as the permutation stride. No second inference kernel — just metadata adaptation.

### `RSRAdaptiveMultiplier`

*Files: `multiplier/bit_1/cpu/rsr_adaptive.py`*

For square matrices not divisible by `k`, pads to the next multiple and delegates to `RSRCppV4_2Multiplier`.

---

## Binary CUDA: `multiplier/bit_1/cuda`

### Shared Preprocessing

*Files: `_prep_cuda.py`, `_prep_cuda_nonsquare.py`*

Preprocessing runs on CPU (counting sort is one-time work). Metadata is rearranged into GPU-friendly tensors. Several versions sort permutation indices within each group — this does not change the sum but makes reads from `v` more spatially local, improving L2 behavior.

### `RSRCudaV4_10Multiplier`

*Files: `rsr_cuda_v4_10.py`, `kernels/bit_1/cuda/rsr_v4_10.cu`*

- 16-bit permutations
- Precomputed `group_starts`
- Sorted perms within groups
- One CUDA block per row block; warps process groups, lane 0 scatters to shared memory
- 8× unrolled gather (256 elements per iteration)
- Adaptive thread count: 128/256/512 based on `k`

### `RSRCudaV5_7Multiplier`

*Files: `rsr_cuda_v5_7.py`, `kernels/bit_1/cuda/rsr_v5_6.cu`*

Introduces **packed metadata**: each group is one `int4(start, end, row_mask, 0)`. Scatter becomes bit operations over the row mask instead of following a variable-length row array. Each warp writes into its own shared-memory partial buffer. Processes two groups per warp step. Fixed 256 threads/block.

One metadata load per group instead of multiple array reads → less global memory traffic.

### `RSRCudaV5_8Multiplier`

*Files: `rsr_cuda_v5_8.py`, `kernels/bit_1/cuda/rsr_v5_8.cu`*

Same packed-metadata design as v5.7. Uses 1024 threads when `k > 4` (256 otherwise) and builds with `--use_fast_math`. More warps per block means more group-level parallelism on larger `k`.

### `RSRCudaV5_9Multiplier`

*Files: `rsr_cuda_v5_9.py`, `kernels/bit_1/cuda/rsr_v5_9.cu`*

Keeps packed metadata and sorted perms. Stores permutations as `uint16` — at large `n`, the permutation array is the biggest metadata stream, so halving it directly lowers bandwidth pressure. Uses 256 threads for `k ≤ 4`, 512 otherwise.

**This is the main large-`n` binary CUDA kernel.**

### `RSRCudaV5_9NonSquareMultiplier`

*Files: `rsr_cuda_v5_9_nonsquare.py`*

Pads rows to a multiple of `k`, runs non-square CPU preprocessing, sorts within groups, and reuses the v5.9 kernel with `n_cols` as stride.

### `RSRCudaV5_10Multiplier`

*Files: `rsr_cuda_v5_10.py`*

Not a new kernel — an empirical dispatcher:

| Condition | Kernel |
|:---|:---|
| `k == 8` and `n ≤ 4096` | v5.7 |
| `k == 16` and `n ≤ 8192` | v5.8 |
| otherwise | v5.9 |

### `RSRCudaAdaptiveMultiplier`

*Files: `rsr_cuda_adaptive.py`*

Pads square matrices to a multiple of `k` and delegates to `RSRCudaV5_10Multiplier`.

---

## Ternary CPU: `multiplier/bit_1_58`

### What Is Different

Binary RSR only needs to know which rows receive `+agg`. Ternary RSR must track both `+agg` and `-agg` rows. The optimization story in the ternary family is reducing the cost of storing and reading this signed scatter information.

### `RSRTernaryV1_4Multiplier`

*Files: `multiplier/bit_1_58/cpu/rsr_v1_4.py`, `kernels/bit_1_58/cpu/rsr_ternary_prep.c`, `kernels/bit_1_58/cpu/rsr_ternary.c`*

Splits each block into positive and negative bit patterns, combines into a `2k`-bit ternary code (`(pos_val << k) | neg_val`), groups with counting sort over `4^k` buckets. Inference gathers, sums, then scatters with explicit `scatter_rows` and `scatter_signs` arrays. Fused C, not Python.

### `RSRTernaryV3_1Multiplier`

*Files: `multiplier/bit_1_58/cpu/rsr_v3_1.py`, `kernels/bit_1_58/cpu/rsr_ternary_v3_1.c`*

Over v1.4:
- `perms` and `group_ends` shrunk to `uint16` — halves metadata bandwidth in the hot loop
- Wrapper caches ctypes pointers (no repeated Python→ctypes setup)
- Kernel uses `schedule(static)`, small-group fast paths (switch for len ≤ 4), and `_mm_prefetch` with T0 hints

### `RSRTernaryV3_3Multiplier`

*Files: `multiplier/bit_1_58/cpu/rsr_v3_3.py`, `kernels/bit_1_58/cpu/rsr_ternary_v3_3.c`*

Replaces the variable-length signed scatter arrays with two fixed-size `uint16` masks per group: `pos_mask` and `neg_mask`. The kernel iterates set bits with `__builtin_ctz` and `mask &= mask - 1`. Requires `k ≤ 16`.

This is **the key ternary CPU optimization**: metadata per group stops depending on how many rows are active. Two compact masks replace a variable-length scatter list.

### `RSRTernaryNonSquareMultiplier`

*Files: `multiplier/bit_1_58/cpu/rsr_nonsquare.py`, `kernels/bit_1_58/cpu/rsr_ternary_prep_nonsquare.c`*

Pads rows to a multiple of `k`. Dispatches:

| Condition | Kernel |
|:---|:---|
| `n_cols ≥ 4096` and `k ≤ 16` | v3.3 |
| otherwise | v3.1 |

v3.3 wins when metadata bandwidth matters enough to justify mask creation; v3.1 is lighter for smaller shapes.

### CPU Runtime Variants for Model Inference

*Files: `multiplier/bit_1_58/cpu/rsr_runtime.py`, `kernels/bit_1_58/cpu/rsr_ternary_v3_1_batch.c`, `kernels/bit_1_58/cpu/rsr_ternary_v3_3_batch.c`*

These explain the large end-to-end CPU inference speedups.

**`RSRPreprocessedMultiplier`** — loads saved RSR tensors from safetensors, skipping preprocessing at serve time. Dispatches to v3.1 or v3.3 based on the same rules.

**`fused_call`** — fuses BitNet activation quantization and RSR GEMV into a single C call. Since the kernel already touches the full input vector, quantizing it in the same call avoids extra Python dispatch and temporary handling.

**`RSRBatchMultiplier` / `RSRBatchMultiplierV31`** — batch multiple layers sharing the same input vector (e.g. `q_proj + k_proj + v_proj`, or `gate_proj + up_proj`). Quantize the input once, execute all GEMVs in one C call, and parallelize across the combined block pool with OpenMP.

This batching and fusion layer is a major reason the CPU LLM path is much faster than calling one optimized GEMV per layer:
- One quantization instead of many
- One native call instead of many
- More total blocks for OpenMP to distribute across cores

---

## Ternary CUDA: `multiplier/bit_1_58/cuda`

### Shared Preprocessing

*Files: `_prep_cuda.py`, `_prep_cuda_nonsquare.py`, `_prep_v2_common.py`*

Preprocessing runs on CPU. Then:
1. Sort permutations inside each group
2. Convert signed scatter lists into `pos_mask` / `neg_mask`
3. Drop all-zero groups (they never change the output)
4. Pack each remaining group into one `uint64`:

```
bits [0:15]   start
bits [16:31]  length
bits [32:47]  pos_mask
bits [48:63]  neg_mask
```

One 64-bit word fully describes a group. The kernel reads one value and starts working immediately. Skipping zero groups removes useless warps and gathers.

### `RSRTernaryCudaV2_0Multiplier`

*Files: `rsr_cuda_v2_0.py`, `kernels/bit_1_58/cuda/rsr_ternary_v2_0.cu`*

- One CUDA block per row block
- One warp processes one group: reduces the group sum, then scatters into per-warp shared-memory partials using the positive/negative masks
- Partials reduced across warps at block end
- Zero groups removed during preprocessing
- 16-bit permutations
- Compile-time `k` specializations for 2, 4, 6, 8, 10, 12 — lets the compiler unroll the scatter loop
- Thread count: 256 for `k ≤ 4`, 512 otherwise

**This is the retained ternary CUDA kernel.**

### `RSRPreprocessedCudaMultiplier`

*Files: `multiplier/bit_1_58/cuda/rsr_runtime.py`*

Loads saved CUDA RSR tensors and reuses the v2.0 kernel at inference time. Unlike the CPU path, the current CUDA runtime does not yet fuse activation quantization across sibling layers — the speedup comes from the compact v2.0 kernel itself.

---

## Currently Active Paths

| Use case | Entry point | Inner kernel |
|:---|:---|:---|
| Binary CPU | `RSRCppNonSquareMultiplier` | v4.2 |
| Binary CUDA | `RSRCudaV5_9NonSquareMultiplier` | v5.10 adaptive dispatcher |
| Ternary CPU | `RSRTernaryNonSquareMultiplier` | v3.1 or v3.3 |
| Ternary CPU (model runtime) | `RSRPreprocessedMultiplier` | fused batch v3.1/v3.3 |
| Ternary CUDA | `RSRTernaryCudaV2_0Multiplier` | v2.0 |

---

## Optimization Progression

The speed comes from this sequence:

1. **Change the algorithm.** Aggregate repeated column patterns once instead of repeating work per column.
2. **Match preprocessing to the discrete problem.** Counting sort over a small known pattern space beats general-purpose sorting.
3. **Shrink the hot metadata.** `uint16` perms, `uint16` masks, packed 64-bit words, dropped zero groups — all cut bandwidth.
4. **Improve gather locality.** Sorting indices within a group does not change the sum but makes the memory system happier.
5. **Remove unnecessary movement.** Direct-gather designs delete the extra write/read of intermediate buffers.
6. **Keep updates local.** Shared-memory or per-warp partials avoid fighting over global output writes.
7. **Fuse surrounding work.** Skipping repeated activation quantization and repeated Python calls matters almost as much as the GEMV kernel itself.
