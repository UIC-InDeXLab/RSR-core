/*
 * RSR CUDA kernel v5.9
 *
 * Large-n focused changes:
 *   1. Same packed metadata path as v5.6/v5.7.
 *   2. Permutation indices stored as uint16 to reduce memory bandwidth.
 *
 * Constraint: n <= 65535.
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

__device__ __forceinline__ float warp_reduce_sum(float x) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_down_sync(0xffffffff, x, offset);
    }
    return x;
}

template <int NUM_WARPS>
__device__ __forceinline__ void process_group(
    int gg,
    int lane,
    const uint16_t* __restrict__ perm,
    const int4* __restrict__ group_packed,
    const float* __restrict__ v,
    float* __restrict__ my_partials
) {
    const int4 meta = group_packed[gg];
    const int start = meta.x;
    const int end = meta.y;
    int32_t mask = meta.z;
    const int len = end - start;

    float local_sum = 0.0f;

    int i = lane;
    for (; i + 96 < len; i += 128) {
        float s = 0.0f;
        #pragma unroll 4
        for (int u = 0; u < 4; ++u) {
            const int idx = static_cast<int>(perm[start + i + u * 32]);
            s += __ldg(&v[idx]);
        }
        local_sum += s;
    }
    for (; i < len; i += 32) {
        const int idx = static_cast<int>(perm[start + i]);
        local_sum += __ldg(&v[idx]);
    }

    local_sum = warp_reduce_sum(local_sum);

    if (lane == 0) {
        while (mask) {
            const int row = __ffs(mask) - 1;
            my_partials[row] += local_sum;
            mask &= (mask - 1);
        }
    }
}

template <int THREADS>
__global__ void rsr_gemv_v5_9_kernel(
    const uint16_t* __restrict__ perms,
    const int4* __restrict__ group_packed,
    const int32_t* __restrict__ block_meta,
    const float*   __restrict__ v,
    float*         __restrict__ out,
    int n, int k
) {
    constexpr int NUM_WARPS = THREADS / 32;

    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    const uint16_t* perm = perms + static_cast<int64_t>(b) * n;
    const int g_off = __ldg(&block_meta[b * 2]);
    const int n_groups = __ldg(&block_meta[b * 2 + 1]);
    float* bout = out + static_cast<int64_t>(b) * k;

    extern __shared__ float smem[];
    float* warp_partials = smem;  // [NUM_WARPS, k]
    float* my_partials = warp_partials + warp_id * k;

    for (int i = tid; i < NUM_WARPS * k; i += THREADS) {
        warp_partials[i] = 0.0f;
    }
    __syncthreads();

    for (int g = warp_id; g < n_groups; g += NUM_WARPS * 2) {
        process_group<NUM_WARPS>(g_off + g, lane, perm, group_packed, v, my_partials);

        const int g2 = g + NUM_WARPS;
        if (g2 < n_groups) {
            process_group<NUM_WARPS>(g_off + g2, lane, perm, group_packed, v, my_partials);
        }
    }

    __syncthreads();

    for (int r = tid; r < k; r += THREADS) {
        float acc = 0.0f;
        #pragma unroll
        for (int w = 0; w < NUM_WARPS; ++w) {
            acc += warp_partials[w * k + r];
        }
        bout[r] = acc;
    }
}

}  // namespace

torch::Tensor compute_group_starts(
    torch::Tensor group_ends,
    torch::Tensor block_meta,
    int num_blocks
) {
    TORCH_CHECK(group_ends.device().is_cpu(), "group_ends must be a CPU tensor");
    TORCH_CHECK(block_meta.device().is_cpu(), "block_meta must be a CPU tensor");
    TORCH_CHECK(group_ends.dtype() == torch::kInt32, "group_ends must be int32");
    TORCH_CHECK(block_meta.dtype() == torch::kInt32, "block_meta must be int32");
    TORCH_CHECK(group_ends.is_contiguous(), "group_ends must be contiguous");
    TORCH_CHECK(block_meta.is_contiguous(), "block_meta must be contiguous");

    auto starts = torch::empty_like(group_ends);
    const auto* ge = group_ends.data_ptr<int32_t>();
    const auto* bm = block_meta.data_ptr<int32_t>();
    auto* gs = starts.data_ptr<int32_t>();

    for (int b = 0; b < num_blocks; ++b) {
        const int g_off = bm[b * 2];
        const int n_groups = bm[b * 2 + 1];
        for (int g = 0; g < n_groups; ++g) {
            const int gg = g_off + g;
            gs[gg] = (g == 0) ? 0 : ge[gg - 1];
        }
    }
    return starts;
}

torch::Tensor compute_group_masks(
    torch::Tensor scatter_offsets,
    torch::Tensor scatter_rows
) {
    TORCH_CHECK(scatter_offsets.device().is_cpu(), "scatter_offsets must be CPU");
    TORCH_CHECK(scatter_rows.device().is_cpu(), "scatter_rows must be CPU");
    TORCH_CHECK(scatter_offsets.dtype() == torch::kInt32, "scatter_offsets must be int32");
    TORCH_CHECK(scatter_rows.dtype() == torch::kUInt8, "scatter_rows must be uint8");
    TORCH_CHECK(scatter_offsets.is_contiguous(), "scatter_offsets must be contiguous");
    TORCH_CHECK(scatter_rows.is_contiguous(), "scatter_rows must be contiguous");

    const int64_t total_groups = scatter_offsets.size(0) - 1;
    auto masks = torch::zeros({total_groups}, torch::dtype(torch::kInt32));

    const auto* offsets = scatter_offsets.data_ptr<int32_t>();
    const auto* rows = scatter_rows.data_ptr<uint8_t>();
    auto* out = masks.data_ptr<int32_t>();

    for (int64_t g = 0; g < total_groups; ++g) {
        uint32_t mask = 0;
        for (int32_t s = offsets[g]; s < offsets[g + 1]; ++s) {
            const uint8_t row = rows[s];
            TORCH_CHECK(row < 32, "k must be <= 32 for bitmask encoding");
            mask |= (1u << row);
        }
        out[g] = static_cast<int32_t>(mask);
    }

    return masks;
}

torch::Tensor pack_group_metadata(
    torch::Tensor group_starts,
    torch::Tensor group_ends,
    torch::Tensor group_masks
) {
    TORCH_CHECK(group_starts.device().is_cpu(), "group_starts must be CPU");
    TORCH_CHECK(group_ends.device().is_cpu(), "group_ends must be CPU");
    TORCH_CHECK(group_masks.device().is_cpu(), "group_masks must be CPU");
    TORCH_CHECK(group_starts.dtype() == torch::kInt32, "group_starts must be int32");
    TORCH_CHECK(group_ends.dtype() == torch::kInt32, "group_ends must be int32");
    TORCH_CHECK(group_masks.dtype() == torch::kInt32, "group_masks must be int32");
    TORCH_CHECK(group_starts.is_contiguous(), "group_starts must be contiguous");
    TORCH_CHECK(group_ends.is_contiguous(), "group_ends must be contiguous");
    TORCH_CHECK(group_masks.is_contiguous(), "group_masks must be contiguous");
    TORCH_CHECK(
        group_starts.numel() == group_ends.numel() && group_ends.numel() == group_masks.numel(),
        "group_* tensors must have matching lengths"
    );

    const int64_t total_groups = group_starts.numel();
    auto packed = torch::empty({total_groups, 4}, torch::dtype(torch::kInt32));

    const auto* gs = group_starts.data_ptr<int32_t>();
    const auto* ge = group_ends.data_ptr<int32_t>();
    const auto* gm = group_masks.data_ptr<int32_t>();
    auto* out = packed.data_ptr<int32_t>();

    for (int64_t g = 0; g < total_groups; ++g) {
        out[g * 4 + 0] = gs[g];
        out[g * 4 + 1] = ge[g];
        out[g * 4 + 2] = gm[g];
        out[g * 4 + 3] = 0;
    }

    return packed;
}

void rsr_gemv_v5_9(
    torch::Tensor perms_u16,
    torch::Tensor group_packed,
    torch::Tensor block_meta,
    torch::Tensor v,
    torch::Tensor out,
    int n, int k, int num_blocks
) {
    TORCH_CHECK(perms_u16.dtype() == torch::kUInt16, "perms_u16 must be uint16");
    TORCH_CHECK(n <= 65535, "v5.9 requires n <= 65535");
    (void)num_blocks;
    const int4* gp4 = reinterpret_cast<const int4*>(group_packed.data_ptr<int32_t>());

    if (k <= 4) {
        constexpr int threads = 256;
        const int smem_bytes = (threads / 32) * k * static_cast<int>(sizeof(float));
        rsr_gemv_v5_9_kernel<threads><<<num_blocks, threads, smem_bytes>>>(
            perms_u16.data_ptr<uint16_t>(),
            gp4,
            block_meta.data_ptr<int32_t>(),
            v.data_ptr<float>(),
            out.data_ptr<float>(),
            n, k
        );
    } else {
        constexpr int threads = 512;
        const int smem_bytes = (threads / 32) * k * static_cast<int>(sizeof(float));
        rsr_gemv_v5_9_kernel<threads><<<num_blocks, threads, smem_bytes>>>(
            perms_u16.data_ptr<uint16_t>(),
            gp4,
            block_meta.data_ptr<int32_t>(),
            v.data_ptr<float>(),
            out.data_ptr<float>(),
            n, k
        );
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rsr_gemv_v5_9", &rsr_gemv_v5_9, "RSR GEMV CUDA v5.9");
    m.def("compute_group_starts", &compute_group_starts, "Precompute group starts");
    m.def("compute_group_masks", &compute_group_masks, "Build per-group row bitmasks");
    m.def("pack_group_metadata", &pack_group_metadata, "Pack (start,end,mask) per group");
}
