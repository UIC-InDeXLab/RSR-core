/*
 * RSR Ternary CUDA kernel v2.0
 *
 * Optimizations over v1.1:
 *   - compact per-group metadata in one 64-bit word
 *   - zero-pattern groups removed during preprocessing
 *   - static-k specializations for the benchmarked k values
 *
 * Algorithm is unchanged: permute -> aggregate contiguous groups -> signed scatter.
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

template <int K>
__device__ __forceinline__ void scatter_group_sum(
    float* __restrict__ partials,
    uint32_t pos_mask,
    uint32_t neg_mask,
    float value,
    int k_runtime
) {
    if constexpr (K > 0) {
        #pragma unroll
        for (int row = 0; row < K; ++row) {
            const uint32_t bit = 1u << row;
            if (pos_mask & bit) partials[row] += value;
            if (neg_mask & bit) partials[row] -= value;
        }
    } else {
        for (int row = 0; row < k_runtime; ++row) {
            const uint32_t bit = 1u << row;
            if (pos_mask & bit) partials[row] += value;
            if (neg_mask & bit) partials[row] -= value;
        }
    }
}

template <int K>
__device__ __forceinline__ void process_group_warp(
    uint64_t meta,
    int lane,
    const uint16_t* __restrict__ perm,
    const float* __restrict__ v,
    float* __restrict__ partials,
    int k_runtime
) {
    const int start = static_cast<int>(meta & 0xffffu);
    const int len = static_cast<int>((meta >> 16) & 0xffffu);
    const uint32_t pos_mask = static_cast<uint32_t>((meta >> 32) & 0xffffu);
    const uint32_t neg_mask = static_cast<uint32_t>((meta >> 48) & 0xffffu);

    float local_sum = 0.0f;
    int i = lane;
    for (; i + 96 < len; i += 128) {
        float acc = 0.0f;
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int idx = static_cast<int>(perm[start + i + u * 32]);
            acc += __ldg(&v[idx]);
        }
        local_sum += acc;
    }
    for (; i < len; i += 32) {
        const int idx = static_cast<int>(perm[start + i]);
        local_sum += __ldg(&v[idx]);
    }

    local_sum = warp_reduce_sum(local_sum);
    if (lane == 0) {
        scatter_group_sum<K>(partials, pos_mask, neg_mask, local_sum, k_runtime);
    }
}

template <int THREADS, int K>
__global__ void rsr_ternary_gemv_v2_0_kernel(
    const uint16_t* __restrict__ perms,
    const uint64_t* __restrict__ group_packed,
    const int32_t* __restrict__ block_meta,
    const float* __restrict__ v,
    float* __restrict__ out,
    int n_cols,
    int k_runtime
) {
    constexpr int NUM_WARPS = THREADS / 32;

    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int k_eff = K > 0 ? K : k_runtime;

    const uint16_t* perm = perms + static_cast<int64_t>(b) * n_cols;
    const int g_off = __ldg(&block_meta[b * 2]);
    const int n_groups = __ldg(&block_meta[b * 2 + 1]);
    float* bout = out + static_cast<int64_t>(b) * k_runtime;

    extern __shared__ float smem[];
    float* warp_partials = smem;  /* [NUM_WARPS, k] */
    float* my_partials = warp_partials + warp_id * k_eff;

    for (int i = tid; i < NUM_WARPS * k_eff; i += THREADS) {
        warp_partials[i] = 0.0f;
    }
    __syncthreads();

    for (int g = warp_id; g < n_groups; g += NUM_WARPS) {
        process_group_warp<K>(group_packed[g_off + g], lane, perm, v, my_partials, k_runtime);
    }

    __syncthreads();

    for (int row = tid; row < k_eff; row += THREADS) {
        float acc = 0.0f;
        #pragma unroll
        for (int w = 0; w < NUM_WARPS; ++w) {
            acc += warp_partials[w * k_eff + row];
        }
        bout[row] = acc;
    }
}

template <int THREADS, int K>
void launch_typed(
    const uint16_t* perms,
    const uint64_t* group_packed,
    const int32_t* block_meta,
    const float* v,
    float* out,
    int n_cols,
    int k,
    int num_blocks
) {
    const int k_eff = K > 0 ? K : k;
    const int smem_bytes = (THREADS / 32) * k_eff * static_cast<int>(sizeof(float));
    rsr_ternary_gemv_v2_0_kernel<THREADS, K><<<num_blocks, THREADS, smem_bytes>>>(
        perms,
        group_packed,
        block_meta,
        v,
        out,
        n_cols,
        k
    );
}

template <int THREADS>
void launch_for_k(
    const uint16_t* perms,
    const uint64_t* group_packed,
    const int32_t* block_meta,
    const float* v,
    float* out,
    int n_cols,
    int k,
    int num_blocks
) {
    switch (k) {
        case 2:  launch_typed<THREADS, 2>(perms, group_packed, block_meta, v, out, n_cols, k, num_blocks); break;
        case 4:  launch_typed<THREADS, 4>(perms, group_packed, block_meta, v, out, n_cols, k, num_blocks); break;
        case 6:  launch_typed<THREADS, 6>(perms, group_packed, block_meta, v, out, n_cols, k, num_blocks); break;
        case 8:  launch_typed<THREADS, 8>(perms, group_packed, block_meta, v, out, n_cols, k, num_blocks); break;
        case 10: launch_typed<THREADS, 10>(perms, group_packed, block_meta, v, out, n_cols, k, num_blocks); break;
        case 12: launch_typed<THREADS, 12>(perms, group_packed, block_meta, v, out, n_cols, k, num_blocks); break;
        default: launch_typed<THREADS, 0>(perms, group_packed, block_meta, v, out, n_cols, k, num_blocks); break;
    }
}

}  // namespace


void rsr_ternary_gemv_v2_0(
    torch::Tensor perms_u16,
    torch::Tensor group_packed_u64,
    torch::Tensor block_meta,
    torch::Tensor v,
    torch::Tensor out,
    int n_cols,
    int k,
    int num_blocks
) {
    TORCH_CHECK(perms_u16.dtype() == torch::kUInt16, "perms must be uint16");
    TORCH_CHECK(group_packed_u64.dtype() == torch::kInt64, "group_packed must be int64");
    TORCH_CHECK(block_meta.dtype() == torch::kInt32, "block_meta must be int32");
    TORCH_CHECK(v.dtype() == torch::kFloat32, "v must be float32");
    TORCH_CHECK(out.dtype() == torch::kFloat32, "out must be float32");
    TORCH_CHECK(n_cols <= 65535, "v2.0 requires n_cols <= 65535");

    const auto* gp = reinterpret_cast<const uint64_t*>(group_packed_u64.data_ptr<int64_t>());

    if (k <= 4) {
        launch_for_k<256>(
            perms_u16.data_ptr<uint16_t>(),
            gp,
            block_meta.data_ptr<int32_t>(),
            v.data_ptr<float>(),
            out.data_ptr<float>(),
            n_cols,
            k,
            num_blocks
        );
    } else {
        launch_for_k<512>(
            perms_u16.data_ptr<uint16_t>(),
            gp,
            block_meta.data_ptr<int32_t>(),
            v.data_ptr<float>(),
            out.data_ptr<float>(),
            n_cols,
            k,
            num_blocks
        );
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rsr_ternary_gemv_v2_0", &rsr_ternary_gemv_v2_0, "RSR Ternary GEMV CUDA v2.0");
}
