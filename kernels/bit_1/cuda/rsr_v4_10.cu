/*
 * RSR CUDA kernel v4.10 — uint16 perms + 8x unroll.
 *
 * Building on v4.8 (int16 perms + adaptive threads + sorted perms):
 *   - uint16_t perms instead of int16_t → supports n up to 65535.
 *   - 8x unroll (256 elements per outer iteration) for better ILP on large groups.
 *   - Adaptive thread count passed from Python.
 */

#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void rsr_gemv_v4_10_kernel(
    const int16_t* __restrict__ perms,
    const int32_t*  __restrict__ group_starts,
    const int32_t*  __restrict__ group_ends,
    const int32_t*  __restrict__ scatter_offsets,
    const uint8_t*  __restrict__ scatter_rows,
    const int32_t*  __restrict__ block_meta,
    const float*    __restrict__ v,
    float*          __restrict__ out,
    int n, int k
) {
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int num_warps = blockDim.x >> 5;

    const int16_t* perm = perms + (int64_t)b * n;
    const int g_off = __ldg(&block_meta[b * 2]);
    const int n_groups = __ldg(&block_meta[b * 2 + 1]);
    float* bout = out + (int64_t)b * k;

    extern __shared__ float smem[];

    for (int r = tid; r < k; r += blockDim.x) {
        smem[r] = 0.0f;
    }
    __syncthreads();

    for (int g = warp_id; g < n_groups; g += num_warps) {
        const int gg = g_off + g;
        const int start = __ldg(&group_starts[gg]);
        const int end = __ldg(&group_ends[gg]);
        const int len = end - start;

        float local_sum = 0.0f;

        int i = lane;
        // 8x unroll: process 256 elements per outer iteration
        for (; i + 224 < len; i += 256) {
            float s = 0.0f;
            #pragma unroll 8
            for (int u = 0; u < 8; u++) {
                s += __ldg(&v[(int)__ldg(&perm[start + i + u * 32])]);
            }
            local_sum += s;
        }
        // 4x unroll remainder
        for (; i + 96 < len; i += 128) {
            float s = 0.0f;
            #pragma unroll 4
            for (int u = 0; u < 4; u++) {
                s += __ldg(&v[(int)__ldg(&perm[start + i + u * 32])]);
            }
            local_sum += s;
        }
        for (; i < len; i += 32) {
            local_sum += __ldg(&v[(int)__ldg(&perm[start + i])]);
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        }

        if (lane == 0) {
            const int s_begin = __ldg(&scatter_offsets[gg]);
            const int s_end = __ldg(&scatter_offsets[gg + 1]);
            float agg = local_sum;
            for (int s = s_begin; s < s_end; s++) {
                atomicAdd(&smem[(int)__ldg(&scatter_rows[s])], agg);
            }
        }
    }

    __syncthreads();

    for (int r = tid; r < k; r += blockDim.x) {
        bout[r] = smem[r];
    }
}


torch::Tensor compute_group_starts(
    torch::Tensor group_ends,
    torch::Tensor block_meta,
    int num_blocks
) {
    auto starts = torch::empty_like(group_ends);
    auto ge = group_ends.data_ptr<int32_t>();
    auto bm = block_meta.data_ptr<int32_t>();
    auto gs = starts.data_ptr<int32_t>();

    for (int b = 0; b < num_blocks; b++) {
        int g_off = bm[b * 2];
        int n_groups = bm[b * 2 + 1];
        for (int g = 0; g < n_groups; g++) {
            int gg = g_off + g;
            gs[gg] = (g == 0) ? 0 : ge[gg - 1];
        }
    }
    return starts;
}


void rsr_gemv_v4_10(
    torch::Tensor perms,
    torch::Tensor group_starts,
    torch::Tensor group_ends,
    torch::Tensor scatter_offsets,
    torch::Tensor scatter_rows,
    torch::Tensor block_meta,
    torch::Tensor v,
    torch::Tensor out,
    int n, int k, int num_blocks, int threads
) {
    const int smem_bytes = k * sizeof(float);
    rsr_gemv_v4_10_kernel<<<num_blocks, threads, smem_bytes>>>(
        perms.data_ptr<int16_t>(),
        group_starts.data_ptr<int32_t>(),
        group_ends.data_ptr<int32_t>(),
        scatter_offsets.data_ptr<int32_t>(),
        scatter_rows.data_ptr<uint8_t>(),
        block_meta.data_ptr<int32_t>(),
        v.data_ptr<float>(),
        out.data_ptr<float>(),
        n, k
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rsr_gemv_v4_10", &rsr_gemv_v4_10, "RSR GEMV CUDA v4.10");
    m.def("compute_group_starts", &compute_group_starts, "Precompute group starts");
}
