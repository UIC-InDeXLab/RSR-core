/*
 * RSR ternary CUDA kernel v6.5 — v6.3 with grid-stride loop (fixed grid).
 *
 * Instead of one CUDA block per RSR block (2048 blocks for n=16384 k=8),
 * launches a tuned fixed number of CUDA blocks and processes multiple RSR
 * blocks per CUDA block via a grid-stride loop.
 *
 * Benefits:
 *   - Reduces CUDA kernel waves (e.g., from 3 to 1–2 for n=16384 k=8)
 *   - Each block reuses the INT8 input in L1 cache across RSR blocks
 *   - Eliminates partial wave overhead
 *
 * grid_size is passed at runtime (caller can tune per GPU/k/n).
 * Uses the K_per_loop=32 (double-load) optimization from v6.3.
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

__global__ void quantize_fp32_to_int8_v65(
    const float* __restrict__ v,
    int8_t*       __restrict__ v_i8,
    float*        __restrict__ inv_scale_out,
    int n
) {
    const int tid = threadIdx.x, THREADS = blockDim.x;
    float local_max = 0.f;
    for (int i = tid; i < n; i += THREADS)
        local_max = fmaxf(local_max, fabsf(v[i]));
    for (int off = 16; off > 0; off >>= 1)
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, off));
    __shared__ float warp_max[32];
    const int warp_id = tid >> 5, lane = tid & 31;
    if (lane == 0) warp_max[warp_id] = local_max;
    __syncthreads();
    if (warp_id == 0) {
        local_max = (lane < THREADS / 32) ? warp_max[lane] : 0.f;
        for (int off = 16; off > 0; off >>= 1)
            local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, off));
    }
    __shared__ float amax_s;
    if (tid == 0) { amax_s = fmaxf(local_max, 1e-5f); *inv_scale_out = amax_s / 127.f; }
    __syncthreads();
    const float scale = 127.f / amax_s;
    for (int i = tid; i < n; i += THREADS) {
        float val = roundf(v[i] * scale);
        v_i8[i] = static_cast<int8_t>(fminf(fmaxf(val, -128.f), 127.f));
    }
}

__device__ __forceinline__ int decode_byte_v65(uint8_t b) {
    int wp = (b & 3)
           | (((b >> 2) & 3) << 8)
           | (((b >> 4) & 3) << 16)
           | (((b >> 6) & 3) << 24);
    return __vsubss4(wp, 0x02020202);
}

template <int K, int K_DIM, int THREADS>
__global__ void __launch_bounds__(THREADS)
rsr_direct_v6_5_kernel(
    const uint8_t* __restrict__ w,
    const int8_t*  __restrict__ v_i8,
    const float*   __restrict__ inv_scale,
    float*         __restrict__ out,
    int n, int num_blocks
) {
    const int tid = threadIdx.x;
    const int row = tid / K_DIM;
    const int tx  = tid % K_DIM;
    const float iscale = *inv_scale;

    const int stride32 = K_DIM * 32;
    const int n32 = (n / stride32) * stride32;

    // Grid-stride loop over RSR blocks
    for (int blk = blockIdx.x; blk < num_blocks; blk += gridDim.x) {
        const uint8_t* w_row = w + (static_cast<int64_t>(blk) * K + row) * (n / 4);

        int32_t acc = 0;

        for (int base = tx * 32; base < n32; base += stride32) {
            int4 va0 = *reinterpret_cast<const int4*>(v_i8 + base);
            int4 va1 = *reinterpret_cast<const int4*>(v_i8 + base + 16);

            const uint32_t wb0 = __ldg(reinterpret_cast<const uint32_t*>(w_row + base / 4));
            const uint32_t wb1 = __ldg(reinterpret_cast<const uint32_t*>(w_row + base / 4 + 4));

            #pragma unroll
            for (int g = 0; g < 4; g++)
                acc = __dp4a(*(reinterpret_cast<const int*>(&va0) + g),
                             decode_byte_v65(static_cast<uint8_t>(wb0 >> (g * 8))), acc);
            #pragma unroll
            for (int g = 0; g < 4; g++)
                acc = __dp4a(*(reinterpret_cast<const int*>(&va1) + g),
                             decode_byte_v65(static_cast<uint8_t>(wb1 >> (g * 8))), acc);
        }
        // Remainder
        for (int base = n32 + tx * 16; base < n; base += K_DIM * 16) {
            if (base + 16 > n) break;
            int4 va = *reinterpret_cast<const int4*>(v_i8 + base);
            uint32_t wb = __ldg(reinterpret_cast<const uint32_t*>(w_row + base / 4));
            #pragma unroll
            for (int g = 0; g < 4; g++)
                acc = __dp4a(*(reinterpret_cast<const int*>(&va) + g),
                             decode_byte_v65(static_cast<uint8_t>(wb >> (g * 8))), acc);
        }

        // Warp reduction
        const int local_r  = row % (32 / K_DIM);
        const uint32_t mask = ((1u << K_DIM) - 1u) << (local_r * K_DIM);
        for (int off = K_DIM / 2; off > 0; off >>= 1)
            acc += __shfl_down_sync(mask, acc, off);

        if (tx == 0)
            out[static_cast<int64_t>(blk) * K + row] =
                static_cast<float>(acc) * iscale;
    }
}

} // namespace

static void launch_v6_5(
    const uint8_t* w, const int8_t* v_i8, const float* inv_scale,
    float* out, int n, int k, int num_blocks, int k_dim, int grid_size
) {
#define LAUNCH(K, KD) \
    rsr_direct_v6_5_kernel<K, KD, K*KD><<<grid_size, K*KD>>>(w, v_i8, inv_scale, out, n, num_blocks)

    if (k_dim == 32) {
        switch (k) {
            case 2:  LAUNCH(2,  32); break; case 4:  LAUNCH(4,  32); break;
            case 6:  LAUNCH(6,  32); break; case 8:  LAUNCH(8,  32); break;
            case 10: LAUNCH(10, 32); break; case 12: LAUNCH(12, 32); break;
            default: TORCH_CHECK(false, "k not supported");
        }
    } else if (k_dim == 16) {
        switch (k) {
            case 2:  LAUNCH(2,  16); break; case 4:  LAUNCH(4,  16); break;
            case 6:  LAUNCH(6,  16); break; case 8:  LAUNCH(8,  16); break;
            case 10: LAUNCH(10, 16); break; case 12: LAUNCH(12, 16); break;
            default: TORCH_CHECK(false, "k not supported");
        }
    } else {
        TORCH_CHECK(false, "k_dim must be 16 or 32");
    }
#undef LAUNCH
}

void rsr_direct_v6_5_fused(
    torch::Tensor packed_w,
    torch::Tensor v,
    torch::Tensor v_i8_buf,
    torch::Tensor inv_scale_buf,
    torch::Tensor out,
    int n, int k, int num_blocks, int k_dim, int grid_size
) {
    TORCH_CHECK(v.dtype() == torch::kFloat32);
    TORCH_CHECK(packed_w.dtype() == torch::kUInt8);

    quantize_fp32_to_int8_v65<<<1, 256>>>(
        v.data_ptr<float>(),
        v_i8_buf.data_ptr<int8_t>(),
        inv_scale_buf.data_ptr<float>(),
        n
    );

    launch_v6_5(packed_w.data_ptr<uint8_t>(), v_i8_buf.data_ptr<int8_t>(),
                inv_scale_buf.data_ptr<float>(), out.data_ptr<float>(),
                n, k, num_blocks, k_dim, grid_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rsr_direct_v6_5_fused", &rsr_direct_v6_5_fused,
          "RSR direct INT2+INT8+dp4a GEMV v6.5 (grid-stride, tuned grid size)");
}
