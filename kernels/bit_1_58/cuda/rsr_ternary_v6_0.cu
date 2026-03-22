/*
 * RSR ternary CUDA kernel v6.0 — INT2 weights + INT8 input + dp4a.
 *
 * Borrows BitNet's full stack:
 *   1. Weights packed as INT2 (4 per byte, -1→1, 0→2, +1→3) — 64 MB at n=16384
 *   2. Input quantized to INT8 (absmax, fused on GPU)
 *   3. __dp4a for INT8×INT8→INT32 (4 MACs per instruction)
 *   4. Warp-level reduction — no shared memory needed
 *
 * Weight layout: [num_blocks * K, n/4] uint8
 *   packed[(b*K + r), j//4] = INT2(w[r,j]) | INT2(w[r,j+1])<<2 | ...
 *
 * Template K_DIM controls threads per output row:
 *   K_DIM=16 → THREADS = K*16, for K=8: 128 threads (same as BitNet)
 *   K_DIM=8  → THREADS = K*8,  for K=8:  64 threads (2× more blocks/SM)
 *
 * K_per_loop = 16: each iteration loads int4 (16 INT8 input) + uint32
 * (4 bytes = 16 INT2 weights), does 4 dp4a calls.
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

// ─── Fused FP32→INT8 quantization (reused from v5.1) ───────────────────────

__global__ void quantize_fp32_to_int8_v6(
    const float* __restrict__ v,
    int8_t*       __restrict__ v_i8,
    float*        __restrict__ inv_scale_out,
    int n
) {
    const int tid = threadIdx.x;
    const int THREADS = blockDim.x;

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
    if (tid == 0) {
        amax_s = fmaxf(local_max, 1e-5f);
        *inv_scale_out = amax_s / 127.f;
    }
    __syncthreads();

    const float scale = 127.f / amax_s;
    for (int i = tid; i < n; i += THREADS) {
        float val = roundf(v[i] * scale);
        v_i8[i] = static_cast<int8_t>(fminf(fmaxf(val, -128.f), 127.f));
    }
}

// ─── Main GEMV kernel ────────────────────────────────────────────────────────
//
// Template parameters:
//   K      : output rows per RSR block (2, 4, 6, 8, 10, 12)
//   K_DIM  : threads per output row (8, 16, or 32)
//   THREADS: total threads = K * K_DIM
//
// Each CUDA block handles one RSR block (K output rows, n input columns).

template <int K, int K_DIM, int THREADS>
__global__ void __launch_bounds__(THREADS)
rsr_direct_v6_0_kernel(
    const uint8_t* __restrict__ w,          // [num_blocks*K, n/4]
    const int8_t*  __restrict__ v_i8,       // [n]
    const float*   __restrict__ inv_scale,  // [1]
    float*         __restrict__ out,        // [num_blocks * K]
    int n
) {
    const int tid     = threadIdx.x;
    const int row     = tid / K_DIM;   // which output row [0..K-1]
    const int tx      = tid % K_DIM;   // K-dim thread [0..K_DIM-1]
    const int blk     = blockIdx.x;

    const uint8_t* w_row = w + (static_cast<int64_t>(blk) * K + row) * (n / 4);

    int32_t acc = 0;

    // Main loop: K_DIM threads span n columns in strides of K_DIM*16
    const int n16 = (n / (K_DIM * 16)) * (K_DIM * 16);
    for (int base = tx * 16; base < n16; base += K_DIM * 16) {
        // 16 INT8 input values → 1 int4 load
        int4 va = *reinterpret_cast<const int4*>(v_i8 + base);

        // 16 INT2 weight values = 4 bytes → 1 uint32 load
        uint32_t wb = *reinterpret_cast<const uint32_t*>(w_row + base / 4);

        // 4 dp4a, one per byte of weights (4 columns each)
        #pragma unroll
        for (int g = 0; g < 4; g++) {
            const uint8_t byte = static_cast<uint8_t>(wb >> (g * 8));
            // Expand 4 × INT2 → 4 × INT8 in one int32
            int wp = (byte & 3)
                   | (((byte >> 2) & 3) << 8)
                   | (((byte >> 4) & 3) << 16)
                   | (((byte >> 6) & 3) << 24);
            // Decode: subtract 2 from each byte (−1→1−2=−1, 0→2−2=0, +1→3−2=+1)
            wp = __vsubss4(wp, 0x02020202);

            const int ap = *(reinterpret_cast<const int*>(&va) + g);
            acc = __dp4a(ap, wp, acc);
        }
    }
    // Scalar remainder (handles n not divisible by K_DIM*16)
    for (int base = n16 + tx * 4; base < n; base += K_DIM * 4) {
        if (base + 4 > n) break;
        const uint8_t byte = w_row[base / 4];
        int wp = (byte & 3)
               | (((byte >> 2) & 3) << 8)
               | (((byte >> 4) & 3) << 16)
               | (((byte >> 6) & 3) << 24);
        wp = __vsubss4(wp, 0x02020202);
        int ap = *reinterpret_cast<const int*>(v_i8 + base);
        acc = __dp4a(ap, wp, acc);
    }

    // Warp reduction within K_DIM threads of the same row
    // Compute subgroup mask: K_DIM threads form a contiguous group within the warp
    {
        const int local_r  = row % (32 / K_DIM);
        const uint32_t mask = ((1u << K_DIM) - 1u) << (local_r * K_DIM);
        for (int off = K_DIM / 2; off > 0; off >>= 1)
            acc += __shfl_down_sync(mask, acc, off);
    }

    if (tx == 0)
        out[static_cast<int64_t>(blk) * K + row] =
            static_cast<float>(acc) * (*inv_scale);
}

} // namespace

// ─── Dispatcher ─────────────────────────────────────────────────────────────

static void launch_v6_0(
    const uint8_t* w, const int8_t* v_i8, const float* inv_scale,
    float* out, int n, int k, int num_blocks, int k_dim
) {
#define LAUNCH(K, KD) \
    rsr_direct_v6_0_kernel<K, KD, K*KD><<<num_blocks, K*KD>>>(w, v_i8, inv_scale, out, n)

    if (k_dim == 16) {
        switch (k) {
            case 2:  LAUNCH(2,  16); break;
            case 4:  LAUNCH(4,  16); break;
            case 6:  LAUNCH(6,  16); break;
            case 8:  LAUNCH(8,  16); break;
            case 10: LAUNCH(10, 16); break;
            case 12: LAUNCH(12, 16); break;
            default: TORCH_CHECK(false, "k not supported");
        }
    } else if (k_dim == 8) {
        switch (k) {
            case 2:  LAUNCH(2,  8); break;
            case 4:  LAUNCH(4,  8); break;
            case 6:  LAUNCH(6,  8); break;
            case 8:  LAUNCH(8,  8); break;
            case 10: LAUNCH(10, 8); break;
            case 12: LAUNCH(12, 8); break;
            default: TORCH_CHECK(false, "k not supported");
        }
    } else if (k_dim == 32) {
        switch (k) {
            case 2:  LAUNCH(2,  32); break;
            case 4:  LAUNCH(4,  32); break;
            case 6:  LAUNCH(6,  32); break;
            case 8:  LAUNCH(8,  32); break;
            case 10: LAUNCH(10, 32); break;
            case 12: LAUNCH(12, 32); break;
            default: TORCH_CHECK(false, "k not supported");
        }
    } else {
        TORCH_CHECK(false, "k_dim must be 8, 16, or 32");
    }
#undef LAUNCH
}

void rsr_direct_v6_0_fused(
    torch::Tensor packed_w,     // [num_blocks*k, n/4] uint8
    torch::Tensor v,            // [n] float32
    torch::Tensor v_i8_buf,     // [n] int8  (pre-allocated)
    torch::Tensor inv_scale_buf,// [1] float (pre-allocated)
    torch::Tensor out,          // [num_blocks * k] float32
    int n, int k, int num_blocks, int k_dim
) {
    TORCH_CHECK(v.dtype() == torch::kFloat32);
    TORCH_CHECK(v_i8_buf.dtype() == torch::kInt8);
    TORCH_CHECK(packed_w.dtype() == torch::kUInt8);

    // Step 1: fused INT8 quantization
    quantize_fp32_to_int8_v6<<<1, 256>>>(
        v.data_ptr<float>(),
        v_i8_buf.data_ptr<int8_t>(),
        inv_scale_buf.data_ptr<float>(),
        n
    );

    // Step 2: GEMV
    launch_v6_0(
        packed_w.data_ptr<uint8_t>(),
        v_i8_buf.data_ptr<int8_t>(),
        inv_scale_buf.data_ptr<float>(),
        out.data_ptr<float>(),
        n, k, num_blocks, k_dim
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rsr_direct_v6_0_fused", &rsr_direct_v6_0_fused,
          "RSR direct INT2+INT8+dp4a GEMV v6.0 (parametric k_dim)");
}
