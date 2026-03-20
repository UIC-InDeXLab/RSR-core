/*
 * T-MAC-style ternary GEMV baseline (LUT-based, following microsoft/T-MAC).
 *
 * Upstream reference:
 *   - Repo: https://github.com/microsoft/T-MAC
 *   - Files:
 *     https://github.com/microsoft/T-MAC/blob/main/python/t_mac/intrins/lut_ctor.cc
 *     https://github.com/microsoft/T-MAC/blob/main/python/t_mac/intrins/tbl.cc
 *     https://github.com/microsoft/T-MAC/blob/main/python/t_mac/weights.py
 *   - This kernel is a simplified/adapted baseline built from those LUT ideas.
 *
 * Core idea from T-MAC (microsoft/T-MAC) and BitNet.cpp TL2 kernel:
 *   - Group 2 consecutive ternary weights per 4-bit nibble index.
 *   - 3^2 = 9 possible ternary combinations fit in a 16-entry LUT.
 *   - For each pair of activations, build a 16-entry int8 LUT of all
 *     possible ternary-weighted partial sums.
 *   - Use _mm256_shuffle_epi8 (pshufb) for 32 parallel 4-bit lookups.
 *   - Accumulate int8->int16->int32, apply single global scale.
 *
 * Weight encoding (2 ternary values per nibble):
 *   index = (w0 + 1) * 3 + (w1 + 1)
 *   where w0, w1 in {-1, 0, +1} -> index in {0..8}, fits in 4 bits.
 *
 * Weight packing: column-group-major for SIMD-friendly row processing.
 *   For each activation group g, all rows' nibbles are contiguous.
 *
 * Compile:
 *   gcc -O3 -march=native -mavx2 -shared -fPIC -o tmac_ternary.so tmac_ternary.c -lm
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

/*
 * Pack ternary matrix M into column-group-major nibble format.
 *
 * Layout: for each activation group g (columns 2*g, 2*g+1),
 *   pack all rows' nibbles contiguously.
 *   packed[g * row_bytes + r/2]:
 *     low nibble  = row r   (even)
 *     high nibble = row r+1 (odd)
 *
 * n_groups = ceil(n/2), row_bytes = ceil(n/2).
 * Total: n_groups * row_bytes bytes.
 */
void tmac_ternary_pack(
    const int8_t *M,       /* n x n, row-major */
    uint8_t      *packed,  /* out: n_groups * row_bytes bytes */
    int           n
)
{
    int n_groups = (n + 1) / 2;
    int row_bytes = (n + 1) / 2;  /* ceil(n/2) nibbles -> ceil(n/2) / 2... wait */

    /* Each group has n nibbles (one per row). Pack 2 nibbles per byte. */
    row_bytes = (n + 1) / 2;  /* bytes to store n nibbles */

    memset(packed, 0, (size_t)n_groups * row_bytes);

    for (int g = 0; g < n_groups; g++) {
        int c0 = g * 2;
        int c1 = g * 2 + 1;
        uint8_t *group_out = packed + (int64_t)g * row_bytes;

        for (int r = 0; r < n; r++) {
            int8_t w0 = M[(int64_t)r * n + c0];
            int8_t w1 = (c1 < n) ? M[(int64_t)r * n + c1] : 0;
            uint8_t nibble = (uint8_t)((w0 + 1) * 3 + (w1 + 1));

            int byte_idx = r / 2;
            if (r & 1) {
                group_out[byte_idx] |= (nibble << 4);
            } else {
                group_out[byte_idx] |= nibble;
            }
        }
    }
}

/*
 * Build LUTs for all activation groups with a single global scale.
 *
 * Following T-MAC's per_tensor_quant approach:
 *   1. Compute global_max = max over all groups of (|b0| + |b1|)
 *   2. lut_scale = global_max / 127
 *   3. For each group, build 16-entry int8 LUT quantized with this scale
 *
 * Output: qlut (n_groups * 16 int8), lut_scale (single float).
 */
void tmac_ternary_build_lut(
    const float *v,          /* n */
    int8_t      *qlut,       /* out: n_groups * 16 */
    float       *lut_scale,  /* out: single float */
    int          n
)
{
    int n_groups = (n + 1) / 2;

    /* Step 1: find global max of |sum of abs activations| per group */
    float global_max = 0.0f;
    for (int g = 0; g < n_groups; g++) {
        float b0 = v[g * 2];
        float b1 = (g * 2 + 1 < n) ? v[g * 2 + 1] : 0.0f;
        float group_max = fabsf(b0) + fabsf(b1);
        if (group_max > global_max) global_max = group_max;
    }

    if (global_max < 1e-10f) {
        *lut_scale = 0.0f;
        memset(qlut, 0, (size_t)n_groups * 16);
        return;
    }

    float scale = global_max / 127.0f;
    float inv_scale = 127.0f / global_max;
    *lut_scale = scale;

    /* Step 2: build quantized LUTs */
    for (int g = 0; g < n_groups; g++) {
        float b0 = v[g * 2];
        float b1 = (g * 2 + 1 < n) ? v[g * 2 + 1] : 0.0f;
        int8_t *lut = qlut + g * 16;

        /* Build 9 entries for (w0,w1) in {-1,0,+1}^2 */
        for (int w0 = -1; w0 <= 1; w0++) {
            for (int w1 = -1; w1 <= 1; w1++) {
                int idx = (w0 + 1) * 3 + (w1 + 1);
                float val = (float)w0 * b0 + (float)w1 * b1;
                int q = (int)roundf(val * inv_scale);
                if (q > 127) q = 127;
                if (q < -127) q = -127;
                lut[idx] = (int8_t)q;
            }
        }
        /* indices 9..15 = 0 (already zeroed by memset below) */
        for (int i = 9; i < 16; i++) lut[i] = 0;
    }
}

/*
 * T-MAC ternary GEMV: out = M @ v
 *
 * Algorithm:
 *   For batches of 32 rows:
 *     For each activation group g:
 *       lut256 = broadcast(qlut[g])
 *       load 16 bytes = 32 weight nibbles for these 32 rows at group g
 *       split into lo/hi nibbles -> 32 indices
 *       result = pshufb(lut256, indices)  -> 32 int8 lookups
 *       widen to int16 and accumulate
 *     Convert int32 accumulators to float, multiply by lut_scale
 */

#if defined(__AVX2__)

void tmac_ternary_gemv(
    const uint8_t *packed,     /* column-group-major packed weights */
    const int8_t  *qlut,       /* n_groups * 16 */
    float          lut_scale,  /* single global scale */
    float         *out,        /* n */
    int            n
)
{
    int n_groups = (n + 1) / 2;
    int row_bytes = (n + 1) / 2;
    const __m256i low_mask = _mm256_set1_epi8(0x0F);

    /* Process 32 rows at a time */
    int row = 0;
    for (; row + 31 < n; row += 32) {
        /* 32 rows -> 16 bytes per group (2 nibbles per byte) */
        __m256i acc_i32_0 = _mm256_setzero_si256();  /* rows 0-7 in int32 */
        __m256i acc_i32_1 = _mm256_setzero_si256();  /* rows 8-15 */
        __m256i acc_i32_2 = _mm256_setzero_si256();  /* rows 16-23 */
        __m256i acc_i32_3 = _mm256_setzero_si256();  /* rows 24-31 */

        __m256i acc_i16_lo = _mm256_setzero_si256();  /* rows 0-15 in int16 */
        __m256i acc_i16_hi = _mm256_setzero_si256();  /* rows 16-31 in int16 */
        int flush_count = 0;

        for (int g = 0; g < n_groups; g++) {
            const uint8_t *gp = packed + (int64_t)g * row_bytes + row / 2;

            /* Load 16 bytes = 32 nibbles for 32 rows */
            __m128i raw128 = _mm_loadu_si128((const __m128i *)gp);
            __m256i raw = _mm256_cvtepu8_epi16(raw128);  /* no, wrong approach */

            /* Actually: raw128 has 16 bytes. Each byte has 2 nibbles.
             * Low nibble = even row, high nibble = odd row within the byte.
             * Byte 0: rows row+0 (lo), row+1 (hi)
             * Byte 1: rows row+2 (lo), row+3 (hi)
             * ...
             * Byte 15: rows row+30 (lo), row+31 (hi)
             *
             * We need 32 separate indices. Split into lo and hi nibbles.
             */
            __m128i lo_nib = _mm_and_si128(raw128, _mm_set1_epi8(0x0F));
            __m128i hi_nib = _mm_and_si128(_mm_srli_epi16(raw128, 4), _mm_set1_epi8(0x0F));

            /* Interleave to get indices in row order:
             * row+0, row+1, row+2, row+3, ...
             * = lo[0], hi[0], lo[1], hi[1], ... */
            __m128i idx_lo16 = _mm_unpacklo_epi8(lo_nib, hi_nib);  /* bytes 0-7 -> rows 0-15 */
            __m128i idx_hi16 = _mm_unpackhi_epi8(lo_nib, hi_nib);  /* bytes 8-15 -> rows 16-31 */

            /* Broadcast the LUT */
            __m128i lut128 = _mm_loadu_si128((const __m128i *)(qlut + g * 16));
            __m256i lut256 = _mm256_set_m128i(lut128, lut128);

            /* pshufb: 32 parallel lookups */
            __m256i indices_lo = _mm256_set_m128i(idx_lo16, idx_lo16);
            __m256i indices_hi = _mm256_set_m128i(idx_hi16, idx_hi16);

            /* Wait — pshufb operates on each 128-bit lane independently.
             * _mm256_set_m128i(lut, lut) means both lanes have the same LUT.
             * For indices, we want: lane0 = first 16 indices, lane1 = next 16 (not needed here).
             * Actually, we want to do 2 separate pshufb:
             *   res_lo = pshufb(lut256, [idx_lo16 in both lanes])  -> we only need one lane
             *   res_hi = pshufb(lut256, [idx_hi16 in both lanes])  -> we only need one lane
             * Or better: combine idx_lo16 and idx_hi16 into one 256-bit register:
             */
            __m256i all_indices = _mm256_set_m128i(idx_hi16, idx_lo16);
            __m256i result = _mm256_shuffle_epi8(lut256, all_indices);

            /* result: 32 int8 values, rows 0-15 in lo 128, rows 16-31 in hi 128 */
            /* Sign-extend int8 to int16 and accumulate */
            __m128i res_lo128 = _mm256_castsi256_si128(result);       /* rows 0-15 */
            __m128i res_hi128 = _mm256_extracti128_si256(result, 1);  /* rows 16-31 */

            /* cvtepi8_epi16: 16 int8 -> 16 int16 (in 256-bit register) */
            acc_i16_lo = _mm256_add_epi16(acc_i16_lo, _mm256_cvtepi8_epi16(res_lo128));
            acc_i16_hi = _mm256_add_epi16(acc_i16_hi, _mm256_cvtepi8_epi16(res_hi128));

            flush_count++;
            /* Flush int16 -> int32 every 128 groups to prevent overflow
             * (max int16 = 32767, max per group = 127, 32767/127 = 258, safe with 128) */
            if (flush_count >= 128) {
                /* Widen int16 to int32: split each 256-bit i16 register (16 values)
                 * into 2 x 256-bit i32 registers (8 values each) */
                __m128i lo_half, hi_half;

                /* acc_i16_lo has rows 0-15 as int16 */
                lo_half = _mm256_castsi256_si128(acc_i16_lo);
                hi_half = _mm256_extracti128_si256(acc_i16_lo, 1);
                acc_i32_0 = _mm256_add_epi32(acc_i32_0, _mm256_cvtepi16_epi32(lo_half));
                acc_i32_1 = _mm256_add_epi32(acc_i32_1, _mm256_cvtepi16_epi32(hi_half));

                /* acc_i16_hi has rows 16-31 as int16 */
                lo_half = _mm256_castsi256_si128(acc_i16_hi);
                hi_half = _mm256_extracti128_si256(acc_i16_hi, 1);
                acc_i32_2 = _mm256_add_epi32(acc_i32_2, _mm256_cvtepi16_epi32(lo_half));
                acc_i32_3 = _mm256_add_epi32(acc_i32_3, _mm256_cvtepi16_epi32(hi_half));

                acc_i16_lo = _mm256_setzero_si256();
                acc_i16_hi = _mm256_setzero_si256();
                flush_count = 0;
            }
        }

        /* Final flush of remaining int16 -> int32 */
        if (flush_count > 0) {
            __m128i lo_half, hi_half;

            lo_half = _mm256_castsi256_si128(acc_i16_lo);
            hi_half = _mm256_extracti128_si256(acc_i16_lo, 1);
            acc_i32_0 = _mm256_add_epi32(acc_i32_0, _mm256_cvtepi16_epi32(lo_half));
            acc_i32_1 = _mm256_add_epi32(acc_i32_1, _mm256_cvtepi16_epi32(hi_half));

            lo_half = _mm256_castsi256_si128(acc_i16_hi);
            hi_half = _mm256_extracti128_si256(acc_i16_hi, 1);
            acc_i32_2 = _mm256_add_epi32(acc_i32_2, _mm256_cvtepi16_epi32(lo_half));
            acc_i32_3 = _mm256_add_epi32(acc_i32_3, _mm256_cvtepi16_epi32(hi_half));
        }

        /* Convert int32 accumulators to float and apply scale */
        __m256 scale_v = _mm256_set1_ps(lut_scale);
        __m256 f0 = _mm256_mul_ps(_mm256_cvtepi32_ps(acc_i32_0), scale_v);
        __m256 f1 = _mm256_mul_ps(_mm256_cvtepi32_ps(acc_i32_1), scale_v);
        __m256 f2 = _mm256_mul_ps(_mm256_cvtepi32_ps(acc_i32_2), scale_v);
        __m256 f3 = _mm256_mul_ps(_mm256_cvtepi32_ps(acc_i32_3), scale_v);

        _mm256_storeu_ps(out + row + 0,  f0);
        _mm256_storeu_ps(out + row + 8,  f1);
        _mm256_storeu_ps(out + row + 16, f2);
        _mm256_storeu_ps(out + row + 24, f3);
    }

    /* Scalar tail for remaining rows */
    for (; row < n; row++) {
        float acc = 0.0f;
        for (int g = 0; g < n_groups; g++) {
            const uint8_t *gp = packed + (int64_t)g * row_bytes;
            int byte_idx = row / 2;
            uint8_t byte_val = gp[byte_idx];
            int idx = (row & 1) ? ((byte_val >> 4) & 0x0F) : (byte_val & 0x0F);

            int8_t val = qlut[g * 16 + idx];
            acc += (float)val * lut_scale;
        }
        out[row] = acc;
    }
}

#else
/* Scalar fallback */

void tmac_ternary_gemv(
    const uint8_t *packed,
    const int8_t  *qlut,
    float          lut_scale,
    float         *out,
    int            n
)
{
    int n_groups = (n + 1) / 2;
    int row_bytes = (n + 1) / 2;

    for (int row = 0; row < n; row++) {
        float acc = 0.0f;
        for (int g = 0; g < n_groups; g++) {
            const uint8_t *gp = packed + (int64_t)g * row_bytes;
            int byte_idx = row / 2;
            uint8_t byte_val = gp[byte_idx];
            int idx = (row & 1) ? ((byte_val >> 4) & 0x0F) : (byte_val & 0x0F);

            int8_t val = qlut[g * 16 + idx];
            acc += (float)val * lut_scale;
        }
        out[row] = acc;
    }
}

#endif
