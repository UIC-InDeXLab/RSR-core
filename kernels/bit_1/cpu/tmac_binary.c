/*
 * T-MAC-style binary GEMV baseline (LUT-based, following microsoft/T-MAC).
 *
 * For binary {0, 1} weights, we map to T-MAC's {-1, +1} representation:
 *   w_tmac = 2*w - 1  (0 -> -1, 1 -> +1)
 *
 * Then: M*v = (M_tmac*v + sum(v) * ones) / 2
 *   where M_tmac is the {-1,+1} matrix.
 *
 * T-MAC groups 4 binary weights per nibble (2^4 = 16 entries in LUT).
 * For each group of 4 activations (b0,b1,b2,b3), build LUT:
 *   lut[idx] = sum(s_i * b_i) where s_i = +1 if bit i of idx is set, else -1
 *
 * Uses _mm256_shuffle_epi8 (pshufb) for 32 parallel 4-bit lookups.
 *
 * Compile:
 *   gcc -O3 -march=native -mavx2 -shared -fPIC -o tmac_binary.so tmac_binary.c -lm
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

/*
 * Pack binary matrix M (float {0,1}, row-major n×n) into nibble format.
 *
 * Groups of 4 columns → 1 nibble. Each nibble is the 4-bit pattern of weights.
 * Layout: column-group-major for SIMD-friendly row processing.
 *   For each group g (columns 4*g to 4*g+3), all rows' nibbles are contiguous.
 *   packed[g * row_bytes + r/2]:
 *     low nibble  = row r (even)
 *     high nibble = row r+1 (odd)
 *
 * n_groups = ceil(n/4), row_bytes = ceil(n/2).
 */
void tmac_binary_pack(
    const float *M,        /* n x n, row-major, values {0, 1} */
    uint8_t     *packed,   /* out: n_groups * row_bytes bytes */
    int          n
)
{
    int n_groups = (n + 3) / 4;
    int row_bytes = (n + 1) / 2;

    memset(packed, 0, (size_t)n_groups * row_bytes);

    for (int g = 0; g < n_groups; g++) {
        int c_base = g * 4;
        uint8_t *group_out = packed + (int64_t)g * row_bytes;

        for (int r = 0; r < n; r++) {
            /* Build 4-bit nibble from 4 binary weights */
            uint8_t nibble = 0;
            for (int b = 0; b < 4; b++) {
                int col = c_base + b;
                if (col < n && M[(int64_t)r * n + col] > 0.5f) {
                    nibble |= (1 << b);
                }
            }

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
 * For each group of 4 activations (b0,b1,b2,b3):
 *   lut[idx] = sum(s_i * b_i) where s_i = +1 if bit i set, else -1
 *
 * T-MAC symmetry trick: lut[idx] = -lut[15-idx], so only compute half.
 *
 * Global lut_scale = max(|lut entries|) / 127.
 *
 * The returned LUT is for the {-1,+1} representation. The caller must
 * correct: result = (tmac_result + sum_v) / 2.
 */
void tmac_binary_build_lut(
    const float *v,          /* n */
    int8_t      *qlut,       /* out: n_groups * 16 */
    float       *lut_scale,  /* out: single float */
    int          n
)
{
    int n_groups = (n + 3) / 4;

    /* Step 1: find global max for quantization */
    float global_max = 0.0f;
    for (int g = 0; g < n_groups; g++) {
        float abssum = 0.0f;
        for (int b = 0; b < 4; b++) {
            int idx = g * 4 + b;
            if (idx < n) abssum += fabsf(v[idx]);
        }
        if (abssum > global_max) global_max = abssum;
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
        float b[4];
        for (int i = 0; i < 4; i++) {
            int idx = g * 4 + i;
            b[i] = (idx < n) ? v[idx] : 0.0f;
        }

        int8_t *lut = qlut + g * 16;

        /* Build all 16 entries using symmetry: lut[idx] = -lut[15-idx] */
        for (int idx = 0; idx < 16; idx++) {
            float val = 0.0f;
            for (int i = 0; i < 4; i++) {
                float sign = (idx & (1 << i)) ? 1.0f : -1.0f;
                val += sign * b[i];
            }
            int q = (int)roundf(val * inv_scale);
            if (q > 127) q = 127;
            if (q < -127) q = -127;
            lut[idx] = (int8_t)q;
        }
    }
}

/*
 * T-MAC binary GEMV: out = M @ v (approximately, due to int8 LUT quantization)
 *
 * Computes M_tmac @ v using LUT+pshufb, then corrects:
 *   out[row] = (tmac_result[row] + sum_v) / 2
 */

#if defined(__AVX2__)

void tmac_binary_gemv(
    const uint8_t *packed,     /* column-group-major packed weights */
    const int8_t  *qlut,       /* n_groups * 16 */
    float          lut_scale,  /* single global scale */
    const float   *v,          /* n (for computing sum_v) */
    float         *out,        /* n */
    int            n
)
{
    int n_groups = (n + 3) / 4;
    int row_bytes = (n + 1) / 2;

    /* Compute sum(v) for bias correction: M*v = (M_tmac*v + sum(v)*ones) / 2 */
    float sum_v = 0.0f;
    for (int i = 0; i < n; i++) sum_v += v[i];
    float half_sum_v = sum_v * 0.5f;

    /* Process 32 rows at a time */
    int row = 0;
    for (; row + 31 < n; row += 32) {
        __m256i acc_i32_0 = _mm256_setzero_si256();
        __m256i acc_i32_1 = _mm256_setzero_si256();
        __m256i acc_i32_2 = _mm256_setzero_si256();
        __m256i acc_i32_3 = _mm256_setzero_si256();

        __m256i acc_i16_lo = _mm256_setzero_si256();
        __m256i acc_i16_hi = _mm256_setzero_si256();
        int flush_count = 0;

        for (int g = 0; g < n_groups; g++) {
            const uint8_t *gp = packed + (int64_t)g * row_bytes + row / 2;

            __m128i raw128 = _mm_loadu_si128((const __m128i *)gp);
            __m128i lo_nib = _mm_and_si128(raw128, _mm_set1_epi8(0x0F));
            __m128i hi_nib = _mm_and_si128(_mm_srli_epi16(raw128, 4), _mm_set1_epi8(0x0F));

            __m128i idx_lo16 = _mm_unpacklo_epi8(lo_nib, hi_nib);
            __m128i idx_hi16 = _mm_unpackhi_epi8(lo_nib, hi_nib);

            __m128i lut128 = _mm_loadu_si128((const __m128i *)(qlut + g * 16));
            __m256i lut256 = _mm256_set_m128i(lut128, lut128);

            __m256i all_indices = _mm256_set_m128i(idx_hi16, idx_lo16);
            __m256i result = _mm256_shuffle_epi8(lut256, all_indices);

            __m128i res_lo128 = _mm256_castsi256_si128(result);
            __m128i res_hi128 = _mm256_extracti128_si256(result, 1);

            acc_i16_lo = _mm256_add_epi16(acc_i16_lo, _mm256_cvtepi8_epi16(res_lo128));
            acc_i16_hi = _mm256_add_epi16(acc_i16_hi, _mm256_cvtepi8_epi16(res_hi128));

            flush_count++;
            if (flush_count >= 128) {
                __m128i lo_half, hi_half;

                lo_half = _mm256_castsi256_si128(acc_i16_lo);
                hi_half = _mm256_extracti128_si256(acc_i16_lo, 1);
                acc_i32_0 = _mm256_add_epi32(acc_i32_0, _mm256_cvtepi16_epi32(lo_half));
                acc_i32_1 = _mm256_add_epi32(acc_i32_1, _mm256_cvtepi16_epi32(hi_half));

                lo_half = _mm256_castsi256_si128(acc_i16_hi);
                hi_half = _mm256_extracti128_si256(acc_i16_hi, 1);
                acc_i32_2 = _mm256_add_epi32(acc_i32_2, _mm256_cvtepi16_epi32(lo_half));
                acc_i32_3 = _mm256_add_epi32(acc_i32_3, _mm256_cvtepi16_epi32(hi_half));

                acc_i16_lo = _mm256_setzero_si256();
                acc_i16_hi = _mm256_setzero_si256();
                flush_count = 0;
            }
        }

        /* Final flush */
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

        /* Convert to float: result = tmac_result * lut_scale * 0.5 + half_sum_v */
        __m256 half_scale = _mm256_set1_ps(lut_scale * 0.5f);
        __m256 bias = _mm256_set1_ps(half_sum_v);

        __m256 f0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc_i32_0), half_scale, bias);
        __m256 f1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc_i32_1), half_scale, bias);
        __m256 f2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc_i32_2), half_scale, bias);
        __m256 f3 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc_i32_3), half_scale, bias);

        _mm256_storeu_ps(out + row + 0,  f0);
        _mm256_storeu_ps(out + row + 8,  f1);
        _mm256_storeu_ps(out + row + 16, f2);
        _mm256_storeu_ps(out + row + 24, f3);
    }

    /* Scalar tail */
    for (; row < n; row++) {
        int acc = 0;
        for (int g = 0; g < n_groups; g++) {
            const uint8_t *gp = packed + (int64_t)g * row_bytes;
            int byte_idx = row / 2;
            uint8_t byte_val = gp[byte_idx];
            int idx = (row & 1) ? ((byte_val >> 4) & 0x0F) : (byte_val & 0x0F);
            acc += (int)qlut[g * 16 + idx];
        }
        out[row] = (float)acc * lut_scale * 0.5f + half_sum_v;
    }
}

#else
/* Scalar fallback */

void tmac_binary_gemv(
    const uint8_t *packed,
    const int8_t  *qlut,
    float          lut_scale,
    const float   *v,
    float         *out,
    int            n
)
{
    int n_groups = (n + 3) / 4;
    int row_bytes = (n + 1) / 2;

    float sum_v = 0.0f;
    for (int i = 0; i < n; i++) sum_v += v[i];
    float half_sum_v = sum_v * 0.5f;

    for (int row = 0; row < n; row++) {
        int acc = 0;
        for (int g = 0; g < n_groups; g++) {
            const uint8_t *gp = packed + (int64_t)g * row_bytes;
            int byte_idx = row / 2;
            uint8_t byte_val = gp[byte_idx];
            int idx = (row & 1) ? ((byte_val >> 4) & 0x0F) : (byte_val & 0x0F);
            acc += (int)qlut[g * 16 + idx];
        }
        out[row] = (float)acc * lut_scale * 0.5f + half_sum_v;
    }
}

#endif
