/*
 * Binary matrix GEMV kernel inspired by BitNet.cpp's I2_S approach.
 *
 * Technique (from microsoft/BitNet src/ggml-bitnet-mad.cpp):
 *   - Pack binary weights (0/1) into 2 bits each, 4 per byte.
 *   - Quantize the float activation vector to int8.
 *   - Compute dot products using SIMD: unpack 2-bit weights with shifts+masks,
 *     then use maddubs (AVX2) or smull/sadalp (NEON) for fast multiply-accumulate.
 *
 * For binary {0,1} we use: 0 -> 0x00, 1 -> 0x01 in 2-bit slots.
 * BitNet uses 0->-1, 1->0, 2->+1 but we only need 0 and 1.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

/* ---------- Weight packing (preprocessing) ----------
 * Pack an n_rows x n_cols binary float matrix (row-major) into 2-bit-per-element
 * format.  4 elements are packed per byte following BitNet's I2_S layout:
 *   byte = (w0 << 6) | (w1 << 4) | (w2 << 2) | w3
 *
 * Groups of 4 rows are interleaved per column so that a single byte load
 * gives 4 row-values at the same column, matching the AVX2 kernel's access pattern.
 *
 * Output size: (n_rows * n_cols) / 4 bytes.
 */
void bitnet_pack_weights(const float *weights, uint8_t *packed,
                         int n_rows, int n_cols) {
    int n_row_groups = n_rows / 4;
    memset(packed, 0, (size_t)(n_rows * n_cols) / 4);

    for (int rg = 0; rg < n_row_groups; rg++) {
        int r0 = rg * 4;
        int base = rg * n_cols;          /* output byte offset */
        for (int col = 0; col < n_cols; col++) {
            uint8_t b0 = (uint8_t)weights[(r0 + 0) * n_cols + col];
            uint8_t b1 = (uint8_t)weights[(r0 + 1) * n_cols + col];
            uint8_t b2 = (uint8_t)weights[(r0 + 2) * n_cols + col];
            uint8_t b3 = (uint8_t)weights[(r0 + 3) * n_cols + col];
            packed[base + col] = (uint8_t)((b0 << 6) | (b1 << 4) | (b2 << 2) | b3);
        }
    }
}

/* ---------- Activation quantization ----------
 * Quantize float vector to int8: scale = 127 / max(|v|), q[i] = round(v[i] * scale).
 * Returns the inverse scale (1/scale) so the caller can rescale the int result.
 */
float bitnet_quantize_activation(const float *v, int8_t *qv, int n) {
    float amax = 0.0f;
    for (int i = 0; i < n; i++) {
        float a = fabsf(v[i]);
        if (a > amax) amax = a;
    }
    if (amax < 1e-10f) {
        memset(qv, 0, (size_t)n);
        return 0.0f;
    }
    float scale = 127.0f / amax;
    for (int i = 0; i < n; i++) {
        float val = v[i] * scale;
        int q = (int)roundf(val);
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        qv[i] = (int8_t)q;
    }
    return 1.0f / scale;   /* inv_scale */
}

/* ---------- GEMV: packed_weights (n_rows x n_cols packed) @ qv (n_cols) ----------
 * Computes dot product of each row of the packed weight matrix with the int8
 * activation vector.  Result is written to out[] as float (after rescaling).
 *
 * n_rows must be a multiple of 4.  n_cols must be a multiple of 32 (AVX2) or
 * 16 (NEON) for the SIMD path; the scalar fallback handles any size.
 */

#if defined(__AVX2__)

void bitnet_gemv(const uint8_t *packed, const int8_t *qv, float *out,
                 int n_rows, int n_cols, float inv_scale) {
    const __m256i mask = _mm256_set1_epi8(0x03);
    const __m256i one16 = _mm256_set1_epi16(1);

    int n_row_groups = n_rows / 4;

    for (int rg = 0; rg < n_row_groups; rg++) {
        const uint8_t *px = packed + rg * n_cols;
        const int8_t *py = qv;

        __m256i accu0 = _mm256_setzero_si256();
        __m256i accu1 = _mm256_setzero_si256();
        __m256i accu2 = _mm256_setzero_si256();
        __m256i accu3 = _mm256_setzero_si256();

        int col = 0;
        for (; col + 31 < n_cols; col += 32) {
            /* Load 32 packed bytes = 32 columns x 4 rows */
            __m256i raw = _mm256_loadu_si256((const __m256i *)(px + col));

            /* Unpack each of the 4 rows (2-bit fields) */
            __m256i w0 = _mm256_and_si256(_mm256_srli_epi16(raw, 6), mask);
            __m256i w1 = _mm256_and_si256(_mm256_srli_epi16(raw, 4), mask);
            __m256i w2 = _mm256_and_si256(_mm256_srli_epi16(raw, 2), mask);
            __m256i w3 = _mm256_and_si256(raw, mask);

            /* Load 32 int8 activation values */
            __m256i act = _mm256_loadu_si256((const __m256i *)(py + col));

            /* maddubs: treats first arg as unsigned, second as signed.
             * Our weights (0 or 1) are unsigned, activations are signed int8.
             * Result: pairs of adjacent u8*i8 products summed into i16. */
            __m256i d0 = _mm256_maddubs_epi16(w0, act);
            __m256i d1 = _mm256_maddubs_epi16(w1, act);
            __m256i d2 = _mm256_maddubs_epi16(w2, act);
            __m256i d3 = _mm256_maddubs_epi16(w3, act);

            /* Widen i16 -> i32 and accumulate */
            accu0 = _mm256_add_epi32(accu0, _mm256_madd_epi16(d0, one16));
            accu1 = _mm256_add_epi32(accu1, _mm256_madd_epi16(d1, one16));
            accu2 = _mm256_add_epi32(accu2, _mm256_madd_epi16(d2, one16));
            accu3 = _mm256_add_epi32(accu3, _mm256_madd_epi16(d3, one16));
        }

        /* Horizontal sum for each row */
        __m128i lo, hi, sum128, hi64, sum64, hi32;

        /* Row 0 */
        lo = _mm256_castsi256_si128(accu0);
        hi = _mm256_extracti128_si256(accu0, 1);
        sum128 = _mm_add_epi32(lo, hi);
        hi64 = _mm_unpackhi_epi64(sum128, sum128);
        sum64 = _mm_add_epi32(sum128, hi64);
        hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
        int s0 = _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));

        /* Row 1 */
        lo = _mm256_castsi256_si128(accu1);
        hi = _mm256_extracti128_si256(accu1, 1);
        sum128 = _mm_add_epi32(lo, hi);
        hi64 = _mm_unpackhi_epi64(sum128, sum128);
        sum64 = _mm_add_epi32(sum128, hi64);
        hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
        int s1 = _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));

        /* Row 2 */
        lo = _mm256_castsi256_si128(accu2);
        hi = _mm256_extracti128_si256(accu2, 1);
        sum128 = _mm_add_epi32(lo, hi);
        hi64 = _mm_unpackhi_epi64(sum128, sum128);
        sum64 = _mm_add_epi32(sum128, hi64);
        hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
        int s2 = _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));

        /* Row 3 */
        lo = _mm256_castsi256_si128(accu3);
        hi = _mm256_extracti128_si256(accu3, 1);
        sum128 = _mm_add_epi32(lo, hi);
        hi64 = _mm_unpackhi_epi64(sum128, sum128);
        sum64 = _mm_add_epi32(sum128, hi64);
        hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
        int s3 = _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));

        /* Scalar tail for remaining columns */
        for (; col < n_cols; col++) {
            uint8_t byte = px[col];
            int8_t a = py[col];
            s0 += ((byte >> 6) & 0x3) * a;
            s1 += ((byte >> 4) & 0x3) * a;
            s2 += ((byte >> 2) & 0x3) * a;
            s3 += ((byte >> 0) & 0x3) * a;
        }

        out[rg * 4 + 0] = (float)s0 * inv_scale;
        out[rg * 4 + 1] = (float)s1 * inv_scale;
        out[rg * 4 + 2] = (float)s2 * inv_scale;
        out[rg * 4 + 3] = (float)s3 * inv_scale;
    }
}

#elif defined(__ARM_NEON)

void bitnet_gemv(const uint8_t *packed, const int8_t *qv, float *out,
                 int n_rows, int n_cols, float inv_scale) {
    const uint8x16_t mask = vdupq_n_u8(0x03);
    int n_row_groups = n_rows / 4;

    for (int rg = 0; rg < n_row_groups; rg++) {
        const uint8_t *px = packed + rg * n_cols;
        const int8_t *py = qv;

        int32x4_t accu0 = vdupq_n_s32(0);
        int32x4_t accu1 = vdupq_n_s32(0);
        int32x4_t accu2 = vdupq_n_s32(0);
        int32x4_t accu3 = vdupq_n_s32(0);

        int col = 0;
        for (; col + 15 < n_cols; col += 16) {
            uint8x16_t raw = vld1q_u8(px + col);
            int8x16_t act = vld1q_s8(py + col);

            /* Unpack 4 rows from 2-bit fields */
            int8x16_t w0 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(raw, 6), mask));
            int8x16_t w1 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(raw, 4), mask));
            int8x16_t w2 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(raw, 2), mask));
            int8x16_t w3 = vreinterpretq_s8_u8(vandq_u8(raw, mask));

            /* Widening multiply and pairwise add: i8*i8 -> i16, then accumulate to i32 */
            int16x8_t p0_lo = vmull_s8(vget_low_s8(w0), vget_low_s8(act));
            int16x8_t p0_hi = vmull_s8(vget_high_s8(w0), vget_high_s8(act));
            accu0 = vpadalq_s16(accu0, p0_lo);
            accu0 = vpadalq_s16(accu0, p0_hi);

            int16x8_t p1_lo = vmull_s8(vget_low_s8(w1), vget_low_s8(act));
            int16x8_t p1_hi = vmull_s8(vget_high_s8(w1), vget_high_s8(act));
            accu1 = vpadalq_s16(accu1, p1_lo);
            accu1 = vpadalq_s16(accu1, p1_hi);

            int16x8_t p2_lo = vmull_s8(vget_low_s8(w2), vget_low_s8(act));
            int16x8_t p2_hi = vmull_s8(vget_high_s8(w2), vget_high_s8(act));
            accu2 = vpadalq_s16(accu2, p2_lo);
            accu2 = vpadalq_s16(accu2, p2_hi);

            int16x8_t p3_lo = vmull_s8(vget_low_s8(w3), vget_low_s8(act));
            int16x8_t p3_hi = vmull_s8(vget_high_s8(w3), vget_high_s8(act));
            accu3 = vpadalq_s16(accu3, p3_lo);
            accu3 = vpadalq_s16(accu3, p3_hi);
        }

        int s0 = vaddvq_s32(accu0);
        int s1 = vaddvq_s32(accu1);
        int s2 = vaddvq_s32(accu2);
        int s3 = vaddvq_s32(accu3);

        for (; col < n_cols; col++) {
            uint8_t byte = px[col];
            int8_t a = py[col];
            s0 += ((byte >> 6) & 0x3) * a;
            s1 += ((byte >> 4) & 0x3) * a;
            s2 += ((byte >> 2) & 0x3) * a;
            s3 += ((byte >> 0) & 0x3) * a;
        }

        out[rg * 4 + 0] = (float)s0 * inv_scale;
        out[rg * 4 + 1] = (float)s1 * inv_scale;
        out[rg * 4 + 2] = (float)s2 * inv_scale;
        out[rg * 4 + 3] = (float)s3 * inv_scale;
    }
}

#else
/* Scalar fallback */

void bitnet_gemv(const uint8_t *packed, const int8_t *qv, float *out,
                 int n_rows, int n_cols, float inv_scale) {
    int n_row_groups = n_rows / 4;

    for (int rg = 0; rg < n_row_groups; rg++) {
        const uint8_t *px = packed + rg * n_cols;
        int s0 = 0, s1 = 0, s2 = 0, s3 = 0;

        for (int col = 0; col < n_cols; col++) {
            uint8_t byte = px[col];
            int8_t a = qv[col];
            s0 += ((byte >> 6) & 0x3) * a;
            s1 += ((byte >> 4) & 0x3) * a;
            s2 += ((byte >> 2) & 0x3) * a;
            s3 += ((byte >> 0) & 0x3) * a;
        }

        out[rg * 4 + 0] = (float)s0 * inv_scale;
        out[rg * 4 + 1] = (float)s1 * inv_scale;
        out[rg * 4 + 2] = (float)s2 * inv_scale;
        out[rg * 4 + 3] = (float)s3 * inv_scale;
    }
}

#endif
