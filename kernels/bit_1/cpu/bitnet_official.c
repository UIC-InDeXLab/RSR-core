/*
 * BitNet.cpp-style I2_S GEMV baseline for this project.
 *
 * This implementation follows the official microsoft/BitNet I2_S logic:
 *   - Ternary-ish 2-bit coding (0,1,2) for (-1,0,+1) style weights.
 *   - 4-row interleaved packing layout from quantize_i2_s non-ACT path.
 *   - int8 activation quantization and SIMD int2xint8 accumulation.
 *
 * Exposed C ABI is tailored for multiplier/bitnet_official.py.
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

/*
 * Pack float weights into official-style 2-bit codes:
 *   code 0 => negative, code 1 => zero, code 2 => positive.
 *
 * Packing layout (non-ACT_PARALLEL path from BitNet.cpp):
 * for each group of 4 rows and each column:
 *   packed_byte = (q0 << 6) | (q1 << 4) | (q2 << 2) | q3
 *
 * i2_scale_out receives max(abs(weights)).
 */
void bitnet_official_pack_weights(const float *weights, uint8_t *packed,
                                  int n_rows, int n_cols, float *i2_scale_out) {
    int64_t n = (int64_t)n_rows * (int64_t)n_cols;
    if (n_rows <= 0 || n_cols <= 0 || (n_rows % 4) != 0) {
        if (i2_scale_out) {
            *i2_scale_out = 0.0f;
        }
        return;
    }

    double max = 0.0;
    for (int64_t i = 0; i < n; i++) {
        double a = fabs((double)weights[i]);
        if (a > max) {
            max = a;
        }
    }

    const float i2_scale = (float)max;
    if (i2_scale_out) {
        *i2_scale_out = i2_scale;
    }

    uint8_t *q8 = (uint8_t *)malloc((size_t)n);
    if (!q8) {
        memset(packed, 0, (size_t)n / 4);
        return;
    }

    for (int64_t i = 0; i < n; i++) {
        if (fabs((double)weights[i]) < 1e-6) {
            q8[i] = 1;
            continue;
        }
        q8[i] = ((double)weights[i] * (double)i2_scale > 0.0) ? 2 : 0;
    }

    memset(packed, 0, (size_t)n / 4);
    const int64_t nrow4 = n_rows / 4;
    for (int64_t rg = 0; rg < nrow4; rg++) {
        const int64_t r0 = rg * 4 + 0;
        const int64_t r1 = rg * 4 + 1;
        const int64_t r2 = rg * 4 + 2;
        const int64_t r3 = rg * 4 + 3;
        const int64_t base = rg * n_cols;

        for (int64_t col = 0; col < n_cols; col++) {
            const uint8_t q0 = q8[r0 * n_cols + col];
            const uint8_t q1 = q8[r1 * n_cols + col];
            const uint8_t q2 = q8[r2 * n_cols + col];
            const uint8_t q3 = q8[r3 * n_cols + col];
            packed[base + col] = (uint8_t)((q0 << 6) | (q1 << 4) | (q2 << 2) | q3);
        }
    }

    free(q8);
}

/*
 * Per-tensor int8 quantization for activation vector.
 * Returns dequant multiplier (inverse quant scale).
 */
float bitnet_official_quantize_activation(const float *v, int8_t *qv, int n) {
    float amax = 0.0f;
    for (int i = 0; i < n; i++) {
        const float a = fabsf(v[i]);
        if (a > amax) {
            amax = a;
        }
    }

    if (amax < 1e-10f) {
        memset(qv, 0, (size_t)n);
        return 0.0f;
    }

    const float scale = 127.0f / amax;
    for (int i = 0; i < n; i++) {
        int q = (int)roundf(v[i] * scale);
        if (q > 127) {
            q = 127;
        }
        if (q < -127) {
            q = -127;
        }
        qv[i] = (int8_t)q;
    }

    return 1.0f / scale;
}

/*
 * GEMV for packed I2_S weights:
 *   packed (n_rows x n_cols, 4 rows per byte) @ qv (n_cols).
 *
 * Stored 2-bit code is shifted ternary: val = code - 1.
 * We compute sum(code * qv) via SIMD and subtract sum(qv) once per row.
 */
#if defined(__AVX2__)

void bitnet_official_gemv(const uint8_t *packed, const int8_t *qv, float *out,
                          int n_rows, int n_cols, float inv_act_scale, float i2_scale) {
    const __m256i mask = _mm256_set1_epi8(0x03);
    const __m256i one16 = _mm256_set1_epi16(1);

    int sum_qv = 0;
    for (int i = 0; i < n_cols; i++) {
        sum_qv += qv[i];
    }

    const float out_scale = inv_act_scale * i2_scale;
    const int n_row_groups = n_rows / 4;

    for (int rg = 0; rg < n_row_groups; rg++) {
        const uint8_t *px = packed + rg * n_cols;
        const int8_t *py = qv;

        __m256i accu0 = _mm256_setzero_si256();
        __m256i accu1 = _mm256_setzero_si256();
        __m256i accu2 = _mm256_setzero_si256();
        __m256i accu3 = _mm256_setzero_si256();

        int col = 0;
        for (; col + 31 < n_cols; col += 32) {
            const __m256i raw = _mm256_loadu_si256((const __m256i *)(px + col));
            const __m256i w0 = _mm256_and_si256(_mm256_srli_epi16(raw, 6), mask);
            const __m256i w1 = _mm256_and_si256(_mm256_srli_epi16(raw, 4), mask);
            const __m256i w2 = _mm256_and_si256(_mm256_srli_epi16(raw, 2), mask);
            const __m256i w3 = _mm256_and_si256(raw, mask);

            const __m256i act = _mm256_loadu_si256((const __m256i *)(py + col));
            const __m256i d0 = _mm256_maddubs_epi16(w0, act);
            const __m256i d1 = _mm256_maddubs_epi16(w1, act);
            const __m256i d2 = _mm256_maddubs_epi16(w2, act);
            const __m256i d3 = _mm256_maddubs_epi16(w3, act);

            accu0 = _mm256_add_epi32(accu0, _mm256_madd_epi16(d0, one16));
            accu1 = _mm256_add_epi32(accu1, _mm256_madd_epi16(d1, one16));
            accu2 = _mm256_add_epi32(accu2, _mm256_madd_epi16(d2, one16));
            accu3 = _mm256_add_epi32(accu3, _mm256_madd_epi16(d3, one16));
        }

        __m128i lo, hi, sum128, hi64, sum64, hi32;

        lo = _mm256_castsi256_si128(accu0);
        hi = _mm256_extracti128_si256(accu0, 1);
        sum128 = _mm_add_epi32(lo, hi);
        hi64 = _mm_unpackhi_epi64(sum128, sum128);
        sum64 = _mm_add_epi32(sum128, hi64);
        hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
        int s0 = _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));

        lo = _mm256_castsi256_si128(accu1);
        hi = _mm256_extracti128_si256(accu1, 1);
        sum128 = _mm_add_epi32(lo, hi);
        hi64 = _mm_unpackhi_epi64(sum128, sum128);
        sum64 = _mm_add_epi32(sum128, hi64);
        hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
        int s1 = _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));

        lo = _mm256_castsi256_si128(accu2);
        hi = _mm256_extracti128_si256(accu2, 1);
        sum128 = _mm_add_epi32(lo, hi);
        hi64 = _mm_unpackhi_epi64(sum128, sum128);
        sum64 = _mm_add_epi32(sum128, hi64);
        hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
        int s2 = _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));

        lo = _mm256_castsi256_si128(accu3);
        hi = _mm256_extracti128_si256(accu3, 1);
        sum128 = _mm_add_epi32(lo, hi);
        hi64 = _mm_unpackhi_epi64(sum128, sum128);
        sum64 = _mm_add_epi32(sum128, hi64);
        hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
        int s3 = _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));

        for (; col < n_cols; col++) {
            const uint8_t byte = px[col];
            const int8_t a = py[col];
            s0 += ((byte >> 6) & 0x3) * a;
            s1 += ((byte >> 4) & 0x3) * a;
            s2 += ((byte >> 2) & 0x3) * a;
            s3 += ((byte >> 0) & 0x3) * a;
        }

        out[rg * 4 + 0] = (float)(s0 - sum_qv) * out_scale;
        out[rg * 4 + 1] = (float)(s1 - sum_qv) * out_scale;
        out[rg * 4 + 2] = (float)(s2 - sum_qv) * out_scale;
        out[rg * 4 + 3] = (float)(s3 - sum_qv) * out_scale;
    }
}

#elif defined(__ARM_NEON)

void bitnet_official_gemv(const uint8_t *packed, const int8_t *qv, float *out,
                          int n_rows, int n_cols, float inv_act_scale, float i2_scale) {
    const uint8x16_t mask = vdupq_n_u8(0x03);

    int sum_qv = 0;
    for (int i = 0; i < n_cols; i++) {
        sum_qv += qv[i];
    }

    const float out_scale = inv_act_scale * i2_scale;
    const int n_row_groups = n_rows / 4;

    for (int rg = 0; rg < n_row_groups; rg++) {
        const uint8_t *px = packed + rg * n_cols;
        const int8_t *py = qv;

        int32x4_t accu0 = vdupq_n_s32(0);
        int32x4_t accu1 = vdupq_n_s32(0);
        int32x4_t accu2 = vdupq_n_s32(0);
        int32x4_t accu3 = vdupq_n_s32(0);

        int col = 0;
        for (; col + 15 < n_cols; col += 16) {
            const uint8x16_t raw = vld1q_u8(px + col);
            const int8x16_t act = vld1q_s8(py + col);

            const int8x16_t w0 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(raw, 6), mask));
            const int8x16_t w1 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(raw, 4), mask));
            const int8x16_t w2 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(raw, 2), mask));
            const int8x16_t w3 = vreinterpretq_s8_u8(vandq_u8(raw, mask));

            const int16x8_t p0_lo = vmull_s8(vget_low_s8(w0), vget_low_s8(act));
            const int16x8_t p0_hi = vmull_s8(vget_high_s8(w0), vget_high_s8(act));
            accu0 = vpadalq_s16(accu0, p0_lo);
            accu0 = vpadalq_s16(accu0, p0_hi);

            const int16x8_t p1_lo = vmull_s8(vget_low_s8(w1), vget_low_s8(act));
            const int16x8_t p1_hi = vmull_s8(vget_high_s8(w1), vget_high_s8(act));
            accu1 = vpadalq_s16(accu1, p1_lo);
            accu1 = vpadalq_s16(accu1, p1_hi);

            const int16x8_t p2_lo = vmull_s8(vget_low_s8(w2), vget_low_s8(act));
            const int16x8_t p2_hi = vmull_s8(vget_high_s8(w2), vget_high_s8(act));
            accu2 = vpadalq_s16(accu2, p2_lo);
            accu2 = vpadalq_s16(accu2, p2_hi);

            const int16x8_t p3_lo = vmull_s8(vget_low_s8(w3), vget_low_s8(act));
            const int16x8_t p3_hi = vmull_s8(vget_high_s8(w3), vget_high_s8(act));
            accu3 = vpadalq_s16(accu3, p3_lo);
            accu3 = vpadalq_s16(accu3, p3_hi);
        }

        int s0 = vaddvq_s32(accu0);
        int s1 = vaddvq_s32(accu1);
        int s2 = vaddvq_s32(accu2);
        int s3 = vaddvq_s32(accu3);

        for (; col < n_cols; col++) {
            const uint8_t byte = px[col];
            const int8_t a = py[col];
            s0 += ((byte >> 6) & 0x3) * a;
            s1 += ((byte >> 4) & 0x3) * a;
            s2 += ((byte >> 2) & 0x3) * a;
            s3 += ((byte >> 0) & 0x3) * a;
        }

        out[rg * 4 + 0] = (float)(s0 - sum_qv) * out_scale;
        out[rg * 4 + 1] = (float)(s1 - sum_qv) * out_scale;
        out[rg * 4 + 2] = (float)(s2 - sum_qv) * out_scale;
        out[rg * 4 + 3] = (float)(s3 - sum_qv) * out_scale;
    }
}

#else

void bitnet_official_gemv(const uint8_t *packed, const int8_t *qv, float *out,
                          int n_rows, int n_cols, float inv_act_scale, float i2_scale) {
    int sum_qv = 0;
    for (int i = 0; i < n_cols; i++) {
        sum_qv += qv[i];
    }

    const float out_scale = inv_act_scale * i2_scale;
    const int n_row_groups = n_rows / 4;

    for (int rg = 0; rg < n_row_groups; rg++) {
        const uint8_t *px = packed + rg * n_cols;
        int s0 = 0;
        int s1 = 0;
        int s2 = 0;
        int s3 = 0;

        for (int col = 0; col < n_cols; col++) {
            const uint8_t byte = px[col];
            const int8_t a = qv[col];
            s0 += ((byte >> 6) & 0x3) * a;
            s1 += ((byte >> 4) & 0x3) * a;
            s2 += ((byte >> 2) & 0x3) * a;
            s3 += ((byte >> 0) & 0x3) * a;
        }

        out[rg * 4 + 0] = (float)(s0 - sum_qv) * out_scale;
        out[rg * 4 + 1] = (float)(s1 - sum_qv) * out_scale;
        out[rg * 4 + 2] = (float)(s2 - sum_qv) * out_scale;
        out[rg * 4 + 3] = (float)(s3 - sum_qv) * out_scale;
    }
}

#endif
