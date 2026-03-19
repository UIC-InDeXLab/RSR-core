/*
 * BitNet.cpp-style ternary GEMV baseline (official I2_S approach).
 *
 * Faithfully follows microsoft/BitNet src/ggml-bitnet-mad.cpp:
 *   - 2-bit weight coding: -1 -> 0, 0 -> 1, +1 -> 2
 *   - 4-row interleaved packing: byte = (q0<<6)|(q1<<4)|(q2<<2)|q3
 *   - int8 activation quantization (per-tensor absmax -> 127)
 *   - AVX2 kernel: _mm256_maddubs_epi16 for uint8_weight * int8_activation
 *   - Bias correction: subtract sum(qv) per row (because code = ternary_val + 1)
 *
 * Compile:
 *   gcc -O3 -march=native -mavx2 -shared -fPIC -o bitnet_ternary.so bitnet_ternary.c -lm
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

/*
 * Pack ternary matrix M (int8, values {-1,0,+1}, row-major n×n)
 * into 2-bit format following BitNet.cpp I2_S non-ACT_PARALLEL layout.
 *
 * Encoding: -1 -> 0, 0 -> 1, +1 -> 2
 *
 * Groups of 4 rows are interleaved per column:
 *   byte = (code_row0 << 6) | (code_row1 << 4) | (code_row2 << 2) | code_row3
 *
 * Output size: (n * n) / 4 bytes.  n must be divisible by 4.
 */
void bitnet_ternary_pack(
    const int8_t *M,       /* n x n, row-major, values {-1,0,+1} */
    uint8_t      *packed,  /* out: n*n/4 bytes */
    int           n
)
{
    int n_row_groups = n / 4;
    memset(packed, 0, (size_t)(n * n) / 4);

    for (int rg = 0; rg < n_row_groups; rg++) {
        int r0 = rg * 4;
        int base = rg * n;  /* output byte offset */

        for (int col = 0; col < n; col++) {
            /* Encode: -1 -> 0, 0 -> 1, +1 -> 2 */
            uint8_t q0 = (uint8_t)(M[(r0 + 0) * n + col] + 1);
            uint8_t q1 = (uint8_t)(M[(r0 + 1) * n + col] + 1);
            uint8_t q2 = (uint8_t)(M[(r0 + 2) * n + col] + 1);
            uint8_t q3 = (uint8_t)(M[(r0 + 3) * n + col] + 1);
            packed[base + col] = (uint8_t)((q0 << 6) | (q1 << 4) | (q2 << 2) | q3);
        }
    }
}

/*
 * Per-tensor int8 quantization for activation vector.
 * scale = 127 / max(|v|),  qv[i] = round(v[i] * scale), clamped to [-127, 127].
 * Returns the inverse scale (1/scale = max(|v|)/127) for dequantization.
 */
float bitnet_ternary_quantize_activation(const float *v, int8_t *qv, int n)
{
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
        int q = (int)roundf(v[i] * scale);
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        qv[i] = (int8_t)q;
    }

    return 1.0f / scale;  /* inv_act_scale */
}

/*
 * GEMV for packed I2_S ternary weights:
 *   out = M @ v  (approximately, due to int8 activation quantization)
 *
 * packed layout: groups of 4 rows interleaved, 2-bit codes per weight.
 * Stored code = ternary_val + 1, so:
 *   sum(code * qv) = sum((ternary_val + 1) * qv) = sum(ternary_val * qv) + sum(qv)
 * We subtract sum(qv) to recover the ternary dot product.
 *
 * Final: out[row] = (raw_dot - sum_qv) * inv_act_scale
 *
 * (No i2_scale needed for ternary {-1,0,+1} — weights are exact.)
 */

#if defined(__AVX2__)

void bitnet_ternary_gemv(
    const uint8_t *packed,  /* n*n/4 bytes, 4-row interleaved */
    const int8_t  *qv,      /* n, quantized activation */
    float         *out,     /* n */
    int            n,
    float          inv_act_scale
)
{
    const __m256i mask = _mm256_set1_epi8(0x03);
    const __m256i one16 = _mm256_set1_epi16(1);

    /* Precompute sum of quantized activations (for bias correction) */
    int sum_qv = 0;
    for (int i = 0; i < n; i++) {
        sum_qv += qv[i];
    }

    int n_row_groups = n / 4;

    for (int rg = 0; rg < n_row_groups; rg++) {
        const uint8_t *px = packed + rg * n;
        const int8_t *py = qv;

        __m256i accu0 = _mm256_setzero_si256();
        __m256i accu1 = _mm256_setzero_si256();
        __m256i accu2 = _mm256_setzero_si256();
        __m256i accu3 = _mm256_setzero_si256();

        int col = 0;
        for (; col + 31 < n; col += 32) {
            /* Load 32 packed bytes = 32 columns x 4 rows */
            __m256i raw = _mm256_loadu_si256((const __m256i *)(px + col));

            /* Extract each of 4 rows (2-bit fields) */
            __m256i w0 = _mm256_and_si256(_mm256_srli_epi16(raw, 6), mask);
            __m256i w1 = _mm256_and_si256(_mm256_srli_epi16(raw, 4), mask);
            __m256i w2 = _mm256_and_si256(_mm256_srli_epi16(raw, 2), mask);
            __m256i w3 = _mm256_and_si256(raw, mask);

            /* Load 32 int8 activation values */
            __m256i act = _mm256_loadu_si256((const __m256i *)(py + col));

            /* maddubs: first arg unsigned (weight codes 0,1,2), second signed (int8 activation)
             * Computes: pairs of u8*s8 summed into s16 */
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

        /* Horizontal sum for each row accumulator */
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

        /* Scalar tail */
        for (; col < n; col++) {
            uint8_t byte = px[col];
            int8_t a = py[col];
            s0 += ((byte >> 6) & 0x3) * a;
            s1 += ((byte >> 4) & 0x3) * a;
            s2 += ((byte >> 2) & 0x3) * a;
            s3 += ((byte >> 0) & 0x3) * a;
        }

        /* Bias correction: code = val + 1, so subtract sum_qv per row */
        out[rg * 4 + 0] = (float)(s0 - sum_qv) * inv_act_scale;
        out[rg * 4 + 1] = (float)(s1 - sum_qv) * inv_act_scale;
        out[rg * 4 + 2] = (float)(s2 - sum_qv) * inv_act_scale;
        out[rg * 4 + 3] = (float)(s3 - sum_qv) * inv_act_scale;
    }
}

#else
/* Scalar fallback */

void bitnet_ternary_gemv(
    const uint8_t *packed,
    const int8_t  *qv,
    float         *out,
    int            n,
    float          inv_act_scale
)
{
    int sum_qv = 0;
    for (int i = 0; i < n; i++) {
        sum_qv += qv[i];
    }

    int n_row_groups = n / 4;

    for (int rg = 0; rg < n_row_groups; rg++) {
        const uint8_t *px = packed + rg * n;
        int s0 = 0, s1 = 0, s2 = 0, s3 = 0;

        for (int col = 0; col < n; col++) {
            uint8_t byte = px[col];
            int8_t a = qv[col];
            s0 += ((byte >> 6) & 0x3) * a;
            s1 += ((byte >> 4) & 0x3) * a;
            s2 += ((byte >> 2) & 0x3) * a;
            s3 += ((byte >> 0) & 0x3) * a;
        }

        out[rg * 4 + 0] = (float)(s0 - sum_qv) * inv_act_scale;
        out[rg * 4 + 1] = (float)(s1 - sum_qv) * inv_act_scale;
        out[rg * 4 + 2] = (float)(s2 - sum_qv) * inv_act_scale;
        out[rg * 4 + 3] = (float)(s3 - sum_qv) * inv_act_scale;
    }
}

#endif
