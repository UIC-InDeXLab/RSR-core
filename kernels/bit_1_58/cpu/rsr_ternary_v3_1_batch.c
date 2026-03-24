/*
 * RSR Ternary Inference Kernel v3.1 — Batched + Fused act_quant
 *
 * Two public entry-points:
 *
 *   rsr_ternary_gemv_v3_1_fused
 *       Single-layer GEMV with act_quant fused in.
 *
 *   rsr_ternary_gemv_v3_1_batch_fused
 *       Multi-layer GEMV for layers that share the same input vector.
 *       act_quant is applied once; all layers' GEMVs run in one call.
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

/* ------------------------------------------------------------------ */
/*  act_quant — matches Python _bitnet_act_quant for a 1-D vector     */
/* ------------------------------------------------------------------ */
static void act_quant_f32(const float *restrict src,
                          float       *restrict dst,
                          int n)
{
    float abs_max = 0.0f;
#if defined(__AVX2__)
    {
        __m256 vmax = _mm256_setzero_ps();
        const __m256 sign_mask = _mm256_set1_ps(-0.0f);
        int i = 0;
        for (; i + 7 < n; i += 8) {
            __m256 v = _mm256_loadu_ps(src + i);
            v = _mm256_andnot_ps(sign_mask, v);
            vmax = _mm256_max_ps(vmax, v);
        }
        __m128 hi = _mm256_extractf128_ps(vmax, 1);
        __m128 lo = _mm256_castps256_ps128(vmax);
        __m128 m  = _mm_max_ps(lo, hi);
        m = _mm_max_ps(m, _mm_movehl_ps(m, m));
        m = _mm_max_ss(m, _mm_movehdup_ps(m));
        abs_max = _mm_cvtss_f32(m);
        for (; i < n; i++) {
            float a = fabsf(src[i]);
            if (a > abs_max) abs_max = a;
        }
    }
#else
    for (int i = 0; i < n; i++) {
        float a = fabsf(src[i]);
        if (a > abs_max) abs_max = a;
    }
#endif
    if (abs_max < 1e-5f) abs_max = 1e-5f;
    const float scale     = 127.0f / abs_max;
    const float inv_scale = abs_max / 127.0f;

#if defined(__AVX2__)
    {
        const __m256 vs  = _mm256_set1_ps(scale);
        const __m256 vis = _mm256_set1_ps(inv_scale);
        const __m256 lo  = _mm256_set1_ps(-128.0f);
        const __m256 hi  = _mm256_set1_ps( 127.0f);
        int i = 0;
        for (; i + 7 < n; i += 8) {
            __m256 v = _mm256_loadu_ps(src + i);
            v = _mm256_mul_ps(v, vs);
            v = _mm256_round_ps(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            v = _mm256_max_ps(v, lo);
            v = _mm256_min_ps(v, hi);
            v = _mm256_mul_ps(v, vis);
            _mm256_storeu_ps(dst + i, v);
        }
        for (; i < n; i++) {
            float q = roundf(src[i] * scale);
            if (q < -128.0f) q = -128.0f;
            else if (q >  127.0f) q =  127.0f;
            dst[i] = q * inv_scale;
        }
    }
#else
    for (int i = 0; i < n; i++) {
        float q = roundf(src[i] * scale);
        if (q < -128.0f) q = -128.0f;
        else if (q >  127.0f) q =  127.0f;
        dst[i] = q * inv_scale;
    }
#endif
}

/* ------------------------------------------------------------------ */
/*  Core GEMV for one block — v3.1 scatter logic                      */
/* ------------------------------------------------------------------ */
static inline void gemv_block_v31(
    const uint16_t *restrict perm,
    const uint16_t *restrict group_ends,
    const int32_t  *restrict scatter_offsets,
    const int8_t   *restrict scatter_rows,
    const int8_t   *restrict scatter_signs,
    int ge_off, int ge_end,
    const float *restrict v,
    float *restrict bout,
    int n_cols, int k)
{
    memset(bout, 0, (size_t)k * sizeof(float));

    int start = 0;
    for (int g = ge_off; g < ge_end && start < n_cols; g++) {
        const int end = (int)group_ends[g];
        const int len = end - start;
        float agg = 0.0f;

        if (__builtin_expect(len <= 4, 1)) {
            switch (len) {
                case 4: agg += v[perm[start + 3]]; /* fall through */
                case 3: agg += v[perm[start + 2]]; /* fall through */
                case 2: agg += v[perm[start + 1]]; /* fall through */
                case 1: agg += v[perm[start + 0]]; /* fall through */
                default: break;
            }
        } else {
            int i = start;
            const int pf_dist = 48;
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

            for (; i + 7 < end; i += 8) {
#if defined(__AVX2__)
                if (i + pf_dist + 7 < end) {
                    _mm_prefetch((const char *)&v[perm[i + pf_dist + 0]], _MM_HINT_T0);
                    _mm_prefetch((const char *)&v[perm[i + pf_dist + 2]], _MM_HINT_T0);
                    _mm_prefetch((const char *)&v[perm[i + pf_dist + 4]], _MM_HINT_T0);
                    _mm_prefetch((const char *)&v[perm[i + pf_dist + 6]], _MM_HINT_T0);
                }
#endif
                s0 += v[perm[i + 0]] + v[perm[i + 1]];
                s1 += v[perm[i + 2]] + v[perm[i + 3]];
                s2 += v[perm[i + 4]] + v[perm[i + 5]];
                s3 += v[perm[i + 6]] + v[perm[i + 7]];
            }
            agg += (s0 + s1) + (s2 + s3);
            for (; i < end; i++)
                agg += v[perm[i]];
        }

        const int s_begin = scatter_offsets[g];
        const int s_end = scatter_offsets[g + 1];
        for (int s = s_begin; s < s_end; s++) {
            bout[(int)scatter_rows[s]] += (float)scatter_signs[s] * agg;
        }

        start = end;
    }
}

/* ================================================================== */
/*  PUBLIC: single-layer fused act_quant + GEMV (v3.1)                */
/* ================================================================== */
void rsr_ternary_gemv_v3_1_fused(
    const uint16_t *restrict perms,
    const uint16_t *restrict group_ends,
    const int32_t  *restrict scatter_offsets,
    const int8_t   *restrict scatter_rows,
    const int8_t   *restrict scatter_signs,
    const int32_t  *restrict block_meta,
    const float    *restrict v_raw,
    float          *restrict out,
    float          *restrict v_scratch,
    int             n_cols,
    int             k,
    int             num_blocks)
{
    act_quant_f32(v_raw, v_scratch, n_cols);

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int b = 0; b < num_blocks; b++) {
        const uint16_t *perm = perms + (size_t)b * (size_t)n_cols;
        const int ge_off = block_meta[2 * b];
        const int ge_end = (b + 1 < num_blocks)
                               ? block_meta[2 * (b + 1)]
                               : (ge_off + n_cols);
        float *bout = out + (size_t)b * (size_t)k;
        gemv_block_v31(perm, group_ends, scatter_offsets,
                       scatter_rows, scatter_signs,
                       ge_off, ge_end, v_scratch, bout, n_cols, k);
    }
}

/* ================================================================== */
/*  PUBLIC: multi-layer batched fused act_quant + GEMV (v3.1)         */
/* ================================================================== */
void rsr_ternary_gemv_v3_1_batch_fused(
    int              num_layers,
    const uint16_t **perms_arr,
    const uint16_t **group_ends_arr,
    const int32_t  **scatter_offsets_arr,
    const int8_t   **scatter_rows_arr,
    const int8_t   **scatter_signs_arr,
    const int32_t  **block_meta_arr,
    const int32_t   *k_arr,
    const int32_t   *num_blocks_arr,
    const float     *v_raw,
    float          **out_arr,
    float           *v_scratch,
    int              n_cols)
{
    /* 1. act_quant once */
    act_quant_f32(v_raw, v_scratch, n_cols);

    /* 2. Build flat index across all layers' blocks */
    int total_blocks = 0;
    int *block_offsets = (int *)alloca((size_t)(num_layers + 1) * sizeof(int));
    block_offsets[0] = 0;
    for (int l = 0; l < num_layers; l++) {
        block_offsets[l + 1] = block_offsets[l] + num_blocks_arr[l];
    }
    total_blocks = block_offsets[num_layers];

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int flat_b = 0; flat_b < total_blocks; flat_b++) {
        int l = 0;
        while (l + 1 < num_layers && flat_b >= block_offsets[l + 1])
            l++;
        const int b = flat_b - block_offsets[l];

        const int            kl  = k_arr[l];
        const int            nb  = num_blocks_arr[l];
        const uint16_t      *perm = perms_arr[l] + (size_t)b * (size_t)n_cols;
        const int32_t       *bm  = block_meta_arr[l];
        const int ge_off = bm[2 * b];
        const int ge_end = (b + 1 < nb) ? bm[2 * (b + 1)] : (ge_off + n_cols);
        float *bout = out_arr[l] + (size_t)b * (size_t)kl;

        gemv_block_v31(perm, group_ends_arr[l],
                       scatter_offsets_arr[l],
                       scatter_rows_arr[l],
                       scatter_signs_arr[l],
                       ge_off, ge_end, v_scratch, bout, n_cols, kl);
    }
}
