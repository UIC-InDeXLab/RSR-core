/*
 * RSR Ternary Inference Kernel v3.0
 *
 * Idea:
 *   - Keep the two-pass layout for large-group workloads.
 *   - Replace per-call scratch allocation with a persistent thread-local buffer.
 *   - Use scalar-unrolled gather for the permutation pass and contiguous AVX2
 *     reduction for the aggregation pass.
 */

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

static _Thread_local float *tls_v_perm = NULL;
static _Thread_local size_t tls_v_perm_cap = 0;

static inline float *ensure_tls_v_perm(size_t n)
{
    if (tls_v_perm_cap >= n) {
        return tls_v_perm;
    }

    const size_t bytes = ((n * sizeof(float)) + 63UL) & ~63UL;
    void *new_buf = NULL;
    if (posix_memalign(&new_buf, 64, bytes) != 0) {
        return NULL;
    }

    free(tls_v_perm);
    tls_v_perm = (float *)new_buf;
    tls_v_perm_cap = bytes / sizeof(float);
    return tls_v_perm;
}

#if defined(__AVX2__)
static inline float hsum256_ps(__m256 x)
{
    __m128 lo = _mm256_castps256_ps128(x);
    __m128 hi = _mm256_extractf128_ps(x, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(sum);
    sum = _mm_add_ps(sum, shuf);
    shuf = _mm_movehl_ps(shuf, sum);
    sum = _mm_add_ss(sum, shuf);
    return _mm_cvtss_f32(sum);
}
#endif

void rsr_ternary_gemv_v3_0(
    const int32_t *restrict perms,
    const int32_t *restrict group_ends,
    const int32_t *restrict scatter_offsets,
    const int8_t  *restrict scatter_rows,
    const int8_t  *restrict scatter_signs,
    const int32_t *restrict block_meta,
    const float   *restrict v,
    float         *restrict out,
    int            n,
    int            k,
    int            num_blocks
)
{
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int b = 0; b < num_blocks; b++) {
        float *v_perm = ensure_tls_v_perm((size_t)n);

        const int32_t *perm = perms + (size_t)b * (size_t)n;
        const int ge_off = block_meta[2 * b];
        const int ge_end = (b + 1 < num_blocks) ? block_meta[2 * (b + 1)] : (ge_off + n);
        float *bout = out + (size_t)b * (size_t)k;

        memset(bout, 0, (size_t)k * sizeof(float));

        if (v_perm != NULL) {
            int i = 0;
            const int pf_dist = 64;

            for (; i + 7 < n; i += 8) {
#if defined(__AVX2__)
                if (i + pf_dist + 7 < n) {
                    _mm_prefetch((const char *)&v[perm[i + pf_dist + 0]], _MM_HINT_T0);
                    _mm_prefetch((const char *)&v[perm[i + pf_dist + 2]], _MM_HINT_T0);
                    _mm_prefetch((const char *)&v[perm[i + pf_dist + 4]], _MM_HINT_T0);
                    _mm_prefetch((const char *)&v[perm[i + pf_dist + 6]], _MM_HINT_T0);
                }
#endif
                v_perm[i + 0] = v[perm[i + 0]];
                v_perm[i + 1] = v[perm[i + 1]];
                v_perm[i + 2] = v[perm[i + 2]];
                v_perm[i + 3] = v[perm[i + 3]];
                v_perm[i + 4] = v[perm[i + 4]];
                v_perm[i + 5] = v[perm[i + 5]];
                v_perm[i + 6] = v[perm[i + 6]];
                v_perm[i + 7] = v[perm[i + 7]];
            }
            for (; i < n; i++) {
                v_perm[i] = v[perm[i]];
            }
        }

        int start = 0;
        for (int g = ge_off; g < ge_end && start < n; g++) {
            const int end = group_ends[g];
            float agg = 0.0f;

            if (v_perm != NULL) {
                int j = start;
#if defined(__AVX2__)
                if (end - start >= 32) {
                    __m256 va0 = _mm256_setzero_ps();
                    __m256 va1 = _mm256_setzero_ps();
                    __m256 va2 = _mm256_setzero_ps();
                    __m256 va3 = _mm256_setzero_ps();

                    for (; j + 31 < end; j += 32) {
                        va0 = _mm256_add_ps(va0, _mm256_loadu_ps(v_perm + j));
                        va1 = _mm256_add_ps(va1, _mm256_loadu_ps(v_perm + j + 8));
                        va2 = _mm256_add_ps(va2, _mm256_loadu_ps(v_perm + j + 16));
                        va3 = _mm256_add_ps(va3, _mm256_loadu_ps(v_perm + j + 24));
                    }
                    va0 = _mm256_add_ps(_mm256_add_ps(va0, va1), _mm256_add_ps(va2, va3));

                    for (; j + 7 < end; j += 8) {
                        va0 = _mm256_add_ps(va0, _mm256_loadu_ps(v_perm + j));
                    }
                    agg += hsum256_ps(va0);
                }
#endif
                float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
                for (; j + 3 < end; j += 4) {
                    s0 += v_perm[j + 0];
                    s1 += v_perm[j + 1];
                    s2 += v_perm[j + 2];
                    s3 += v_perm[j + 3];
                }
                agg += (s0 + s1) + (s2 + s3);
                for (; j < end; j++) {
                    agg += v_perm[j];
                }
            } else {
                int i = start;
                float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
                for (; i + 7 < end; i += 8) {
                    s0 += v[perm[i + 0]] + v[perm[i + 1]];
                    s1 += v[perm[i + 2]] + v[perm[i + 3]];
                    s2 += v[perm[i + 4]] + v[perm[i + 5]];
                    s3 += v[perm[i + 6]] + v[perm[i + 7]];
                }
                agg += (s0 + s1) + (s2 + s3);
                for (; i < end; i++) {
                    agg += v[perm[i]];
                }
            }

            const int s_begin = scatter_offsets[g];
            const int s_end = scatter_offsets[g + 1];
            for (int s = s_begin; s < s_end; s++) {
                bout[(int)scatter_rows[s]] += (float)scatter_signs[s] * agg;
            }

            start = end;
        }
    }
}
