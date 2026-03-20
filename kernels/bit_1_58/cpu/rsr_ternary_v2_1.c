/*
 * RSR Ternary Inference Kernel v2.1
 *
 * Idea:
 *   - Keep the two-pass strategy from v1.6 for workloads with larger groups.
 *   - Use static scheduling and per-block zeroing to reduce output contention.
 *   - Unroll gather and contiguous reduction more aggressively.
 *
 * Compile:
 *   gcc -O3 -march=native -mavx2 -fopenmp -shared -fPIC -o rsr_ternary_v2_1.so rsr_ternary_v2_1.c
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__AVX2__)
static inline float hsum256_ps(__m256 x) {
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

void rsr_ternary_gemv_v2_1(
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
    int num_threads = 1;
#ifdef _OPENMP
    num_threads = omp_get_max_threads();
#endif

    const size_t v_perm_bytes = (((size_t)n * sizeof(float)) + 63UL) & ~63UL;
    float **v_perm_bufs = (float **)malloc((size_t)num_threads * sizeof(float *));
    if (v_perm_bufs == NULL) {
        return;
    }

    for (int t = 0; t < num_threads; t++) {
        v_perm_bufs[t] = (float *)aligned_alloc(64, v_perm_bytes);
        if (v_perm_bufs[t] == NULL) {
            for (int i = 0; i < t; i++) {
                free(v_perm_bufs[i]);
            }
            free(v_perm_bufs);
            return;
        }
    }

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int b = 0; b < num_blocks; b++) {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        float *v_perm = v_perm_bufs[tid];

        const int32_t *perm = perms + (size_t)b * (size_t)n;
        const int ge_off = block_meta[2 * b];
        const int ge_end = (b + 1 < num_blocks) ? block_meta[2 * (b + 1)] : (ge_off + n);
        float *bout = out + (size_t)b * (size_t)k;

        memset(bout, 0, (size_t)k * sizeof(float));

        int i = 0;
#if defined(__AVX2__)
        for (; i + 15 < n; i += 16) {
            if (i + 64 < n) {
                _mm_prefetch((const char *)&v[perm[i + 32]], _MM_HINT_T0);
                _mm_prefetch((const char *)&v[perm[i + 40]], _MM_HINT_T0);
                _mm_prefetch((const char *)&v[perm[i + 48]], _MM_HINT_T0);
                _mm_prefetch((const char *)&v[perm[i + 56]], _MM_HINT_T0);
            }

            __m256i idx0 = _mm256_loadu_si256((const __m256i *)(perm + i));
            __m256i idx1 = _mm256_loadu_si256((const __m256i *)(perm + i + 8));
            __m256 g0 = _mm256_i32gather_ps(v, idx0, 4);
            __m256 g1 = _mm256_i32gather_ps(v, idx1, 4);
            _mm256_storeu_ps(v_perm + i, g0);
            _mm256_storeu_ps(v_perm + i + 8, g1);
        }
#endif
        for (; i < n; i++) {
            v_perm[i] = v[perm[i]];
        }

        int start = 0;
        for (int g = ge_off; g < ge_end && start < n; g++) {
            const int end = group_ends[g];
            float agg = 0.0f;
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
                s0 += v_perm[j];
                s1 += v_perm[j + 1];
                s2 += v_perm[j + 2];
                s3 += v_perm[j + 3];
            }
            agg += (s0 + s1) + (s2 + s3);
            for (; j < end; j++) {
                agg += v_perm[j];
            }

            const int s_begin = scatter_offsets[g];
            const int s_end = scatter_offsets[g + 1];
            for (int s = s_begin; s < s_end; s++) {
                bout[(int)scatter_rows[s]] += (float)scatter_signs[s] * agg;
            }

            start = end;
        }
    }

    for (int t = 0; t < num_threads; t++) {
        free(v_perm_bufs[t]);
    }
    free(v_perm_bufs);
}
