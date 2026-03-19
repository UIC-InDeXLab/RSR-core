/*
 * RSR Ternary Inference Kernel v1.6 — two-pass: permute then linear scan.
 *
 * Improvements over v1.5:
 *   - Pass 1: create v_perm[b] = v[perm[b]] (sequential write, random read)
 *   - Pass 2: aggregate from v_perm with sequential reads (cache-friendly)
 *   - AVX2 for both passes
 *   - Aligned temporary buffer for v_perm
 *
 * Compile:
 *   gcc -O3 -march=native -mavx2 -fopenmp -shared -fPIC -o rsr_ternary_v1_6.so rsr_ternary_v1_6.c
 */

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <immintrin.h>

#ifdef _OPENMP
#include <omp.h>
#endif

void rsr_ternary_gemv_v1_6(
    const int32_t *perms,
    const int32_t *group_ends,
    const int32_t *scatter_offsets,
    const int8_t  *scatter_rows,
    const int8_t  *scatter_signs,
    const int32_t *block_meta,
    const float   *v,
    float         *out,
    int            n,
    int            k,
    int            num_blocks
)
{
    memset(out, 0, n * sizeof(float));

    /* Aligned buffer size: n floats per thread */
    int n_aligned = (n + 7) & ~7;  /* round up to multiple of 8 */

    #pragma omp parallel
    {
        /* Per-thread aligned v_perm buffer */
        float *v_perm = (float *)aligned_alloc(32, n_aligned * sizeof(float));

        #pragma omp for schedule(dynamic)
        for (int b = 0; b < num_blocks; b++) {
            int ge_off   = block_meta[2 * b];
            int row_base = b * k;

            int ge_end;
            if (b + 1 < num_blocks)
                ge_end = block_meta[2 * (b + 1)];
            else
                ge_end = ge_off + n;  /* upper bound */

            const int32_t *perm = perms + (int64_t)b * n;

            /* Pass 1: Permute v into contiguous v_perm */
            int j = 0;
#ifdef __AVX2__
            for (; j + 7 < n; j += 8) {
                __m256i vidx = _mm256_loadu_si256((const __m256i *)(perm + j));
                __m256 gathered = _mm256_i32gather_ps(v, vidx, 4);
                _mm256_store_ps(v_perm + j, gathered);
            }
#endif
            for (; j < n; j++) {
                v_perm[j] = v[perm[j]];
            }

            /* Pass 2: Aggregate + scatter from contiguous v_perm */
            int col_start = 0;
            int g = ge_off;

            for (; g < ge_end && col_start < n; g++) {
                int col_end = group_ends[g];
                int count = col_end - col_start;

                /* Sequential sum from v_perm */
                float agg;
                {
                    const float *src = v_perm + col_start;
                    int i = 0;
#ifdef __AVX2__
                    __m256 vacc = _mm256_setzero_ps();
                    for (; i + 7 < count; i += 8) {
                        __m256 chunk = _mm256_load_ps(src + i);
                        vacc = _mm256_add_ps(vacc, chunk);
                    }
                    __m128 lo = _mm256_castps256_ps128(vacc);
                    __m128 hi = _mm256_extractf128_ps(vacc, 1);
                    lo = _mm_add_ps(lo, hi);
                    lo = _mm_hadd_ps(lo, lo);
                    lo = _mm_hadd_ps(lo, lo);
                    agg = _mm_cvtss_f32(lo);
#else
                    agg = 0.0f;
#endif
                    for (; i < count; i++) {
                        agg += src[i];
                    }
                }

                /* Scatter with signs */
                int s_begin = scatter_offsets[g];
                int s_end   = scatter_offsets[g + 1];
                for (int s = s_begin; s < s_end; s++) {
                    int row = row_base + scatter_rows[s];
                    out[row] += (float)scatter_signs[s] * agg;
                }

                col_start = col_end;
            }
        }

        free(v_perm);
    }
}
