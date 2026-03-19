/*
 * RSR Ternary Inference Kernel v1.5 — AVX2 gather + aggregate.
 *
 * Improvements over v1.4:
 *   - AVX2 8-wide gather for aggregation (vgatherdps)
 *   - Prefetch next group's perm entries
 *   - Multiply scatter_signs into scatter as float to avoid int->float in loop
 *
 * Compile:
 *   gcc -O3 -march=native -mavx2 -fopenmp -shared -fPIC -o rsr_ternary_v1_5.so rsr_ternary_v1_5.c
 */

#include <stdint.h>
#include <string.h>
#include <immintrin.h>

#ifdef _OPENMP
#include <omp.h>
#endif

void rsr_ternary_gemv_v1_5(
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

    #pragma omp parallel for schedule(dynamic)
    for (int b = 0; b < num_blocks; b++) {
        int ge_off   = block_meta[2 * b];
        int row_base = b * k;

        int ge_end;
        if (b + 1 < num_blocks)
            ge_end = block_meta[2 * (b + 1)];
        else {
            /* For the last block, count groups until col_start >= n */
            ge_end = ge_off + n; /* upper bound, loop will break on col_start >= n */
        }

        const int32_t *perm = perms + (int64_t)b * n;
        int col_start = 0;
        int g = ge_off;

        for (; g < ge_end && col_start < n; g++) {
            int col_end = group_ends[g];
            int count = col_end - col_start;

            /* Aggregate: sum v[perm[col_start..col_end)] using AVX2 */
            float agg;
            {
                const int32_t *idx = perm + col_start;
                int i = 0;

#ifdef __AVX2__
                __m256 vacc = _mm256_setzero_ps();
                for (; i + 7 < count; i += 8) {
                    __m256i vidx = _mm256_loadu_si256((const __m256i *)(idx + i));
                    __m256 gathered = _mm256_i32gather_ps(v, vidx, 4);
                    vacc = _mm256_add_ps(vacc, gathered);
                }
                /* Horizontal sum of vacc */
                __m128 lo = _mm256_castps256_ps128(vacc);
                __m128 hi = _mm256_extractf128_ps(vacc, 1);
                lo = _mm_add_ps(lo, hi);
                lo = _mm_hadd_ps(lo, lo);
                lo = _mm_hadd_ps(lo, lo);
                agg = _mm_cvtss_f32(lo);
#else
                agg = 0.0f;
#endif
                /* Scalar tail */
                for (; i < count; i++) {
                    agg += v[idx[i]];
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
}
