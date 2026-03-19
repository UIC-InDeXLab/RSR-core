/*
 * RSR (Row-Sort-Reduce) binary matrix GEMV kernel.
 *
 * Inference path:
 *   1. Gather: collect input vector elements via precomputed permutation
 *   2. Aggregate: sum gathered elements within each unique pattern group
 *   3. Scatter: distribute aggregated sums to output rows via predecoded indices
 *
 * Preprocessing (sorting, grouping, bit decoding) is done in Python.
 */

#include <stdint.h>
#include <string.h>

#if defined(__AVX2__)
#include <immintrin.h>
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

/*
 * rsr_gemv — RSR binary matrix-vector multiply (inference only).
 *
 * Parameters:
 *   perms            [num_blocks * n]      int32  — permutation indices per block
 *   group_ends       [total_groups]        int32  — cumulative group sizes within each block
 *   scatter_offsets  [total_groups + 1]    int32  — prefix offsets into scatter_rows
 *   scatter_rows     [total_set_bits]      uint8  — output row indices to update per group
 *   block_meta       [num_blocks * 2]      int32  — (group_array_offset, num_groups) per block
 *   v                [n]                   float  — input vector
 *   out              [n]                   float  — output vector
 *   n                matrix dimension (square n x n)
 *   k                block height (rows per block, must be <= 64)
 *   num_blocks       n / k
 */
void rsr_gemv(
    const int32_t *restrict perms,
    const int32_t *restrict group_ends,
    const int32_t *restrict scatter_offsets,
    const uint8_t *restrict scatter_rows,
    const int32_t *restrict block_meta,
    const float *restrict v,
    float *restrict out,
    int n,
    int k,
    int num_blocks
) {
    for (int b = 0; b < num_blocks; b++) {
        const int32_t *perm = perms + (size_t)b * n;
        const int g_off = block_meta[b * 2];
        const int n_groups = block_meta[b * 2 + 1];
        float *bout = out + (size_t)b * k;

        memset(bout, 0, (size_t)k * sizeof(float));

        int start = 0;
        for (int g = 0; g < n_groups; g++) {
            const int gg = g_off + g;
            const int end = group_ends[gg];
            const int len = end - start;

            float agg = 0.0f;
            if (__builtin_expect(len <= 4, 1)) {
                switch (len) {
                    case 4:
                        agg += v[perm[start + 3]];
                        /* fallthrough */
                    case 3:
                        agg += v[perm[start + 2]];
                        /* fallthrough */
                    case 2:
                        agg += v[perm[start + 1]];
                        /* fallthrough */
                    case 1:
                        agg += v[perm[start]];
                        break;
                    default:
                        break;
                }
            } else {
                int i = start;
#if defined(__AVX2__)
                if (len >= 32) {
                    __m256 va0 = _mm256_setzero_ps();
                    __m256 va1 = _mm256_setzero_ps();
                    __m256 va2 = _mm256_setzero_ps();
                    __m256 va3 = _mm256_setzero_ps();

                    for (; i + 31 < end; i += 32) {
                        va0 = _mm256_add_ps(va0, _mm256_i32gather_ps(v,
                            _mm256_loadu_si256((const __m256i *)(perm + i)), 4));
                        va1 = _mm256_add_ps(va1, _mm256_i32gather_ps(v,
                            _mm256_loadu_si256((const __m256i *)(perm + i + 8)), 4));
                        va2 = _mm256_add_ps(va2, _mm256_i32gather_ps(v,
                            _mm256_loadu_si256((const __m256i *)(perm + i + 16)), 4));
                        va3 = _mm256_add_ps(va3, _mm256_i32gather_ps(v,
                            _mm256_loadu_si256((const __m256i *)(perm + i + 24)), 4));
                    }

                    va0 = _mm256_add_ps(_mm256_add_ps(va0, va1),
                                        _mm256_add_ps(va2, va3));

                    for (; i + 7 < end; i += 8) {
                        va0 = _mm256_add_ps(va0, _mm256_i32gather_ps(v,
                            _mm256_loadu_si256((const __m256i *)(perm + i)), 4));
                    }
                    agg += hsum256_ps(va0);
                }
#endif
                float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
                for (; i + 3 < end; i += 4) {
                    s0 += v[perm[i]];
                    s1 += v[perm[i + 1]];
                    s2 += v[perm[i + 2]];
                    s3 += v[perm[i + 3]];
                }
                agg += (s0 + s1) + (s2 + s3);
                for (; i < end; i++) {
                    agg += v[perm[i]];
                }
            }

            const int s_begin = scatter_offsets[gg];
            const int s_end = scatter_offsets[gg + 1];
            for (int s = s_begin; s < s_end; s++) {
                bout[(int)scatter_rows[s]] += agg;
            }

            start = end;
        }
    }
}

