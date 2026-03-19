/*
 * RSR GEMV kernel v4.2
 *
 * Idea:
 *   - Eliminate the temporary v_perm buffer entirely.
 *   - Do gather + aggregate in one pass per group, then scatter.
 *   - Use scalar-unrolled gather and software prefetch (AVX2 gather is often
 *     slower on random accesses for this workload).
 */

#include <stdint.h>
#include <string.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

void rsr_gemv_v4_2(
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
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
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
            int i = start;

            if (__builtin_expect(len <= 4, 1)) {
                switch (len) {
                    case 4:
                        agg += v[perm[start + 3]];
                    case 3:
                        agg += v[perm[start + 2]];
                    case 2:
                        agg += v[perm[start + 1]];
                    case 1:
                        agg += v[perm[start]];
                    default:
                        break;
                }
            } else {
                const int pf_dist = 64;
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
