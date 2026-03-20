/*
 * RSR Ternary Inference Kernel v2.0
 *
 * Idea:
 *   - Eliminate the temporary v_perm buffer entirely.
 *   - Use static scheduling and per-block zeroing to avoid false sharing.
 *   - Favor scalar, unrolled gathers because ternary RSR groups are usually tiny.
 *
 * Compile:
 *   gcc -O3 -march=native -mavx2 -fopenmp -shared -fPIC -o rsr_ternary_v2_0.so rsr_ternary_v2_0.c
 */

#include <stdint.h>
#include <string.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

void rsr_ternary_gemv_v2_0(
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
        (void)n;
        const int32_t *perm = perms + (size_t)b * (size_t)n;
        const int ge_off = block_meta[2 * b];
        const int ge_end = (b + 1 < num_blocks) ? block_meta[2 * (b + 1)] : (ge_off + n);
        float *bout = out + (size_t)b * (size_t)k;

        memset(bout, 0, (size_t)k * sizeof(float));

        int start = 0;
        for (int g = ge_off; g < ge_end && start < n; g++) {
            const int end = group_ends[g];
            const int len = end - start;
            float agg = 0.0f;

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
                int i = start;
                const int pf_dist = 32;
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

            const int s_begin = scatter_offsets[g];
            const int s_end = scatter_offsets[g + 1];
            for (int s = s_begin; s < s_end; s++) {
                bout[(int)scatter_rows[s]] += (float)scatter_signs[s] * agg;
            }

            start = end;
        }
    }
}
