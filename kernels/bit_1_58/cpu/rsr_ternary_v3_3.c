/*
 * RSR Ternary Inference Kernel v3.3
 *
 * Idea:
 *   - Build on v3.1's 16-bit direct-gather path.
 *   - Replace variable-length scatter lists with fixed-size positive/negative
 *     row masks, cutting hot-loop metadata traffic to two uint16 loads.
 */

#include <stdint.h>
#include <string.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

void rsr_ternary_gemv_v3_3(
    const uint16_t *restrict perms,
    const uint16_t *restrict group_ends,
    const uint16_t *restrict pos_masks,
    const uint16_t *restrict neg_masks,
    const int32_t  *restrict block_meta,
    const float    *restrict v,
    float          *restrict out,
    int             n,
    int             k,
    int             num_blocks
)
{
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int b = 0; b < num_blocks; b++) {
        const uint16_t *perm = perms + (size_t)b * (size_t)n;
        const int ge_off = block_meta[2 * b];
        const int ge_end = (b + 1 < num_blocks) ? block_meta[2 * (b + 1)] : (ge_off + n);
        float *bout = out + (size_t)b * (size_t)k;

        memset(bout, 0, (size_t)k * sizeof(float));

        int start = 0;
        for (int g = ge_off; g < ge_end && start < n; g++) {
            const int end = (int)group_ends[g];
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
                        agg += v[perm[start + 0]];
                    default:
                        break;
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
                for (; i < end; i++) {
                    agg += v[perm[i]];
                }
            }

            uint16_t pos_mask = pos_masks[g];
            uint16_t neg_mask = neg_masks[g];

            while (pos_mask != 0) {
                const unsigned row = (unsigned)__builtin_ctz((unsigned)pos_mask);
                bout[row] += agg;
                pos_mask &= (uint16_t)(pos_mask - 1);
            }
            while (neg_mask != 0) {
                const unsigned row = (unsigned)__builtin_ctz((unsigned)neg_mask);
                bout[row] -= agg;
                neg_mask &= (uint16_t)(neg_mask - 1);
            }

            start = end;
        }
    }
}
