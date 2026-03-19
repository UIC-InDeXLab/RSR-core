/*
 * RSR Ternary Inference Kernel — fused gather + aggregate + signed scatter.
 *
 * For each block:
 *   1. For each group: gather v[perm[start..end]] and sum (aggregate)
 *   2. For each scatter entry: out[row] += sign * aggregated
 *
 * Compile:
 *   gcc -O3 -march=native -fopenmp -shared -fPIC -o rsr_ternary.so rsr_ternary.c
 */

#include <stdint.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

void rsr_ternary_gemv(
    const int32_t *perms,             /* num_blocks * n */
    const int32_t *group_ends,        /* total_groups */
    const int32_t *scatter_offsets,    /* total_groups + 1 */
    const int8_t  *scatter_rows,      /* total_scatter */
    const int8_t  *scatter_signs,     /* total_scatter: +1 or -1 */
    const int32_t *block_meta,        /* 2 * num_blocks */
    const float   *v,                 /* n */
    float         *out,               /* n */
    int            n,
    int            k,
    int            num_blocks
)
{
    memset(out, 0, n * sizeof(float));

    #pragma omp parallel for schedule(dynamic)
    for (int b = 0; b < num_blocks; b++) {
        int ge_off  = block_meta[2 * b];
        int sc_off  = block_meta[2 * b + 1];
        int row_base = b * k;

        /* Find this block's group range */
        int ge_end;
        if (b + 1 < num_blocks) {
            ge_end = block_meta[2 * (b + 1)];
        } else {
            /* Last block: need total_groups from scatter_offsets sentinel */
            /* We read until group_ends runs out; use a sentinel approach */
            /* Actually, we can compute from the next block or from total */
            ge_end = -1;  /* will be set below */
        }

        const int32_t *perm = perms + b * n;
        int col_start = 0;

        /* Iterate groups for this block */
        int g = ge_off;
        /* For last block, iterate until scatter_offsets[g+1] exists */
        for (;;) {
            /* Check if we've exhausted this block's groups */
            if (ge_end >= 0 && g >= ge_end) break;
            if (col_start >= n) break;

            int col_end = group_ends[g];

            /* Aggregate: sum v[perm[col_start..col_end)] */
            float agg = 0.0f;
            for (int j = col_start; j < col_end; j++) {
                agg += v[perm[j]];
            }

            /* Scatter with signs */
            int s_begin = scatter_offsets[g];
            int s_end   = scatter_offsets[g + 1];
            for (int s = s_begin; s < s_end; s++) {
                int row = row_base + scatter_rows[s];
                float contrib = scatter_signs[s] * agg;
                /* No race: each block writes to its own row range [b*k, (b+1)*k) */
                out[row] += contrib;
            }

            col_start = col_end;
            g++;
        }
    }
}
