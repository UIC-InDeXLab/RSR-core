/*
 * RSR Ternary Preprocessing Kernel
 *
 * Takes a ternary matrix M (stored as int8 with values {-1, 0, +1}),
 * decomposes into M_pos and M_neg binary matrices, and runs counting-sort
 * based preprocessing for each.
 *
 * For each block b:
 *   - Encodes columns of M_pos as k-bit integers (pos_pattern)
 *   - Encodes columns of M_neg as k-bit integers (neg_pattern)
 *   - Combines into 2k-bit ternary pattern: (pos_pattern << k) | neg_pattern
 *   - Counting-sort columns by ternary pattern
 *   - Outputs: perms, group_ends, scatter_offsets, scatter_rows, scatter_signs
 *
 * Compile:
 *   gcc -O3 -march=native -fopenmp -shared -fPIC -o rsr_ternary_prep.so rsr_ternary_prep.c
 */

#include <stdint.h>
#include <string.h>
#include <stdlib.h>

void rsr_ternary_prep(
    const int8_t  *M,               /* n x n ternary matrix, row-major, values {-1,0,+1} */
    int            n,
    int            k,
    int32_t       *perms,            /* out: num_blocks * n */
    int32_t       *group_ends,       /* out: up to num_blocks * max_groups */
    int32_t       *scatter_offsets,   /* out: total_groups + 1 */
    int8_t        *scatter_rows,     /* out: row indices for scatter */
    int8_t        *scatter_signs,    /* out: +1 or -1 for each scatter entry */
    int32_t       *block_meta,       /* out: 2 * num_blocks (group_ends_off, scatter_off) */
    int32_t       *out_sizes         /* out: [total_groups, total_scatter] */
)
{
    int num_blocks = n / k;
    int num_buckets = 1;
    /* num_buckets = 4^k = (1 << 2k), but capped for sanity */
    if (2 * k <= 24) {
        num_buckets = 1 << (2 * k);
    } else {
        /* For large k, use hash-based approach. For now, limit to k <= 12. */
        num_buckets = 1 << (2 * k);
    }

    int total_groups = 0;
    int total_scatter = 0;

    /* Temp buffer for counting sort */
    int32_t *counts = (int32_t *)calloc(num_buckets, sizeof(int32_t));
    int32_t *col_values = (int32_t *)malloc(n * sizeof(int32_t));

    for (int b = 0; b < num_blocks; b++) {
        int row_start = b * k;

        /* Step 1: Encode columns as 2k-bit integers */
        for (int j = 0; j < n; j++) {
            int32_t pos_val = 0;
            int32_t neg_val = 0;
            for (int i = 0; i < k; i++) {
                int8_t elem = M[(row_start + i) * n + j];
                if (elem == 1) {
                    pos_val |= (1 << (k - 1 - i));
                } else if (elem == -1) {
                    neg_val |= (1 << (k - 1 - i));
                }
            }
            col_values[j] = (pos_val << k) | neg_val;
        }

        /* Step 2: Counting sort */
        memset(counts, 0, num_buckets * sizeof(int32_t));
        for (int j = 0; j < n; j++) {
            counts[col_values[j]]++;
        }

        /* Prefix sum */
        int32_t *offsets = (int32_t *)malloc(num_buckets * sizeof(int32_t));
        offsets[0] = 0;
        for (int i = 1; i < num_buckets; i++) {
            offsets[i] = offsets[i-1] + counts[i-1];
        }

        /* Place columns in sorted order */
        int32_t *perm = perms + b * n;
        int32_t *tmp_off = (int32_t *)malloc(num_buckets * sizeof(int32_t));
        memcpy(tmp_off, offsets, num_buckets * sizeof(int32_t));
        for (int j = 0; j < n; j++) {
            int32_t bucket = col_values[j];
            perm[tmp_off[bucket]++] = j;
        }

        /* Step 3: Find non-empty groups */
        block_meta[2 * b] = total_groups;
        block_meta[2 * b + 1] = total_scatter;

        int32_t cum = 0;
        for (int bucket = 0; bucket < num_buckets; bucket++) {
            if (counts[bucket] == 0) continue;
            cum += counts[bucket];
            group_ends[total_groups] = cum;

            /* Step 4: Decode pattern to scatter rows + signs */
            scatter_offsets[total_groups] = total_scatter;

            int pos_pattern = bucket >> k;
            int neg_pattern = bucket & ((1 << k) - 1);

            /* Positive scatter entries */
            for (int bit = 0; bit < k; bit++) {
                if (pos_pattern & (1 << (k - 1 - bit))) {
                    scatter_rows[total_scatter] = (int8_t)bit;
                    scatter_signs[total_scatter] = 1;
                    total_scatter++;
                }
            }
            /* Negative scatter entries */
            for (int bit = 0; bit < k; bit++) {
                if (neg_pattern & (1 << (k - 1 - bit))) {
                    scatter_rows[total_scatter] = (int8_t)bit;
                    scatter_signs[total_scatter] = -1;
                    total_scatter++;
                }
            }

            total_groups++;
        }
        scatter_offsets[total_groups] = total_scatter;

        free(offsets);
        free(tmp_off);
    }

    out_sizes[0] = total_groups;
    out_sizes[1] = total_scatter;

    free(counts);
    free(col_values);
}
