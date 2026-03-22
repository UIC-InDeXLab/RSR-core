/*
 * RSR Ternary Preprocessing Kernel — non-square matrices.
 *
 * Fork of rsr_ternary_prep.c with separate n_rows and n_cols parameters.
 * M is n_rows x n_cols, row-major, values {-1, 0, +1}.
 * num_blocks = n_rows / k. Each block has n_cols columns.
 *
 * Compile:
 *   gcc -O3 -march=native -fopenmp -shared -fPIC -o rsr_ternary_prep_nonsquare.so rsr_ternary_prep_nonsquare.c
 */

#include <stdint.h>
#include <string.h>
#include <stdlib.h>

void rsr_ternary_prep_nonsquare(
    const int8_t  *M,               /* n_rows x n_cols ternary matrix, row-major */
    int            n_rows,
    int            n_cols,
    int            k,
    int32_t       *perms,            /* out: num_blocks * n_cols */
    int32_t       *group_ends,       /* out: up to total_groups */
    int32_t       *scatter_offsets,   /* out: total_groups + 1 */
    int8_t        *scatter_rows,     /* out: row indices for scatter */
    int8_t        *scatter_signs,    /* out: +1 or -1 per scatter entry */
    int32_t       *block_meta,       /* out: 2 * num_blocks */
    int32_t       *out_sizes         /* out: [total_groups, total_scatter] */
)
{
    int num_blocks = n_rows / k;
    int num_buckets = 1;
    if (2 * k <= 24) {
        num_buckets = 1 << (2 * k);
    } else {
        num_buckets = 1 << (2 * k);
    }

    int total_groups = 0;
    int total_scatter = 0;

    int32_t *counts = (int32_t *)calloc(num_buckets, sizeof(int32_t));
    int32_t *col_values = (int32_t *)malloc(n_cols * sizeof(int32_t));

    for (int b = 0; b < num_blocks; b++) {
        int row_start = b * k;

        /* Step 1: Encode columns as 2k-bit integers */
        for (int j = 0; j < n_cols; j++) {
            int32_t pos_val = 0;
            int32_t neg_val = 0;
            for (int i = 0; i < k; i++) {
                int8_t elem = M[(int64_t)(row_start + i) * n_cols + j];
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
        for (int j = 0; j < n_cols; j++) {
            counts[col_values[j]]++;
        }

        /* Prefix sum */
        int32_t *offsets = (int32_t *)malloc(num_buckets * sizeof(int32_t));
        offsets[0] = 0;
        for (int i = 1; i < num_buckets; i++) {
            offsets[i] = offsets[i-1] + counts[i-1];
        }

        /* Place columns in sorted order */
        int32_t *perm = perms + (int64_t)b * n_cols;
        int32_t *tmp_off = (int32_t *)malloc(num_buckets * sizeof(int32_t));
        memcpy(tmp_off, offsets, num_buckets * sizeof(int32_t));
        for (int j = 0; j < n_cols; j++) {
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

            for (int bit = 0; bit < k; bit++) {
                if (pos_pattern & (1 << (k - 1 - bit))) {
                    scatter_rows[total_scatter] = (int8_t)bit;
                    scatter_signs[total_scatter] = 1;
                    total_scatter++;
                }
            }
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
