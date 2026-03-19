/*
 * RSR preprocessing kernel — counting-sort based.
 *
 * Two entry points:
 *   rsr_prep       — single-threaded
 *   rsr_prep_omp   — OpenMP parallel over blocks (two-pass)
 *
 * Key advantage over the Python prep:
 *   - Counting sort is O(n + 2^k) per block instead of O(n log n) comparison sort
 *   - Within each bucket the indices are already ascending (stable counting sort),
 *     so the extra within-group sort that Python does is free.
 *   - No Python-loop / PyTorch-dispatch overhead.
 */

#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ------------------------------------------------------------------ */
/*  Serial version                                                     */
/* ------------------------------------------------------------------ */

void rsr_prep(
    const uint8_t *M,          /* n x n binary matrix, row-major, 0/1  */
    int n, int k,
    int32_t *perms,            /* out: num_blocks * n                  */
    int32_t *group_ends,       /* out: total_groups                    */
    int32_t *scatter_offsets,   /* out: total_groups + 1                */
    uint8_t *scatter_rows,     /* out: total_scatter                   */
    int32_t *block_meta,       /* out: 2 * num_blocks                  */
    int32_t *out_sizes         /* out: [total_groups, total_scatter]    */
)
{
    int num_blocks = n / k;
    int num_buckets = 1 << k;

    int *counts   = (int *)malloc(num_buckets * sizeof(int));
    int *bkt_off  = (int *)malloc(num_buckets * sizeof(int));
    int *pos      = (int *)malloc(num_buckets * sizeof(int));
    int *col_vals = (int *)malloc(n * sizeof(int));

    int total_groups  = 0;
    int total_scatter = 0;
    scatter_offsets[0] = 0;

    for (int b = 0; b < num_blocks; b++) {
        int32_t *bp = perms + (int64_t)b * n;

        /* 1. Encode columns as k-bit integers + count buckets */
        memset(counts, 0, num_buckets * sizeof(int));
        for (int j = 0; j < n; j++) {
            int val = 0;
            for (int i = 0; i < k; i++)
                val = (val << 1) | M[((int64_t)b * k + i) * n + j];
            col_vals[j] = val;
            counts[val]++;
        }

        /* 2. Counting sort → permutation (stable, ascending within bucket) */
        bkt_off[0] = 0;
        for (int i = 1; i < num_buckets; i++)
            bkt_off[i] = bkt_off[i - 1] + counts[i - 1];
        memcpy(pos, bkt_off, num_buckets * sizeof(int));
        for (int j = 0; j < n; j++)
            bp[pos[col_vals[j]]++] = j;

        /* 3. group_ends, scatter_offsets, scatter_rows */
        int lg = 0;
        for (int v = 0; v < num_buckets; v++) {
            if (counts[v] == 0) continue;
            group_ends[total_groups + lg] = bkt_off[v] + counts[v];

            for (int bit = k - 1; bit >= 0; bit--)
                if (v & (1 << bit))
                    scatter_rows[total_scatter++] = (uint8_t)(k - 1 - bit);
            scatter_offsets[total_groups + lg + 1] = total_scatter;
            lg++;
        }

        block_meta[2 * b]     = total_groups;
        block_meta[2 * b + 1] = lg;
        total_groups += lg;
    }

    out_sizes[0] = total_groups;
    out_sizes[1] = total_scatter;

    free(counts); free(bkt_off); free(pos); free(col_vals);
}

/* ------------------------------------------------------------------ */
/*  OpenMP parallel version (two-pass)                                 */
/* ------------------------------------------------------------------ */

void rsr_prep_omp(
    const uint8_t *M,
    int n, int k,
    int32_t *perms,
    int32_t *group_ends,
    int32_t *scatter_offsets,
    uint8_t *scatter_rows,
    int32_t *block_meta,
    int32_t *out_sizes
)
{
    int num_blocks  = n / k;
    int num_buckets = 1 << k;

    int *blk_nuniq    = (int *)malloc(num_blocks * sizeof(int));
    int *blk_nscatter = (int *)malloc(num_blocks * sizeof(int));

    /* ---------- Pass 1: per-block sizes (parallel) ---------- */
    #pragma omp parallel
    {
        int *counts = (int *)malloc(num_buckets * sizeof(int));

        #pragma omp for schedule(static)
        for (int b = 0; b < num_blocks; b++) {
            memset(counts, 0, num_buckets * sizeof(int));
            for (int j = 0; j < n; j++) {
                int val = 0;
                for (int i = 0; i < k; i++)
                    val = (val << 1) | M[((int64_t)b * k + i) * n + j];
                counts[val]++;
            }
            int nu = 0, ns = 0;
            for (int v = 0; v < num_buckets; v++) {
                if (counts[v] > 0) {
                    nu++;
                    ns += __builtin_popcount(v);
                }
            }
            blk_nuniq[b]    = nu;
            blk_nscatter[b] = ns;
        }

        free(counts);
    }

    /* ---------- Prefix sums (serial — tiny) ---------- */
    int *g_off = (int *)malloc((num_blocks + 1) * sizeof(int));
    int *s_off = (int *)malloc((num_blocks + 1) * sizeof(int));
    g_off[0] = 0;
    s_off[0] = 0;
    for (int b = 0; b < num_blocks; b++) {
        g_off[b + 1] = g_off[b] + blk_nuniq[b];
        s_off[b + 1] = s_off[b] + blk_nscatter[b];
    }
    int total_groups  = g_off[num_blocks];
    int total_scatter = s_off[num_blocks];

    /* Set boundary scatter_offsets (one per block boundary) */
    for (int b = 0; b <= num_blocks; b++)
        scatter_offsets[g_off[b]] = s_off[b];

    /* ---------- Pass 2: fill arrays (parallel) ---------- */
    #pragma omp parallel
    {
        int *counts   = (int *)malloc(num_buckets * sizeof(int));
        int *bkt_off  = (int *)malloc(num_buckets * sizeof(int));
        int *pos      = (int *)malloc(num_buckets * sizeof(int));
        int *col_vals = (int *)malloc(n * sizeof(int));

        #pragma omp for schedule(static)
        for (int b = 0; b < num_blocks; b++) {
            int32_t *bp = perms + (int64_t)b * n;
            int go = g_off[b];
            int so = s_off[b];

            /* Encode + count */
            memset(counts, 0, num_buckets * sizeof(int));
            for (int j = 0; j < n; j++) {
                int val = 0;
                for (int i = 0; i < k; i++)
                    val = (val << 1) | M[((int64_t)b * k + i) * n + j];
                col_vals[j] = val;
                counts[val]++;
            }

            /* Counting sort */
            bkt_off[0] = 0;
            for (int i = 1; i < num_buckets; i++)
                bkt_off[i] = bkt_off[i - 1] + counts[i - 1];
            memcpy(pos, bkt_off, num_buckets * sizeof(int));
            for (int j = 0; j < n; j++)
                bp[pos[col_vals[j]]++] = j;

            /* Build group_ends + scatter */
            int lg = 0;
            int ls = so;
            for (int v = 0; v < num_buckets; v++) {
                if (counts[v] == 0) continue;
                group_ends[go + lg] = bkt_off[v] + counts[v];

                for (int bit = k - 1; bit >= 0; bit--)
                    if (v & (1 << bit))
                        scatter_rows[ls++] = (uint8_t)(k - 1 - bit);
                scatter_offsets[go + lg + 1] = ls;
                lg++;
            }

            block_meta[2 * b]     = go;
            block_meta[2 * b + 1] = blk_nuniq[b];
        }

        free(counts); free(bkt_off); free(pos); free(col_vals);
    }

    out_sizes[0] = total_groups;
    out_sizes[1] = total_scatter;

    free(blk_nuniq); free(blk_nscatter);
    free(g_off); free(s_off);
}

