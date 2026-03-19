/*
 * BitNet.cpp-style ternary GEMV baseline.
 *
 * TQ2_0-style 2-bit packing: each ternary value {-1, 0, +1} is stored as 2 bits.
 *   00 = 0,  01 = +1,  10 = -1
 *
 * The kernel does conditional add/sub — no multiplications in the inner loop.
 * AVX2 path processes 16 ternary weights per iteration.
 *
 * Compile:
 *   gcc -O3 -march=native -mavx2 -fopenmp -shared -fPIC -o bitnet_ternary.so bitnet_ternary.c
 */

#include <stdint.h>
#include <string.h>
#include <immintrin.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/*
 * Pack ternary matrix M (int8, values {-1,0,+1}, row-major n×n)
 * into 2-bit format: 4 values per byte, row-major.
 * packed size = n * ceil(n/4) bytes.
 */
void bitnet_ternary_pack(
    const int8_t *M,       /* n x n, row-major, values {-1,0,+1} */
    uint8_t      *packed,  /* out: n * ((n+3)/4) bytes */
    int           n
)
{
    int cols_packed = (n + 3) / 4;

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        const int8_t *row = M + (int64_t)i * n;
        uint8_t *out_row = packed + (int64_t)i * cols_packed;

        for (int j4 = 0; j4 < cols_packed; j4++) {
            uint8_t byte = 0;
            for (int sub = 0; sub < 4; sub++) {
                int j = j4 * 4 + sub;
                uint8_t code = 0;  /* 0 */
                if (j < n) {
                    if (row[j] == 1)  code = 1;  /* 01 = +1 */
                    if (row[j] == -1) code = 2;  /* 10 = -1 */
                }
                byte |= (code << (sub * 2));
            }
            out_row[j4] = byte;
        }
    }
}

/*
 * Ternary GEMV: y = packed_M @ v
 *
 * For each row: decode 2-bit weights and do conditional add/sub.
 */
void bitnet_ternary_gemv(
    const uint8_t *packed,  /* n * ((n+3)/4) bytes */
    const float   *v,       /* n */
    float         *out,     /* n */
    int            n
)
{
    int cols_packed = (n + 3) / 4;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        const uint8_t *row = packed + (int64_t)i * cols_packed;
        float acc = 0.0f;

        int j4 = 0;

#ifdef __AVX2__
        /* Process 16 ternary values (4 bytes) per iteration */
        __m256 vacc = _mm256_setzero_ps();

        for (; j4 + 3 < cols_packed && (j4 * 4 + 15) < n; j4 += 4) {
            /* Load 4 bytes = 16 ternary values */
            uint32_t packed4;
            memcpy(&packed4, row + j4, 4);

            /* Process in groups of 8 (2 bytes each) */
            for (int half = 0; half < 2; half++) {
                uint16_t packed2 = (packed4 >> (half * 16)) & 0xFFFF;
                int base_j = j4 * 4 + half * 8;

                /* Load 8 floats */
                __m256 vx = _mm256_loadu_ps(v + base_j);

                /* Decode 8 ternary values from 16 bits */
                /* Even bits = bit0, odd bits = bit1 for each 2-bit field */
                /* For each of 8 values: code = (packed2 >> (i*2)) & 3 */
                /* +1 mask: code == 1 (bit0=1, bit1=0) */
                /* -1 mask: code == 2 (bit0=0, bit1=1) */

                /* Build masks for 8 values */
                __m256 plus_mask_f = _mm256_setzero_ps();
                __m256 minus_mask_f = _mm256_setzero_ps();

                /* Decode using scalar, build float masks */
                float pm[8], mm[8];
                for (int s = 0; s < 8; s++) {
                    int code = (packed2 >> (s * 2)) & 3;
                    pm[s] = (code == 1) ? 1.0f : 0.0f;
                    mm[s] = (code == 2) ? 1.0f : 0.0f;
                }
                plus_mask_f = _mm256_loadu_ps(pm);
                minus_mask_f = _mm256_loadu_ps(mm);

                /* acc += x * plus_mask - x * minus_mask */
                __m256 pos_contrib = _mm256_mul_ps(vx, plus_mask_f);
                __m256 neg_contrib = _mm256_mul_ps(vx, minus_mask_f);
                vacc = _mm256_add_ps(vacc, pos_contrib);
                vacc = _mm256_sub_ps(vacc, neg_contrib);
            }
        }

        /* Horizontal sum */
        __m128 lo = _mm256_castps256_ps128(vacc);
        __m128 hi = _mm256_extractf128_ps(vacc, 1);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);
        acc = _mm_cvtss_f32(lo);
#endif

        /* Scalar tail */
        for (; j4 < cols_packed; j4++) {
            uint8_t byte = row[j4];
            for (int sub = 0; sub < 4; sub++) {
                int j = j4 * 4 + sub;
                if (j >= n) break;
                int code = (byte >> (sub * 2)) & 3;
                if (code == 1) acc += v[j];        /* +1 */
                else if (code == 2) acc -= v[j];   /* -1 */
            }
        }

        out[i] = acc;
    }
}
