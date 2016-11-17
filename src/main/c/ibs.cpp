#include <popcntintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "ibs.h"

#define EXPORT __attribute__((visibility("default")))

using namespace simdpp;

void ibs256(__restrict__ uint64_t* result, uint64v x, uint64v y, uint64v xna, uint64v yna) {
  uint64v allones = make_ones();
  uint64v leftAllele = make_uint(0xAAAAAAAAAAAAAAAA);

  uint64v nxor = ~(x ^ y);
  uint64v nxorSR1 = shift_r(nxor, 1);

  uint64v na = ~(xna & yna) | leftAllele;

  uint64v ibs2 = bit_andnot(nxorSR1 & nxor, na);
  uint64_t ibs2sum = _mm_popcnt_u64(extract<0>(ibs2));
  ibs2sum += _mm_popcnt_u64(extract<1>(ibs2));
  #if SIMDPP_FAST_INT64_SIZE >= 4
  ibs2sum += _mm_popcnt_u64(extract<2>(ibs2));
  ibs2sum += _mm_popcnt_u64(extract<3>(ibs2));
  #endif
  #if SIMDPP_FAST_INT64_SIZE >= 8
  ibs2sum += _mm_popcnt_u64(extract<4>(ibs2));
  ibs2sum += _mm_popcnt_u64(extract<5>(ibs2));
  ibs2sum += _mm_popcnt_u64(extract<6>(ibs2));
  ibs2sum += _mm_popcnt_u64(extract<7>(ibs2));
  #endif

  uint64v ibs1 = bit_andnot(nxorSR1 ^ nxor, na);
  uint64_t ibs1sum = _mm_popcnt_u64(extract<0>(ibs1));
  ibs1sum += _mm_popcnt_u64(extract<1>(ibs1));
  #if SIMDPP_FAST_INT64_SIZE >= 4
  ibs1sum += _mm_popcnt_u64(extract<2>(ibs1));
  ibs1sum += _mm_popcnt_u64(extract<3>(ibs1));
  #endif
  #if SIMDPP_FAST_INT64_SIZE >= 8
  ibs1sum += _mm_popcnt_u64(extract<4>(ibs1));
  ibs1sum += _mm_popcnt_u64(extract<5>(ibs1));
  ibs1sum += _mm_popcnt_u64(extract<6>(ibs1));
  ibs1sum += _mm_popcnt_u64(extract<7>(ibs1));
  #endif

  uint64v ibs0 = bit_andnot(~(nxorSR1 | nxor), na);
  uint64_t ibs0sum = _mm_popcnt_u64(extract<0>(ibs0));
  ibs0sum += _mm_popcnt_u64(extract<1>(ibs0));
  #if SIMDPP_FAST_INT64_SIZE >= 4
  ibs0sum += _mm_popcnt_u64(extract<2>(ibs0));
  ibs0sum += _mm_popcnt_u64(extract<3>(ibs0));
  #endif
  #if SIMDPP_FAST_INT64_SIZE >= 8
  ibs0sum += _mm_popcnt_u64(extract<4>(ibs0));
  ibs0sum += _mm_popcnt_u64(extract<5>(ibs0));
  ibs0sum += _mm_popcnt_u64(extract<6>(ibs0));
  ibs0sum += _mm_popcnt_u64(extract<7>(ibs0));
  #endif

  result[0] += ibs0sum;
  result[1] += ibs1sum;
  result[2] += ibs2sum;
}

uint64v naMaskForGenotypePack(uint64v block) {
  uint64v allNA = make_uint(0xAAAAAAAAAAAAAAAA);
  uint64v isna = allNA ^ block;
  return (shift_r(isna, 1)) | isna;
}

void ibs256_with_na(__restrict__ uint64_t* result, uint64v x, uint64v y) {
  ibs256(result, x, y, naMaskForGenotypePack(x), naMaskForGenotypePack(y));
}

void ibsVec(__restrict__ uint64_t* result,
            uint64_t length,
            __restrict__ uint64_t* x,
            __restrict__ uint64_t* y,
            __restrict__ uint64v * x_na_masks,
            __restrict__ uint64v * y_na_masks) {
  uint64_t i = 0;
  for (; i <= (length - SIMDPP_FAST_INT64_SIZE); i += SIMDPP_FAST_INT64_SIZE) {
    uint64v x256 = load_u(x+i);
    uint64v y256 = load_u(y+i);
    ibs256(result, x256, y256, x_na_masks[i/SIMDPP_FAST_INT64_SIZE], y_na_masks[i/SIMDPP_FAST_INT64_SIZE]);
  }
  for (; i < length; ++i) {
    uint64_t rightAllele64 = 0x5555555555555555;
    uint64_t allNA64 = 0xAAAAAAAAAAAAAAAA;
    uint64_t xb = x[i];
    uint64_t yb = y[i];
    uint64_t nxor = ~(xb ^ yb);
    uint64_t xna_tmp = ~(allNA64 ^ xb);
    uint64_t xna = (xna_tmp >> 1) & xna_tmp;
    uint64_t yna_tmp = ~(allNA64 ^ yb);
    uint64_t yna = (yna_tmp >> 1) & yna_tmp;
    uint64_t na = (xna | yna) & rightAllele64;
    result[2] += _mm_popcnt_u64(((nxor >> 1) & nxor) & rightAllele64 & ~na);
    result[1] += _mm_popcnt_u64(((nxor >> 1) ^ nxor) & rightAllele64 & ~na);
    result[0] += _mm_popcnt_u64(~((nxor >> 1) | nxor) & rightAllele64 & ~na);
  }
}

void allocateNaMasks(uint64v ** mask1,
                     uint64v ** mask2,
                     uint64_t nSamples,
                     uint64_t nGenotypePacks,
                     __restrict__ uint64_t* x,
                     __restrict__ uint64_t* y) {
  int err1 = posix_memalign((void **)mask1, 32, nSamples*(nGenotypePacks/SIMDPP_FAST_INT64_SIZE)*sizeof(uint64v));
  int err2 = posix_memalign((void **)mask2, 32, nSamples*(nGenotypePacks/SIMDPP_FAST_INT64_SIZE)*sizeof(uint64v));
  if (err1 || err2) {
    printf("Not enough memory to allocate space for the naMasks: %d %d\n", err1, err2);
    exit(-1);
  }

  for (uint64_t i = 0; i <= (length - SIMDPP_FAST_INT64_SIZE); i += SIMDPP_FAST_INT64_SIZE) {
    naMasks1[i/SIMDPP_FAST_INT64_SIZE] =
      naMaskForGenotypePack(load_u(x+i));
    naMasks2[i/SIMDPP_FAST_INT64_SIZE] =
      naMaskForGenotypePack(load_u(y+i));
  }
}

// used to test ibsVec: allocates naMasks locally rather than accepting it as an argument; only for one pair of samples
void ibsVec2(__restrict__ uint64_t* result,
             uint64_t nGenotypePacks,
             __restrict__ uint64_t* x,
             __restrict__ uint64_t* y) {
  uint64v * naMasks1 = 0;
  uint64v * naMasks2 = 0;
  allocateNaMasks(&naMasks1, &naMasks2, 2, nGenotypePacks, x, y);
  ibsVec(result, length, x, y, naMasks1, naMasks2);
}

#ifndef CACHE_SIZE_PER_MATRIX_IN_KB
#define CACHE_SIZE_PER_MATRIX_IN_KB 4
#endif

#ifndef CACHE_SIZE_IN_MATRIX_ROWS
#define CACHE_SIZE_IN_MATRIX_ROWS 4 * CACHE_SIZE_PER_MATRIX_IN_KB
#endif

// samples in rows, genotypes in columns
extern "C"
EXPORT
void ibsMat(__restrict__ uint64_t* result, uint64_t nSamples, uint64_t nGenotypePacks, __restrict__ uint64_t* genotypes1, __restrict__ uint64_t* genotypes2) {
  uint64v * naMasks1 = 0;
  uint64v * naMasks2 = 0;
  allocateNaMasks(&naMasks1, &naMasks2, nSamples, nGenotypePacks, x, y);

  uint64_t i_block_end;
  for (i_block_end = CACHE_SIZE_IN_MATRIX_ROWS;
       i_block_end <= nSamples;
       i_block_end += CACHE_SIZE_IN_MATRIX_ROWS) {
    uint64_t j_block_end;
    for (j_block_end = CACHE_SIZE_IN_MATRIX_ROWS;
         j_block_end <= nSamples;
         j_block_end += CACHE_SIZE_IN_MATRIX_ROWS) {
      for (uint64_t si = i_block_end - CACHE_SIZE_IN_MATRIX_ROWS;
           si != i_block_end;
           ++si) {
        for (uint64_t sj = j_block_end - CACHE_SIZE_IN_MATRIX_ROWS;
             sj != j_block_end;
             ++sj) {
          ibsVec(result + si*nSamples*3 + sj*3,
                 nGenotypePacks,
                 genotypes1 + si*nGenotypePacks,
                 genotypes2 + sj*nGenotypePacks,
                 naMasks1 + si*(nGenotypePacks/SIMDPP_FAST_INT64_SIZE),
                 naMasks2 + sj*(nGenotypePacks/SIMDPP_FAST_INT64_SIZE)
                 );
        }
      }
    }
    for (uint64_t si = i_block_end - CACHE_SIZE_IN_MATRIX_ROWS;
         si != i_block_end;
         ++si) {
      for (uint64_t sj = j_block_end - CACHE_SIZE_IN_MATRIX_ROWS;
           sj < nSamples;
           ++sj) {
        ibsVec(result + si*nSamples*3 + sj*3,
               nGenotypePacks,
               genotypes1 + si*nGenotypePacks,
               genotypes2 + sj*nGenotypePacks,
               naMasks1 + si*(nGenotypePacks/SIMDPP_FAST_INT64_SIZE),
               naMasks2 + sj*(nGenotypePacks/SIMDPP_FAST_INT64_SIZE)
               );
      }
    }
  }
  for (uint64_t si = i_block_end - CACHE_SIZE_IN_MATRIX_ROWS;
       si < nSamples;
       ++si) {
    uint64_t j_block_end;
    for (j_block_end = CACHE_SIZE_IN_MATRIX_ROWS;
         j_block_end <= nSamples;
         j_block_end += CACHE_SIZE_IN_MATRIX_ROWS) {
      for (uint64_t sj = j_block_end - CACHE_SIZE_IN_MATRIX_ROWS;
           sj != j_block_end;
           ++sj) {
        ibsVec(result + si*nSamples*3 + sj*3,
               nGenotypePacks,
               genotypes1 + si*nGenotypePacks,
               genotypes2 + sj*nGenotypePacks,
               naMasks1 + si*(nGenotypePacks/SIMDPP_FAST_INT64_SIZE),
               naMasks2 + sj*(nGenotypePacks/SIMDPP_FAST_INT64_SIZE)
               );
      }
    }
    for (uint64_t sj = j_block_end - CACHE_SIZE_IN_MATRIX_ROWS;
         sj < nSamples;
         ++sj) {
      ibsVec(result + si*nSamples*3 + sj*3,
             nGenotypePacks,
             genotypes1 + si*nGenotypePacks,
             genotypes2 + sj*nGenotypePacks,
             naMasks1 + si*(nGenotypePacks/SIMDPP_FAST_INT64_SIZE),
             naMasks2 + sj*(nGenotypePacks/SIMDPP_FAST_INT64_SIZE)
             );
    }
  }


  free(naMasks1);
  free(naMasks2);
}
