#include <popcntintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "ibs.h"

#define EXPORT __attribute__((visibility("default")))

using namespace simdpp;

void ibs256(uint64_t* __restrict__ result, hailvec x, hailvec y, hailvec xna, hailvec yna) {
  hailvec allones = make_ones();
  hailvec leftAllele = make_uint(0xAAAAAAAAAAAAAAAA);

  hailvec nxor = ~(x ^ y);
  hailvec nxorSR1 = shift_r(nxor, 1);

  hailvec na = ~(xna & yna) | leftAllele;

  hailvec ibs2 = bit_andnot(nxorSR1 & nxor, na);
  uint64_t ibs2sum = _mm_popcnt_u64(extract<0>(ibs2));
  #if HAIL_VECTOR_WIDTH >= 2
  ibs2sum += _mm_popcnt_u64(extract<1>(ibs2));
  #endif
  #if HAIL_VECTOR_WIDTH >= 4
  ibs2sum += _mm_popcnt_u64(extract<2>(ibs2));
  ibs2sum += _mm_popcnt_u64(extract<3>(ibs2));
  #endif
  #if HAIL_VECTOR_WIDTH >= 8
  ibs2sum += _mm_popcnt_u64(extract<4>(ibs2));
  ibs2sum += _mm_popcnt_u64(extract<5>(ibs2));
  ibs2sum += _mm_popcnt_u64(extract<6>(ibs2));
  ibs2sum += _mm_popcnt_u64(extract<7>(ibs2));
  #endif

  hailvec ibs1 = bit_andnot(nxorSR1 ^ nxor, na);
  uint64_t ibs1sum = _mm_popcnt_u64(extract<0>(ibs1));
  #if HAIL_VECTOR_WIDTH >= 2
  ibs1sum += _mm_popcnt_u64(extract<1>(ibs1));
  #endif
  #if HAIL_VECTOR_WIDTH >= 4
  ibs1sum += _mm_popcnt_u64(extract<2>(ibs1));
  ibs1sum += _mm_popcnt_u64(extract<3>(ibs1));
  #endif
  #if HAIL_VECTOR_WIDTH >= 8
  ibs1sum += _mm_popcnt_u64(extract<4>(ibs1));
  ibs1sum += _mm_popcnt_u64(extract<5>(ibs1));
  ibs1sum += _mm_popcnt_u64(extract<6>(ibs1));
  ibs1sum += _mm_popcnt_u64(extract<7>(ibs1));
  #endif

  hailvec ibs0 = bit_andnot(~(nxorSR1 | nxor), na);
  uint64_t ibs0sum = _mm_popcnt_u64(extract<0>(ibs0));
  #if HAIL_VECTOR_WIDTH >= 2
  ibs0sum += _mm_popcnt_u64(extract<1>(ibs0));
  #endif
  #if HAIL_VECTOR_WIDTH >= 4
  ibs0sum += _mm_popcnt_u64(extract<2>(ibs0));
  ibs0sum += _mm_popcnt_u64(extract<3>(ibs0));
  #endif
  #if HAIL_VECTOR_WIDTH >= 8
  ibs0sum += _mm_popcnt_u64(extract<4>(ibs0));
  ibs0sum += _mm_popcnt_u64(extract<5>(ibs0));
  ibs0sum += _mm_popcnt_u64(extract<6>(ibs0));
  ibs0sum += _mm_popcnt_u64(extract<7>(ibs0));
  #endif

  result[0] += ibs0sum;
  result[1] += ibs1sum;
  result[2] += ibs2sum;
}

hailvec naMaskForGenotypePack(hailvec block) {
  hailvec allNA = make_uint(0xAAAAAAAAAAAAAAAA);
  hailvec isna = allNA ^ block;
  return (shift_r(isna, 1)) | isna;
}

void ibs256_with_na(uint64_t* __restrict__ result, hailvec x, hailvec y) {
  ibs256(result, x, y, naMaskForGenotypePack(x), naMaskForGenotypePack(y));
}

void ibsVec(uint64_t* __restrict__ result,
            uint64_t length,
            uint64_t* __restrict__ x,
            uint64_t* __restrict__ y,
            hailvec * __restrict__ x_na_masks,
            hailvec * __restrict__ y_na_masks) {
  uint64_t i = 0;
  for (; i <= (length - HAIL_VECTOR_WIDTH); i += HAIL_VECTOR_WIDTH) {
    hailvec x256 = load_u(x+i);
    hailvec y256 = load_u(y+i);
    ibs256(result, x256, y256, x_na_masks[i/HAIL_VECTOR_WIDTH], y_na_masks[i/HAIL_VECTOR_WIDTH]);
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

void createNaMasks(hailvec ** mask1,
                   hailvec ** mask2,
                   uint64_t nSamples,
                   uint64_t nGenotypePacks,
                   uint64_t* __restrict__ x,
                   uint64_t* __restrict__ y) {
  int err1 = posix_memalign((void **)mask1, 32, nSamples*(nGenotypePacks/HAIL_VECTOR_WIDTH)*sizeof(hailvec));
  int err2 = posix_memalign((void **)mask2, 32, nSamples*(nGenotypePacks/HAIL_VECTOR_WIDTH)*sizeof(hailvec));
  if (err1 || err2) {
    printf("Not enough memory to allocate space for the naMasks: %d %d\n", err1, err2);
    exit(-1);
  }

  for (uint64_t i = 0; i != nSamples; ++i) {
    for (uint64_t j = 0; j <= (nGenotypePacks - HAIL_VECTOR_WIDTH); j += HAIL_VECTOR_WIDTH) {
      (*mask1)[i*(nGenotypePacks/HAIL_VECTOR_WIDTH)+(j/HAIL_VECTOR_WIDTH)] =
        naMaskForGenotypePack(load_u(x+i*nGenotypePacks+j));
      (*mask2)[i*(nGenotypePacks/HAIL_VECTOR_WIDTH)+(j/HAIL_VECTOR_WIDTH)] =
        naMaskForGenotypePack(load_u(y+i*nGenotypePacks+j));
    }
  }
}

// used to test ibsVec: allocates naMasks locally rather than accepting it as an argument; only for one pair of samples
void ibsVec2(uint64_t* __restrict__ result,
             uint64_t nGenotypePacks,
             uint64_t* __restrict__ x,
             uint64_t* __restrict__ y) {
  hailvec * naMasks1 = 0;
  hailvec * naMasks2 = 0;
  createNaMasks(&naMasks1, &naMasks2, 1, nGenotypePacks, x, y);
  ibsVec(result, nGenotypePacks, x, y, naMasks1, naMasks2);
}

// samples in rows, genotypes in columns
extern "C"
EXPORT
void ibsMat(uint64_t* __restrict__ result, uint64_t nSamples, uint64_t nGenotypePacks, uint64_t* __restrict__ genotypes1, uint64_t* __restrict__ genotypes2) {
  hailvec * naMasks1 = 0;
  hailvec * naMasks2 = 0;
  createNaMasks(&naMasks1, &naMasks2, nSamples, nGenotypePacks, genotypes1, genotypes2);

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
                 naMasks1 + si*(nGenotypePacks/HAIL_VECTOR_WIDTH),
                 naMasks2 + sj*(nGenotypePacks/HAIL_VECTOR_WIDTH)
                 );
        }
      }
    }
  }

  free(naMasks1);
  free(naMasks2);
}
