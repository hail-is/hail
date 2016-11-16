#include <popcntintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <simdpp/simd.h>

#define EXPORT __attribute__((visibility("default")))

using namespace simdpp;

void ibs256(uint64_t* result, uint64v x, uint64v y, uint64v xna, uint64v yna) {
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

void ibs256_with_na(uint64_t* result, uint64v x, uint64v y) {
  ibs256(result, x, y, naMaskForGenotypePack(x), naMaskForGenotypePack(y));
}

void ibsVec(uint64_t* result, uint64_t length, uint64_t* x, uint64_t* y, uint64v * x_na_masks, uint64v * y_na_masks) {
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

void ibsVec2(uint64_t* result, uint64_t length, uint64_t* x, uint64_t* y) {
  uint64v * naMasks1 = 0;
  int err1 = posix_memalign((void **)&naMasks1, 32, (length/SIMDPP_FAST_INT64_SIZE)*sizeof(uint64v));
  uint64v * naMasks2 = 0;
  int err2 = posix_memalign((void **)&naMasks2, 32, (length/SIMDPP_FAST_INT64_SIZE)*sizeof(uint64v));
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
  ibsVec(result, length, x, y, naMasks1, naMasks2);
}

#define CACHE_SIZE_PER_MATRIX_IN_KB 4
#define CACHE_SIZE_IN_MATRIX_ROWS 4 * CACHE_SIZE_PER_MATRIX_IN_KB

// samples in rows, genotypes in columns
extern "C"
EXPORT
void ibsMat(uint64_t* result, uint64_t nSamples, uint64_t nGenotypePacks, uint64_t* genotypes1, uint64_t* genotypes2) {
  uint64v * naMasks1 = 0;
  int err1 = posix_memalign((void **)&naMasks1, 32, nSamples*(nGenotypePacks/SIMDPP_FAST_INT64_SIZE)*sizeof(uint64v));
  uint64v * naMasks2 = 0;
  int err2 = posix_memalign((void **)&naMasks2, 32, nSamples*(nGenotypePacks/SIMDPP_FAST_INT64_SIZE)*sizeof(uint64v));
  if (err1 || err2) {
    printf("Not enough memory to allocate space for the naMasks: %d %d\n", err1, err2);
    exit(-1);
  }

  for (uint64_t i = 0; i != nSamples; ++i) {
    for (uint64_t j = 0; j <= (nGenotypePacks - SIMDPP_FAST_INT64_SIZE); j += SIMDPP_FAST_INT64_SIZE) {
      naMasks1[i*(nGenotypePacks/SIMDPP_FAST_INT64_SIZE)+(j/SIMDPP_FAST_INT64_SIZE)] =
        naMaskForGenotypePack(load_u(genotypes1+i*nGenotypePacks+j));
      naMasks2[i*(nGenotypePacks/SIMDPP_FAST_INT64_SIZE)+(j/SIMDPP_FAST_INT64_SIZE)] =
        naMaskForGenotypePack(load_u(genotypes2+i*nGenotypePacks+j));
    }
    // NA mask for trailing genotype blocks will be calculated in the loop
  }

  uint64_t i_block_end;
  for (i_block_end = CACHE_SIZE_IN_MATRIX_ROWS;
       i_block_end <= (nSamples - CACHE_SIZE_IN_MATRIX_ROWS);
       i_block_end += CACHE_SIZE_IN_MATRIX_ROWS) {
    uint64_t j_block_end;
    for (j_block_end = CACHE_SIZE_IN_MATRIX_ROWS;
         j_block_end <= (nSamples - CACHE_SIZE_IN_MATRIX_ROWS);
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
         j_block_end <= (nSamples - CACHE_SIZE_IN_MATRIX_ROWS);
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


  free(naMasks1);
  free(naMasks2);
}

#define expect(name, x) if (!(x)) { ++failures; printf(name ": expected " #x " to be true, but was false\n\n"); } else { ++successes; }
#define expect_equal(name, fmt, x, y) if ((x) != (y)) { ++failures; printf(name ": expected " #x " to equal " #y ", but actually got " fmt " and " fmt"\n\n", x, y); } else { ++successes; }

uint64_t resultIndex(uint64_t* result, int nSamples, int si, int sj, int ibs) {
  return result[si*nSamples*3 + sj*3 + ibs];
}

int main(int argc, char** argv) {
  if (argc != 1) {
    printf("Expected zero arguments.\n");
    return -1;
  }

  uint64_t failures = 0;
  uint64_t successes = 0;

  // ibs256 tests
  {
    uint64_t result[3] = { 0, 0, 0 };
    uint64v allNA1 = make_uint(0xAAAAAAAAAAAAAAAA);
    uint64v allNA2 = make_uint(0xAAAAAAAAAAAAAAAA);
    ibs256_with_na(result, allNA1, allNA2);

    expect_equal("allNA", "%" PRIu64, result[0], ((uint64_t)0));
    expect_equal("allNA", "%" PRIu64, result[1], ((uint64_t)0));
    expect_equal("allNA", "%" PRIu64, result[2], ((uint64_t)0));
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    uint64v allHomRef1 = make_uint(0x0000000000000000);
    uint64v allHomRef2 = make_uint(0x0000000000000000);
    ibs256_with_na(result, allHomRef1, allHomRef2);

    expect_equal("allHomRef", "%" PRIu64, result[0], ((uint64_t)0));
    expect_equal("allHomRef", "%" PRIu64, result[1], ((uint64_t)0));
    expect_equal("allHomRef", "%" PRIu64, result[2], ((uint64_t)128));
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    uint64v allHet1 = make_uint(0x5555555555555555);
    uint64v allHet2 = make_uint(0x5555555555555555);
    ibs256_with_na(result, allHet1, allHet2);

    expect_equal("allHet", "%" PRIu64, result[0], ((uint64_t)0));
    expect_equal("allHet", "%" PRIu64, result[1], ((uint64_t)0));
    expect_equal("allHet", "%" PRIu64, result[2], ((uint64_t)128));
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    uint64v allHomAlt1 = make_uint(0xFFFFFFFFFFFFFFFF);
    uint64v allHomAlt2 = make_uint(0xFFFFFFFFFFFFFFFF);
    ibs256_with_na(result, allHomAlt1, allHomAlt2);

    expect_equal("allHomAlt", "%" PRIu64, result[0], ((uint64_t)0));
    expect_equal("allHomAlt", "%" PRIu64, result[1], ((uint64_t)0));
    expect_equal("allHomAlt", "%" PRIu64, result[2], ((uint64_t)128));
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    uint64v allHomAlt = make_uint(0xFFFFFFFFFFFFFFFF);
    uint64v allHomRef = make_uint(0x0000000000000000);
    ibs256_with_na(result, allHomAlt, allHomRef);

    expect_equal("homAlt v homRef", "%" PRIu64, result[0], ((uint64_t)128));
    expect_equal("homAlt v homRef", "%" PRIu64, result[1], ((uint64_t)0));
    expect_equal("homAlt v homRef", "%" PRIu64, result[2], ((uint64_t)0));
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    uint64v allHet = make_uint(0x5555555555555555);
    uint64v allHomRef = make_uint(0x0000000000000000);
    ibs256_with_na(result, allHet, allHomRef);

    expect_equal("het v homRef", "%" PRIu64, result[0], ((uint64_t)0));
    expect_equal("het v homRef", "%" PRIu64, result[1], ((uint64_t)128));
    expect_equal("het v homRef", "%" PRIu64, result[2], ((uint64_t)0));
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    uint64v allHet = make_uint(0x5555555555555555);
    uint64v allHomAlt = make_uint(0xFFFFFFFFFFFFFFFF);
    ibs256_with_na(result, allHet, allHomAlt);

    expect_equal("het v homAlt", "%" PRIu64, result[0], ((uint64_t)0));
    expect_equal("het v homAlt", "%" PRIu64, result[1], ((uint64_t)128));
    expect_equal("het v homAlt", "%" PRIu64, result[2], ((uint64_t)0));
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    uint64v allHet = make_uint(0x5555555555555555);
    uint64v allHomAltOneNA = make_uint(0xBFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF);
    ibs256_with_na(result, allHet, allHomAltOneNA);

    expect_equal("het v homAltOneNA", "%" PRIu64, result[0], ((uint64_t)0));
    expect_equal("het v homAltOneNA", "%" PRIu64, result[1], ((uint64_t)127));
    expect_equal("het v homAltOneNA", "%" PRIu64, result[2], ((uint64_t)0));
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    uint64v allHet = make_uint(0x5555555555555555);
    uint64v allHomAltTwoNA = make_uint(0xAFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF);
    ibs256_with_na(result, allHet, allHomAltTwoNA);

    expect_equal("het v homAltTwoNA", "%" PRIu64, result[0], ((uint64_t)0));
    expect_equal("het v homAltTwoNA", "%" PRIu64, result[1], ((uint64_t)126));
    expect_equal("het v homAltTwoNA", "%" PRIu64, result[2], ((uint64_t)0));
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    uint64v ref_het_alt_het = make_uint(0x1DAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA);
    uint64v het_ref_het_alt = make_uint(0x47AAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA);
    ibs256_with_na(result, ref_het_alt_het, het_ref_het_alt);

    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[0], ((uint64_t)0));
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[1], ((uint64_t)4));
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[2], ((uint64_t)0));
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    uint64v ref_het_alt_het = make_uint(0xAAAAAAAAAAAAAAAA, 0x1DAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA);
    uint64v het_ref_het_alt = make_uint(0xAAAAAAAAAAAAAAAA, 0x47AAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA);
    ibs256_with_na(result, ref_het_alt_het, het_ref_het_alt);

    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[0], ((uint64_t)0));
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[1], ((uint64_t)4));
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[2], ((uint64_t)0));
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    uint64v ref_het_alt_het = make_uint(0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0x1DAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA);
    uint64v het_ref_het_alt = make_uint(0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0x47AAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA);
    ibs256_with_na(result, ref_het_alt_het, het_ref_het_alt);

    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[0], ((uint64_t)0));
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[1], ((uint64_t)4));
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[2], ((uint64_t)0));
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    uint64v ref_het_alt_het = make_uint(0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0x1DAAAAAAAAAAAAAA);
    uint64v het_ref_het_alt = make_uint(0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0x47AAAAAAAAAAAAAA);
    ibs256_with_na(result, ref_het_alt_het, het_ref_het_alt);

    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[0], ((uint64_t)0));
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[1], ((uint64_t)4));
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[2], ((uint64_t)0));
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    uint64v ref_het_alt_het = make_uint(0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAA1D);
    uint64v het_ref_het_alt = make_uint(0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAA47);
    ibs256_with_na(result, ref_het_alt_het, het_ref_het_alt);

    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[0], ((uint64_t)0));
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[1], ((uint64_t)4));
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[2], ((uint64_t)0));
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    uint64v ref_het_alt_het = make_uint(0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAAAA);
    uint64v het_ref_het_alt = make_uint(0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAA47, 0xAAAAAAAAAAAAAAAA);
    ibs256_with_na(result, ref_het_alt_het, het_ref_het_alt);

    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[0], ((uint64_t)0));
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[1], ((uint64_t)4));
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[2], ((uint64_t)0));
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    uint64v ref_het_alt_het = make_uint(0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA);
    uint64v het_ref_het_alt = make_uint(0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAA47, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA);
    ibs256_with_na(result, ref_het_alt_het, het_ref_het_alt);

    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[0], ((uint64_t)0));
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[1], ((uint64_t)4));
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[2], ((uint64_t)0));
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    uint64v ref_het_alt_het = make_uint(0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA);
    uint64v het_ref_het_alt = make_uint(0xAAAAAAAAAAAAAA47, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA);
    ibs256_with_na(result, ref_het_alt_het, het_ref_het_alt);

    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[0], ((uint64_t)0));
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[1], ((uint64_t)4));
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[2], ((uint64_t)0));
  }

  // ibsVec tests
  {
    uint64_t result[3] = { 0, 0, 0 };
    uint64_t ref_het_alt_het[7] =
      { 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
        0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA };
    uint64_t het_ref_het_alt[7] =
      { 0xAAAAAAAAAAAAAA47, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
        0xAAAAAAAAAAAAAA47, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA };
    ibsVec2(result, 7, ref_het_alt_het, het_ref_het_alt);

    expect_equal("ibsVec2 ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[0], ((uint64_t)0));
    expect_equal("ibsVec2 ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[1], ((uint64_t)8));
    expect_equal("ibsVec2 ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[2], ((uint64_t)0));
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    uint64_t ref_het_alt_het[7] =
      { 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D,
        0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D };
    uint64_t het_ref_het_alt[7] =
      { 0xAAAAAAAAAAAAAA47, 0xAAAAAAAAAAAAAA47, 0xAAAAAAAAAAAAAA47, 0xAAAAAAAAAAAAAA47,
        0xAAAAAAAAAAAAAA47, 0xAAAAAAAAAAAAAA47, 0xAAAAAAAAAAAAAA47 };
    ibsVec2(result, 7, ref_het_alt_het, het_ref_het_alt);

    expect_equal("ibsVec2 7 ref_het_alt_het v 7 het_ref_het_alt", "%" PRIu64, result[0], ((uint64_t)0));
    expect_equal("ibsVec2 7 ref_het_alt_het v 7 het_ref_het_alt", "%" PRIu64, result[1], 7 * ((uint64_t)4));
    expect_equal("ibsVec2 7 ref_het_alt_het v 7 het_ref_het_alt", "%" PRIu64, result[2], ((uint64_t)0));
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    uint64_t ref_het_alt_het[7] =
      { 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D,
        0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D };
    ibsVec2(result, 7, ref_het_alt_het, ref_het_alt_het);

    expect_equal("ibsVec2 ref_het_alt_het v self", "%" PRIu64, result[0], ((uint64_t)0));
    expect_equal("ibsVec2 ref_het_alt_het v self", "%" PRIu64, result[1], ((uint64_t)0));
    expect_equal("ibsVec2 ref_het_alt_het v self", "%" PRIu64, result[2], 4*((uint64_t)7));
  }
  // ibsMat
  {
    uint64_t result[2*2*3] = { 0, 0, 0,  0, 0, 0,
                               0, 0, 0,  0, 0, 0 };
    uint64_t all_ref_het_alt_het[2*7] =
      { 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D,
        0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D };
    ibsMat(result, 2, 7, all_ref_het_alt_het, all_ref_het_alt_het);

    expect_equal("ibsMat identical 0 0, ibs0", "%" PRIu64, resultIndex(result, 2, 0, 0, 0), ((uint64_t)0));
    expect_equal("ibsMat identical 0 0, ibs1", "%" PRIu64, resultIndex(result, 2, 0, 0, 1), ((uint64_t)0));
    expect_equal("ibsMat identical 0 0, ibs2", "%" PRIu64, resultIndex(result, 2, 0, 0, 2), 4*((uint64_t)7));
    expect_equal("ibsMat identical 0 1, ibs0", "%" PRIu64, resultIndex(result, 2, 0, 1, 0), ((uint64_t)0));
    expect_equal("ibsMat identical 0 1, ibs1", "%" PRIu64, resultIndex(result, 2, 0, 1, 1), ((uint64_t)0));
    expect_equal("ibsMat identical 0 1, ibs2", "%" PRIu64, resultIndex(result, 2, 0, 1, 2), 4*((uint64_t)7));
    expect_equal("ibsMat identical 1 1, ibs0", "%" PRIu64, resultIndex(result, 2, 1, 1, 0), ((uint64_t)0));
    expect_equal("ibsMat identical 1 1, ibs1", "%" PRIu64, resultIndex(result, 2, 1, 1, 1), ((uint64_t)0));
    expect_equal("ibsMat identical 1 1, ibs2", "%" PRIu64, resultIndex(result, 2, 1, 1, 2), 4*((uint64_t)7));
  }
  {
    uint64_t result[2*2*3] = { 0, 0, 0,  0, 0, 0,
                               0, 0, 0,  0, 0, 0 };
    uint64_t one_ibs1_rest_ibs2[2*7] =
      { 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D,
        0xAAAAAAAAAAAAAA1F, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D };
    ibsMat(result, 2, 7, one_ibs1_rest_ibs2, one_ibs1_rest_ibs2);

    expect_equal("ibsMat one-ibs1 0 0, ibs0", "%" PRIu64, resultIndex(result, 2, 0, 0, 0), ((uint64_t)0));
    expect_equal("ibsMat one-ibs1 0 0, ibs1", "%" PRIu64, resultIndex(result, 2, 0, 0, 1), ((uint64_t)0));
    expect_equal("ibsMat one-ibs1 0 0, ibs2", "%" PRIu64, resultIndex(result, 2, 0, 0, 2), 4*((uint64_t)7));
    expect_equal("ibsMat one-ibs1 0 1, ibs0", "%" PRIu64, resultIndex(result, 2, 0, 1, 0), ((uint64_t)0));
    expect_equal("ibsMat one-ibs1 0 1, ibs1", "%" PRIu64, resultIndex(result, 2, 0, 1, 1), ((uint64_t)1));
    expect_equal("ibsMat one-ibs1 0 1, ibs2", "%" PRIu64, resultIndex(result, 2, 0, 1, 2), 3+4*((uint64_t)6));
    expect_equal("ibsMat one-ibs1 1 1, ibs0", "%" PRIu64, resultIndex(result, 2, 1, 1, 0), ((uint64_t)0));
    expect_equal("ibsMat one-ibs1 1 1, ibs1", "%" PRIu64, resultIndex(result, 2, 1, 1, 1), ((uint64_t)0));
    expect_equal("ibsMat one-ibs1 1 1, ibs2", "%" PRIu64, resultIndex(result, 2, 1, 1, 2), 4*((uint64_t)7));
  }

  if (failures != 0) {
    printf("%" PRIu64 " test(s) failed.\n", failures);
  } else {
    printf("%" PRIu64 " test(s) succeeded.\n", successes);
  }

  return 0;
}
