#include <x86intrin.h>
#include <avxintrin.h>
#include <avx2intrin.h>
#include <popcntintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <inttypes.h>

#define EXPORT __attribute__((visibility("default")))

__m256i allones = { 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF };
__m256i allNA = { 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA };
__m256i leftAllele = { 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA };


#ifdef __AVX__
void ibs256(uint64_t* result, __m256i x, __m256i y, __m256i xna, __m256i yna) {
  __m256i nxor = _mm256_xor_si256(_mm256_xor_si256(x, y), allones);
  __m256i nxorSR1 = _mm256_srli_epi64(nxor, 1);

  __m256i na = _mm256_or_si256(_mm256_xor_si256(_mm256_and_si256(xna, yna), allones), leftAllele);

  __m256i ibs2 = _mm256_andnot_si256(na, _mm256_and_si256(nxorSR1, nxor));
  uint64_t ibs2sum = _mm_popcnt_u64(_mm256_extract_epi64(ibs2, 0));
  ibs2sum += _mm_popcnt_u64(_mm256_extract_epi64(ibs2, 1));
  ibs2sum += _mm_popcnt_u64(_mm256_extract_epi64(ibs2, 2));
  ibs2sum += _mm_popcnt_u64(_mm256_extract_epi64(ibs2, 3));

  __m256i ibs1 = _mm256_andnot_si256(na, _mm256_xor_si256(nxorSR1, nxor));
  uint64_t ibs1sum = _mm_popcnt_u64(_mm256_extract_epi64(ibs1, 0));
  ibs1sum += _mm_popcnt_u64(_mm256_extract_epi64(ibs1, 1));
  ibs1sum += _mm_popcnt_u64(_mm256_extract_epi64(ibs1, 2));
  ibs1sum += _mm_popcnt_u64(_mm256_extract_epi64(ibs1, 3));

  __m256i ibs0 = _mm256_andnot_si256(na, _mm256_xor_si256(_mm256_or_si256(nxorSR1, nxor), allones));
  uint64_t ibs0sum = _mm_popcnt_u64(_mm256_extract_epi64(ibs0, 0));
  ibs0sum += _mm_popcnt_u64(_mm256_extract_epi64(ibs0, 1));
  ibs0sum += _mm_popcnt_u64(_mm256_extract_epi64(ibs0, 2));
  ibs0sum += _mm_popcnt_u64(_mm256_extract_epi64(ibs0, 3));

  result[0] += ibs0sum;
  result[1] += ibs1sum;
  result[2] += ibs2sum;
}

__m256i naMaskForGenotypePack(__m256i block) {
  __m256i isna = _mm256_xor_si256(allNA, block);
  return _mm256_or_si256(_mm256_srli_epi64(isna, 1), isna);
}

void ibs256_with_na(uint64_t* result, __m256i x, __m256i y) {
  ibs256(result, x, y, naMaskForGenotypePack(x), naMaskForGenotypePack(y));
}
#endif // __AVX__

void ibsVec(uint64_t* result, uint64_t length, uint64_t* x, uint64_t* y
            #ifdef __AVX__
            , __m256i * x_na_masks, __m256i * y_na_masks
            #endif // __AVX__
            ) {
  uint64_t i = 0;
  #ifdef __AVX__
  for (; i < (length - 3); i += 4) {
    __m256i x256 = _mm256_load_si256((__m256i*)(x+i));
    __m256i y256 = _mm256_load_si256((__m256i*)(y+i));
    ibs256(result, x256, y256, x_na_masks[i/4], y_na_masks[i/4]);
  }
  #endif // __AVX__
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
  #ifdef __AVX__
  __m256i * naMasks1 = malloc((length/4)*sizeof(__m256i));
  __m256i * naMasks2 = malloc((length/4)*sizeof(__m256i));
  if (!(naMasks1 && naMasks2)) {
    printf("Not enough memory to allocate space for the naMasks\n");
    exit(-1);
  }

  for (uint64_t i = 0; i < (length - 3); i += 4) {
    naMasks1[i/4] =
      naMaskForGenotypePack(_mm256_load_si256((__m256i*)(x+i)));
    naMasks2[i/4] =
      naMaskForGenotypePack(_mm256_load_si256((__m256i*)(y+i)));
  }
  ibsVec(result, length, x, y, naMasks1, naMasks2);
  #else
  ibsVec(result, length, x, y);
  #endif // __AVX__
}

// samples in rows, genotypes in columns
EXPORT
void ibsMat(uint64_t* result, uint64_t nSamples, uint64_t nGenotypePacks, uint64_t* genotypes1, uint64_t* genotypes2) {
  __m256i * naMasks1 = malloc(nSamples*(nGenotypePacks/4)*sizeof(__m256i));
  __m256i * naMasks2 = malloc(nSamples*(nGenotypePacks/4)*sizeof(__m256i));
  if (!(naMasks1 && naMasks2)) {
    printf("Not enough memory to allocate space for the naMasks\n");
    exit(-1);
  }

  for (uint64_t i = 0; i != nSamples; ++i) {
    for (uint64_t j = 0; j < (nGenotypePacks - 3); j += 4) {
      naMasks1[i*(nGenotypePacks/4)+(j/4)] =
        naMaskForGenotypePack(_mm256_load_si256((__m256i*)(genotypes1+i*nGenotypePacks+j)));
      naMasks2[i*(nGenotypePacks/4)+(j/4)] =
        naMaskForGenotypePack(_mm256_load_si256((__m256i*)(genotypes2+i*nGenotypePacks+j)));
    }
    // NA mask for trailing genotype blocks will be calculated in the loop
  }

  for (uint64_t si = 0; si != nSamples; ++si) {
    for (uint64_t sj = 0; sj != nSamples; ++sj) {
      ibsVec(result + si*nSamples*3 + sj*3,
             nGenotypePacks,
             genotypes1 + si*nGenotypePacks,
             genotypes2 + sj*nGenotypePacks
             #ifdef __AVX__
             , naMasks1 + si*(nGenotypePacks/4)
             , naMasks2 + sj*(nGenotypePacks/4)
             #endif // __AVX__
             );
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
    __m256i allNA1 = { 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA };
    __m256i allNA2 = { 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA };
    ibs256_with_na(result, allNA1, allNA2);

    expect_equal("allNA", "%llu", result[0], 0ULL);
    expect_equal("allNA", "%llu", result[1], 0ULL);
    expect_equal("allNA", "%llu", result[2], 0ULL);
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    __m256i allHomRef1 = { 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 };
    __m256i allHomRef2 = { 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 };
    ibs256_with_na(result, allHomRef1, allHomRef2);

    expect_equal("allHomRef", "%llu", result[0], 0ULL);
    expect_equal("allHomRef", "%llu", result[1], 0ULL);
    expect_equal("allHomRef", "%llu", result[2], 128ULL);
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    __m256i allHet1 = { 0x5555555555555555, 0x5555555555555555, 0x5555555555555555, 0x5555555555555555 };
    __m256i allHet2 = { 0x5555555555555555, 0x5555555555555555, 0x5555555555555555, 0x5555555555555555 };
    ibs256_with_na(result, allHet1, allHet2);

    expect_equal("allHet", "%llu", result[0], 0ULL);
    expect_equal("allHet", "%llu", result[1], 0ULL);
    expect_equal("allHet", "%llu", result[2], 128ULL);
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    __m256i allHomAlt1 = { 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF };
    __m256i allHomAlt2 = { 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF };
    ibs256_with_na(result, allHomAlt1, allHomAlt2);

    expect_equal("allHomAlt", "%llu", result[0], 0ULL);
    expect_equal("allHomAlt", "%llu", result[1], 0ULL);
    expect_equal("allHomAlt", "%llu", result[2], 128ULL);
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    __m256i allHomAlt = { 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF };
    __m256i allHomRef = { 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 };
    ibs256_with_na(result, allHomAlt, allHomRef);

    expect_equal("homAlt v homRef", "%llu", result[0], 128ULL);
    expect_equal("homAlt v homRef", "%llu", result[1], 0ULL);
    expect_equal("homAlt v homRef", "%llu", result[2], 0ULL);
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    __m256i allHet = { 0x5555555555555555, 0x5555555555555555, 0x5555555555555555, 0x5555555555555555 };
    __m256i allHomRef = { 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 };
    ibs256_with_na(result, allHet, allHomRef);

    expect_equal("het v homRef", "%llu", result[0], 0ULL);
    expect_equal("het v homRef", "%llu", result[1], 128ULL);
    expect_equal("het v homRef", "%llu", result[2], 0ULL);
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    __m256i allHet = { 0x5555555555555555, 0x5555555555555555, 0x5555555555555555, 0x5555555555555555 };
    __m256i allHomAlt = { 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF };
    ibs256_with_na(result, allHet, allHomAlt);

    expect_equal("het v homAlt", "%llu", result[0], 0ULL);
    expect_equal("het v homAlt", "%llu", result[1], 128ULL);
    expect_equal("het v homAlt", "%llu", result[2], 0ULL);
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    __m256i allHet = { 0x5555555555555555, 0x5555555555555555, 0x5555555555555555, 0x5555555555555555 };
    __m256i allHomAltOneNA = { 0xBFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF };
    ibs256_with_na(result, allHet, allHomAltOneNA);

    expect_equal("het v homAltOneNA", "%llu", result[0], 0ULL);
    expect_equal("het v homAltOneNA", "%llu", result[1], 127ULL);
    expect_equal("het v homAltOneNA", "%llu", result[2], 0ULL);
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    __m256i allHet = { 0x5555555555555555, 0x5555555555555555, 0x5555555555555555, 0x5555555555555555 };
    __m256i allHomAltTwoNA = { 0xAFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF };
    ibs256_with_na(result, allHet, allHomAltTwoNA);

    expect_equal("het v homAltTwoNA", "%llu", result[0], 0ULL);
    expect_equal("het v homAltTwoNA", "%llu", result[1], 126ULL);
    expect_equal("het v homAltTwoNA", "%llu", result[2], 0ULL);
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    __m256i ref_het_alt_het = { 0x1DAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA };
    __m256i het_ref_het_alt = { 0x47AAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA };
    ibs256_with_na(result, ref_het_alt_het, het_ref_het_alt);

    expect_equal("ref_het_alt_het v het_ref_het_alt", "%llu", result[0], 0ULL);
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%llu", result[1], 4ULL);
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%llu", result[2], 0ULL);
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    __m256i ref_het_alt_het = { 0xAAAAAAAAAAAAAAAA, 0x1DAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA };
    __m256i het_ref_het_alt = { 0xAAAAAAAAAAAAAAAA, 0x47AAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA };
    ibs256_with_na(result, ref_het_alt_het, het_ref_het_alt);

    expect_equal("ref_het_alt_het v het_ref_het_alt", "%llu", result[0], 0ULL);
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%llu", result[1], 4ULL);
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%llu", result[2], 0ULL);
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    __m256i ref_het_alt_het = { 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0x1DAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA };
    __m256i het_ref_het_alt = { 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0x47AAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA };
    ibs256_with_na(result, ref_het_alt_het, het_ref_het_alt);

    expect_equal("ref_het_alt_het v het_ref_het_alt", "%llu", result[0], 0ULL);
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%llu", result[1], 4ULL);
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%llu", result[2], 0ULL);
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    __m256i ref_het_alt_het = { 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0x1DAAAAAAAAAAAAAA };
    __m256i het_ref_het_alt = { 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0x47AAAAAAAAAAAAAA };
    ibs256_with_na(result, ref_het_alt_het, het_ref_het_alt);

    expect_equal("ref_het_alt_het v het_ref_het_alt", "%llu", result[0], 0ULL);
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%llu", result[1], 4ULL);
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%llu", result[2], 0ULL);
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    __m256i ref_het_alt_het = { 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAA1D };
    __m256i het_ref_het_alt = { 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAA47 };
    ibs256_with_na(result, ref_het_alt_het, het_ref_het_alt);

    expect_equal("ref_het_alt_het v het_ref_het_alt", "%llu", result[0], 0ULL);
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%llu", result[1], 4ULL);
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%llu", result[2], 0ULL);
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    __m256i ref_het_alt_het = { 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAAAA };
    __m256i het_ref_het_alt = { 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAA47, 0xAAAAAAAAAAAAAAAA };
    ibs256_with_na(result, ref_het_alt_het, het_ref_het_alt);

    expect_equal("ref_het_alt_het v het_ref_het_alt", "%llu", result[0], 0ULL);
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%llu", result[1], 4ULL);
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%llu", result[2], 0ULL);
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    __m256i ref_het_alt_het = { 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA };
    __m256i het_ref_het_alt = { 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAA47, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA };
    ibs256_with_na(result, ref_het_alt_het, het_ref_het_alt);

    expect_equal("ref_het_alt_het v het_ref_het_alt", "%llu", result[0], 0ULL);
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%llu", result[1], 4ULL);
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%llu", result[2], 0ULL);
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    __m256i ref_het_alt_het = { 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA };
    __m256i het_ref_het_alt = { 0xAAAAAAAAAAAAAA47, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA };
    ibs256_with_na(result, ref_het_alt_het, het_ref_het_alt);

    expect_equal("ref_het_alt_het v het_ref_het_alt", "%llu", result[0], 0ULL);
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%llu", result[1], 4ULL);
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%llu", result[2], 0ULL);
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

    expect_equal("ibsVec2 ref_het_alt_het v het_ref_het_alt", "%llu", result[0], 0ULL);
    expect_equal("ibsVec2 ref_het_alt_het v het_ref_het_alt", "%llu", result[1], 8ULL);
    expect_equal("ibsVec2 ref_het_alt_het v het_ref_het_alt", "%llu", result[2], 0ULL);
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

    expect_equal("ibsVec2 7 ref_het_alt_het v 7 het_ref_het_alt", "%llu", result[0], 0ULL);
    expect_equal("ibsVec2 7 ref_het_alt_het v 7 het_ref_het_alt", "%llu", result[1], 7 * 4ULL);
    expect_equal("ibsVec2 7 ref_het_alt_het v 7 het_ref_het_alt", "%llu", result[2], 0ULL);
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    uint64_t ref_het_alt_het[7] =
      { 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D,
        0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D };
    ibsVec2(result, 7, ref_het_alt_het, ref_het_alt_het);

    expect_equal("ibsVec2 ref_het_alt_het v self", "%llu", result[0], 0ULL);
    expect_equal("ibsVec2 ref_het_alt_het v self", "%llu", result[1], 0ULL);
    expect_equal("ibsVec2 ref_het_alt_het v self", "%llu", result[2], 4*7ULL);
  }
  // ibsMat
  {
    uint64_t result[2*2*3] = { 0, 0, 0,  0, 0, 0,
                               0, 0, 0,  0, 0, 0 };
    uint64_t all_ref_het_alt_het[2*7] =
      { 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D,
        0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D };
    ibsMat(result, 2, 7, all_ref_het_alt_het, all_ref_het_alt_het);

    expect_equal("ibsMat identical 0 0, ibs0", "%llu", resultIndex(result, 2, 0, 0, 0), 0ULL);
    expect_equal("ibsMat identical 0 0, ibs1", "%llu", resultIndex(result, 2, 0, 0, 1), 0ULL);
    expect_equal("ibsMat identical 0 0, ibs2", "%llu", resultIndex(result, 2, 0, 0, 2), 4*7ULL);
    expect_equal("ibsMat identical 0 1, ibs0", "%llu", resultIndex(result, 2, 0, 1, 0), 0ULL);
    expect_equal("ibsMat identical 0 1, ibs1", "%llu", resultIndex(result, 2, 0, 1, 1), 0ULL);
    expect_equal("ibsMat identical 0 1, ibs2", "%llu", resultIndex(result, 2, 0, 1, 2), 4*7ULL);
    expect_equal("ibsMat identical 1 1, ibs0", "%llu", resultIndex(result, 2, 1, 1, 0), 0ULL);
    expect_equal("ibsMat identical 1 1, ibs1", "%llu", resultIndex(result, 2, 1, 1, 1), 0ULL);
    expect_equal("ibsMat identical 1 1, ibs2", "%llu", resultIndex(result, 2, 1, 1, 2), 4*7ULL);
  }
  {
    uint64_t result[2*2*3] = { 0, 0, 0,  0, 0, 0,
                               0, 0, 0,  0, 0, 0 };
    uint64_t one_ibs1_rest_ibs2[2*7] =
      { 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D,
        0xAAAAAAAAAAAAAA1F, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D };
    ibsMat(result, 2, 7, one_ibs1_rest_ibs2, one_ibs1_rest_ibs2);

    expect_equal("ibsMat one-ibs1 0 0, ibs0", "%llu", resultIndex(result, 2, 0, 0, 0), 0ULL);
    expect_equal("ibsMat one-ibs1 0 0, ibs1", "%llu", resultIndex(result, 2, 0, 0, 1), 0ULL);
    expect_equal("ibsMat one-ibs1 0 0, ibs2", "%llu", resultIndex(result, 2, 0, 0, 2), 4*7ULL);
    expect_equal("ibsMat one-ibs1 0 1, ibs0", "%llu", resultIndex(result, 2, 0, 1, 0), 0ULL);
    expect_equal("ibsMat one-ibs1 0 1, ibs1", "%llu", resultIndex(result, 2, 0, 1, 1), 1ULL);
    expect_equal("ibsMat one-ibs1 0 1, ibs2", "%llu", resultIndex(result, 2, 0, 1, 2), 3+4*6ULL);
    expect_equal("ibsMat one-ibs1 1 1, ibs0", "%llu", resultIndex(result, 2, 1, 1, 0), 0ULL);
    expect_equal("ibsMat one-ibs1 1 1, ibs1", "%llu", resultIndex(result, 2, 1, 1, 1), 0ULL);
    expect_equal("ibsMat one-ibs1 1 1, ibs2", "%llu", resultIndex(result, 2, 1, 1, 2), 4*7ULL);
  }

  if (failures != 0) {
    printf("%llu test(s) failed.\n", failures);
  } else {
    printf("%llu test(s) succeeded.\n", successes);
  }

  return 0;
}
