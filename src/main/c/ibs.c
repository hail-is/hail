#include <x86intrin.h>
#include <avxintrin.h>
#include <avx2intrin.h>
#include <popcntintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <inttypes.h>

#define EXPORT __attribute__((visibility("default")))

#define fmt256 "%016llx  %016llx  %016llx  %016llx"
// XXX: only call with identifiers
#define splat256(x) _mm256_extract_epi64((x), 0), _mm256_extract_epi64((x), 1), _mm256_extract_epi64((x), 2), _mm256_extract_epi64((x), 3)

#define echo(x) printf(#x " = %016llx\n", x)

__m256i allones = { 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF };
__m256i rightAllele = { 0x5555555555555555, 0x5555555555555555, 0x5555555555555555, 0x5555555555555555 };
__m256i allNA = { 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA };

void ibs256(uint64_t* result, __m256i x, __m256i y) {
  __m256i nxor = _mm256_xor_si256(_mm256_xor_si256(x, y), allones);

  // if both bits are one, then the genotype is missing
  __m256i xna_tmp = _mm256_xor_si256(_mm256_xor_si256(allNA, x), allones);
  __m256i xna = _mm256_and_si256(_mm256_srli_epi64(xna_tmp, 1), xna_tmp);
  __m256i yna_tmp = _mm256_xor_si256(_mm256_xor_si256(allNA, y), allones);
  __m256i yna = _mm256_and_si256(_mm256_srli_epi64(yna_tmp, 1), yna_tmp);
  // if either sample is missing a genotype, we ignore that genotype pair
  __m256i na = _mm256_and_si256(_mm256_or_si256(xna, yna), rightAllele);
  // 1. shift the left alleles over the right ones
  // 2. and the alleles
  // 3. mask to the right ones
  __m256i ibs2 = _mm256_andnot_si256(na, _mm256_and_si256(_mm256_and_si256(_mm256_srli_epi64(nxor, 1), nxor), rightAllele));
  // 4. popcnt
  uint64_t ibs2sum = _mm_popcnt_u64(_mm256_extract_epi64(ibs2, 0));
  ibs2sum += _mm_popcnt_u64(_mm256_extract_epi64(ibs2, 1));
  ibs2sum += _mm_popcnt_u64(_mm256_extract_epi64(ibs2, 2));
  ibs2sum += _mm_popcnt_u64(_mm256_extract_epi64(ibs2, 3));

  // 1. shift the left alleles over the right ones
  // 2. or the alleles
  // 3. mask to the right ones
  __m256i ibs1 = _mm256_andnot_si256(na, _mm256_and_si256(_mm256_xor_si256(_mm256_srli_epi64(nxor, 1), nxor), rightAllele));
  // 4. popcnt
  uint64_t ibs1sum = _mm_popcnt_u64(_mm256_extract_epi64(ibs1, 0));
  ibs1sum += _mm_popcnt_u64(_mm256_extract_epi64(ibs1, 1));
  ibs1sum += _mm_popcnt_u64(_mm256_extract_epi64(ibs1, 2));
  ibs1sum += _mm_popcnt_u64(_mm256_extract_epi64(ibs1, 3));

  // 1. shift the left alleles over the right ones
  // 2. or the alleles
  // 3. negate the alleles
  // 4. mask to the right ones
  __m256i ibs0 = _mm256_andnot_si256(na, _mm256_andnot_si256(_mm256_or_si256(_mm256_srli_epi64(nxor, 1), nxor), rightAllele));
  // 5. popcnt
  uint64_t ibs0sum = _mm_popcnt_u64(_mm256_extract_epi64(ibs0, 0));
  ibs0sum += _mm_popcnt_u64(_mm256_extract_epi64(ibs0, 1));
  ibs0sum += _mm_popcnt_u64(_mm256_extract_epi64(ibs0, 2));
  ibs0sum += _mm_popcnt_u64(_mm256_extract_epi64(ibs0, 3));

  result[0] += ibs0sum;
  result[1] += ibs1sum;
  result[2] += ibs2sum;
}

void ibsVec(uint64_t* result, uint64_t length, uint64_t* x, uint64_t* y) {
  uint64_t i;
  /* printf("x 0x%" PRIXPTR "\n", (uintptr_t)x); */
  /* printf("y 0x%" PRIXPTR "\n", (uintptr_t)y); */
  for (i = 0; i < (length - 3); i += 4) {
    /* printf("i %llu\n", i); */
    /* printf("x %llx  %llx  %llx  %llx\n", x[i], x[i+1], x[i+2], x[i+3]); */
    /* printf("y %llx  %llx  %llx  %llx\n", y[i], y[i+1], y[i+2], y[i+3]); */
    __m256i x256 = _mm256_set_epi64x(x[i], x[i+1], x[i+2], x[i+3]);
    __m256i y256 = _mm256_set_epi64x(y[i], y[i+1], y[i+2], y[i+3]);
    ibs256(result, x256, y256);
  }
  while (i < length) {
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
    ++i;
  }
}

// samples in rows, genotypes in columns
EXPORT
void ibsMat(uint64_t* result, uint64_t nSamples, uint64_t nGenotypePacks, uint64_t* genotypes1, uint64_t* genotypes2) {
  /* printf("genotypes1 0x%" PRIXPTR "\n", (uintptr_t)genotypes1); */
  /* printf("genotypes2 0x%" PRIXPTR "\n", (uintptr_t)genotypes2); */
  /* printf("sizeof(uint64_t) %lu\n", sizeof(uint64_t)); */
  for (uint64_t si = 0; si != nSamples; ++si) {
    /* printf("si %llu\n", si); */
    for (uint64_t sj = 0; sj != nSamples; ++sj) {
      /* printf("**** si,sj %llu, %llu\n", si, sj); */
      ibsVec(result + si*nSamples*3 + sj*3,
             nGenotypePacks,
             genotypes1 + si*nGenotypePacks,
             genotypes2 + sj*nGenotypePacks);
    }
  }
}

#define expect(name, x) if (!(x)) { ++failures; printf(name ": expected " #x " to be true, but was false\n\n"); } else { ++successes; }
#define expect_equal(name, fmt, x, y) if ((x) != (y)) { ++failures; printf(name ": expected " #x " to equal " #y ", but actually got " fmt " and " fmt"\n\n", x, y); } else { ++successes; }

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
    ibs256(result, allNA1, allNA2);

    expect_equal("allNA", "%llu", result[0], 0ULL);
    expect_equal("allNA", "%llu", result[1], 0ULL);
    expect_equal("allNA", "%llu", result[2], 0ULL);
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    __m256i allHomRef1 = { 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 };
    __m256i allHomRef2 = { 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 };
    ibs256(result, allHomRef1, allHomRef2);

    expect_equal("allHomRef", "%llu", result[0], 0ULL);
    expect_equal("allHomRef", "%llu", result[1], 0ULL);
    expect_equal("allHomRef", "%llu", result[2], 128ULL);
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    __m256i allHet1 = { 0x5555555555555555, 0x5555555555555555, 0x5555555555555555, 0x5555555555555555 };
    __m256i allHet2 = { 0x5555555555555555, 0x5555555555555555, 0x5555555555555555, 0x5555555555555555 };
    ibs256(result, allHet1, allHet2);

    expect_equal("allHet", "%llu", result[0], 0ULL);
    expect_equal("allHet", "%llu", result[1], 0ULL);
    expect_equal("allHet", "%llu", result[2], 128ULL);
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    __m256i allHomAlt1 = { 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF };
    __m256i allHomAlt2 = { 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF };
    ibs256(result, allHomAlt1, allHomAlt2);

    expect_equal("allHomAlt", "%llu", result[0], 0ULL);
    expect_equal("allHomAlt", "%llu", result[1], 0ULL);
    expect_equal("allHomAlt", "%llu", result[2], 128ULL);
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    __m256i allHomAlt = { 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF };
    __m256i allHomRef = { 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 };
    ibs256(result, allHomAlt, allHomRef);

    expect_equal("homAlt v homRef", "%llu", result[0], 128ULL);
    expect_equal("homAlt v homRef", "%llu", result[1], 0ULL);
    expect_equal("homAlt v homRef", "%llu", result[2], 0ULL);
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    __m256i allHet = { 0x5555555555555555, 0x5555555555555555, 0x5555555555555555, 0x5555555555555555 };
    __m256i allHomRef = { 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 };
    ibs256(result, allHet, allHomRef);

    expect_equal("het v homRef", "%llu", result[0], 0ULL);
    expect_equal("het v homRef", "%llu", result[1], 128ULL);
    expect_equal("het v homRef", "%llu", result[2], 0ULL);
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    __m256i allHet = { 0x5555555555555555, 0x5555555555555555, 0x5555555555555555, 0x5555555555555555 };
    __m256i allHomAlt = { 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF };
    ibs256(result, allHet, allHomAlt);

    expect_equal("het v homAlt", "%llu", result[0], 0ULL);
    expect_equal("het v homAlt", "%llu", result[1], 128ULL);
    expect_equal("het v homAlt", "%llu", result[2], 0ULL);
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    __m256i allHet = { 0x5555555555555555, 0x5555555555555555, 0x5555555555555555, 0x5555555555555555 };
    __m256i allHomAltOneNA = { 0xBFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF };
    ibs256(result, allHet, allHomAltOneNA);

    expect_equal("het v homAltOneNA", "%llu", result[0], 0ULL);
    expect_equal("het v homAltOneNA", "%llu", result[1], 127ULL);
    expect_equal("het v homAltOneNA", "%llu", result[2], 0ULL);
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    __m256i allHet = { 0x5555555555555555, 0x5555555555555555, 0x5555555555555555, 0x5555555555555555 };
    __m256i allHomAltTwoNA = { 0xAFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF };
    ibs256(result, allHet, allHomAltTwoNA);

    expect_equal("het v homAltTwoNA", "%llu", result[0], 0ULL);
    expect_equal("het v homAltTwoNA", "%llu", result[1], 126ULL);
    expect_equal("het v homAltTwoNA", "%llu", result[2], 0ULL);
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    __m256i ref_het_alt_het = { 0x1DAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA };
    __m256i het_ref_het_alt = { 0x47AAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA };
    ibs256(result, ref_het_alt_het, het_ref_het_alt);

    expect_equal("ref_het_alt_het v het_ref_het_alt", "%llu", result[0], 0ULL);
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%llu", result[1], 4ULL);
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%llu", result[2], 0ULL);
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    __m256i ref_het_alt_het = { 0xAAAAAAAAAAAAAAAA, 0x1DAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA };
    __m256i het_ref_het_alt = { 0xAAAAAAAAAAAAAAAA, 0x47AAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA };
    ibs256(result, ref_het_alt_het, het_ref_het_alt);

    expect_equal("ref_het_alt_het v het_ref_het_alt", "%llu", result[0], 0ULL);
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%llu", result[1], 4ULL);
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%llu", result[2], 0ULL);
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    __m256i ref_het_alt_het = { 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0x1DAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA };
    __m256i het_ref_het_alt = { 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0x47AAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA };
    ibs256(result, ref_het_alt_het, het_ref_het_alt);

    expect_equal("ref_het_alt_het v het_ref_het_alt", "%llu", result[0], 0ULL);
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%llu", result[1], 4ULL);
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%llu", result[2], 0ULL);
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    __m256i ref_het_alt_het = { 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0x1DAAAAAAAAAAAAAA };
    __m256i het_ref_het_alt = { 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0x47AAAAAAAAAAAAAA };
    ibs256(result, ref_het_alt_het, het_ref_het_alt);

    expect_equal("ref_het_alt_het v het_ref_het_alt", "%llu", result[0], 0ULL);
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%llu", result[1], 4ULL);
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%llu", result[2], 0ULL);
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    __m256i ref_het_alt_het = { 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAA1D };
    __m256i het_ref_het_alt = { 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAA47 };
    ibs256(result, ref_het_alt_het, het_ref_het_alt);

    expect_equal("ref_het_alt_het v het_ref_het_alt", "%llu", result[0], 0ULL);
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%llu", result[1], 4ULL);
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%llu", result[2], 0ULL);
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    __m256i ref_het_alt_het = { 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAAAA };
    __m256i het_ref_het_alt = { 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAA47, 0xAAAAAAAAAAAAAAAA };
    ibs256(result, ref_het_alt_het, het_ref_het_alt);

    expect_equal("ref_het_alt_het v het_ref_het_alt", "%llu", result[0], 0ULL);
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%llu", result[1], 4ULL);
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%llu", result[2], 0ULL);
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    __m256i ref_het_alt_het = { 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA };
    __m256i het_ref_het_alt = { 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAA47, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA };
    ibs256(result, ref_het_alt_het, het_ref_het_alt);

    expect_equal("ref_het_alt_het v het_ref_het_alt", "%llu", result[0], 0ULL);
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%llu", result[1], 4ULL);
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%llu", result[2], 0ULL);
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    __m256i ref_het_alt_het = { 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA };
    __m256i het_ref_het_alt = { 0xAAAAAAAAAAAAAA47, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA };
    ibs256(result, ref_het_alt_het, het_ref_het_alt);

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
    ibsVec(result, 7, ref_het_alt_het, het_ref_het_alt);

    expect_equal("ibsVec ref_het_alt_het v het_ref_het_alt", "%llu", result[0], 0ULL);
    expect_equal("ibsVec ref_het_alt_het v het_ref_het_alt", "%llu", result[1], 8ULL);
    expect_equal("ibsVec ref_het_alt_het v het_ref_het_alt", "%llu", result[2], 0ULL);
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    uint64_t ref_het_alt_het[7] =
      { 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D,
        0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D };
    uint64_t het_ref_het_alt[7] =
      { 0xAAAAAAAAAAAAAA47, 0xAAAAAAAAAAAAAA47, 0xAAAAAAAAAAAAAA47, 0xAAAAAAAAAAAAAA47,
        0xAAAAAAAAAAAAAA47, 0xAAAAAAAAAAAAAA47, 0xAAAAAAAAAAAAAA47 };
    ibsVec(result, 7, ref_het_alt_het, het_ref_het_alt);

    expect_equal("ibsVec 7 ref_het_alt_het v 7 het_ref_het_alt", "%llu", result[0], 0ULL);
    expect_equal("ibsVec 7 ref_het_alt_het v 7 het_ref_het_alt", "%llu", result[1], 7 * 4ULL);
    expect_equal("ibsVec 7 ref_het_alt_het v 7 het_ref_het_alt", "%llu", result[2], 0ULL);
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    uint64_t ref_het_alt_het[7] =
      { 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D,
        0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D };
    ibsVec(result, 7, ref_het_alt_het, ref_het_alt_het);

    expect_equal("ibsVec ref_het_alt_het v self", "%llu", result[0], 0ULL);
    expect_equal("ibsVec ref_het_alt_het v self", "%llu", result[1], 0ULL);
    expect_equal("ibsVec ref_het_alt_het v self", "%llu", result[2], 4*7ULL);
  }

  if (failures != 0) {
    printf("%llu test(s) failed.\n", failures);
  } else {
    printf("%llu test(s) succeeded.\n", successes);
  }

  return 0;
}
