#include <x86intrin.h>
#include <avxintrin.h>
#include <avx2intrin.h>
#include <popcntintrin.h>
#include <stdint.h>
#include <stdio.h>

__m256i allones = { 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF };
__m256i rightAllele = { 0x5555555555555555, 0x5555555555555555, 0x5555555555555555, 0x5555555555555555 };

void ibs256(uint64_t* result, __m256i x, __m256i y) {
  __m256i nxor = _mm256_xor_si256(_mm256_xor_si256(x, y), allones);
  // if both bits are one, then the genotype is missing
  __m256i xna = _mm256_and_si256(_mm256_srli_si256(x, 1), x);
  __m256i yna = _mm256_and_si256(_mm256_srli_si256(y, 1), y);
  // if either sample is missing a genotype, we ignore that genotype pair
  __m256i na = _mm256_or_si256(xna, yna);
  // 1. shift the left alleles over the right ones
  // 2. and the alleles
  // 3. mask to the right ones
  __m256i ibs2 = _mm256_and_si256(_mm256_and_si256(_mm256_and_si256(_mm256_srli_si256(nxor, 1), nxor), rightAllele), na);
  // 4. popcnt
  uint64_t ibs2sum = _mm_popcnt_u64(_mm256_extract_epi64(ibs2, 0));
  ibs2sum += _mm_popcnt_u64(_mm256_extract_epi64(ibs2, 1));
  ibs2sum += _mm_popcnt_u64(_mm256_extract_epi64(ibs2, 2));
  ibs2sum += _mm_popcnt_u64(_mm256_extract_epi64(ibs2, 3));

  // 1. shift the left alleles over the right ones
  // 2. or the alleles
  // 3. mask to the right ones
  __m256i ibs1 = _mm256_and_si256(_mm256_and_si256(_mm256_xor_si256(_mm256_srli_si256(nxor, 1), nxor), rightAllele), na);
  // 4. popcnt
  uint64_t ibs1sum = _mm_popcnt_u64(_mm256_extract_epi64(ibs1, 0));
  ibs1sum += _mm_popcnt_u64(_mm256_extract_epi64(ibs1, 1));
  ibs1sum += _mm_popcnt_u64(_mm256_extract_epi64(ibs1, 2));
  ibs1sum += _mm_popcnt_u64(_mm256_extract_epi64(ibs1, 3));

  // 1. shift the left alleles over the right ones
  // 2. or the alleles
  // 3. negate the alleles
  // 4. mask to the right ones
  __m256i ibs0 = _mm256_and_si256(_mm256_andnot_si256(rightAllele, _mm256_or_si256(_mm256_srli_si256(nxor, 1), nxor)), na);
  // 5. popcnt
  uint64_t ibs0sum = _mm_popcnt_u64(_mm256_extract_epi64(ibs0, 0));
  ibs0sum += _mm_popcnt_u64(_mm256_extract_epi64(ibs0, 1));
  ibs0sum += _mm_popcnt_u64(_mm256_extract_epi64(ibs0, 2));
  ibs0sum += _mm_popcnt_u64(_mm256_extract_epi64(ibs0, 3));

  *(result + 0) += ibs0sum;
  *(result + 1) += ibs1sum;
  *(result + 2) += ibs2sum;
}

uint8_t bytepopcnt(uint8_t x) {
  uint8_t cnt = 0;
  for (int i = 0; i < 8; ++i) {
    cnt += (x >> i) & 0xFF;
  }
  return cnt;
}

void ibsVec(uint64_t* result, uint64_t length, uint64_t* x, uint64_t* y) {
  uint64_t i;
  for (i = 0; i < (length - 3); i += 4) {
    __m256i x256 = _mm256_set_epi64x(x[i], x[i+1], x[i+2], x[i+3]);
    __m256i y256 = _mm256_set_epi64x(y[i], y[i+1], y[i+2], y[i+3]);
    ibs256(result, x256, y256);
  }
  while (i < length) {
    uint64_t rightAllele64 = 0xFFFFFFFFFFFFFFFF;
    uint64_t xb = x[i];
    uint64_t yb = y[i];
    uint64_t nxor = ~(xb ^ yb);
    uint64_t xna = (xb >> 1) & xb;
    uint64_t yna = (yb >> 1) & yb;
    uint64_t na = xna | yna;
    *(result+2) += bytepopcnt(((nxor >> 1) & nxor) & na & rightAllele64);
    *(result+1) += bytepopcnt(((nxor >> 1) ^ nxor) & na & rightAllele64);
    *(result+0) += bytepopcnt(~((nxor >> 1) | nxor) & na & rightAllele64);
  }
}

// samples in rows, genotypes in columns
void ibsMat(uint64_t* result, uint64_t nSamples, uint64_t nGenotypePacks, uint64_t* genotypes1, uint64_t* genotypes2) {
  for (uint64_t si = 0; si != nSamples; ++si) {
    for (uint64_t sj = 0; sj != nSamples; ++sj) {
      ibsVec(result + si*nSamples*3*sizeof(uint64_t) + sj*3*sizeof(uint64_t),
             nGenotypePacks,
             genotypes1 + si*nGenotypePacks*sizeof(uint64_t),
             genotypes2 + sj*nGenotypePacks*sizeof(uint64_t));
    }
  }
}

int main(int argc, char** argv) {
  if (argc != 2) {
    return -1;
  }
  int itr = atoi(argv[1]);
  uint64_t arr[3] = { 0, 0, 0 };

  for (int i = 0; i < itr; ++i ) {
    uint64_t x0 = 0; //(((uint64_t)rand()) << 32) | rand();
    uint64_t x1 = 0;//(((uint64_t)rand()) << 32) | rand();
    uint64_t x2 = 0; //(((uint64_t)rand()) << 32) | rand();
    uint64_t x3 = 0;//(((uint64_t)rand()) << 32) | rand();
    uint64_t y0 = 0;//(((uint64_t)rand()) << 32) | rand();
    uint64_t y1 = 0;//(((uint64_t)rand()) << 32) | rand();
    uint64_t y2 = 0;//(((uint64_t)rand()) << 32) | rand();
    uint64_t y3 = 0;//(((uint64_t)rand()) << 32) | rand();
    __m256i x = _mm256_set_epi64x(x0, x1, x2, x3);
    __m256i y = _mm256_set_epi64x(y0, y1, y2, y3);
    ibs256(arr, x, y);
  }

  printf("%llu", arr[0]);

  return 0;
}
