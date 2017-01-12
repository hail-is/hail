#ifndef HAIL_OVERRIDE_ARCH
#if __AVX2__
#define SIMDPP_ARCH_X86_AVX2
#elif __AVX__
#define SIMDPP_ARCH_X86_AVX
#elif __SSE4_1__
#define SIMDPP_ARCH_X86_SSE4_1
#elif __SSSE3__
#define SIMDPP_ARCH_X86_SSSE3
#elif __SSE3__
#define SIMDPP_ARCH_X86_SSE3
#elif __SSE2__
#define SIMDPP_ARCH_X86_SSE2
#endif
#endif // HAIL_OVERRIDE_ARCH

#include <simdpp/simd.h>
#include <inttypes.h>

using namespace simdpp;

#ifndef HAIL_OVERRIDE_WIDTH
#define UINT64_VECTOR_SIZE SIMDPP_FAST_INT64_SIZE
#else
#define UINT64_VECTOR_SIZE HAIL_OVERRIDE_WIDTH
#endif // HAIL_OVERRIDE_WIDTH

using uint64vector = uint64<UINT64_VECTOR_SIZE>;

// should be equal to chunkSize from IBD.scala
#ifndef NUMBER_OF_GENOTYPES_PER_ROW
#define NUMBER_OF_GENOTYPES_PER_ROW 1024
#endif
#define NUMBER_OF_UINT64_GENOTYPE_PACKS_PER_ROW (NUMBER_OF_GENOTYPES_PER_ROW / 32)

#if (NUMBER_OF_UINT64_GENOTYPE_PACKS_PER_ROW % UINT64_VECTOR_SIZE) != 0
#error "genotype packs per row, NUMBER_OF_UINT64_GENOTYPE_PACKS_PER_ROW, must be multuple of vector width, UINT64_VECTOR_SIZE."
#endif

#ifndef CACHE_SIZE_PER_MATRIX_IN_KB
#define CACHE_SIZE_PER_MATRIX_IN_KB 4
#endif

#ifndef CACHE_SIZE_IN_MATRIX_ROWS
#define CACHE_SIZE_IN_MATRIX_ROWS (((CACHE_SIZE_PER_MATRIX_IN_KB * 1024) / 64) / NUMBER_OF_UINT64_GENOTYPE_PACKS_PER_ROW)
#endif

void ibs256(uint64_t* __restrict__ result, uint64vector x, uint64vector y, uint64vector xna, uint64vector yna);
uint64vector naMaskForGenotypePack(uint64vector block);
void ibs256_with_na(uint64_t* __restrict__ result, uint64vector x, uint64vector y);
void ibsVec(uint64_t* __restrict__ result,
            uint64_t length,
            uint64_t* __restrict__ x,
            uint64_t* __restrict__ y,
            uint64vector* __restrict__ x_na_masks,
            uint64vector* __restrict__ y_na_masks);
void allocateNaMasks(uint64vector ** __restrict__ mask1,
                     uint64vector ** __restrict__ mask2,
                     uint64_t nSamples,
                     uint64_t nGenotypePacks,
                     uint64_t* __restrict__ x,
                     uint64_t* __restrict__ y);
void ibsVec2(uint64_t* __restrict__ result,
             uint64_t nGenotypePacks,
             uint64_t* __restrict__ x,
             uint64_t* __restrict__ y);
extern "C"
void ibsMat(uint64_t* __restrict__ result,
            uint64_t nSamples,
            uint64_t nGenotypePacks,
            uint64_t* __restrict__ genotypes1,
            uint64_t* __restrict__ genotypes2);
