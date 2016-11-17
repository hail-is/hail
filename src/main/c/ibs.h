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
#define HAIL_VECTOR_WIDTH SIMDPP_FAST_INT64_SIZE
#else
#define HAIL_VECTOR_WIDTH HAIL_OVERRIDE_WIDTH
#endif // HAIL_OVERRIDE_WIDTH

using hailvec = uint64<HAIL_VECTOR_WIDTH>;

void ibs256(uint64_t* __restrict__ result, hailvec x, hailvec y, hailvec xna, hailvec yna);
hailvec naMaskForGenotypePack(hailvec block);
void ibs256_with_na(uint64_t* __restrict__ result, hailvec x, hailvec y);
void ibsVec(uint64_t* __restrict__ result,
            uint64_t length,
            uint64_t* __restrict__ x,
            uint64_t* __restrict__ y,
            hailvec* __restrict__ x_na_masks,
            hailvec* __restrict__ y_na_masks);
void allocateNaMasks(hailvec ** __restrict__ mask1,
                     hailvec ** __restrict__ mask2,
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
