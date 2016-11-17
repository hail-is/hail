#include <simdpp/simd.h>
#include <inttypes.h>

using namespace simdpp;

void ibs256(uint64_t* __restrict__ result, uint64v x, uint64v y, uint64v xna, uint64v yna);
uint64v naMaskForGenotypePack(uint64v block);
void ibs256_with_na(uint64_t* __restrict__ result, uint64v x, uint64v y);
void ibsVec(uint64_t* __restrict__ result,
            uint64_t length,
            uint64_t* __restrict__ x,
            uint64_t* __restrict__ y,
            uint64v* __restrict__ x_na_masks,
            uint64v* __restrict__ y_na_masks);
void allocateNaMasks(uint64v ** __restrict__ mask1,
                     uint64v ** __restrict__ mask2,
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
