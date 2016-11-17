#include <simdpp/simd.h>
#include <inttypes.h>

void ibs256(__restrict__ uint64_t* result, uint64v x, uint64v y, uint64v xna, uint64v yna);
uint64v naMaskForGenotypePack(uint64v block);
void ibs256_with_na(__restrict__ uint64_t* result, uint64v x, uint64v y);
void ibsVec(__restrict__ uint64_t* result,
            uint64_t length,
            __restrict__ uint64_t* x,
            __restrict__ uint64_t* y,
            __restrict__ uint64v * x_na_masks,
            __restrict__ uint64v * y_na_masks);
void allocateNaMasks(uint64v ** mask1,
                     uint64v ** mask2,
                     uint64_t nSamples,
                     uint64_t nGenotypePacks,
                     __restrict__ uint64_t* x,
                     __restrict__ uint64_t* y);
void ibsVec2(__restrict__ uint64_t* result,
             uint64_t nGenotypePacks,
             __restrict__ uint64_t* x,
             __restrict__ uint64_t* y);
extern "C"
void ibsMat(uint64_t* result, uint64_t nSamples, uint64_t nGenotypePacks, uint64_t* genotypes1, uint64_t* genotypes2);
