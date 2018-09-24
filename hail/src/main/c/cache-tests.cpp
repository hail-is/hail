#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <time.h>
#include "ibs.h"

#define N_SAMPLES 1024
#ifndef ITERATIONS
#define ITERATIONS 100
#endif

int main(int argc, char** argv) {
  printf("in kb, iterations: %d %d\n", CACHE_SIZE_PER_MATRIX_IN_KB, ITERATIONS);
  if (argc != 1) {
    printf("Expected zero arguments.\n");
    return -1;
  }

  uint64_t * genotypes1 = 0;
  int err1 = posix_memalign((void **)&genotypes1, 32, ITERATIONS*N_SAMPLES*NUMBER_OF_UINT64_GENOTYPE_PACKS_PER_ROW*sizeof(uint64_t));
  uint64_t * genotypes2 = 0;
  int err2 = posix_memalign((void **)&genotypes2, 32, ITERATIONS*N_SAMPLES*NUMBER_OF_UINT64_GENOTYPE_PACKS_PER_ROW*sizeof(uint64_t));
  if (err1 || err2) {
    printf("Not enough memory to allocate space for the genotypes: %d %d\n", err1, err2);
    exit(-1);
  }
  uint64_t * result = (uint64_t*)malloc(N_SAMPLES*N_SAMPLES*3*sizeof(uint64_t));

  clock_t t1, t2;

  t1 = clock();
  for (int i = 0; i < ITERATIONS; ++i) {
    ibsMat(result,
           N_SAMPLES,
           NUMBER_OF_UINT64_GENOTYPE_PACKS_PER_ROW,
           genotypes1+i*N_SAMPLES*NUMBER_OF_UINT64_GENOTYPE_PACKS_PER_ROW,
           genotypes2+i*N_SAMPLES*NUMBER_OF_UINT64_GENOTYPE_PACKS_PER_ROW);
  }
  t2 = clock();

  free(genotypes1);
  free(genotypes2);

  printf("%" PRIu64 "  ", result[0]);

  float diff = ((float)(t2 - t1) / CLOCKS_PER_SEC) / ITERATIONS;
  printf("%f\n",diff);
}
