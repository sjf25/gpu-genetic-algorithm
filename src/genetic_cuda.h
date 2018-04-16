#ifndef GENETIC_CUDA_H
#define GENETIC_CUDA_H

#include <cstddef>
#include <cstdint>

typedef double (*fitness_t)(uint8_t*);

void run_genetic_cuda(size_t, size_t, double, double, unsigned, fitness_t);

#endif
