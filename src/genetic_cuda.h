#ifndef GENETIC_CUDA_H
#define GENETIC_CUDA_H

#include <cstddef>
#include <cstdint>
#include <iostream>

typedef double (*fitness_t)(uint8_t*);

double run_genetic_cuda(unsigned, size_t, size_t, double, double, unsigned, fitness_t);

#endif
