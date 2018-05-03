#ifndef MAXONE_H
#define MAXONE_H

#include <cstdint>
#include <cstddef>

void set_length(unsigned);
double maxone_fitness(uint8_t*);
__device__ double d_maxone_fitness(uint8_t*);

#endif
