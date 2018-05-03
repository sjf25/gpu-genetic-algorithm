#ifndef UTILS_H
#define UTILS_H

__device__ void prefix_sum(double* __restrict__, double* __restrict__, unsigned);
__device__ void parallel_max(double* __restrict__, double* __restrict__, unsigned);

#endif
