#include "utils.h"
#include <cstdio>

// TODO: see if there's a better way to do all of this

// TODO: consider inlining the following two functions
__device__ static double add(double x, double y) { return x + y; }
__device__ static double max_wrapper(double x, double y) { return fmax(x, y); }

template <typename F>
__device__ static void parallel_reduce(double* arr, double* result, unsigned n, F func) {

	unsigned idx = threadIdx.x;
	__shared__ double* prefix_arr;
	if(idx == 0) {
		prefix_arr = new double[n];
		memcpy(prefix_arr, arr, sizeof(double) * n);	
	}
	__syncthreads();
	for(unsigned i = 1; i < n; i *= 2) {
		if(idx >= i) {
			//prefix_arr[idx] += prefix_arr[idx-i];
			prefix_arr[idx] = func(prefix_arr[idx], prefix_arr[idx-i]);
		}
		__syncthreads();
	}
	if(idx == 0) {
		*result = prefix_arr[n-1];
		delete[] prefix_arr;
	}
}

// works for single block with multiple threads
__device__ void prefix_sum(double* arr, double* result, unsigned n) {
	parallel_reduce(arr, result, n, add);
}

// works for single block with multiple threads
__device__ void parallel_max(double* arr, double* result, unsigned n) {
	parallel_reduce(arr, result, n, max_wrapper);
}
