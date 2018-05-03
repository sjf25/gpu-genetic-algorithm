#include "maxone.h"
#include <iostream>
#include <cassert>

unsigned length;
__constant__ unsigned d_length;

void set_length(unsigned len) {
	length = len;
	cudaMemcpyToSymbol(d_length, &length, sizeof(unsigned));
	cudaDeviceSynchronize();
	cudaError err2 = cudaGetLastError();
	if(err2 != cudaSuccess) {
		std::cerr << "error when memcpying to d_num_vertices\n";
		exit(1);
	}
}

double maxone_fitness(uint8_t* member) {
	double member_fitness = 0.0;
	for(unsigned i = 0; i < length; i++) {
		assert(member[i] == 0 || member[i] == 1);
		member_fitness += member[i];
	}
	return member_fitness;
}

__device__ double d_maxone_fitness(uint8_t* member) {
	double member_fitness = 0.0;
	for(unsigned i = 0; i < d_length; i++) {
		assert(member[i] == 0 || member[i] == 1);
		member_fitness += member[i];
	}
	return member_fitness;
}
