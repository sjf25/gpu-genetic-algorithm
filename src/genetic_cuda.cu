#include "genetic_cuda.h"

__constant__ size_t pop_size;
__constant__ size_t member_size;
__constant__ double crossover_rate;
__constant__ double mutation_rate;

__global__ void genetic_kernel() {
}

void run_genetic_seq(size_t p_size, size_t m_size, double cr_rate,
		double m_rate, unsigned max_iter, double (*fitness_func)(uint8_t*)) {
	cudaMemcpyToSymbol(pop_size, &p_size, sizeof(size_t));
	cudaMemcpyToSymbol(member_size, &m_size, sizeof(size_t));
	cudaMemcpyToSymbol(crossover_rate, &cr_rate, sizeof(double));
	cudaMemcpyToSymbol(mutation_rate, &m_rate, sizeof(double));
	cudaDeviceSynchronize();
}
