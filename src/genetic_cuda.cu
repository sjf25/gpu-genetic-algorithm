#include "genetic_cuda.h"
#include <curand_kernel.h>
#include <thrust/random.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>

__constant__ static size_t pop_size;
__constant__ static size_t member_size;
__constant__ static double crossover_rate;
__constant__ static double mutation_rate;
__constant__ static unsigned max_iterations;
__constant__ double (*fitness)(uint8_t*);



inline void check_cuda_error(std::string msg) {
        cudaError err = cudaGetLastError();
        if(cudaSuccess != err) {
                std::cerr << "\033[1;31mcuda error: " << msg << std::endl;
                std::cerr << "\t" << cudaGetErrorString(err) << std::endl;
                exit(1);
        }
}

static __device__ uint8_t* get_member(uint8_t* member_array, int i) {
	return &member_array[i * member_size];
}

__device__ void init_population(int idx, uint8_t* population,
	thrust::default_random_engine& rand_gen) {
	thrust::uniform_int_distribution<uint8_t> dist(0, 1);
	for(unsigned i = 0; i < member_size; i++) {
		population[member_size * idx + i] = dist(rand_gen);
	}
}

__device__ void record_fitness(int idx, double* fitness_arr,
	uint8_t* population) {
	double current_fitness = fitness(get_member(population, idx));
	fitness_arr[idx] = current_fitness;
}

__device__ void selection(int idx, uint8_t* population, double* fitness_arr) {
	__shared__ double fitness_sum;
	record_fitness(idx, fitness_arr, population);
	if(idx == 0) {
		//fitness_sum = thrust::reduce(fitness_arr, fitness_arr + pop_size, 0.0, thrust::plus<double>());
	}
	__syncthreads();
}

// I think it's safe to assume that pop_size <= 1024
// so just one block with many threads
// TODO: initalize population
__global__ void genetic_kernel() {
	int idx = threadIdx.x;
	printf("idx is %d\n", idx);
	__shared__ uint8_t* population;
	__shared__ double* fitness_arr;
	thrust::default_random_engine rand_gen;
	//__shared__ curandState_t* rand_state;
	if(idx == 0) {
		population = new uint8_t[pop_size * member_size];	
		fitness_arr = new double[pop_size];
	}
	__syncthreads();
	init_population(idx, population, rand_gen);

	// selection
	__syncthreads();
	selection(idx, population, fitness_arr);
	__syncthreads();
}

void run_genetic_cuda(size_t p_size, size_t m_size, double cr_rate,
		double m_rate, unsigned max_iter, double (*fitness_func)(uint8_t*)) {
		//double m_rate, unsigned max_iter) {
	cudaMemcpyToSymbol(pop_size, &p_size, sizeof(size_t));
	cudaMemcpyToSymbol(member_size, &m_size, sizeof(size_t));
	cudaMemcpyToSymbol(crossover_rate, &cr_rate, sizeof(double));
	cudaMemcpyToSymbol(mutation_rate, &m_rate, sizeof(double));
	cudaMemcpyToSymbol(max_iterations, &max_iter, sizeof(unsigned));
	cudaMemcpyToSymbol(fitness, &fitness_func, sizeof(fitness_func));
	cudaDeviceSynchronize();
	check_cuda_error("memcpying arguments");

	// TODO: change numbers in angle brackets later
	genetic_kernel<<<1, 100>>>();
	cudaDeviceSynchronize();
	check_cuda_error("after running kernel");
	std::cout << "done with cuda genetic\n";
}
