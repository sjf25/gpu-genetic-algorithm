#include "genetic_cuda.h"
#include "utils.h"
#include <curand_kernel.h>
#include <thrust/random.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>

__constant__ static size_t pop_size;
__constant__ static size_t member_size;
__constant__ static double crossover_rate;
__constant__ static double mutation_rate;
__constant__ static unsigned max_iterations;
__constant__ double (*fitness)(uint8_t*);

//__device__ double final_fitness;

inline void check_cuda_error(std::string msg) {
        cudaError err = cudaGetLastError();
        if(cudaSuccess != err) {
                std::cerr << "\033[1;31mcuda error: " << msg << std::endl;
                std::cerr << "\t" << cudaGetErrorString(err) << std::endl;
                exit(1);
        }
}

__device__ static void print_member(uint8_t* member) {
	for(size_t j = 0; j < member_size; j++) {
		printf("%d ", member[j]);
	}
	printf("\n");
}

__device__ static void print_pop(uint8_t* member_arr) {
	for(size_t i = 0; i < pop_size; i++) {
		print_member(&member_arr[i*member_size]);
	}
}

static __device__ uint8_t* get_member(uint8_t* member_array, int i) {
	return &member_array[i * member_size];
}

__device__ void init_population(int idx, uint8_t* population,
	thrust::default_random_engine& rand_gen) {
	thrust::uniform_int_distribution<uint8_t> dist(0, 1);
	for(unsigned i = 0; i < member_size; i++) {
		rand_gen.discard(idx);
		population[member_size * idx + i] = dist(rand_gen);
	}
}

__device__ void record_fitness(int idx, double* __restrict__ fitness_arr,
	uint8_t* __restrict__ population, double* __restrict__ best_fitnesses) {
	double current_fitness = fitness(get_member(population, idx));
	fitness_arr[idx] = current_fitness;
	if(current_fitness > best_fitnesses[idx])
		best_fitnesses[idx] = current_fitness;
}

__device__ static unsigned roulette(double* fitness_arr, double fitness_sum,
	thrust::default_random_engine& rand_gen) {

	rand_gen.discard(threadIdx.x);

	thrust::uniform_real_distribution<double> dist(0.0, fitness_sum);
	//thrust::default_random_engine rand_gen;
	double rand_num = dist(rand_gen);
	
	double partial_sum = 0.0;
	for(unsigned i = 0; i < pop_size; i++) {
		partial_sum += fitness_arr[i];
		if(partial_sum >= rand_num)
			return i;
	}
	return 0.0;
}

__device__ void selection(int idx, uint8_t* __restrict__ population,
	double* __restrict__ fitness_arr, uint8_t* __restrict__ new_population,
	thrust::default_random_engine& rand_gen, double* __restrict__ best_fitnesses,
	bool* __restrict__ ready) {
	__shared__ double fitness_sum;
	record_fitness(idx, fitness_arr, population, best_fitnesses);

	__syncthreads();
	prefix_sum(fitness_arr, &fitness_sum, pop_size);

	__syncthreads();

	unsigned selected_idx = roulette(fitness_arr, fitness_sum, rand_gen);
	memcpy(get_member(new_population, idx), get_member(population, selected_idx),
		member_size);
	ready[idx] = true;
}

__device__ static void two_point_crossover(int idx, uint8_t* __restrict__ parent1,
	uint8_t* __restrict__ parent2, thrust::default_random_engine& rand_gen) {
	// crossover only with probability of crossover rate
	thrust::uniform_real_distribution<double> crossover_dist(0.0, 1.0);
	rand_gen.discard(idx);
	double crossover_random = crossover_dist(rand_gen);
	if(crossover_random > crossover_rate)
		return;

	// TODO: verify if points valid
	thrust::uniform_int_distribution<unsigned> start_dist(0, member_size - 1);
	rand_gen.discard(idx);
	unsigned start = start_dist(rand_gen);
	thrust::uniform_int_distribution<unsigned> size_dist(0,
			member_size - 1 - start);
	rand_gen.discard(idx);
	unsigned swap_size = size_dist(rand_gen);
	uint8_t* temp_buffer = new uint8_t[swap_size];

	memcpy(temp_buffer, parent1 + start, swap_size);
	memcpy(parent1 + start, parent2 + start, swap_size);
	memcpy(parent2 + start, temp_buffer, swap_size);
	delete[] temp_buffer;
}

__device__ static void crossover(int idx, uint8_t* __restrict__ selected,
	thrust::default_random_engine& rand_gen, bool* __restrict__ ready) {
	if(idx % 2 == 0) {
		while(!ready[idx + 1]) {}
		two_point_crossover(idx, get_member(selected, idx),
			get_member(selected, idx+1), rand_gen);
		ready[idx] = true;
		ready[idx+1] = true;
	}
	//ready[idx] = true;
}

// TODO: consider speeding up mutation by not waiting to sync threads after crossover
// TODO: consider speeding up by having thread per bit and mutating
__device__ static void mutation(int idx, uint8_t* __restrict__ crossed_over,
	thrust::default_random_engine& rand_gen, bool* __restrict__ ready) {
	while(!ready[idx]) {}
	thrust::uniform_real_distribution<double> mutation_dist(0.0, 1.0);
	uint8_t* member = get_member(crossed_over, idx);
	for(unsigned i = 0; i < member_size; i++) {
		rand_gen.discard(idx);
		double mutation_prob = mutation_dist(rand_gen);
		if(mutation_prob > mutation_rate)
			continue;
		assert(member[i] == 0 || member[i] == 1);
		// flip the 'bit' when mutating
		member[i] ^= 1;
	}
	ready[idx] = false;
}

__device__ void swap_population(uint8_t** population_1, uint8_t** population_2) {
	uint8_t* temp = *population_1;
	*population_1 = *population_2;
	*population_2 = temp;
}

// I think it's safe to assume that pop_size <= 1024
// so just one block with many threads
__global__ void genetic_kernel(double* final_fitnesses) {
	int idx = threadIdx.x;
	__shared__ uint8_t* population;
	__shared__ uint8_t* new_population;
	__shared__ double* fitness_arr;
	__shared__ double* best_fitnesses;
	__shared__ double best_fit;
	
	__shared__ bool* ready;

	thrust::default_random_engine rand_gen;
	rand_gen.discard(blockIdx.x);
	

	if(idx == 0) {
		population = new uint8_t[pop_size * member_size];
		new_population = new uint8_t[pop_size * member_size];
		fitness_arr = new double[pop_size];
		best_fitnesses = new double[pop_size];
		ready = new bool[pop_size]();
	}
	__syncthreads();
	best_fitnesses[idx] = -DBL_MAX;
	init_population(idx, population, rand_gen);
	
	for(unsigned i = 0; i < max_iterations; i++) {
		__syncthreads();
		selection(idx, population, fitness_arr, new_population, rand_gen, best_fitnesses, ready);

		//__syncthreads();
		crossover(idx, new_population, rand_gen, ready);

		//__syncthreads();
		mutation(idx, new_population, rand_gen, ready);
		
		__syncthreads();
		if(idx == 0)
			swap_population(&population, &new_population);
	}
	__syncthreads();
	parallel_max(best_fitnesses, &best_fit, pop_size);
	__syncthreads();
	#if 0
	if(idx == 0) {
		printf("best fitness: %f\n", 1/best_fit);
	}
	#endif
	//final_fitness = best_fit;
	if(idx == 0) {
		final_fitnesses[blockIdx.x] = best_fit;
	}
}

double run_genetic_cuda(unsigned run_count, size_t p_size, size_t m_size, double cr_rate,
		double m_rate, unsigned max_iter, double (*fitness_func)(uint8_t*)) {

	// population size must be even
	assert(p_size % 2 == 0);

	cudaMemcpyToSymbol(pop_size, &p_size, sizeof(size_t));
	cudaMemcpyToSymbol(member_size, &m_size, sizeof(size_t));
	cudaMemcpyToSymbol(crossover_rate, &cr_rate, sizeof(double));
	cudaMemcpyToSymbol(mutation_rate, &m_rate, sizeof(double));
	cudaMemcpyToSymbol(max_iterations, &max_iter, sizeof(unsigned));
	cudaMemcpyToSymbol(fitness, &fitness_func, sizeof(fitness_func));
	cudaDeviceSynchronize();
	check_cuda_error("memcpying arguments");

	double* final_fitnesses;
	cudaMalloc(&final_fitnesses, run_count * sizeof(double));
	genetic_kernel<<<run_count, p_size>>>(final_fitnesses);
	cudaDeviceSynchronize();
	check_cuda_error("after running kernel");

	double return_val;
	thrust::device_ptr<double> fit_ptr = thrust::device_pointer_cast(final_fitnesses);
	return_val = *thrust::max_element(fit_ptr, fit_ptr + run_count);
	return return_val;
}
