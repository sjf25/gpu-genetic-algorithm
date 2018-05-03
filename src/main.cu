#include "genetic_seq.h"
#include "genetic_cuda.h"
#include <cassert>
#include <iostream>
#include <fstream>
#include <cstring>
#include <stack>
#include "args_parser.h"
#include "vertex_cover.h"
#include "maxone.h"
#include <vector>

std::string input_file;
unsigned pop_size;
unsigned function_evals;
RunMode run_mode;
Problem problem;
unsigned num_runs;
unsigned maxone_len;

__device__ fitness_t cuda_vercov_cpy = d_vertex_cover_fitness;
__device__ fitness_t cuda_maxone_cpy = d_maxone_fitness;

int main(int argc, char**argv) {
	handle_args(argc, argv);

	std::cout << "problem is " << problem << std::endl;
	std::cout << "running mode is " << run_mode << std::endl;

	unsigned member_size;
	double mut_rate;
	double cross_rate;

	
	fitness_t cuda_fitness;
	fitness_t seq_fitness;

	if(problem == VERCOV) {
		unsigned num_vertices = load_graph(input_file);
		std::cout << "|V| = " << num_vertices << std::endl;

		seq_fitness = vertex_cover_fitness;
		cudaMemcpyFromSymbol(&cuda_fitness, cuda_vercov_cpy, sizeof(fitness_t));

		member_size = num_vertices;
		mut_rate = 1/(double)num_vertices;
		cross_rate = 0.6;
	}
	else if(problem == MAXONE) {
		set_length(maxone_len);

		seq_fitness = maxone_fitness;
		cudaMemcpyFromSymbol(&cuda_fitness, cuda_maxone_cpy, sizeof(fitness_t));

		member_size = maxone_len;
		mut_rate = 1/(double)member_size;
		cross_rate = 0.7;
	}

	cudaError err1 = cudaGetLastError();
	if(err1 != cudaSuccess) {
		std::cerr << "couldn't copy func ptr\n";
		std::cerr << "\t" << cudaGetErrorString(err1) << std::endl;
		return 1;
	}


	double best_fit;

	float time;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	if(run_mode == CUDA) {
		best_fit = run_genetic_cuda(num_runs, pop_size, member_size, cross_rate, mut_rate,
		function_evals/pop_size, cuda_fitness);
	}

	else if(run_mode == SEQUENTIAL) {
		best_fit = run_genetic_seq(num_runs, pop_size, member_size, cross_rate, mut_rate,
		function_evals/pop_size, seq_fitness);
	}

	
	cudaDeviceSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	
	std::cout << "final fitness is " << best_fit << std::endl;
	std::cout << "elapsed time: " << time << std::endl;

	destroy_graph();
}
