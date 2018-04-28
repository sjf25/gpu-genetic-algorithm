#include "genetic_seq.h"
#include "genetic_cuda.h"
#include <cassert>
#include <iostream>
#include <fstream>
#include <cstring>
#include <stack>
#include "args_parser.h"
#include "vertex_cover.h"
#include <vector>

std::string input_file;
unsigned pop_size;
unsigned function_evals;
RunMode run_mode;

__device__ fitness_t dummy = d_vertex_cover_fitness;

int main(int argc, char**argv) {
	handle_args(argc, argv);
	unsigned num_vertices = load_graph(input_file);
	std::cout << "|V| = " << num_vertices << std::endl;
	//std::cout << "greedy solution is " << greedy_vertex_cover() << std::endl;
	/*run_genetic_seq(50, num_vertices, 0.6, 1/(double)num_vertices, 20000/50,
			&vertex_cover_fitness);*/

	fitness_t cuda_fitness;
	cudaMemcpyFromSymbol(&cuda_fitness, dummy, sizeof(fitness_t));
	cudaError err1 = cudaGetLastError();
	if(err1 != cudaSuccess) {
		std::cerr << "couldn't copy func ptr\n";
		std::cerr << "\t" << cudaGetErrorString(err1) << std::endl;
		return 1;
	}
	/*run_genetic_cuda(50, num_vertices, 0.6, 1/(double)num_vertices, 20000/50,
			cuda_fitness);*/

	std::cout << "running mode is " << run_mode << std::endl;

	float time;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	if(run_mode == CUDA) {
		run_genetic_cuda(pop_size, num_vertices, 0.6, 1/(double)num_vertices, function_evals/pop_size,
			cuda_fitness);
	}

	else if(run_mode == SEQUENTIAL) {
		run_genetic_seq(pop_size, num_vertices, 0.6, 1/(double)num_vertices, function_evals/pop_size,
			&vertex_cover_fitness);
	}
	
	cudaDeviceSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	
	std::cout << "elapsed time: " << time << std::endl;

	destroy_graph();
}
