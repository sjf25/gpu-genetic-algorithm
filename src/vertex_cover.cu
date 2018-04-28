#include "vertex_cover.h"
#include <iostream>
#include <fstream>
#include <cassert>

static unsigned num_vertices;
// TODO: consider using bitarray-type structure
// TODO: consider using triangular structure
static uint8_t* adjacency;

__constant__ static unsigned d_num_vertices;
__constant__ static uint8_t* d_adjacency;

// Note: graph is undirected in vertex cover problem
unsigned load_graph(std::string filename) {
	std::ifstream graph_file(filename);
	std::string p, edge;
	unsigned num_edges;
	graph_file >> p >> edge >> num_vertices >> num_edges;
	cudaMemcpyToSymbol(d_num_vertices, &num_vertices, sizeof(unsigned));
	cudaDeviceSynchronize();
	cudaError err2 = cudaGetLastError();
	if(err2 != cudaSuccess) {
		std::cerr << "error when memcpying to d_num_vertices\n";
		exit(1);
	}



	adjacency = new uint8_t[num_vertices * num_vertices]();
	uint8_t* adj_temp;
	cudaError err1 = cudaMalloc(&adj_temp, num_vertices * num_vertices);
	if(err1 != cudaSuccess) {
		std::cerr << "error when cudaMalloc-ing temporary adjacency arr\n";
		exit(1);
	}

	for(unsigned i = 0; i < num_edges; i++) {
		std::string dummy;
		unsigned u, v;
		graph_file >> dummy >> u >> v;
		adjacency[u * num_vertices + v] = 1;
		adjacency[v * num_vertices + u] = 1;
	}

	cudaMemcpy(adj_temp, adjacency, num_vertices * num_vertices, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_adjacency, &adj_temp, sizeof(uint8_t*));
	cudaError err3 = cudaGetLastError();
	cudaDeviceSynchronize();
	if(err3 != cudaSuccess) {
		std::cerr << "error copying adjacency matrix to cuda\n";
		exit(1);
	}

	return num_vertices;
}

void destroy_graph() {
	delete[] adjacency;
}

double vertex_cover_fitness(uint8_t* member) {
	double total = 0.0;
	for(unsigned i = 0; i < num_vertices; i++) {
		assert(member[i] == 0 || member[i] == 1);
		if(member[i] == 1)
			total += 1.0;
		// else member[i] == 0
		else {
			double penalty = 0.0;
			for(unsigned j = 0; j < num_vertices; j++) {
				/*if(member[j] == 0 && adjacency[i*num_vertices + j] == 1)
					penalty += 1.0;*/
				penalty += adjacency[i*num_vertices + j] * (1-member[j]);
			}
			total += num_vertices * penalty;
		}
	}
	return 1/total;
}

__device__ double d_vertex_cover_fitness(uint8_t* member) {
	//printf("from device, |V| = %d\n", d_num_vertices);
	double total = 0.0;
	for(unsigned i = 0; i < d_num_vertices; i++) {
		#if 1
		if(member[i] != 0 && member[i] != 1) {
			printf("member: %d\n", member[i]);
		}
		#endif
		assert(member[i] == 0 || member[i] == 1);
		if(member[i] == 1)
			total += 1.0;
		// else member[i] == 0
		else {
			double penalty = 0.0;
			for(unsigned j = 0; j < d_num_vertices; j++) {
				/*if(member[j] == 0 && adjacency[i*num_vertices + j] == 1)
					penalty += 1.0;*/
				penalty += d_adjacency[i*d_num_vertices + j] * (1-member[j]);
			}
			total += d_num_vertices * penalty;
		}
	}
	return 1/total;
}

//extern __device__ double (*dummy)(uint8_t*) = d_vertex_cover_fitness;

// used to test if genetic algorithm is worth-while
// algorithm obainted from khuri-back paper and geeksforgeeks
// TODO: randomly select edge
#if 1
unsigned greedy_vertex_cover() {
	bool* visited = new bool[num_vertices]();
	
	for(unsigned u = 0; u < num_vertices; u++) {
		if(visited[u])
			continue;
		for(unsigned v = 0; v < num_vertices; v++) {
			if(adjacency[u * num_vertices + v] == 0)
				continue;
			if(visited[v])
				continue;
			visited[v] = true;
			visited[u] = true;
		}
	}

	unsigned cover_size = 0;
	for(unsigned i = 0; i < num_vertices; i++) {
		if(visited[i])
			cover_size++;
	}

	delete[] visited;
	//return cover_size;
	return cover_size;
}
#endif

// TODO: consider if this works with self pointing edges
#if 0
unsigned greedy_vertex_cover() {
	bool* visited = new bool[num_vertices]();
	std::vector<unsigned> to_visit;
	for(unsigned i = 0; i < num_vertices; i++)
		to_visit.push_back(i);

	std::random_device rd;
	std::mt19937 g(rd());
	std::shuffle(to_visit.begin(), to_visit.end(), g);
	
	for(unsigned i = 0; i < num_vertices; i++) {
		unsigned u = to_visit[i];
		std::vector<unsigned> v_list;
		for(unsigned i = 0; i < num_vertices; i++)
			v_list.push_back(i);

		std::shuffle(v_list.begin(), v_list.end(), g);	

		if(visited[u])
			continue;
		for(unsigned j = 0; j < num_vertices; j++) {
			unsigned v = v_list[j];
			if(adjacency[u * num_vertices + v] == 0)
				continue;
			if(visited[v])
				continue;
			visited[v] = true;
			visited[u] = true;
		}
	}

	unsigned cover_size = 0;
	for(unsigned i = 0; i < num_vertices; i++) {
		if(visited[i])
			cover_size++;
	}

	delete[] visited;
	//return cover_size;
	return cover_size;
}
#endif
