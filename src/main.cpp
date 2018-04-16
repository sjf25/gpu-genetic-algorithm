#include "genetic_seq.h"
#include <cassert>
#include <iostream>
#include <fstream>
#include <cstring>
#include <stack>
#include "args_parser.h"

#include <vector>

std::string input_file;

unsigned num_vertices = 10;
// TODO: consider using bitarray-type structure
// TODO: consider using triangular structure
uint8_t* adjacency;

double fitness(uint8_t*) {
	return 42;
}

// Note: graph is undirected in vertex cover problem
void load_vertex_cover(std::string filename) {
	std::ifstream graph_file(filename);
	std::string p, edge;
	unsigned num_edges;
	graph_file >> p >> edge >> num_vertices >> num_edges;
	adjacency = new uint8_t[num_vertices * num_vertices]();
	for(unsigned i = 0; i < num_edges; i++) {
		std::string dummy;
		unsigned u, v;
		graph_file >> dummy >> u >> v;
		adjacency[u * num_vertices + v] = 1;
		adjacency[v * num_vertices + u] = 1;
	}
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


#if 0
unsigned greedy_vertex_cover() {
	unsigned cover_size = 0;
	//std::unordered_set<std::pair<unsigned, unsigned>> e_prime;
	std::stack<unsigned> e_prime;
	uint8_t* adj_cpy = new uint8_t[num_vertices * num_vertices];
	memcpy(adj_cpy, adjacency, num_vertices * num_vertices);
	for(unsigned i = 0; i < num_vertices; i++) {
		for(unsigned j = i; j < num_vertices; j++) {
			if(adjacency[i * num_vertices + j] == 1)
				//e_prime.push_back(std::pair(i, j));
				e_prime.push(i*num_vertices + j);
		}
	}
	while(e_prime.size() > 0) {
		unsigned edge = e_prime.top();
		e_prime.pop();
		unsigned u = edge / num_vertices;
		unsigned v = edge % num_vertices;
		assert(u < num_vertices && v < num_vertices);
		assert(u <= v);
		if(adj_cpy[u*num_vertices + v] == 0)
			continue;
		cover_size+=2;
		for(unsigned i = 0; i < num_vertices; i++) {
			adj_cpy[u * num_vertices + i] = 0;
			adj_cpy[i * num_vertices + v] = 0;
			adj_cpy[v * num_vertices + i] = 0;
			adj_cpy[i * num_vertices + u] = 0;
		}
	}
	return cover_size;
}
#endif

// used to test if genetic algorithm is worth-while
// algorithm obainted from khuri-back paper and geeksforgeeks
// TODO: randomly select edge
#if 0
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

int main(int argc, char**argv) {
	handle_args(argc, argv);
	load_vertex_cover(input_file);
	std::cout << "greedy solution is " << greedy_vertex_cover() << std::endl;
	/*run_genetic(50, num_vertices, 0.6, 1/(double)num_vertices, 20'000,
			&vertex_cover_fitness);*/
	delete[] adjacency;
}
