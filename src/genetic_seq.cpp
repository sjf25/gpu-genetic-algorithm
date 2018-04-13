#include "genetic_seq.h"
#include <random>
#include <limits>
#include <iostream>
#include <cstring>
#include <cassert>

size_t pop_size, member_size;
uint8_t* population;
std::default_random_engine rand_gen;
unsigned survivor_count;
//uint8_t* survivors;

void init() {
	population = new uint8_t[pop_size * member_size];
	//survivors = new uint8_t[survivor_count * member_size];

	std::random_device r;
	rand_gen = std::default_random_engine{r()};
	std::uniform_int_distribution<uint8_t> dist(0,
			std::numeric_limits<uint8_t>::max());
	population = new uint8_t[pop_size * member_size];
	for(size_t i = 0; i < pop_size * member_size; i++) {
		population[i] = dist(rand_gen);
	}
}

uint8_t* get_member(uint8_t* member_array, int i) {
	return &member_array[i * member_size];
}

void destroy() {
	delete[] population;
}

// note: must minimize fitness function
// place-holder fitness function
double fitness(uint8_t*) {
	return 42;
}

double* pop_fitnesses() {
	double* fitnesses = new double[pop_size];
	for(unsigned i = 0; i < pop_size; i++) {
		fitnesses[i] = fitness(get_member(population, i));
	}
	return fitnesses;
}

// returns index of member of population selected via roulette wheel selection
unsigned roulette(double* fitness_arr, double fitness_sum) {
	// TODO: consider making distribution static
	std::uniform_real_distribution<double> dist(0, fitness_sum);
	double random_num = dist(rand_gen);

	double partial_sum = 0.0;
	for(unsigned i = 0; i < pop_size; i++) {
		partial_sum += fitness_arr[i];
		if(partial_sum >= random_num)
			return i;
	}
	// should be unreachable here
	assert("unreachable section of roulette");
	return 0.0;
}

// returns indices of selected individuals
unsigned* selection() {
	double* fitness_arr = pop_fitnesses();
	double fitness_sum = 0.0;
	for(unsigned i = 0; i < pop_size; i++)
		fitness_sum += fitness_arr[i];

	unsigned* selected = new unsigned[pop_size];
	for(unsigned i = 0; i < pop_size; i++)
		selected[i] = roulette(fitness_arr, fitness_sum);

	delete[] fitness_arr;
	return selected;
}

void two_point_crossover(uint8_t* parent1, uint8_t parent2) {
	// crossover only with probability of crossover rate
	std::uinform_real_distribution crossover_dist(0, 1);
	double crossover_random = rand_gen(crossover_dist);
	if(crossover_random > crossover_rate)
		return;

	// TODO: verify if points valid
	std::uniform_int_distribution<unsigned> point1_dist(0,
			sizeof(uint8_t) * member_size - 1);
	unsigned start_in_bits = point1_dist(rand_gen);
	unsigned start_in_bytes = start_in_bits % 8;

	std::uniform_int_distribution<unsigned> crossover_size_dist(0,
			sizeof(uint8_t) * member_size - 1 - start_in_bits);
	unsigned size_in_bits = crossover_size_dist(rand_gen);
	unsigned size_in_bytes = size_in_bits % 8;
	/*std::uniform_int_distribution<unsigned> point2_dist(point1,
			member_size-1);*/
	//unsigned point2 = point2_dist(rand_gen);
}

// uses two-point crossover
void crossover(unsigned* selected) {

	delete[] selected;
}

void print_member(std::ostream& out, uint8_t* member) {
	for(size_t j = 0; j < member_size; j++) {
		out << (unsigned int)member[j] << " ";
	}
	out << std::endl;
}

void print_pop(std::ostream& out, uint8_t* member_arr=population) {
	for(size_t i = 0; i < pop_size; i++) {
		print_member(out, &member_arr[i*member_size]);
	}
}

void run_genetic(size_t p_size, size_t m_size/*, unsigned sur_count*/) {
	pop_size = p_size;
	member_size = m_size;
	//survivor_count = sur_count;

	init();
	print_pop(std::cout);
	unsigned* selected = selection();
	std::cout << "-------------------------------------\n";
	for(unsigned i = 0; i < pop_size; i++) {
		print_member(std::cout, &population[member_size * selected[i]]);
	}
	destroy();
}
