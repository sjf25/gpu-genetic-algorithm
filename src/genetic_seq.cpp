#include "genetic_seq.h"
#include <random>
#include <limits>
#include <iostream>
#include <cstring>
#include <cassert>

static size_t pop_size, member_size;
static uint8_t* population;
static std::default_random_engine rand_gen;
static double crossover_rate;
static double mutation_rate;
static double (*fitness)(uint8_t*);

//double best_fitness = std::numeric_limits<double>::max();
double best_fitness = std::numeric_limits<double>::min();

void init() {
	population = new uint8_t[pop_size * member_size];

	std::random_device r;
	rand_gen = std::default_random_engine{r()};
	/*std::uniform_int_distribution<uint8_t> dist(0,
			std::numeric_limits<uint8_t>::max());*/
	std::uniform_int_distribution<uint8_t> dist(0, 1);
	population = new uint8_t[pop_size * member_size];
	for(size_t i = 0; i < pop_size * member_size; i++) {
		population[i] = dist(rand_gen);
	}
	// TODO: decide whether or not to use following line
	//memset(population, 1, member_size);
}

static uint8_t* get_member(uint8_t* member_array, int i) {
	return &member_array[i * member_size];
}

void destroy() {
	delete[] population;
}

double* pop_fitnesses() {
	double* fitnesses = new double[pop_size];
	for(unsigned i = 0; i < pop_size; i++) {
		fitnesses[i] = fitness(get_member(population, i));
		//std::cout << fitnesses[i] << std::endl;
		//if(fitnesses[i] < best_fitness)
		if(fitnesses[i] > best_fitness)
			best_fitness = fitnesses[i];
	}
	return fitnesses;
}

// returns index of member of population selected via roulette wheel selection
// TODO: problem is this only works when maxing fitness not minimizing
#if 1
static unsigned roulette(double* fitness_arr, double fitness_sum) {
	// TODO: consider making distribution static
	std::uniform_real_distribution<double> dist(0.0, fitness_sum);
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
#endif

#if 0
unsigned roulette(double* probs) {
	std::uniform_real_distribution<double> dist(0.0, 1.0);
	double random_num = dist(rand_gen);
	for(unsigned i = 0; i < pop_size; i++) {
		// TODO: make less redundant/stupid
		/*if(i == pop_size - 1)
			return i;
		if(random_num > probs[i] && random_num < probs[i+1])
			return i;*/
		if(probs[i] >= random_num)
			return i;
	}
	assert("can't get here");
	return 0;
}
#endif

// returns indices of selected individuals
uint8_t* selection() {
	uint8_t* new_population = new uint8_t[pop_size * member_size];
	double* fitness_arr = pop_fitnesses();
	double fitness_sum = 0.0;
	double max_fit = 0.0;
	for(unsigned i = 0; i < pop_size; i++) {
		fitness_sum += fitness_arr[i];
		if(fitness_arr[i] > max_fit)
			max_fit = fitness_arr[i];
	}

	//double new_fitness_sum = pop_size * max_fit - fitness_sum;

	//std::cerr << "new_fitness_sum = " << new_fitness_sum << std::endl;
	//std::cerr << "fitness_arr[0] = " << fitness_arr[0] << std::endl;

	double* probabilities = new double[pop_size]();
	double partial_prob_sum = 0.0;
	for(unsigned i = 0; i < pop_size; i++) {
		//fitness_arr[i] = max_fit - fitness_arr[i];
		//probabilities[i] = partial_prob_sum + fitness_arr[i] / new_fitness_sum;
		probabilities[i] = partial_prob_sum + fitness_arr[i] / fitness_sum;
		partial_prob_sum += probabilities[i];
	}
	//std::cerr << "fitness_arr[0] = " << fitness_arr[0] << std::endl;
	//std::cerr << "fitness_arr[1] = " << fitness_arr[1] << std::endl;
	//std::cerr << "prob[0] = " << probabilities[0] << std::endl;
	//std::cerr << "prob[1] = " << probabilities[1] << std::endl;

	//unsigned* selected = new unsigned[pop_size];
	for(unsigned i = 0; i < pop_size; i++) {
		unsigned selected_idx = roulette(fitness_arr, fitness_sum);
		//unsigned selected_idx = roulette(fitness_arr, new_fitness_sum);
		//unsigned selected_idx = roulette(probabilities);
		memcpy(get_member(new_population, i),
				get_member(population, selected_idx), member_size);
	}

	delete[] fitness_arr;
	delete[] probabilities;
	return new_population;
}

void static two_point_crossover(uint8_t* parent1, uint8_t* parent2) {
	// crossover only with probability of crossover rate
	std::uniform_real_distribution<double> crossover_dist(0, 1);
	double crossover_random = crossover_dist(rand_gen);
	if(crossover_random > crossover_rate)
		return;

	// TODO: verify if points valid
	std::uniform_int_distribution<unsigned> start_dist(0, member_size - 1);
	unsigned start = start_dist(rand_gen);
	std::uniform_int_distribution<unsigned> size_dist(0,
			member_size - 1 - start);
	unsigned swap_size = size_dist(rand_gen);
	uint8_t* temp_buffer = new uint8_t[swap_size];

	// TODO: remove the print statements
	/*std::cout << "\tcrossing over with start = " << start << ", size = "
		<< swap_size << std::endl;*/

	memcpy(temp_buffer, parent1 + start, swap_size);
	memcpy(parent1 + start, parent2 + start, swap_size);
	memcpy(parent2 + start, temp_buffer, swap_size);
	delete[] temp_buffer;
}

// uses two-point crossover
void static crossover(uint8_t* selected) {
	for(unsigned i = 0; i < pop_size; i+=2) {
		// TODO: remove print statement
		//std::cout << "in crossover, i = " << i << std::endl;
		two_point_crossover(get_member(selected, i),
				get_member(selected, i+1));
	}
}

void static mutation(uint8_t* crossed_over) {
	std::uniform_real_distribution<double> mutation_dist(0.0, 1.0);
	for(unsigned i = 0; i < pop_size * member_size; i++) {
		double mutation_prob = mutation_dist(rand_gen);
		if(mutation_prob > mutation_rate)
			continue;
		assert(crossed_over[i] == 0 || crossed_over[i] == 1);
		// flip the 'bit' when mutating
		crossed_over[i] ^= 1;
	}
}

static void print_member(std::ostream& out, uint8_t* member) {
	for(size_t j = 0; j < member_size; j++) {
		out << (unsigned int)member[j] << " ";
	}
	out << std::endl;
}

static void print_pop(std::ostream& out, uint8_t* member_arr=population) {
	for(size_t i = 0; i < pop_size; i++) {
		print_member(out, &member_arr[i*member_size]);
	}
}

void run_genetic_seq(size_t p_size, size_t m_size, double cr_rate,
		double m_rate, unsigned max_iter, double (*fitness_func)(uint8_t*)) {
	// population size must be even
	assert(p_size % 2 == 0);

	pop_size = p_size;
	member_size = m_size;
	crossover_rate = cr_rate;
	mutation_rate = m_rate;
	fitness = fitness_func;

	init();
	//print_pop(std::cout);
	for(unsigned i = 0; i < max_iter; i++) {
		uint8_t* selected = selection();
		//std::cout << "-------------------------------------\n";
		//print_pop(std::cout, selected);

		crossover(selected);
		//print_pop(std::cout, selected);

		mutation(selected);
		//std::cout << "-------------------------------------\n";
		//print_pop(std::cout, selected);
		delete[] population;
		population = selected;
		//std::cout << "best fitness: " << 1/best_fitness << std::endl;
		//best_fitness = std::numeric_limits<double>::min();
	}
	std::cout << "best fitness: " << 1/best_fitness << std::endl;
	destroy();
}
