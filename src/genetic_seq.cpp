#include "genetic_seq.h"
#include <random>
#include <limits>
#include <iostream>

static size_t pop_size, member_size;
static uint8_t* population;
static std::default_random_engine rand_gen;

static void init() {
	std::random_device r;
	rand_gen = std::default_random_engine{r()};
	std::uniform_int_distribution<uint8_t> dist(0,
			std::numeric_limits<uint8_t>::max());
	population = new uint8_t[pop_size * member_size];
	for(size_t i = 0; i < pop_size * member_size; i++) {
		population[i] = dist(rand_gen);
	}
}

static void destroy() {
	delete[] population;
}

void run_genetic(size_t p_size, size_t m_size) {
	pop_size = p_size;
	member_size = m_size;
	init();
	for(size_t i = 0; i < pop_size; i++) {
		for(size_t j = 0; j < member_size; j++) {
			std::cout << (unsigned int)population[i*member_size + j] << " ";
		}
		std::cout << std::endl;
	}
	destroy();
}
