#ifndef GENETIC_SEQ_H
#define GENETIC_SEQ_H

#include <cstddef>
#include <cstdint>

typedef double (*fitness_t)(uint8_t*);

double run_genetic_seq(unsigned, size_t, size_t, double, double, unsigned, fitness_t);

#endif
