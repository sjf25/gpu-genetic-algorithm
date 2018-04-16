#ifndef GENETIC_SEQ_H
#define GENETIC_SEQ_H

#include <cstddef>
#include <cstdint>

typedef double (*fitness_t)(uint8_t*);

void run_genetic(size_t, size_t, double, double, unsigned, fitness_t);

#endif
