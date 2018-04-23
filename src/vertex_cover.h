#ifndef VERTEX_COVER_H
#define VERTEX_COVER_H

#include <string>

unsigned load_graph(std::string);
double vertex_cover_fitness(uint8_t*);
__device__ double d_vertex_cover_fitness(uint8_t*);
unsigned greedy_vertex_cover();
void destroy_graph();

#endif
