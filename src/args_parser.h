#ifndef ARGS_PARSER
#define ARGS_PARSER

#include <boost/program_options.hpp>
#include <string>
#include <iostream>
#include <vector>

enum RunMode { SEQUENTIAL, CUDA/*, GREEDY*/ };

extern std::string input_file;
extern unsigned pop_size;
extern unsigned function_evals;
extern RunMode run_mode;

void handle_args(int, char**);

std::ostream& operator<<(std::ostream&, const RunMode);

#endif
