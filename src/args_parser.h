#ifndef ARGS_PARSER
#define ARGS_PARSER

#include <boost/program_options.hpp>
#include <string>
#include <iostream>
#include <vector>

enum RunMode { SEQUENTIAL, CUDA/*, GREEDY*/ };
enum Problem { VERCOV, MAXONE };

extern std::string input_file;
extern unsigned pop_size;
extern unsigned function_evals;
extern RunMode run_mode;
extern Problem problem;
extern unsigned num_runs;
extern unsigned maxone_len;

void handle_args(int, char**);

std::ostream& operator<<(std::ostream&, const RunMode);
std::ostream& operator<<(std::ostream&, const Problem);

#endif
