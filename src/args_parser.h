#ifndef ARGS_PARSER
#define ARGS_PARSER

#include <boost/program_options.hpp>
#include <string>
#include <iostream>

extern std::string input_file;

void handle_args(int, char**);

#endif
