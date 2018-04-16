#include "args_parser.h"

void handle_args(int argc, char** argv) {
	namespace po = boost::program_options;
	po::options_description desc("Options");
	desc.add_options()
		("input", po::value<std::string>(&input_file), "input file");
	po::variables_map vm;
	try {
		po::store(po::parse_command_line(argc, argv, desc), vm);
		po::notify(vm);
	}
	catch(po::error& e) {
		std::cerr << "argv error: " << e.what() << std::endl;
	}
}
