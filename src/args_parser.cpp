#include "args_parser.h"

void validate(boost::any& v, std::vector<std::string> const& choices, RunMode*, int) {
	using namespace boost::program_options;
	validators::check_first_occurrence(v);
	std::string const& str = validators::get_single_string(choices);
	if(str ==  "seq") {
		v = boost::any(SEQUENTIAL);
	}
	else if(str == "cuda") {
		v = boost::any(CUDA);
	}
	else {
		throw validation_error(validation_error::invalid_option_value);
	}
}


void validate(boost::any& v, std::vector<std::string> const& choices, Problem*, int) {
	using namespace boost::program_options;
	validators::check_first_occurrence(v);
	std::string const& str = validators::get_single_string(choices);
	if(str ==  "vercov") {
		v = boost::any(VERCOV);
	}
	else if(str == "maxone") {
		v = boost::any(MAXONE);
	}
	else {
		throw validation_error(validation_error::invalid_option_value);
	}
}

void handle_args(int argc, char** argv) {
	namespace po = boost::program_options;
	po::options_description desc("Options");
	desc.add_options()
		("input", po::value<std::string>(&input_file), "input file")
		("pop-size", po::value<unsigned>(&pop_size)->required(), "size of population")
		("func-evals", po::value<unsigned>(&function_evals)->required(), "number of function evals")
		("run-mode", po::value<RunMode>(&run_mode)->required(), "mode to run in (seq or cuda)")
		("problem", po::value<Problem>(&problem)->required(), "problem to run (vercov or maxone)")
		("num-runs", po::value<unsigned>(&num_runs)->required(), "number of runs of the genetic algorithm")
		("maxone-len", po::value<unsigned>(&maxone_len), "length of maxone");
	po::variables_map vm;
	try {
		po::store(po::parse_command_line(argc, argv, desc), vm);
		po::notify(vm);
	}
	catch(po::error& e) {
		std::cerr << "argv error: " << e.what() << std::endl;
		exit(1);
	}
}

std::ostream& operator<<(std::ostream& out, const RunMode mode) {
	switch(mode) {
	case SEQUENTIAL:
		return out << "sequential";
	case CUDA:
		return out << "cuda";
	default:
		return out;
	}
}

std::ostream& operator<<(std::ostream& out, const Problem prob) {
	switch(prob) {
	case VERCOV:
		return out << "vertex cover";
	case MAXONE:
		return out << "maxone";
	default:
		return out;
	}
}
