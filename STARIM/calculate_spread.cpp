#include "stdafx.h"
#include "SFMT/dSFMT/dSFMT.c"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>

int main(int argc, char* argv[])
{
	// Randomize the seed for generating random numbers
	dsfmt_gv_init_gen_rand(static_cast<uint32_t>(time(nullptr)));

	const TArgument Arg(argc, argv); 
	
	TResult tRes;

	Multiplex multi_graph = Multiplex(Arg._dir, tRes, false);

	multi_graph.to_normal_accum_prob();

	multi_graph.reserve_FRsets();

    multi_graph.calculateSpread_build_rrset(Arg._samplesize, "S");

	std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream iss(line);
        std::string token;
		std::vector<Node> seed_nodes;

        while (iss >> token) {
            size_t comma_pos = token.find(',');
            if (comma_pos != std::string::npos) {
                int layer = std::stoi(token.substr(0, comma_pos));
                int node = std::stoi(token.substr(comma_pos + 1));
                seed_nodes.emplace_back(layer, node);
            }
        }

        double spread = multi_graph.calculateSpread_calculate(seed_nodes);
        std::cout << spread << std::endl;
        std::cout.flush();
	}

    
	return 0;
}