#include "stdafx.h"
#include "SFMT/dSFMT/dSFMT.c"

int main(int argc, char* argv[])
{
	// Randomize the seed for generating random numbers
	dsfmt_gv_init_gen_rand(static_cast<uint32_t>(time(nullptr)));

	const TArgument Arg(argc, argv); 
	
	TResult tRes;

	Multiplex multi_graph = Multiplex(Arg._dir, tRes, false);

	multi_graph.to_normal_accum_prob();

	multi_graph.reserve_FRsets();

	if (Arg._mode == "M")
	{
		auto delta = Arg._delta;
		if (delta < 0) delta = 0.01;
		auto epsilon = Arg._epsilon;
		multi_graph.mgrrFlexible(Arg._seedsize, Arg._mode, delta, Arg._dir, epsilon);
	}
	else 
	{
		std::cerr << "Error: Mode must be 'M'.\n";
        return 2;  
	}
	
	TIO::write_result(Arg._outFileName, tRes, Arg._resultFolder);
	TIO::write_order_seeds(Arg._outFileName, tRes, Arg._resultFolder);	
	return 0;
}