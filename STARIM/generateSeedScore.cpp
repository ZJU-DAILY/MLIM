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

	auto delta = Arg._delta;
    if (delta < 0) delta = 0.01;
    auto epsilon = Arg._epsilon;
    multi_graph.seedScore(Arg._seedsize, "M", delta, epsilon, Arg._dir);

	return 0;
}