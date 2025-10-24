/**
* @file MGRR.cpp
* @brief multi-graph RR set
* @author LRZ
*
*/

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

	const TArgument Arg(argc, argv); //使用 TArgument 类处理命令行参数，argc 是参数的数量，argv 是参数数组。
	
	TResult tRes;//初始化结果对象 tRes

	//先读图，再格式化图,并初始化结果对象成员
	Multiplex multi_graph = Multiplex(Arg._dir, tRes, false);

    //LT模型概率变成积累化
	multi_graph.to_normal_accum_prob();

	//为_FRsets预留空间，提升性能
	multi_graph.reserve_FRsets();

    //修改，只生成验证集的RR set，这样时间会更快
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