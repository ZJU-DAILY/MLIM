/**
* @file MGRR.cpp
* @brief multi-graph RR set
* @author LRZ
*
*/

#include "stdafx.h"
#include "SFMT/dSFMT/dSFMT.c"

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


	//根据输入参数设置 delta 的值,delta一般等于1/n，n为多层图RR总数
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
        return 2;  // 非正常结束
	}
	
	//将结果和种子顺序写入输出文件,之后将输出文件名整理一下
	TIO::write_result(Arg._outFileName, tRes, Arg._resultFolder);
	TIO::write_order_seeds(Arg._outFileName, tRes, Arg._resultFolder);	
	return 0;
}