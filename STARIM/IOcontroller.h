#pragma once

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#endif

class IOcontroller
{
public:
	static void mkdir_absence(const char* outFolder)
	{
#if defined(_WIN32)
		CreateDirectoryA(outFolder, nullptr); // can be used on Windows
#else
		mkdir(outFolder, 0733); // can be used on non-Windows
#endif
	}

	/*
	/// Save a serialized file
	template <class T>
	static void save_file(const std::string filename, const T& output)
	{
		std::ofstream outfile(filename, std::ios::binary);
		if (!outfile.eof() && !outfile.fail())
		{
			StreamType res;
			serialize(output, res);
			outfile.write(reinterpret_cast<char*>(&res[0]), res.size());
			outfile.close();
			res.clear();
			std::cout << "Save file successfully: " << filename << '\n';
		}
		else
		{
			std::cout << "Save file failed: " + filename << '\n';
			exit(1);
		}
	}

	/// Load a serialized file
	//模板参数 T：此函数是一个模板函数，T 可以是任何数据类型。
	//filename：输入参数，表示要加载的文件名。
	//input：引用类型参数，用于存储反序列化后的数据。
	template <class T>
	static void load_file(const std::string filename, T& input)
	{
		//使用 std::ifstream 打开指定的文件，模式为二进制 (std::ios::binary)。
		//这允许读取文件中的原始二进制数据，适用于存储非文本格式的数据。
		std::ifstream infile(filename, std::ios::binary);
		
		if (!infile.eof() && !infile.fail())
		{//检查文件是否成功打开，并且未到达文件结束标志
			//使用 seekg 函数移动文件指针到文件末尾，以获取文件的大小
			infile.seekg(0, std::ios_base::end);
			//tellg 返回当前文件指针的位置
			const std::streampos fileSize = infile.tellg();
			//将指针重置回文件开头
			infile.seekg(0, std::ios_base::beg);

			//创建一个 std::vector<uint8_t>，大小为 fileSize，用于存储读取的文件数据
			std::vector<uint8_t> res(fileSize);
			//使用 read 函数将文件内容读取到 res 中，
			//reinterpret_cast 将 uint8_t 指针转换为 char 指针以符合 read 函数的参数要求
			infile.read(reinterpret_cast<char*>(&res[0]), fileSize);
			infile.close();//关闭文件流，释放资源

			input.clear();//清空 input，以确保其状态为初始状态
			auto it = res.cbegin();//创建一个迭代器 it，指向 res 的起始位置
			//调用 deserialize 函数将数据反序列化到 input 中。deserialize 函数假定已经定义，
			//并负责将二进制数据转换为类型 T 的对象。
			input = deserialize<T>(it, res.cend());
			res.clear();清空 res 向量，释放内存
		}
		else
		{
			std::cout << "Cannot open file: " + filename << '\n';
			exit(1);
		}
	}

	/// Save graph structure to a file
	static void save_graph_struct(const std::string graphName, const Graph& vecGraph, const bool isReverse)
	{
		std::string postfix = ".vec.graph";
		if (isReverse) postfix = ".vec.rvs.graph";
		const std::string filename = graphName + postfix;
		save_file(filename, vecGraph);
	}

	/// Load graph structure from a file
	//graphName：图文件的基本名称，不包含文件后缀。
	//vecGraph：引用类型参数，用于存储加载后的图结构。
	//isReverse：布尔值，指示是否加载反向图结构。
	static void load_graph_struct(const std::string graphName, Graph& vecGraph, const bool isReverse)
	{
		//设置文件后缀
		std::string postfix = ".vec.graph";
		if (isReverse) postfix = ".vec.rvs.graph";
		const std::string filename = graphName + postfix;//构造完整的文件名

		//vecGraph 是一个引用类型参数，这意味着在函数内部对 vecGraph 所做的任何修改
		//都会直接影响到调用者所提供的图数据结构。
		load_file(filename, vecGraph);//调用函数 load_file，将图数据从指定的文件中加载到 vecGraph 中。
	}
	*/

	/// Get out-file name
	static std::string get_out_file_name(const std::string graphName, const std::string algName, const int seedsize,
		const std::string probDist, const float probEdge)
	{
		if (probDist == "UNI")
		{
			return graphName + "_" + algName + "_k" + std::to_string(seedsize) + "_" + probDist + std::
				to_string(probEdge);
		}
		return graphName + "_" + algName + "_k" + std::to_string(seedsize) + "_" + probDist;
	}

	
	/// Print the results
	static void write_result(const std::string& outFileName, const TResult& resultObj, const std::string& outFolder)
	{
		//const auto approx = resultObj.get_approximation();
		const auto runTime = resultObj.get_running_time();
		const auto influence = resultObj.get_influence();
		const auto influenceOriginal = resultObj.get_influence_original();
		const auto influenceMC = resultObj.get_influence_MC();
		const auto seedSize = resultObj.get_seed_size();
		const auto RRsetsSize = resultObj.get_RRsets_size();
		const auto round = resultObj.get_round();

		std::cout << "   --------------------" << std::endl;
		//std::cout << "  |Approx.: " << approx << std::endl;
		std::cout << "  |Time (sec): " << runTime << std::endl;
		std::cout << "  |Influence Validate: " << influence << std::endl;
		std::cout << "  |Influence Original: " << influenceOriginal << std::endl;
		std::cout << "  |Influence MC: " << influenceMC << std::endl;
		std::cout << "  |#Seeds: " << seedSize << std::endl;
		std::cout << "  |#RR sets: " << RRsetsSize << std::endl;
		std::cout << "  |#Round: " << round << std::endl;
		std::cout << "   --------------------" << std::endl;
		
		mkdir_absence(outFolder.c_str());
		std::ofstream outFileNew(outFolder + "/" + outFileName);
		if (outFileNew.is_open())
		{
			//outFileNew << "Approx.: " << approx << std::endl;
			outFileNew << "Time (sec): " << runTime << std::endl;
			outFileNew << "Influence: " << influence << std::endl;
			outFileNew << "Self-estimated influence: " << influenceMC << std::endl;
			outFileNew << "#Seeds: " << seedSize << std::endl;
			outFileNew << "#RR sets: " << RRsetsSize << std::endl;
			outFileNew << "#Round: " << round << std::endl;
			outFileNew.close();
		}
	}

	/// Print the seeds
	static void write_order_seeds(const std::string& outFileName, const TResult& resultObj, const std::string& outFolder)
	{
		auto vecSeed = resultObj.get_seed_vec();
		mkdir_absence(outFolder.c_str());
		const auto outpath = outFolder + "/seed";
		mkdir_absence(outpath.c_str());
		std::ofstream outFile(outpath + "/seed_" + outFileName);
		for (auto i = 0; i < vecSeed.size(); i++)
		{
			std::cout << "(" <<vecSeed[i].first << "," << vecSeed[i].second << ")" << '\n';
		}
		outFile.close();
	}
};

using TIO = IOcontroller;
using PIO = std::shared_ptr<IOcontroller>;
