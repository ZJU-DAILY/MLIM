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
