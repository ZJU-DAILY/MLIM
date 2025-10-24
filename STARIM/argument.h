#pragma once


class Argument
{
public:
	std::string _mode = "M"; // I->integrate 整体方法  S->stratification 分层方法  M->MGRR 我们提出的方法
	std::string _dir = "dataset/Venetie"; // Directory 
	size_t _seedsize = 10; // The number of nodes to be selected. Default is 50.
	size_t _samplesize = 10000; // The number of RR sets to be generated.
	double _epsilon = 0.2; // Error threshold 1-1/e-epsilon. 
	double _delta = 0.01; // Failure probability delta. Default is 1/#nodes.
	size_t _grapgNumber = 10; //用于指定处理几个训练图 
	std::vector<int> _budget = {10, 30, 50, 70, 90, 100, 150, 300, 500}; // 用于存储预算值

	std::string _resultFolder = "result"; // Result folder. Default is "test".
	std::string _algName = "MGRR"; // Algorithm. Default is oneHop.
	std::string _graphname; // Graph name. Default is "facebook".
	std::string _outFileName; // File name of the result

	std::string _seedFileName = ""; // File name of the result
	std::string _resultFileName = ""; // File name of the result

	/*
	float _probEdge = float(0.1); // For the UNI setting, every edge has the same diffusion probability.
	std::string _probDist = "load"; // Probability distribution for diffusion model. Option: load, WC, TR, UNI. Default is loaded from the file.
	*/
	
	Argument(int argc, char* argv[])
	{
		std::string param, value;
		for (int ind = 1; ind < argc; ind++)
		{
			if (argv[ind][0] != '-') break;
			std::stringstream sstr(argv[ind]);
			getline(sstr, param, '=');
			getline(sstr, value, '=');
			if (!param.compare("-mode")) _mode = value;
			else if (!param.compare("-dir")) _dir = value;
			else if (!param.compare("-seedFile")) _seedFileName = value;
			else if (!param.compare("-resultFile")) _resultFileName = value;
			else if (!param.compare("-seedsize")) _seedsize = stoi(value);
			else if (!param.compare("-samplesize")) _samplesize = stoull(value);
			else if (!param.compare("-epsilon")) _epsilon = stod(value);
			else if (!param.compare("-delta")) _delta = stod(value);
		
			else if (!param.compare("-outpath")) _resultFolder = value;
			else if (!param.compare("-alg")) _algName = value;
			else if (!param.compare("-gname")) _graphname = value;

			else if (!param.compare("-graphnumber")) _grapgNumber = stoi(value);//stoi用于将字符串转换为整数
			else if (!param.compare("-budget")) {
				_budget.clear();  // 清空默认值，避免重复添加
				// 使用string分割获取多个预算值
				std::stringstream valueStream(value);
				std::string token;
				while (getline(valueStream, token, ',')) {
					_budget.push_back(stoi(token));  // 将每个预算值转换为整数并存储
				}
			}

			/*
			else if (!param.compare("-pedge")) _probEdge = stof(value);
			else if (!param.compare("-pdist")) _probDist = value;
			*/
		}
		
		_graphname = _dir;
		_outFileName = get_outfilename_MGRR();
		//之后可能会用到
		//_outFileName = TIO::get_out_file_name(_graphname, _algName + postfix, _seedsize, _probDist, _probEdge);
	}

	std::string get_outfilename_MGRR()
	{
		//目前边的权重都是load，没有其他的
		if (_mode == "I")
		{
			return _graphname + "_" + "Integrate" + "_k" + std::to_string(_seedsize) + "_s" + std::to_string(_samplesize) + "_load";
		}
		else if (_mode == "S")
		{
			return _graphname + "_" + "Stratifie" + "_k" + std::to_string(_seedsize) + "_s" + std::to_string(_samplesize) + "_load";
		}
		else if (_mode == "M")
		{
			return _graphname + "_" + "MGRR" + "_k" + std::to_string(_seedsize) + "_de" + std::to_string(_delta) + "_ep" + std::to_string(_epsilon) +"_load";
		}
		else{
			return "error";
		}
	}
};

using TArgument = Argument;
using PArgument = std::shared_ptr<TArgument>;
