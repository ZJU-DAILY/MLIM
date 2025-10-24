#pragma once

class Multiplex
{ 
private:
    /// Result object.
	TResult& _tRes;


    /// 多层图的层数
    uint32_t _nLayers;  
    /// 多层图的节点总数，有重复节点按照重复计算（用加法）
    uint32_t _nNodesAllLayers;  
    /// 存储多层图结构
    std::vector<Graph> _Layers;
    /// 存储每层对应的
    std::vector<CascadeModel> _LayerModels;

    /// 用于选择种子集的RR set个数
	size_t _numRRsets;  
    /// 储存用于选择种子集的RRsets
    RRsets _RRsets; 
    /// 用于验证种子集影响力的RR set个数
    size_t _numRRsetsVldt;  
    /// 储存用于验证种子集影响力的RRsets
    RRsets _RRsetsVldt;  

    /// 种子集
    Nodelist _vecSeed; 

    /// 存储分解出来的RR(1) RR(2) ... RR(n)的个数
    std::vector<uint32_t> _RRSub;//问一下命名的问题，是否需要在前面加上_
    std::vector<uint32_t> _RRSubVldt;

    /// 分层方法中，每一层上覆盖的RR set个数,在验证的快速方法中会用到
    std::vector<uint32_t> _coveredRRSetVldt;

    //////上面的都初始化过，下面的都没有初始化
    //GCN方法的中间表示
    std::map<Node, int> _nodeRankMap;
    std::vector< std::vector< std::pair<Node, Inf> > > _seedNodeWithGain; //最外层的vector表示m个种子集，里层的vector表示每一个种子集种具体的node和inf
    std::vector<Inf> _totalGain;

public:
    Multiplex( std::string fname, TResult& tRes, bool processTrainingData): _tRes(tRes)
    {
        //初始化变量
        _nLayers = 0;
        _nNodesAllLayers = 0;
        _numRRsets = 0;
        _numRRsetsVldt = 0;

        //预分配空间 
        _RRsets.reserve(5000);//尽量一次分配较多内存
        _RRsetsVldt.reserve(5000);
        _vecSeed.reserve(500);

        //其他私有属性全部使用默认初始化，在这里不写了，后续会读取数据
        init(fname, processTrainingData); 
    }
	
    ~Multiplex()
	{
	}

    void init(std::string input, bool processTrainingData);

    /*
	对于LT传播模型，将边上的传播概率变成积累的形式并进行归一化
    这样做是为了后续生成RR set：e.g., [0.2, 0.5, 0.3]->[0.2, 0.7, 1.0]*/
    void to_normal_accum_prob();

    void reserve_FRsets();

    void calculateSpread_build_rrset(const size_t numRRsets, const std::string mode);

    double calculateSpread_calculate(const std::vector<Node> &seed);

    void build_n_RRsets(const size_t numSamples, const std::string mode, bool processTrainingData = false, bool validateRRset = true);//bool validateRRset表示是否生成验证集的RR set

    void build_one_RRset(const Node uStart, const size_t hyperIdx, const bool validate, const std::string mode);

    double max_cover_stratifie(const int targetSize);//分层方法下的maxCover算法，不论是stratifie还是MGRR方法都用这个

    double self_inf_cal_stratifie();

    /// MGRR: 数据量小时epsilon = 0.2，数据量小时epsilon = 0.3 epsilon∈(0.1,0.5)
    void mgrrFlexible(const int targetSize, const std::string mode, const double delta, std::string fname, const double epsilon = 0.2);

    double monteCarloInfluence(const Nodelist& vecSeed, uint32_t nIter = 10000);

    void sampleMultiplex(std::vector<Graph>& sampled_multigraph);

    uint32_t forwardProp(std::vector<Graph> sampled_multigraph, const Nodelist& vecSeed);

    double seedScore(const int targetSize, const std::string mode, const double delta, const double epsilon, std::string fname);

    void RandomK(const int targetSize);

    void BestDegree(const int targetSize);

    void deepIM_influence(const int targetSize, std::string dir);

    void KSN_influence(const int targetSize, std::string dir);

    double find_min_RRSub(const std::vector<uint32_t> &RRSub)
    {
        double min = 1e100;

        for (size_t i = 0; i < _nLayers; i++)
        {
            if(_Layers[i]._Capacity == 0) continue;
            double tmp = static_cast<double>(RRSub[i]) / _Layers[i]._Node_number;
            if(tmp < min){
                min =tmp;
            }
        }
        return min;
    }

    bool file_exists (const std::string& name) {
        if (FILE *file = fopen(name.c_str(), "r")) {
            fclose(file);
            return true;
        } else {
            return false;
        }   
    }

    //检查读图是否正确的代码，读者可以直接使用，本代码没有使用这个函数，因为本代码读图已经验证过为正确的
    void checkGraph(){
         std::ofstream outFile("check.txt"); // 打开文件

        // 检查文件是否成功打开
        if (!outFile) {
            std::cerr << "Error opening file: " << "check.txt" << std::endl;
            return;
        }

        outFile << "nLayers "<< _nLayers <<std::endl;
        for (size_t i = 0; i < _nLayers; i++) {
            outFile << _LayerModels[i] <<std::endl;
        }

        for (size_t i = 0; i < _Layers[0]._Node_number; i++) {
            for (auto edge : _Layers[0]._GraphContent[i]) {
                // 将信息写入文件
                outFile << edge.first.second << " " << i << " " << edge.second * 255 << std::endl;
            }
        }

        outFile.close(); // 关闭文件
        
    }
    
    //检查读图是否正确的代码，读者可以直接使用，本代码没有使用这个函数，因为本代码读图已经验证过为正确的
    void checkOverlapGraph(){
         std::ofstream outFile("checkOV.txt"); // 打开文件

        // 检查文件是否成功打开
        if (!outFile) {
            std::cerr << "Error opening file: " << "check.txt" << std::endl;
            return;
        }

        outFile << "nLayers "<< _nLayers <<std::endl;
        for (size_t i = 0; i < _nLayers; i++)
        {
            outFile << "LayerID:"<<i<<std::endl;
            for (size_t j = 0; j < _Layers[i]._OverlapGraph.size(); j++) 
            {
                if(_Layers[i]._OverlapGraph[j].size() != 0){
                    for (auto edge : _Layers[i]._OverlapGraph[j]){
                        outFile << j << " " << edge.first.first << " " << edge.first.second << std::endl;
                    }
                }
            }
            outFile << " "<<std::endl;
        }

        outFile.close(); // 关闭文件
    }

    int selectRandomLayer(const std::vector<Graph>& _Layers, int _nNodesAllLayers) {
        if (_Layers.empty() || _nNodesAllLayers <= 0) {
            throw std::runtime_error("Invalid input: no layers or zero total nodes.");
        }

        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, _nNodesAllLayers - 1);

        int r = dis(gen); // [0, _nNodesAllLayers-1]

        int cumulative = 0;
        for (const auto& g : _Layers) {
            if (g._Node_number <= 0 || g._Capacity == 0) {
                continue; // 跳过空图
            }

            cumulative += g._Node_number;
            if (r < cumulative) {
                return g._layer_id;  // 注意这里保持 layer_id，而不是 vector 下标
            }
        }

        // 如果逻辑正确，不会走到这里
        throw std::runtime_error("selectRandomLayer: no valid layer found!");
    }

};