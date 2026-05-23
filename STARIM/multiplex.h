#pragma once

class Multiplex
{ 
private:
    /// Result object.
	TResult& _tRes;



    uint32_t _nLayers;  

    uint32_t _nNodesAllLayers;  
    
    std::vector<Graph> _Layers;
    
    std::vector<CascadeModel> _LayerModels;

    
	size_t _numRRsets;  
    
    RRsets _RRsets; 
    
    size_t _numRRsetsVldt;  
    
    RRsets _RRsetsVldt;  

    
    Nodelist _vecSeed; 

    std::vector<uint32_t> _RRSub;
    std::vector<uint32_t> _RRSubVldt;

    std::vector<uint32_t> _coveredRRSetVldt;


    std::map<Node, int> _nodeRankMap;
    std::vector< std::vector< std::pair<Node, Inf> > > _seedNodeWithGain; 
    std::vector<Inf> _totalGain;

public:
    Multiplex( std::string fname, TResult& tRes, bool processTrainingData): _tRes(tRes)
    {

        _nLayers = 0;
        _nNodesAllLayers = 0;
        _numRRsets = 0;
        _numRRsetsVldt = 0;

        //预分配空间 
        _RRsets.reserve(5000);
        _RRsetsVldt.reserve(5000);
        _vecSeed.reserve(500);


        init(fname, processTrainingData); 
    }
	
    ~Multiplex()
	{
	}

    void init(std::string input, bool processTrainingData);

    void to_normal_accum_prob();

    void reserve_FRsets();

    void calculateSpread_build_rrset(const size_t numRRsets, const std::string mode);

    double calculateSpread_calculate(const std::vector<Node> &seed);

    void build_n_RRsets(const size_t numSamples, const std::string mode, bool processTrainingData = false, bool validateRRset = true);//bool validateRRset表示是否生成验证集的RR set

    void build_one_RRset(const Node uStart, const size_t hyperIdx, const bool validate, const std::string mode);

    double max_cover_stratifie(const int targetSize);

    double self_inf_cal_stratifie();

    
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

    void checkGraph(){
         std::ofstream outFile("check.txt"); 

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
                outFile << edge.first.second << " " << i << " " << edge.second * 255 << std::endl;
            }
        }

        outFile.close(); 
        
    }
    
    void checkOverlapGraph(){
         std::ofstream outFile("checkOV.txt"); 

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

        outFile.close(); 
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
                continue; 
            }

            cumulative += g._Node_number;
            if (r < cumulative) {
                return g._layer_id;  
            }
        }

        throw std::runtime_error("selectRandomLayer: no valid layer found!");
    }

};