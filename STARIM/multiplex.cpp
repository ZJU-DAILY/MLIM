#include "stdafx.h" 

void Multiplex::init(std::string input, bool processTrainingData) {

	uint32_t max_nodeID = GraphBase::read_max_nodeID(input);
	uint32_t n_layers = GraphBase::read_total_layers(input);
	uint32_t layers_capacity = n_layers + 1;
	uint32_t nodes_capacity = max_nodeID + 1;
	_Layers.resize(layers_capacity); 
	_LayerModels.resize(layers_capacity); 

	input += "/layer";
	std::string currFName = input + std::to_string(_nLayers);
	while(!file_exists(currFName + ".txt")){
		_nLayers++;
		currFName = input + std::to_string( _nLayers );
	}

	Nodelist crossLayerNode;
	while (file_exists(currFName + ".txt")) {
		++_nLayers;

		Graph singleGraph = GraphBase::read_graph(currFName + ".txt", _nLayers-1, nodes_capacity);
		_nNodesAllLayers += singleGraph._Node_number;

		currFName += "ov";
		GraphBase::read_overlapGraph(currFName + ".txt", singleGraph, crossLayerNode);

		_Layers[_nLayers - 1]	= singleGraph;

		currFName = input + std::to_string( _nLayers - 1 ) + "model";
		std::ifstream infile(currFName + ".txt");
		uint32_t cascadeModel;
		infile >> cascadeModel;
		CascadeModel layerModel = (CascadeModel) cascadeModel;
		_LayerModels[_nLayers - 1] = layerModel;
		infile.close();

		currFName = input + std::to_string( _nLayers );
		//std::cout << "read file finish: layer" << _nLayers - 1 << std::endl;
	}

	for(Node n : crossLayerNode){
		_Layers[n.first]._hasNode[n.second] = true;
	}

	_RRSub.resize(_nLayers);
	_RRSubVldt.resize(_nLayers);
	_coveredRRSetVldt.resize(_nLayers);
}

void Multiplex::to_normal_accum_prob()
{	
	for(int i = 0; i < _nLayers; i++){
		if(_Layers[i]._Capacity == 0) continue;

		if(_LayerModels[i] == LT){
			for(auto &edgelist : _Layers[i]._GraphContent){
				float accumVal = float(0.0);
				for (auto& edge : edgelist)
				{
					accumVal += edge.second;
					edge.second = accumVal;
				}
				for (auto& edge : edgelist)
				{
					edge.second /= accumVal;
				}
			}
		}
	}

}

void Multiplex::reserve_FRsets()
{
	for (int i = 0; i < _nLayers; i++)
	{
		if(_Layers[i]._Capacity == 0) continue;
		_Layers[i].reserve_singleGraph_FRsets();
	}
}

void Multiplex::calculateSpread_build_rrset(const size_t numRRsets, const std::string mode)
{
	build_n_RRsets(numRRsets, mode, false, false);

	return ;
}

double Multiplex::calculateSpread_calculate(const std::vector<Node> &seed)
{
	_vecSeed.clear();
	for (auto node : seed)
	{
		_vecSeed.push_back(node);
	}

	std::vector<bool> vecBoolVst(_numRRsets, false);
	std::vector<uint32_t> coveredRRSet(_nLayers,0);

	for (auto seed : _vecSeed)
	{
		for (auto RRsetIdx : _Layers[seed.first]._FRsets[seed.second])
		{
			if(vecBoolVst[RRsetIdx]) continue;

			coveredRRSet[_RRsets[RRsetIdx].sourceLayer]++;

			vecBoolVst[RRsetIdx] = true;
		}
	}

	double finalInf = 0.0;
	for (size_t i = 0; i < _nLayers; i++)
	{
		if(_Layers[i]._Capacity == 0) continue;
		if(_RRSub[i] == 0) continue; 
		finalInf += static_cast<double>(coveredRRSet[i]) / _RRSub[i] * _Layers[i]._Node_number;
	}
	
	return finalInf;
}

void Multiplex::build_n_RRsets(const size_t numSamples, const std::string mode, bool processTrainingData, bool validateRRset) //validateRRset默认为true
{
	if (numSamples > SIZE_MAX)
	{
		std::cout << "Error:R too large" << std::endl;
		exit(1);
	}

	const auto prevSize = _numRRsets;
	//std::cout<< prevSize <<std::endl;
	_numRRsets = _numRRsets > numSamples ? _numRRsets : numSamples;
	

	for (auto i = prevSize; i < numSamples; i++)
	{

		LayerID layer_id = selectRandomLayer(_Layers, _nNodesAllLayers);


		NodeID node_id = dsfmt_gv_genrand_uint32_range(_Layers[layer_id]._Capacity);
		while (_Layers[layer_id]._GraphContent[node_id].size() == 0 and _Layers[layer_id]._OverlapGraph[node_id].size() == 0) 
		{
			node_id = dsfmt_gv_genrand_uint32_range(_Layers[layer_id]._Capacity);
		}
		
		build_one_RRset(Node(layer_id,node_id), i, false, mode);
	}

	if(validateRRset)
	{
		const auto prevSizeVldt = _numRRsetsVldt;
		_numRRsetsVldt = _numRRsetsVldt > numSamples ? _numRRsetsVldt : numSamples;
		
		for (auto i = prevSizeVldt; i < numSamples; i++)
		{
			LayerID layer_id = selectRandomLayer(_Layers, _nNodesAllLayers);

			NodeID node_id = dsfmt_gv_genrand_uint32_range(_Layers[layer_id]._Capacity);
			while (_Layers[layer_id]._GraphContent[node_id].size() == 0 and _Layers[layer_id]._OverlapGraph[node_id].size() == 0) 
			{
				node_id = dsfmt_gv_genrand_uint32_range(_Layers[layer_id]._Capacity);
			}
			
			build_one_RRset(Node(layer_id,node_id), i, true, mode);
		}
	}
}

void Multiplex::build_one_RRset(const Node uStart, const size_t hyperIdx, const bool validate, const std::string mode)
{
	size_t numVisitNode = 0, currIdx = 0;
	LayerID uStart_layerID = uStart.first;
	NodeID uStart_nodeID = uStart.second;
	Nodelist vecVisitNode;
	vecVisitNode.reserve(5000);

	numVisitNode++;
	vecVisitNode.push_back(uStart);
	_Layers[uStart_layerID]._vecVisitBoolPerGraph[uStart_nodeID] = true;
	if(validate){
		_Layers[uStart_layerID]._FRsetsVldt[uStart_nodeID].push_back(hyperIdx);
	}else{
		_Layers[uStart_layerID]._FRsets[uStart_nodeID].push_back(hyperIdx);
	}

	while (currIdx < numVisitNode){
		const Node expand = vecVisitNode[currIdx++];
		const LayerID currLayerID = expand.first;
		const NodeID currNodeID = expand.second;

		for (Edge& inEdge : _Layers[currLayerID]._OverlapGraph[currNodeID])
        {
			const Node nbrNode = inEdge.first;
			const float edgeWeight = inEdge.second;
			const LayerID nbrLayerID = nbrNode.first;
			const NodeID nbrNodeID = nbrNode.second;

            if (_Layers[nbrLayerID]._vecVisitBoolPerGraph[nbrNodeID])
                continue;
            
            const auto randDouble = dsfmt_gv_genrand_open_close();
            if (randDouble > edgeWeight)
                continue;
            
            numVisitNode++;
            vecVisitNode.push_back(nbrNode);
			_Layers[nbrLayerID]._vecVisitBoolPerGraph[nbrNodeID] = true;
			if(validate){
				_Layers[nbrLayerID]._FRsetsVldt[nbrNodeID].push_back(hyperIdx);
			}else{
				_Layers[nbrLayerID]._FRsets[nbrNodeID].push_back(hyperIdx);
			}
        }

		if (_LayerModels[currLayerID] == IC)
		{
			for (Edge& inEdge : _Layers[currLayerID]._GraphContent[currNodeID])
			{
				const Node nbrNode = inEdge.first;
				const float edgeWeight = inEdge.second;
				const LayerID nbrLayerID = nbrNode.first;
				const NodeID nbrNodeID = nbrNode.second;

				if (_Layers[nbrLayerID]._vecVisitBoolPerGraph[nbrNodeID])
					continue;
				
				const auto randDouble = dsfmt_gv_genrand_open_close();
				if (randDouble > edgeWeight)
					continue;
				
				numVisitNode++;
				vecVisitNode.push_back(nbrNode);
				_Layers[nbrLayerID]._vecVisitBoolPerGraph[nbrNodeID] = true;
				if(validate){
					_Layers[nbrLayerID]._FRsetsVldt[nbrNodeID].push_back(hyperIdx);
				}else{
					_Layers[nbrLayerID]._FRsets[nbrNodeID].push_back(hyperIdx);
				}
			}
		}
		else if (_LayerModels[currLayerID] == LT)
		{
			if (_Layers[currLayerID]._GraphContent[currNodeID].size() == 0)
				continue;

			const auto nextNbrNodeIdx = gen_random_node_by_weight_LT(_Layers[currLayerID]._GraphContent[currNodeID]);
			if (nextNbrNodeIdx >= _Layers[currLayerID]._GraphContent[currNodeID].size()) 
				continue; // No element activated

			const LayerID nextNbrLayerID = currLayerID;
			const NodeID nextNbrNodeID = _Layers[currLayerID]._GraphContent[currNodeID][nextNbrNodeIdx].first.second;
			if (_Layers[nextNbrLayerID]._vecVisitBoolPerGraph[nextNbrNodeID])
				continue;

			numVisitNode++;
			vecVisitNode.push_back(Node(nextNbrLayerID,nextNbrNodeID));
			_Layers[nextNbrLayerID]._vecVisitBoolPerGraph[nextNbrNodeID] = true;
			if(validate){
				_Layers[nextNbrLayerID]._FRsetsVldt[nextNbrNodeID].push_back(hyperIdx);
			}else{
				_Layers[nextNbrLayerID]._FRsets[nextNbrNodeID].push_back(hyperIdx);
			}
		}
	}
	
	for (int i = 0; i < _nLayers; i++)
	{
		if(_Layers[i]._Capacity == 0) continue;
		_Layers[i].clean_visitedMark();
	}

	RRset rrSet;
	rrSet.rrSetContent = std::vector<Node>(vecVisitNode.begin(), vecVisitNode.begin() + numVisitNode);
	rrSet.sourceLayer = uStart_layerID;

	if(validate)
	{
		_RRsetsVldt.push_back(rrSet);
		if (mode == "S" || mode == "M"){
			_RRSubVldt[uStart_layerID]++;
		}
	}
	else
	{
		_RRsets.push_back(rrSet);
		if (mode == "S" || mode == "M"){
			_RRSub[uStart_layerID]++;
		}
	}
}

double Multiplex::max_cover_stratifie(const int targetSize)
{
	auto cmp = [](const std::pair<Node, Inf>& a, const std::pair<Node, Inf>& b) {
		return a.second > b.second || (a.second == b.second && a.first < b.first); // 先按 Inf 降序排序，再按 Node 升序排序（从小到大）
	};
	std::set<std::pair<Node, Inf>, decltype(cmp)> sortedSet(cmp);

	std::map<Node, Inf> nodeInfMap;

    auto insertNode = [&](Node node, Inf inf) {
        if (nodeInfMap.count(node)) {
            sortedSet.erase({node, nodeInfMap[node]});
        }
        nodeInfMap[node] = inf;
        sortedSet.insert({node, inf});
    };

	std::vector<uint32_t> coveredRRSet(_nLayers, 0);
	std::vector<bool> edgeMark(_numRRsets, false);
	_vecSeed.clear();

	for (auto i = 0; i < _nLayers; i++)
	{
		if(_Layers[i]._Capacity == 0) continue;

		for(auto j = 0; j < _Layers[i]._Capacity; j++){
			if(_Layers[i]._hasNode[j] == false) continue;

			std::fill(coveredRRSet.begin(), coveredRRSet.end(), 0);

			for (auto edgeIdx : _Layers[i]._FRsets[j]){
				coveredRRSet[_RRsets[edgeIdx].sourceLayer]++;
			}

			double nodeInf = 0.0;
			for (size_t t = 0; t < _nLayers; t++)
			{
				if(_Layers[t]._Capacity == 0) continue;
				if( _RRSub[t] == 0) continue; 

				nodeInf += static_cast<double>(coveredRRSet[t]) 
				/ _RRSub[t] * _Layers[t]._Node_number;
			}
			insertNode(Node(i,j), nodeInf);
		}
	}

	Inf sumInf = 0.0;
	if(targetSize<=_nNodesAllLayers){
		for (size_t idx = 0; idx < targetSize; idx++)
		{
			auto maxIt = sortedSet.begin();
			Node maxNode = maxIt->first;
			Inf maxInf = maxIt->second;
			sumInf += maxInf;

			_vecSeed.push_back(maxNode);
			sortedSet.erase({maxNode, nodeInfMap[maxNode]});
			nodeInfMap[maxNode] = 0.0;
			
			for (auto edgeIdx : _Layers[maxNode.first]._FRsets[maxNode.second]){
				if(edgeMark[edgeIdx]) continue;

				for (auto node : _RRsets[edgeIdx].rrSetContent)
				{
					if(nodeInfMap[node] == 0.0) continue;

					Inf nodeInf = nodeInfMap[node];
					nodeInf -= 1.0 / _RRSub[_RRsets[edgeIdx].sourceLayer] * _Layers[_RRsets[edgeIdx].sourceLayer]._Node_number;
					insertNode(node, nodeInf);
				}

				edgeMark[edgeIdx] = true;
			}
		}
		return sumInf;
	}
	else{
		std::cout<<"error ,targetSize is larger than nNodesAllLayers"<<std::endl;
		std::exit(EXIT_FAILURE);  
	}
}

double Multiplex::self_inf_cal_stratifie()
{
	std::vector<bool> vecBoolVst(_numRRsetsVldt, false);
	std::vector<uint32_t> coveredRRSet(_nLayers,0);

	for (auto seed : _vecSeed)
	{
		for (auto RRsetIdx : _Layers[seed.first]._FRsetsVldt[seed.second])
		{
			if(vecBoolVst[RRsetIdx]) continue;

			coveredRRSet[_RRsetsVldt[RRsetIdx].sourceLayer]++;

			vecBoolVst[RRsetIdx] = true;
		}
	}

	double finalInf = 0.0;
	for (size_t i = 0; i < _nLayers; i++)
	{
		if(_Layers[i]._Capacity == 0) continue;
		if(_RRSubVldt[i] == 0) continue; 
		finalInf += static_cast<double>(coveredRRSet[i]) / _RRSubVldt[i] * _Layers[i]._Node_number;
	}
	
	return finalInf;
}

void Multiplex::mgrrFlexible(const int targetSize, const std::string mode, const double delta, std::string fname, const double epsilon)
{
	size_t numRRsets = 500; 
	size_t iterator = 1;  
	double infOrigin = 0.0; 
	double infVldt = 0.0;
	const double e = exp(1);//e
	Timer timerMGRR("MGRR");
	while(true)
	{
		build_n_RRsets(numRRsets, mode);

		infOrigin = max_cover_stratifie(targetSize);//fR1(S)
		infVldt = self_inf_cal_stratifie();//fR2(S)

		const auto lamda = infOrigin / infVldt;
		const auto tmp = infVldt / log(5 * pow2(iterator) / delta);
		const auto minVldt = find_min_RRSub(_RRSubVldt);
		const auto min = find_min_RRSub(_RRSub);
		double epsilon1 = 0.0;
		try {
			epsilon1 = solveQuadratic(1-tmp * minVldt,3,2);
		} catch (const std::runtime_error& e) {
			std::cerr << e.what() << std::endl;
		}

		const auto epsilon2 = std::sqrt((2 * epsilon1 + 2) / (tmp * min));
		const auto threshold = (1-1/e) / (1-1/e-epsilon) * ((1-epsilon2) / (1+epsilon1));

		if((lamda>0) && (lamda<=threshold)){
			if( ((epsilon1>0) && (epsilon1<1)) && ((epsilon2>0) && (epsilon2<1))){
				break;
			}
		}
		
		const auto bound = (8 + 2 * epsilon) * (1 + epsilon1) 
							* (log(6/delta) + _nNodesAllLayers * log(2)) / (epsilon * epsilon * infVldt);
		
		if(min >= bound){
			break;
		}

		iterator++;
		numRRsets *= 2;

	}
	double total_time = timerMGRR.get_total_time();
	_tRes.set_running_time(total_time);

	
	_tRes.set_influence(infVldt);//保存影响力结果
	_tRes.set_influence_original(infOrigin);//保存原始影响力
	_tRes.set_influence_MC(0);//保存MC影响力
	_tRes.set_seed_vec(_vecSeed);//保存种子集
	_tRes.set_RR_sets_size(_numRRsets * 2);//保存创建RR集合的大小
	_tRes.set_round(iterator);

}

double Multiplex::monteCarloInfluence(const Nodelist& vecSeed, uint32_t nIter)
{
	double est = 0.0;
	for (size_t i = 0; i < nIter; ++i) {
		std::vector<Graph> sampled_multigraph;

		sampleMultiplex(sampled_multigraph);

		est += static_cast< double >( forwardProp(sampled_multigraph, vecSeed) );

		if(i%100==0){
			std::cout<<"MC iter i="<< i <<std::endl;
		}
	}

	est /= nIter;

	return est;
}

void Multiplex::sampleMultiplex(std::vector<Graph>& sampled_multigraph)
{
	sampled_multigraph.clear();
	for (size_t layer_id = 0; layer_id < _nLayers; layer_id++){
		if(_Layers[layer_id]._Capacity == 0){
			Graph sampled_graph = Graph(layer_id, 0, 0, 0);
			sampled_multigraph.push_back(sampled_graph);
			continue;
		}
		Graph& origin_graph = _Layers[layer_id]; 
		Graph sampled_graph = Graph(layer_id, origin_graph._Node_number, 
			origin_graph._Edge_number, origin_graph._Capacity);
		if (_LayerModels[layer_id] == IC){
			for(auto i = 0; i < origin_graph._Capacity; i++){//1-654
				if(origin_graph._hasNode[i] == false) continue;
				for (Edge& inEdge : origin_graph._GraphContent[i]){//nbrNode->i
					const Node nbrNode = inEdge.first;
					const float edgeWeight = inEdge.second;
					const auto randDouble = dsfmt_gv_genrand_open_close();
					if(randDouble < edgeWeight){
						sampled_graph._GraphContent[nbrNode.second].push_back(Edge(Node(layer_id,i),1));
					}
				}
			}

			sampled_graph._hasNode = origin_graph._hasNode;
			sampled_graph._OverlapGraph.resize(sampled_graph._Capacity);
			for (size_t i = 0; i < sampled_graph._Capacity; i++)
			{
				sampled_graph._OverlapGraph[i].reserve(10);
			}
			sampled_graph._vecIsSeed.resize(sampled_graph._Capacity);
		}
		else if (_LayerModels[layer_id] == LT){
			//层内边复制
			for(auto i = 0; i < origin_graph._Capacity; i++){
				if(origin_graph._hasNode[i] == false) continue;
				for (Edge& inEdge : origin_graph._GraphContent[i]){
					const Node nbrNode = inEdge.first;
					const float edgeWeight = inEdge.second;
					sampled_graph._GraphContent[nbrNode.second].push_back(Edge(Node(layer_id,i),edgeWeight));
				}
			}

			sampled_graph._hasNode = origin_graph._hasNode;
			sampled_graph._OverlapGraph.resize(sampled_graph._Capacity);
			for (size_t i = 0; i < sampled_graph._Capacity; i++)
			{
				sampled_graph._OverlapGraph[i].reserve(10);
			}
			sampled_graph._vecIsSeed.resize(sampled_graph._Capacity);
			sampled_graph.setThreshold();
		}

		sampled_multigraph.push_back(sampled_graph);
	} 

	for (unsigned layer_id = 0; layer_id < _nLayers; ++layer_id){
		if(_Layers[layer_id]._Capacity == 0) continue;

		Graph& origin_graph = _Layers[layer_id];
		for(auto i = 0; i < origin_graph._Capacity; i++){
			if(origin_graph._hasNode[i] == false) continue;
			for (Edge& inEdge : origin_graph._OverlapGraph[i]){
				const Node nbrNode = inEdge.first;
				const float edgeWeight = inEdge.second;
				const auto randDouble = dsfmt_gv_genrand_open_close();
				if(randDouble < edgeWeight){
					sampled_multigraph[nbrNode.first]._OverlapGraph[nbrNode.second].push_back(Edge(Node(layer_id,i),1));
				}
			}
		}
	}
}

uint32_t Multiplex::forwardProp(std::vector<Graph> sampled_multigraph, const Nodelist& vecSeed)
{
	uint32_t total_activate = 0;
	std::queue<Node> Q;

	for (auto seed : vecSeed){
		Q.push(seed);
		sampled_multigraph[seed.first]._vecIsSeed[seed.second] = true;
	}

	while ((!Q.empty()))
	{
		Node curr = Q.front();//<layer,node>
		Q.pop();
		total_activate++;

		const LayerID currLayerID = curr.first;
		const NodeID currNodeID = curr.second;
		for (Edge& inEdge : sampled_multigraph[currLayerID]._OverlapGraph[currNodeID])
		{
			const Node nbrNode = inEdge.first;
			if(sampled_multigraph[nbrNode.first]._vecIsSeed[nbrNode.second] == true) continue;
			Q.push(nbrNode);
			sampled_multigraph[nbrNode.first]._vecIsSeed[nbrNode.second] = true;
		}

		if(_LayerModels[currLayerID] == IC){
			for (Edge& inEdge : sampled_multigraph[currLayerID]._GraphContent[currNodeID])
			{
				const Node nbrNode = inEdge.first;
				if(sampled_multigraph[nbrNode.first]._vecIsSeed[nbrNode.second] == true) continue;
				Q.push(nbrNode);
				sampled_multigraph[nbrNode.first]._vecIsSeed[nbrNode.second] = true;
			}
		}
		else if (_LayerModels[currLayerID] == LT){
			for (Edge& inEdge : _Layers[currLayerID]._GraphContent[currNodeID]){
				const Node nbrNode = inEdge.first;
				const double edgeWeight = inEdge.second;
				const LayerID nbrLayerID = nbrNode.first;
				const NodeID nbrNodeID = nbrNode.second;
				sampled_multigraph[nbrLayerID].Cumulative_weight[nbrNodeID] += edgeWeight;

				const double weight = sampled_multigraph[nbrLayerID].Cumulative_weight[nbrNodeID];
				if(weight>sampled_multigraph[nbrLayerID].Threshold[nbrNodeID]){
					if(sampled_multigraph[nbrLayerID]._vecIsSeed[nbrNodeID] == true) continue;
					Q.push(nbrNode);
					sampled_multigraph[nbrLayerID]._vecIsSeed[nbrNodeID] = true;
				}
			}
		}

	}
	
	return total_activate;
}

double Multiplex::seedScore(const int targetSize, const std::string mode, const double delta, const double epsilon, std::string dir)
{
	size_t numRRsets = 10000; 
	size_t iterator = 1;  
	double infOrigin = 0.0; 
	double infVldt = 0.0;
	const double e = exp(1);//e
	Timer timerMGRR("MGRR");

	build_n_RRsets(numRRsets, mode);

	std::map<Node, Inf> nodeInfMap;


	std::vector<uint32_t> coveredRRSet(_nLayers, 0);
	std::vector<bool> edgeMark(_numRRsets, false);

	for (auto i = 0; i < _nLayers; i++)
	{
		if(_Layers[i]._Capacity == 0) continue;

		for(auto j = 0; j < _Layers[i]._Capacity; j++){
			if(_Layers[i]._hasNode[j] == false) continue;

			std::fill(coveredRRSet.begin(), coveredRRSet.end(), 0);

			for (auto edgeIdx : _Layers[i]._FRsets[j]){
				coveredRRSet[_RRsets[edgeIdx].sourceLayer]++;
			}

			double nodeInf = 0.0;
			for (size_t t = 0; t < _nLayers; t++)
			{
				if(_Layers[t]._Capacity == 0) continue;
				if( _RRSub[t] == 0) continue; 

				nodeInf += static_cast<double>(coveredRRSet[t]) 
				/ _RRSub[t] * _Layers[t]._Node_number;
			}
			nodeInfMap[Node(i,j)] = nodeInf;		
		}
	}
    std::vector<std::pair<Node, double>> vec(nodeInfMap.begin(), nodeInfMap.end());
	std::sort(vec.begin(), vec.end(), [](const std::pair<Node, double>& a, const std::pair<Node, double>& b) {
		return a.second > b.second; 
	});
	double total_time = timerMGRR.get_total_time();
	std::cout<<"time to generate node score: "<< total_time <<std::endl;


	return 0.0;
}

void Multiplex::RandomK(const int targetSize)
{
	_vecSeed.clear();
	Timer timerMGRR("MGRR");
	for (size_t i = 0; i < targetSize; i++)
	{
		LayerID layer_id = selectRandomLayer(_Layers, _nNodesAllLayers);

		NodeID node_id = dsfmt_gv_genrand_uint32_range(_Layers[layer_id]._Capacity);
		while (_Layers[layer_id]._GraphContent[node_id].size() == 0 and _Layers[layer_id]._OverlapGraph[node_id].size() == 0) 
		{
			node_id = dsfmt_gv_genrand_uint32_range(_Layers[layer_id]._Capacity);
		}

		_vecSeed.push_back(Node(layer_id, node_id));
	}

	uint32_t RRset_num = 4000;
	for (size_t i = 0; i < RRset_num; i++)
	{

		LayerID layer_id = selectRandomLayer(_Layers, _nNodesAllLayers);

		NodeID node_id = dsfmt_gv_genrand_uint32_range(_Layers[layer_id]._Capacity);
		while (_Layers[layer_id]._GraphContent[node_id].size() == 0 and _Layers[layer_id]._OverlapGraph[node_id].size() == 0) 
		{
			node_id = dsfmt_gv_genrand_uint32_range(_Layers[layer_id]._Capacity);
		}
		
		build_one_RRset(Node(layer_id,node_id), i, true, "M");
	}
	_numRRsetsVldt = 4000;
	double infVldt = self_inf_cal_stratifie();
	//double infMC = monteCarloInfluence(_vecSeed);
	double total_time = timerMGRR.get_total_time();
	_tRes.set_running_time(total_time);

	_tRes.set_influence(infVldt);
	_tRes.set_influence_original(0);
	_tRes.set_influence_MC(0);
	_tRes.set_seed_vec(_vecSeed);
	_tRes.set_RR_sets_size(4000);
	_tRes.set_round(0);
}

void Multiplex::BestDegree(const int targetSize)
{
	_vecSeed.clear();

	Timer timerMGRR("MGRR");
	std::map<Node, int> nodeDegreeMap;
	for (size_t i = 0; i < _nLayers; i++)
	{
		if (_Layers[i]._Capacity == 0) continue;
		
		for (size_t j = 0; j < _Layers[i]._Capacity; j++)
		{
			if(_Layers[i]._hasNode[j] == false) continue;

			for (Edge& inEdge : _Layers[i]._GraphContent[j]){
				const Node nbrNode = inEdge.first;
				nodeDegreeMap[nbrNode]++;
			}

			for (Edge& inEdge : _Layers[i]._OverlapGraph[j]){
				const Node nbrNode = inEdge.first;
				nodeDegreeMap[nbrNode]++;
			}
		}
		
	}

	std::vector<std::pair<Node, int>> vec(nodeDegreeMap.begin(), nodeDegreeMap.end());
	sort(vec.begin(), vec.end(), [](const std::pair<Node, int>& a, const std::pair<Node, int>& b) {
        return a.second > b.second;
    });

	for (int i = 0; i < targetSize && i < vec.size(); ++i) {
        _vecSeed.push_back(vec[i].first);
    }

	uint32_t RRset_num = 4000;
	for (size_t i = 0; i < RRset_num; i++)
	{
			LayerID layer_id = selectRandomLayer(_Layers, _nNodesAllLayers);

			NodeID node_id = dsfmt_gv_genrand_uint32_range(_Layers[layer_id]._Capacity);
			while (_Layers[layer_id]._GraphContent[node_id].size() == 0 and _Layers[layer_id]._OverlapGraph[node_id].size() == 0) 
			{
				node_id = dsfmt_gv_genrand_uint32_range(_Layers[layer_id]._Capacity);
			}
			
			build_one_RRset(Node(layer_id,node_id), i, true, "M");
	}
	_numRRsetsVldt = 4000;
	double infVldt = self_inf_cal_stratifie();
	//double infMC = monteCarloInfluence(_vecSeed);
	double total_time = timerMGRR.get_total_time();
	_tRes.set_running_time(total_time);
	
	_tRes.set_influence(infVldt);
	_tRes.set_influence_original(0);
	_tRes.set_influence_MC(0);
	_tRes.set_seed_vec(_vecSeed);
	_tRes.set_RR_sets_size(4000);
	_tRes.set_round(0);
}

void Multiplex::deepIM_influence(const int targetSize, std::string dir)
{
	_vecSeed.clear();
	std::string filename = dir + "/seed_set.txt";
	std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        exit(1);
    }

    int totalSize = 0;
    infile >> totalSize;

	if (totalSize != targetSize) {
        std::cerr << "Error: file node count " << totalSize 
                  << " does not match targetSize " << targetSize << std::endl;
        exit(1);
    }

    int layer_id, node_id;
    char comma;  

    while (infile >> layer_id >> comma >> node_id) {
        _vecSeed.push_back(Node(layer_id, node_id));
    }

    if (_vecSeed.size() != static_cast<size_t>(targetSize)) {
        std::cerr << "Error: read node count " << _vecSeed.size() 
                  << " does not match targetSize " << targetSize << std::endl;
        exit(1);
    }

    infile.close();


	uint32_t RRset_num = 10000;
	for (size_t i = 0; i < RRset_num; i++)
	{

		LayerID layer_id = dsfmt_gv_genrand_uint32_range(_nLayers);
		while(_Layers[layer_id]._Capacity == 0){
			layer_id = dsfmt_gv_genrand_uint32_range(_nLayers);
		}

		NodeID node_id = dsfmt_gv_genrand_uint32_range(_Layers[layer_id]._Capacity);
		while (_Layers[layer_id]._GraphContent[node_id].size() == 0 and _Layers[layer_id]._OverlapGraph[node_id].size() == 0) 
		{
			node_id = dsfmt_gv_genrand_uint32_range(_Layers[layer_id]._Capacity);
		}
		
		build_one_RRset(Node(layer_id,node_id), i, true, "M");
	}
	_numRRsetsVldt = 10000;
	double infVldt = self_inf_cal_stratifie();

	_tRes.set_influence(infVldt);
	_tRes.set_influence_original(0);
	_tRes.set_influence_MC(0);
	_tRes.set_seed_vec(_vecSeed);
	_tRes.set_RR_sets_size(10000);
	_tRes.set_round(0);
}

void Multiplex::KSN_influence(const int targetSize, std::string dir)
{
	_vecSeed.clear();
	std::string filename = dir + "/seed_set_k_"+ std::to_string(targetSize) +".txt";

	std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        exit(1);
    }

    int totalSize = 0;
    infile >> totalSize;

    int layer_id, node_id;
    char comma; 

    while (infile >> layer_id >> comma >> node_id) {
        _vecSeed.push_back(Node(layer_id, node_id));
    }

    infile.close();

	uint32_t RRset_num = 10000;
	for (size_t i = 0; i < RRset_num; i++)
	{
		LayerID layer_id = selectRandomLayer(_Layers, _nNodesAllLayers);

		NodeID node_id = dsfmt_gv_genrand_uint32_range(_Layers[layer_id]._Capacity);
		while (_Layers[layer_id]._GraphContent[node_id].size() == 0 and _Layers[layer_id]._OverlapGraph[node_id].size() == 0) 
		{
			node_id = dsfmt_gv_genrand_uint32_range(_Layers[layer_id]._Capacity);
		}
		
		build_one_RRset(Node(layer_id,node_id), i, true, "M");
	}
	_numRRsetsVldt = 10000;
	double infVldt = self_inf_cal_stratifie();

	_tRes.set_influence(infVldt);
	_tRes.set_influence_original(0);
	_tRes.set_influence_MC(0);
	_tRes.set_seed_vec(_vecSeed);
	_tRes.set_RR_sets_size(10000);
	_tRes.set_round(0);
}
