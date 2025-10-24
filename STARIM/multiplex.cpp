#include "stdafx.h" //stdafx.h已经包含了multiplex.h

void Multiplex::init(std::string input, bool processTrainingData) {
	//默认：若不是生成训练图，那么节点编号没有大于等于节点总数的；若是利用训练图生成数据，则需要处理节点编号与节点总数的关系

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

	Nodelist crossLayerNode;//KSN中没使用
	while (file_exists(currFName + ".txt")) {
		//为下次读图做准备
		++_nLayers;

		//读图
		Graph singleGraph = GraphBase::read_graph(currFName + ".txt", _nLayers-1, nodes_capacity);
		_nNodesAllLayers += singleGraph._Node_number;

		//读取本图上的重叠节点
		currFName += "ov";
		GraphBase::read_overlapGraph(currFName + ".txt", singleGraph, crossLayerNode);

		//将当前层的图 singleGraph 添加到 Layers 容器中，存储所有层的图信息
		_Layers[_nLayers - 1]	= singleGraph;

		//读取当前图的传播模型
		currFName = input + std::to_string( _nLayers - 1 ) + "model";
		std::ifstream infile(currFName + ".txt");
		uint32_t cascadeModel;
		infile >> cascadeModel;
		CascadeModel layerModel = (CascadeModel) cascadeModel;
		_LayerModels[_nLayers - 1] = layerModel;
		infile.close();

		//更新 lfname 为下一个层的文件名，准备下一次循环。
		currFName = input + std::to_string( _nLayers );
		//std::cout << "read file finish: layer" << _nLayers - 1 << std::endl;
	}

	//处理跨层节点，把他们标注为存在
	for(Node n : crossLayerNode){
		_Layers[n.first]._hasNode[n.second] = true;
	}

	//初始化RRSub与RRSubVldt与coveredRRSetVldt
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
				// Normalization
				//这里应该是对应LT模型的一种方式，LT模型的边权重等于1/入度，所以这里应该是对边权重进行归一化
				//也保证了任一节点入边权重和<=1
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
	//生成指定数量的RR set，选择集和验证集都生成numRRsets
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
	std::vector<uint32_t> coveredRRSet(_nLayers,0);//分层方法中，覆盖的子RR set个数

	//计算分层方法中，种子集在每一层上覆盖的RR set个数
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
		if(_RRSub[i] == 0) continue; //若该层就没有被采样到
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

	//生成选择种子集的RR set
	const auto prevSize = _numRRsets;
	//std::cout<< prevSize <<std::endl;
	_numRRsets = _numRRsets > numSamples ? _numRRsets : numSamples;
	
	//从已有集合数量（prevSize）开始，生成新的 RR 集合，直到达到目标数量 numSamples。
	//这里for循环的终止条件没有问题
	for (auto i = prevSize; i < numSamples; i++)
	{
		//生成随机的起点
		LayerID layer_id = selectRandomLayer(_Layers, _nNodesAllLayers);
		/*LayerID layer_id = dsfmt_gv_genrand_uint32_range(_nLayers);
		while(_Layers[layer_id]._Capacity == 0){
			layer_id = dsfmt_gv_genrand_uint32_range(_nLayers);
		}*/

		NodeID node_id = dsfmt_gv_genrand_uint32_range(_Layers[layer_id]._Capacity);
		while (_Layers[layer_id]._GraphContent[node_id].size() == 0 and _Layers[layer_id]._OverlapGraph[node_id].size() == 0) 
		{//说明这个点要么不存在，要么没有一条入边
			node_id = dsfmt_gv_genrand_uint32_range(_Layers[layer_id]._Capacity);//再次进行随机
		}
		
		build_one_RRset(Node(layer_id,node_id), i, false, mode);
	}

	if(validateRRset)
	{
		//生成验证影响力的RR set
		const auto prevSizeVldt = _numRRsetsVldt;
		_numRRsetsVldt = _numRRsetsVldt > numSamples ? _numRRsetsVldt : numSamples;
		
		//从已有集合数量（prevSize）开始，生成新的 RR 集合，直到达到目标数量 numSamples。
		for (auto i = prevSizeVldt; i < numSamples; i++)
		{
			//生成随机的起点
			LayerID layer_id = selectRandomLayer(_Layers, _nNodesAllLayers);

			NodeID node_id = dsfmt_gv_genrand_uint32_range(_Layers[layer_id]._Capacity);//需要验证集则说明不是训练图，则_Capacity和_Node_number是一样的
			while (_Layers[layer_id]._GraphContent[node_id].size() == 0 and _Layers[layer_id]._OverlapGraph[node_id].size() == 0) 
			{//说明这个点要么不存在，要么没有一条入边
				node_id = dsfmt_gv_genrand_uint32_range(_Layers[layer_id]._Capacity);//再次进行随机
			}
			
			build_one_RRset(Node(layer_id,node_id), i, true, mode);
		}
	}
}

void Multiplex::build_one_RRset(const Node uStart, const size_t hyperIdx, const bool validate, const std::string mode)
{
	//numVisitNode：记录已访问的节点数  currIdx：表示当前处理的节点在队列中的位置
	size_t numVisitNode = 0, currIdx = 0;
	LayerID uStart_layerID = uStart.first;
	NodeID uStart_nodeID = uStart.second;
	Nodelist vecVisitNode;
	vecVisitNode.reserve(5000);//预留空间

	//从起点开始
	numVisitNode++;
	vecVisitNode.push_back(uStart);
	_Layers[uStart_layerID]._vecVisitBoolPerGraph[uStart_nodeID] = true;
	if(validate){
		_Layers[uStart_layerID]._FRsetsVldt[uStart_nodeID].push_back(hyperIdx);
	}else{
		_Layers[uStart_layerID]._FRsets[uStart_nodeID].push_back(hyperIdx);
	}

	while (currIdx < numVisitNode){
		//expand 表示当前正在扩展的节点，currIdx++ 用于更新当前节点的索引。
		const Node expand = vecVisitNode[currIdx++];
		const LayerID currLayerID = expand.first;
		const NodeID currNodeID = expand.second;

		//层间传播,先对每一个节点进行层间传播，避免被LT模型的代码逻辑给跳过
		for (Edge& inEdge : _Layers[currLayerID]._OverlapGraph[currNodeID])
        {
			const Node nbrNode = inEdge.first;
			const float edgeWeight = inEdge.second;
			const LayerID nbrLayerID = nbrNode.first;
			const NodeID nbrNodeID = nbrNode.second;

            //如果邻居节点 nbrId 已被访问，则跳过。
            if (_Layers[nbrLayerID]._vecVisitBoolPerGraph[nbrNodeID])
                continue;
            
            //生成一个随机数 randDouble，如果这个随机数大于邻居节点的激活概率，则跳过。
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

		//层内传播
		if (_LayerModels[currLayerID] == IC)
		{
			for (Edge& inEdge : _Layers[currLayerID]._GraphContent[currNodeID])
			{//遍历当前节点 expand 的邻居
				const Node nbrNode = inEdge.first;
				const float edgeWeight = inEdge.second;
				const LayerID nbrLayerID = nbrNode.first;
				const NodeID nbrNodeID = nbrNode.second;

				//如果邻居节点 nbrId 已被访问，则跳过。
				if (_Layers[nbrLayerID]._vecVisitBoolPerGraph[nbrNodeID])
					continue;
				
				//生成一个随机数 randDouble，如果这个随机数大于邻居节点的激活概率，则跳过。
				const auto randDouble = dsfmt_gv_genrand_open_close();
				if (randDouble > edgeWeight)
					continue;
				
				//如果条件满足，则将邻居节点 nbrId 添加到访问列表，并标记为已访问。
				//同时，将当前 RR 集合索引 hyperIdx 添加到邻居节点的前向传播集合 _FRsets 中
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
			//如果当前节点没有邻居，则继续处理下一个节点。
			if (_Layers[currLayerID]._GraphContent[currNodeID].size() == 0)
				continue;

			//根据权重选择一个邻居节点 nextNbrIdx，如果无激活的节点，则处理下一个节点。
			const auto nextNbrNodeIdx = gen_random_node_by_weight_LT(_Layers[currLayerID]._GraphContent[currNodeID]);
			if (nextNbrNodeIdx >= _Layers[currLayerID]._GraphContent[currNodeID].size()) 
				continue; // No element activated

			//如果选择的邻居节点已经被访问，则处理下一个节点。
			const LayerID nextNbrLayerID = currLayerID;
			const NodeID nextNbrNodeID = _Layers[currLayerID]._GraphContent[currNodeID][nextNbrNodeIdx].first.second;
			if (_Layers[nextNbrLayerID]._vecVisitBoolPerGraph[nextNbrNodeID])
				continue;

			//如果条件满足，将邻居节点添加到访问列表，并标记为已访问，同时记录其逆传播集合
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
	
	//清除访问标记，将访问列表中的节点标记为未访问，为下次构建 RR 集合做准备。
	for (int i = 0; i < _nLayers; i++)
	{
		if(_Layers[i]._Capacity == 0) continue;
		_Layers[i].clean_visitedMark();
	}

	//构建当前生成的 RR 集合，记录当前逆传播集合中的所有节点与采样源节点所属的层编号
	RRset rrSet;
	rrSet.rrSetContent = std::vector<Node>(vecVisitNode.begin(), vecVisitNode.begin() + numVisitNode);
	rrSet.sourceLayer = uStart_layerID;

	if(validate)
	{
		_RRsetsVldt.push_back(rrSet);//添加到验证集中,整体方法就到这里就结束了
		if (mode == "S" || mode == "M"){
			_RRSubVldt[uStart_layerID]++;//验证集上采样源节点所属的层编号上的RR set加一
		}
	}
	else
	{
		_RRsets.push_back(rrSet);//添加到选择集中,整体方法就到这里就结束了
		if (mode == "S" || mode == "M"){
			_RRSub[uStart_layerID]++;//选择集上采样源节点所属的层编号上的RR set加一
		}
	}
}

double Multiplex::max_cover_stratifie(const int targetSize)
{
	auto cmp = [](const std::pair<Node, Inf>& a, const std::pair<Node, Inf>& b) {
		return a.second > b.second || (a.second == b.second && a.first < b.first); // 先按 Inf 降序排序，再按 Node 升序排序（从小到大）
	};
	std::set<std::pair<Node, Inf>, decltype(cmp)> sortedSet(cmp);

	std::map<Node, Inf> nodeInfMap;//sortedSet是有序结构，不能直接访问指定node的Inf，所以需要辅助存储结构
	// 插入数据的同时维护排序集合
    auto insertNode = [&](Node node, Inf inf) {
        if (nodeInfMap.count(node)) {
            // 如果节点已存在，从排序集合中移除旧值
            sortedSet.erase({node, nodeInfMap[node]});
        }
        nodeInfMap[node] = inf;
        sortedSet.insert({node, inf});
    };

	std::vector<uint32_t> coveredRRSet(_nLayers, 0);//分层方法中，每一层上覆盖的RR set个数
	std::vector<bool> edgeMark(_numRRsets, false);//初始化布尔数组 edgeMark 用于标记边是否已被处理过
	_vecSeed.clear();//OPIM中没有对种子集初始化数量

	for (auto i = 0; i < _nLayers; i++)
	{
		if(_Layers[i]._Capacity == 0) continue;

		for(auto j = 0; j < _Layers[i]._Capacity; j++){
			if(_Layers[i]._hasNode[j] == false) continue;

			std::fill(coveredRRSet.begin(), coveredRRSet.end(), 0);//计算开始前设置为0

			for (auto edgeIdx : _Layers[i]._FRsets[j]){
				coveredRRSet[_RRsets[edgeIdx].sourceLayer]++;
			}

			//无偏估计，计算影响力
			double nodeInf = 0.0;
			for (size_t t = 0; t < _nLayers; t++)
			{
				if(_Layers[t]._Capacity == 0) continue;
				if( _RRSub[t] == 0) continue; //若该层就没有被采样到

				nodeInf += static_cast<double>(coveredRRSet[t]) 
				/ _RRSub[t] * _Layers[t]._Node_number;
			}
			insertNode(Node(i,j), nodeInf);
		}
	}

	Inf sumInf = 0.0;
	//按照inf从低到高去选择节点
	if(targetSize<=_nNodesAllLayers){
		for (size_t idx = 0; idx < targetSize; idx++)
		{
			auto maxIt = sortedSet.begin();
			Node maxNode = maxIt->first;
			Inf maxInf = maxIt->second;
			sumInf += maxInf;

			_vecSeed.push_back(maxNode);
			sortedSet.erase({maxNode, nodeInfMap[maxNode]});//debug检查一下
			nodeInfMap[maxNode] = 0.0;
			
			for (auto edgeIdx : _Layers[maxNode.first]._FRsets[maxNode.second]){
				if(edgeMark[edgeIdx]) continue;

				for (auto node : _RRsets[edgeIdx].rrSetContent)
				{//这里有很多重复的插入操作，除了这个已经很优化了
					if(nodeInfMap[node] == 0.0) continue; //已经被选为种子节点，跳过

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
		std::exit(EXIT_FAILURE);  // 使用标准退出，EXIT_FAILURE 表示失败状态
	}
}

double Multiplex::self_inf_cal_stratifie()
{
	std::vector<bool> vecBoolVst(_numRRsetsVldt, false);
	std::vector<uint32_t> coveredRRSet(_nLayers,0);//分层方法中，覆盖的子RR set个数

	//计算分层方法中，种子集在每一层上覆盖的RR set个数
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
		if(_RRSubVldt[i] == 0) continue; //若该层就没有被采样到
		finalInf += static_cast<double>(coveredRRSet[i]) / _RRSubVldt[i] * _Layers[i]._Node_number;
	}
	
	return finalInf;
}

void Multiplex::mgrrFlexible(const int targetSize, const std::string mode, const double delta, std::string fname, const double epsilon)
{
	size_t numRRsets = 500; //一开始要生成的RR set个数设为总节点个数
	size_t iterator = 1;  //迭代次数
	double infOrigin = 0.0; 
	double infVldt = 0.0;
	const double e = exp(1);//e
	Timer timerMGRR("MGRR");//初始化一个计时器
	while(true)
	{
		build_n_RRsets(numRRsets, mode);//选择集和验证集都生成numRRsets个

		//TODO: 1.检查一下max_cover是否有更优化的算法  2.看看用effic_inf_valid_algo是否更快

		//这个函数估计的影响力还是略有问题，需要修改
		//max_cover_lazyNew也可以只计算新的，把旧的一起加合
		infOrigin = max_cover_stratifie(targetSize);//fR1(S) 需要把max_cover_lazyNew弄成两个函数，另一个叫effect
		infVldt = self_inf_cal_stratifie();//fR2(S)
		//infVldt = effic_inf_valid_algo(numRRsets);

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
		//需要修改连续生成RRset，主要是分层方法
	}
	double total_time = timerMGRR.get_total_time();
	_tRes.set_running_time(total_time);//保存时间

	/*
	double infMC = monteCarloInfluence(_vecSeed);
	for (size_t i = 0; i < _vecSeed.size(); i++)
	{
		std::cout<< _vecSeed[i].first<<","<< _vecSeed[i].second <<std::endl;
	}*/

	/*
	//定义存储节点影响力的数据结构
	std::map<Node, Inf> nodeInfMap;

	//计算每个节点的影响力
	std::vector<uint32_t> coveredRRSet(_nLayers, 0);//分层方法中，每一层上覆盖的RR set个数
	std::vector<bool> edgeMark(_numRRsets, false);//初始化布尔数组 edgeMark 用于标记边是否已被处理过

	//初始化每个节点的影响力
	for (auto i = 0; i < _nLayers; i++)
	{
		if(_Layers[i]._Capacity == 0) continue;

		for(auto j = 0; j < _Layers[i]._Capacity; j++){
			if(_Layers[i]._hasNode[j] == false) continue;

			std::fill(coveredRRSet.begin(), coveredRRSet.end(), 0);//计算开始前设置为0

			for (auto edgeIdx : _Layers[i]._FRsets[j]){
				coveredRRSet[_RRsets[edgeIdx].sourceLayer]++;
			}

			//无偏估计，计算影响力
			double nodeInf = 0.0;
			for (size_t t = 0; t < _nLayers; t++)
			{
				if(_Layers[t]._Capacity == 0) continue;
				if( _RRSub[t] == 0) continue; //若该层就没有被采样到

				nodeInf += static_cast<double>(coveredRRSet[t]) 
				/ _RRSub[t] * _Layers[t]._Node_number;
			}
			nodeInfMap[Node(i,j)] = nodeInf;		
		}
	}
    std::vector<std::pair<Node, double>> vec(nodeInfMap.begin(), nodeInfMap.end());
	// 使用 Lambda 表达式对 vector 进行降序排序
	std::sort(vec.begin(), vec.end(), [](const std::pair<Node, double>& a, const std::pair<Node, double>& b) {
		return a.second > b.second; // 按 double 值降序排序
	});

	// 写入文件
	std::ofstream fout("/home/lrz/MGRR-2/node_score.txt");
	if (!fout.is_open()) {
		std::cerr << "无法打开输出文件 node_score.txt" << std::endl;
	} else {
		for (const auto& p : vec) {
			const Node& node = p.first;
			double score = p.second;
			fout << "(" <<node.first << "," << node.second << ") " << score << "\n";
		}
		fout.close();
	}
	*/
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
		//创建一个 tinyGraph 类型的向量 detLayers，用于存储每次模拟的网络层。
		std::vector<Graph> sampled_multigraph;
		//调用 sampleMultiplex 函数生成一次新的样本网络，并将结果存储在 detLayers 中。
		sampleMultiplex(sampled_multigraph);
		//调用 forwardProp 函数，以 detLayers 和 S 为输入，计算此次模拟中的影响力扩散结果。
		est += static_cast< double >( forwardProp(sampled_multigraph, vecSeed) );

		if(i%100==0){
			std::cout<<"MC iter i="<< i <<std::endl;
		}
	}

	est /= nIter;//将累加的总影响力 est 除以 nIter，得到平均影响力扩散值

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
		Graph& origin_graph = _Layers[layer_id]; //不修改就用&
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
						//这里就需要按照出边的形式记录了
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

		Graph& origin_graph = _Layers[layer_id]; //不修改就用&
		for(auto i = 0; i < origin_graph._Capacity; i++){
			if(origin_graph._hasNode[i] == false) continue;
			for (Edge& inEdge : origin_graph._OverlapGraph[i]){
				const Node nbrNode = inEdge.first;
				const float edgeWeight = inEdge.second;
				const auto randDouble = dsfmt_gv_genrand_open_close();
				if(randDouble < edgeWeight){
					//这里就需要按照出边的形式记录了
					sampled_multigraph[nbrNode.first]._OverlapGraph[nbrNode.second].push_back(Edge(Node(layer_id,i),1));
				}
			}
		}
	}
	//弄完之后加入hasnode检查
}

uint32_t Multiplex::forwardProp(std::vector<Graph> sampled_multigraph, const Nodelist& vecSeed)
{
	uint32_t total_activate = 0;
	//Q 用于进行广度优先搜索（BFS），里面是预激活的节点。已经满足激活条件，但是还没被正式激活
	std::queue<Node> Q;

	for (auto seed : vecSeed){
		Q.push(seed);
		sampled_multigraph[seed.first]._vecIsSeed[seed.second] = true; //意思是标记为已经激活
	}

	while ((!Q.empty()))
	{
		Node curr = Q.front();//<layer,node>
		Q.pop();
		total_activate++;

		//层间边激活
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
	size_t numRRsets = 10000; //一开始要生成的RR set个数设为总节点个数
	size_t iterator = 1;  //迭代次数
	double infOrigin = 0.0; 
	double infVldt = 0.0;
	const double e = exp(1);//e
	Timer timerMGRR("MGRR");//初始化一个计时器

	build_n_RRsets(numRRsets, mode);//选择集和验证集都生成numRRsets个

	//定义存储节点影响力的数据结构
	std::map<Node, Inf> nodeInfMap;

	//计算每个节点的影响力
	std::vector<uint32_t> coveredRRSet(_nLayers, 0);//分层方法中，每一层上覆盖的RR set个数
	std::vector<bool> edgeMark(_numRRsets, false);//初始化布尔数组 edgeMark 用于标记边是否已被处理过

	//初始化每个节点的影响力
	for (auto i = 0; i < _nLayers; i++)
	{
		if(_Layers[i]._Capacity == 0) continue;

		for(auto j = 0; j < _Layers[i]._Capacity; j++){
			if(_Layers[i]._hasNode[j] == false) continue;

			std::fill(coveredRRSet.begin(), coveredRRSet.end(), 0);//计算开始前设置为0

			for (auto edgeIdx : _Layers[i]._FRsets[j]){
				coveredRRSet[_RRsets[edgeIdx].sourceLayer]++;
			}

			//无偏估计，计算影响力
			double nodeInf = 0.0;
			for (size_t t = 0; t < _nLayers; t++)
			{
				if(_Layers[t]._Capacity == 0) continue;
				if( _RRSub[t] == 0) continue; //若该层就没有被采样到

				nodeInf += static_cast<double>(coveredRRSet[t]) 
				/ _RRSub[t] * _Layers[t]._Node_number;
			}
			nodeInfMap[Node(i,j)] = nodeInf;		
		}
	}
    std::vector<std::pair<Node, double>> vec(nodeInfMap.begin(), nodeInfMap.end());
	// 使用 Lambda 表达式对 vector 进行降序排序
	std::sort(vec.begin(), vec.end(), [](const std::pair<Node, double>& a, const std::pair<Node, double>& b) {
		return a.second > b.second; // 按 double 值降序排序
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
		{//说明这个点要么不存在，要么没有一条入边
			node_id = dsfmt_gv_genrand_uint32_range(_Layers[layer_id]._Capacity);//再次进行随机
		}

		_vecSeed.push_back(Node(layer_id, node_id));
	}

	uint32_t RRset_num = 4000;
	for (size_t i = 0; i < RRset_num; i++)
	{
		//生成随机的起点
		LayerID layer_id = selectRandomLayer(_Layers, _nNodesAllLayers);

		NodeID node_id = dsfmt_gv_genrand_uint32_range(_Layers[layer_id]._Capacity);
		while (_Layers[layer_id]._GraphContent[node_id].size() == 0 and _Layers[layer_id]._OverlapGraph[node_id].size() == 0) 
		{//说明这个点要么不存在，要么没有一条入边
			node_id = dsfmt_gv_genrand_uint32_range(_Layers[layer_id]._Capacity);//再次进行随机
		}
		
		build_one_RRset(Node(layer_id,node_id), i, true, "M");
	}
	_numRRsetsVldt = 4000;
	double infVldt = self_inf_cal_stratifie();
	//double infMC = monteCarloInfluence(_vecSeed);
	double total_time = timerMGRR.get_total_time();
	_tRes.set_running_time(total_time);//保存时间

	_tRes.set_influence(infVldt);//保存影响力结果
	_tRes.set_influence_original(0);//保存原始影响力
	_tRes.set_influence_MC(0);//保存MC影响力
	_tRes.set_seed_vec(_vecSeed);//保存种子集
	_tRes.set_RR_sets_size(4000);//保存创建RR集合的大小
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
			{//说明这个点要么不存在，要么没有一条入边
				node_id = dsfmt_gv_genrand_uint32_range(_Layers[layer_id]._Capacity);//再次进行随机
			}
			
			build_one_RRset(Node(layer_id,node_id), i, true, "M");
	}
	_numRRsetsVldt = 4000;
	double infVldt = self_inf_cal_stratifie();
	//double infMC = monteCarloInfluence(_vecSeed);
	double total_time = timerMGRR.get_total_time();
	_tRes.set_running_time(total_time);//保存时间
	
	_tRes.set_influence(infVldt);//保存影响力结果
	_tRes.set_influence_original(0);//保存原始影响力
	_tRes.set_influence_MC(0);//保存MC影响力
	_tRes.set_seed_vec(_vecSeed);//保存种子集
	_tRes.set_RR_sets_size(4000);//保存创建RR集合的大小
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
    char comma;  // 用于跳过逗号

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
		//生成随机的起点
		LayerID layer_id = dsfmt_gv_genrand_uint32_range(_nLayers);
		while(_Layers[layer_id]._Capacity == 0){
			layer_id = dsfmt_gv_genrand_uint32_range(_nLayers);
		}

		NodeID node_id = dsfmt_gv_genrand_uint32_range(_Layers[layer_id]._Capacity);//需要验证集则说明不是训练图，则_Capacity和_Node_number是一样的
		while (_Layers[layer_id]._GraphContent[node_id].size() == 0 and _Layers[layer_id]._OverlapGraph[node_id].size() == 0) 
		{//说明这个点要么不存在，要么没有一条入边
			node_id = dsfmt_gv_genrand_uint32_range(_Layers[layer_id]._Capacity);//再次进行随机
		}
		
		build_one_RRset(Node(layer_id,node_id), i, true, "M");
	}
	_numRRsetsVldt = 10000;
	double infVldt = self_inf_cal_stratifie();

	_tRes.set_influence(infVldt);//保存影响力结果
	_tRes.set_influence_original(0);//保存原始影响力
	_tRes.set_influence_MC(0);//保存MC影响力
	_tRes.set_seed_vec(_vecSeed);//保存种子集
	_tRes.set_RR_sets_size(10000);//保存创建RR集合的大小
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
    char comma;  // 用于跳过逗号

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
		{//说明这个点要么不存在，要么没有一条入边
			node_id = dsfmt_gv_genrand_uint32_range(_Layers[layer_id]._Capacity);//再次进行随机
		}
		
		build_one_RRset(Node(layer_id,node_id), i, true, "M");
	}
	_numRRsetsVldt = 10000;
	double infVldt = self_inf_cal_stratifie();

	_tRes.set_influence(infVldt);//保存影响力结果
	_tRes.set_influence_original(0);//保存原始影响力
	_tRes.set_influence_MC(0);//保存MC影响力
	_tRes.set_seed_vec(_vecSeed);//保存种子集
	_tRes.set_RR_sets_size(10000);//保存创建RR集合的大小
	_tRes.set_round(0);
}
