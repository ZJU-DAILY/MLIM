#pragma once

class GraphBase
{
public:
	
	static int read_max_nodeID(const std::string &dir) {
		std::string filename = dir + "/max_nodeID.txt";
		std::ifstream infile(filename);

		if (!infile.is_open()) {
			std::cerr << "Error: Cannot open file " << filename << std::endl;
			exit(EXIT_FAILURE);  
		}

		int max_nodeID;
		if (!(infile >> max_nodeID)) {
			std::cerr << "Error: Failed to read integer from " << filename << std::endl;
			exit(EXIT_FAILURE);
		}

		infile.close();
		return max_nodeID;
	}

	static int read_total_layers(const std::string &dir) {
		std::string filename = dir + "/total_layers.txt";
		std::ifstream infile(filename);

		if (!infile.is_open()) {
			std::cerr << "Error: Cannot open file " << filename << std::endl;
			exit(EXIT_FAILURE);  
		}

		int total_nodes;
		if (!(infile >> total_nodes)) {
			std::cerr << "Error: Failed to read integer from " << filename << std::endl;
			exit(EXIT_FAILURE);
		}

		infile.close();
		return total_nodes;
	}

	//这里读取的是有weight的图
	static Graph read_graph(const std::string filename, const LayerID layer_id, const uint32_t nodes_capacity)
	{
		size_t numV, numE;
		uint32_t srcId, dstId;
		float weight = 0.0;

		//打开指定的文件 filename 并检查是否成功打开
		std::ifstream infile(filename);
		if (!infile.is_open())
		{
			std::cout << "The file \"" + filename + "\" can NOT be opened\n";
			exit(EXIT_FAILURE);  
		}

		//从文件中读取图的节点数 numV 和边数 numE , 作用仅仅是检查
		infile >> numV >> numE;

		uint32_t capacity = nodes_capacity;
		Graph vecGRev = Graph(layer_id, numV, numE, nodes_capacity);

		size_t checkEdgeNumber = 0;
		std::set<int> uniqueNodes;
		while (infile >> srcId >> dstId >> weight) {
			// 如果有问题（如某行格式不对），会跳过这一行
			if (infile.fail()) {
				std::cout << "Warning: invalid data format at line: " << srcId << " " << dstId << " " << weight << std::endl;
				infile.clear();  // 重置流状态
				infile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');  // 忽略当前行
				continue;
			}

			vecGRev._GraphContent[dstId].push_back( Edge( Node(layer_id, srcId), weight ) );

			uniqueNodes.insert(srcId);
			uniqueNodes.insert(dstId);
			vecGRev._hasNode[srcId] = true;
			vecGRev._hasNode[dstId] = true;
			checkEdgeNumber++;
		}
		if(checkEdgeNumber!=numE){//检查边的数量
			std::cout<<"checkEdgeNumber!=numE, read graph wrong"<<std::endl;
			std::cout<<"read_graph error from"<<" filename:"<<filename<<std::endl;
			exit(EXIT_FAILURE);
		}

		if(uniqueNodes.size()!=numV){//检查节点数量
			std::cout<<"checkNodeNumber!=numV, read graph wrong"<<std::endl;
			std::cout<<"read_graph error from"<<" filename:"<<filename<<std::endl;
			exit(EXIT_FAILURE);
		}
		infile.close();

		vecGRev.setVariable();
		
		return vecGRev;
	}

	/*
    * Overlap file contains lines
    * i j k
    * where i is node_id mapped to layer j, node k
	* 层间传播全都是按照入边的形式存储的，，每对重叠节点都有两条方向相反的边，在设定概率的时候需要注意
    */
	static void read_overlapGraph(const std::string filename, Graph &singleGraph, Nodelist &crossLayerNode)
	{
		//本层节点ID ，对方节点ID
		NodeID node_id, overlap_node_id;
		//对方节点层编号
        LayerID overlap_layer_id;
		float weight = 0.0;

		//打开指定的文件 filename 并检查是否成功打开
		std::ifstream infile(filename);
		if (!infile.is_open())
		{
			std::cout << "The file \"" + filename + "\" can NOT be opened\n";
			exit(EXIT_FAILURE);  
		}

        while (infile >> node_id) {
            infile >> overlap_layer_id;
            infile >> overlap_node_id;
			infile >> weight;
			singleGraph._OverlapGraph[node_id].push_back( Edge( Node(overlap_layer_id, overlap_node_id), weight ) );
			singleGraph._hasNode[node_id] = true;
			crossLayerNode.push_back(Node(overlap_layer_id, overlap_node_id));
        }

        infile.close();
	}
};
