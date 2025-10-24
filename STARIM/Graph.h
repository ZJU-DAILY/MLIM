#pragma once

#include "commonFunc.h"
/// @brief 存储单层图结构
///以类的形式定义图结构，可以比struct结构更方便初始化
///将图结构全部设置成为public，是为了方便其他类访问
class Graph
{

public:
    /// 本层的层编号
    LayerID _layer_id;

    ///图中节点的个数
	uint32_t _Node_number;

    ///图中边的个数，不包括层间的边
	uint32_t _Edge_number;

    ///图中存储信息的数据结构的容量
	uint32_t _Capacity;

    ///以入边集合的形式存储图
    std::vector<Edgelist> _GraphContent;

    /// 用于存储层间重叠节点
    std::vector<Edgelist> _OverlapGraph;

    /// 节点是否存在
    std::vector<bool> _hasNode;

    /// FRsets是节点可以覆盖的RR set的集合，用于选择种子集 
    FRsets _FRsets;

    /// 用于验证生成的种子集的影响力
    FRsets _FRsetsVldt;

    /// 用于标记图中节点是否访问过
    std::vector<bool> _vecVisitBoolPerGraph;

    /// 用于标记哪些节点属于种子集
    std::vector<bool> _vecIsSeed; //MC模拟的时候会用到

    std::vector<double> Threshold;

    std::vector<double> Cumulative_weight;

    Graph() = default;
    
	Graph(const LayerID layer_id, const uint32_t Node_number, const uint32_t Edge_number, const uint32_t nodes_capacity)
    {
        //前三个变量有重名，则使用this->，后面的变量没有重名，方便起见不适用this->
        this->_layer_id = layer_id; //useless?
		this->_Node_number = Node_number; //检查一下_Node_number、_Edge_number是否有用？
		this->_Edge_number = Edge_number;
        _Capacity = nodes_capacity;

		_GraphContent.resize(_Capacity);
        for (size_t i = 0; i < _Capacity; i++)
        {//预分配空间 50是一个经验数字，可以修改不唯一。一个节点的邻居，也就是度，一般在10-30，所以分配50一般是够的
            _GraphContent[i].reserve(200);
        }

        _hasNode.resize(_Capacity);//默认为false
	}

	~Graph(){
		
	}

    void setVariable(bool effic_inf_valid_algo = false){        
        _OverlapGraph.resize(_Capacity);
        for (size_t i = 0; i < _Capacity; i++)
        {//预分配空间 10是一个经验数字，可以修改不唯一，我目前认为这是与层数挂钩的，即一层最多有一个共同用户。我没有考虑一层上有多个用户同时和别的用户关联的情况
            _OverlapGraph[i].reserve(10);//resize强于reserve
        }

        _FRsets = FRsets(_Capacity);
        _FRsetsVldt = FRsets(_Capacity);

        _vecVisitBoolPerGraph.resize(_Capacity);//在 std::vector<bool> 中，capacity() 显示的是位（bit）数，而 底层内存分配是按字节（byte）对齐的。
        
        if (effic_inf_valid_algo) _vecIsSeed.resize(_Capacity);//std::vector<bool> 被 压缩存储，每 8 个 bool 占用 1 字节。100需要至少 13 字节，内存对齐后16字节 = 128 bit
        //_vecIsSeed.resize(_Capacity); 
    }

    void setThreshold(){
        Threshold.resize(_Capacity);
        Cumulative_weight.resize(_Capacity);
        for (size_t i = 0; i < _Capacity; i++)
        {
            if(_hasNode[i] == false){
                Threshold[i] = 100;
                Cumulative_weight[i] = -1; 
                continue;
            }
            auto randDouble = dsfmt_gv_genrand_open_close();
            Threshold[i] = randDouble;
            Cumulative_weight[i] = 0;    
        }
    }

    void reserve_singleGraph_FRsets(){
        for(size_t i = 0; i < _Capacity; i++){
            if(_hasNode[i] == false) continue;
            _FRsets[i].reserve(5000);//初始化后reserve500和5000耗时几乎一样，但是内存预留不一样
        }
        for(size_t i = 0; i < _Capacity; i++){
            if(_hasNode[i] == false) continue;
            _FRsetsVldt[i].reserve(5000);
        }
    }
    
    void clean_visitedMark(){
        for(int i = 0; i < _Capacity; i++){
            _vecVisitBoolPerGraph[i] = false;
        }
    }

    void clean_seedMark(){
        for(int i = 0; i < _Capacity; i++){
            _vecIsSeed[i] = false;
        }
    }

    void clean_FRset(){
        for(int i = 0; i < _Capacity; i++){
            _FRsets[i].clear();
        }
    }
	
};