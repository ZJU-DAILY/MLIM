#pragma once

#include "commonFunc.h"

class Graph
{

public:
    LayerID _layer_id;

	uint32_t _Node_number;

	uint32_t _Edge_number;

	uint32_t _Capacity;

    std::vector<Edgelist> _GraphContent;

    std::vector<Edgelist> _OverlapGraph;

    std::vector<bool> _hasNode;

    FRsets _FRsets;

    FRsets _FRsetsVldt;

    std::vector<bool> _vecVisitBoolPerGraph;

    std::vector<bool> _vecIsSeed; 

    std::vector<double> Threshold;

    std::vector<double> Cumulative_weight;

    Graph() = default;
    
	Graph(const LayerID layer_id, const uint32_t Node_number, const uint32_t Edge_number, const uint32_t nodes_capacity)
    {
        this->_layer_id = layer_id; 
		this->_Node_number = Node_number; 
		this->_Edge_number = Edge_number;
        _Capacity = nodes_capacity;

		_GraphContent.resize(_Capacity);
        for (size_t i = 0; i < _Capacity; i++)
        {
            _GraphContent[i].reserve(200);
        }

        _hasNode.resize(_Capacity);
	}

	~Graph(){
		
	}

    void setVariable(bool effic_inf_valid_algo = false){        
        _OverlapGraph.resize(_Capacity);
        for (size_t i = 0; i < _Capacity; i++)
        {
            _OverlapGraph[i].reserve(10);
        }

        _FRsets = FRsets(_Capacity);
        _FRsetsVldt = FRsets(_Capacity);

        _vecVisitBoolPerGraph.resize(_Capacity);
        
        if (effic_inf_valid_algo) _vecIsSeed.resize(_Capacity);
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
            _FRsets[i].reserve(5000);
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