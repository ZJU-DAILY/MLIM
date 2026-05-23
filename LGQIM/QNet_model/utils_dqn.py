from collections import defaultdict

import torch
import pickle
import os
import re
import sys
import subprocess
import random
import networkx as nx
import numpy as np
from typing import Union, List, Tuple

from scipy.sparse import csr_matrix

from q_network import QNetwork







def read_data_from_file(file_path):
    data = {}
    with open(file_path, 'r') as f:
        for line in f:
            match = re.match(r"\((\d+),(\d+)\)\s+([0-9\.e\+-]+)", line)
            if match:
                layer = int(match.group(1))
                node = int(match.group(2))
                score = float(match.group(3))
                data[(layer, node)] = score
    return data

def load_graph_old(base_dir):
  
    with open(os.path.join(base_dir, "total_layers.txt"), 'r') as f:
        total_layers = int(f.read().strip())


    layer0_file = os.path.join(base_dir, "layer0.txt")
    if os.path.exists(layer0_file):
        layer_start = 0
    else:
        layer_start = 1


    G = nx.DiGraph()
   
    for i in range(layer_start, layer_start + total_layers):

        layer_file = os.path.join(base_dir, f"layer{i}.txt")
        with open(layer_file, 'r') as f:
            lines = f.readlines()
        
            node_count, edge_count = map(int, lines[0].strip().split())
            for line in lines[1:]:
                src, dst, weight = line.strip().split()
                src = (i, int(src))
                dst = (i, int(dst))
                weight = float(weight)
                G.add_edge(src, dst, weight=weight, layer=f"layer{i}")


        ov_file = os.path.join(base_dir, f"layer{i}ov.txt")
        with open(ov_file, 'r') as f:
            for line in f:
                cur_node_id, other_layer_id, other_node_id, weight = line.strip().split()
                src = (int(other_layer_id), int(other_node_id))
                dst = (i, int(cur_node_id))
                weight = float(weight)
                G.add_edge(src, dst, weight=weight, layer=f"layer{i}overlap")
    return G


class GraphWrapper:


    def __init__(self, adj_matrix, node_to_idx, idx_to_node, edge_attrs=None):
        self.adj_matrix = adj_matrix
        self.adj_matrix_csc = adj_matrix.tocsc()  
        self.node_to_idx = node_to_idx
        self.idx_to_node = idx_to_node
        self.n_nodes = len(node_to_idx)
        self.edge_attrs = edge_attrs  
        self._neighbor_cache = {}
        self._predecessor_cache = {}
        self._precompute_all()

    def _precompute_all(self):

   
        for idx in range(self.n_nodes):
            row_start = self.adj_matrix.indptr[idx]
            row_end = self.adj_matrix.indptr[idx + 1]
            neighbor_indices = self.adj_matrix.indices[row_start:row_end]
            node = self.idx_to_node[idx]
            self._neighbor_cache[node] = [self.idx_to_node[i] for i in neighbor_indices]

        for idx in range(self.n_nodes):
            col_start = self.adj_matrix_csc.indptr[idx]
            col_end = self.adj_matrix_csc.indptr[idx + 1]
            predecessor_indices = self.adj_matrix_csc.indices[col_start:col_end]
            node = self.idx_to_node[idx]
            self._predecessor_cache[node] = [self.idx_to_node[i] for i in predecessor_indices]

    def nodes(self):

        return iter(self.node_to_idx.keys())

    def neighbors(self, node):

        return iter(self._neighbor_cache.get(node, []))

    def predecessors(self, node):

        return iter(self._predecessor_cache.get(node, []))

    def successors(self, node):

        return self.neighbors(node)

    def has_node(self, node):

        return node in self.node_to_idx

    def has_edge(self, src, dst):

        if src not in self.node_to_idx or dst not in self.node_to_idx:
            return False
        src_idx = self.node_to_idx[src]
        dst_idx = self.node_to_idx[dst]
        return self.adj_matrix[src_idx, dst_idx] != 0

    def get_edge_data(self, src, dst, default=None):

        if src not in self.node_to_idx or dst not in self.node_to_idx:
            return default
        src_idx = self.node_to_idx[src]
        dst_idx = self.node_to_idx[dst]
        weight = self.adj_matrix[src_idx, dst_idx]
        if weight == 0:
            return default

        result = {'weight': weight}
   
        if self.edge_attrs and (src, dst) in self.edge_attrs:
            result.update(self.edge_attrs[(src, dst)])
        return result

    def edges(self, data=False):

        rows, cols = self.adj_matrix.nonzero()
        if data:
            for i, j in zip(rows, cols):
                src = self.idx_to_node[i]
                dst = self.idx_to_node[j]
                weight = self.adj_matrix[i, j]
                edge_data = {'weight': weight}
                if self.edge_attrs and (src, dst) in self.edge_attrs:
                    edge_data.update(self.edge_attrs[(src, dst)])
                yield (src, dst, edge_data)
        else:
            for i, j in zip(rows, cols):
                src = self.idx_to_node[i]
                dst = self.idx_to_node[j]
                yield (src, dst)

    def number_of_nodes(self):

        return self.n_nodes

    def number_of_edges(self):

        return self.adj_matrix.nnz

    def __len__(self):

        return self.n_nodes

    def __contains__(self, node):

        return node in self.node_to_idx

    def __getitem__(self, node):

        if node not in self.node_to_idx:
            raise KeyError(node)
        result = {}
        idx = self.node_to_idx[node]
        row_start = self.adj_matrix.indptr[idx]
        row_end = self.adj_matrix.indptr[idx + 1]
        neighbor_indices = self.adj_matrix.indices[row_start:row_end]
        weights = self.adj_matrix.data[row_start:row_end]
        for neighbor_idx, weight in zip(neighbor_indices, weights):
            neighbor = self.idx_to_node[neighbor_idx]
            edge_data = {'weight': weight}
            if self.edge_attrs and (node, neighbor) in self.edge_attrs:
                edge_data.update(self.edge_attrs[(node, neighbor)])
            result[neighbor] = edge_data
        return result

    def in_degree(self, node=None):

        if node is None:
            in_degrees = np.diff(self.adj_matrix_csc.indptr)
            return {self.idx_to_node[i]: int(deg) for i, deg in enumerate(in_degrees)}
        else:
            if node not in self.node_to_idx:
                return 0
            return len(self._predecessor_cache.get(node, []))

    def out_degree(self, node=None):

        if node is None:
            out_degrees = np.diff(self.adj_matrix.indptr)
            return {self.idx_to_node[i]: int(deg) for i, deg in enumerate(out_degrees)}
        else:
            if node not in self.node_to_idx:
                return 0
            return len(self._neighbor_cache.get(node, []))


def load_graph(base_dir):
    import gc

   
    with open(os.path.join(base_dir, "total_layers.txt"), 'r') as f:
        total_layers = int(f.read().strip())

   
    layer0_file = os.path.join(base_dir, "layer0.txt")
    layer_start = 0 if os.path.exists(layer0_file) else 1

  
    node_to_idx = {}
    idx_counter = 0


    edges_row = []
    edges_col = []
    edges_data = []

    for i in range(layer_start, layer_start + total_layers):

     
        layer_file = os.path.join(base_dir, f"layer{i}.txt")
        with open(layer_file, 'r') as f:
            next(f) 
            for line in f:
                parts = line.strip().split()
                if len(parts) != 3:
                    continue

                src, dst, weight = parts
                src_node = (i, int(src))
                dst_node = (i, int(dst))


                if src_node not in node_to_idx:
                    node_to_idx[src_node] = idx_counter
                    idx_counter += 1
                if dst_node not in node_to_idx:
                    node_to_idx[dst_node] = idx_counter
                    idx_counter += 1


                edges_row.append(node_to_idx[src_node])
                edges_col.append(node_to_idx[dst_node])
                edges_data.append(float(weight))

      
        ov_file = os.path.join(base_dir, f"layer{i}ov.txt")
        if os.path.exists(ov_file):
            with open(ov_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 4:
                        continue

                    cur_node_id, other_layer_id, other_node_id, weight = parts
                    src_node = (int(other_layer_id), int(other_node_id))
                    dst_node = (i, int(cur_node_id))

            
                    if src_node not in node_to_idx:
                        node_to_idx[src_node] = idx_counter
                        idx_counter += 1
                    if dst_node not in node_to_idx:
                        node_to_idx[dst_node] = idx_counter
                        idx_counter += 1

                    edges_row.append(node_to_idx[src_node])
                    edges_col.append(node_to_idx[dst_node])
                    edges_data.append(float(weight))

        if i % 10 == 0:
            gc.collect()

    print(f"num of nodes: {len(node_to_idx)}")
    print(f"num of edges: {len(edges_data)}")

  
    n = len(node_to_idx)

 
    row = np.array(edges_row, dtype=np.int32)
    col = np.array(edges_col, dtype=np.int32)
    data = np.array(edges_data, dtype=np.float32)

  
    del edges_row, edges_col, edges_data
    gc.collect()

    adj_matrix = csr_matrix((data, (row, col)), shape=(n, n))

  
    del row, col, data
    gc.collect()


    idx_to_node = {v: k for k, v in node_to_idx.items()}

    G = GraphWrapper(adj_matrix, node_to_idx, idx_to_node)

    return G



def construct_nodeid2idx(node_list):
    nodeid2idx = {}
    idx2node = {}
    idx = 0
    for node in node_list:
        if node not in nodeid2idx:
            nodeid2idx[node] = idx
            idx2node[idx] = node
            idx += 1
    return nodeid2idx, idx2node

def normalize_tensor(tensor: torch.Tensor, dim: Union[int, List[int], Tuple[int]] = None) -> torch.Tensor:
    
    tensor_clone = tensor.clone()
    
    if dim is None:
        dims = range(tensor.shape[1])
    elif isinstance(dim, int):
        dims = [dim]
    else:
        dims = dim

    for d in dims:
        col = tensor[:, d]
        mean = col.mean()
        std = col.std(unbiased=False)
        tensor_clone[:, d] = (col - mean) / (std + 1e-8)

    return tensor_clone

def get_graph(graph_dir, pred_dir, use_gts, ratio, cfg):
   
    print('load graph begin')
    G = load_graph(graph_dir)
    print('load graph end')

  
    groundtruth = use_gts
    if groundtruth:
        print('use ground truth scores')
        node_scores_file_path = graph_dir + "/node_score.txt"  # ground truth
        node_score_dict = read_data_from_file(node_scores_file_path)
       
        total_nodes = len(node_score_dict)
        
        top_k = max(1, int(total_nodes * ratio))
       
        sorted_nodes = sorted(node_score_dict.items(), key=lambda item: item[1], reverse=True)
       
        top_ten_nodes = [node for node, score in sorted_nodes[:top_k]]

    else:
        print('use predicted scores')
        node_scores_file_path = pred_dir
        node_score_dict = read_data_from_file(node_scores_file_path)
        top_ten_nodes = list(node_score_dict.keys())



    
    node_neighbor_dict = {}
    for node in G.nodes():
        node_neighbor_dict[node] = list(G.neighbors(node))





  
    goodNode2idx, idx2goodNode = construct_nodeid2idx(top_ten_nodes)
    num_nodes = len(goodNode2idx)
   
    embedding_tensor = torch.zeros((num_nodes, cfg.EMBEDDING_DIM), dtype=torch.float32)

   
    for node_id, idx in goodNode2idx.items():
        num_neighbors = len(node_neighbor_dict[node_id])
        if node_id in node_score_dict:
            score = node_score_dict[node_id]
        else:
            raise KeyError(f"error")
        embedding_tensor[idx] = torch.tensor([num_neighbors, score, 0], dtype=torch.float32)
    embedding_archive = embedding_tensor.to(cfg.DEVICE).clone()

  
    index = list(range(cfg.EMBEDDING_DIM - 1))  # [0, 1, ..., n-2]
    embedding_tensor = normalize_tensor(embedding_archive, index) 

    return G, top_ten_nodes, goodNode2idx, embedding_archive, embedding_tensor, node_neighbor_dict, node_score_dict      

def get_best_node(graph, stateIdx, model):   
    mu_s = []
    mu_l = []
    mu_v = []
    vertices = []
    mu_s_single, mu_l_single = graph.get_environment_embedding(stateIdx) 
    
    for nd in graph.top_tenpct_nodes:
        if graph.is_selected[nd] == -1 or graph.is_selected[nd] >= stateIdx:
       
            mu_v_single = graph.embedding_time[stateIdx][graph.goodNode2idx[nd]]
            mu_s.append(mu_s_single)
            mu_l.append(mu_l_single)
            mu_v.append(mu_v_single)
            vertices.append(nd)

    if not vertices:
        return (-1, -1) 

  
    mu_s_tensor = torch.stack(mu_s)
    mu_l_tensor = torch.stack(mu_l)
    mu_v_tensor = torch.stack(mu_v)

    model.eval()
    with torch.no_grad():
        predictions = model(mu_v_tensor, mu_s_tensor, mu_l_tensor) 

    best_index = torch.argmax(predictions).item()
    bestNode = vertices[best_index]

    return bestNode

def get_short_reward(seed_nodes, previous_spread, proc):
    spread  = calculate_spread(seed_nodes, proc)
    shortReward = spread - previous_spread
    return shortReward, spread


def calculate_spread_old(seedset, proc):
    seed_str = " ".join([f"{layer},{node}" for layer, node in seedset]) + "\n"
    proc.stdin.write(seed_str)
    proc.stdin.flush()

    result = proc.stdout.readline()
    return float(result.strip())
    
def calculate_spread(seedset, proc):
    seed_str = " ".join([f"{layer},{node}" for layer, node in seedset]) + "\n"
    proc.stdin.write(seed_str)
    proc.stdin.flush()  
    import select
    if 1: #select.select([proc.stdout], [], [], 30.0)[0]:
        result = proc.stdout.readline()
        return float(result.strip())
    else:
        print("Warning: rr_server timeout")
        return 0.0






def find_overlap_targets(G, action):

    overlap_targets = []
    for _, target, edge_data in G.out_edges(action, data=True):
        if "overlap" in edge_data.get("layer", ""):
            overlap_targets.append(target)

    if overlap_targets:
        return True, overlap_targets
    else:
        return False, []

