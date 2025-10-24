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

from q_network import QNetwork

#get_graph和createMu还在用

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

def load_graph(base_dir):
    # 读取总层数
    with open(os.path.join(base_dir, "total_layers.txt"), 'r') as f:
        total_layers = int(f.read().strip())

    # 尝试判断layer编号起始（0或1），通过检查文件是否存在
    layer0_file = os.path.join(base_dir, "layer0.txt")
    if os.path.exists(layer0_file):
        layer_start = 0
    else:
        layer_start = 1


    G = nx.DiGraph()
    # ========== 遍历每一层 ==========
    for i in range(layer_start, layer_start + total_layers):
        # ---------- 1. 读取 layer{i}.txt ----------
        layer_file = os.path.join(base_dir, f"layer{i}.txt")
        with open(layer_file, 'r') as f:
            lines = f.readlines()
            # 读取节点数和边数（可选，用于验证）
            node_count, edge_count = map(int, lines[0].strip().split())
            for line in lines[1:]:
                src, dst, weight = line.strip().split()
                src = (i, int(src))
                dst = (i, int(dst))
                weight = float(weight)
                G.add_edge(src, dst, weight=weight, layer=f"layer{i}")

    # ---------- 2. 读取 layer{i}ov.txt ----------
        ov_file = os.path.join(base_dir, f"layer{i}ov.txt")
        with open(ov_file, 'r') as f:
            for line in f:
                cur_node_id, other_layer_id, other_node_id, weight = line.strip().split()
                src = (int(other_layer_id), int(other_node_id))
                dst = (i, int(cur_node_id))
                weight = float(weight)
                G.add_edge(src, dst, weight=weight, layer=f"layer{i}overlap")
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
    """
    标准化输入张量的指定列（或所有列），返回一个新张量。

    参数:
        tensor (torch.Tensor): 输入张量，形状为 (num_rows, num_features)
        dim (int, list, tuple, or None): 指定归一化的列索引，None 表示所有列

    返回:
        torch.Tensor: 标准化后的新张量
    """
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

def get_graph(graph_dir, cfg):
    # 加载图
    G = load_graph(graph_dir)

    # 加载节点预测得分 现在读入的是node_score.txt  之后需要读入predict_score.txt
    node_scores_file_path = graph_dir + "/node_score.txt"#现在的
    node_score_dict = read_data_from_file(node_scores_file_path)


    # 取 top 10% 节点
    # 1. 获取所有节点的数量
    total_nodes = len(node_score_dict)
    # 2. 计算前10%的数量，至少取一个
    top_k = max(1, int(total_nodes * 0.10))
    # 3. 按照score从高到低排序
    sorted_nodes = sorted(node_score_dict.items(), key=lambda item: item[1], reverse=True)
    # 4. 提取前10%的节点（只取键，即 (layer, node)）
    top_ten_nodes = [node for node, score in sorted_nodes[:top_k]]


    # 加载节点采样邻居信息 - 这里是不采样的节点信息
    node_neighbor_dict = {}
    for node in G.nodes():
        node_neighbor_dict[node] = list(G.neighbors(node))



    # 处理 embedding
    goodNode2idx, idx2goodNode = construct_nodeid2idx(top_ten_nodes)
    num_nodes = len(goodNode2idx)
    # 初始化张量
    embedding_tensor = torch.zeros((num_nodes, cfg.EMBEDDING_DIM), dtype=torch.float32)

    # 构造张量
    for node_id, idx in goodNode2idx.items():
        num_neighbors = len(node_neighbor_dict[node_id])
        if node_id in node_score_dict:
            score = node_score_dict[node_id]
        else:
            raise KeyError(f"错误：节点 {node_id} 不在 dict_sup_gs_scores 中，无法获取分数！")
        embedding_tensor[idx] = torch.tensor([num_neighbors, score, 0], dtype=torch.float32)
    embedding_archive = embedding_tensor.to(cfg.DEVICE).clone()

    """with open("qnet_features.txt", "w") as f:
            # 将 Tensor 转为列表后逐行写入
            for row in embedding_tensor.cpu().tolist():
                f.write(" ".join([f"{x:.4f}" for x in row]) + "\n") # 用空格分隔元素"""
    
    # 归一化 embedding
    index = list(range(cfg.EMBEDDING_DIM - 1))  # [0, 1, ..., n-2]
    embedding_tensor = normalize_tensor(embedding_archive, index) #或者第一列minmax，第二列log+minmax

    return G, top_ten_nodes, goodNode2idx, embedding_archive, embedding_tensor, node_neighbor_dict, node_score_dict      

def get_best_node(graph, stateIdx, model):   
    mu_s = []
    mu_l = []
    mu_v = []
    vertices = []
    mu_s_single, mu_l_single = graph.get_environment_embedding(stateIdx) # 在GPU上

    # 所有的action_t : 即 top_tenpct_nodes 选择未被选中的节点
    for nd in graph.top_tenpct_nodes:
        if graph.is_selected[nd] == -1 or graph.is_selected[nd] >= stateIdx:
            #若选择nd为下一次的action,其向量表示
            mu_v_single = graph.embedding_time[stateIdx][graph.goodNode2idx[nd]] # 在GPU上
            mu_s.append(mu_s_single)
            mu_l.append(mu_l_single)
            mu_v.append(mu_v_single)
            vertices.append(nd)

    if not vertices:
        return (-1, -1)  # 如果没有候选节点，直接返回

    # 将列表堆叠为 Tensor（仍然在 GPU 上）
    mu_s_tensor = torch.stack(mu_s)
    mu_l_tensor = torch.stack(mu_l)
    mu_v_tensor = torch.stack(mu_v)

    model.eval()
    with torch.no_grad():
        predictions = model(mu_v_tensor, mu_s_tensor, mu_l_tensor)  # 计算 Q 值

    best_index = torch.argmax(predictions).item()
    bestNode = vertices[best_index]

    return bestNode

def get_short_reward(seed_nodes, previous_spread, proc):
    spread  = calculate_spread(seed_nodes, proc)
    shortReward = spread - previous_spread
    return shortReward, spread

def calculate_spread(seedset, proc):
    seed_str = " ".join([f"{layer},{node}" for layer, node in seedset]) + "\n"
    proc.stdin.write(seed_str)
    proc.stdin.flush()

    result = proc.stdout.readline()
    return float(result.strip())
    



def load_best_model():
    if os.path.exists("saved_model/best_model.pth"):
        input_dim = 2
        q_network = QNetwork(input_dim, 1)  
        checkpoint = torch.load("saved_model/best_model.pth")
        q_network.load_state_dict(checkpoint["model_state_dict"])
    else:
        input_dim = 2
        q_network = QNetwork(input_dim, 1)  
        checkpoint = torch.load("saved_model/newest_model.pth")
        q_network.load_state_dict(checkpoint["model_state_dict"])
    return q_network

def load_newest_model():
    if not os.path.exists("saved_model/newest_model.pth"):
        return None #表示目前没有可以使用的模型
    else:
        input_dim = 2
        q_network = QNetwork(input_dim, 1)  
        checkpoint = torch.load("saved_model/newest_model.pth")
        q_network.load_state_dict(checkpoint["model_state_dict"])
    return q_network

def save_tensor_info(tensor, filename):
    with open(filename, "w") as f:
        f.write(f"Shape: {tensor.shape}\n")
        f.write(f"Type: {tensor.dtype}\n")
        f.write("Values:\n")
        np.savetxt(f, tensor.numpy(), fmt="%.6f")  # 保存浮点数，保留6位小数


def find_overlap_targets(G, action):
    """
    查找从 action 出发的所有层间边（overlap edge）指向的终点节点。

    参数：
    - G: networkx.DiGraph 对象
    - action: 起点节点 (layer_id, node_id)

    返回：
    - (True, [终点节点, ...]) 或 (False, [])
    """
    overlap_targets = []
    for _, target, edge_data in G.out_edges(action, data=True):
        if "overlap" in edge_data.get("layer", ""):
            overlap_targets.append(target)

    if overlap_targets:
        return True, overlap_targets
    else:
        return False, []

