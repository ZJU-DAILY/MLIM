import os
import re
import sys
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from typing import Union, List, Tuple
import networkx as nx
#之后熟悉代码，把注释全部删除 仅仅留下函数功能注释

"""读取节点特征和得分"""
#需要检查数据结构
def load_feature_and_score(base_dir, nodeid2idx):
    xv_dict_path = os.path.join(base_dir, 'graph0_nodexv.txt')
    score_dict_path = os.path.join(base_dir, 'graph0_seedscore.txt')

    xv_dict = read_data_from_file(xv_dict_path)
    features = torch.stack([xv_dict[node] for node in nodeid2idx.keys()])#torch.Tensor
    scaler = StandardScaler()
    features_standardized = torch.tensor(scaler.fit_transform(features), dtype=torch.float32)

    score_dict = read_data_from_file(score_dict_path)
    score_groundtruth = torch.stack([score_dict[node] for node in nodeid2idx.keys()])
    return xv_dict, score_dict, features_standardized, score_groundtruth

""" 从文件读取数据并解析为字典形式
    文件形式为：(layer, node) score
    字典形式为：{(layer, node): tensor([score])}"""
def read_data_from_file(file_path):
    data = {}
    with open(file_path, 'r') as f:
        for line in f:
            match = re.match(r"\((\d+),(\d+)\)\s+([0-9\.e\+-]+)", line)
            if match:
                layer = int(match.group(1))
                node = int(match.group(2))
                score = float(match.group(3))
                data[(layer, node)] = torch.tensor([score])
    return data

'''选出xv排名前rbmax个节点,返回节点索引列表,之后应该要修改根据budget-rank进行选择'''
def select_top_nodes(xv_dict, rbmax, nodeid2idx):
    sorted_nodes = sorted(xv_dict.items(), key=lambda item: item[1].item(), reverse=True)
    selected_nodes = [node for idx, (node, _) in enumerate(sorted_nodes) if idx < rbmax]
    selected_nodes_idx = [nodeid2idx[node] for node in selected_nodes]
    return selected_nodes_idx

'''构造每个节点的邻居组成邻接矩阵。邻居个数为max_degree,如果邻居数量大于max_degree,则随机选取max_degree个邻居
    如果邻居数量小于 max_degree, 首先取前 len(neighbors) 个不重复的邻居, 然后使用有放回采样补充不足部分'''
def construct_adj(G, max_degree, nodeid2idx):
    num_nodes = G.number_of_nodes()
    # 用-1表示填充无邻居的位置
    adj = torch.full((num_nodes, max_degree), -1, dtype=torch.long)

    for nodeid in G.nodes():
        neighbors = [nodeid2idx[neighbor] for neighbor in G.neighbors(nodeid)]
        neighbors = torch.tensor(neighbors, dtype=torch.long)

        if len(neighbors) == 0:
            continue

        if len(neighbors) > max_degree:
            neighbors = neighbors[torch.randperm(len(neighbors))[:max_degree]]
        elif len(neighbors) < max_degree:
            remaining_count = max_degree - len(neighbors)
            sampled_neighbors = neighbors[
                torch.multinomial(torch.ones(len(neighbors)), remaining_count, replacement=True)]
            neighbors = torch.cat([neighbors, sampled_neighbors])

        adj[nodeid2idx[nodeid], :] = neighbors

    return adj


'''构造节点到新索引的映射字典'''
def construct_nodeid2idx(G):
    nodeid2idx = {}
    idx = 0
    for node in G.nodes():
        if node not in nodeid2idx:
            nodeid2idx[node] = idx
            idx = idx + 1
    return nodeid2idx

class GraphSAGEDataset(Dataset):
    def __init__(self, train_nodes, score_groundtruth):
        self.train_nodes = train_nodes #只包含部分节点
        self.score_groundtruth = score_groundtruth #包含全部节点的真值

    def __len__(self):
        return len(self.train_nodes)

    def __getitem__(self, idx):
        node_id = self.train_nodes[idx]
        score = self.score_groundtruth[node_id]
        return node_id, score

class SubgraphRankingDataset(Dataset):
    def __init__(self, all_node_indices, all_scores, subgraph_size):
        self.all_nodes = all_node_indices
        self.scores = all_scores
        self.subgraph_size = subgraph_size

    def __len__(self):
        return 100  # 固定迭代次数，不是节点数

    def __getitem__(self, idx):
        sampled_indices = random.sample(self.all_nodes, self.subgraph_size)
        sampled_indices = torch.tensor(sampled_indices)

        sampled_score = self.scores[sampled_indices]    # (subgraph_size,)

        return sampled_indices, sampled_score

def precision_at_k(pred, label, k):
    # pred, label: shape (1, n)
    pred_indices = torch.topk(pred, k=k, dim=1).indices
    label_indices = torch.topk(label, k=k, dim=1).indices
    intersection = (pred_indices == label_indices).sum().item()
    return intersection / k

def L2_normalize(x):
    return F.normalize(x, p=2, dim=1)




################################################### 节点特征 ##################################
#新修改的代码
def build_node_features(G,cfg):
    '''
    node2idx = {node: idx for idx, node in enumerate(G.nodes())}
    idx2node = {idx: node for node, idx in node2idx.items()}
    num_nodes = len(G.nodes())
    features = torch.zeros((num_nodes, 9), dtype=torch.float32, device=cfg.device)

    out_deg_count = torch.zeros(num_nodes, device=cfg.device)
    in_deg_count = torch.zeros(num_nodes, device=cfg.device)

    # 1️⃣ is_bridge_node（是否为桥接节点）
    bridge_nodes = set()
    for u, v, data in G.edges(data=True):
        if 'overlap' in data.get('layer', ''):
            bridge_nodes.add(u)
            bridge_nodes.add(v)
    for node in G.nodes():
        features[node2idx[node], 0] = 1.0 if node in bridge_nodes else 0.0

    # 2️⃣ second_order_degree（二阶度数）
    for node in G.nodes():
        neighbors = set(G.neighbors(node))
        two_hop_neighbors = set()
        for n in neighbors:
            two_hop_neighbors.update(G.neighbors(n))
        two_hop_neighbors.discard(node)
        features[node2idx[node], 1] = len(two_hop_neighbors)

    # 3️⃣ inter_in_weight_sum（层间入边权重和）
    # 4️⃣ inter_out_weight_sum（层间出边权重和）
    for u, v, data in G.edges(data=True):
        src_idx = node2idx[u]
        dst_idx = node2idx[v]
        weight = data.get('weight', 0.0)
        layer_type = data.get('layer', '')

        if "overlap" in layer_type:
            features[src_idx, 3] += weight  # 第4列: inter_out_weight_sum
            features[dst_idx, 2] += weight  # 第3列: inter_in_weight_sum

        # 为 avg_in_weight / avg_out_weight 做准备
        out_deg_count[src_idx] += 1
        in_deg_count[dst_idx] += 1

        features[src_idx, 6] += weight  # 第7列: avg_out_weight（临时累加）
        features[dst_idx, 5] += weight  # 第6列: avg_in_weight（临时累加）

    # 5️⃣ clustering_coefficient（聚类系数）
    clustering = nx.clustering(G, weight='weight')
    for node, cl in clustering.items():
        features[node2idx[node], 4] = cl

    # 6️⃣ avg_in_weight，7️⃣ avg_out_weight
    features[:, 5] = features[:, 5] / (in_deg_count + 1e-8)   # avg_in_weight
    features[:, 6] = features[:, 6] / (out_deg_count + 1e-8)  # avg_out_weight

    # 8️⃣ pagerank
    pagerank = nx.pagerank(G, weight='weight')
    for node, pr in pagerank.items():
        features[node2idx[node], 7] = pr

    # 9️⃣ closeness
    closeness = nx.closeness_centrality(G)
    for node, cc in closeness.items():
        features[node2idx[node], 8] = cc

    normalize_columns = [1, 2, 3, 5, 6, 7, 8]
    features = min_max_normalize_tensor(features,dim=normalize_columns)
    '''
    node2idx = {node: idx for idx, node in enumerate(G.nodes())}
    idx2node = {idx: node for node, idx in node2idx.items()}
    num_nodes = len(G.nodes())
    features = torch.zeros((num_nodes, 4), dtype=torch.float32, device=cfg.device)

    out_deg_count = torch.zeros(num_nodes, device=cfg.device)
    in_deg_count = torch.zeros(num_nodes, device=cfg.device)

    for u, v, data in G.edges(data=True):
        src_idx = node2idx[u]
        dst_idx = node2idx[v]
        weight = data.get('weight', 0.0)
        layer_type = data.get('layer', '')

        # 出边特征
        if "overlap" in layer_type:
            # 2 inter_out_weight_sum（层间出边权重和）
            features[src_idx, 1] += weight  
        else:
            # 3 intra_out_weight_sum（层内出边权重和）
            features[src_idx, 2] += weight  
        # 1 出度
        features[src_idx, 0] += 1  # 第1维:出度
        features[dst_idx, 3] += 1  # 第4维:入度

    normalize_columns = [0, 1, 2, 3]
    features = min_max_normalize_tensor(features,dim=normalize_columns)
    return features, node2idx, idx2node


def load_scores_to_tensor_old(score_file, node2idx, cfg):
    # 初始化一个得分向量，大小等于节点数量
    scores = torch.zeros(len(node2idx), dtype=torch.float32, device=cfg.device)

    with open(score_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                node_str = parts[0]         # e.g. "(1,9)"
                score = float(parts[1])

                # 解析 node_str 为元组 (layerID, nodeID)
                layer_node = eval(node_str)

                if layer_node in node2idx:
                    idx = node2idx[layer_node]
                    scores[idx] = score
                else:
                    print(f"Warning: Node {layer_node} not found in node2idx.")

    return scores

import ast
import torch


def load_scores_to_tensor(score_file, node2idx, cfg,
                          top_ratio: float = 0.3,
                          normalize: bool = True,
                          use_log: bool = False,
                          eps: float = 1e-8):
    """
    读取 score 文件并返回张量形式的 scores（放在 cfg.device）。
    可选返回 top_ratio 对应的 top_idx（按 score 降序）。
    可选对标签做标准化（基于 top_idx，如果 top_ratio 给定，否则基于全部节点）。
    可选对标签做 log1p 变换（先 transform 再标准化）。

    返回: (scores, top_idx, mean, std)
      - scores: torch.Tensor, shape [N], device=cfg.device (原始分数，未标准化)
      - top_idx: torch.LongTensor or None (如果 top_ratio 给定则为长度 k 的 idx，在device上)
      - mean: torch.scalar Tensor or None (如果 normalize=True 则返回，基于top节点计算)
      - std:  torch.scalar Tensor or None (如果 normalize=True 则返回，基于top节点计算)
    """
    import ast

    # 初始化
    N = len(node2idx)
    scores = torch.zeros(N, dtype=torch.float32, device=cfg.device)

    print(f"Loading scores from: {score_file}")

    # 读取文件（安全解析 node 字符串）
    missing = 0
    loaded = 0
    with open(score_file, 'r') as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):  # 跳过空行和注释
                continue

            parts = line.split()
            if len(parts) < 2:
                print(f"Warning: line {line_no} has insufficient parts: {line}")
                continue

            node_str = parts[0]
            try:
                s = float(parts[1])
            except Exception as e:
                print(f"Warning: line {line_no} score parsing error: {parts[1]} - {e}")
                continue

            # 解析 "(layer,node)" 安全使用 ast.literal_eval
            try:
                layer_node = ast.literal_eval(node_str)
            except Exception:
                # 兜底解析 (比如 "1,9" 等)
                try:
                    txt = node_str.strip().lstrip('(').rstrip(')')
                    a, b = txt.split(',')
                    layer_node = (int(a.strip()), int(b.strip()))
                except Exception as e:
                    print(f"Warning: line {line_no} cannot parse node string: {node_str} - {e}")
                    continue

            if layer_node in node2idx:
                idx = node2idx[layer_node]
                scores[idx] = float(s)
                loaded += 1
            else:
                missing += 1

    print(f"Loaded {loaded} scores, {missing} entries not found in node2idx")

    # 基本统计信息
    nonzero_count = (scores != 0).sum().item()
    print(f"Total nodes: {N}, Non-zero scores: {nonzero_count}")
    print(f"Score range: [{scores.min().item():.6f}, {scores.max().item():.6f}]")

    # log 变换
    if use_log:
        print("Applying log1p transformation...")
        scores = torch.log1p(scores)
        print(f"After log1p - Score range: [{scores.min().item():.6f}, {scores.max().item():.6f}]")

    # 计算 top_idx
    top_idx = None
    if top_ratio is not None:
        if not (0.0 < top_ratio <= 1.0):
            raise ValueError("top_ratio must be in (0, 1].")
        k = max(1, int(N * top_ratio))
        sorted_idx = torch.argsort(scores, descending=True)
        top_idx = sorted_idx[:k]  # 确保在同一设备上

        print(f"Top {top_ratio * 100:.1f}% nodes: {k} nodes")
        print(f"Top {k} score range: [{scores[top_idx[-1]].item():.6f}, {scores[top_idx[0]].item():.6f}]")

    # 标准化，基于 top_idx，否则基于全部节点
    mean = None
    std = None
    if normalize:
        if top_idx is not None:
            vals = scores[top_idx]
            print(f"Computing normalization parameters based on top {len(top_idx)} nodes")
        else:
            vals = scores
            print("Computing normalization parameters based on all nodes")

        mean = vals.mean()
        std = vals.std(unbiased=False)

        # 防止除零错误
        if std.item() < eps:
            print(f"Warning: std ({std.item():.8f}) is too small, setting to 1.0")
            std = torch.tensor(1.0, device=cfg.device)

        print(f"Normalization parameters: mean={mean.item():.6f}, std={std.item():.6f}")

    return scores, top_idx, mean, std



def custom_normalize(x: torch.Tensor) -> torch.Tensor:
    # 保留原始设备信息
    assert x.ndim == 2 and x.shape[1] == 3, "输入必须是形状为 [N, 3] 的二维 Tensor"
    x_norm = torch.empty_like(x)

    # 第 1 列（索引 0） Min-Max 归一化
    col0 = x[:, 0]
    min0, max0 = col0.min(), col0.max()
    x_norm[:, 0] = (col0 - min0) / (max0 - min0 + 1e-8)

    # 第 2 列（索引 1）先 log(x + 1)，再 Min-Max 归一化
    col1 = x[:, 1].log1p()
    min1, max1 = col1.min(), col1.max()
    x_norm[:, 1] = (col1 - min1) / (max1 - min1 + 1e-8)

    # 第 3 列（索引 2） Min-Max 归一化
    col2 = x[:, 2]
    min2, max2 = col2.min(), col2.max()
    x_norm[:, 2] = (col2 - min2) / (max2 - min2 + 1e-8)

    return x_norm


def min_max_normalize_tensor(tensor: torch.Tensor, dim: Union[int, List[int], Tuple[int]] = None) -> torch.Tensor:
    """
    对输入张量的指定列（或所有列）进行 Min-Max 归一化，返回一个新张量。

    参数:
        tensor (torch.Tensor): 输入张量，形状为 (num_rows, num_features)
        dim (int, list, tuple, or None): 指定归一化的列索引，None 表示所有列

    返回:
        torch.Tensor: 归一化后的新张量
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
        min_val = col.min()
        max_val = col.max()
        denom = max_val - min_val
        if denom > 1e-8:
            tensor_clone[:, d] = (col - min_val) / (denom + 1e-8)
        else:
            tensor_clone[:, d] = 0.0  # 如果全列数值相同，归一化为 0

    return tensor_clone