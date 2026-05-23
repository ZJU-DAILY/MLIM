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


def select_top_nodes(xv_dict, rbmax, nodeid2idx):
    sorted_nodes = sorted(xv_dict.items(), key=lambda item: item[1].item(), reverse=True)
    selected_nodes = [node for idx, (node, _) in enumerate(sorted_nodes) if idx < rbmax]
    selected_nodes_idx = [nodeid2idx[node] for node in selected_nodes]
    return selected_nodes_idx


def construct_adj(G, max_degree, nodeid2idx):
    num_nodes = G.number_of_nodes()

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
        self.train_nodes = train_nodes #
        self.score_groundtruth = score_groundtruth #

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
        return 100  #

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



def build_node_features(G,cfg):
    
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
            features[src_idx, 1] += weight  
        else:
            features[src_idx, 2] += weight  
      
        features[src_idx, 0] += 1  
        features[dst_idx, 3] += 1 

    normalize_columns = [0, 1, 2, 3]
    features = min_max_normalize_tensor(features,dim=normalize_columns)
    return features, node2idx, idx2node


def load_scores_to_tensor_old(score_file, node2idx, cfg):
 
    scores = torch.zeros(len(node2idx), dtype=torch.float32, device=cfg.device)

    with open(score_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                node_str = parts[0]         # e.g. "(1,9)"
                score = float(parts[1])

               
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

    import ast


    N = len(node2idx)
    scores = torch.zeros(N, dtype=torch.float32, device=cfg.device)

    print(f"Loading scores from: {score_file}")


    missing = 0
    loaded = 0
    with open(score_file, 'r') as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'): 
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

         
            try:
                layer_node = ast.literal_eval(node_str)
            except Exception:
              
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

  
    nonzero_count = (scores != 0).sum().item()
    print(f"Total nodes: {N}, Non-zero scores: {nonzero_count}")
    print(f"Score range: [{scores.min().item():.6f}, {scores.max().item():.6f}]")

    
    if use_log:
        print("Applying log1p transformation...")
        scores = torch.log1p(scores)
        print(f"After log1p - Score range: [{scores.min().item():.6f}, {scores.max().item():.6f}]")


    top_idx = None
    if top_ratio is not None:
        if not (0.0 < top_ratio <= 1.0):
            raise ValueError("top_ratio must be in (0, 1].")
        k = max(1, int(N * top_ratio))
        sorted_idx = torch.argsort(scores, descending=True)
        top_idx = sorted_idx[:k] 

        print(f"Top {top_ratio * 100:.1f}% nodes: {k} nodes")
        print(f"Top {k} score range: [{scores[top_idx[-1]].item():.6f}, {scores[top_idx[0]].item():.6f}]")


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

      
        if std.item() < eps:
            print(f"Warning: std ({std.item():.8f}) is too small, setting to 1.0")
            std = torch.tensor(1.0, device=cfg.device)

        print(f"Normalization parameters: mean={mean.item():.6f}, std={std.item():.6f}")

    return scores, top_idx, mean, std



def custom_normalize(x: torch.Tensor) -> torch.Tensor:
   
    assert x.ndim == 2 and x.shape[1] == 3, 
    x_norm = torch.empty_like(x)

  
    col0 = x[:, 0]
    min0, max0 = col0.min(), col0.max()
    x_norm[:, 0] = (col0 - min0) / (max0 - min0 + 1e-8)

    
    col1 = x[:, 1].log1p()
    min1, max1 = col1.min(), col1.max()
    x_norm[:, 1] = (col1 - min1) / (max1 - min1 + 1e-8)

    
    col2 = x[:, 2]
    min2, max2 = col2.min(), col2.max()
    x_norm[:, 2] = (col2 - min2) / (max2 - min2 + 1e-8)

    return x_norm


def min_max_normalize_tensor(tensor: torch.Tensor, dim: Union[int, List[int], Tuple[int]] = None) -> torch.Tensor:
   
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
            tensor_clone[:, d] = 0.0  

    return tensor_clone
