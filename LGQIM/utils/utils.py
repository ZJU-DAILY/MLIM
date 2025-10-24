import os
import random
import networkx as nx

"""构建多层图"""
#有时间可以检查层内边，一个月之后吧
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

def print_graph_node_info(G):
    nodes = list(G.nodes)
    num_nodes = len(nodes)
    num_edges = G.number_of_edges()
    print(f"number of nodes: {num_nodes}")
    print(f"number of edges: {num_edges}")


    # 如果想看所有层有哪些，也可以：
    layers = sorted(set(layer for layer, _ in nodes))
    print(f"number of layers: {len(layers)}: {layers}")
    print()


def print_sorted_edges_by_layer_for_check(G):
    layers = sorted(set(node[0] for node in G.nodes))  # 获取所有层，并排序
    
    for layer in layers:
        edges_in_layer = [
            (u[1], v[1], int(d['weight'] * 255))  # 取 nodeID 并转换权重
            for u, v, d in G.edges(data=True) 
            if u[0] == v[0] == layer  # 确保边在同一层
        ]
        
        if edges_in_layer:  # 该层有边才打印
            print(layer)  # 先打印层编号
            edges_in_layer.sort()  # 先按 source 排序，再按 target 排序
            for src, dst, weight in edges_in_layer:
                print(src, dst, weight)



def print_config(config, config_name):
    print(f"\n{config_name} Configuration:")
    for key, value in vars(config).items():
        print(f"  {key}: {value}")

def print_dict(data):
    for key, value in data.items():
        print(f"{key}: {value}")