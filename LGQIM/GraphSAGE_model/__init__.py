# src/GraphSAGE_model/__init__.py

from .model import SupervisedGraphsage  # 导入SupervisedGraphsage类 
from .trainer import train_model, save_model, load_model, predict  # 导入训练、保存、加载和预测函数
from .sage_utils import (
    read_data_from_file,
    select_top_nodes,
    construct_adj,
    construct_nodeid2idx,
    GraphSAGEDataset,
    L2_normalize
)  # 导入其他实用函数

#__all__ =什么，那么from . import * 就会导入这些函数
__all__ = [
    "SupervisedGraphsage",
    "train_model",
    "save_model",
    "load_model",
    "predict",
    "read_data_from_file",
    "print_dict",
    "select_top_nodes",
    "construct_adj",
    "construct_nodeid2idx",
    "GraphSAGEDataset",
    "UniformNeighborSampler",
    "SAGEInfo"
]

