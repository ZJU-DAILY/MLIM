import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple

from GraphSAGE_model.sage_utils import L2_normalize

def get_neighbor_features(features, neighbors):
    """
    从 features 中提取邻居的特征。
    - neighbors 中可能包含 -1，表示 padding 节点。
    - 将 -1 替换成 0，再创建 mask，后续可用于聚合时忽略无效邻居。
    """
    mask = (neighbors != -1).float().unsqueeze(-1)  # shape: (batch_size * k, 1)
    safe_neighbors = neighbors.clone()
    safe_neighbors[safe_neighbors == -1] = 0  # 避免访问 features[-1]
    neighbor_feats = features[safe_neighbors]  # shape: (batch_size, k, feature_dim)
    return neighbor_feats, mask

'''
    SupervisedGraphsage类要在后面整体检查
'''
class SupervisedGraphsage(nn.Module):
    """Implementation of supervised GraphSAGE in PyTorch."""
    def __init__(self, adj, input_dim, embedding_dim, final_output_dim, dropout_rate=0.1, **kwargs):
        super(SupervisedGraphsage, self).__init__()
        #只定义模型相关，不定义其他任何数据
        self.sampler = UniformNeighborSampler(adj)
        self.aggregator_firstPropagation = MeanPoolingAggregator(input_dim=input_dim, output_dim=embedding_dim, 
                                                                 dropout_rate=dropout_rate, act=F.relu)
        self.aggregator_secondPropagation = MeanPoolingAggregator(input_dim=2*embedding_dim, output_dim=embedding_dim, 
                                                                  dropout_rate=dropout_rate, act=L2_normalize)
        self.fc_pred = nn.Sequential( 
            nn.Linear(2*embedding_dim, final_output_dim),  # 线性变换
            nn.Dropout(p=dropout_rate)  # Dropout
        )
        

    def forward(self, batch_training_nodes, features, num_samples):
        #数据准备，是为了计算聚合所需要的数据，让代码更加清晰
        self_feature, hop1_neighbor_feature, hop2_neighbor_feature = self.sample(batch_training_nodes, num_samples, features)
        batch_size = len(batch_training_nodes)
        features_dim = features.shape[1]

        #模型结构与数据流
        self_embedding = self.aggregator_firstPropagation(self_feature, 
                                                          hop1_neighbor_feature.view(batch_size, num_samples[0], features_dim))
        hop1_embedding = self.aggregator_firstPropagation(hop1_neighbor_feature, 
                                                          hop2_neighbor_feature.view(batch_size * num_samples[0], num_samples[1], features_dim))
        final_self_embedding = self.aggregator_secondPropagation(self_embedding, 
                                                                 hop1_embedding.view(batch_size, num_samples[0], -1))
        node_preds = self.fc_pred(final_self_embedding)

        return node_preds

    def sample(self, batch_training_nodes, num_samples, features):
        hop1_neighbor, hop2_neighbor = self.sampler.sample_2hop_neighbor(batch_training_nodes, num_samples, features.shape[0]-1)
        self_feature = features[batch_training_nodes]

        hop1_neighbor_feature = features[hop1_neighbor].view(-1, features.shape[1]) #表示reshape为二维张量，-1表示自动计算维度，1表示第二维的维度为1
        hop2_neighbor_feature = features[hop2_neighbor].view(-1, features.shape[1])
        return self_feature, hop1_neighbor_feature, hop2_neighbor_feature 
    
    #不使用了
    def loss(self, preds, labels, weight_decay=0.00002):
        # 每个aggregator的 L2 正则化项（仅包含 self.fc_self 和 self.fc_neighbor）
        agg_l2_loss = 0.5*sum(torch.sum(p ** 2) for p in self.aggregator_firstPropagation.fc_self.parameters())
        agg_l2_loss += 0.5*sum(torch.sum(p ** 2) for p in self.aggregator_firstPropagation.fc_neighbor.parameters())
        agg_l2_loss += 0.5*sum(torch.sum(p ** 2) for p in self.aggregator_secondPropagation.fc_self.parameters())
        agg_l2_loss += 0.5*sum(torch.sum(p ** 2) for p in self.aggregator_secondPropagation.fc_neighbor.parameters())

        pred_layer_l2_loss = 0.5 * sum(torch.sum(p ** 2) for p in self.fc_pred.parameters())

        l2_loss = agg_l2_loss + pred_layer_l2_loss

        # 计算 MSE 损失
        mse_loss = F.mse_loss(preds.view(-1), labels.view(-1))

        return mse_loss + weight_decay * l2_loss



class MeanPoolingAggregator(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.1, act=F.relu):
        super(MeanPoolingAggregator, self).__init__()
        #所需参数
        self.input_dim = input_dim
        self.neighbor_hidden_dim = 512

        #模型结构
        self.fc_self = nn.Linear(input_dim, output_dim, bias=False)
        self.neighbor_embedding_layer = nn.Sequential( 
            nn.Linear(input_dim, self.neighbor_hidden_dim),  # 线性变换
            nn.ReLU(),  # ReLU 激活
            nn.Dropout(p=dropout_rate)  # Dropout
        )
        self.fc_neighbor = nn.Linear(self.neighbor_hidden_dim, output_dim, bias=False)
        self.act = act #激活函数


    def forward(self, self_feature, neighbor_feature):
        self_feature = self.fc_self(self_feature) 

        batch_size, num_neighbor, _ = neighbor_feature.shape
        neighbor_feature = neighbor_feature.view(batch_size*num_neighbor, self.input_dim)  
        neighbor_new_feature = self.neighbor_embedding_layer(neighbor_feature)  
        neighbor_new_feature = neighbor_new_feature.view(batch_size, num_neighbor, self.neighbor_hidden_dim)  
        
        aggregated_feature = neighbor_new_feature.mean(dim=1)
        aggregated_feature = self.fc_neighbor(aggregated_feature)

        output_feature = torch.cat((self_feature, aggregated_feature), dim=-1)
        output_feature = self.act(output_feature)
        return output_feature
    

class UniformNeighborSampler:
    def __init__(self, adj):
        self.adj_info = adj #邻接矩阵

    '''采样传入的batch节点的第一跳和第二跳邻居, 每次采样邻居的时候尽量采样不同的邻居
        采样第二跳邻居的时候，实际上是对第一跳邻居的邻居进行采样，但是不会采样到该节点本身'''
    def sample_2hop_neighbor(self, batch, num_samples, num_nodes):
        hop1_neighbors = self.adj_info[batch, :num_samples[0]]  #num_samples[0]=samples_1

        hop2_neighbors = []
        for i in range(len(batch)):
            first_hop = hop1_neighbors[i]

            second_hop = []
            for node in first_hop:
                node_neighbors = self.adj_info[node]  # 获取该节点的邻居        
                node_neighbors = node_neighbors[node_neighbors != batch[i]] # 排除掉节点本身，避免自己成为邻居
                sampled_neighbors = node_neighbors[:num_samples[1]] #num_samples[1]=samples_2

                if len(sampled_neighbors) < num_samples[1]:
                    pad_len = num_samples[1] - len(sampled_neighbors)
                    sampled_neighbors = torch.cat([sampled_neighbors, torch.full((pad_len,), num_nodes, dtype=torch.long, device=sampled_neighbors.device)]) 
                
                second_hop.extend(sampled_neighbors)

            hop2_neighbors.append(second_hop)

        hop2_neighbors = torch.tensor(hop2_neighbors)

        return hop1_neighbors, hop2_neighbors
