import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple

from GraphSAGE_model.sage_utils import L2_normalize

def get_neighbor_features(features, neighbors):

    mask = (neighbors != -1).float().unsqueeze(-1)  # shape: (batch_size * k, 1)
    safe_neighbors = neighbors.clone()
    safe_neighbors[safe_neighbors == -1] = 0 
    neighbor_feats = features[safe_neighbors]  # shape: (batch_size, k, feature_dim)
    return neighbor_feats, mask


class SupervisedGraphsage(nn.Module):
    """Implementation of supervised GraphSAGE in PyTorch."""
    def __init__(self, adj, input_dim, embedding_dim, final_output_dim, dropout_rate=0.1, **kwargs):
        super(SupervisedGraphsage, self).__init__()
       
        self.sampler = UniformNeighborSampler(adj)
        self.aggregator_firstPropagation = MeanPoolingAggregator(input_dim=input_dim, output_dim=embedding_dim, 
                                                                 dropout_rate=dropout_rate, act=F.relu)
        self.aggregator_secondPropagation = MeanPoolingAggregator(input_dim=2*embedding_dim, output_dim=embedding_dim, 
                                                                  dropout_rate=dropout_rate, act=L2_normalize)
        self.fc_pred = nn.Sequential( 
            nn.Linear(2*embedding_dim, final_output_dim),  
            nn.Dropout(p=dropout_rate)  # Dropout
        )
        

    def forward(self, batch_training_nodes, features, num_samples):
       
        self_feature, hop1_neighbor_feature, hop2_neighbor_feature = self.sample(batch_training_nodes, num_samples, features)
        batch_size = len(batch_training_nodes)
        features_dim = features.shape[1]

       
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
    
    
    def loss(self, preds, labels, weight_decay=0.00002):
       
        agg_l2_loss = 0.5*sum(torch.sum(p ** 2) for p in self.aggregator_firstPropagation.fc_self.parameters())
        agg_l2_loss += 0.5*sum(torch.sum(p ** 2) for p in self.aggregator_firstPropagation.fc_neighbor.parameters())
        agg_l2_loss += 0.5*sum(torch.sum(p ** 2) for p in self.aggregator_secondPropagation.fc_self.parameters())
        agg_l2_loss += 0.5*sum(torch.sum(p ** 2) for p in self.aggregator_secondPropagation.fc_neighbor.parameters())

        pred_layer_l2_loss = 0.5 * sum(torch.sum(p ** 2) for p in self.fc_pred.parameters())

        l2_loss = agg_l2_loss + pred_layer_l2_loss

        
        mse_loss = F.mse_loss(preds.view(-1), labels.view(-1))

        return mse_loss + weight_decay * l2_loss



class MeanPoolingAggregator(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.1, act=F.relu):
        super(MeanPoolingAggregator, self).__init__()
      
        self.input_dim = input_dim
        self.neighbor_hidden_dim = 512

       
        self.fc_self = nn.Linear(input_dim, output_dim, bias=False)
        self.neighbor_embedding_layer = nn.Sequential( 
            nn.Linear(input_dim, self.neighbor_hidden_dim), 
            nn.ReLU(),  
            nn.Dropout(p=dropout_rate)  # Dropout
        )
        self.fc_neighbor = nn.Linear(self.neighbor_hidden_dim, output_dim, bias=False)
        self.act = act 


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
        self.adj_info = adj 

    def sample_2hop_neighbor(self, batch, num_samples, num_nodes):
        hop1_neighbors = self.adj_info[batch, :num_samples[0]]  #num_samples[0]=samples_1

        hop2_neighbors = []
        for i in range(len(batch)):
            first_hop = hop1_neighbors[i]

            second_hop = []
            for node in first_hop:
                node_neighbors = self.adj_info[node]        
                node_neighbors = node_neighbors[node_neighbors != batch[i]] 
                sampled_neighbors = node_neighbors[:num_samples[1]] #num_samples[1]=samples_2

                if len(sampled_neighbors) < num_samples[1]:
                    pad_len = num_samples[1] - len(sampled_neighbors)
                    sampled_neighbors = torch.cat([sampled_neighbors, torch.full((pad_len,), num_nodes, dtype=torch.long, device=sampled_neighbors.device)]) 
                
                second_hop.extend(sampled_neighbors)

            hop2_neighbors.append(second_hop)

        hop2_neighbors = torch.tensor(hop2_neighbors)

        return hop1_neighbors, hop2_neighbors
