import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, input_dim, embedding_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, embedding_dim)  
        self.fc2 = nn.Linear(input_dim-1, embedding_dim)  
        self.fc3 = nn.Linear(input_dim-1, embedding_dim)  
        self.output_layer = nn.Linear(embedding_dim * 3, output_dim)  

    def forward(self, mu_v, mu_selected, mu_left):
        h1 = self.fc1(mu_v) 
        h2 = self.fc2(mu_selected) 
        h3 = self.fc3(mu_left)

        concat_layer = torch.cat([h1, h2, h3], dim=1) 
        output = self.output_layer(concat_layer)
        return output
