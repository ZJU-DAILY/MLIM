import torch
import torch.nn as nn
import torch.optim as optim

#是否最好是input_dim=embedding_dim?
# QNet的结构
# input_dim=configs.EMBEDDING_DIM, embedding_dim=configs.HIDDEN_DIM
class QNetwork(nn.Module):
    def __init__(self, input_dim, embedding_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, embedding_dim)  # 等价于 layer_1
        self.fc2 = nn.Linear(input_dim-1, embedding_dim)  # 等价于 layer_2
        self.fc3 = nn.Linear(input_dim-1, embedding_dim)  # 等价于 layer_3
        self.output_layer = nn.Linear(embedding_dim * 3, output_dim)  # 拼接后全连接
        # 不使用 ReLU，之后可以尝试使用

    def forward(self, mu_v, mu_selected, mu_left):
        h1 = self.fc1(mu_v) # 向量，代表一个节点
        h2 = self.fc2(mu_selected) # 已经被选择为种子节点的向量，取所有向量每一维的max作为最终的向量
        h3 = self.fc3(mu_left) # 未被选择为种子节点的向量，取所有向量每一维的max作为最终的向量

        concat_layer = torch.cat([h1, h2, h3], dim=1) # 拼接
        output = self.output_layer(concat_layer)
        return output