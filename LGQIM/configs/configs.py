import torch

class GraphSAGEConfig:
    max_degree = 128
    samples_1 = 10 # 1-hop neibor
    samples_2 = 5 # 2-hop neibor
    embedding_dim = 60 # feature 4维 -> 60维 -> 120维 -> 1维（score）# parameter sensitive [30 60 90 120 150]
    output_dim = 1
    #batch_size = 100
    num_epochs = 500 # 待定
    ratio = 0.1
    device = "cpu"
    dataset = "ff-tw-yt"

class QNetConfig:
    K = 100 # 种子节点数 （默认为种子数最大值）
    GAME_EPISODES = 200 #一共玩几次游戏; 一共进行几次IM选择, 待定
    EMBEDDING_DIM = 3 #强化学习中，每个节点embedding需要用3个特征来表示
    HIDDEN_DIM = 10 #QNet的隐藏层维度 , parameter sensitive [10 20 30 40 50]
    N_STEP = 2  # n步回报，算法参数 [1 2 3 4 5]
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 1 #在一次训练周期中，进行num_epochs次采样经验、训练、更新参数的操作。
    BATCH_SIZE = 6 #每次从replay buffer中选择batchSize条数据进行训练
    DEVICE = "cpu"
    GAMMA = 0.8 # 折扣因子  [0 0.2 0.4 0.6 0.8 1.0] (GAMMA越大，影响力越大)
    EPSILON_START = 1.0
    EPSILON_END = 0.1
    EPSILON_DECAY_RATIO = 0.8
    TARGET_UPDATE = 100
    DATASET = "ff-tw-yt"
    R = 200 # GNN预测score时的epoch次数
    gts = 1 # use ground truth score
    ratio = 0.1

    '''BUFFER_SIZE = 10000  # 经验回放缓存大小
    BATCH_SIZE = 64
    EPSILON_START = 1.0
    EPSILON_END = 0.1
    EPSILON_DECAY = 1000'''

