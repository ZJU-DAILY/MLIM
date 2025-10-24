import argparse
import torch

from configs import GraphSAGEConfig, QNetConfig

def parse_args():
    parser = argparse.ArgumentParser(description="GraphSAGE & QNet Training Configuration")

    # GraphSAGE 参数
    parser.add_argument("--max_degree", type=int, default=GraphSAGEConfig.max_degree, help="Max degree for neighbor sampling")
    parser.add_argument("--samples_1", type=int, default=GraphSAGEConfig.samples_1, help="Number of samples for 1-hop neighbors")
    parser.add_argument("--samples_2", type=int, default=GraphSAGEConfig.samples_2, help="Number of samples for 2-hop neighbors")
    parser.add_argument("--embedding_dim", type=int, default=GraphSAGEConfig.embedding_dim, help="Embedding dimension")
    parser.add_argument("--output_dim", type=int, default=GraphSAGEConfig.output_dim, help="Output feature dimension")
    #parser.add_argument("--batch_size", type=int, default=GraphSAGEConfig.batch_size, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=GraphSAGEConfig.num_epochs, help="Number of training epochs")
    parser.add_argument("--device", type=str, default=GraphSAGEConfig.device, help="Specify device (e.g., 'cuda:0', 'cuda:1', 'cpu')")
    parser.add_argument("--dataset", type=str, default=GraphSAGEConfig.dataset, help="Specify dataset")
    parser.add_argument("--ratio", type=float, default=GraphSAGEConfig.ratio, help="Output training ratio")

    # QNet 参数
    parser.add_argument("--qnet_k", type=int, default=QNetConfig.K, help="K value for QNet")
    parser.add_argument("--qnet_episodes", type=int, default=QNetConfig.GAME_EPISODES, help="Number of training episodes for QNet")
    parser.add_argument("--qnet_embedding_dim", type=int, default=QNetConfig.EMBEDDING_DIM, help="Embedding dimension for QNet")
    parser.add_argument("--qnet_n_step", type=int, default=QNetConfig.N_STEP, help="Sliding window size for QNet")
    parser.add_argument("--qnet_learning_rate", type=float, default=QNetConfig.LEARNING_RATE, help="Learning rate for QNet")
    parser.add_argument("--qnet_num_epochs", type=int, default=QNetConfig.NUM_EPOCHS, help="Number of epochs per environment state for QNet")
    parser.add_argument("--qnet_batch_size", type=int, default=QNetConfig.BATCH_SIZE, help="Batch size for QNet training")
    parser.add_argument("--qnet_device", type=str, default=QNetConfig.DEVICE, help="Specify device for QNet")
    parser.add_argument("--qnet_gamma", type=float, default=QNetConfig.GAMMA, help="Discount factor for Q-learning")
    parser.add_argument("--qnet_epsilon_start", type=float, default=QNetConfig.EPSILON_START, help="Starting epsilon value")
    parser.add_argument("--qnet_epsilon_end", type=float, default=QNetConfig.EPSILON_END, help="Final epsilon value")
    parser.add_argument("--qnet_epsilon_decay_ratio", type=float, default=QNetConfig.EPSILON_DECAY_RATIO, help="Number of steps over which epsilon decays")
    parser.add_argument("--qnet_target_update", type=int, default=QNetConfig.TARGET_UPDATE, help="Target network update frequency")
    parser.add_argument("--qnet_hidden_dim", type=int, default=QNetConfig.HIDDEN_DIM, help="Q-network hidden dim")
    parser.add_argument("--qnet_dataset", type=str, default=QNetConfig.DATASET, help="Q-network hidden dim")
    parser.add_argument("--qnet_R", type=int, default=QNetConfig.R, help="R value for QNet, GNN epoch")
    parser.add_argument("--qnet_gts", type=int, default=QNetConfig.gts, help="ground truth score for QNet")
    parser.add_argument("--qnet_ratio", type=float, default=QNetConfig.ratio, help="training size for QNet")
    return parser.parse_args()

def load_config():
    args = parse_args()

    # 更新 GraphSAGEConfig
    graph_sage_config = GraphSAGEConfig()
    graph_sage_config.max_degree = args.max_degree
    graph_sage_config.samples_1 = args.samples_1
    graph_sage_config.samples_2 = args.samples_2
    graph_sage_config.embedding_dim = args.embedding_dim
    graph_sage_config.output_dim = args.output_dim
    #graph_sage_config.batch_size = args.batch_size
    graph_sage_config.num_epochs = args.num_epochs
    graph_sage_config.device = torch.device(args.device)
    graph_sage_config.dataset = args.dataset
    graph_sage_config.ratio = args.ratio
    
    # 更新 QNetConfig
    qnet_config = QNetConfig()
    qnet_config.K = args.qnet_k
    qnet_config.GAME_EPISODES = args.qnet_episodes
    qnet_config.EMBEDDING_DIM = args.qnet_embedding_dim
    qnet_config.N_STEP = args.qnet_n_step
    qnet_config.LEARNING_RATE = args.qnet_learning_rate
    qnet_config.NUM_EPOCHS = args.qnet_num_epochs
    qnet_config.BATCH_SIZE = args.qnet_batch_size
    qnet_config.DEVICE = torch.device(args.qnet_device)
    qnet_config.GAMMA = args.qnet_gamma
    qnet_config.EPSILON_START = args.qnet_epsilon_start
    qnet_config.EPSILON_END = args.qnet_epsilon_end
    qnet_config.EPSILON_DECAY_RATIO = args.qnet_epsilon_decay_ratio
    qnet_config.TARGET_UPDATE = args.qnet_target_update
    qnet_config.HIDDEN_DIM = args.qnet_hidden_dim
    qnet_config.DATASET = args.qnet_dataset
    qnet_config.R = args.qnet_R
    qnet_config.gts = args.qnet_gts
    qnet_config.ratio = args.qnet_ratio

    return graph_sage_config, qnet_config
