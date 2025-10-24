import torch
import torch.nn as nn
import random
from q_network import QNetwork
from replay_buffer import ReplayBuffer
from utils_dqn import get_best_node


class Agent:
    def __init__(self, cfg):
        self.device = cfg.DEVICE
        self.gamma = cfg.GAMMA  # 折扣因子
        self.epsilon = cfg.EPSILON_START #用于打印
        self.epsilon_start = cfg.EPSILON_START
        self.epsilon_end = cfg.EPSILON_END
        self.epsilon_decay_steps = int(cfg.GAME_EPISODES * cfg.K * cfg.EPSILON_DECAY_RATIO)
        self.batch_size = cfg.BATCH_SIZE

        self.q_network = QNetwork(cfg.EMBEDDING_DIM, cfg.HIDDEN_DIM, 1).to(self.device)
        self.target_network = QNetwork(cfg.EMBEDDING_DIM, cfg.HIDDEN_DIM, 1).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_update = cfg.TARGET_UPDATE


        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=cfg.LEARNING_RATE)
        self.replay_buffer = ReplayBuffer(3*cfg.K)

        self.total_steps = 0

    
    def select_action(self, env, step):
        self.total_steps += 1
        decay_ratio = min(self.total_steps / self.epsilon_decay_steps, 1.0)
        self.epsilon = self.epsilon_start - decay_ratio * (self.epsilon_start - self.epsilon_end)
        if random.random() < self.epsilon: #random.random() ∈ [0.0, 1.0)
            if step==0 :
                selected_node = env.get_random_step0_node()
            else:
                selected_node = env.get_random_node()
        else:
            selected_node = get_best_node(env, step, self.q_network)
            if selected_node==(-1, -1):
                selected_node = env.get_random_node()
        return selected_node
    
    def update(self, cfg):
        if len(self.replay_buffer) < cfg.BATCH_SIZE:
            return
        
        state, reward, next_n_state = self.replay_buffer.sample(cfg.BATCH_SIZE)
        #state [mu_v, mu_selected, mu_left], #reward 选择mu_v的N_STEP实际边际增益, next_n_state t+N_STEP的state

        q_values = self.q_network(state[0], state[1], state[2]) #预测选择节点之后整个种子集的影响力
        next_n_q_values = self.target_network(next_n_state[0], next_n_state[1], next_n_state[2])
        target_q_values = reward + (cfg.GAMMA**cfg.N_STEP) * next_n_q_values
        
        assert q_values.shape == target_q_values.shape, f"Shape mismatch: q_values {q_values.shape}, target_q_values {target_q_values.shape}"
        loss = nn.MSELoss()(q_values, target_q_values.detach())  # 计算均方根损失      
        
        # 优化更新模型
        self.optimizer.zero_grad()  
        loss.backward()
        # clip防止梯度爆炸
        for param in self.q_network.parameters():  
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step() 
        if self.total_steps % self.target_update == 0: # 每隔一段时间，将策略网络的参数复制到目标网络
            self.target_network.load_state_dict(self.q_network.state_dict())

    def predict_action(self, env, step):
        selected_node = get_best_node(env, step, self.q_network)
        return selected_node