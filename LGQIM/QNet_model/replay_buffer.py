import torch
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
       
        self.buffer.append(transition)

    def sample(self, batch_size: int, sequential: bool = False):
        batch_size = min(batch_size, len(self.buffer))
        if sequential:
            start = random.randint(0, len(self.buffer) - batch_size)
            batch = [self.buffer[i] for i in range(start, start + batch_size)]
        else:
            batch = random.sample(self.buffer, batch_size)

        states, rewards, next_states = zip(*batch) 

    
        mu_v_s, mu_s_s, mu_l_s = zip(*states)
        mu_v_n, mu_s_n, mu_l_n = zip(*next_states)

        device = mu_v_s[0].device

     
        mu_v_s = torch.stack(mu_v_s)
        mu_s_s = torch.stack(mu_s_s)
        mu_l_s = torch.stack(mu_l_s)

        mu_v_n = torch.stack(mu_v_n)
        mu_s_n = torch.stack(mu_s_n)
        mu_l_n = torch.stack(mu_l_n)

    
        if isinstance(rewards[0], torch.Tensor):
            rewards = torch.stack(rewards).to(device=device)
        else:
            rewards = torch.tensor(rewards, dtype=torch.float32, device=device)

        rewards = rewards.view(-1, 1)
        return [mu_v_s, mu_s_s, mu_l_s], rewards, [mu_v_n, mu_s_n, mu_l_n]

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)

