import copy
import torch
import random
from types import MappingProxyType

from utils_dqn import get_short_reward, get_graph, normalize_tensor, find_overlap_targets

class GraphEnvironment:
    def __init__(self, cfg, graph_path, pred_path, use_gts, ratio):
        self.num_k = cfg.K
        self.graph_dir = graph_path
        self.pred_path = pred_path
        self.use_gts = use_gts
        self.ratio = ratio

        #读入数据
        #self.top_tenpct_nodes, self.dict_node_sampled_neighbors 是真实节点, 在CPU上
        #self.embedding 是张量,已经放在了GPU上,需要配合索引使用
        (self.graphX, self.top_tenpct_nodes, self.goodNode2idx, self.embedding_archive_0time, self.embedding, 
         self.dict_node_sampled_neighbors, self.node_score) = get_graph(self.graph_dir, self.pred_path, self.use_gts, self.ratio,cfg)
        self.node_score = MappingProxyType(self.node_score)

        # 初始化状态
        self.reset()


    def reset(self):
        """重置环境状态，为新的 episode 做准备"""
        self.is_selected = {key: -1 for key in self.top_tenpct_nodes}  # 每个节点被选中的时间步

        self.state = []  # 当前已选节点序列
        #self.cumulativeReward = []  # 累计奖励
        self.rewards = []
        self.neighbors_chosen_till_now = set()  # 已选节点的邻居集合

        # self.embedding_time是dict, self.embedding_time[i]是放在GPU上的张量
        self.embedding_time = {0: self.embedding.clone()}
        self.embedding_test_archive = self.embedding_archive_0time.detach().clone() #0时刻未归一化的嵌入
        self.embedding_train_archive = self.embedding_archive_0time.detach().clone() #0时刻未归一化的嵌入


    #可能还需要进行调整
    def step_old(self, action, step, previous_spread, proc):
        #更新t时刻相关状态
        self.is_selected[action] = step
        self.state.append(action)

        #更新reward,但是我还是想惩罚同一节点的选择，想想方法
        short_reward, previous_spread = get_short_reward(self.state[:], previous_spread, proc)
        self.rewards.append(short_reward)

        #更新embedding
        # 缓存局部变量
        embedding = self.embedding_time
        goodNode2idx = self.goodNode2idx
        top_nodes = self.top_tenpct_nodes
        graphX = self.graphX
        node_score = self.node_score

        #更新t+1时刻嵌入
        if (step + 1) not in embedding:
            embedding[step + 1] = self.embedding_train_archive.clone() #device一致


        #仅仅针对新增加的邻居所涉及的节点进行更新，提升速度
        neighbors_of_chosen_node = self.dict_node_sampled_neighbors[action]#无需使用索引
        new_neighbors = set(neighbors_of_chosen_node) - self.neighbors_chosen_till_now
        self.neighbors_chosen_till_now.update(neighbors_of_chosen_node)

        # 用局部 embedding
        embed_next = embedding[step + 1]

        #更新邻居数量
        idx_list = []
        for new_node in new_neighbors:
            for src in graphX.predecessors(new_node):
                if src in top_nodes:
                    idx_list.append(goodNode2idx[src])

        if idx_list:
            idx_tensor = torch.tensor(idx_list, device=embed_next.device)
            # 每个 idx 对应 -1
            updates = torch.ones_like(idx_tensor, dtype=embed_next.dtype, device=embed_next.device)
            # index_add_ 默认是加，这里加负数就是减
            embed_next[:, 0].index_add_(0, idx_tensor, -updates)
            embed_next[:, 0].clamp_(min=0)

        #寻找有没有相同节点
        has_same_node, same_nodes = find_overlap_targets(self.graphX, action)                
        for node in neighbors_of_chosen_node:
            if node in top_nodes:
                idx = goodNode2idx[node]
                edge_data = graphX.get_edge_data(action, node)
                w = edge_data.get("weight", None)
                embed_next[idx, 1] -= w * node_score[action]
                embed_next[idx, 1].clamp_(min=0)
                if node in same_nodes:
                    embed_next[idx, 2] += 1

        self.embedding_train_archive = embed_next.detach().clone()       

        #没有就直接进行归一化, 注意标识符位不归一化
        index = list(range(embed_next.shape[1] - 1))
        embedding[step + 1] = normalize_tensor(embed_next, index)

        return previous_spread

    def step_old2(self, action, step, previous_spread, proc):
        # -------------------- 1️⃣ 更新状态 --------------------
        self.is_selected[action] = step
        self.state.append(action)

        # 更新 reward
        short_reward, previous_spread = get_short_reward(self.state[:], previous_spread, proc)
        self.rewards.append(short_reward)

        # -------------------- 2️⃣ 缓存局部变量 --------------------
        embedding = self.embedding_time
        goodNode2idx = self.goodNode2idx
        top_nodes = self.top_tenpct_nodes
        graphX = self.graphX
        node_score = self.node_score

        # -------------------- 3️⃣ 更新 t+1 embedding --------------------
        if (step + 1) not in embedding:
            embedding[step + 1] = self.embedding_train_archive.clone()  # device 一致
        embed_next = embedding[step + 1]

        # -------------------- 4️⃣ 更新邻居影响 --------------------
        neighbors_of_chosen_node = self.dict_node_sampled_neighbors[action]
        new_neighbors = set(neighbors_of_chosen_node) - self.neighbors_chosen_till_now
        self.neighbors_chosen_till_now.update(neighbors_of_chosen_node)

        # 批量索引更新 embed_next[:,0]
        # 批量索引更新 embed_next[:,0]
        idx_list = [goodNode2idx[src]
                    for new_node in new_neighbors
                    for src in graphX.predecessors(new_node)
                    if src in top_nodes]
        if idx_list:
            idx_tensor = torch.tensor(idx_list, device=embed_next.device)
            updates = torch.ones_like(idx_tensor, dtype=embed_next.dtype, device=embed_next.device)
            embed_next[:, 0].index_add_(0, idx_tensor, -updates)
            embed_next[:, 0].clamp_(min=0)

        # -------------------- 5️⃣ 更新边权 + 同节点惩罚 --------------------
        same_nodes = set(find_overlap_targets(graphX, action)[1])
        top_neighbors = [n for n in neighbors_of_chosen_node if n in top_nodes]
        if top_neighbors:
            idx_tensor = torch.tensor([goodNode2idx[n] for n in top_neighbors], device=embed_next.device)
            weights = torch.tensor([graphX.get_edge_data(action, n).get("weight", 0) * node_score[action]
                                    for n in top_neighbors], dtype=embed_next.dtype, device=embed_next.device)
            embed_next[idx_tensor, 1] -= weights
            embed_next[idx_tensor, 1].clamp_(min=0)

        # 同节点惩罚
        same_idx = [goodNode2idx[n] for n in same_nodes if n in top_nodes]
        if same_idx:
            same_idx_tensor = torch.tensor(same_idx, device=embed_next.device)
            embed_next[same_idx_tensor, 2] += 1

        self.embedding_train_archive = embed_next.detach().clone()

        # -------------------- 6️⃣ 延迟归一化 --------------------
        # 可每 5 step 批量归一化
        index = list(range(embed_next.shape[1] - 1))
        if step % 5 == 0:
            embedding[step + 1] = normalize_tensor(embed_next, index)
        else:
            embedding[step + 1] = embed_next

        return previous_spread

    def step_old3(self, action, step, previous_spread, proc):

        # -------------------- 1️ 更新状态 --------------------
        self.is_selected[action] = step
        self.state.append(action)

        # 更新 reward
        short_reward, previous_spread = get_short_reward(self.state[:], previous_spread, proc)
        self.rewards.append(short_reward)

        # -------------------- 2️ 缓存局部变量 --------------------
        embedding = self.embedding_time
        goodNode2idx = self.goodNode2idx
        top_nodes = self.top_tenpct_nodes
        top_nodes_set = set(top_nodes)  # 转为set提高查找效率
        graphX = self.graphX
        node_score = self.node_score

        # -------------------- 3️ 更新 t+1 embedding --------------------
        if (step + 1) not in embedding:
            embedding[step + 1] = self.embedding_train_archive.clone()
        embed_next = embedding[step + 1]

        # -------------------- 4️ 优化的邻居影响更新 --------------------
        neighbors_of_chosen_node = self.dict_node_sampled_neighbors.get(action, [])
        new_neighbors = set(neighbors_of_chosen_node) - self.neighbors_chosen_till_now

        # 早期退出：如果没有新邻居，跳过后续计算
        if not new_neighbors:
            return previous_spread

        self.neighbors_chosen_till_now.update(neighbors_of_chosen_node)

        # 关键优化：批量收集所有前驱节点，避免嵌套循环
        # 原来的嵌套循环是主要性能瓶颈
        all_predecessors_in_top = set()
        for new_node in new_neighbors:
            # 使用缓存避免重复计算predecessors
            cache_key = f"pred_{new_node}"
            if not hasattr(self, '_pred_cache'):
                self._pred_cache = {}

            if cache_key not in self._pred_cache:
                predecessors = set(graphX.predecessors(new_node))
                self._pred_cache[cache_key] = predecessors & top_nodes_set

            all_predecessors_in_top.update(self._pred_cache[cache_key])

        # 批量创建idx_list，避免重复的集合查找
        if all_predecessors_in_top:
            idx_list = [goodNode2idx[src] for src in all_predecessors_in_top]

            if idx_list:
                idx_tensor = torch.tensor(idx_list, device=embed_next.device)
                updates = torch.ones_like(idx_tensor, dtype=embed_next.dtype, device=embed_next.device)
                embed_next[:, 0].index_add_(0, idx_tensor, -updates)
                embed_next[:, 0].clamp_(min=0)

        # -------------------- 5️ 优化的边权 + 同节点惩罚 --------------------
        #  优化边权更新：预过滤top_neighbors
        top_neighbors = [n for n in neighbors_of_chosen_node if n in top_nodes_set]

        if top_neighbors:
            idx_tensor = torch.tensor([goodNode2idx[n] for n in top_neighbors], device=embed_next.device)

            # 批量获取边权重，使用get避免KeyError
            weights = []
            for n in top_neighbors:
                edge_data = graphX.get_edge_data(action, n)
                weight = edge_data.get("weight", 0) if edge_data else 0
                weights.append(weight * node_score[action])

            weights_tensor = torch.tensor(weights, dtype=embed_next.dtype, device=embed_next.device)
            embed_next[idx_tensor, 1] -= weights_tensor
            embed_next[idx_tensor, 1].clamp_(min=0)

        #  优化同节点惩罚：使用缓存
        overlap_cache_key = f"overlap_{action}"
        if not hasattr(self, '_overlap_cache'):
            self._overlap_cache = {}

        if overlap_cache_key not in self._overlap_cache:
            self._overlap_cache[overlap_cache_key] = set(find_overlap_targets(graphX, action)[1])

        same_nodes = self._overlap_cache[overlap_cache_key] & top_nodes_set

        if same_nodes:
            same_idx = [goodNode2idx[n] for n in same_nodes if n in goodNode2idx]
            if same_idx:
                same_idx_tensor = torch.tensor(same_idx, device=embed_next.device)
                embed_next[same_idx_tensor, 2] += 1

        self.embedding_train_archive = embed_next.detach().clone()

        # -------------------- 6️ 延迟归一化 --------------------
        index = list(range(embed_next.shape[1] - 1))
        if step % 5 == 0:
            embedding[step + 1] = normalize_tensor(embed_next, index)
        else:
            embedding[step + 1] = embed_next

        #  内存管理：定期清理缓存防止内存溢出
        if step % 100 == 0:
            if hasattr(self, '_pred_cache') and len(self._pred_cache) > 1000:
                # 保留最近的500个缓存
                items = list(self._pred_cache.items())[-500:]
                self._pred_cache = dict(items)

            if hasattr(self, '_overlap_cache') and len(self._overlap_cache) > 1000:
                items = list(self._overlap_cache.items())[-500:]
                self._overlap_cache = dict(items)

        return previous_spread

    def step(self, action, step, previous_spread, proc):

        import gc

        # -------------------- 1️ 更新状态 --------------------
        self.is_selected[action] = step
        self.state.append(action)

        # 更新 reward
        short_reward, previous_spread = get_short_reward(self.state[:], previous_spread, proc)
        self.rewards.append(short_reward)

        max_history = getattr(self, 'n_step', 5) + 10
        if len(self.state) > max_history:
            self.state = self.state[-max_history:]
            self.rewards = self.rewards[-max_history:]

        # -------------------- 2️ 缓存局部变量 --------------------
        embedding = self.embedding_time
        goodNode2idx = self.goodNode2idx
        top_nodes = self.top_tenpct_nodes
        top_nodes_set = set(top_nodes)  # 转为set提高查找效率
        graphX = self.graphX
        node_score = self.node_score

        max_embedding_steps = getattr(self, 'n_step', 5) + 5
        if len(embedding) > max_embedding_steps:
            # 删除旧的 embedding
            old_keys = [k for k in embedding.keys() if k < step - max_embedding_steps]
            for k in old_keys:
                del embedding[k]
            # 强制垃圾回收
            if len(old_keys) > 0:
                gc.collect()

        # -------------------- 3️ 更新 t+1 embedding --------------------
        if (step + 1) not in embedding:
            embedding[step + 1] = self.embedding_train_archive.clone()
        embed_next = embedding[step + 1]

        # -------------------- 4️ 优化的邻居影响更新 --------------------
        neighbors_of_chosen_node = self.dict_node_sampled_neighbors.get(action, [])
        new_neighbors = set(neighbors_of_chosen_node) - self.neighbors_chosen_till_now

        # 早期退出：如果没有新邻居，跳过后续计算
        if not new_neighbors:
            return previous_spread

        self.neighbors_chosen_till_now.update(neighbors_of_chosen_node)

        # ===== 简化缓存逻辑，使用 LRU 缓存 =====
        if not hasattr(self, '_pred_cache'):
            from functools import lru_cache
            # 使用 LRU 缓存，自动淘汰旧数据
            self._pred_cache = {}

        all_predecessors_in_top = set()
        for new_node in new_neighbors:
            cache_key = new_node  # 直接使用 node 作为 key

            if cache_key not in self._pred_cache:
                predecessors = set(graphX.predecessors(new_node))
                # 只缓存结果，不缓存中间变量
                self._pred_cache[cache_key] = predecessors & top_nodes_set

            all_predecessors_in_top.update(self._pred_cache[cache_key])

        # 批量更新 embedding
        if all_predecessors_in_top:
            idx_list = [goodNode2idx[src] for src in all_predecessors_in_top]

            if idx_list:
                idx_tensor = torch.tensor(idx_list, device=embed_next.device, dtype=torch.long)
                updates = torch.ones(len(idx_list), dtype=embed_next.dtype, device=embed_next.device)
                embed_next[:, 0].index_add_(0, idx_tensor, -updates)
                embed_next[:, 0].clamp_(min=0)

                # ===== 显式删除临时 tensor =====
                del idx_tensor, updates

        # -------------------- 5️ 优化的边权 + 同节点惩罚 --------------------
        top_neighbors = [n for n in neighbors_of_chosen_node if n in top_nodes_set]

        if top_neighbors:
            idx_list = [goodNode2idx[n] for n in top_neighbors]
            idx_tensor = torch.tensor(idx_list, device=embed_next.device, dtype=torch.long)

            weights = []
            for n in top_neighbors:
                edge_data = graphX.get_edge_data(action, n)
                weight = edge_data.get("weight", 0) if edge_data else 0
                weights.append(weight * node_score[action])

            weights_tensor = torch.tensor(weights, dtype=embed_next.dtype, device=embed_next.device)
            embed_next[idx_tensor, 1] -= weights_tensor
            embed_next[idx_tensor, 1].clamp_(min=0)

            # ===== 删除临时 tensor =====
            del idx_tensor, weights_tensor

        # 同节点惩罚
        if not hasattr(self, '_overlap_cache'):
            self._overlap_cache = {}

        overlap_cache_key = action  # 直接使用 action 作为 key

        if overlap_cache_key not in self._overlap_cache:
            self._overlap_cache[overlap_cache_key] = set(find_overlap_targets(graphX, action)[1])

        same_nodes = self._overlap_cache[overlap_cache_key] & top_nodes_set

        if same_nodes:
            same_idx = [goodNode2idx[n] for n in same_nodes if n in goodNode2idx]
            if same_idx:
                same_idx_tensor = torch.tensor(same_idx, device=embed_next.device, dtype=torch.long)
                embed_next[same_idx_tensor, 2] += 1
                del same_idx_tensor

        # ===== 直接更新 embedding_train_archive，避免 clone =====
        # 使用 in-place 操作
        self.embedding_train_archive.copy_(embed_next.detach())

        # -------------------- 6️ 延迟归一化 --------------------
        index = list(range(embed_next.shape[1] - 1))
        if step % 5 == 0:
            embedding[step + 1] = normalize_tensor(embed_next, index)
        else:
            embedding[step + 1] = embed_next

        # ===== 更激进的缓存清理 =====
        # 每 20 步清理一次缓存（原来是 100 步）
        if step % 20 == 0:
            # 限制缓存大小为 200（原来是 500）
            if hasattr(self, '_pred_cache') and len(self._pred_cache) > 200:
                # 只保留最近使用的 100 个
                items = list(self._pred_cache.items())[-100:]
                self._pred_cache = dict(items)

            if hasattr(self, '_overlap_cache') and len(self._overlap_cache) > 200:
                items = list(self._overlap_cache.items())[-100:]
                self._overlap_cache = dict(items)

            # 强制垃圾回收
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return previous_spread

    def get_experience(self, step, n_step, gamma):
        #不需要终止状态的标识符done, 因为我们是倒着组织状态的
        if step < n_step:
            return None
        else:
            #state, n_step_reward, next_n_step_state
            action_t = self.state[step - n_step]
            mu_v = self.embedding_time[step - n_step][self.goodNode2idx[action_t]]
            mu_s, mu_l = self.get_environment_embedding(step - n_step) 
            state = [mu_v, mu_s, mu_l]

            action_next_n_t = self.state[step]
            mu_v = self.embedding_time[step][self.goodNode2idx[action_next_n_t]]
            mu_s, mu_l = self.get_environment_embedding(step - n_step)  
            next_n_step_state = [mu_v, mu_s, mu_l]

            n_step_reward = 0
            for i in range(n_step):
                n_step_reward += (gamma**i) * self.rewards[(step - n_step) + i]
            
            return state, n_step_reward, next_n_step_state



    
    def test_step(self, action, step):
        #更新t时刻相关状态
        self.is_selected[action] = step
        self.state.append(action)

        #更新embedding
        # 缓存局部变量
        embedding = self.embedding_time
        goodNode2idx = self.goodNode2idx
        top_nodes = self.top_tenpct_nodes
        graphX = self.graphX
        node_score = self.node_score

        #更新t+1时刻嵌入
        if (step + 1) not in embedding:
            embedding[step + 1] = self.embedding_test_archive.clone() #device一致

        #仅仅针对新增加的邻居所涉及的节点进行更新，提升速度
        neighbors_of_chosen_node = self.dict_node_sampled_neighbors[action]#无需使用索引
        new_neighbors = set(neighbors_of_chosen_node) - self.neighbors_chosen_till_now
        self.neighbors_chosen_till_now.update(neighbors_of_chosen_node)

        # 用局部 embedding
        embed_next = embedding[step + 1]

        idx_list = []
        for new_node in new_neighbors:
            for src in graphX.predecessors(new_node):
                if src in top_nodes:
                    idx_list.append(goodNode2idx[src])

        if idx_list:
            idx_tensor = torch.tensor(idx_list, device=embed_next.device)
            # 每个 idx 对应 -1
            updates = torch.ones_like(idx_tensor, dtype=embed_next.dtype, device=embed_next.device)
            # index_add_ 默认是加，这里加负数就是减
            embed_next[:, 0].index_add_(0, idx_tensor, -updates)
            embed_next[:, 0].clamp_(min=0)


        #寻找有没有相同节点
        has_same_node, same_nodes = find_overlap_targets(graphX, action) 
        for node in neighbors_of_chosen_node:
            if node in top_nodes:
                idx = goodNode2idx[node]
                edge_data = graphX.get_edge_data(action, node)
                w = edge_data.get("weight", None)
                embed_next[idx, 1] -= w * node_score[action]
                embed_next[idx, 1].clamp_(min=0)
                if node in same_nodes:
                    embed_next[idx, 2] += 1  


        self.embedding_test_archive = embed_next.detach().clone() 

        #没有就直接进行归一化, 注意标识符位不归一化
        index = list(range(embed_next.shape[1] - 1))
        embedding[step + 1] = normalize_tensor(embed_next, index)




    def get_environment_embedding(self, step):
        # 获取设备和嵌入维度（我们只关心前两列）
        device = self.embedding_time[step].device
        dimension = self.embedding_time[step].shape[1] - 1  # 忽略第3列的标志位

        # 取出这些节点在当前 step 的嵌入，并只保留前两列（维度: [N, 2]）
        embeddings = self.embedding_time[step][:, :2] 


        is_selected_list = [0] * len(self.top_tenpct_nodes)  # 初始化为 0（未选中过）
        for node in self.top_tenpct_nodes:
            idx = self.goodNode2idx[node]
            if self.is_selected[node] != -1 and self.is_selected[node] < step:
                is_selected_list[idx] = 1  # 表示这个位置的节点是被选中过的

        # 转为 tensor，在 GPU 上进行掩码构造
        is_selected_tensor = torch.tensor(is_selected_list, dtype=torch.bool, device=device) 
        mask_selected = is_selected_tensor
        mask_left = ~is_selected_tensor

        # 分别取出被选中过的嵌入、未被选中过的嵌入
        selected_embs = embeddings[mask_selected]  
        left_embs = embeddings[mask_left]          

        # 初始化 mu_selected 和 mu_left，默认是 -1e6
        mu_selected = torch.full((dimension,), -1e6, dtype=torch.float32, device=device)
        mu_left = torch.full((dimension,), -1e6, dtype=torch.float32, device=device)

        # 如果有选中过的节点，计算其列最大值
        if selected_embs.shape[0] > 0:
            mu_selected = torch.amax(selected_embs, dim=0)  # 对每一列取最大值
        # 同理对未被选中的节点
        if left_embs.shape[0] > 0:
            mu_left = torch.amax(left_embs, dim=0)

        return mu_selected, mu_left



    def get_topScore_node(self):
        max_node = max(self.node_score, key=self.node_score.get)
        return max_node
    
    def get_random_step0_node(self, k=10):
        sorted_nodes = sorted(self.node_score.items(), key=lambda x: x[1], reverse=True)
        top_k_nodes = [node for node, score in sorted_nodes[:k]]
        return random.choice(top_k_nodes)
    
    def get_random_node(self):
        unselected_nodes = [node for node in self.top_tenpct_nodes if self.is_selected[node] == -1]
        selection = random.choice(unselected_nodes)
        return selection