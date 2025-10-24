import gc

from utils_dqn import calculate_spread
import time
import torch
import os

def train(cfg, env, agent, proc):
    print('begin training...', flush=True)
    total_start_time = time.perf_counter()
    interval_start_time = time.perf_counter()  # 每5回合计时器
    for episode in range(cfg.GAME_EPISODES):
        #开始一轮游戏
        env.reset()  # 重置环境，返回初始状态
        previous_spread = 0
        for step in range(cfg.K):
            action = agent.select_action(env, step)  # 选择动作

            previous_spread = env.step(action, step, previous_spread, proc)

            experience = env.get_experience(step, cfg.N_STEP, cfg.GAMMA)
            if experience is not None:
                state, n_step_reward, next_n_step_state = experience
                if len(agent.replay_buffer) >= cfg.REPLAY_BUFFER_SIZE:
                    agent.replay_buffer.pop(0)  # 移除最旧的经验
                agent.replay_buffer.push((state, n_step_reward, next_n_step_state))  # 保存transition
            
            # agent.update(cfg)  # 更新智能体
            if step % 5 == 0:  # 批量更新智能体
                agent.update(cfg)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if (episode + 1) % 5 == 0:
            interval_end_time = time.perf_counter()
            print(f"round：{episode+1}/{cfg.GAME_EPISODES}，reward：{previous_spread:.2f}，Epsilon：{agent.epsilon:.3f}，time of recent 5 rounds：{interval_end_time - interval_start_time:.2f} s",flush=True)
            interval_start_time = time.perf_counter()  # 重置5回合计时器
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    total_end_time = time.perf_counter()
    training_time = total_end_time - total_start_time
    print(f"training time：{training_time:.2f} s")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "saved_model")
    model_name = cfg.DATASET + "_DQN_model.pth"  # 一个节点被选之后，邻居的 score-=被选节点score * weight (被选->邻居)
    #model_name = f"{cfg.DATASET}_{cfg.ratio}_DQN_model.pth" # 测para
    model_path = os.path.join(base_dir, model_name)
    save_model(agent, cfg.GAME_EPISODES, path=model_path)

    # write training ouput
    if 1:
        parent_dir = os.path.dirname(script_dir)
        dqn_dir = os.path.join(parent_dir, "dqn_perf")
        perf_output_path = os.path.join(dqn_dir, f"training-{cfg.DATASET}-k{cfg.K}.txt")
        os.makedirs(os.path.dirname(perf_output_path), exist_ok=True)
        with open(perf_output_path, 'w') as f:
            f.write("# Training Performance Metrics\n")
            f.write(f"Dataset: {cfg.DATASET}\n")
            f.write(f"K: {cfg.K}\n")
            f.write(f"Training Time (seconds): {training_time:.4f}\n")

        print(f"training performance saved to: {perf_output_path}")
        print()




#预测种子集并输出影响力
def test(cfg, env, agent, proc):
    #TODO:可以一次性预测出来选择前10个
    spread = calculate_spread([(1,1)], proc)
    seed_set = []
    env.reset()
    max_node = env.get_topScore_node()
    total_start_time = time.perf_counter()
    for step in range(0, cfg.K):
        if(step==0):
            selected_node = max_node #可以尝试一下直接预测
        else:
            selected_node = agent.predict_action(env, step)
        env.test_step(selected_node, step)
        seed_set.append(selected_node)   

    total_end_time = time.perf_counter()
    spread = calculate_spread(seed_set, proc)
    print(f"Prediction time：{total_end_time - total_start_time:.2f} s")
    print(seed_set)
    print(spread)

    # write testing ouput
    if 1:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        dqn_dir = os.path.join(parent_dir, "dqn_perf")
        perf_test_output_path = os.path.join(dqn_dir, f"testing-{cfg.DATASET}-k{cfg.K}-pred.txt")
        os.makedirs(os.path.dirname(perf_test_output_path), exist_ok=True)
        with open(perf_test_output_path, 'w') as f:
            f.write("# Testing Performance Metrics\n")
            f.write(f"Dataset: {cfg.DATASET}\n")
            f.write(f"K: {cfg.K}\n")
            f.write(" ".join(str(seed) for seed in seed_set) + "\n")
            f.write(f"spread: {spread}\n")
            f.write(f"Testing Time (seconds): {total_end_time - total_start_time:.4f}\n")

        print(f"testing performance saved to: {perf_test_output_path}")
        print()


def save_model(agent, episode, path='checkpoint.pth'):
    torch.save({
        'episode': episode,
        'q_network_state_dict': agent.q_network.state_dict(),
        'target_network_state_dict': agent.target_network.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'epsilon': agent.epsilon,
        'total_steps': agent.total_steps,
    }, path)
    print(f"model have saved：{path}")


def load_model(agent, path):
    checkpoint = torch.load(path, map_location=agent.device, weights_only=True)
    agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
    agent.target_network.load_state_dict(checkpoint['target_network_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    agent.epsilon = checkpoint.get('epsilon', agent.epsilon)
    agent.total_steps = checkpoint.get('total_steps', 0)
    episode = checkpoint.get('episode', 0)
    print(f"model have loaded：continue training from episode {episode}")
    return episode

