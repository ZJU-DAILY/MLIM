import sys
import os
import subprocess

# 获取当前脚本的绝对路径，并向上找到 src 目录
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(src_dir)

from configs.config_loader import load_config
from environment import GraphEnvironment
from agent import Agent
from QNet_trainer import train, test, load_model

graph_path = "../process_dataset/dataset/"

#main.py按照强化学习的框架
#TODO:优先经验回放，保存模型，加载模型以继续训练
def main():
    # 初始化配置
    _, qnet_config = load_config()
    print(f"Dataset: {qnet_config.DATASET}, "
          f"gts: {qnet_config.gts}, "
          f"K: {qnet_config.K}, "
          f"R: {qnet_config.R}")

    use_gts = qnet_config.gts
    dataset_path = os.path.join("../process_dataset/dataset/", qnet_config.DATASET)
    proc = subprocess.Popen(
        ['/home/cxq/lrz/GNN-QNet-20250808/QNet_model/rr_server', f'-dir={dataset_path}'],  # 输入种子节点集合, 用rr_server计算节点的socre
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    R = qnet_config.R
    pred_path = os.path.join(script_dir, "../pred_score", f"{qnet_config.DATASET}-{R}-predict-score.txt")

    graph_env = GraphEnvironment(qnet_config, dataset_path, pred_path, use_gts)
    agent = Agent(qnet_config)

    model_name = f"{qnet_config.DATASET}_DQN_model.pth"
    model_path = os.path.join(script_dir, "saved_model", model_name)
    load_model(agent, model_path)

    test(qnet_config, graph_env, agent, proc)

    # 训练结束，关闭 C++ 子进程
    proc.stdin.close()
    proc.terminate()
    proc.wait()

if __name__ == "__main__":
    main()