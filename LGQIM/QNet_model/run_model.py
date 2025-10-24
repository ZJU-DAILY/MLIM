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

graph_path = "/root/GNN_QNet/data/"

#main.py按照强化学习的框架
#TODO:优先经验回放，保存模型，加载模型以继续训练
def main():
    # 初始化配置
    _, qnet_config = load_config()
    print(qnet_config.DATASET)

    dataset_dir = os.path.join("/root/GNN_QNet/data", qnet_config.DATASET)
    proc = subprocess.Popen(
        ['./rr_server', f'-dir={dataset_dir}'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    dataset_path = os.path.join(graph_path, qnet_config.DATASET)

    graph_env = GraphEnvironment(qnet_config, dataset_path)
    agent = Agent(qnet_config)
    path = "/root/GNN_QNet/src/QNet_model/saved_model/" + qnet_config.DATASET + "_3_NeighborMinusScore_model.pth"
    load_model(agent, path)

    test(qnet_config, graph_env, agent, proc)

    # 训练结束，关闭 C++ 子进程
    proc.stdin.close()
    proc.terminate()
    proc.wait()

if __name__ == "__main__":
    main()