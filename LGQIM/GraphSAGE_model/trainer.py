import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
from sklearn.metrics import ndcg_score

from GraphSAGE_model.sage_utils import GraphSAGEDataset
from GraphSAGE_model.model import SupervisedGraphsage
from GraphSAGE_model.loss_function import CombinedListMLE_MSE_Loss

#device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = "cpu"


def train_model_old(model, optimizer, cfg, data, num_samples):
    features, score_groundtruth, train_nodes = (
        data['features'],
        data['score_groundtruth'],
        data['train_nodes']
    )
    features = features
    dataset = GraphSAGEDataset(train_nodes, score_groundtruth)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True, drop_last=False)

    for epoch in range(cfg.num_epochs):
        model.train() # 设置模型为训练模式
        total_loss = 0 
        for batch in dataloader: 
            batch_nodes, batch_scores = batch 

            batch_nodes = batch_nodes.squeeze(0).to(device)
            batch_scores = batch_scores.squeeze(0).to(device)

            optimizer.zero_grad() # 梯度清零

            predictions = model(batch_nodes, features, num_samples).squeeze(-1) #去掉最后一维度

            predictions = predictions.view(1, -1) #(1, n)
            batch_scores = batch_scores.view(1, -1)
            #到这里，修改loss即可
            loss = F.mse_loss(predictions, batch_scores)

            loss.backward() # 反向传播

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)#梯度裁剪

            optimizer.step() # 更新模型参数
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{cfg.num_epochs}, Loss: {total_loss:.4f}")


def train_model(model, optimizer, cfg, data, num_samples):
    """
    训练 GraphSAGE 模型
    data:
        features: 节点特征 [N, F]
        score_groundtruth: torch.Tensor, 标准化后的真实分数 [N] (只有top节点有非零值)
        train_nodes: list, 训练节点索引（如 top 30%的节点索引列表）
    """
    features = data['features']
    score_groundtruth = data['score_groundtruth']  # 已标准化，shape [N]
    train_nodes = data['train_nodes']  # list of indices

    print(f"Training on {len(train_nodes)} nodes out of {features.shape[0]} total nodes")

    # 将train_nodes转换为tensor（如果还不是的话）
    if isinstance(train_nodes, list):
        train_nodes_tensor = torch.tensor(train_nodes, dtype=torch.long, device=cfg.device)
    else:
        train_nodes_tensor = train_nodes.to(cfg.device)

    # 提取训练节点对应的标准化分数
    train_scores = score_groundtruth[train_nodes_tensor]  # shape [K] where K=len(train_nodes)

    print(f"Train scores range: [{train_scores.min().item():.6f}, {train_scores.max().item():.6f}]")
    print(f"Train scores mean/std: {train_scores.mean().item():.6f}/{train_scores.std().item():.6f}")

    # 创建数据集和数据加载器
    #dataset = GraphSAGEDataset(train_nodes_tensor, train_scores)
    dataset = GraphSAGEDataset(train_nodes_tensor, score_groundtruth)

    dataloader = DataLoader(dataset, batch_size=min(512, len(train_nodes)),
                            shuffle=True, drop_last=False)

    print(f"Created dataset with {len(dataset)} samples, batch_size={dataloader.batch_size}")
    print()

    model.train()

    for epoch in range(cfg.num_epochs):
        total_loss = 0.0
        num_batches = 0

        for batch_nodes, batch_scores in dataloader:
            batch_nodes = batch_nodes.to(cfg.device)
            batch_scores = batch_scores.to(cfg.device)

            optimizer.zero_grad()

            try:
                # GraphSAGE 前向计算
                predictions = model(batch_nodes, features, num_samples)

                # 确保predictions是1D张量
                if predictions.dim() > 1:
                    predictions = predictions.squeeze(-1)  # shape [B]

                # 确保维度匹配
                assert predictions.shape == batch_scores.shape, \
                    f"Shape mismatch: predictions {predictions.shape} vs batch_scores {batch_scores.shape}"

                # MSE loss
                loss = F.mse_loss(predictions, batch_scores)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                total_loss += loss.item() * batch_nodes.size(0)  # 按样本数累计
                num_batches += 1

            except Exception as e:
                print(f"Error in epoch {epoch + 1}, batch {num_batches + 1}: {e}")
                print(f"batch_nodes shape: {batch_nodes.shape}")
                print(f"batch_scores shape: {batch_scores.shape}")
                if 'predictions' in locals():
                    print(f"predictions shape: {predictions.shape}")
                raise e

        avg_loss = total_loss / len(train_nodes)

        # 更频繁的输出在前期，后期减少
        print(f"Epoch {epoch + 1:3d}/{cfg.num_epochs}, Loss: {avg_loss:.6f}, "
              f"Batches: {num_batches}")
        # if epoch < 10 or (epoch + 1) % max(1, cfg.num_epochs // 10) == 0:
        #     print(f"Epoch {epoch + 1:3d}/{cfg.num_epochs}, Loss: {avg_loss:.6f}, "
        #           f"Batches: {num_batches}")

        # 早期验证（可选）
        if (epoch + 1) % max(10, cfg.num_epochs // 5) == 0:
            model.eval()
            with torch.no_grad():
                # 在训练集上做一次预测，检查过拟合
                sample_nodes = train_nodes_tensor[:min(100, len(train_nodes_tensor))]
                sample_preds = model(sample_nodes, features, num_samples)
                if sample_preds.dim() > 1:
                    sample_preds = sample_preds.squeeze(-1)
                sample_targets = score_groundtruth[sample_nodes]

                sample_mse = F.mse_loss(sample_preds, sample_targets)
                sample_mae = F.l1_loss(sample_preds, sample_targets)

                print(f"    Sample validation - MSE: {sample_mse.item():.6f}, "
                      f"MAE: {sample_mae.item():.6f}")
            model.train()

    print(f"Training completed. Final average loss: {avg_loss:.6f}")
    print()


def improved_train_model(model, optimizer, cfg, data, num_samples):
    """
    改进的训练函数，解决loss波动问题
    """
    features = data['features']
    score_groundtruth = data['score_groundtruth']
    train_nodes = data['train_nodes']

    print(f"Training on {len(train_nodes)} nodes out of {features.shape[0]} total nodes")

    # 转换数据类型
    if isinstance(train_nodes, list):
        train_nodes_tensor = torch.tensor(train_nodes, dtype=torch.long, device=cfg.device)
    else:
        train_nodes_tensor = train_nodes.to(cfg.device)


    # 分析训练数据
    train_scores = score_groundtruth[train_nodes_tensor]
    print(f"Train scores stats:")
    print(f"  Range: [{train_scores.min().item():.6f}, {train_scores.max().item():.6f}]")
    print(f"  Mean/Std: {train_scores.mean().item():.6f}/{train_scores.std().item():.6f}")
    print(f"  Non-zero count: {(train_scores != 0).sum().item()}")

    # 检查是否有异常值（极端值）
    # q95 = torch.quantile(torch.abs(train_scores), 0.95)
    # q99 = torch.quantile(torch.abs(train_scores), 0.99)
    # print(f"  95th percentile |score|: {q95.item():.6f}")
    # print(f"  99th percentile |score|: {q99.item():.6f}")

    # 调整批次大小，确保足够大以获得稳定的梯度估计
    optimal_batch_size = min(256, max(64, len(train_nodes) // 8))   # 至少8个batch
    print(f"  Adjusting batch_size to: {optimal_batch_size}")

    dataset = GraphSAGEDataset(train_nodes_tensor, score_groundtruth)
    dataloader = DataLoader(dataset, batch_size=optimal_batch_size,
                            shuffle=True, drop_last=False)

    print(f"Created dataset with {len(dataset)} samples, {len(dataloader)} batches")
    print()

    # 1. 设置初始学习率（保守起步）
    base_lr = cfg.learning_rate if hasattr(cfg, 'learning_rate') else 0.001
    lr = min(base_lr, 0.0005)  # 如果给的太高，就限制到 0.0005

    # 2. 初始化优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 3. 添加学习率调度器（自动根据验证集 loss 降低 lr）
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=50,min_lr=1e-6)

    model.train()

    # 用于记录训练历史
    best_loss = float('inf')
    patience_counter = 0
    patience_limit = 200  # 如果200个epoch没改善就停止

    for epoch in range(cfg.num_epochs):
        epoch_losses = []
        total_loss = 0.0
        num_batches = 0

        for batch_nodes, batch_scores in dataloader:
            batch_nodes = batch_nodes.to(cfg.device)
            batch_scores = batch_scores.to(cfg.device)

            optimizer.zero_grad()

            # 前向传播
            predictions = model(batch_nodes, features, num_samples)
            if predictions.dim() > 1:
                predictions = predictions.squeeze(-1)

            # 计算loss
            loss = F.mse_loss(predictions, batch_scores)

            # 检查loss是否异常
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Invalid loss at epoch {epoch + 1}, batch {num_batches + 1}")
                continue

            epoch_losses.append(loss.item())
            # 反向传播
            loss.backward()
            # 更保守的梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            total_loss += loss.item() * batch_nodes.size(0)
            num_batches += 1

        if num_batches == 0:  # 如果所有batch都有问题
            print(f"Epoch {epoch + 1}: All batches had invalid loss, skipping...")
            continue

        avg_loss = total_loss / len(train_nodes)
        scheduler.step(avg_loss) # 学习率调度

        # 早停检查
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1

        # 输出训练信息
        if epoch < 20 or (epoch + 1) % max(1, cfg.num_epochs // 20) == 0:
            batch_loss_std = np.std(epoch_losses) if epoch_losses else 0
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch + 1:4d}/{cfg.num_epochs}, "
                  f"Loss: {avg_loss:.6f} (±{batch_loss_std:.6f}), "
                  f"LR: {current_lr:.2e}, "
                  f"Batches: {num_batches}")

        # 详细验证
        if (epoch + 1) % max(20, cfg.num_epochs // 10) == 0:
            model.eval()
            with torch.no_grad():
                # 在训练集上评估
                sample_size = min(200, len(train_nodes_tensor))
                sample_nodes = train_nodes_tensor[:sample_size]
                sample_preds = model(sample_nodes, features, num_samples)
                if sample_preds.dim() > 1:
                    sample_preds = sample_preds.squeeze(-1)
                sample_targets = score_groundtruth[sample_nodes]

                sample_mse = F.mse_loss(sample_preds, sample_targets)
                sample_mae = F.l1_loss(sample_preds, sample_targets)

                # 计算相关系数
                pred_np = sample_preds.cpu().numpy()
                target_np = sample_targets.cpu().numpy()
                correlation = np.corrcoef(pred_np, target_np)[0, 1] if len(pred_np) > 1 else 0.0

                print(f"    Validation - MSE: {sample_mse.item():.6f}, "
                      f"MAE: {sample_mae.item():.6f}, "
                      f"Corr: {correlation:.4f}")

                # 显示预测范围
                print(f"    Pred range: [{sample_preds.min().item():.4f}, {sample_preds.max().item():.4f}], "
                      f"Target range: [{sample_targets.min().item():.4f}, {sample_targets.max().item():.4f}]")

            model.train()

        # 早停
        if patience_counter >= patience_limit:
            print(f"Early stopping at epoch {epoch + 1} (no improvement for {patience_limit} epochs)")
            break

    print(f"Training completed. Best loss: {best_loss:.6f}")


def improved_clear_train_model(model, optimizer, cfg, data, num_samples):
    """
    精简版训练函数，删除所有输出信息
    """
    features = data['features']
    score_groundtruth = data['score_groundtruth']
    train_nodes = data['train_nodes']

    # 转换数据类型
    if isinstance(train_nodes, list):
        train_nodes_tensor = torch.tensor(train_nodes, dtype=torch.long, device=cfg.device)
    else:
        train_nodes_tensor = train_nodes.to(cfg.device)

    # 调整批次大小
    optimal_batch_size = min(256, max(64, len(train_nodes) // 8))
    dataset = GraphSAGEDataset(train_nodes_tensor, score_groundtruth)
    dataloader = DataLoader(dataset, batch_size=optimal_batch_size, shuffle=True, drop_last=False)

    # 设置学习率和优化器
    base_lr = getattr(cfg, 'learning_rate', 0.001)
    lr = min(base_lr, 0.0005)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=50, min_lr=1e-6, verbose=False
    )

    model.train()

    # 早停机制
    best_loss = float('inf')
    patience_counter = 0
    patience_limit = 200

    for epoch in range(cfg.num_epochs):
        total_loss = 0.0
        valid_batches = 0

        for batch_nodes, batch_scores in dataloader:
            batch_nodes = batch_nodes.to(cfg.device)
            batch_scores = batch_scores.to(cfg.device)

            optimizer.zero_grad()
            predictions = model(batch_nodes, features, num_samples)

            if predictions.dim() > 1:
                predictions = predictions.squeeze(-1)

            loss = F.mse_loss(predictions, batch_scores)

            # 跳过无效loss
            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            total_loss += loss.item() * batch_nodes.size(0)
            valid_batches += 1

        if valid_batches == 0:
            continue

        avg_loss = total_loss / len(train_nodes)
        scheduler.step(avg_loss)

        # 早停检查
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience_limit:
            break

    return best_loss


#############################################################################################
def improved_train_model_temp(model, optimizer, cfg, data, num_samples):
    """
    改进的训练函数：
    1. 目标归一化（稳定训练）
    2. 预测范围惩罚（控制预测不偏高/偏低）
    3. 保留早停、梯度裁剪等机制
    """
    features = data['features']
    score_groundtruth = data['score_groundtruth']
    train_nodes = data['train_nodes']

    print(f"Training on {len(train_nodes)} nodes out of {features.shape[0]} total nodes")

    # 转换训练节点格式
    if isinstance(train_nodes, list):
        train_nodes_tensor = torch.tensor(train_nodes, dtype=torch.long, device=cfg.device)
    else:
        train_nodes_tensor = train_nodes.to(cfg.device)

    # ======== 目标归一化 ========
    y_min = score_groundtruth.min()
    y_max = score_groundtruth.max()
    score_groundtruth_norm = (score_groundtruth - y_min) / (y_max - y_min + 1e-8)  # 防除零

    # 打印归一化后的训练集统计
    train_scores = score_groundtruth_norm[train_nodes_tensor]
    print(f"Train scores stats (normalized):")
    print(f"  Range: [{train_scores.min().item():.6f}, {train_scores.max().item():.6f}]")
    print(f"  Mean/Std: {train_scores.mean().item():.6f}/{train_scores.std().item():.6f}")

    # 批次大小
    #optimal_batch_size = min(256, max(64, len(train_nodes) // 8))  # 至少8个batch
    #optimal_batch_size = min(500, max(200, len(train_nodes) // 4))
    #optimal_batch_size = min(300, max(100, len(train_nodes) // 6))
    optimal_batch_size = min(256, max(64, len(train_nodes) // 8))
    print(f"  Adjusting batch_size to: {optimal_batch_size}")

    dataset = GraphSAGEDataset(train_nodes_tensor, score_groundtruth_norm)
    dataloader = DataLoader(dataset, batch_size=optimal_batch_size,
                            shuffle=True, drop_last=False)
    print(f"Created dataset with {len(dataset)} samples, {len(dataloader)} batches\n")

    # 学习率设置
    lr = getattr(cfg, 'learning_rate', 0.001)
    suggested_lr = min(lr, 0.0005)
    if suggested_lr < lr:
        print(f"降低学习率从 {lr} 到 {suggested_lr}")
        for param_group in optimizer.param_groups:
            param_group['lr'] = suggested_lr

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=50,
        min_lr=1e-6, verbose=True
    )

    model.train()

    best_loss = float('inf')
    patience_counter = 0
    patience_limit = 200

    for epoch in range(cfg.num_epochs):
        epoch_losses = []
        total_loss = 0.0
        num_batches = 0

        for batch_nodes, batch_scores in dataloader:
            batch_nodes = batch_nodes.to(cfg.device)
            batch_scores = batch_scores.to(cfg.device)  # 已经是归一化的

            optimizer.zero_grad()

            # 前向传播
            predictions = model(batch_nodes, features, num_samples)
            if predictions.dim() > 1:
                predictions = predictions.squeeze(-1)

            # ======== MSE + 预测范围惩罚 ========
            lambda_range = 5.0  # 惩罚系数
            over_max = torch.relu(predictions - 1.0)
            under_min = torch.relu(-predictions)
            range_penalty = lambda_range * torch.mean(over_max**2 + under_min**2)

            loss = F.mse_loss(predictions, batch_scores) + range_penalty

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Invalid loss at epoch {epoch + 1}, batch {num_batches + 1}")
                continue

            epoch_losses.append(loss.item())

            # 反向传播
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            total_loss += loss.item() * batch_nodes.size(0)
            num_batches += 1

        if num_batches == 0:
            print(f"Epoch {epoch + 1}: All batches invalid, skipping...")
            continue

        avg_loss = total_loss / len(train_nodes)
        scheduler.step(avg_loss)

        # 早停逻辑
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch < 20 or (epoch + 1) % max(1, cfg.num_epochs // 20) == 0:
            batch_loss_std = np.std(epoch_losses) if epoch_losses else 0
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch + 1:4d}/{cfg.num_epochs}, "
                  f"Loss: {avg_loss:.6f} (±{batch_loss_std:.6f}), "
                  f"LR: {current_lr:.2e}, "
                  f"Batches: {num_batches}")

        # 验证
        if (epoch + 1) % max(20, cfg.num_epochs // 10) == 0:
            model.eval()
            with torch.no_grad():
                sample_size = min(200, len(train_nodes_tensor))
                sample_nodes = train_nodes_tensor[:sample_size]
                sample_preds = model(sample_nodes, features, num_samples)
                if sample_preds.dim() > 1:
                    sample_preds = sample_preds.squeeze(-1)

                # 反归一化
                sample_preds = sample_preds * (y_max - y_min) + y_min
                sample_targets = score_groundtruth[sample_nodes]

                sample_mse = F.mse_loss(sample_preds, sample_targets)
                sample_mae = F.l1_loss(sample_preds, sample_targets)
                correlation = np.corrcoef(
                    sample_preds.cpu().numpy(),
                    sample_targets.cpu().numpy()
                )[0, 1] if sample_size > 1 else 0.0

                print(f"    Validation - MSE: {sample_mse.item():.6f}, "
                      f"MAE: {sample_mae.item():.6f}, "
                      f"Corr: {correlation:.4f}")
                print(f"    Pred range: [{sample_preds.min().item():.4f}, {sample_preds.max().item():.4f}], "
                      f"Target range: [{sample_targets.min().item():.4f}, {sample_targets.max().item():.4f}]")
            model.train()

        if patience_counter >= patience_limit:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    print(f"Training completed. Best loss: {best_loss:.6f}")



def save_model(model, optimizer, epoch, file_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, file_path)
    print(f'Model saved at epoch {epoch}.')

def load_model(model, optimizer, file_path, device):
    epoch = 0
    if os.path.exists(file_path):
        checkpoint = torch.load(file_path, weights_only=True, map_location=device) # 加载checkpoint,包含保存的模型信息
        model.load_state_dict(checkpoint["model_state_dict"]) # 加载模型参数，其中包括模型参数的名称和对应的张量值
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"]) # 加载优化器参数，例如动量、学习率等
        epoch = checkpoint.get('epoch', 0) # 加载epoch
        print("load model success")
    else:
        print("load model failed")
    return epoch

def predict(model, inputs, features):
    num_samples = [5,5]
    model.eval()
    with torch.no_grad():
        predictions = model(inputs, features.to(torch.device(device)), num_samples)
    return predictions