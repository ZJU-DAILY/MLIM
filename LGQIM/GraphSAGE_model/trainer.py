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
        model.train()
        total_loss = 0 
        for batch in dataloader: 
            batch_nodes, batch_scores = batch 

            batch_nodes = batch_nodes.squeeze(0).to(device)
            batch_scores = batch_scores.squeeze(0).to(device)

            optimizer.zero_grad()

            predictions = model(batch_nodes, features, num_samples).squeeze(-1) 

            predictions = predictions.view(1, -1) #(1, n)
            batch_scores = batch_scores.view(1, -1)
           
            loss = F.mse_loss(predictions, batch_scores)

            loss.backward() 

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{cfg.num_epochs}, Loss: {total_loss:.4f}")


def train_model(model, optimizer, cfg, data, num_samples):
    
    features = data['features']
    score_groundtruth = data['score_groundtruth']  
    train_nodes = data['train_nodes']  # list of indices

    print(f"Training on {len(train_nodes)} nodes out of {features.shape[0]} total nodes")

   
    if isinstance(train_nodes, list):
        train_nodes_tensor = torch.tensor(train_nodes, dtype=torch.long, device=cfg.device)
    else:
        train_nodes_tensor = train_nodes.to(cfg.device)


    train_scores = score_groundtruth[train_nodes_tensor]  # shape [K] where K=len(train_nodes)

    print(f"Train scores range: [{train_scores.min().item():.6f}, {train_scores.max().item():.6f}]")
    print(f"Train scores mean/std: {train_scores.mean().item():.6f}/{train_scores.std().item():.6f}")

    
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
                predictions = model(batch_nodes, features, num_samples)

                
                if predictions.dim() > 1:
                    predictions = predictions.squeeze(-1)  # shape [B]

               
                assert predictions.shape == batch_scores.shape, \
                    f"Shape mismatch: predictions {predictions.shape} vs batch_scores {batch_scores.shape}"

                # MSE loss
                loss = F.mse_loss(predictions, batch_scores)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                total_loss += loss.item() * batch_nodes.size(0) 
                num_batches += 1

            except Exception as e:
                print(f"Error in epoch {epoch + 1}, batch {num_batches + 1}: {e}")
                print(f"batch_nodes shape: {batch_nodes.shape}")
                print(f"batch_scores shape: {batch_scores.shape}")
                if 'predictions' in locals():
                    print(f"predictions shape: {predictions.shape}")
                raise e

        avg_loss = total_loss / len(train_nodes)

       
        print(f"Epoch {epoch + 1:3d}/{cfg.num_epochs}, Loss: {avg_loss:.6f}, "
              f"Batches: {num_batches}")
        # if epoch < 10 or (epoch + 1) % max(1, cfg.num_epochs // 10) == 0:
        #     print(f"Epoch {epoch + 1:3d}/{cfg.num_epochs}, Loss: {avg_loss:.6f}, "
        #           f"Batches: {num_batches}")

       
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

  
    if isinstance(train_nodes, list):
        train_nodes_tensor = torch.tensor(train_nodes, dtype=torch.long, device=cfg.device)
    else:
        train_nodes_tensor = train_nodes.to(cfg.device)


 
    train_scores = score_groundtruth[train_nodes_tensor]
    print(f"Train scores stats:")
    print(f"  Range: [{train_scores.min().item():.6f}, {train_scores.max().item():.6f}]")
    print(f"  Mean/Std: {train_scores.mean().item():.6f}/{train_scores.std().item():.6f}")
    print(f"  Non-zero count: {(train_scores != 0).sum().item()}")


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

   
    base_lr = cfg.learning_rate if hasattr(cfg, 'learning_rate') else 0.001
    lr = min(base_lr, 0.0005)  

  
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

  
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=50,min_lr=1e-6)

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
            batch_scores = batch_scores.to(cfg.device)

            optimizer.zero_grad()

         
            predictions = model(batch_nodes, features, num_samples)
            if predictions.dim() > 1:
                predictions = predictions.squeeze(-1)

            # 
            loss = F.mse_loss(predictions, batch_scores)

            # 
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Invalid loss at epoch {epoch + 1}, batch {num_batches + 1}")
                continue

            epoch_losses.append(loss.item())
            # 
            loss.backward()
            # 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            total_loss += loss.item() * batch_nodes.size(0)
            num_batches += 1

        if num_batches == 0:  # 
            print(f"Epoch {epoch + 1}: All batches had invalid loss, skipping...")
            continue

        avg_loss = total_loss / len(train_nodes)
        scheduler.step(avg_loss) # 

        
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

        
        if (epoch + 1) % max(20, cfg.num_epochs // 10) == 0:
            model.eval()
            with torch.no_grad():
                # 
                sample_size = min(200, len(train_nodes_tensor))
                sample_nodes = train_nodes_tensor[:sample_size]
                sample_preds = model(sample_nodes, features, num_samples)
                if sample_preds.dim() > 1:
                    sample_preds = sample_preds.squeeze(-1)
                sample_targets = score_groundtruth[sample_nodes]

                sample_mse = F.mse_loss(sample_preds, sample_targets)
                sample_mae = F.l1_loss(sample_preds, sample_targets)

                # 
                pred_np = sample_preds.cpu().numpy()
                target_np = sample_targets.cpu().numpy()
                correlation = np.corrcoef(pred_np, target_np)[0, 1] if len(pred_np) > 1 else 0.0

                print(f"    Validation - MSE: {sample_mse.item():.6f}, "
                      f"MAE: {sample_mae.item():.6f}, "
                      f"Corr: {correlation:.4f}")

                
                print(f"    Pred range: [{sample_preds.min().item():.4f}, {sample_preds.max().item():.4f}], "
                      f"Target range: [{sample_targets.min().item():.4f}, {sample_targets.max().item():.4f}]")

            model.train()

    
        if patience_counter >= patience_limit:
            print(f"Early stopping at epoch {epoch + 1} (no improvement for {patience_limit} epochs)")
            break

    print(f"Training completed. Best loss: {best_loss:.6f}")


def improved_clear_train_model(model, optimizer, cfg, data, num_samples):
    
    features = data['features']
    score_groundtruth = data['score_groundtruth']
    train_nodes = data['train_nodes']

    
    if isinstance(train_nodes, list):
        train_nodes_tensor = torch.tensor(train_nodes, dtype=torch.long, device=cfg.device)
    else:
        train_nodes_tensor = train_nodes.to(cfg.device)

    
    optimal_batch_size = min(256, max(64, len(train_nodes) // 8))
    dataset = GraphSAGEDataset(train_nodes_tensor, score_groundtruth)
    dataloader = DataLoader(dataset, batch_size=optimal_batch_size, shuffle=True, drop_last=False)

    
    base_lr = getattr(cfg, 'learning_rate', 0.001)
    lr = min(base_lr, 0.0005)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=50, min_lr=1e-6, verbose=False
    )

    model.train()

    
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
    
    features = data['features']
    score_groundtruth = data['score_groundtruth']
    train_nodes = data['train_nodes']

    print(f"Training on {len(train_nodes)} nodes out of {features.shape[0]} total nodes")


    if isinstance(train_nodes, list):
        train_nodes_tensor = torch.tensor(train_nodes, dtype=torch.long, device=cfg.device)
    else:
        train_nodes_tensor = train_nodes.to(cfg.device)


    y_min = score_groundtruth.min()
    y_max = score_groundtruth.max()
    score_groundtruth_norm = (score_groundtruth - y_min) / (y_max - y_min + 1e-8)  


    train_scores = score_groundtruth_norm[train_nodes_tensor]
    print(f"Train scores stats (normalized):")
    print(f"  Range: [{train_scores.min().item():.6f}, {train_scores.max().item():.6f}]")
    print(f"  Mean/Std: {train_scores.mean().item():.6f}/{train_scores.std().item():.6f}")

    
    #optimal_batch_size = min(256, max(64, len(train_nodes) // 8))  # 至少8个batch
    #optimal_batch_size = min(500, max(200, len(train_nodes) // 4))
    #optimal_batch_size = min(300, max(100, len(train_nodes) // 6))
    optimal_batch_size = min(256, max(64, len(train_nodes) // 8))
    print(f"  Adjusting batch_size to: {optimal_batch_size}")

    dataset = GraphSAGEDataset(train_nodes_tensor, score_groundtruth_norm)
    dataloader = DataLoader(dataset, batch_size=optimal_batch_size,
                            shuffle=True, drop_last=False)
    print(f"Created dataset with {len(dataset)} samples, {len(dataloader)} batches\n")


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
            batch_scores = batch_scores.to(cfg.device)  

            optimizer.zero_grad()

         
            predictions = model(batch_nodes, features, num_samples)
            if predictions.dim() > 1:
                predictions = predictions.squeeze(-1)

         
            lambda_range = 5.0  
            over_max = torch.relu(predictions - 1.0)
            under_min = torch.relu(-predictions)
            range_penalty = lambda_range * torch.mean(over_max**2 + under_min**2)

            loss = F.mse_loss(predictions, batch_scores) + range_penalty

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Invalid loss at epoch {epoch + 1}, batch {num_batches + 1}")
                continue

            epoch_losses.append(loss.item())

            
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

        
        if (epoch + 1) % max(20, cfg.num_epochs // 10) == 0:
            model.eval()
            with torch.no_grad():
                sample_size = min(200, len(train_nodes_tensor))
                sample_nodes = train_nodes_tensor[:sample_size]
                sample_preds = model(sample_nodes, features, num_samples)
                if sample_preds.dim() > 1:
                    sample_preds = sample_preds.squeeze(-1)

                
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
        checkpoint = torch.load(file_path, weights_only=True, map_location=device) 
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"]) 
        epoch = checkpoint.get('epoch', 0) 
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
