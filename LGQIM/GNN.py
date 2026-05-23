import os
import torch
import time
import torch.optim as optim
import torch.nn.functional as F
from sympy.codegen import Print

from configs.config_loader import load_config
from utils.utils import load_graph, print_graph_node_info
from GraphSAGE_model.model import SupervisedGraphsage
from GraphSAGE_model.trainer import load_model, train_model, save_model, predict, improved_train_model
from GraphSAGE_model.sage_utils import build_node_features, construct_adj, select_top_nodes, load_scores_to_tensor
from utils import *


def main():
    import faulthandler  
    faulthandler.enable()

    
    graph_sage_config, _ = load_config()

   
    base_dir = os.path.join("../process_dataset/dataset/", graph_sage_config.dataset)
    G = load_graph(base_dir)
    print(graph_sage_config.dataset)
    print_graph_node_info(G)

    # features, score, adj
    features, node2idx, idx2node = build_node_features(G, graph_sage_config)
    adj = construct_adj(G, graph_sage_config.max_degree, node2idx).to(graph_sage_config.device)

    score_file = os.path.join(base_dir, "node_score.txt")

    
    p=graph_sage_config.ratio
    score, top_idx, mean, std = load_scores_to_tensor(
        score_file, node2idx, graph_sage_config,
        top_ratio=p, normalize=True, use_log=False)
        #top_ratio=0.10, normalize=True, use_log=False)

  
    print("Num nodes:", score.shape[0])
    print("Top ratio nodes:", None if top_idx is None else top_idx.shape[0])
    print()
    print("score mean/std (used for normalization):",
          mean.item() if mean is not None else None,
          std.item() if std is not None else None)
    print()


    score_norm = torch.zeros_like(score)
    if top_idx is not None:
         score_norm[top_idx] = (score[top_idx] - mean) / (std + 1e-8)


    V_g = [node2idx[node] for node in G.nodes()] 
    num_samples = [graph_sage_config.samples_1, graph_sage_config.samples_2]

   
    train_nodes = top_idx.tolist() if top_idx is not None else V_g

    data = {
        'features': features,
        'score_groundtruth': score_norm,  
        'train_nodes': train_nodes  
    }

    
    model = SupervisedGraphsage(adj, input_dim=features.shape[1],
                                embedding_dim=graph_sage_config.embedding_dim,
                                final_output_dim=graph_sage_config.output_dim).to(graph_sage_config.device)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

 
    print("load model...")
    #model_path = ("result_GNNmodel/" + "GNN_model-" + graph_sage_config.dataset + "-" + str(graph_sage_config.num_epochs)+ "-"
    #              + str(graph_sage_config.embedding_dim) + "-" + str(graph_sage_config.ratio)  + ".pth")
    model_path = "result_GNNmodel/" + "GNN_model-" + graph_sage_config.dataset + "-" + str(graph_sage_config.num_epochs) + ".pth"
    epoch = load_model(model, optimizer, model_path, graph_sage_config.device)
    print("load model end.")
    print()

   
    training_time = 0.0
    prediction_time = 0.0
    spearman_corr = 0.0
    pearson_corr = 0.0

 
    if epoch == 0: #1:  # epoch == 0:
        
        print("start training...")
        start_time = time.perf_counter()
        improved_train_model(model, optimizer, graph_sage_config, data, num_samples)
        save_model(model, optimizer, epoch + graph_sage_config.num_epochs, model_path)
        end_time = time.perf_counter()
        training_time = end_time - start_time
        print(f"training model time: {training_time:.4f} s")
        print()
    else:
        print(f"model is loaded，trained in {epoch} rounds，skip training，directly predicting")
        print()

    print("start predicting...")
    start_time = time.perf_counter()

   
    predict_nodes = train_nodes 
    predict_score_norm = predict(model, predict_nodes, features)  

    predict_score = predict_score_norm * (std + 1e-8) + mean

    end_time = time.perf_counter()
    prediction_time = end_time - start_time
    print(f"predicting time: {prediction_time:.4f} s")
    print()


#########################################################################
  
    if top_idx is not None:
        actual_top = score[top_idx]

      
        predict_score = predict_score.view(-1)
        actual_top = actual_top.view(-1)

        print("Top X0% nodes evaluation:")
        print("Predicted scores (first 10):", predict_score[:10])
        print("Actual scores (first 10):", actual_top[:10])
        print()

        mse_loss = F.mse_loss(predict_score, actual_top)
        mae_loss = F.l1_loss(predict_score, actual_top)
        print(f'MSE Loss (top X0%): {mse_loss.item():.6f}')
        print(f'MAE Loss (top X0%): {mae_loss.item():.6f}')
        print()

   
    sorted_indices = torch.argsort(predict_score, descending=True)
    top10_predicted_indices = sorted_indices[:10]

    print("Top 10 predicted nodes (within top 10%):")
    for i, sort_idx in enumerate(top10_predicted_indices):
        original_idx = top_idx[sort_idx] 
        layer, node = idx2node[original_idx.item()]
        pred_score = predict_score[sort_idx].item()
        actual_score = score[original_idx].item()
        print(f"Rank {i + 1}: ({layer},{node}) Predicted: {pred_score:.6f}, Actual: {actual_score:.6f}")
    print()

  
    #output_path = f"pred_score_p/{graph_sage_config.dataset}-{graph_sage_config.num_epochs}-{graph_sage_config.embedding_dim}-{graph_sage_config.ratio}-predict-score.txt"
    output_path = f"pred_score/{graph_sage_config.dataset}-{graph_sage_config.num_epochs}-predict-score.txt"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        for sort_idx in sorted_indices:
            original_idx = top_idx[sort_idx] 
            layer, node = idx2node[original_idx.item()]
            pred_score = predict_score[sort_idx].item()
            f.write(f"({layer},{node}) {pred_score:.2f}\n")

    print(f"Results saved to: {output_path}")
    print()

 

    if top_idx is not None:
        
        pred_ranks = torch.argsort(torch.argsort(predict_score, descending=True))
        actual_ranks = torch.argsort(torch.argsort(actual_top, descending=True))

 
        pred_ranks_np = pred_ranks.cpu().numpy()
        actual_ranks_np = actual_ranks.cpu().numpy()

        from scipy.stats import spearmanr, pearsonr
        spearman_corr, spearman_p = spearmanr(pred_ranks_np, actual_ranks_np)
        pearson_corr, pearson_p = pearsonr(predict_score.cpu().numpy(), actual_top.cpu().numpy())

        print(f"Spearman correlation (ranking): {spearman_corr:.4f} (p={spearman_p:.4f})")
        print(f"Pearson correlation (values): {pearson_corr:.4f} (p={pearson_p:.4f})")

 
    perf_output_path = f"pred_perf/{graph_sage_config.dataset}-r{graph_sage_config.num_epochs}.txt"
    #perf_output_path = f"pred_perf_p/{graph_sage_config.dataset}-r{graph_sage_config.num_epochs}-e{graph_sage_config.embedding_dim}-p{graph_sage_config.ratio}.txt"
    os.makedirs(os.path.dirname(perf_output_path), exist_ok=True)

    with open(perf_output_path, 'w') as f:
        f.write("# Performance Metrics\n")
        f.write(f"Dataset: {graph_sage_config.dataset}\n")
        f.write(f"Epochs: {graph_sage_config.num_epochs}\n")
        f.write(f"Training Time (seconds): {training_time:.4f}\n")
        f.write(f"Prediction Time (seconds): {prediction_time:.4f}\n")
        f.write(f"Spearman Correlation: {spearman_corr:.6f}\n")
        f.write(f"Pearson Correlation: {pearson_corr:.6f}\n")

        if top_idx is not None:
            f.write(f"MSE Loss: {mse_loss.item():.6f}\n")
            f.write(f"MAE Loss: {mae_loss.item():.6f}\n")
            f.write(f"Number of training nodes: {len(train_nodes)}\n")
            f.write(f"Total nodes: {score.shape[0]}\n")

    print(f"Performance metrics saved to: {perf_output_path}")
    print()

if __name__ == "__main__":

    main()

