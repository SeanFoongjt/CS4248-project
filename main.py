import os
import pandas as pd
import torch
import logging
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
import copy

# Import refactored modules
import logger_setup
import global_state
from embedder import TextEmbedder
from graph import build_graph_from_title
from model import SarcasmGNN, calculate_irf_weights, semantic_contrastive_loss
from evaluate import evaluate
from visualise import save_gnn_graph, track_and_log_weights, plot_weight_trajectories, plot_training_metrics, weight_history

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = torch.nn.BCELoss(reduction='none') # Don't average immediately

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        # Calculate p_t (probability of the true class)
        pt = torch.exp(-bce_loss) 
        # Apply the focal modulating factor
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

if __name__ == "__main__":
    logger_setup.setup_logger()
    logging.info("Loading dataset...")
    
    try:
        df = pd.read_json(r"data\Sarcasm_Headlines_Dataset_v2.json", lines=True)
    except FileNotFoundError:
        logging.info("Dataset not found. Please ensure 'Sarcasm_Headlines_Dataset_v2.json' is in the directory.")
        exit()
    
    df_subset = df.sample(1000, random_state=42) 
    train_df, test_df = train_test_split(df_subset, test_size=0.2, random_state=42, stratify=df_subset['is_sarcastic'])
    
    embedder = TextEmbedder(method='bert') 
    
    logging.info("\nConstructing training graphs... (API rate limits and fallbacks apply)")
    train_graphs = [build_graph_from_title(row['headline'], row['is_sarcastic'], embedder) 
                    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Training Graphs")]

    logging.info("\nConstructing testing graphs...")
    test_graphs = [build_graph_from_title(row['headline'], row['is_sarcastic'], embedder) 
                   for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Testing Graphs")]

    train_loader = DataLoader(train_graphs, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=64, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    total_unique_relations = len(global_state.RELATION_VOCAB)
    logging.info(f"Total unique edge relations discovered: {total_unique_relations}")
    
    irf_weights = calculate_irf_weights(train_graphs, total_unique_relations).to(device)
    
    gnn_model = SarcasmGNN(
        input_dim=embedder.dim, 
        hidden_dim=32, 
        edge_embed_dim=16, 
        num_relations=total_unique_relations,
        irf_weights=irf_weights
    ).to(device)
    
    # 1. Added weight_decay (L2 Regularisation) to penalise massive weights
    optimizer = torch.optim.Adam([
        {'params': [p for n, p in gnn_model.named_parameters() if 'edge_embedding' not in n], 'lr': 0.001, 'weight_decay': 1e-4},
        {'params': gnn_model.edge_embedding.parameters(), 'lr': 0.05} 
    ])
    
    # 2. Added a Scheduler: Reduces LR by 50% if the test loss doesn't improve for 2 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=20, 
        min_lr=1e-6
    )
    
    criterion = FocalLoss(alpha=1.0, gamma=2.0)
    lambda_sem = 0.5 

    training_history = {
        'train_loss': [], 'test_loss': [],
        'test_acc': [], 'test_prec': [],
        'test_rec': [], 'test_f1': []
    }

    logging.info("\nInitiating training with Differential Learning Rates and Scheduling...")
    epochs = 150

    best_test_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    
    for epoch in tqdm(range(epochs), desc="Training Epochs", unit="epoch"): 
        gnn_model.train()
        train_loss = 0
        
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out, node_embeds = gnn_model(data.x, data.edge_index, data.edge_attr, data.edge_weight, data.batch)
            probs = torch.sigmoid(out.squeeze(-1))
            
            loss_bce = criterion(probs, data.y)
            loss_sem = semantic_contrastive_loss(node_embeds, data.edge_index, data.edge_attr)
            loss = loss_bce + lambda_sem * loss_sem
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(gnn_model, test_loader, criterion, device)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(gnn_model.state_dict())
            logging.info(f"[Checkpoint] New best model saved at Epoch {best_epoch} (Test Loss: {best_test_loss:.4f})")
        
        # Track the primary network learning rate before the step
        old_lr = optimizer.param_groups[0]['lr']

        # 3. Step the scheduler based on the test loss
        scheduler.step(test_loss)

        # Check if the scheduler forced a topological decay
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < old_lr:
            logging.info(f"[Scheduler] Plateau detected! Network LR decayed from {old_lr:.6f} to {new_lr:.6f}.")
        
        # Buffer metrics for plotting
        training_history['train_loss'].append(train_loss)
        training_history['test_loss'].append(test_loss)
        training_history['test_acc'].append(test_acc)
        training_history['test_prec'].append(test_prec)
        training_history['test_rec'].append(test_rec)
        training_history['test_f1'].append(test_f1)
        
        logging.info(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Acc: {test_acc*100:.1f}% | Prec: {test_prec:.3f} | Rec: {test_rec:.3f} | F1: {test_f1:.3f}")
        
        track_and_log_weights(gnn_model, global_state.RELATION_VOCAB, epoch)

    logging.info(f"\nTraining complete. Restoring best model weights from Epoch {best_epoch}...")
    if best_model_state is not None:
        gnn_model.load_state_dict(best_model_state)
        
    logging.info("\n==================================================")
    logging.info("   Exporting Topology and Visualisations ")
    logging.info("==================================================")
    
    # Ensure all directories exist before saving plots
    os.makedirs("visualisations", exist_ok=True)
    os.makedirs("visualisations/train_graphs", exist_ok=True)
    os.makedirs("visualisations/test_graphs", exist_ok=True)
    
    # --- NEW: Generate Progression Plots ---
    plot_training_metrics(training_history, "visualisations")
    plot_weight_trajectories(weight_history, os.path.join("visualisations", "weight_trajectories.png"))
    
    def export_dataset_results(df, dataset_name):
        csv_data = []
        gnn_model.eval()
        
        with torch.no_grad():
            for idx, (_, row) in tqdm(enumerate(df.iterrows()), total=len(df), desc=f"Exporting {dataset_name.capitalize()} Data"):
                headline = row['headline']
                true_label = row['is_sarcastic']
                
                data, concepts = build_graph_from_title(headline, true_label, embedder, return_concepts=True)
                data = data.to(device)
                batch = torch.zeros(data.x.size(0), dtype=torch.long).to(device)
                
                out, _ = gnn_model(data.x, data.edge_index, data.edge_attr, data.edge_weight, batch)
                pred_prob = torch.sigmoid(out.squeeze(-1)).item()
                pred_label = 1 if pred_prob >= 0.5 else 0
                
                edge_attr = data.edge_attr.squeeze(-1)
                num_seq_edges = (edge_attr == 0).sum().item() // 2
                num_sem_edges = (edge_attr > 0).sum().item() // 2
                
                found_relations = []
                for rel_id in edge_attr[edge_attr > 0].unique():
                    for label, v_id in global_state.RELATION_VOCAB.items():
                        if v_id == rel_id.item():
                            found_relations.append(label)
                            break
                            
                csv_data.append({
                    "Sample_ID": idx,
                    "Headline": headline,
                    "True_Label": true_label,
                    "Predicted_Label": pred_label,
                    "Prediction_Probability": round(pred_prob, 4),
                    "Correct_Prediction": true_label == pred_label,
                    "Num_Concepts": len(concepts),
                    "Sequential_Edges": num_seq_edges,
                    "Semantic_Edges": num_sem_edges,
                    "Relations_Found": " | ".join(found_relations) if found_relations else "None",
                    "Concepts_List": ", ".join(concepts)
                })
                
                save_path = os.path.join("visualisations", f"{dataset_name}_graphs", f"{dataset_name}_sample_{idx:04d}.png")
                save_gnn_graph(data, concepts, headline, true_label, pred_label, save_path)
                
        export_df = pd.DataFrame(csv_data)
        csv_export_path = f"{dataset_name}_data_analysis.csv"
        export_df.to_csv(csv_export_path, index=False)
        logging.info(f"Successfully exported {len(export_df)} {dataset_name} samples to '{csv_export_path}'.")

    export_dataset_results(train_df, "train")
    export_dataset_results(test_df, "test")
    
    global_state.save_cache(global_state.conceptnet_cache)