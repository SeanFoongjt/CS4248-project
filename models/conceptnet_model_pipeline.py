from __future__ import annotations
 
from datetime import datetime
import os
import sys
import json
import math
import random
import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data

# ─────────────────────────────────────────────
# IMPORT CONCEPTNET STATE, VISUALISATIONS & LOGGER
# ─────────────────────────────────────────────
import utils.global_state as global_state
from utils.api import get_node_data
import utils.visualise as visualise
from utils.logger_setup import setup_logger

import spacy
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_web_md")
    nlp = spacy.load("en_core_web_md")
 
# ─────────────────────────────────────────────
# 1. CONFIG
# ─────────────────────────────────────────────
 
@dataclass
class RobertaConfig:
    pretrained_name: str = "roberta-base"
    max_length: int = 128
    num_labels: int = 2
    dropout: float = 0.1
    learning_rate: float = 2e-5
    gnn_learning_rate: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    edge_embed_dim: int = 16  
    use_conceptnet: bool = True  
    export_visualisations: bool = True
    output_dir: str = "result"  
 
 
# ─────────────────────────────────────────────
# 2. DATASET & GRAPH BUILDER
# ─────────────────────────────────────────────
 
class SarcasmGraphDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[str, int]],
        tokenizer: RobertaTokenizer,
        max_length: int,
        use_conceptnet: bool = True
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_conceptnet = use_conceptnet
        self.graphs = []
        
        self._build_graphs()

    def _build_graphs(self):
        import concurrent.futures
        
        for text, label in tqdm(self.samples, desc="Initialising Graph Representations"):
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].squeeze(0)
            attention_mask = encoding["attention_mask"].squeeze(0)
            
            seq_len_actual = attention_mask.sum().item()
            edge_index, edge_attr, edge_weight = [], [], []
            concepts_in_text = []
            
            for i in range(seq_len_actual - 1):
                edge_index.append([i, i + 1])
                edge_index.append([i + 1, i])
                edge_attr.extend([[0], [0]]) 
                edge_weight.extend([1.0, 1.0])
                
            if self.use_conceptnet:
                tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
                doc = nlp(text.lower())
                valid_pos = {"NOUN", "VERB", "ADJ", "PROPN"}
                concepts_in_text = [token.lemma_.lower().strip().replace(" ", "_") 
                                    for token in doc if token.pos_ in valid_pos and not token.is_stop]
                concepts_in_text = list(dict.fromkeys(concepts_in_text))
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                    futures = [executor.submit(get_node_data, concept) for concept in concepts_in_text]
                    concurrent.futures.wait(futures)
                
                concept_to_indices = {}
                for i, token in enumerate(tokens):
                    if token in ["[CLS]", "[SEP]", "<pad>"]: continue
                    clean_token = token.replace("Ġ", "").lower()
                    
                    for c in concepts_in_text:
                        if c == clean_token or c.startswith(clean_token) or clean_token.startswith(c):
                            if c not in concept_to_indices:
                                concept_to_indices[c] = []
                            concept_to_indices[c].append(i)
                            
                for i in range(len(concepts_in_text)):
                    for j in range(i + 1, len(concepts_in_text)):
                        c_i = concepts_in_text[i]
                        c_j = concepts_in_text[j]
                        
                        with global_state.cache_lock:
                            node_i_cache = global_state.conceptnet_cache.get(c_i, {})
                            
                        if c_j in node_i_cache:
                            for rel_label, weight in node_i_cache[c_j]:
                                rel_id = global_state.get_relation_id(rel_label)
                                
                                if c_i in concept_to_indices and c_j in concept_to_indices:
                                    for idx_i in concept_to_indices[c_i]:
                                        for idx_j in concept_to_indices[c_j]:
                                            edge_index.append([idx_i, idx_j])
                                            edge_index.append([idx_j, idx_i])
                                            edge_attr.extend([[rel_id], [rel_id]])
                                            edge_weight.extend([weight, weight])
                                        
            if not edge_index:
                edge_index = [[0, 0]]
                edge_attr = [[0]]
                edge_weight = [1.0]
                
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.long)
            edge_weight = torch.tensor(edge_weight, dtype=torch.float)
            
            weight_sum = edge_weight.sum()
            if weight_sum > 0:
                edge_weight = edge_weight / weight_sum
                
            self.graphs.append({
                "headline": text,
                "concepts": concepts_in_text if concepts_in_text else self.tokenizer.convert_ids_to_tokens(input_ids),
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "label": torch.tensor(label, dtype=torch.long),
                "edge_index": edge_index,
                "edge_attr": edge_attr,
                "edge_weight": edge_weight
            })
 
    def __len__(self) -> int:
        return len(self.graphs)
 
    def __getitem__(self, idx: int) -> dict:
        return self.graphs[idx]


def graph_collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    
    batched_edge_index = []
    batched_edge_attr = []
    batched_edge_weight = []
    
    max_len = input_ids.shape[1]
    
    for i, item in enumerate(batch):
        offset = i * max_len 
        edges = item['edge_index'] + offset
        batched_edge_index.append(edges)
        batched_edge_attr.append(item['edge_attr'])
        batched_edge_weight.append(item['edge_weight'])
        
    return {
        "headlines": [item['headline'] for item in batch],
        "concepts": [item['concepts'] for item in batch],
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label": labels,
        "edge_index": torch.cat(batched_edge_index, dim=1),
        "edge_attr": torch.cat(batched_edge_attr, dim=0),
        "edge_weight": torch.cat(batched_edge_weight, dim=0),
        "raw_graphs": batch
    }

def calculate_irf_weights(dataset, num_relations):
    edge_counts = torch.zeros(num_relations)
    for data in dataset.graphs:
        unique, counts = torch.unique(data["edge_attr"], return_counts=True)
        for u, c in zip(unique, counts):
            edge_counts[u] += c
            
    total_edges = edge_counts.sum()
    irf_weights = torch.zeros(num_relations)
    
    for i in range(num_relations):
        if edge_counts[i] > 0: 
            irf_weights[i] = math.log((total_edges / edge_counts[i]) + 1)
        else: 
            irf_weights[i] = 1.0 
            
    if num_relations > 1:
        max_irf = irf_weights[1:].max()
        irf_weights[0] = max_irf
            
    weight_sum = irf_weights.sum()
    if weight_sum > 0:
        irf_weights = irf_weights / weight_sum
        
    return irf_weights


# ─────────────────────────────────────────────
# 3. MODEL
# ─────────────────────────────────────────────
 
class RobertaGNNModel(nn.Module):
    def __init__(self, cfg: RobertaConfig, num_relations: int, irf_weights: torch.Tensor):
        super().__init__()
        self.cfg = cfg
 
        self.encoder = RobertaModel.from_pretrained(cfg.pretrained_name)
        hidden_size = self.encoder.config.hidden_size 
        
        self.edge_embedding = nn.Embedding(num_relations, cfg.edge_embed_dim)
        torch.nn.init.orthogonal_(self.edge_embedding.weight)
        self.register_buffer('irf_weights', irf_weights)
        
        self.gat = GATv2Conv(hidden_size, hidden_size, heads=1, edge_dim=cfg.edge_embed_dim, dropout=0.3, concat=False)
 
        self.classifier = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden_size // 2, cfg.num_labels),
        )
 
    def forward(self, input_ids, attention_mask, edge_index, edge_attr, edge_weight):
        batch_size, seq_len = input_ids.size()

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        transformer_embeddings = outputs.last_hidden_state 
 
        x_flat = transformer_embeddings.view(batch_size * seq_len, -1)

        edge_attr_1d = edge_attr.view(-1) 
        raw_edge_feats = self.edge_embedding(edge_attr_1d)
        # normalized_edge_feats = F.normalize(raw_edge_feats, p=2, dim=-1)
        irf_scalars = self.irf_weights[edge_attr_1d].unsqueeze(-1)
        empirical_scalars = edge_weight.view(-1).unsqueeze(-1)
        # scaled_edge_feats = normalized_edge_feats * irf_scalars * empirical_scalars
        scaled_edge_feats = raw_edge_feats * irf_scalars * empirical_scalars

        x_gnn_flat = self.gat(x_flat, edge_index, edge_attr=scaled_edge_feats)
        x_gnn_flat = F.relu(x_gnn_flat + x_flat) 
        
        x_gnn = x_gnn_flat.view(batch_size, seq_len, -1)
        cls_repr = x_gnn[:, 0, :] 
 
        logits = self.classifier(cls_repr)             
        return logits
 
 
# ─────────────────────────────────────────────
# 4. TRAINING LOOP
# ─────────────────────────────────────────────
 
def train(
    model: RobertaGNNModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: RobertaConfig,
    device: torch.device,
) -> dict:
 
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()

    transformer_params = []
    gnn_params = []
    
    for name, param in model.named_parameters():
        if "encoder" in name:
            transformer_params.append(param)
        else:
            gnn_params.append(param)
            
    optimizer = AdamW([
        {"params": transformer_params, "lr": cfg.learning_rate},
        {"params": gnn_params, "lr": cfg.gnn_learning_rate}
    ], weight_decay=0.01)
 
    total_steps = len(train_loader) * cfg.num_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
 
    training_history = {
        'train_loss': [], 'test_loss': [],
        'test_acc': [], 'test_prec': [],
        'test_rec': [], 'test_f1': []
    }

    for epoch in tqdm(range(cfg.num_epochs), desc="Training"):
        model.train()
        train_loss = 0.0
 
        for batch in train_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)
            edge_index     = batch["edge_index"].to(device)
            edge_attr      = batch["edge_attr"].to(device)
            edge_weight    = batch["edge_weight"].to(device)
 
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask, edge_index, edge_attr, edge_weight)
            loss   = loss_fn(logits, labels)
            loss.backward()
 
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
 
            train_loss += loss.item()
 
        avg_train_loss = train_loss / len(train_loader)
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(model, val_loader, loss_fn, device)
 
        training_history['train_loss'].append(avg_train_loss)
        training_history['test_loss'].append(val_loss)
        training_history['test_acc'].append(val_acc)
        training_history['test_prec'].append(val_prec)
        training_history['test_rec'].append(val_rec)
        training_history['test_f1'].append(val_f1)

        logging.info(
            f"Epoch {epoch + 1}/{cfg.num_epochs} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"val_f1={val_f1:.4f}"
        )

        if cfg.export_visualisations:
            visualise.track_and_log_weights(model, global_state.RELATION_VOCAB, epoch)        

    return training_history
 
 
# ─────────────────────────────────────────────
# 5. EVALUATION
# ─────────────────────────────────────────────
 
@torch.no_grad()
def evaluate(
    model: RobertaGNNModel,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float, float, float, float]:
 
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
 
    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)
        edge_index     = batch["edge_index"].to(device)
        edge_attr      = batch["edge_attr"].to(device)
        edge_weight    = batch["edge_weight"].to(device)
 
        logits = model(input_ids, attention_mask, edge_index, edge_attr, edge_weight)
        loss   = loss_fn(logits, labels)
 
        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
 
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels) if all_labels else 0
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    return total_loss / len(loader), accuracy, precision, recall, f1
 
 
# ─────────────────────────────────────────────
# 6. PIPELINE & INFERENCE METHODS
# ─────────────────────────────────────────────
 
def build_pipeline(
    train_samples: list[tuple[str, int]],
    val_samples: list[tuple[str, int]],
    cfg: Optional[RobertaConfig] = None,
) -> None:
    cfg = cfg or RobertaConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    tokenizer = RobertaTokenizer.from_pretrained(cfg.pretrained_name)
 
    train_ds = SarcasmGraphDataset(train_samples, tokenizer, cfg.max_length, use_conceptnet=cfg.use_conceptnet)
    val_ds   = SarcasmGraphDataset(val_samples,   tokenizer, cfg.max_length, use_conceptnet=cfg.use_conceptnet)
    
    if cfg.use_conceptnet:
        global_state.save_cache(global_state.conceptnet_cache)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=graph_collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, collate_fn=graph_collate_fn)
 
    total_unique_relations = len(global_state.RELATION_VOCAB)
    logging.info(f"Total unique relationships tracked: {total_unique_relations}")
    irf_weights = calculate_irf_weights(train_ds, total_unique_relations).to(device)
 
    model = RobertaGNNModel(cfg, total_unique_relations, irf_weights)
    history = train(model, train_loader, val_loader, cfg, device)

    # Output Model
    output_path = os.path.join(cfg.output_dir, f"sarcasm_gnn_model_{cfg.use_conceptnet}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pt")
    save_dir = os.path.dirname(output_path)
    if save_dir:  # Only attempt to make directories if a parent folder is specified
        os.makedirs(save_dir, exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_relations': total_unique_relations,
        'irf_weights': irf_weights,
        'use_conceptnet': cfg.use_conceptnet
    }, output_path)
    
    logging.info(f"Graph-augmented Transformer Checkpoint saved to {output_path}.")

    # Render Visualisations
    if cfg.export_visualisations:
        visualisation_dir = os.path.join(cfg.output_dir, "visualisations")
        os.makedirs(visualisation_dir, exist_ok=True)
        os.makedirs(os.path.join(visualisation_dir, "graphs"), exist_ok=True)
        
        visualise.plot_training_metrics(history, visualisation_dir)
        visualise.plot_weight_trajectories(visualise.weight_history, os.path.join(visualisation_dir, "weight_trajectories.png"))

        history_df = pd.DataFrame(history)
        history_df.index = history_df.index + 1  # 1-based indexing for epochs
        history_df.index.name = 'Epoch'
        history_csv_path = os.path.join(visualisation_dir, "training_history.csv")
        history_df.to_csv(history_csv_path)
        logging.info(f"Training metrics exported to {history_csv_path}")

        if visualise.weight_history:
            weights_df = pd.DataFrame(visualise.weight_history)
            weights_df.index = weights_df.index + 1 
            weights_df.index.name = 'Epoch'
            weights_csv_path = os.path.join(visualisation_dir, "relation_weights.csv")
            weights_df.to_csv(weights_csv_path)
            logging.info(f"ConceptNet relation weights exported to {weights_csv_path}")
        
        # Export a few sample graphs
        model.eval()
        for idx, item in enumerate(val_ds.graphs[:10]):  
            input_ids = item['input_ids'].unsqueeze(0).to(device)
            attention_mask = item['attention_mask'].unsqueeze(0).to(device)
            edge_index = item['edge_index'].to(device)
            edge_attr = item['edge_attr'].unsqueeze(-1).to(device)
            edge_weight = item['edge_weight'].to(device)
            
            with torch.no_grad():
                logits = model(input_ids, attention_mask, edge_index, edge_attr, edge_weight)
                pred_label = logits.argmax(dim=-1).item()
                
            pyg_data = Data(edge_index=item['edge_index'], num_nodes=len(item['concepts']))
            save_path = os.path.join(visualisation_dir, "graphs", f"sample_{idx:03d}.png")
            visualise.save_gnn_graph(
                data=pyg_data, 
                concepts=item['concepts'], 
                headline=item['headline'], 
                true_label=item['label'].item(), 
                pred_label=pred_label, 
                save_path=save_path
            )
            
        logging.info("Visualisations saved to the 'visualisations' directory.")

    return model

def predict(texts: list[str], model_path: str = "sarcasm_gnn_model.pt", output_dir: str = "results") -> list[str]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(model_path, map_location=device)
    num_relations = checkpoint['num_relations']
    irf_weights = checkpoint['irf_weights']
    use_conceptnet = checkpoint.get('use_conceptnet', True)

    cfg = RobertaConfig(use_conceptnet=use_conceptnet, output_dir=output_dir)
    tokenizer = RobertaTokenizer.from_pretrained(cfg.pretrained_name)

    model = RobertaGNNModel(cfg, num_relations, irf_weights)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    dummy_samples = [(text, 0) for text in texts]
    eval_ds = SarcasmGraphDataset(dummy_samples, tokenizer, cfg.max_length, use_conceptnet=use_conceptnet)
    eval_loader = DataLoader(eval_ds, batch_size=cfg.batch_size, collate_fn=graph_collate_fn)

    results = []
    with torch.no_grad():
        for batch in eval_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            edge_index     = batch["edge_index"].to(device)
            edge_attr      = batch["edge_attr"].to(device)
            edge_weight    = batch["edge_weight"].to(device)

            logits = model(input_ids, attention_mask, edge_index, edge_attr, edge_weight)
            preds = logits.argmax(dim=-1).cpu().numpy()
            
            for pred in preds:
                results.append("sarcastic" if pred == 1 else "not sarcastic")

    return results

# ─────────────────────────────────────────────
# Example usage (with Argparse)
# ─────────────────────────────────────────────
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate the Sarcasm GNN Model")
    parser.add_argument("--input", type=str, default="Sarcasm_Headlines_Dataset_v2.json", help="Path to the input JSON dataset")
    parser.add_argument("--output", type=str, default="result", help="Path to save the trained model checkpoint")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training and evaluation")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate for the optimizer")
    parser.add_argument("--max-length", type=int, default=128, help="Maximum sequence length for tokenisation")
    parser.add_argument("--no-conceptnet", action="store_true", help="Disable ConceptNet API processing")
    parser.add_argument("--no-visualisations", action="store_true", help="Disable exporting performance visualisations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    parser.add_argument("--predict", action="store_true", help="Run in inference mode rather than training")
    parser.add_argument("--model-path", type=str, default="sarcasm_gnn_model.pt", help="Path to the trained model checkpoint (for prediction)")

    args = parser.parse_args()

    # Apply global custom logger rules 
    log_path = os.path.join(args.output, f"training_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
    setup_logger(path=log_path)

    samples = []
    
    logging.info(f"Loading dataset from {args.input}...")

    try:
        with open(args.input) as f:
            for line in f:
                item = json.loads(line)
                headline = item["headline"]
                label = item["is_sarcastic"]
                samples.append((headline, label))
    except FileNotFoundError:
        logging.error("Dataset files not found. Please ensure the Sarcasm JSON files are in the directory.")
        sys.exit(1)

    if args.predict:
        logging.info("\n--- Initiating Inference Mode ---")
        logging.info(f"Loading checkpoint from: {args.model_path}")
        
        if not os.path.exists(args.model_path):
            logging.error(f"Checkpoint file not found at {args.model_path}. Please train the model first or provide a valid path.")
            sys.exit(1)
            
        predictions = predict(samples, model_path=args.model_path, output_dir=args.output)
        
        logging.info("\n--- Prediction Results ---")
        for text, pred in zip(samples, predictions):
            logging.info(f"Text: {text}\nPred: {pred}\n")
        sys.exit(0)

    else:
        seed = args.seed if args.seed is not None else random.randint(1, 100)
        logging.info("The seed is " + str(seed))

        data, data_test = train_test_split(samples, test_size=0.2, random_state=seed)
        data_train, data_valid = train_test_split(data, test_size=0.25, random_state=seed)
    
        cfg = RobertaConfig(
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            max_length=args.max_length,
            use_conceptnet=not args.no_conceptnet,
            export_visualisations=not args.no_visualisations,
            output_dir=args.output
        )
        
        logging.info("\n--- Initiating Model Pipeline ---")
        model = build_pipeline(data_train, data_valid, cfg)

        logging.info("\n--- Evaluating Model on Unseen Test Set ---")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = RobertaTokenizer.from_pretrained(cfg.pretrained_name)
        
        test_ds = SarcasmGraphDataset(data_test, tokenizer, cfg.max_length, use_conceptnet=cfg.use_conceptnet)
        test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, collate_fn=graph_collate_fn)
        loss_fn = nn.CrossEntropyLoss()
        
        test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(model, test_loader, loss_fn, device)
        
        logging.info(f"\nFinal Test Set Results:")
        logging.info(f"Loss:      {test_loss:.4f}")
        logging.info(f"Accuracy:  {test_acc:.4f}")
        logging.info(f"Precision: {test_prec:.4f}")
        logging.info(f"Recall:    {test_rec:.4f}")
        logging.info(f"F1-Score:  {test_f1:.4f}")