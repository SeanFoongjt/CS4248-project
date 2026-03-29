import os
import sys
import json
import logging
import random
import argparse
import optuna
import torch
import torch.nn as nn
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer

# Import your pipeline components
from models.conceptnet_model_pipeline import (
    RobertaConfig, RobertaGNNModel, SarcasmGraphDataset, 
    graph_collate_fn, calculate_irf_weights, train, evaluate
)
import utils.global_state as global_state
from utils.logger_setup import setup_logger

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Hyperparameter Tuning for Sarcasm GNN")
    parser.add_argument("--input", type=str, default=".", help="Directory containing the Sarcasm JSON datasets")
    parser.add_argument("--output", type=str, default="results/tuning", help="Directory to save tuning results and reports")
    parser.add_argument("--n-trials", type=int, default=20, help="Number of Optuna trials to run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--no-conceptnet", action="store_true", help="Disable ConceptNet API processing")
    parser.add_argument("--text-format", type=str, choices=["headline", "headline_section", "all"], default="headline", help="Which text fields to construct the graph from")
    
    
    args = parser.parse_args()
    
    log_path = os.path.join(args.output, f"training_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
    setup_logger(path=log_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ─────────────────────────────────────────────
    # 1. LOAD DATA (Text only, no preprocessing yet)
    # ─────────────────────────────────────────────
    samples = []
    
    try:
        with open(args.input, encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                samples.append({
                    "headline": item.get("headline", ""),
                    "section": item.get("section", ""),
                    "description": item.get("description", ""),
                    "label": item.get("is_sarcastic", 0)
                })
    except FileNotFoundError:
        logging.error(f"Dataset files not found in '{args.input}'.")
        sys.exit(1)

    data, _ = train_test_split(samples, test_size=0.2, random_state=args.seed)
    data_train, data_valid = train_test_split(data, test_size=0.25, random_state=args.seed)

    # ─────────────────────────────────────────────
    # 2. OPTUNA OBJECTIVE CLOSURE
    # ─────────────────────────────────────────────
    def objective(trial):
        # Dynamically Construct the Configuration per Trial
        cfg = RobertaConfig(
            # --- Locked Parameters ---
            num_labels = 2, 
            export_visualisations = False,
            text_format = args.text_format,
            output_dir = args.output,
            
            # --- Tuned Hyperparameters ---
            max_length = trial.suggest_categorical("max_length", [64, 128, 256, 512]),
            dropout = trial.suggest_float("dropout", 0.1, 0.4),
            learning_rate = trial.suggest_float("lr", 1e-6, 5e-5, log=True),
            gnn_learning_rate = trial.suggest_float("gnn_lr", 5e-4, 1e-2, log=True),
            batch_size = trial.suggest_categorical("batch_size", [16, 32, 48, 64]),
            num_epochs = trial.suggest_int("num_epochs", 5, 20),
            warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.2),
            edge_embed_dim = trial.suggest_categorical("edge_embed_dim", [8, 16, 32, 48, 64]),
            use_conceptnet = not args.no_conceptnet,
            weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
        )
        
        logging.info(f"\n=========================================")
        logging.info(f"  STARTING TRIAL {trial.number}")
        logging.info(f"  Max Len: {cfg.max_length} | CNet: {cfg.use_conceptnet} | Batch: {cfg.batch_size}")
        logging.info(f"=========================================")

        tokenizer = RobertaTokenizer.from_pretrained(cfg.pretrained_name)

        # Build graphs dynamically (ConceptNet API responses will be loaded from RAM after Trial 1)
        train_ds = SarcasmGraphDataset(data_train, tokenizer, cfg.max_length, use_conceptnet=cfg.use_conceptnet, text_format=cfg.text_format)
        val_ds   = SarcasmGraphDataset(data_valid, tokenizer, cfg.max_length, use_conceptnet=cfg.use_conceptnet, text_format=cfg.text_format)
        
        if cfg.use_conceptnet:
            global_state.save_cache(global_state.conceptnet_cache)

        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=graph_collate_fn)
        val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, collate_fn=graph_collate_fn)

        total_relations = len(global_state.RELATION_VOCAB)
        irf_weights = calculate_irf_weights(train_ds, total_relations).to(device)

        model = RobertaGNNModel(cfg, total_relations, irf_weights)
        
        # Execute Training Loop
        train(model, train_loader, val_loader, cfg, device)
        
        # Evaluate & Return Objective F1
        loss_fn = nn.CrossEntropyLoss()
        _, _, _, _, val_f1 = evaluate(model, val_loader, loss_fn, device)
        
        return val_f1

    # ─────────────────────────────────────────────
    # 3. RUN OPTIMISATION & EXPORT RESULTS
    # ─────────────────────────────────────────────
    study = optuna.create_study(direction="maximize", study_name="sarcasm_gnn_comprehensive_tuning")
    study.optimize(objective, n_trials=args.n_trials)
    
    os.makedirs(args.output, exist_ok=True)
    summary_path = os.path.join(args.output, "tuning_summary_comprehensive.txt")
    
    try:
        importances = optuna.importance.get_param_importances(study)
    except Exception as e:
        logging.warning(f"Could not calculate parameter importances: {e}")
        importances = {}

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=========================================\n")
        f.write("    COMPREHENSIVE OPTIMISATION REPORT    \n")
        f.write("=========================================\n\n")
        f.write(f"Total Trials Run: {args.n_trials}\n")
        f.write(f"Best Trial: #{study.best_trial.number}\n")
        f.write(f"Best Validation F1-Score: {study.best_trial.value:.4f}\n\n")
        
        f.write("--- Optimal Parameters ---\n")
        for key, value in study.best_trial.params.items():
            f.write(f"  --{key}: {value}\n")
            
        f.write("\n--- Hyperparameter Importance ---\n")
        f.write("(Shows which parameters had the greatest impact on the F1-Score)\n")
        if importances:
            for key, val in importances.items():
                f.write(f"  {key}: {val:.4f} ({val * 100:.1f}%)\n")
        else:
            f.write("  Not enough variance to calculate importances.\n")
            
        f.write("\n\n--- Full Trial History ---\n")
        df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
        df = df.rename(columns={"value": "f1_score"})
        f.write(df.to_string(index=False))

    print(f"\nOptimisation finished. A comprehensive report has been saved to: {summary_path}")
    print(f"Best F1-Score achieved: {study.best_trial.value:.4f}")

if __name__ == "__main__":
    main()