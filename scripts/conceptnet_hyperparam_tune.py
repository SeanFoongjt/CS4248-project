from datetime import datetime
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
    # ─────────────────────────────────────────────
    # 1. ARGUMENT PARSING
    # ─────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="Automated Hyperparameter Tuning for Sarcasm GNN")
    parser.add_argument("--input", type=str, default=".", help="File path to the Sarcasm JSON datasets")
    parser.add_argument("--output", type=str, default="tuning_results", help="Directory to save the tuning summary text file")
    parser.add_argument("--n-trials", type=int, default=15, help="Number of Optuna trials to run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--no-conceptnet", action="store_true", help="Disable ConceptNet API processing")
    
    args = parser.parse_args()
    
    log_path = os.path.join(args.output, f"training_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
    setup_logger(path=log_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ─────────────────────────────────────────────
    # 2. DATA LOADING & PRE-COMPUTATION
    # ─────────────────────────────────────────────
    samples = []
    
    try:
        with open(args.input, encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                samples.append((item["headline"], item["is_sarcastic"]))
    except FileNotFoundError:
        logging.error(f"Dataset file not found in '{args.input}'. Please verify the --input argument.")
        sys.exit(1)

    # Keep the split consistent across all trials
    data, _ = train_test_split(samples, test_size=0.2, random_state=args.seed)
    data_train, data_valid = train_test_split(data, test_size=0.25, random_state=args.seed)

    base_cfg = RobertaConfig(use_conceptnet=not args.no_conceptnet)
    tokenizer = RobertaTokenizer.from_pretrained(base_cfg.pretrained_name)

    logging.info("--- Pre-computing ConceptNet Graphs (This runs only once) ---")
    train_ds = SarcasmGraphDataset(data_train, tokenizer, base_cfg.max_length, use_conceptnet=base_cfg.use_conceptnet)
    val_ds   = SarcasmGraphDataset(data_valid, tokenizer, base_cfg.max_length, use_conceptnet=base_cfg.use_conceptnet)

    if base_cfg.use_conceptnet:
        global_state.save_cache(global_state.conceptnet_cache)

    total_relations = len(global_state.RELATION_VOCAB)
    global_irf_weights = calculate_irf_weights(train_ds, total_relations).to(device)

    # ─────────────────────────────────────────────
    # 3. OPTUNA OBJECTIVE CLOSURE
    # ─────────────────────────────────────────────
    def objective(trial):
        cfg = RobertaConfig(
            use_conceptnet = base_cfg.use_conceptnet,
            max_length = base_cfg.max_length,
            num_epochs = 50,
            export_visualisations = False, 
            
            # ── Parameter Search Space ──
            learning_rate = trial.suggest_float("lr", 1e-5, 5e-5, log=True),
            gnn_learning_rate = trial.suggest_float("gnn_lr", 5e-4, 5e-3, log=True),
            batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 48, 64]),
            edge_embed_dim = trial.suggest_categorical("edge_embed_dim", [8, 16, 32]),
            dropout = trial.suggest_float("dropout", 0.1, 0.4)
        )
        
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=graph_collate_fn)
        val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, collate_fn=graph_collate_fn)

        model = RobertaGNNModel(cfg, total_relations, global_irf_weights)
        
        logging.info(f"\n--- Starting Trial {trial.number} ---")
        train(model, train_loader, val_loader, cfg, device)
        
        loss_fn = nn.CrossEntropyLoss()
        _, _, _, _, val_f1 = evaluate(model, val_loader, loss_fn, device)
        
        return val_f1

    # ─────────────────────────────────────────────
    # 4. RUN OPTIMISATION & EXPORT RESULTS
    # ─────────────────────────────────────────────
    study = optuna.create_study(direction="maximize", study_name="sarcasm_gnn_conceptnet_tuning")
    study.optimize(objective, n_trials=args.n_trials)
    
    os.makedirs(args.output, exist_ok=True)
    summary_path = os.path.join(args.output, "tuning_summary.txt")
    
    # Calculate Hyperparameter Importances
    try:
        importances = optuna.importance.get_param_importances(study)
    except Exception as e:
        logging.warning(f"Could not calculate parameter importances: {e}")
        importances = {}

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=========================================\n")
        f.write("        OPTIMISATION COMPLETE            \n")
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
            f.write("  Not enough data/variance to calculate importances.\n")
            
        f.write("\n\n--- Full Trial History ---\n")
        # Export the full dataframe summary of all trials
        df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
        df = df.rename(columns={"value": "f1_score"})
        f.write(df.to_string(index=False))

    print(f"\nOptimisation finished. A comprehensive report has been saved to: {summary_path}")
    print(f"Best F1-Score achieved: {study.best_trial.value:.4f}")

if __name__ == "__main__":
    main()