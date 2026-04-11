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
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import numpy as np

# Import your pipeline components
from models.general_conceptnet_gnn_pipeline import (
    TransformerGNNConfig, TransformerGNNModel, SarcasmGraphDataset, 
    graph_collate_fn, calculate_irf_weights, train, evaluate
)
import utils.global_state as global_state
from utils.logger_setup import setup_logger
import utils.visualise as visualise

def set_global_seed(seed: int):
    """Enforces global determinism across all random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    # Forces deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Hyperparameter Tuning for Sarcasm GNN")
    parser.add_argument("--input", type=str, default=".", help="Directory containing the Sarcasm JSON datasets")
    parser.add_argument("--output", type=str, default="results/tuning", help="Directory to save tuning results and reports")
    parser.add_argument("--n-trials", type=int, default=20, help="Number of Optuna trials to run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--no-conceptnet", action="store_true", help="Disable ConceptNet API processing")
    parser.add_argument("--text-format", type=str, choices=["headline", "headline_section", "all"], default="headline", help="Which text fields to construct the graph from")
    parser.add_argument("--model-type", type=str, choices=["roberta", "distilbert"], default="roberta", help="Which transformer architecture to use")
    parser.add_argument("--pretrained-name", type=str, default=None, help="HuggingFace model name (defaults to roberta-base or distilbert-base-uncased)")

    parser.add_argument(
        '--pos',
        nargs='+',           # Gathers 1 or more arguments into a list
        type=str.upper,      # Automatically converts lowercase inputs to uppercase
        choices={"NOUN", "VERB", "ADJ", "PROPN"},   # Restricts inputs strictly to your set
        default=["NOUN", "VERB", "ADJ", "PROPN"], # Optional: Set a default if the flag isn't called
        help=f"Specify one or more POS tags. Allowed values: {', '.join({"NOUN", "VERB", "ADJ", "PROPN"})}"
    )
    
    args = parser.parse_args()

    selected_pos = set(args.pos)
    
    log_path = os.path.join(args.output, f"training_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
    setup_logger(path=log_path)

    set_global_seed(args.seed)

    logging.info(f"Starting comprehensive hyperparameter tuning with the following settings:")
    logging.info(f"  Input Directory: {args.input}")
    logging.info(f"  Output Directory: {args.output}")
    logging.info(f"  Number of Trials: {args.n_trials}")
    logging.info(f"  Random Seed: {args.seed}")
    logging.info(f"  Use ConceptNet: {not args.no_conceptnet}")
    logging.info(f"  Text Format: {args.text_format}")
    logging.info(f"  Model Type: {args.model_type}")
    logging.info(f"  Pretrained Model: {args.pretrained_name if args.pretrained_name else 'Default'}")
    logging.info(f"  Selected POS Tags: {', '.join(selected_pos)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.pretrained_name is None:
        pretrained_name = "roberta-base" if args.model_type == "roberta" else "distilbert-base-uncased"
    else:
        pretrained_name = args.pretrained_name

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

    data, data_test = train_test_split(samples, test_size=0.2, random_state=args.seed)
    data_train, data_valid = train_test_split(data, test_size=0.25, random_state=args.seed)

    # ─────────────────────────────────────────────
    # 2. OPTUNA OBJECTIVE CLOSURE
    # ─────────────────────────────────────────────
    def objective(trial):
        # Dynamically Construct the Configuration per Trial
        cfg = TransformerGNNConfig(
            # --- Locked Parameters ---
            num_labels = 2, 
            export_visualisations = False,
            text_format = args.text_format,
            output_dir = args.output,
            model_type = args.model_type,
            pretrained_name = pretrained_name,
            selected_pos=selected_pos,
            
            # --- Tuned Hyperparameters ---
            max_length = trial.suggest_categorical("max_length", [128, 256, 512]),
            dropout = trial.suggest_categorical("dropout", [0.1, 0.2, 0.3, 0.4]),
            learning_rate = trial.suggest_float("lr", 1e-7, 1e-5, log=True),
            gnn_learning_rate = trial.suggest_float("gnn_lr", 1e-5, 1e-2, log=True),
            batch_size = trial.suggest_categorical("batch_size", [16, 32, 48, 64]),
            num_epochs = trial.suggest_int("num_epochs", 3, 10),
            warmup_ratio = trial.suggest_categorical("warmup_ratio", [0.0, 0.001, 0.1, 0.15, 0.2]),
            edge_embed_dim = trial.suggest_categorical("edge_embed_dim", [8, 16, 32, 48, 64]),
            use_conceptnet = not args.no_conceptnet,
            weight_decay = trial.suggest_categorical("weight_decay", [1e-4, 1e-3, 1e-2, 1e-1])
        )
        
        logging.info(f"\n=========================================")
        logging.info(f"  STARTING TRIAL {trial.number}")
        logging.info(f"  Max Len: {cfg.max_length} | CNet: {cfg.use_conceptnet} | Batch: {cfg.batch_size}")
        logging.info(f"=========================================")

        tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_name)

        train_ds = SarcasmGraphDataset(data_train, tokenizer, cfg.max_length, use_conceptnet=cfg.use_conceptnet, text_format=cfg.text_format, selected_pos=cfg.selected_pos)
        val_ds   = SarcasmGraphDataset(data_valid, tokenizer, cfg.max_length, use_conceptnet=cfg.use_conceptnet, text_format=cfg.text_format, selected_pos=cfg.selected_pos)

        if cfg.use_conceptnet:
            global_state.save_cache(global_state.conceptnet_cache)

        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=graph_collate_fn)
        val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, collate_fn=graph_collate_fn)

        total_relations = len(global_state.RELATION_VOCAB)
        irf_weights = calculate_irf_weights(train_ds, total_relations).to(device)

        model = TransformerGNNModel(cfg, total_relations, irf_weights)
        
        # Execute Training Loop
        history = train(model, train_loader, val_loader, cfg, device)
        
        val_f1_scores = history['test_f1'] 
        final_val_f1 = history['test_f1'][-1]

        logging.info(f"Trial {trial.number} Results: Best Validation F1={final_val_f1:.4f}")
        
        return final_val_f1

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
        f.write(f"Best Validation F1-Score: {study.best_trial.value:.6f}\n\n")
        
        f.write("--- Optimal Parameters ---\n")
        for key, value in study.best_trial.params.items():
            f.write(f"  --{key}: {value}\n")
            
        f.write("\n--- Hyperparameter Importance ---\n")
        f.write("(Shows which parameters had the greatest impact on the F1-Score)\n")
        if importances:
            for key, val in importances.items():
                f.write(f"  {key}: {val:.6f} ({val * 100:.1f}%)\n")
        else:
            f.write("  Not enough variance to calculate importances.\n")
            
        f.write("\n\n--- Full Trial History ---\n")
        df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
        df = df.rename(columns={"value": "f1_score"})
        f.write(df.to_string(index=False))

    print(f"\nOptimisation finished. A comprehensive report has been saved to: {summary_path}")
    print(f"Best F1-Score achieved: {study.best_trial.value:.6f}")

# ─────────────────────────────────────────────
    # 4. TRAIN, SAVE, AND VISUALISE BEST MODEL
    # ─────────────────────────────────────────────
    logging.info("\n=========================================")
    logging.info("  TRAINING FINAL MODEL ON BEST PARAMETERS  ")
    logging.info("=========================================")
    print("\nTraining final model on best hyperparameters...")

    best_params = study.best_trial.params

    # Construct a new config using the optimal parameters
    final_cfg = TransformerGNNConfig(
        num_labels=2,
        export_visualisations=True, # Explicitly enable visualisations
        text_format=args.text_format,
        output_dir=os.path.join(args.output, "final_best_model"),
        model_type=args.model_type,
        pretrained_name=pretrained_name,
        max_length=best_params["max_length"],
        dropout=best_params["dropout"],
        learning_rate=best_params["lr"],
        gnn_learning_rate=best_params["gnn_lr"],
        batch_size=best_params["batch_size"],
        num_epochs=best_params["num_epochs"],
        warmup_ratio=best_params["warmup_ratio"],
        edge_embed_dim=best_params["edge_embed_dim"],
        use_conceptnet=not args.no_conceptnet,
        weight_decay=best_params["weight_decay"],
        selected_pos=selected_pos
    )

    tokenizer = AutoTokenizer.from_pretrained(final_cfg.pretrained_name)

    train_ds = SarcasmGraphDataset(data_train, tokenizer, final_cfg.max_length, use_conceptnet=final_cfg.use_conceptnet, text_format=final_cfg.text_format, selected_pos=final_cfg.selected_pos)
    val_ds   = SarcasmGraphDataset(data_valid, tokenizer, final_cfg.max_length, use_conceptnet=final_cfg.use_conceptnet, text_format=final_cfg.text_format, selected_pos=final_cfg.selected_pos)
    test_ds  = SarcasmGraphDataset(data_test, tokenizer, final_cfg.max_length, use_conceptnet=final_cfg.use_conceptnet, text_format=final_cfg.text_format, selected_pos=final_cfg.selected_pos)

    train_loader = DataLoader(train_ds, batch_size=final_cfg.batch_size, shuffle=True, collate_fn=graph_collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=final_cfg.batch_size, collate_fn=graph_collate_fn)
    test_loader  = DataLoader(test_ds, batch_size=final_cfg.batch_size, collate_fn=graph_collate_fn)

    total_relations = len(global_state.RELATION_VOCAB)
    irf_weights = calculate_irf_weights(train_ds, total_relations).to(device)

    final_model = TransformerGNNModel(final_cfg, total_relations, irf_weights)

    # CRITICAL: Clear the visualisation history from previous Optuna trials
    if hasattr(visualise, 'weight_history'):
        visualise.weight_history.clear()

    # Train the final model and capture history
    history = train(final_model, train_loader, val_loader, final_cfg, device)

    # --- SAVE THE MODEL CHECKPOINT ---
    os.makedirs(final_cfg.output_dir, exist_ok=True)
    output_path = os.path.join(final_cfg.output_dir, f"sarcasm_gnn_model_tuned_{final_cfg.model_type}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pt")
    
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'num_relations': total_relations,
        'irf_weights': irf_weights,
        'model_type': final_cfg.model_type,
        'pretrained_name': final_cfg.pretrained_name,
        'use_conceptnet': final_cfg.use_conceptnet,
        'text_format': final_cfg.text_format,
        'gnn_learning_rate': final_cfg.gnn_learning_rate,
        'dropout': final_cfg.dropout,
        'weight_decay': final_cfg.weight_decay,
        'warmup_ratio': final_cfg.warmup_ratio,
        'edge_embed_dim': final_cfg.edge_embed_dim,
        'learning_rate': final_cfg.learning_rate,
        'max_length': final_cfg.max_length,
        'num_epochs': final_cfg.num_epochs,
        'batch_size': final_cfg.batch_size,
        'selected_pos': final_cfg.selected_pos
    }, output_path)
    logging.info(f"Final tuned model checkpoint saved to {output_path}.")

    # --- EXPORT VISUALISATIONS ---
    visualisation_dir = os.path.join(final_cfg.output_dir, "visualisations")
    os.makedirs(visualisation_dir, exist_ok=True)
    os.makedirs(os.path.join(visualisation_dir, "graphs"), exist_ok=True)
    
    visualise.plot_training_metrics(history, visualisation_dir)
    visualise.plot_weight_trajectories(visualise.weight_history, os.path.join(visualisation_dir, "weight_trajectories.png"))

    history_df = pd.DataFrame(history)
    history_df.index = history_df.index + 1  
    history_df.index.name = 'Epoch'
    history_csv_path = os.path.join(visualisation_dir, "training_history.csv")
    history_df.to_csv(history_csv_path)

    if hasattr(visualise, 'weight_history') and visualise.weight_history:
        weights_df = pd.DataFrame(visualise.weight_history)
        weights_df.index = weights_df.index + 1 
        weights_df.index.name = 'Epoch'
        weights_csv_path = os.path.join(visualisation_dir, "relation_weights.csv")
        weights_df.to_csv(weights_csv_path)
    
    # Export sample graphs from the validation set
    final_model.eval()
    from torch_geometric.data import Data
    for idx, item in enumerate(val_ds.graphs[:10]):  
        input_ids = item['input_ids'].unsqueeze(0).to(device)
        attention_mask = item['attention_mask'].unsqueeze(0).to(device)
        edge_index = item['edge_index'].to(device)
        edge_attr = item['edge_attr'].unsqueeze(-1).to(device)
        edge_weight = item['edge_weight'].to(device)
        
        with torch.no_grad():
            logits = final_model(input_ids, attention_mask, edge_index, edge_attr, edge_weight)
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
    logging.info(f"Visualisations and training artifacts exported to {visualisation_dir}.")

    # ─────────────────────────────────────────────
    # 5. UNSEEN TEST SET EVALUATION
    # ─────────────────────────────────────────────
    logging.info("\nEvaluating final model on the hold-out test set...")
    loss_fn = nn.CrossEntropyLoss()
    
    test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(final_model, test_loader, loss_fn, device)

    final_results = (
        "\n=========================================\n"
        "        FINAL UNSEEN TEST RESULTS        \n"
        "=========================================\n"
        f"Loss:      {test_loss:.4f}\n"
        f"Accuracy:  {test_acc:.4f}\n"
        f"Precision: {test_prec:.4f}\n"
        f"Recall:    {test_rec:.4f}\n"
        f"F1-Score:  {test_f1:.4f}\n"
        "========================================="
    )
    
    logging.info(final_results)
    print(final_results)
    
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(final_results)

    # ─────────────────────────────────────────────
    # 6. EXPORT FINAL MODEL PREDICTIONS (OUTPUT)
    # ─────────────────────────────────────────────
    logging.info("\nExtracting final model predictions on the test set...")
    final_model.eval()
    test_outputs = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            edge_index     = batch["edge_index"].to(device)
            edge_attr      = batch["edge_attr"].to(device)
            edge_weight    = batch["edge_weight"].to(device)
            
            headlines      = batch["headlines"]
            labels         = batch["label"].cpu().numpy()

            logits = final_model(input_ids, attention_mask, edge_index, edge_attr, edge_weight)
            preds = logits.argmax(dim=-1).cpu().numpy()

            for headline, true_label, pred_label in zip(headlines, labels, preds):
                test_outputs.append({
                    "Headline": headline,
                    "True_Label": "Sarcastic" if true_label == 1 else "Not Sarcastic",
                    "Predicted_Label": "Sarcastic" if pred_label == 1 else "Not Sarcastic",
                    "Correct": true_label == pred_label
                })

    output_df = pd.DataFrame(test_outputs)
    output_csv_path = os.path.join(final_cfg.output_dir, "final_test_predictions.csv")
    output_df.to_csv(output_csv_path, index=False, encoding='utf-8')
    logging.info(f"Saved {len(test_outputs)} detailed test predictions to: {output_csv_path}")

    print(f"\n--- Sample Outputs from the Final Tuned Model ---")
    print(f"Detailed outputs saved to: {output_csv_path}\n")
    
    sample_df = output_df.sample(min(5, len(output_df)))
    for _, row in sample_df.iterrows():
        match_str = "✅ Correct" if row['Correct'] else "❌ Incorrect"
        print(f"Headline:  {row['Headline']}")
        print(f"Actual:    {row['True_Label']}")
        print(f"Predicted: {row['Predicted_Label']} ({match_str})\n")
        print("-" * 50)

if __name__ == "__main__":
    main()