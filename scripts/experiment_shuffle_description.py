import os
import sys
import json
import logging
import argparse
import torch
import torch.nn as nn
from datetime import datetime
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Import your pipeline components
from models.general_conceptnet_gnn_pipeline import (
    TransformerGNNModel, SarcasmGraphDataset, 
    graph_collate_fn, evaluate
)
from utils.logger_setup import setup_logger

def shuffle(input_df, bin_size=5, seed=42):
    """Groups description sections into bins of similar length and shuffles descriptions within each bin."""
    
    df = input_df.copy()
    df['desc_length'] = df['description'].str.split().str.len().fillna(0)
    df['desc_bin_id'] = df['desc_length'] // bin_size
    df['shuffled_description'] = df.groupby('desc_bin_id')['description'].transform(
        lambda x: x.sample(frac=1, random_state=seed).values
    )
    df['shuffled_preprocessed_description'] = df.groupby('desc_bin_id')['preprocessed_description'].transform(
        lambda x: x.sample(frac=1, random_state=seed).values
    )
    return df

def main():
    parser = argparse.ArgumentParser(description="Experiment on Shuffled Descriptions of Similar Length")
    parser.add_argument("--input", type=str, default=".", help="Directory containing the Sarcasm JSON datasets")
    parser.add_argument("--output", type=str, default="results/experiments", help="Directory to save results and reports")
    parser.add_argument("--model", type=int, default=42, help="Filepath of the model's saved PT file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    log_path = os.path.join(args.output, f"training_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
    setup_logger(path=log_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ─────────────────────────────────────────────
    # 1. LOAD MODEL FROM SAVE
    # ─────────────────────────────────────────────
    save = torch.load(args.model)
    model = TransformerGNNModel()
    model.load_state_dict(save['model_state_dict'])
    model.eval()

    # ─────────────────────────────────────────────
    # 2. LOAD DATA AND SHUFFLE TEST SET
    # ─────────────────────────────────────────────
    samples = []
    
    try:
        with open(args.input, encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                samples.append({
                    "headline": item.get("headline", ""),
                    "section": item.get("preprocessed_section", ""),
                    "description": item.get("shuffled_preprocessed_description", ""), # make sure we use shuffled here
                    "label": item.get("is_sarcastic", 0)
                })
    except FileNotFoundError:
        logging.error(f"Dataset files not found in '{args.input}'.")
        sys.exit(1)

    _, data_test = train_test_split(samples, test_size=0.2, random_state=args.seed)

    # shuffle training data
    data_test = shuffle(data_test, 5, args.seed)

    tokenizer = AutoTokenizer.from_pretrained(save['pretrained_name'])
    test_ds  = SarcasmGraphDataset(data_test, tokenizer, save['max_length'], use_conceptnet=save['use_conceptnet'], text_format=save['text_format'])
    test_loader  = DataLoader(test_ds, batch_size=save['batch_size'], collate_fn=graph_collate_fn)

    # ─────────────────────────────────────────────
    # 3. EVALUATION ON TEST SET WITH SHUFFLED DESCRIPTIONS
    # ─────────────────────────────────────────────
    logging.info("\nEvaluating final model on shuffled test set...")
    loss_fn = nn.CrossEntropyLoss()
    
    test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(model, test_loader, loss_fn, device)

    final_results = (
        "\n=========================================\n"
        "        FINAL SHUFFLED TEST RESULTS        \n"
        "=========================================\n"
        f"Model:     {args.model}\n"
        f"Loss:      {test_loss:.4f}\n"
        f"Accuracy:  {test_acc:.4f}\n"
        f"Precision: {test_prec:.4f}\n"
        f"Recall:    {test_rec:.4f}\n"
        f"F1-Score:  {test_f1:.4f}\n"
        "========================================="
    )
    
    logging.info(final_results)
    print(final_results)

    os.makedirs(args.output, exist_ok=True)
    summary_path = os.path.join(args.output, "experiment_shuffle_description.txt")
    
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(final_results)

if __name__ == "__main__":
    main()