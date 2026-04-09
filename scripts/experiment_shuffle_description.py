import os
import json
import logging
import argparse
import random
from datetime import datetime

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from models.general_conceptnet_gnn_pipeline import (
    TransformerGNNConfig,
    TransformerGNNModel,
    SarcasmGraphDataset,
    graph_collate_fn,
    evaluate,
    set_global_seed,
)
from utils.logger_setup import setup_logger


def load_samples(input_path: str) -> list[dict]:
    """Load either a JSONL dataset or a JSON array dataset."""

    with open(input_path, encoding="utf-8") as f:
        first_non_ws = ""
        while True:
            pos = f.tell()
            ch = f.read(1)
            if not ch:
                break
            if not ch.isspace():
                first_non_ws = ch
                f.seek(pos)
                break

        if first_non_ws == "[":
            raw_items = json.load(f)
        else:
            raw_items = [json.loads(line) for line in f if line.strip()]

    samples = []
    for item in raw_items:
        samples.append(
            {
                "headline": item.get("headline", ""),
                "section": item.get(
                    "preprocessed_article_section",
                    item.get("preprocessed_section", item.get("article_section", "")),
                ),
                "description": item.get(
                    "preprocessed_description",
                    item.get("shuffled_preprocessed_description", item.get("description", "")),
                ),
                "label": item.get("is_sarcastic", 0),
            }
        )

    return samples


def shuffle_descriptions(samples: list[dict], bin_size: int = 5, seed: int = 42) -> list[dict]:
    """Shuffle descriptions within coarse length bins while keeping other fields fixed."""

    shuffled_samples = [dict(sample) for sample in samples]
    rng = random.Random(seed)
    bin_to_indices: dict[int, list[int]] = {}

    for idx, sample in enumerate(shuffled_samples):
        description = sample.get("description", "") or ""
        desc_length = len(description.split())
        bin_id = desc_length // bin_size
        bin_to_indices.setdefault(bin_id, []).append(idx)

    for indices in bin_to_indices.values():
        descriptions = [shuffled_samples[idx].get("description", "") or "" for idx in indices]
        rng.shuffle(descriptions)
        for idx, shuffled_description in zip(indices, descriptions):
            shuffled_samples[idx]["description"] = shuffled_description

    return shuffled_samples


def build_model_from_checkpoint(checkpoint: dict, device: torch.device, output_dir: str) -> TransformerGNNModel:
    cfg = TransformerGNNConfig(
        model_type=checkpoint.get("model_type", "roberta"),
        pretrained_name=checkpoint.get("pretrained_name", "roberta-base"),
        max_length=checkpoint.get("max_length", 128),
        dropout=checkpoint.get("dropout", 0.1),
        learning_rate=checkpoint.get("learning_rate", 2e-5),
        gnn_learning_rate=checkpoint.get("gnn_learning_rate", 1e-3),
        batch_size=checkpoint.get("batch_size", 32),
        num_epochs=checkpoint.get("num_epochs", 3),
        warmup_ratio=checkpoint.get("warmup_ratio", 0.1),
        edge_embed_dim=checkpoint.get("edge_embed_dim", 16),
        use_conceptnet=checkpoint.get("use_conceptnet", True),
        export_visualisations=False,
        output_dir=output_dir,
        weight_decay=checkpoint.get("weight_decay", 0.01),
        text_format=checkpoint.get("text_format", "headline"),
    )

    model = TransformerGNNModel(
        cfg,
        checkpoint["num_relations"],
        checkpoint["irf_weights"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained GNN model on a test split with shuffled descriptions.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/Preprocessed_Article_Section_Description.json",
        help="Path to the dataset file. Supports JSONL and JSON array formats.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="result/experiments",
        help="Directory to save logs and summary outputs.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the model checkpoint (.pt).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--bin-size",
        type=int,
        default=5,
        help="Description length bin size used before shuffling.",
    )

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    log_path = os.path.join(args.output, f"training_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
    setup_logger(path=log_path)

    set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info(f"Loading checkpoint from {args.model}...")
    checkpoint = torch.load(args.model, map_location=device)
    model = build_model_from_checkpoint(checkpoint, device, args.output)

    logging.info(f"Loading samples from {args.input}...")
    samples = load_samples(args.input)
    logging.info(f"Loaded {len(samples)} total samples.")

    _, data_test = train_test_split(samples, test_size=0.2, random_state=args.seed)
    logging.info(f"Recovered test split with {len(data_test)} samples using seed={args.seed}.")

    shuffled_test = shuffle_descriptions(data_test, bin_size=args.bin_size, seed=args.seed)
    logging.info(f"Shuffled descriptions within length bins of size {args.bin_size}.")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint["pretrained_name"])
    test_ds = SarcasmGraphDataset(
        shuffled_test,
        tokenizer,
        checkpoint["max_length"],
        use_conceptnet=checkpoint["use_conceptnet"],
        text_format=checkpoint["text_format"],
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=checkpoint["batch_size"],
        collate_fn=graph_collate_fn,
    )

    logging.info("Evaluating final model on shuffled test set...")
    loss_fn = nn.CrossEntropyLoss()
    test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(model, test_loader, loss_fn, device)

    final_results = (
        "\n=========================================\n"
        "        FINAL SHUFFLED TEST RESULTS      \n"
        "=========================================\n"
        f"Model:     {args.model}\n"
        f"Input:     {args.input}\n"
        f"Seed:      {args.seed}\n"
        f"Bin Size:  {args.bin_size}\n"
        f"Loss:      {test_loss:.4f}\n"
        f"Accuracy:  {test_acc:.4f}\n"
        f"Precision: {test_prec:.4f}\n"
        f"Recall:    {test_rec:.4f}\n"
        f"F1-Score:  {test_f1:.4f}\n"
        "=========================================\n"
    )

    logging.info(final_results)
    print(final_results)

    summary_path = os.path.join(args.output, "experiment_shuffle_description.txt")
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(final_results)


if __name__ == "__main__":
    main()
