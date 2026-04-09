from __future__ import annotations

import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


def discover_default_all_checkpoints(result_root: str = "result") -> list[str]:
    """Return the four tuned checkpoints trained with the `all` recipe."""

    root = Path(result_root)
    checkpoints = []

    for path in root.rglob("sarcasm_gnn_model_tuned_*.pt"):
        run_dir = path.parents[1].name
        if "_all_" in run_dir:
            checkpoints.append(path)

    return [str(path) for path in sorted(checkpoints)]


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


def build_model_from_checkpoint(checkpoint: dict, device: torch.device, output_dir: str):
    from models.general_conceptnet_gnn_pipeline import TransformerGNNConfig, TransformerGNNModel

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


def evaluate_checkpoint(model_path: str, samples: list[dict], output_dir: str, device: torch.device) -> dict:
    from models.general_conceptnet_gnn_pipeline import SarcasmGraphDataset, graph_collate_fn, evaluate

    checkpoint = torch.load(model_path, map_location=device)
    model = build_model_from_checkpoint(checkpoint, device, output_dir)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint["pretrained_name"])
    test_ds = SarcasmGraphDataset(
        samples,
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

    loss_fn = nn.CrossEntropyLoss()
    test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(model, test_loader, loss_fn, device)

    return {
        "model_path": model_path,
        "model_type": checkpoint.get("model_type", "roberta"),
        "pretrained_name": checkpoint.get("pretrained_name", "roberta-base"),
        "use_conceptnet": checkpoint.get("use_conceptnet", True),
        "text_format": checkpoint.get("text_format", "headline"),
        "max_length": checkpoint.get("max_length", 128),
        "batch_size": checkpoint.get("batch_size", 32),
        "loss": test_loss,
        "accuracy": test_acc,
        "precision": test_prec,
        "recall": test_rec,
        "f1": test_f1,
    }


def predict_checkpoint(model_path: str, samples: list[dict], output_dir: str, device: torch.device) -> list[dict]:
    from models.general_conceptnet_gnn_pipeline import SarcasmGraphDataset, graph_collate_fn

    checkpoint = torch.load(model_path, map_location=device)
    model = build_model_from_checkpoint(checkpoint, device, output_dir)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint["pretrained_name"])
    test_ds = SarcasmGraphDataset(
        samples,
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

    predictions = []
    model.eval()

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            edge_index = batch["edge_index"].to(device)
            edge_attr = batch["edge_attr"].to(device)
            edge_weight = batch["edge_weight"].to(device)

            logits = model(input_ids, attention_mask, edge_index, edge_attr, edge_weight)
            probs = F.softmax(logits, dim=-1).cpu()
            preds = logits.argmax(dim=-1).cpu()
            labels = batch["label"].cpu()
            headlines = batch["headlines"]

            for headline, label, pred, prob in zip(headlines, labels, preds, probs):
                predictions.append(
                    {
                        "model_path": model_path,
                        "model_type": checkpoint.get("model_type", "roberta"),
                        "pretrained_name": checkpoint.get("pretrained_name", "roberta-base"),
                        "use_conceptnet": checkpoint.get("use_conceptnet", True),
                        "text_format": checkpoint.get("text_format", "headline"),
                        "headline": headline,
                        "true_label": int(label.item()),
                        "predicted_label": int(pred.item()),
                        "correct": bool(label.item() == pred.item()),
                        "prob_not_sarcastic": float(prob[0].item()),
                        "prob_sarcastic": float(prob[1].item()),
                        "confidence": float(prob.max().item()),
                    }
                )

    return predictions
