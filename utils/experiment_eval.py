import json
import logging
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from models.general_conceptnet_gnn_pipeline import (
    SarcasmGraphDataset,
    TransformerGNNConfig,
    TransformerGNNModel,
    graph_collate_fn,
)


def discover_default_all_checkpoints(result_root: str | Path = "result") -> list[str]:
    """Find tuned checkpoints for the four `all` model variants."""
    root = Path(result_root)
    candidates = sorted(root.glob("*_all_*/final_best_model/*.pt"))
    return [str(path).replace("\\", "/") for path in candidates]


def load_json_records(input_path: str | Path) -> list[dict]:
    path = Path(input_path)
    text = path.read_text(encoding="utf-8").lstrip()

    if not text:
        return []

    if text.startswith("["):
        return json.loads(text)

    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_samples(input_path: str | Path) -> list[dict]:
    """Load the dataset into the sample schema expected by SarcasmGraphDataset."""
    samples = []
    for item in load_json_records(input_path):
        samples.append(
            {
                "headline": item.get("headline", ""),
                "section": item.get(
                    "preprocessed_article_section",
                    item.get("section", ""),
                ),
                "description": item.get(
                    "preprocessed_description",
                    item.get("description", ""),
                ),
                "label": item.get("is_sarcastic", item.get("label", 0)),
            }
        )
    return samples


def torch_load_checkpoint(model_path: str | Path, device: torch.device) -> dict:
    try:
        return torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(model_path, map_location=device)


def config_from_checkpoint(checkpoint: dict, output_dir: str | Path) -> TransformerGNNConfig:
    return TransformerGNNConfig(
        model_type=checkpoint.get("model_type", "roberta"),
        pretrained_name=checkpoint.get("pretrained_name", "roberta-base"),
        use_conceptnet=checkpoint.get("use_conceptnet", True),
        output_dir=str(output_dir),
        text_format=checkpoint.get("text_format", "headline"),
        gnn_learning_rate=checkpoint.get("gnn_learning_rate", 1e-3),
        dropout=checkpoint.get("dropout", 0.1),
        weight_decay=checkpoint.get("weight_decay", 0.01),
        warmup_ratio=checkpoint.get("warmup_ratio", 0.1),
        edge_embed_dim=checkpoint.get("edge_embed_dim", 16),
        max_length=checkpoint.get("max_length", 128),
        num_epochs=checkpoint.get("num_epochs", 3),
        learning_rate=checkpoint.get("learning_rate", 2e-5),
        batch_size=checkpoint.get("batch_size", 32),
    )


def build_model_from_checkpoint(
    model_path: str | Path,
    output_dir: str | Path,
    device: torch.device,
) -> tuple[TransformerGNNModel, TransformerGNNConfig, dict]:
    checkpoint = torch_load_checkpoint(model_path, device)
    cfg = config_from_checkpoint(checkpoint, output_dir)

    model = TransformerGNNModel(
        cfg,
        checkpoint["num_relations"],
        checkpoint["irf_weights"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, cfg, checkpoint


def model_metadata(
    model_path: str | Path,
    cfg: TransformerGNNConfig,
) -> dict:
    return {
        "model_path": str(model_path).replace("\\", "/"),
        "model_type": cfg.model_type,
        "pretrained_name": cfg.pretrained_name,
        "use_conceptnet": cfg.use_conceptnet,
        "text_format": cfg.text_format,
        "max_length": cfg.max_length,
        "batch_size": cfg.batch_size,
    }


def evaluate_and_predict_checkpoint(
    model_path: str | Path,
    samples: list[dict],
    output_dir: str | Path,
    device: torch.device,
    prediction_extra_fields: Iterable[str] = (),
) -> tuple[dict, list[dict]]:
    """Evaluate a checkpoint and collect per-sample predictions in one graph build."""
    model, cfg, _ = build_model_from_checkpoint(model_path, output_dir, device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_name)
    dataset = SarcasmGraphDataset(
        samples,
        tokenizer,
        cfg.max_length,
        use_conceptnet=cfg.use_conceptnet,
        text_format=cfg.text_format,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        collate_fn=graph_collate_fn,
    )
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    all_preds = []
    all_labels = []
    prediction_rows = []
    sample_offset = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            edge_index = batch["edge_index"].to(device)
            edge_attr = batch["edge_attr"].to(device)
            edge_weight = batch["edge_weight"].to(device)

            logits = model(input_ids, attention_mask, edge_index, edge_attr, edge_weight)
            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)
            batch_size = labels.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            for local_idx in range(batch_size):
                sample = samples[sample_offset + local_idx]
                prob_not_sarcastic = probs[local_idx, 0].item()
                prob_sarcastic = probs[local_idx, 1].item()
                predicted_label = preds[local_idx].item()
                true_label = labels[local_idx].item()
                row = {
                    **model_metadata(model_path, cfg),
                    "headline": sample.get("headline", ""),
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "correct": predicted_label == true_label,
                    "prob_not_sarcastic": prob_not_sarcastic,
                    "prob_sarcastic": prob_sarcastic,
                    "confidence": max(prob_not_sarcastic, prob_sarcastic),
                }
                for field in prediction_extra_fields:
                    row[field] = sample.get(field, "")
                prediction_rows.append(row)

            sample_offset += batch_size

    accuracy = (
        sum(pred == label for pred, label in zip(all_preds, all_labels)) / len(all_labels)
        if all_labels
        else 0
    )
    result = {
        **model_metadata(model_path, cfg),
        "loss": total_loss / len(loader) if len(loader) else 0,
        "accuracy": accuracy,
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
    }

    return result, prediction_rows


def evaluate_checkpoint(
    model_path: str | Path,
    samples: list[dict],
    output_dir: str | Path,
    device: torch.device,
) -> dict:
    result, _ = evaluate_and_predict_checkpoint(model_path, samples, output_dir, device)
    return result


def predict_checkpoint(
    model_path: str | Path,
    samples: list[dict],
    output_dir: str | Path,
    device: torch.device,
    prediction_extra_fields: Iterable[str] = (),
) -> list[dict]:
    _, prediction_rows = evaluate_and_predict_checkpoint(
        model_path,
        samples,
        output_dir,
        device,
        prediction_extra_fields=prediction_extra_fields,
    )
    return prediction_rows


def log_checkpoints(model_paths: list[str]) -> None:
    logging.info("Evaluating %d checkpoint(s):", len(model_paths))
    for model_path in model_paths:
        logging.info(" - %s", model_path)
