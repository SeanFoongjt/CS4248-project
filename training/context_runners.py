from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .context_common import (
    ClassicalRunConfig,
    ContextRecipe,
    SplitConfig,
    TransformerRunConfig,
)
from models.tfidf_lr import TfidfLogRegModel, TfidfLrConfig
from models.tfidf_nb import TfidfNbConfig, TfidfNbModel


def build_classical_model(name: str, model_params: Optional[dict[str, Any]] = None):
    """Return one TF-IDF baseline model wrapper."""

    params = model_params or {}
    if name == "tfidf_nb":
        return TfidfNbModel(TfidfNbConfig(**params))
    if name == "tfidf_lr":
        return TfidfLogRegModel(TfidfLrConfig(**params))
    raise ValueError(f"Unknown classical model: {name}")


def build_transformer_wrapper(
    name: str,
    max_length: int,
    pretrained_name: Optional[str] = None,
    dropout: Optional[float] = None,
):
    """Return one transformer wrapper."""

    if name == "distilbert":
        from models.distilbert import DistilBertConfig, DistilBertSarcasmModel

        kwargs = {"max_length": max_length}
        if pretrained_name:
            kwargs["pretrained_name"] = pretrained_name
        return DistilBertSarcasmModel(DistilBertConfig(**kwargs))
    if name == "roberta":
        from models.roberta import RobertaConfig, RobertaSarcasmModel

        kwargs = {"max_length": max_length}
        if pretrained_name:
            kwargs["pretrained_name"] = pretrained_name
        if dropout is not None:
            kwargs["dropout"] = dropout
        return RobertaSarcasmModel(RobertaConfig(**kwargs))
    raise ValueError(f"Unknown transformer model: {name}")


class SimpleTextDataset(torch.utils.data.Dataset):
    """Small wrapper around the existing HF sarcasm dataset builder."""

    def __init__(self, base_ds):
        self.base_ds = base_ds

    def __len__(self) -> int:
        return len(self.base_ds)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.base_ds[idx]


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move one collated batch to the target device."""

    out: dict[str, Any] = {}
    for key, value in batch.items():
        out[key] = value.to(device) if hasattr(value, "to") else value
    return out


def evaluate_transformer(
    wrapper,
    loader: DataLoader,
    device: torch.device,
    compute_metrics: Callable[[list[int], list[int]], dict[str, float]],
) -> tuple[float, dict[str, float]]:
    """Run evaluation for one transformer model."""

    wrapper.model.eval()
    losses: list[float] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            out = wrapper.forward_batch(batch)
            losses.append(float(out.loss.detach().cpu()))
            pred = out.logits.argmax(dim=-1).detach().cpu().tolist()
            gold = batch["labels"].detach().cpu().tolist()
            y_pred.extend(pred)
            y_true.extend(gold)
    return float(np.mean(losses)) if losses else 0.0, compute_metrics(y_true, y_pred)


def train_one_epoch(
    wrapper,
    loader: DataLoader,
    optimizer: AdamW,
    scheduler,
    device: torch.device,
    clip_grad_norm: Callable[..., Any],
) -> float:
    """Train one epoch and return mean loss."""

    wrapper.model.train()
    losses: list[float] = []
    for batch in loader:
        batch = _move_batch(batch, device)
        optimizer.zero_grad()
        out = wrapper.forward_batch(batch)
        out.loss.backward()
        clip_grad_norm(wrapper.model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        losses.append(float(out.loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


def run_classical_recipe(
    recipe_frame: pd.DataFrame,
    recipe: ContextRecipe,
    cfg: ClassicalRunConfig,
    split_cfg: SplitConfig,
    *,
    split_frame: Callable[[pd.DataFrame, SplitConfig], dict[str, pd.DataFrame]],
    save_split_files: Callable[[str | Path, dict[str, pd.DataFrame]], None],
    save_json: Callable[[str | Path, dict[str, Any]], None],
    compute_metrics: Callable[[list[int], list[int]], dict[str, float]],
    build_classical_model_fn: Callable[..., Any],
) -> dict[str, Any]:
    """Train and evaluate one TF-IDF baseline on one recipe."""

    split_map = split_frame(recipe_frame, split_cfg)
    out_dir = Path(cfg.output_dir) / cfg.model_name / recipe.name
    save_split_files(out_dir / "splits", split_map)
    wrapper = build_classical_model_fn(cfg.model_name, model_params=cfg.model_params)
    pipe = wrapper.build_pipeline()
    train_df = split_map["train"]
    val_df = split_map["val"]
    test_df = split_map["test"]
    pipe.fit(train_df["text"], train_df["label"])
    val_pred = pipe.predict(val_df["text"])
    test_pred = pipe.predict(test_df["text"])
    metrics = {
        "model": cfg.model_name,
        "recipe": recipe.name,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "run_config": asdict(cfg),
        "model_config": asdict(wrapper.cfg),
        "val": compute_metrics(val_df["label"], val_pred),
        "test": compute_metrics(test_df["label"], test_pred),
    }
    save_json(out_dir / "metrics.json", metrics)
    pred_df = test_df.copy()
    pred_df["pred"] = test_pred
    pred_df.to_csv(out_dir / "test_predictions.csv", index=False)
    return metrics


def run_transformer_recipe(
    recipe_frame: pd.DataFrame,
    recipe: ContextRecipe,
    cfg: TransformerRunConfig,
    split_cfg: SplitConfig,
    *,
    set_seed: Callable[[int], None],
    pick_device: Callable[[str], torch.device],
    split_frame: Callable[[pd.DataFrame, SplitConfig], dict[str, pd.DataFrame]],
    save_split_files: Callable[[str | Path, dict[str, pd.DataFrame]], None],
    save_json: Callable[[str | Path, dict[str, Any]], None],
    compute_metrics: Callable[[list[int], list[int]], dict[str, float]],
    build_transformer_wrapper_fn: Callable[..., Any],
    clip_grad_norm: Callable[..., Any],
    epoch_callback: Optional[Callable[[int, dict[str, Any]], None]] = None,
) -> dict[str, Any]:
    """Train and evaluate one transformer on one recipe."""

    set_seed(cfg.seed)
    split_map = split_frame(recipe_frame, split_cfg)
    out_dir = Path(cfg.output_dir) / cfg.model_name / recipe.name
    save_split_files(out_dir / "splits", split_map)
    device = pick_device(cfg.device)
    wrapper = build_transformer_wrapper_fn(
        cfg.model_name,
        max_length=cfg.max_length,
        pretrained_name=cfg.pretrained_name,
        dropout=cfg.dropout,
    ).to(device)
    train_ds = SimpleTextDataset(
        wrapper.make_dataset(split_map["train"]["text"].tolist(), split_map["train"]["label"].tolist())
    )
    val_ds = SimpleTextDataset(
        wrapper.make_dataset(split_map["val"]["text"].tolist(), split_map["val"]["label"].tolist())
    )
    test_ds = SimpleTextDataset(
        wrapper.make_dataset(split_map["test"]["text"].tolist(), split_map["test"]["label"].tolist())
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=wrapper.collator)
    val_loader = DataLoader(val_ds, batch_size=cfg.eval_batch_size, shuffle=False, collate_fn=wrapper.collator)
    test_loader = DataLoader(test_ds, batch_size=cfg.eval_batch_size, shuffle=False, collate_fn=wrapper.collator)
    optimizer = AdamW(wrapper.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = max(1, len(train_loader) * cfg.epochs)
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    from transformers import get_linear_schedule_with_warmup

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    history: list[dict[str, Any]] = []
    best_state: Optional[dict[str, Any]] = None
    best_epoch: Optional[int] = None
    best_val_loss: Optional[float] = None
    best_val_metrics: Optional[dict[str, float]] = None
    best_train_loss: Optional[float] = None
    best_metric = -1.0
    for ep in range(cfg.epochs):
        train_loss = train_one_epoch(wrapper, train_loader, optimizer, scheduler, device, clip_grad_norm)
        val_loss, val_metrics = evaluate_transformer(wrapper, val_loader, device, compute_metrics)
        row = {
            "epoch": ep + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        if epoch_callback is not None:
            epoch_callback(ep + 1, row)
        if val_metrics["macro_f1"] > best_metric:
            best_metric = val_metrics["macro_f1"]
            best_epoch = ep + 1
            best_train_loss = train_loss
            best_val_loss = val_loss
            best_val_metrics = val_metrics
            best_state = {key: value.detach().cpu().clone() for key, value in wrapper.model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("No checkpoint was saved during training")
    wrapper.model.load_state_dict(best_state)
    if hasattr(wrapper, "save_pretrained"):
        wrapper.save_pretrained(out_dir / "checkpoint")
    else:
        wrapper.model.save_pretrained(out_dir / "checkpoint")
    wrapper.tokenizer.save_pretrained(out_dir / "checkpoint")
    test_loss, test_metrics = evaluate_transformer(wrapper, test_loader, device, compute_metrics)
    metrics = {
        "model": cfg.model_name,
        "recipe": recipe.name,
        "device": str(device),
        "n_train": int(len(split_map["train"])),
        "n_val": int(len(split_map["val"])),
        "n_test": int(len(split_map["test"])),
        "best_epoch": best_epoch,
        "train_loss": best_train_loss,
        "val_loss": best_val_loss,
        "val_accuracy": None if best_val_metrics is None else best_val_metrics["accuracy"],
        "val": best_val_metrics,
        "train_config": asdict(cfg),
        "model_config": asdict(wrapper.cfg),
        "history": history,
        "test_loss": test_loss,
        "test": test_metrics,
    }
    save_json(out_dir / "metrics.json", metrics)
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    return metrics
