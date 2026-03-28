from __future__ import annotations
import argparse
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader
from models.tfidf_lr import TfidfLogRegModel, TfidfLrConfig
from models.tfidf_nb import TfidfNbConfig, TfidfNbModel

def _load_preprocessors() -> tuple[Callable[..., str], Callable[..., str], Callable[..., str]]:
    """Load preprocessors lazily.
    We import these only when needed because `utils.preprocess` loads a large
    spaCy model at import time.
    """

    from utils.preprocess import (
        preprocess_article_section,
        preprocess_description,
        preprocess_for_bow,
    )
    
    return preprocess_article_section, preprocess_description, preprocess_for_bow

@dataclass(frozen=True)
class ContextRecipe:
    """One Variant 2 input recipe."""

    name: str
    needs_section: bool = False
    needs_description: bool = False

@dataclass
class SplitConfig:
    """How we split data for training and evaluation."""

    train_size: float = 0.8
    val_size: float = 0.1
    test_size: float = 0.1
    random_state: int = 42

@dataclass
class ClassicalRunConfig:
    """Training config for TF-IDF baselines."""

    model_name: str
    output_dir: str = "runs/variant2"
    model_params: dict[str, Any] = field(default_factory=dict)

@dataclass
class TransformerRunConfig:
    """Training config for DistilBERT/RoBERTa."""

    model_name: str
    output_dir: str = "runs/variant2"
    batch_size: int = 16
    eval_batch_size: int = 32
    lr: float = 2e-5
    weight_decay: float = 0.01
    epochs: int = 3
    warmup_ratio: float = 0.1
    max_length: int = 128
    device: str = "auto"
    seed: int = 42
    pretrained_name: Optional[str] = None

VARIANT2_RECIPES = (
    ContextRecipe(name="headline_section", needs_section=True),
    ContextRecipe(name="headline_description", needs_description=True),
    ContextRecipe(name="headline_section_description", needs_section=True, needs_description=True),
)

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def pick_device(name: str = "auto") -> torch.device:
    """Pick a device for torch code."""

    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_jsonl(path: str | Path) -> pd.DataFrame:
    """Read a json-lines dataset into a DataFrame."""

    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    df = pd.DataFrame(rows)
    df = df.copy()
    df["row_id"] = np.arange(len(df))
    return df

def load_variant2_data(path: str | Path, max_rows: Optional[int] = None) -> pd.DataFrame:
    """Load and prepare the Variant 2 dataframe once for repeated runs."""

    raw_df = load_jsonl(path)
    if max_rows is not None:
        raw_df = raw_df.iloc[:max_rows].copy().reset_index(drop=True)
    return prepare_variant2_frame(raw_df)

def _is_missing(value: Any) -> bool:
    """Return True when a value should be treated as missing."""

    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    return False

def _to_text(value: Any) -> str:
    """Convert metadata value to a plain string."""

    if _is_missing(value):
        return ""
    if isinstance(value, list):
        return " ".join(str(x).strip() for x in value if str(x).strip())
    return str(value).strip()

def _prep_headline_bow(text: Any) -> str:
    """Prepare headline text for TF-IDF models."""

    _, _, prep_bow = _load_preprocessors()
    return prep_bow(_to_text(text))

def _prep_headline_transformer(text: Any) -> str:
    """Prepare headline text for transformer models.
    We keep the surface form here because the existing `preprocess_for_bow`
    function is explicitly written for bag-of-words models.
    """

    return _to_text(text)

def _prep_section_raw(text: Any) -> str:
    """Standardize article section in a source-blind way."""

    prep_sec, _, _ = _load_preprocessors()
    raw = _to_text(text)
    if not raw:
        return ""
    return prep_sec(raw).strip()

def _prep_section_bow(text: Any) -> str:
    """Standardize article section, then prepare it for TF-IDF models."""

    _, _, prep_bow = _load_preprocessors()
    out = _prep_section_raw(text)
    if not out:
        return ""
    return prep_bow(out)

def _prep_description_raw(text: Any) -> str:
    """Mask description entities and remove obvious source clues."""

    _, prep_desc, _ = _load_preprocessors()
    raw = _to_text(text)
    if not raw:
        return ""
    return prep_desc(raw).strip()

def _prep_description_bow(text: Any) -> str:
    """Prepare description for TF-IDF models."""

    _, _, prep_bow = _load_preprocessors()
    out = _prep_description_raw(text)
    if not out:
        return ""
    return prep_bow(out)

def prepare_variant2_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Build all preprocessed columns needed for Variant 2.
    This function does not do general dataset cleaning. It only prepares the
    fields used by Variant 2.
    """

    out = df.copy()
    out["label"] = out["is_sarcastic"].astype(int)
    out["headline_bow"] = out["headline"].map(_prep_headline_bow)
    out["headline_tf"] = out["headline"].map(_prep_headline_transformer)
    out["section_raw"] = out["article_section"].map(_prep_section_raw)
    out["section_bow"] = out["article_section"].map(_prep_section_bow)
    out["description_raw"] = out["description"].map(_prep_description_raw)
    out["description_bow"] = out["description"].map(_prep_description_bow)
    return out

def _join_parts(parts: Iterable[str]) -> str:
    """Join non-empty text parts with one space."""

    return " ".join(part.strip() for part in parts if part and part.strip()).strip()

def build_recipe_text(row: pd.Series, recipe: ContextRecipe, family: str) -> str:
    """Build one training text for a given model family.
    Args:
        row: One row from the prepared dataframe.
        recipe: Which Variant 2 context recipe to use.
        family: Either `classical` or `transformer`.
    """

    if family not in {"classical", "transformer"}:
        raise ValueError(f"Unknown family: {family}")
    if family == "classical":
        h = row["headline_bow"]
        sec = row["section_bow"]
        desc = row["description_bow"]
    else:
        h = row["headline_tf"]
        sec = row["section_raw"]
        desc = row["description_raw"]
    parts = [f"headline: {h}"]
    if recipe.needs_section:
        parts.append(f"section: {sec}")
    if recipe.needs_description:
        parts.append(f"description: {desc}")
    return _join_parts(parts)

def subset_for_recipe(df: pd.DataFrame, recipe: ContextRecipe) -> pd.DataFrame:
    """Keep only rows that truly satisfy the context recipe.
    We do not allow headline-only rows in Variant 2. If a recipe needs a field,
    that field must be present after preprocessing.
    """

    mask = pd.Series(True, index=df.index)
    if recipe.needs_section:
        mask &= df["section_raw"].astype(str).str.len() > 0
    if recipe.needs_description:
        mask &= df["description_raw"].astype(str).str.len() > 0
    out = df.loc[mask].copy().reset_index(drop=True)
    return out

def build_recipe_frame(df: pd.DataFrame, recipe: ContextRecipe, family: str) -> pd.DataFrame:
    """Build the actual text column used by one recipe/model-family pair."""

    sub = subset_for_recipe(df, recipe)
    sub = sub.copy()
    sub["text"] = sub.apply(build_recipe_text, axis=1, recipe=recipe, family=family)
    return sub[["row_id", "label", "text"]].copy()

def _safe_stratify(labels: np.ndarray) -> Optional[np.ndarray]:
    """Return labels when stratification is safe, else None.
    This mainly matters for tiny smoke tests. On the full dataset the normal
    stratified path should be used.
    """

    if len(labels) < 4:
        return None
    _, counts = np.unique(labels, return_counts=True)
    if counts.min() < 2:
        return None
    return labels

def split_frame(df: pd.DataFrame, cfg: SplitConfig) -> dict[str, pd.DataFrame]:
    """Create train/val/test splits.
    We stratify whenever it is safe. For very small smoke tests we fall back to
    non-stratified splitting for the tiny holdout split.
    """

    total = cfg.train_size + cfg.val_size + cfg.test_size
    if abs(total - 1.0) > 1e-8:
        raise ValueError("Split sizes must sum to 1.0")
    idx = np.arange(len(df))
    labels = df["label"].to_numpy()
    train_idx, hold_idx = train_test_split(
        idx,
        test_size=cfg.val_size + cfg.test_size,
        random_state=cfg.random_state,
        stratify=_safe_stratify(labels),
    )
    hold_labels = labels[hold_idx]
    rel_test_size = cfg.test_size / (cfg.val_size + cfg.test_size)
    val_idx, test_idx = train_test_split(
        hold_idx,
        test_size=rel_test_size,
        random_state=cfg.random_state,
        stratify=_safe_stratify(hold_labels),
    )
    return {
        "train": df.iloc[train_idx].reset_index(drop=True),
        "val": df.iloc[val_idx].reset_index(drop=True),
        "test": df.iloc[test_idx].reset_index(drop=True),
    }

def compute_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> dict[str, float]:
    """Compute the main classification metrics for binary sarcasm detection."""

    y_true = list(y_true)
    y_pred = list(y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    bin_p, bin_r, bin_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "precision": float(bin_p),
        "recall": float(bin_r),
        "f1": float(bin_f1),
    }

def build_classical_model(name: str, model_params: Optional[dict[str, Any]] = None):
    """Return one TF-IDF baseline model wrapper."""

    model_params = model_params or {}
    if name == "tfidf_nb":
        return TfidfNbModel(TfidfNbConfig(**model_params))
    if name == "tfidf_lr":
        return TfidfLogRegModel(TfidfLrConfig(**model_params))
    raise ValueError(f"Unknown classical model: {name}")

def build_transformer_wrapper(name: str, max_length: int, pretrained_name: Optional[str] = None):
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
        return RobertaSarcasmModel(RobertaConfig(**kwargs))
    raise ValueError(f"Unknown transformer model: {name}")

def save_json(path: str | Path, obj: dict[str, Any]) -> None:
    """Save one dict as pretty JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def save_split_files(out_dir: str | Path, split_map: dict[str, pd.DataFrame]) -> None:
    """Save splits so later runs stay comparable."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in split_map.items():
        frame.to_csv(out_dir / f"{name}.csv", index=False)

def run_classical_recipe(
    recipe_frame: pd.DataFrame,
    recipe: ContextRecipe,
    cfg: ClassicalRunConfig,
    split_cfg: SplitConfig,
) -> dict[str, Any]:
    """Train and evaluate one TF-IDF baseline on one recipe."""

    split_map = split_frame(recipe_frame, split_cfg)
    out_dir = Path(cfg.output_dir) / cfg.model_name / recipe.name
    save_split_files(out_dir / "splits", split_map)
    wrapper = build_classical_model(cfg.model_name, model_params=cfg.model_params)
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

    out = {}
    for k, v in batch.items():
        if hasattr(v, "to"):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out

def evaluate_transformer(wrapper, loader: DataLoader, device: torch.device) -> tuple[float, dict[str, float]]:
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
    metrics = compute_metrics(y_true, y_pred)
    avg_loss = float(np.mean(losses)) if losses else 0.0
    return avg_loss, metrics

def train_one_epoch(
    wrapper,
    loader: DataLoader,
    optimizer: AdamW,
    scheduler,
    device: torch.device,
) -> float:
    """Train one epoch and return mean loss."""

    wrapper.model.train()
    losses: list[float] = []
    for batch in loader:
        batch = _move_batch(batch, device)
        optimizer.zero_grad()
        out = wrapper.forward_batch(batch)
        out.loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        losses.append(float(out.loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0

def run_transformer_recipe(
    recipe_frame: pd.DataFrame,
    recipe: ContextRecipe,
    cfg: TransformerRunConfig,
    split_cfg: SplitConfig,
    epoch_callback: Optional[Callable[[int, dict[str, Any]], None]] = None,
) -> dict[str, Any]:
    """Train and evaluate one transformer on one recipe."""

    set_seed(cfg.seed)
    split_map = split_frame(recipe_frame, split_cfg)
    out_dir = Path(cfg.output_dir) / cfg.model_name / recipe.name
    save_split_files(out_dir / "splits", split_map)
    device = pick_device(cfg.device)
    wrapper = build_transformer_wrapper(
        cfg.model_name,
        max_length=cfg.max_length,
        pretrained_name=cfg.pretrained_name,
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
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=wrapper.collator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        collate_fn=wrapper.collator,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        collate_fn=wrapper.collator,
    )
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
    best_metric = -1.0
    for ep in range(cfg.epochs):
        train_loss = train_one_epoch(wrapper, train_loader, optimizer, scheduler, device)
        val_loss, val_metrics = evaluate_transformer(wrapper, val_loader, device)
        row = {
            "epoch": ep + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        if epoch_callback is not None:
            epoch_callback(ep + 1, row)
        if val_metrics["macro_f1"] > best_metric:
            best_metric = val_metrics["macro_f1"]
            best_epoch = ep + 1
            best_val_loss = val_loss
            best_val_metrics = val_metrics
            best_state = {k: v.detach().cpu().clone() for k, v in wrapper.model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("No checkpoint was saved during training")
    wrapper.model.load_state_dict(best_state)
    wrapper.model.save_pretrained(out_dir / "checkpoint")
    wrapper.tokenizer.save_pretrained(out_dir / "checkpoint")
    test_loss, test_metrics = evaluate_transformer(wrapper, test_loader, device)
    metrics = {
        "model": cfg.model_name,
        "recipe": recipe.name,
        "device": str(device),
        "n_train": int(len(split_map["train"])),
        "n_val": int(len(split_map["val"])),
        "n_test": int(len(split_map["test"])),
        "best_epoch": best_epoch,
        "val_loss": best_val_loss,
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

def build_run_plan(
    classical_models: Iterable[str],
    transformer_models: Iterable[str],
) -> list[tuple[str, str]]:
    """Build the list of runs we want to execute."""

    plan: list[tuple[str, str]] = []
    for model_name in classical_models:
        for recipe in VARIANT2_RECIPES:
            plan.append((model_name, recipe.name))
    for model_name in transformer_models:
        for recipe in VARIANT2_RECIPES:
            plan.append((model_name, recipe.name))
    return plan

def _recipe_by_name(name: str) -> ContextRecipe:
    """Find a recipe object by name."""

    for recipe in VARIANT2_RECIPES:
        if recipe.name == name:
            return recipe
    raise ValueError(f"Unknown recipe: {name}")

def run_variant2_suite(
    data_path: str | Path,
    output_dir: str | Path,
    split_cfg: SplitConfig,
    classical_models: Iterable[str],
    transformer_models: Iterable[str],
    max_rows: Optional[int] = None,
    transformer_cfg: Optional[TransformerRunConfig] = None,
) -> list[dict[str, Any]]:
    """Run every requested Variant 2 experiment.
    The same recipe-specific subset and split logic is used for all models.
    """

    prep_df = load_variant2_data(data_path, max_rows=max_rows)
    results: list[dict[str, Any]] = []
    for recipe in VARIANT2_RECIPES:
        cls_df = build_recipe_frame(prep_df, recipe, family="classical")
        tf_df = build_recipe_frame(prep_df, recipe, family="transformer")
        for model_name in classical_models:
            cfg = ClassicalRunConfig(model_name=model_name, output_dir=str(output_dir))
            results.append(run_classical_recipe(cls_df, recipe, cfg, split_cfg))
        for model_name in transformer_models:
            base_cfg = transformer_cfg or TransformerRunConfig(model_name=model_name, output_dir=str(output_dir))
            cfg = TransformerRunConfig(**{**asdict(base_cfg), "model_name": model_name, "output_dir": str(output_dir)})
            results.append(run_transformer_recipe(tf_df, recipe, cfg, split_cfg))
    save_json(Path(output_dir) / "summary.json", {"runs": results})
    return results

def parse_args() -> argparse.Namespace:
    """Parse CLI args for manual runs."""

    p = argparse.ArgumentParser(description="Run Variant 2 context experiments")
    p.add_argument("--data", required=True, help="Path to Sarcasm_Headlines_Dataset_With_Metadata.json")
    p.add_argument("--out", default="runs/variant2", help="Output dir")
    p.add_argument(
        "--classical",
        nargs="*",
        default=["tfidf_nb", "tfidf_lr"],
        help="Classical models to run",
    )
    p.add_argument(
        "--transformers",
        nargs="*",
        default=[],
        help="Transformer models to run",
    )
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-rows", type=int, default=None)
    return p.parse_args()

def main() -> None:
    """CLI entry point."""

    args = parse_args()
    split_cfg = SplitConfig()
    tf_cfg = TransformerRunConfig(
        model_name="distilbert",
        output_dir=args.out,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        max_length=args.max_length,
        device=args.device,
        seed=args.seed,
    )
    run_variant2_suite(
        data_path=args.data,
        output_dir=args.out,
        split_cfg=split_cfg,
        classical_models=args.classical,
        transformer_models=args.transformers,
        max_rows=args.max_rows,
        transformer_cfg=tf_cfg,
    )

if __name__ == "__main__":
    main()
