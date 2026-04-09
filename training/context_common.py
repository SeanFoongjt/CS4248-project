from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class ContextRecipe:
    """One supported input recipe for the shared sarcasm pipeline."""

    name: str
    needs_section: bool = False
    needs_description: bool = False


@dataclass
class SplitConfig:
    """How we split data for training and evaluation."""

    train_size: float = 0.6
    val_size: float = 0.2
    test_size: float = 0.2
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
    early_stopping_patience: Optional[int] = None
    early_stopping_min_delta: float = 0.0
    pretrained_name: Optional[str] = None
    dropout: Optional[float] = None


VARIANT2_RECIPES = (
    ContextRecipe(name="headline_only"),
    ContextRecipe(name="headline_section", needs_section=True),
    ContextRecipe(name="headline_section_description", needs_section=True, needs_description=True),
)


def default_transformer_run_config(model_name: str, output_dir: str, **overrides: Any) -> TransformerRunConfig:
    """Build a transformer run config with model-specific defaults."""

    defaults: dict[str, Any] = {
        "model_name": model_name,
        "output_dir": output_dir,
    }
    if model_name == "roberta":
        defaults["batch_size"] = 32
    clean_overrides = {key: value for key, value in overrides.items() if value is not None}
    return TransformerRunConfig(**{**defaults, **clean_overrides})


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
    # Some exported dataset files begin with a UTF-8 BOM. Using utf-8-sig
    # keeps standard UTF-8 behavior while transparently stripping the BOM.
    with open(path, "r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    df = pd.DataFrame(rows).copy()
    df["row_id"] = np.arange(len(df))
    return df


def _safe_stratify(labels: np.ndarray) -> Optional[np.ndarray]:
    """Return labels when stratification is safe, else None."""

    if len(labels) < 4:
        return None
    _, counts = np.unique(labels, return_counts=True)
    if counts.min() < 2:
        return None
    return labels


def split_frame(df: pd.DataFrame, cfg: SplitConfig) -> dict[str, pd.DataFrame]:
    """Create train/val/test splits."""
    from sklearn.model_selection import train_test_split

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
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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


def save_json(path: str | Path, obj: dict[str, Any]) -> None:
    """Save one dict as pretty JSON."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)


def save_split_files(out_dir: str | Path, split_map: dict[str, pd.DataFrame]) -> None:
    """Save splits so later runs stay comparable."""

    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    for name, frame in split_map.items():
        frame.to_csv(target / f"{name}.csv", index=False)


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


def recipe_by_name(name: str) -> ContextRecipe:
    """Find a recipe object by name."""

    for recipe in VARIANT2_RECIPES:
        if recipe.name == name:
            return recipe
    raise ValueError(f"Unknown recipe: {name}")


def clone_transformer_config(
    base_cfg: TransformerRunConfig,
    model_name: str,
    output_dir: str,
) -> TransformerRunConfig:
    """Clone a transformer config and override model/output fields."""

    cfg = asdict(base_cfg)
    cfg["model_name"] = model_name
    cfg["output_dir"] = output_dir
    return default_transformer_run_config(**cfg)