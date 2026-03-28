from __future__ import annotations

import argparse
from typing import Any, Callable, Iterable, Optional

import pandas as pd
import torch.nn as nn

from .context_common import (
    ClassicalRunConfig,
    ContextRecipe,
    SplitConfig,
    TransformerRunConfig,
    VARIANT2_RECIPES,
    build_run_plan,
    clone_transformer_config,
    compute_metrics,
    default_transformer_run_config,
    load_jsonl,
    pick_device,
    recipe_by_name,
    save_json,
    save_split_files,
    set_seed,
    split_frame,
)
from .context_data import build_recipe_frame as _build_recipe_frame_impl
from .context_data import build_recipe_text as _build_recipe_text_impl
from .context_data import load_prepared_data
from .context_data import prepare_frame as _prepare_frame_impl
from .context_data import subset_for_recipe as _subset_for_recipe_impl
from .context_runners import (
    SimpleTextDataset,
    build_classical_model as _build_classical_model_impl,
    build_transformer_wrapper as _build_transformer_wrapper_impl,
    evaluate_transformer,
    run_classical_recipe as _run_classical_recipe_impl,
    run_transformer_recipe as _run_transformer_recipe_impl,
    train_one_epoch,
)


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


def load_variant2_data(path: str, max_rows: Optional[int] = None) -> pd.DataFrame:
    """Load and prepare the shared experiment dataframe once for repeated runs."""

    return load_prepared_data(path, prepare_variant2_frame, max_rows=max_rows)


def prepare_variant2_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Build all preprocessed columns needed for the shared input recipes."""

    return _prepare_frame_impl(df, _load_preprocessors)


def build_recipe_text(row: pd.Series, recipe: ContextRecipe, family: str) -> str:
    """Build one training text for a given model family."""

    return _build_recipe_text_impl(row, recipe, family)


def subset_for_recipe(df: pd.DataFrame, recipe: ContextRecipe) -> pd.DataFrame:
    """Keep only rows that truly satisfy the context recipe."""

    return _subset_for_recipe_impl(df, recipe)


def build_recipe_frame(df: pd.DataFrame, recipe: ContextRecipe, family: str) -> pd.DataFrame:
    """Build the actual text column used by one recipe/model-family pair."""

    return _build_recipe_frame_impl(df, recipe, family)


def build_classical_model(name: str, model_params: Optional[dict[str, Any]] = None):
    """Return one TF-IDF baseline model wrapper."""

    return _build_classical_model_impl(name, model_params=model_params)


def build_transformer_wrapper(
    name: str,
    max_length: int,
    pretrained_name: Optional[str] = None,
    dropout: Optional[float] = None,
):
    """Return one transformer wrapper."""

    return _build_transformer_wrapper_impl(
        name,
        max_length=max_length,
        pretrained_name=pretrained_name,
        dropout=dropout,
    )


def run_classical_recipe(
    recipe_frame: pd.DataFrame,
    recipe: ContextRecipe,
    cfg: ClassicalRunConfig,
    split_cfg: SplitConfig,
) -> dict[str, Any]:
    """Train and evaluate one TF-IDF baseline on one recipe."""

    return _run_classical_recipe_impl(
        recipe_frame,
        recipe,
        cfg,
        split_cfg,
        split_frame=split_frame,
        save_split_files=save_split_files,
        save_json=save_json,
        compute_metrics=compute_metrics,
        build_classical_model_fn=build_classical_model,
    )


def run_transformer_recipe(
    recipe_frame: pd.DataFrame,
    recipe: ContextRecipe,
    cfg: TransformerRunConfig,
    split_cfg: SplitConfig,
    epoch_callback: Optional[Callable[[int, dict[str, Any]], None]] = None,
) -> dict[str, Any]:
    """Train and evaluate one transformer on one recipe."""

    return _run_transformer_recipe_impl(
        recipe_frame,
        recipe,
        cfg,
        split_cfg,
        set_seed=set_seed,
        pick_device=pick_device,
        split_frame=split_frame,
        save_split_files=save_split_files,
        save_json=save_json,
        compute_metrics=compute_metrics,
        build_transformer_wrapper_fn=build_transformer_wrapper,
        clip_grad_norm=nn.utils.clip_grad_norm_,
        epoch_callback=epoch_callback,
    )


def _recipe_by_name(name: str) -> ContextRecipe:
    """Find a recipe object by name."""

    return recipe_by_name(name)


def run_variant2_suite(
    data_path: str,
    output_dir: str,
    split_cfg: SplitConfig,
    classical_models: Iterable[str],
    transformer_models: Iterable[str],
    max_rows: Optional[int] = None,
    transformer_cfg: Optional[TransformerRunConfig] = None,
) -> list[dict[str, Any]]:
    """Run every requested experiment with shared recipe-specific logic."""

    prep_df = load_variant2_data(data_path, max_rows=max_rows)
    results: list[dict[str, Any]] = []
    for recipe in VARIANT2_RECIPES:
        cls_df = build_recipe_frame(prep_df, recipe, family="classical")
        tf_df = build_recipe_frame(prep_df, recipe, family="transformer")
        for model_name in classical_models:
            cfg = ClassicalRunConfig(model_name=model_name, output_dir=str(output_dir))
            results.append(run_classical_recipe(cls_df, recipe, cfg, split_cfg))
        for model_name in transformer_models:
            base_cfg = transformer_cfg or default_transformer_run_config(model_name=model_name, output_dir=str(output_dir))
            cfg = clone_transformer_config(base_cfg, model_name=model_name, output_dir=str(output_dir))
            results.append(run_transformer_recipe(tf_df, recipe, cfg, split_cfg))
    save_json(str(output_dir) + "/summary.json", {"runs": results})
    return results


def parse_args() -> argparse.Namespace:
    """Parse CLI args for manual runs."""

    parser = argparse.ArgumentParser(description="Run shared sarcasm experiments across input variants")
    parser.add_argument("--data", required=True, help="Path to Sarcasm_Headlines_Dataset_With_Metadata.json")
    parser.add_argument("--out", default="runs/variant2", help="Output dir")
    parser.add_argument(
        "--classical",
        nargs="*",
        default=["tfidf_nb", "tfidf_lr"],
        help="Classical models to run",
    )
    parser.add_argument(
        "--transformers",
        nargs="*",
        default=[],
        help="Transformer models to run",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    split_cfg = SplitConfig()
    tf_cfg = None
    if any(
        [
            args.batch_size is not None,
            args.eval_batch_size is not None,
            args.lr != 2e-5,
            args.weight_decay != 0.01,
            args.epochs != 3,
            args.max_length != 128,
            args.device != "auto",
            args.seed != 42,
        ]
    ):
        tf_cfg = default_transformer_run_config(
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
