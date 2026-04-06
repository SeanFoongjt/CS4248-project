from __future__ import annotations
import argparse
import ast
import json
import random
import re
import sys
import pandas as pd
import torch
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from statistics import mean, median
from typing import Iterable, Sequence
from models.roberta import RobertaSarcasmModel
from training.context_common import load_jsonl, pick_device, recipe_by_name
from utils.constants import SECTION_MAPPINGS

EMPTY_TOKEN = "[EMPTY]"

@dataclass(frozen=True)
class VariantResult:
    """One counterfactual input and its effect on the base prediction."""

    name: str
    text: str
    predicted_label: int
    non_sarcastic_prob: float
    sarcastic_prob: float
    confidence_delta: float
    sarcastic_delta: float
    label_flip: bool

@dataclass
class RowAnalysis:
    """Analysis record for one dataset row."""

    row_id: int
    label: int | None
    headline: str
    section: str
    description: str
    base_text: str
    base_predicted_label: int
    base_non_sarcastic_prob: float
    base_sarcastic_prob: float
    variants: list[VariantResult]

def resolve_checkpoint_dir(path: str | Path) -> Path:
    """Accept either a raw checkpoint dir or a study dir."""

    candidate = Path(path).resolve()
    if (candidate / "model_state.pt").exists() and (candidate / "model_config.json").exists():
        return candidate
    best_model_meta = candidate / "best_model" / "materialization.json"
    if best_model_meta.exists():
        meta = json.loads(best_model_meta.read_text(encoding="utf-8"))
        return Path(meta["best_model_path"]).resolve()

    raise ValueError(f"Could not resolve a RoBERTa checkpoint from: {candidate}")

def _normalize_space(text: object) -> str:
    """Normalize whitespace and safely handle missing values."""

    if text is None:
        return ""
    if isinstance(text, float) and pd.isna(text):
        return ""
    text = str(text).strip()
    if not text:
        return ""
    return re.sub(r"\s+", " ", text)

def preprocess_section_local(value: str | None, mappings: dict[str, str] = SECTION_MAPPINGS) -> str:
    """Match the training-side section normalization as closely as possible."""

    if not value:
        return ""
    if isinstance(value, list):
        items = value
    elif isinstance(value, str) and value.startswith("["):
        try:
            items = ast.literal_eval(value)
        except Exception:
            items = [value]
    else:
        items = [value]
    mapped = [mappings.get(str(item).lower(), "other") for item in items]
    mapped = [item for item in dict.fromkeys(mapped) if item]
    return ", ".join(mapped)

def preprocess_description_local(text: str | None) -> str:
    """Apply the same lightweight cleanup used in the analysis script."""

    text = _normalize_space(text)
    if not text:
        return ""
    return re.sub(r"^[A-Z\s,.]+—", "", text).strip()

def build_recipe_text(
    recipe_name: str,
    headline: str,
    *,
    section: str | None = None,
    description: str | None = None,
    empty_token: str = EMPTY_TOKEN,
) -> str:
    """Serialize the input exactly in the shared recipe style."""

    recipe = recipe_by_name(recipe_name)
    head = _normalize_space(headline) or empty_token
    sec = preprocess_section_local(section) or empty_token
    desc = preprocess_description_local(description) or empty_token
    parts = [f"headline: {head}"]
    if recipe.needs_section:
        parts.append(f"section: {sec}")
    if recipe.needs_description:
        parts.append(f"description: {desc}")
    return " ".join(parts)

def shuffle_words(text: str, rng: random.Random) -> str:
    """Shuffle words but keep the bag of words identical."""

    tokens = text.split()
    if len(tokens) <= 1:
        return text
    out = list(tokens)
    rng.shuffle(out)
    return " ".join(out)

def truncate_words(text: str, max_words: int) -> str:
    """Keep only the first max_words words."""

    if max_words <= 0:
        return ""
    return " ".join(text.split()[:max_words])

def drop_last_words(text: str, n_words: int) -> str:
    """Drop the last n_words words from a text."""

    tokens = text.split()
    if n_words <= 0 or len(tokens) <= n_words:
        return ""
    return " ".join(tokens[:-n_words])

def drop_first_words(text: str, n_words: int) -> str:
    """Drop the first n_words words from a text."""

    tokens = text.split()
    if n_words <= 0 or len(tokens) <= n_words:
        return ""
    return " ".join(tokens[n_words:])

def load_input_rows(path: str | Path) -> pd.DataFrame:
    """Read either CSV or JSONL input rows."""

    src = Path(path)
    suffix = src.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(src)
        if "row_id" not in df.columns:
            df = df.copy()
            df["row_id"] = range(len(df))
        return df
    if suffix in {".jsonl", ".json"}:
        return load_jsonl(src)
    raise ValueError(f"Unsupported dataset format for {src}")

def _move_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device) if hasattr(value, "to") else value
    return out

def predict_probabilities_batch(
    wrapper: RobertaSarcasmModel,
    texts: Sequence[str],
    device: torch.device,
    *,
    batch_size: int = 64,
) -> list[tuple[int, list[float]]]:
    """Run batched inference and return predictions with probabilities."""

    if not texts:
        return []
    wrapper.model.eval()
    out_rows: list[tuple[int, list[float]]] = []
    use_amp = device.type == "cuda"
    for start in range(0, len(texts), batch_size):
        chunk = list(texts[start : start + batch_size])
        encoded = wrapper.tokenizer(
            chunk,
            truncation=True,
            max_length=wrapper.cfg.max_length,
            padding=True,
            return_tensors="pt",
        )
        batch = _move_batch(dict(encoded), device)
        with torch.inference_mode():
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    model_out = wrapper.forward_batch(batch)
            else:
                model_out = wrapper.forward_batch(batch)
            probs = torch.softmax(model_out.logits, dim=-1).detach().cpu().tolist()
        for row in probs:
            pred = int(max(range(len(row)), key=lambda idx: row[idx]))
            out_rows.append((pred, [float(x) for x in row]))
    return out_rows

def build_candidate_pools(df: pd.DataFrame) -> dict[int, dict[str, list[dict[str, object]]]]:
    """Group rows by label for later same-label / opposite-label swaps."""

    pools = {
        0: {"section": [], "description": []},
        1: {"section": [], "description": []},
    }
    for row in df.to_dict(orient="records"):
        label = int(row["is_sarcastic"])
        sec = preprocess_section_local(row.get("article_section"))
        desc = preprocess_description_local(row.get("description"))
        row_id = int(row.get("row_id", -1))
        if sec:
            pools[label]["section"].append({"row_id": row_id, "value": sec})
        if desc:
            pools[label]["description"].append({"row_id": row_id, "value": desc})
    return pools

def pick_swap_value(
    pools: dict[int, dict[str, list[dict[str, object]]]],
    *,
    label: int,
    field_name: str,
    current_row_id: int,
    rng: random.Random,
) -> str:
    """Pick a replacement value from the pool, avoiding the same row when possible."""

    choices = [row["value"] for row in pools[label][field_name] if row["row_id"] != current_row_id]
    if not choices:
        choices = [row["value"] for row in pools[label][field_name]]
    if not choices:
        return ""
    return str(rng.choice(choices))

def make_variants(
    *,
    recipe_name: str,
    headline: str,
    section: str,
    description: str,
    current_row_id: int,
    label: int | None,
    pools: dict[int, dict[str, list[dict[str, object]]]],
    rng: random.Random,
) -> list[tuple[str, str]]:
    """Generate field-level counterfactual inputs for one row."""

    recipe = recipe_by_name(recipe_name)
    variants: list[tuple[str, str]] = []
    variants.append(
        (
            "full",
            build_recipe_text(
                recipe_name,
                headline,
                section=section,
                description=description,
            ),
        )
    )
    variants.append(
        (
            "shuffle_headline_words",
            build_recipe_text(
                recipe_name,
                shuffle_words(headline, rng),
                section=section,
                description=description,
            ),
        )
    )
    variants.append(
        (
            "drop_headline_last_4",
            build_recipe_text(
                recipe_name,
                drop_last_words(headline, 4),
                section=section,
                description=description,
            ),
        )
    )
    variants.append(
        (
            "drop_headline_first_4",
            build_recipe_text(
                recipe_name,
                drop_first_words(headline, 4),
                section=section,
                description=description,
            ),
        )
    )
    if recipe.needs_section:
        variants.append(
            (
                "no_section",
                build_recipe_text(
                    recipe_name,
                    headline,
                    section="",
                    description=description,
                ),
            )
        )
        variants.append(
            (
                "section_only",
                build_recipe_text(
                    recipe_name,
                    EMPTY_TOKEN,
                    section=section,
                    description="",
                ),
            )
        )
    if recipe.needs_description:
        variants.append(
            (
                "no_description",
                build_recipe_text(
                    recipe_name,
                    headline,
                    section=section,
                    description="",
                ),
            )
        )
        variants.append(
            (
                "description_only",
                build_recipe_text(
                    recipe_name,
                    EMPTY_TOKEN,
                    section="",
                    description=description,
                ),
            )
        )
        variants.append(
            (
                "shuffle_description_words",
                build_recipe_text(
                    recipe_name,
                    headline,
                    section=section,
                    description=shuffle_words(description, rng),
                ),
            )
        )
        variants.append(
            (
                "truncate_description_16",
                build_recipe_text(
                    recipe_name,
                    headline,
                    section=section,
                    description=truncate_words(description, 16),
                ),
            )
        )
        variants.append(
            (
                "truncate_description_32",
                build_recipe_text(
                    recipe_name,
                    headline,
                    section=section,
                    description=truncate_words(description, 32),
                ),
            )
        )
    if recipe.needs_section and recipe.needs_description:
        variants.append(
            (
                "headline_only_template",
                build_recipe_text(recipe_name, headline, section="", description=""),
            )
        )
        variants.append(
            (
                "context_only",
                build_recipe_text(recipe_name, EMPTY_TOKEN, section=section, description=description),
            )
        )
    if label is not None and recipe.needs_section:
        same_section = pick_swap_value(
            pools,
            label=label,
            field_name="section",
            current_row_id=current_row_id,
            rng=rng,
        )
        opp_section = pick_swap_value(
            pools,
            label=1 - label,
            field_name="section",
            current_row_id=current_row_id,
            rng=rng,
        )
        if same_section:
            variants.append(
                (
                    "swap_section_same_label",
                    build_recipe_text(recipe_name, headline, section=same_section, description=description),
                )
            )
        if opp_section:
            variants.append(
                (
                    "swap_section_opposite_label",
                    build_recipe_text(recipe_name, headline, section=opp_section, description=description),
                )
            )
    if label is not None and recipe.needs_description:
        same_desc = pick_swap_value(
            pools,
            label=label,
            field_name="description",
            current_row_id=current_row_id,
            rng=rng,
        )
        opp_desc = pick_swap_value(
            pools,
            label=1 - label,
            field_name="description",
            current_row_id=current_row_id,
            rng=rng,
        )
        if same_desc:
            variants.append(
                (
                    "swap_description_same_label",
                    build_recipe_text(recipe_name, headline, section=section, description=same_desc),
                )
            )
        if opp_desc:
            variants.append(
                (
                    "swap_description_opposite_label",
                    build_recipe_text(recipe_name, headline, section=section, description=opp_desc),
                )
            )

    return variants

def analyze_row(
    wrapper: RobertaSarcasmModel,
    *,
    recipe_name: str,
    row: dict[str, object],
    pools: dict[int, dict[str, list[dict[str, object]]]],
    rng: random.Random,
    device: torch.device,
    batch_size: int,
) -> RowAnalysis:
    """Run all variants for one row."""

    row_id = int(row.get("row_id", -1))
    label = None if row.get("is_sarcastic") is None else int(row["is_sarcastic"])
    headline = _normalize_space(str(row.get("headline", "")))
    section = preprocess_section_local(row.get("article_section"))
    description = preprocess_description_local(row.get("description"))
    variants = make_variants(
        recipe_name=recipe_name,
        headline=headline,
        section=section,
        description=description,
        current_row_id=row_id,
        label=label,
        pools=pools,
        rng=rng,
    )
    texts = [text for _, text in variants]
    outputs = predict_probabilities_batch(wrapper, texts, device, batch_size=batch_size)
    base_name, base_text = variants[0]
    if base_name != "full":
        raise RuntimeError("The first variant must be the full input")
    base_pred, base_probs = outputs[0]
    base_conf = base_probs[base_pred]
    scored_variants: list[VariantResult] = []
    for (name, text), (pred, probs) in zip(variants[1:], outputs[1:]):
        scored_variants.append(
            VariantResult(
                name=name,
                text=text,
                predicted_label=pred,
                non_sarcastic_prob=probs[0],
                sarcastic_prob=probs[1],
                confidence_delta=float(base_conf - probs[base_pred]),
                sarcastic_delta=float(base_probs[1] - probs[1]),
                label_flip=bool(pred != base_pred),
            )
        )
    return RowAnalysis(
        row_id=row_id,
        label=label,
        headline=headline,
        section=section,
        description=description,
        base_text=base_text,
        base_predicted_label=base_pred,
        base_non_sarcastic_prob=base_probs[0],
        base_sarcastic_prob=base_probs[1],
        variants=scored_variants,
    )

def analyze_dataset(
    wrapper: RobertaSarcasmModel,
    dataset: pd.DataFrame,
    *,
    recipe_name: str,
    seed: int,
    device: torch.device,
    batch_size: int,
    limit: int | None = None,
    progress_every: int = 100,
) -> list[RowAnalysis]:
    """Run the full ablation suite over a dataset."""

    rng = random.Random(seed)
    frame = dataset if limit is None else dataset.iloc[:limit].copy()
    pools = build_candidate_pools(frame)
    analyses: list[RowAnalysis] = []
    for idx, row in enumerate(frame.to_dict(orient="records"), start=1):
        analyses.append(
            analyze_row(
                wrapper,
                recipe_name=recipe_name,
                row=row,
                pools=pools,
                rng=rng,
                device=device,
                batch_size=batch_size,
            )
        )
        if progress_every > 0 and idx % progress_every == 0:
            print(f"Processed {idx} rows...", flush=True)
    return analyses

def aggregate_variant_metrics(
    analyses: Sequence[RowAnalysis],
    *,
    label_filter: int | None = None,
) -> pd.DataFrame:
    """Summarize one row-level analysis table into one variant-level table."""

    rows: list[dict[str, object]] = []
    for analysis in analyses:
        if label_filter is not None and analysis.label != label_filter:
            continue
        for variant in analysis.variants:
            rows.append(
                {
                    "variant": variant.name,
                    "confidence_delta": variant.confidence_delta,
                    "sarcastic_delta": variant.sarcastic_delta,
                    "label_flip": int(variant.label_flip),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "variant",
                "count",
                "mean_confidence_delta",
                "median_confidence_delta",
                "mean_sarcastic_delta",
                "flip_rate",
            ]
        )
    df = pd.DataFrame(rows)
    grouped = (
        df.groupby("variant", as_index=False)
        .agg(
            count=("variant", "count"),
            mean_confidence_delta=("confidence_delta", "mean"),
            median_confidence_delta=("confidence_delta", "median"),
            mean_sarcastic_delta=("sarcastic_delta", "mean"),
            flip_rate=("label_flip", "mean"),
        )
        .sort_values(["mean_confidence_delta", "flip_rate"], ascending=[False, False])
        .reset_index(drop=True)
    )
    return grouped

def row_table(analyses: Sequence[RowAnalysis]) -> pd.DataFrame:
    """Flatten the row analyses into a CSV-friendly table."""

    rows: list[dict[str, object]] = []
    for analysis in analyses:
        for variant in analysis.variants:
            rows.append(
                {
                    "row_id": analysis.row_id,
                    "label": analysis.label,
                    "headline": analysis.headline,
                    "section": analysis.section,
                    "description": analysis.description,
                    "base_predicted_label": analysis.base_predicted_label,
                    "base_non_sarcastic_prob": analysis.base_non_sarcastic_prob,
                    "base_sarcastic_prob": analysis.base_sarcastic_prob,
                    "variant": variant.name,
                    "variant_predicted_label": variant.predicted_label,
                    "variant_non_sarcastic_prob": variant.non_sarcastic_prob,
                    "variant_sarcastic_prob": variant.sarcastic_prob,
                    "confidence_delta": variant.confidence_delta,
                    "sarcastic_delta": variant.sarcastic_delta,
                    "label_flip": int(variant.label_flip),
                }
            )
    return pd.DataFrame(rows)

def write_markdown_report(out_path: Path, analyses: Sequence[RowAnalysis]) -> None:
    """Write a short report with overall and label-wise summaries."""

    overall = aggregate_variant_metrics(analyses)
    label0 = aggregate_variant_metrics(analyses, label_filter=0)
    label1 = aggregate_variant_metrics(analyses, label_filter=1)
    lines = [
        "# Context Ablation Report",
        "",
        f"- Rows analyzed: `{len(analyses)}`",
        "- `mean_confidence_delta` = drop in support for the original predicted class.",
        "- `mean_sarcastic_delta` = drop in sarcastic probability specifically.",
        "- Higher `flip_rate` means the variant changes the model decision more often.",
        "",
    ]
    def add_table(title: str, table: pd.DataFrame) -> None:
        lines.extend(
            [
                f"## {title}",
                "",
                "| Variant | Count | Mean Confidence Δ | Median Confidence Δ | Mean Sarcastic Δ | Flip Rate |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in table.to_dict(orient="records"):
            lines.append(
                f"| {row['variant']} | {int(row['count'])} | {row['mean_confidence_delta']:.6f} | "
                f"{row['median_confidence_delta']:.6f} | {row['mean_sarcastic_delta']:.6f} | {row['flip_rate']:.6f} |"
            )
        lines.append("")
    add_table("Overall", overall)
    add_table("True label = 0", label0)
    add_table("True label = 1", label1)
    out_path.write_text("\n".join(lines), encoding="utf-8")

def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

def write_outputs(out_dir: Path, analyses: Sequence[RowAnalysis]) -> None:
    """Save the main tables and a detailed JSONL file."""

    out_dir.mkdir(parents=True, exist_ok=True)
    flat = row_table(analyses)
    flat.to_csv(out_dir / "context_ablation_rows.csv", index=False)
    aggregate_variant_metrics(analyses).to_csv(out_dir / "context_ablation_summary.csv", index=False)
    aggregate_variant_metrics(analyses, label_filter=0).to_csv(
        out_dir / "context_ablation_summary_label0.csv",
        index=False,
    )
    aggregate_variant_metrics(analyses, label_filter=1).to_csv(
        out_dir / "context_ablation_summary_label1.csv",
        index=False,
    )
    write_markdown_report(out_dir / "context_ablation_report.md", analyses)
    detailed_rows = []
    for analysis in analyses:
        detailed_rows.append(
            {
                "row_id": analysis.row_id,
                "label": analysis.label,
                "headline": analysis.headline,
                "section": analysis.section,
                "description": analysis.description,
                "base_text": analysis.base_text,
                "base_prediction": {
                    "label": analysis.base_predicted_label,
                    "non_sarcastic_prob": analysis.base_non_sarcastic_prob,
                    "sarcastic_prob": analysis.base_sarcastic_prob,
                },
                "variants": [
                    {
                        "name": variant.name,
                        "text": variant.text,
                        "predicted_label": variant.predicted_label,
                        "non_sarcastic_prob": variant.non_sarcastic_prob,
                        "sarcastic_prob": variant.sarcastic_prob,
                        "confidence_delta": variant.confidence_delta,
                        "sarcastic_delta": variant.sarcastic_delta,
                        "label_flip": variant.label_flip,
                    }
                    for variant in analysis.variants
                ],
            }
        )
    write_jsonl(out_dir / "context_ablation_details.jsonl", detailed_rows)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run field-level context ablations on a saved RoBERTa checkpoint.")
    parser.add_argument("checkpoint_or_study_dir", help="Checkpoint dir or study dir with best_model/materialization.json")
    parser.add_argument("--recipe", required=True, choices=["headline_only", "headline_section", "headline_section_description"])
    parser.add_argument("--data-path", required=True, help="CSV or JSONL dataset to analyze")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of rows")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    checkpoint_dir = resolve_checkpoint_dir(args.checkpoint_or_study_dir)
    out_dir = Path(args.out_dir).resolve() if args.out_dir else checkpoint_dir.parent
    device = pick_device(args.device)
    wrapper = RobertaSarcasmModel.from_checkpoint(checkpoint_dir, device=device)
    dataset = load_input_rows(args.data_path)
    analyses = analyze_dataset(
        wrapper,
        dataset,
        recipe_name=args.recipe,
        seed=args.seed,
        device=device,
        batch_size=args.batch_size,
        limit=args.limit,
        progress_every=args.progress_every,
    )
    write_outputs(out_dir, analyses)

if __name__ == "__main__":
    main()