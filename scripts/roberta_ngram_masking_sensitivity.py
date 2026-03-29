from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.roberta import RobertaSarcasmModel
from training.context_common import VARIANT2_RECIPES, load_jsonl, pick_device, recipe_by_name
from utils.constants import SECTION_MAPPINGS


@dataclass
class SpanScore:
    ngram_size: int
    start: int
    end: int
    span: str
    masked_text: str
    sarcastic_prob: float
    sarcastic_delta: float
    predicted_label: int


@dataclass
class ExampleAnalysis:
    row_id: int
    label: int | None
    headline: str
    section: str
    description: str
    input_text: str
    predicted_label: int
    non_sarcastic_prob: float
    sarcastic_prob: float
    top_word: str | None
    top_word_delta: float | None
    top_word_masked_sarcastic_prob: float | None
    top_scores: list[SpanScore]


def resolve_checkpoint_dir(path: str | Path) -> Path:
    candidate = Path(path).resolve()
    if (candidate / "model_state.pt").exists() and (candidate / "model_config.json").exists():
        return candidate
    best_model_meta = candidate / "best_model" / "materialization.json"
    if best_model_meta.exists():
        meta = json.loads(best_model_meta.read_text(encoding="utf-8"))
        return Path(meta["best_model_path"]).resolve()
    raise ValueError(f"Could not resolve a RoBERTa checkpoint from: {candidate}")


def _normalize_whitespace(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def preprocess_article_section_local(entry: str | None, mappings: dict[str, str] = SECTION_MAPPINGS) -> str:
    if not entry:
        return ""
    if isinstance(entry, list):
        items = entry
    elif isinstance(entry, str) and entry.startswith("["):
        try:
            items = ast.literal_eval(entry)
        except Exception:
            items = [entry]
    else:
        items = [entry]
    standardized = [mappings.get(str(item).lower(), "other") for item in items]
    standardized = [item for item in dict.fromkeys(standardized) if item]
    return ", ".join(standardized)


def preprocess_description_local(text: str | None) -> str:
    text = _normalize_whitespace(text)
    if not text:
        return ""
    return re.sub(r"^[A-Z\s,.]+—", "", text).strip()


def build_recipe_input(
    recipe_name: str,
    headline: str,
    section: str | None = None,
    description: str | None = None,
) -> str:
    recipe = recipe_by_name(recipe_name)
    headline_text = _normalize_whitespace(headline)
    section_text = preprocess_article_section_local(section)
    description_text = preprocess_description_local(description)
    if recipe.needs_section and not section_text:
        raise ValueError(f"The provided section is empty after preprocessing for recipe `{recipe_name}`.")
    if recipe.needs_description and not description_text:
        raise ValueError(f"The provided description is empty after preprocessing for recipe `{recipe_name}`.")
    parts = [f"headline: {headline_text}"]
    if recipe.needs_section:
        parts.append(f"section: {section_text}")
    if recipe.needs_description:
        parts.append(f"description: {description_text}")
    return " ".join(parts)


def load_input_rows(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(source)
        if "row_id" not in df.columns:
            df = df.copy()
            df["row_id"] = range(len(df))
        return df
    if suffix in {".jsonl", ".json"}:
        return load_jsonl(source)
    raise ValueError(f"Unsupported dataset format for {source}. Use .csv or .jsonl")


def whitespace_spans(tokens: Sequence[str], ngram_sizes: Iterable[int]) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for n in ngram_sizes:
        if n <= 0:
            continue
        for start in range(0, len(tokens) - n + 1):
            end = start + n
            spans.append((start, end, " ".join(tokens[start:end])))
    return spans


def mask_span(tokens: Sequence[str], start: int, end: int, mask_token: str) -> str:
    return " ".join(list(tokens[:start]) + [mask_token] + list(tokens[end:])).strip()


def _move_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


def predict_probabilities_batch(
    wrapper: RobertaSarcasmModel,
    texts: Sequence[str],
    device: torch.device,
    *,
    batch_size: int = 64,
) -> list[tuple[int, list[float]]]:
    if not texts:
        return []
    wrapper.model.eval()
    results: list[tuple[int, list[float]]] = []
    autocast_enabled = device.type == "cuda"
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
            if autocast_enabled:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = wrapper.forward_batch(batch)
            else:
                output = wrapper.forward_batch(batch)
            probs = torch.softmax(output.logits, dim=-1).detach().cpu().tolist()
        for row in probs:
            pred = int(max(range(len(row)), key=lambda idx: row[idx]))
            results.append((pred, [float(x) for x in row]))
    return results


def predict_probabilities(
    wrapper: RobertaSarcasmModel,
    text: str,
    device: torch.device,
    *,
    batch_size: int = 64,
) -> tuple[int, list[float]]:
    return predict_probabilities_batch(wrapper, [text], device, batch_size=batch_size)[0]


def score_masked_spans(
    wrapper: RobertaSarcasmModel,
    text: str,
    *,
    ngram_sizes: Sequence[int],
    device: torch.device,
    batch_size: int = 64,
) -> tuple[int, list[float], list[SpanScore]]:
    base_pred, base_probs = predict_probabilities(wrapper, text, device, batch_size=batch_size)
    tokens = text.split()
    mask_token = wrapper.tokenizer.mask_token or "<mask>"
    masked_specs: list[tuple[int, int, str, str]] = []
    masked_texts: list[str] = []
    for start, end, span in whitespace_spans(tokens, ngram_sizes):
        masked_text = mask_span(tokens, start, end, mask_token)
        masked_specs.append((start, end, span, masked_text))
        masked_texts.append(masked_text)
    masked_outputs = predict_probabilities_batch(wrapper, masked_texts, device, batch_size=batch_size)
    scores: list[SpanScore] = []
    for (start, end, span, masked_text), (masked_pred, masked_probs) in zip(masked_specs, masked_outputs):
        scores.append(
            SpanScore(
                ngram_size=end - start,
                start=start,
                end=end,
                span=span,
                masked_text=masked_text,
                sarcastic_prob=masked_probs[1],
                sarcastic_delta=base_probs[1] - masked_probs[1],
                predicted_label=masked_pred,
            )
        )
    scores.sort(key=lambda item: item.sarcastic_delta, reverse=True)
    return base_pred, base_probs, scores


def analyze_example(
    wrapper: RobertaSarcasmModel,
    *,
    recipe_name: str,
    row_id: int,
    headline: str,
    section: str | None,
    description: str | None,
    label: int | None,
    ngram_sizes: Sequence[int],
    device: torch.device,
    top_k: int,
    batch_size: int,
) -> ExampleAnalysis:
    input_text = build_recipe_input(recipe_name, headline, section=section, description=description)
    base_pred, base_probs, scores = score_masked_spans(
        wrapper,
        input_text,
        ngram_sizes=ngram_sizes,
        device=device,
        batch_size=batch_size,
    )
    top_scores = list(scores[:top_k])
    word_scores = [score for score in scores if score.ngram_size == 1]
    top_word_score = word_scores[0] if word_scores else None
    return ExampleAnalysis(
        row_id=row_id,
        label=label,
        headline=_normalize_whitespace(headline),
        section=preprocess_article_section_local(section),
        description=preprocess_description_local(description),
        input_text=input_text,
        predicted_label=base_pred,
        non_sarcastic_prob=base_probs[0],
        sarcastic_prob=base_probs[1],
        top_word=None if top_word_score is None else top_word_score.span,
        top_word_delta=None if top_word_score is None else top_word_score.sarcastic_delta,
        top_word_masked_sarcastic_prob=None if top_word_score is None else top_word_score.sarcastic_prob,
        top_scores=top_scores,
    )


def analyze_dataset(
    wrapper: RobertaSarcasmModel,
    dataset: pd.DataFrame,
    *,
    recipe_name: str,
    ngram_sizes: Sequence[int],
    device: torch.device,
    top_k: int,
    limit: int | None = None,
    batch_size: int = 64,
    progress_every: int = 100,
) -> list[ExampleAnalysis]:
    analyses: list[ExampleAnalysis] = []
    frame = dataset if limit is None else dataset.iloc[:limit].copy()
    for idx, row in enumerate(frame.to_dict(orient="records"), start=1):
        try:
            analyses.append(
                analyze_example(
                    wrapper,
                    recipe_name=recipe_name,
                    row_id=int(row.get("row_id", len(analyses))),
                    headline=str(row.get("headline", "")),
                    section=row.get("article_section"),
                    description=row.get("description"),
                    label=None if row.get("is_sarcastic") is None else int(row.get("is_sarcastic")),
                    ngram_sizes=ngram_sizes,
                    device=device,
                    top_k=top_k,
                    batch_size=batch_size,
                )
            )
        except ValueError:
            continue
        if progress_every > 0 and idx % progress_every == 0:
            print(f"Processed {idx} rows...", flush=True)
    return analyses


def aggregate_chosen_words(analyses: Sequence[ExampleAnalysis]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for analysis in analyses:
        if analysis.top_word is None or analysis.top_word_delta is None:
            continue
        rows.append(
            {
                "word": analysis.top_word,
                "sarcastic_delta": analysis.top_word_delta,
                "row_id": analysis.row_id,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["word", "count", "mean_sarcastic_delta", "median_sarcastic_delta", "std_sarcastic_delta", "min_sarcastic_delta", "max_sarcastic_delta"])
    df = pd.DataFrame(rows)
    grouped = (
        df.groupby(["word"], as_index=False)
        .agg(
            count=("row_id", "count"),
            mean_sarcastic_delta=("sarcastic_delta", "mean"),
            median_sarcastic_delta=("sarcastic_delta", "median"),
            std_sarcastic_delta=("sarcastic_delta", "std"),
            min_sarcastic_delta=("sarcastic_delta", "min"),
            max_sarcastic_delta=("sarcastic_delta", "max"),
        )
        .fillna({"std_sarcastic_delta": 0.0})
        .sort_values(["count", "mean_sarcastic_delta"], ascending=[False, False])
        .reset_index(drop=True)
    )
    return grouped


def write_dataset_report(out_dir: Path, analyses: Sequence[ExampleAnalysis], *, report_top_k: int) -> None:
    aggregate = aggregate_chosen_words(analyses)
    lines = [
        "# RoBERTa Word Masking Dataset Summary",
        "",
        f"- Rows analyzed: `{len(analyses)}`",
        f"- Rows with a chosen top word: `{sum(1 for item in analyses if item.top_word is not None)}`",
        "",
        "## Per-row summary",
        "",
        f"- `ngram_masking_dataset_summary.csv` contains the top single word for each row and its delta.",
        "",
        "## Top chosen words",
        "",
        "| Rank | Word | Count | Mean Delta | Median Delta | Std Delta | Min Delta | Max Delta |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for idx, row in enumerate(aggregate.head(report_top_k).to_dict(orient="records"), start=1):
        lines.append(
            f"| {idx} | {row['word']} | {int(row['count'])} | {row['mean_sarcastic_delta']:.6f} | "
            f"{row['median_sarcastic_delta']:.6f} | {row['std_sarcastic_delta']:.6f} | "
            f"{row['min_sarcastic_delta']:.6f} | {row['max_sarcastic_delta']:.6f} |"
        )
    (out_dir / "ngram_masking_dataset_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_report(
    out_path: Path,
    *,
    checkpoint_dir: Path,
    recipe_name: str,
    input_text: str,
    base_pred: int,
    base_probs: list[float],
    scores: Sequence[SpanScore],
    top_k: int,
) -> None:
    top_scores = list(scores[:top_k])
    lines = [
        "# RoBERTa N-gram Masking Sensitivity",
        "",
        f"- Checkpoint: `{checkpoint_dir.as_posix()}`",
        f"- Recipe: `{recipe_name}`",
        f"- Predicted label: `{base_pred}`",
        f"- Base non-sarcastic probability: `{base_probs[0]:.6f}`",
        f"- Base sarcastic probability: `{base_probs[1]:.6f}`",
        "",
        "## Input",
        "",
        "```text",
        input_text,
        "```",
        "",
        "## Most Sarcasm-Supporting Spans",
        "",
        "| Rank | n-gram | Span | Sarcastic Prob After Mask | Delta |",
        "| --- | ---: | --- | ---: | ---: |",
    ]
    for idx, score in enumerate(top_scores, start=1):
        lines.append(
            f"| {idx} | {score.ngram_size} | {score.span} | {score.sarcastic_prob:.6f} | {score.sarcastic_delta:.6f} |"
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_dataset_outputs(out_dir: Path, analyses: Sequence[ExampleAnalysis], *, report_top_k: int) -> None:
    summary_rows = [
        {
            "row_id": analysis.row_id,
            "label": analysis.label,
            "headline": analysis.headline,
            "section": analysis.section,
            "description": analysis.description,
            "predicted_label": analysis.predicted_label,
            "non_sarcastic_prob": analysis.non_sarcastic_prob,
            "sarcastic_prob": analysis.sarcastic_prob,
            "top_word": analysis.top_word,
            "top_word_delta": analysis.top_word_delta,
            "top_word_masked_sarcastic_prob": analysis.top_word_masked_sarcastic_prob,
        }
        for analysis in analyses
    ]
    pd.DataFrame(summary_rows).to_csv(out_dir / "ngram_masking_dataset_summary.csv", index=False)
    write_jsonl(
        out_dir / "ngram_masking_dataset_top_spans.jsonl",
        [
            {
                "row_id": analysis.row_id,
                "label": analysis.label,
                "input_text": analysis.input_text,
                "predicted_label": analysis.predicted_label,
                "probabilities": {
                    "non_sarcastic": analysis.non_sarcastic_prob,
                    "sarcastic": analysis.sarcastic_prob,
                },
                "top_scores": [
                    {
                        "ngram_size": score.ngram_size,
                        "start": score.start,
                        "end": score.end,
                        "span": score.span,
                        "masked_text": score.masked_text,
                        "sarcastic_prob": score.sarcastic_prob,
                        "sarcastic_delta": score.sarcastic_delta,
                        "predicted_label": score.predicted_label,
                    }
                    for score in analysis.top_scores
                ],
            }
            for analysis in analyses
        ],
    )
    aggregate_chosen_words(analyses).to_csv(out_dir / "ngram_masking_aggregate_words.csv", index=False)
    write_dataset_report(out_dir, analyses, report_top_k=report_top_k)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run n-gram masking sensitivity on a saved RoBERTa checkpoint.")
    parser.add_argument("checkpoint_or_study_dir", help="Checkpoint dir or study dir containing best_model/materialization.json")
    parser.add_argument("--recipe", required=True, choices=[recipe.name for recipe in VARIANT2_RECIPES])
    parser.add_argument("--headline")
    parser.add_argument("--section", default=None)
    parser.add_argument("--description", default=None)
    parser.add_argument("--data-path", default=None, help="Optional CSV or JSONL dataset path for batch analysis.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of rows to analyze in batch mode.")
    parser.add_argument("--ngram-sizes", default="1,2,3", help="Comma-separated whitespace n-gram sizes to test")
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--report-top-k", type=int, default=25, help="How many words to include in the aggregate dataset report.")
    parser.add_argument("--inference-batch-size", type=int, default=64, help="How many masked texts to score per GPU batch.")
    parser.add_argument("--progress-every", type=int, default=100, help="Print batch-mode progress every N rows. Use 0 to disable.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", default=None, help="Directory for markdown/json outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data_path and not args.headline:
        raise ValueError("Provide --headline for single-example mode, or --data-path for dataset mode.")
    checkpoint_dir = resolve_checkpoint_dir(args.checkpoint_or_study_dir)
    out_dir = Path(args.out_dir).resolve() if args.out_dir else checkpoint_dir.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    wrapper = RobertaSarcasmModel.from_checkpoint(checkpoint_dir, device=device)
    ngram_sizes = [int(part.strip()) for part in args.ngram_sizes.split(",") if part.strip()]
    if args.data_path:
        dataset = load_input_rows(args.data_path)
        analyses = analyze_dataset(
            wrapper,
            dataset,
            recipe_name=args.recipe,
            ngram_sizes=ngram_sizes,
            device=device,
            top_k=args.top_k,
            limit=args.limit,
            batch_size=args.inference_batch_size,
            progress_every=args.progress_every,
        )
        write_dataset_outputs(out_dir, analyses, report_top_k=args.report_top_k)
        return

    recipe_text = build_recipe_input(args.recipe, args.headline, section=args.section, description=args.description)
    base_pred, base_probs, scores = score_masked_spans(
        wrapper,
        recipe_text,
        ngram_sizes=ngram_sizes,
        device=device,
        batch_size=args.inference_batch_size,
    )
    write_report(
        out_dir / "ngram_masking_report.md",
        checkpoint_dir=checkpoint_dir,
        recipe_name=args.recipe,
        input_text=recipe_text,
        base_pred=base_pred,
        base_probs=base_probs,
        scores=scores,
        top_k=args.top_k,
    )
    write_json(
        out_dir / "ngram_masking_scores.json",
        {
            "checkpoint_dir": str(checkpoint_dir),
            "recipe": args.recipe,
            "input_text": recipe_text,
            "base_prediction": {"label": base_pred, "probabilities": base_probs},
            "scores": [
                {
                    "ngram_size": score.ngram_size,
                    "start": score.start,
                    "end": score.end,
                    "span": score.span,
                    "masked_text": score.masked_text,
                    "sarcastic_prob": score.sarcastic_prob,
                    "sarcastic_delta": score.sarcastic_delta,
                    "predicted_label": score.predicted_label,
                }
                for score in scores
            ],
        },
    )


if __name__ == "__main__":
    main()
