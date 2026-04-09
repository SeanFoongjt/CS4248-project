from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class StudyRow:
    study_name: str
    model: str
    recipe: str
    best_trial_id: int | None
    objective_metric: str
    val_accuracy: float | None
    val_macro_f1: float | None
    test_accuracy: float | None
    test_macro_f1: float | None
    train_loss: float | None
    val_loss: float | None
    test_loss: float | None
    best_epoch: int | None
    metrics_path: str | None
    artifact_path: str | None


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def discover_study_dirs(root: Path) -> list[Path]:
    if (root / "study_summary.json").exists():
        return [root]
    return sorted(path for path in root.iterdir() if path.is_dir() and (path / "study_summary.json").exists())


def resolve_metrics_path(study_dir: Path) -> tuple[Path | None, Path | None]:
    summary = load_json(study_dir / "study_summary.json")
    best_model_meta = study_dir / "best_model" / "materialization.json"
    if best_model_meta.exists():
        meta = load_json(best_model_meta)
        metrics_path = Path(meta["metrics_path"])
        if not metrics_path.exists():
            metrics_path = study_dir / "best_model" / summary["model"] / summary["recipe"] / "metrics.json"
        artifact_path = Path(meta["best_model_path"])
        if not artifact_path.exists():
            artifact_path = study_dir / "best_model" / summary["model"] / summary["recipe"] / "checkpoint"
        return metrics_path, artifact_path

    best_trial_id = summary.get("best_trial_id")
    if best_trial_id is None:
        return None, None
    trial_root = study_dir / "trials" / f"trial_{int(best_trial_id):05d}" / summary["model"] / summary["recipe"]
    metrics_path = trial_root / "metrics.json"
    artifact_path = trial_root / "checkpoint"
    return metrics_path if metrics_path.exists() else None, artifact_path if artifact_path.exists() else None


def build_study_row(study_dir: Path) -> StudyRow:
    summary = load_json(study_dir / "study_summary.json")
    metrics_path, artifact_path = resolve_metrics_path(study_dir)
    metrics = load_json(metrics_path) if metrics_path and metrics_path.exists() else {}
    val_metrics = metrics.get("val") or {}
    test_metrics = metrics.get("test") or {}
    return StudyRow(
        study_name=summary["study_name"],
        model=summary["model"],
        recipe=summary["recipe"],
        best_trial_id=summary.get("best_trial_id"),
        objective_metric=summary["objective_metric"],
        val_accuracy=val_metrics.get("accuracy"),
        val_macro_f1=summary.get("best_validation_metric", val_metrics.get("macro_f1")),
        test_accuracy=test_metrics.get("accuracy"),
        test_macro_f1=summary.get("corresponding_test_metric", test_metrics.get("macro_f1")),
        train_loss=metrics.get("train_loss"),
        val_loss=metrics.get("val_loss"),
        test_loss=metrics.get("test_loss"),
        best_epoch=metrics.get("best_epoch"),
        metrics_path=str(metrics_path) if metrics_path else None,
        artifact_path=str(artifact_path) if artifact_path else summary.get("artifact_path"),
    )


def load_loss_history(study_dir: Path) -> tuple[str, list[dict[str, float]]]:
    summary = load_json(study_dir / "study_summary.json")
    metrics_path, _ = resolve_metrics_path(study_dir)
    if not metrics_path or not metrics_path.exists():
        return f"{summary['model']} / {summary['recipe']}", []
    metrics = load_json(metrics_path)
    history = metrics.get("history") or []
    cleaned: list[dict[str, float]] = []
    for row in history:
        if "epoch" not in row or "train_loss" not in row or "val_loss" not in row:
            continue
        cleaned.append(
            {
                "epoch": float(row["epoch"]),
                "train_loss": float(row["train_loss"]),
                "val_loss": float(row["val_loss"]),
            }
        )
    return f"{summary['model']} / {summary['recipe']}", cleaned


def write_csv(rows: list[StudyRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(StudyRow.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def fmt(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def load_supplemental_results(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return load_json(path)


def prettify_recipe_name(recipe: str) -> str:
    return recipe.replace("_", " ")


def build_variant_summary(
    rows: list[StudyRow],
    *,
    models: tuple[str, ...] = ("distilbert", "roberta"),
) -> list[dict[str, str | float | None]]:
    grouped: dict[str, dict[str, StudyRow]] = defaultdict(dict)
    for row in rows:
        if row.model not in models:
            continue
        existing = grouped[row.recipe].get(row.model)
        if existing is None or ((row.val_macro_f1 or float("-inf")) > (existing.val_macro_f1 or float("-inf"))):
            grouped[row.recipe][row.model] = row

    variant_rows: list[dict[str, str | float | None]] = []
    for recipe in sorted(grouped):
        summary_row: dict[str, str | float | None] = {
            "variant": recipe,
            "variant_label": prettify_recipe_name(recipe),
        }
        for model in models:
            picked = grouped[recipe].get(model)
            summary_row[model] = picked.test_macro_f1 if picked else None
        variant_rows.append(summary_row)
    return variant_rows


def write_variant_summary_csv(summary_rows: list[dict[str, str | float | None]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["variant", "variant_label", "distilbert", "roberta"]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


def write_supplemental_results_csv(payload: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["variant", "variant_label", "model", "model_label", "loss", "accuracy", "precision", "recall", "f1"]
    models = {item["id"]: item["label"] for item in payload.get("models", [])}
    variants = payload.get("variants", [])
    results = payload.get("results", {})
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for variant in variants:
            variant_id = variant["id"]
            variant_results = results.get(variant_id, {})
            for model_id, metrics in variant_results.items():
                writer.writerow(
                    {
                        "variant": variant_id,
                        "variant_label": variant["label"],
                        "model": model_id,
                        "model_label": models.get(model_id, model_id),
                        "loss": metrics.get("loss"),
                        "accuracy": metrics.get("accuracy"),
                        "precision": metrics.get("precision"),
                        "recall": metrics.get("recall"),
                        "f1": metrics.get("f1"),
                    }
                )


def render_supplemental_results_markdown(payload: dict[str, Any]) -> list[str]:
    if not payload:
        return []
    metric_specs = [
        ("loss", "Loss"),
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1-Score"),
    ]
    model_defs = payload.get("models", [])
    variants = payload.get("variants", [])
    results = payload.get("results", {})
    if not model_defs or not variants or not results:
        return []

    lines = [
        f"## {payload.get('title', 'Supplemental Results')}",
        "",
        payload.get("description", "Additional test-set results."),
        "",
    ]
    header = "| Data Variant | " + " | ".join(model["label"] for model in model_defs) + " |"
    divider = "| --- | " + " | ".join("---:" for _ in model_defs) + " |"
    for metric_key, metric_label in metric_specs:
        lines.extend(
            [
                f"### {metric_label}",
                "",
                header,
                divider,
            ]
        )
        for variant in variants:
            row = [variant["label"]]
            variant_results = results.get(variant["id"], {})
            for model in model_defs:
                value = (variant_results.get(model["id"]) or {}).get(metric_key)
                row.append(fmt(value))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    return lines


def render_markdown(
    rows: list[StudyRow],
    out_path: Path,
    loss_plot_pdf_name: str,
    supplemental_results: dict[str, Any] | None = None,
) -> None:
    variant_summary = build_variant_summary(rows)
    lines = [
        "# Model Comparison Report",
        "",
        "## Best Test Macro-F1 by Data Variant",
        "",
        "| Data Variant | DistilBERT | RoBERTa |",
        "| --- | ---: | ---: |",
    ]
    for row in variant_summary:
        lines.append(
            f"| {row['variant_label']} | {fmt(row['distilbert'])} | {fmt(row['roberta'])} |"
        )
    lines.append("")
    lines.extend(render_supplemental_results_markdown(supplemental_results or {}))
    lines.extend(
        [
        "## Best Trial Summary",
        "",
        "| Study | Model | Recipe | Best Trial | Val Acc | Val Macro-F1 | Test Acc | Test Macro-F1 | Train Loss | Val Loss | Test Loss | Best Epoch |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(rows, key=lambda item: (item.model, item.recipe, item.study_name)):
        lines.append(
            f"| {row.study_name} | {row.model} | {row.recipe} | {fmt(row.best_trial_id, 0)} | "
            f"{fmt(row.val_accuracy)} | {fmt(row.val_macro_f1)} | {fmt(row.test_accuracy)} | "
            f"{fmt(row.test_macro_f1)} | {fmt(row.train_loss)} | {fmt(row.val_loss)} | {fmt(row.test_loss)} | {fmt(row.best_epoch, 0)} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `Train/Val/Test Loss` and `Best Epoch` are available for transformer studies that save epoch history.",
            "- Classical studies will show `-` for loss-over-time fields because they do not train in epochs.",
            "",
            "## Loss Over Time",
            "",
            f"[Download loss plot as PDF]({loss_plot_pdf_name})",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,sans-serif;fill:#1f2937}.title{font-size:18px;font-weight:bold}.axis{font-size:11px}.label{font-size:12px}.small{font-size:10px}.grid{stroke:#e5e7eb;stroke-width:1}.axis-line{stroke:#6b7280;stroke-width:1.2}</style>',
    ]


def palette(index: int) -> str:
    colors = ["#2563eb", "#059669", "#dc2626", "#7c3aed", "#ea580c", "#0891b2"]
    return colors[index % len(colors)]


def render_loss_plot(series: list[tuple[str, list[dict[str, float]]]], out_path: Path) -> None:
    usable = [(label, rows) for label, rows in series if rows]
    width, height = 960, 480
    left, right, top, bottom = 80, 220, 50, 60
    inner_w = width - left - right
    inner_h = height - top - bottom
    lines = svg_header(width, height)
    write_text(lines, left, 28, "Loss Over Time for Best Models", "title")

    if not usable:
        write_text(lines, width / 2, height / 2, "No epoch loss history available in the selected studies.", "label", "middle")
        lines.append("</svg>")
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return

    all_epochs = [row["epoch"] for _, rows in usable for row in rows]
    all_losses = [row["train_loss"] for _, rows in usable for row in rows] + [row["val_loss"] for _, rows in usable for row in rows]
    x_min, x_max = min(all_epochs), max(all_epochs)
    y_min, y_max = min(all_losses), max(all_losses)
    y_pad = (y_max - y_min) * 0.1 if y_max > y_min else 0.1
    y_lo, y_hi = max(0.0, y_min - y_pad), y_max + y_pad

    def x_map(x: float) -> float:
        if x_max == x_min:
            return left + inner_w / 2
        return left + (x - x_min) / (x_max - x_min) * inner_w

    def y_map(y: float) -> float:
        return top + (1 - (y - y_lo) / (y_hi - y_lo)) * inner_h

    for i in range(5):
        frac = i / 4
        y = top + frac * inner_h
        val = y_hi - frac * (y_hi - y_lo)
        lines.append(f'<line x1="{left}" y1="{y}" x2="{width - right}" y2="{y}" class="grid"/>')
        write_text(lines, left - 8, y + 4, f"{val:.3f}", "axis", "end")

    for epoch in range(int(x_min), int(x_max) + 1):
        x = x_map(float(epoch))
        lines.append(f'<line x1="{x}" y1="{top}" x2="{x}" y2="{height - bottom}" class="grid"/>')
        write_text(lines, x, height - bottom + 18, str(epoch), "axis", "middle")

    lines.append(f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" class="axis-line"/>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" class="axis-line"/>')
    write_text(lines, width / 2, height - 14, "Epoch", "label", "middle")
    write_text(lines, 24, top - 10, "Loss", "label")

    legend_y = 72
    for idx, (label, rows) in enumerate(usable):
        color = palette(idx)
        train_points = " ".join(f"{x_map(row['epoch']):.1f},{y_map(row['train_loss']):.1f}" for row in rows)
        val_points = " ".join(f"{x_map(row['epoch']):.1f},{y_map(row['val_loss']):.1f}" for row in rows)
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{train_points}" stroke-dasharray="6 4"/>')
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{val_points}"/>')
        for row in rows:
            lines.append(f'<circle cx="{x_map(row["epoch"]):.1f}" cy="{y_map(row["val_loss"]):.1f}" r="3" fill="{color}"/>')
        text_y = legend_y + idx * 32
        lines.append(f'<line x1="{width - right + 10}" y1="{text_y}" x2="{width - right + 40}" y2="{text_y}" stroke="{color}" stroke-width="2.5"/>')
        lines.append(
            f'<line x1="{width - right + 10}" y1="{text_y + 12}" x2="{width - right + 40}" y2="{text_y + 12}" stroke="{color}" stroke-width="2" stroke-dasharray="6 4"/>'
        )
        write_text(lines, width - right + 48, text_y + 4, f"{label} val", "small")
        write_text(lines, width - right + 48, text_y + 16, f"{label} train", "small")

    lines.append("</svg>")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _hex_to_rgb(color: str) -> tuple[float, float, float]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def _pdf_text(commands: list[str], x: float, y: float, text: str, size: int = 11) -> None:
    commands.append(f"BT /F1 {size} Tf 1 0 0 1 {x:.2f} {y:.2f} Tm ({_pdf_escape(text)}) Tj ET")


def _pdf_line(
    commands: list[str],
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    color: str = "#6b7280",
    width: float = 1.0,
    dash: str | None = None,
) -> None:
    r, g, b = _hex_to_rgb(color)
    commands.append(f"{r:.3f} {g:.3f} {b:.3f} RG")
    commands.append(f"{width:.2f} w")
    commands.append(f"{dash if dash else '[] 0 d'}")
    commands.append(f"{x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S")


def _pdf_polyline(
    commands: list[str],
    points: list[tuple[float, float]],
    *,
    color: str,
    width: float = 2.0,
    dash: str | None = None,
) -> None:
    if len(points) < 2:
        return
    r, g, b = _hex_to_rgb(color)
    commands.append(f"{r:.3f} {g:.3f} {b:.3f} RG")
    commands.append(f"{width:.2f} w")
    commands.append(f"{dash if dash else '[] 0 d'}")
    start_x, start_y = points[0]
    segments = [f"{start_x:.2f} {start_y:.2f} m"]
    for x, y in points[1:]:
        segments.append(f"{x:.2f} {y:.2f} l")
    segments.append("S")
    commands.append(" ".join(segments))


def _write_single_page_pdf(width: int, height: int, commands: list[str], out_path: Path) -> None:
    content = "\n".join(commands).encode("latin-1", "replace")
    objects = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        (
            f"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] "
            "/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>\nendobj\n"
        ).encode("latin-1"),
        b"4 0 obj\n<< /Length " + str(len(content)).encode("latin-1") + b" >>\nstream\n" + content + b"\nendstream\nendobj\n",
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
    ]
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objects:
        offsets.append(len(pdf))
        pdf.extend(obj)
    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(offsets)}\n".encode("latin-1"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))
    pdf.extend(
        (
            f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n"
        ).encode("latin-1")
    )
    out_path.write_bytes(pdf)


def render_loss_plot_pdf(series: list[tuple[str, list[dict[str, float]]]], out_path: Path) -> None:
    usable = [(label, rows) for label, rows in series if rows]
    width, height = 960, 480
    left, right, top, bottom = 80, 220, 50, 60
    inner_w = width - left - right
    inner_h = height - top - bottom
    commands: list[str] = []
    _pdf_text(commands, left, height - 28, "Loss Over Time for Best Models", size=18)

    if not usable:
        _pdf_text(commands, width / 2 - 160, height / 2, "No epoch loss history available in the selected studies.", size=12)
        _write_single_page_pdf(width, height, commands, out_path)
        return

    all_epochs = [row["epoch"] for _, rows in usable for row in rows]
    all_losses = [row["train_loss"] for _, rows in usable for row in rows] + [row["val_loss"] for _, rows in usable for row in rows]
    x_min, x_max = min(all_epochs), max(all_epochs)
    y_min, y_max = min(all_losses), max(all_losses)
    y_pad = (y_max - y_min) * 0.1 if y_max > y_min else 0.1
    y_lo, y_hi = max(0.0, y_min - y_pad), y_max + y_pad

    def x_map(x: float) -> float:
        if x_max == x_min:
            return left + inner_w / 2
        return left + (x - x_min) / (x_max - x_min) * inner_w

    def y_map(y: float) -> float:
        return bottom + (y - y_lo) / (y_hi - y_lo) * inner_h

    for i in range(5):
        frac = i / 4
        y = bottom + frac * inner_h
        val = y_lo + frac * (y_hi - y_lo)
        _pdf_line(commands, left, y, width - right, y, color="#e5e7eb", width=1.0)
        _pdf_text(commands, left - 52, y - 4, f"{val:.3f}", size=10)

    for epoch in range(int(x_min), int(x_max) + 1):
        x = x_map(float(epoch))
        _pdf_line(commands, x, bottom, x, height - top, color="#e5e7eb", width=1.0)
        _pdf_text(commands, x - 4, bottom - 20, str(epoch), size=10)

    _pdf_line(commands, left, bottom, width - right, bottom, color="#6b7280", width=1.2)
    _pdf_line(commands, left, bottom, left, height - top, color="#6b7280", width=1.2)
    _pdf_text(commands, width / 2, 18, "Epoch", size=12)
    _pdf_text(commands, 24, height - top + 6, "Loss", size=12)

    legend_y = height - 72
    for idx, (label, rows) in enumerate(usable):
        color = palette(idx)
        train_points = [(x_map(row["epoch"]), y_map(row["train_loss"])) for row in rows]
        val_points = [(x_map(row["epoch"]), y_map(row["val_loss"])) for row in rows]
        _pdf_polyline(commands, train_points, color=color, width=2.0, dash="[6 4] 0 d")
        _pdf_polyline(commands, val_points, color=color, width=2.5)
        legend_y_item = legend_y - idx * 32
        _pdf_line(commands, width - right + 10, legend_y_item, width - right + 40, legend_y_item, color=color, width=2.5)
        _pdf_line(
            commands,
            width - right + 10,
            legend_y_item - 12,
            width - right + 40,
            legend_y_item - 12,
            color=color,
            width=2.0,
            dash="[6 4] 0 d",
        )
        _pdf_text(commands, width - right + 48, legend_y_item - 4, f"{label} val", size=10)
        _pdf_text(commands, width - right + 48, legend_y_item - 16, f"{label} train", size=10)

    _write_single_page_pdf(width, height, commands, out_path)


def default_output_dir(study_dirs: list[Path]) -> Path:
    common = Path(os.path.commonpath([str(path) for path in study_dirs]))
    return common


def write_text(lines: list[str], x: float, y: float, text: str, klass: str = "axis", anchor: str = "start") -> None:
    lines.append(f'<text x="{x}" y="{y}" class="{klass}" text-anchor="{anchor}">{text}</text>')


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a comparison table and loss plot across tuning studies.")
    parser.add_argument(
        "study_paths",
        nargs="+",
        help="One or more study directories, or a root directory containing multiple study directories.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory for the generated report assets. Defaults to the common parent of the selected studies.",
    )
    args = parser.parse_args()

    selected_dirs: list[Path] = []
    for raw in args.study_paths:
        selected_dirs.extend(discover_study_dirs(Path(raw).resolve()))
    unique_dirs = sorted({path.resolve() for path in selected_dirs})
    if not unique_dirs:
        raise ValueError("No study directories with study_summary.json were found.")

    rows = [build_study_row(study_dir) for study_dir in unique_dirs]
    loss_series = [load_loss_history(study_dir) for study_dir in unique_dirs]

    out_dir = Path(args.out_dir).resolve() if args.out_dir else default_output_dir(unique_dirs)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "model_comparison_best_trials_summary.csv"
    variant_csv_path = out_dir / "model_comparison_variant_test_macro_f1.csv"
    supplemental_path = out_dir / "conceptnet_feature_results.json"
    supplemental_csv_path = out_dir / "conceptnet_feature_results_summary.csv"
    report_path = out_dir / "model_comparison_report.md"
    loss_pdf_path = out_dir / "model_comparison_loss_over_time.pdf"
    supplemental_results = load_supplemental_results(supplemental_path)

    write_csv(rows, csv_path)
    write_variant_summary_csv(build_variant_summary(rows), variant_csv_path)
    if supplemental_results:
        write_supplemental_results_csv(supplemental_results, supplemental_csv_path)
    render_markdown(rows, report_path, loss_pdf_path.name, supplemental_results=supplemental_results)
    render_loss_plot_pdf(loss_series, loss_pdf_path)


if __name__ == "__main__":
    main()
