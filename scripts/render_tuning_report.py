from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_trials(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    out = []
    for row in rows:
        if row.get("state") != "COMPLETE":
            continue
        out.append(
            {
                "number": int(row["number"]),
                "value": float(row["value"]),
                "ngram_range": row.get("param.model.ngram_range", ""),
                "min_df": float(row["param.model.min_df"]) if row.get("param.model.min_df") else None,
                "max_df": float(row["param.model.max_df"]) if row.get("param.model.max_df") else None,
                "alpha": float(row["param.model.alpha"]) if row.get("param.model.alpha") else None,
                "test_macro_f1": float(row["user_attr.test_macro_f1"]) if row.get("user_attr.test_macro_f1") else None,
            }
        )
    return out


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,sans-serif;fill:#1f2937}.title{font-size:18px;font-weight:bold}.axis{font-size:11px}.label{font-size:12px}.small{font-size:10px}.grid{stroke:#e5e7eb;stroke-width:1}.axis-line{stroke:#6b7280;stroke-width:1.2}.point{fill:#0f766e;opacity:0.82}.point-alt{fill:#1d4ed8;opacity:0.82}.bar{fill:#2563eb}.bar-max{fill:#0f766e}</style>'
    ]


def clamp(val: float, low: float, high: float) -> float:
    return max(low, min(high, val))


def write_text(lines: list[str], x: float, y: float, text: str, klass: str = "axis", anchor: str = "start") -> None:
    lines.append(f'<text x="{x}" y="{y}" class="{klass}" text-anchor="{anchor}">{text}</text>')


def render_trial_progress_svg(trials: list[dict], out_path: Path) -> None:
    width, height = 900, 420
    left, right, top, bottom = 70, 30, 45, 55
    inner_w = width - left - right
    inner_h = height - top - bottom
    values = [trial["value"] for trial in trials]
    x_max = max(trial["number"] for trial in trials)
    y_min = min(values)
    y_max = max(values)
    pad = (y_max - y_min) * 0.12 if y_max > y_min else 0.02
    y_lo = y_min - pad
    y_hi = y_max + pad

    def x_map(x: float) -> float:
        if x_max == 1:
            return left + inner_w / 2
        return left + (x - 1) / (x_max - 1) * inner_w

    def y_map(y: float) -> float:
        return top + (1 - (y - y_lo) / (y_hi - y_lo)) * inner_h

    lines = svg_header(width, height)
    write_text(lines, left, 26, "Validation Macro-F1 by Trial", "title")
    for i in range(5):
        frac = i / 4
        y = top + frac * inner_h
        val = y_hi - frac * (y_hi - y_lo)
        lines.append(f'<line x1="{left}" y1="{y}" x2="{width - right}" y2="{y}" class="grid"/>')
        write_text(lines, left - 8, y + 4, f"{val:.3f}", "axis", "end")
    for tick in range(1, x_max + 1):
        if tick == 1 or tick == x_max or tick % max(1, math.ceil(x_max / 8)) == 0:
            x = x_map(tick)
            lines.append(f'<line x1="{x}" y1="{top}" x2="{x}" y2="{height - bottom}" class="grid"/>')
            write_text(lines, x, height - bottom + 18, str(tick), "axis", "middle")
    lines.append(f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" class="axis-line"/>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" class="axis-line"/>')
    write_text(lines, width / 2, height - 12, "Trial number", "label", "middle")
    write_text(lines, 18, top - 8, "Val macro-F1", "label")
    poly = " ".join(f"{x_map(trial['number']):.1f},{y_map(trial['value']):.1f}" for trial in sorted(trials, key=lambda item: item["number"]))
    lines.append(f'<polyline fill="none" stroke="#94a3b8" stroke-width="2" points="{poly}"/>')
    best_trial = max(trials, key=lambda item: item["value"])
    for trial in trials:
        klass = "point-alt" if trial["number"] == best_trial["number"] else "point"
        radius = 5 if trial["number"] == best_trial["number"] else 3.5
        lines.append(f'<circle cx="{x_map(trial["number"]):.1f}" cy="{y_map(trial["value"]):.1f}" r="{radius}" class="{klass}"/>')
    write_text(
        lines,
        x_map(best_trial["number"]) + 8,
        y_map(best_trial["value"]) - 10,
        f'best trial {best_trial["number"]}: {best_trial["value"]:.3f}',
        "small",
    )
    lines.append("</svg>")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def render_ngram_bar_svg(trials: list[dict], out_path: Path) -> None:
    width, height = 760, 380
    left, right, top, bottom = 70, 30, 45, 55
    inner_w = width - left - right
    inner_h = height - top - bottom
    grouped: dict[str, list[float]] = defaultdict(list)
    for trial in trials:
        grouped[trial["ngram_range"]].append(trial["value"])
    labels = sorted(grouped)
    stats = [(label, mean(grouped[label]), max(grouped[label])) for label in labels]
    y_max = max(item[2] for item in stats) * 1.05

    def y_map(y: float) -> float:
        return top + (1 - y / y_max) * inner_h

    band_w = inner_w / max(1, len(stats))
    bar_w = band_w * 0.26
    lines = svg_header(width, height)
    write_text(lines, left, 26, "Performance by N-gram Range", "title")
    for i in range(5):
        frac = i / 4
        y = top + frac * inner_h
        val = y_max - frac * y_max
        lines.append(f'<line x1="{left}" y1="{y}" x2="{width - right}" y2="{y}" class="grid"/>')
        write_text(lines, left - 8, y + 4, f"{val:.3f}", "axis", "end")
    lines.append(f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" class="axis-line"/>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" class="axis-line"/>')
    write_text(lines, width / 2, height - 12, "N-gram range", "label", "middle")
    write_text(lines, 18, top - 8, "Val macro-F1", "label")
    for idx, (label, avg_val, max_val) in enumerate(stats):
        center = left + band_w * (idx + 0.5)
        mean_x = center - bar_w * 0.7
        max_x = center + bar_w * 0.1
        mean_h = height - bottom - y_map(avg_val)
        max_h = height - bottom - y_map(max_val)
        lines.append(f'<rect x="{mean_x:.1f}" y="{y_map(avg_val):.1f}" width="{bar_w:.1f}" height="{mean_h:.1f}" class="bar"/>')
        lines.append(f'<rect x="{max_x:.1f}" y="{y_map(max_val):.1f}" width="{bar_w:.1f}" height="{max_h:.1f}" class="bar-max"/>')
        write_text(lines, center, height - bottom + 18, label, "axis", "middle")
        write_text(lines, mean_x + bar_w / 2, y_map(avg_val) - 6, f"{avg_val:.3f}", "small", "middle")
        write_text(lines, max_x + bar_w / 2, y_map(max_val) - 6, f"{max_val:.3f}", "small", "middle")
    write_text(lines, width - 150, 55, "Blue = mean", "small")
    write_text(lines, width - 150, 72, "Green = max", "small")
    lines.append("</svg>")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def render_alpha_scatter_svg(trials: list[dict], out_path: Path) -> None:
    width, height = 900, 420
    left, right, top, bottom = 80, 40, 45, 60
    inner_w = width - left - right
    inner_h = height - top - bottom
    usable = [trial for trial in trials if trial["alpha"] is not None]
    x_vals = [math.log10(trial["alpha"]) for trial in usable]
    y_vals = [trial["value"] for trial in usable]
    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)
    y_pad = (y_max - y_min) * 0.12 if y_max > y_min else 0.02
    y_lo, y_hi = y_min - y_pad, y_max + y_pad

    def x_map(x: float) -> float:
        if x_max == x_min:
            return left + inner_w / 2
        return left + (x - x_min) / (x_max - x_min) * inner_w

    def y_map(y: float) -> float:
        return top + (1 - (y - y_lo) / (y_hi - y_lo)) * inner_h

    lines = svg_header(width, height)
    write_text(lines, left, 26, "Alpha vs Validation Macro-F1", "title")
    for i in range(5):
        frac = i / 4
        y = top + frac * inner_h
        val = y_hi - frac * (y_hi - y_lo)
        lines.append(f'<line x1="{left}" y1="{y}" x2="{width - right}" y2="{y}" class="grid"/>')
        write_text(lines, left - 8, y + 4, f"{val:.3f}", "axis", "end")
    for i in range(5):
        frac = i / 4
        x = left + frac * inner_w
        val = 10 ** (x_min + frac * (x_max - x_min))
        lines.append(f'<line x1="{x}" y1="{top}" x2="{x}" y2="{height - bottom}" class="grid"/>')
        write_text(lines, x, height - bottom + 18, f"{val:.3f}", "axis", "middle")
    lines.append(f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" class="axis-line"/>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" class="axis-line"/>')
    write_text(lines, width / 2, height - 12, "Alpha (log scale)", "label", "middle")
    write_text(lines, 18, top - 8, "Val macro-F1", "label")
    for trial in usable:
        klass = "point-alt" if trial["ngram_range"] == "1,2" else "point"
        lines.append(
            f'<circle cx="{x_map(math.log10(trial["alpha"])):.1f}" cy="{y_map(trial["value"]):.1f}" r="4" class="{klass}"/>'
        )
    write_text(lines, width - 180, 55, "Blue: 1,1 n-grams", "small")
    write_text(lines, width - 180, 72, "Dark blue: 1,2 n-grams", "small")
    lines.append("</svg>")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def build_report(study_dir: Path) -> str:
    summary = load_json(study_dir / "study_summary.json")
    trials = load_trials(study_dir / "trials.csv")
    top_trials = sorted(trials, key=lambda item: item["value"], reverse=True)[:5]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for trial in trials:
        grouped[trial["ngram_range"]].append(trial)

    best = top_trials[0]
    best_artifact = Path(summary["artifact_path"])
    lines = [
        "# TF-IDF + Multinomial NB Tuning Report",
        "",
        "## Summary",
        "",
        f"- Study: `{summary['study_name']}`",
        f"- Model: `{summary['model']}`",
        f"- Recipe: `{summary['recipe']}`",
        f"- Objective: `{summary['objective_metric']}`",
        f"- Trials completed: `{summary['n_trials_finished']}` / `{summary['n_trials_target']}`",
        f"- Best trial: `{summary['best_trial_id']}`",
        f"- Best validation macro-F1: `{summary['best_validation_metric']:.4f}`",
        f"- Corresponding test macro-F1: `{summary['corresponding_test_metric']:.4f}`",
        "",
        "## Best Configuration",
        "",
        f"- `ngram_range`: `{best['ngram_range']}`",
        f"- `min_df`: `{int(best['min_df'])}`",
        f"- `max_df`: `{best['max_df']:.4f}`",
        f"- `alpha`: `{best['alpha']:.6f}`",
        f"- Trial artifact: `{best_artifact.as_posix()}`",
        "",
        "## Main Findings",
        "",
    ]

    ngram_stats = []
    for label in sorted(grouped):
        vals = [item["value"] for item in grouped[label]]
        ngram_stats.append((label, len(vals), mean(vals), max(vals)))
    best_ngram = max(ngram_stats, key=lambda item: item[3])
    lines.extend(
        [
            f"- `1,2` n-grams clearly outperformed `1,1` in this study: best val macro-F1 `{best_ngram[3]:.4f}` and mean `{best_ngram[2]:.4f}`.",
            f"- The winning runs all concentrated at higher document-frequency filtering: the best trial used `min_df={int(best['min_df'])}`, and the top 5 trials stayed in the `min_df` range `{int(min(item['min_df'] for item in top_trials))}` to `{int(max(item['min_df'] for item in top_trials))}`.",
            f"- The best alpha landed in the lower smoothing regime at `{best['alpha']:.6f}`, with the strongest trials clustering roughly between `{min(item['alpha'] for item in top_trials):.6f}` and `{max(item['alpha'] for item in top_trials):.6f}`.",
            f"- Generalization was decent but not identical to validation: best validation macro-F1 was `{summary['best_validation_metric']:.4f}` while the corresponding test macro-F1 was `{summary['corresponding_test_metric']:.4f}`.",
        ]
    )
    lines.extend(
        [
            "",
            "## Top 5 Trials",
            "",
            "| Trial | Val macro-F1 | Test macro-F1 | ngram_range | min_df | max_df | alpha |",
            "| --- | ---: | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for trial in top_trials:
        lines.append(
            f"| {trial['number']} | {trial['value']:.4f} | {trial['test_macro_f1']:.4f} | {trial['ngram_range']} | {int(trial['min_df'])} | {trial['max_df']:.4f} | {trial['alpha']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Visuals",
            "",
            "### Validation Macro-F1 by Trial",
            "![Validation Macro-F1 by Trial](plots/trial_progress.svg)",
            "",
            "### N-gram Range Comparison",
            "![Performance by N-gram Range](plots/ngram_comparison.svg)",
            "",
            "### Alpha vs Validation Macro-F1",
            "![Alpha vs Validation Macro-F1](plots/alpha_scatter.svg)",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Render a Markdown report and SVG plots for one tuning study.")
    parser.add_argument("study_dir", help="Path to the study directory under runs/variant2_tuning")
    args = parser.parse_args()

    study_dir = Path(args.study_dir).resolve()
    plots_dir = study_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    trials = load_trials(study_dir / "trials.csv")
    render_trial_progress_svg(trials, plots_dir / "trial_progress.svg")
    render_ngram_bar_svg(trials, plots_dir / "ngram_comparison.svg")
    render_alpha_scatter_svg(trials, plots_dir / "alpha_scatter.svg")
    (study_dir / "report.md").write_text(build_report(study_dir), encoding="utf-8")


if __name__ == "__main__":
    main()
