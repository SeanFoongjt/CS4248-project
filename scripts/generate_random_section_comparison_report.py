import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


METRIC_FIELDS = ("accuracy", "precision", "recall", "f1", "loss")
ORIGINAL_RESULTS_FILENAME = "original_test_set_results.csv"
ORIGINAL_PREDICTIONS_FILENAME = "original_test_set_predictions.csv"
RANDOM_SECTION_RESULTS_FILENAME = "experiment_random_section_replacement.csv"
RANDOM_SECTION_PREDICTIONS_FILENAME = (
    "experiment_random_section_replacement_predictions.csv"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a Markdown comparison report for original vs random-section replacement runs."
    )
    parser.add_argument(
        "--original-root",
        type=Path,
        default=Path("runs") / "eval_all_models",
        help="Directory containing original-test run artifacts.",
    )
    parser.add_argument(
        "--random-section-root",
        type=Path,
        default=Path("runs") / "random_section_replacement",
        help="Directory containing random-section replacement run artifacts.",
    )
    parser.add_argument(
        "--original-dir",
        type=Path,
        default=None,
        help="Specific original-test run directory. Defaults to the latest run with original result CSVs.",
    )
    parser.add_argument(
        "--random-section-dir",
        type=Path,
        default=None,
        help="Specific random-section run directory. Defaults to the latest run with result CSVs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Markdown output path. Defaults to runs/random_section_replacement/comparison_report.md.",
    )
    return parser.parse_args()


def find_latest_run_dir(root: Path, required_filename: str) -> Path:
    candidates = [
        path.parent
        for path in root.rglob(required_filename)
        if path.parent.name != "slurm"
    ]
    if not candidates:
        raise FileNotFoundError(f"No run under {root} contains {required_filename}.")
    return max(candidates, key=lambda path: path.name)


def read_csv_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def to_bool(value: str) -> bool:
    return str(value).strip().lower() == "true"


def to_float(value: str) -> float:
    return float(value)


def shorten_model_name(row):
    concept = "with_cn" if to_bool(row["use_conceptnet"]) else "without_cn"
    return f"{row['model_type']}_{concept}"


def load_result_map(results_path: Path):
    results_by_model = {}
    for row in read_csv_rows(results_path):
        row = dict(row)
        row["short_model"] = shorten_model_name(row)
        for field in METRIC_FIELDS:
            row[field] = to_float(row[field])
        results_by_model[row["model_path"]] = row
    return results_by_model


def load_predictions_map(predictions_path: Path):
    predictions_by_model = defaultdict(list)
    for row in read_csv_rows(predictions_path):
        predictions_by_model[row["model_path"]].append(row)
    return predictions_by_model


def build_metric_comparison(original_results, random_results):
    rows = []
    common_models = sorted(set(original_results) & set(random_results))
    for model_path in common_models:
        original = original_results[model_path]
        random_section = random_results[model_path]
        row = {
            "model_path": model_path,
            "short_model": original["short_model"],
        }
        for field in METRIC_FIELDS:
            row[f"original_{field}"] = original[field]
            row[f"random_{field}"] = random_section[field]
            row[f"delta_{field}"] = random_section[field] - original[field]
        rows.append(row)
    rows.sort(key=lambda row: row["delta_f1"])
    return rows


def build_prediction_comparison(original_predictions, random_predictions):
    rows = []
    common_models = sorted(set(original_predictions) & set(random_predictions))
    for model_path in common_models:
        original_rows = original_predictions[model_path]
        random_rows = random_predictions[model_path]
        if len(original_rows) != len(random_rows):
            raise ValueError(
                f"Prediction row count mismatch for {model_path}: "
                f"{len(original_rows)} original vs {len(random_rows)} random-section."
            )

        changed_prediction = 0
        changed_correctness = 0
        correct_to_incorrect = 0
        incorrect_to_correct = 0
        confidence_delta_sum = 0.0
        replaced_section_count = 0
        unchanged_section_count = 0
        replacement_pair_counts = Counter()
        regression_pair_counts = Counter()
        impactful_examples = []

        for index, (orig, rand) in enumerate(zip(original_rows, random_rows), start=1):
            orig_pred = int(orig["predicted_label"])
            rand_pred = int(rand["predicted_label"])
            orig_correct = to_bool(orig["correct"])
            rand_correct = to_bool(rand["correct"])
            orig_conf = to_float(orig["confidence"])
            rand_conf = to_float(rand["confidence"])
            confidence_delta = rand_conf - orig_conf
            original_section = rand.get("original_section", "")
            replacement_section = rand.get("replacement_section", "")

            if orig_pred != rand_pred:
                changed_prediction += 1
            if orig_correct != rand_correct:
                changed_correctness += 1
            if orig_correct and not rand_correct:
                correct_to_incorrect += 1
                regression_pair_counts[(original_section, replacement_section)] += 1
            if not orig_correct and rand_correct:
                incorrect_to_correct += 1

            confidence_delta_sum += confidence_delta

            if original_section != replacement_section:
                replaced_section_count += 1
                replacement_pair_counts[(original_section, replacement_section)] += 1
            else:
                unchanged_section_count += 1

            if orig_correct and not rand_correct:
                impactful_examples.append(
                    {
                        "index": index,
                        "headline": orig["headline"],
                        "true_label": orig["true_label"],
                        "original_prediction": orig["predicted_label"],
                        "random_prediction": rand["predicted_label"],
                        "original_confidence": orig_conf,
                        "random_confidence": rand_conf,
                        "confidence_delta": confidence_delta,
                        "original_section": original_section,
                        "replacement_section": replacement_section,
                    }
                )

        impactful_examples.sort(key=lambda row: row["confidence_delta"])
        rows.append(
            {
                "model_path": model_path,
                "total_examples": len(original_rows),
                "changed_prediction": changed_prediction,
                "changed_prediction_rate": changed_prediction / len(original_rows),
                "changed_correctness": changed_correctness,
                "correct_to_incorrect": correct_to_incorrect,
                "incorrect_to_correct": incorrect_to_correct,
                "net_accuracy_change_examples": incorrect_to_correct - correct_to_incorrect,
                "avg_confidence_delta": confidence_delta_sum / len(original_rows),
                "replaced_section_count": replaced_section_count,
                "unchanged_section_count": unchanged_section_count,
                "top_replacement_pairs": replacement_pair_counts.most_common(8),
                "top_regression_pairs": regression_pair_counts.most_common(8),
                "top_regressions": impactful_examples[:5],
            }
        )

    rows.sort(key=lambda row: row["correct_to_incorrect"], reverse=True)
    return rows


def format_float(value: float) -> str:
    return f"{value:.4f}"


def format_delta(value: float) -> str:
    return f"{value:+.4f}"


def escape_md(value) -> str:
    return str(value).replace("|", "\\|")


def append_pair_table(lines, title: str, pairs):
    lines.append(title)
    lines.append("")
    if not pairs:
        lines.append("No pairs found.")
        lines.append("")
        return

    lines.append("| Original Section | Replacement Section | Count |")
    lines.append("| --- | --- | ---: |")
    for (original_section, replacement_section), count in pairs:
        lines.append(
            f"| {escape_md(original_section)} | {escape_md(replacement_section)} | {count} |"
        )
    lines.append("")


def build_markdown(
    original_dir: Path,
    random_section_dir: Path,
    original_results_path: Path,
    random_results_path: Path,
    original_predictions_path: Path,
    random_predictions_path: Path,
    metric_rows,
    prediction_rows,
):
    lines = []
    lines.append("# Random Section Replacement Comparison Report")
    lines.append("")
    lines.append(
        f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} from downloaded evaluation artifacts."
    )
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- Original run: `{original_dir}`")
    lines.append(f"- Random-section run: `{random_section_dir}`")
    lines.append(f"- Original metrics CSV: `{original_results_path}`")
    lines.append(f"- Random-section metrics CSV: `{random_results_path}`")
    lines.append(f"- Original predictions CSV: `{original_predictions_path}`")
    lines.append(f"- Random-section predictions CSV: `{random_predictions_path}`")
    lines.append("")

    if metric_rows:
        best_original = max(metric_rows, key=lambda row: row["original_f1"])
        best_random = max(metric_rows, key=lambda row: row["random_f1"])
        biggest_drop = min(metric_rows, key=lambda row: row["delta_f1"])

        lines.append("## Headline Findings")
        lines.append("")
        lines.append(
            f"- Best original-set F1: `{best_original['short_model']}` at {format_float(best_original['original_f1'])}."
        )
        lines.append(
            f"- Best random-section F1: `{best_random['short_model']}` at {format_float(best_random['random_f1'])}."
        )
        lines.append(
            f"- Largest F1 drop after random section replacement: `{biggest_drop['short_model']}` at {format_delta(biggest_drop['delta_f1'])}."
        )
        lines.append("")

        lines.append("## Metric Comparison")
        lines.append("")
        lines.append(
            "| Model | Original Acc | Random Acc | Delta Acc | Original F1 | Random F1 | Delta F1 | Original Loss | Random Loss | Delta Loss |"
        )
        lines.append(
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        )
        for row in metric_rows:
            lines.append(
                f"| `{row['short_model']}` | "
                f"{format_float(row['original_accuracy'])} | "
                f"{format_float(row['random_accuracy'])} | "
                f"{format_delta(row['delta_accuracy'])} | "
                f"{format_float(row['original_f1'])} | "
                f"{format_float(row['random_f1'])} | "
                f"{format_delta(row['delta_f1'])} | "
                f"{format_float(row['original_loss'])} | "
                f"{format_float(row['random_loss'])} | "
                f"{format_delta(row['delta_loss'])} |"
            )
        lines.append("")

    if prediction_rows:
        lines.append("## Prediction Stability")
        lines.append("")
        lines.append(
            "| Model | Replaced Sections | Changed Predictions | Changed Pred % | Correct -> Incorrect | Incorrect -> Correct | Net Example Change | Avg Confidence Delta |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for prediction_row in prediction_rows:
            matching_metric = next(
                row
                for row in metric_rows
                if row["model_path"] == prediction_row["model_path"]
            )
            lines.append(
                f"| `{matching_metric['short_model']}` | "
                f"{prediction_row['replaced_section_count']} / {prediction_row['total_examples']} | "
                f"{prediction_row['changed_prediction']} / {prediction_row['total_examples']} | "
                f"{format_float(prediction_row['changed_prediction_rate'])} | "
                f"{prediction_row['correct_to_incorrect']} | "
                f"{prediction_row['incorrect_to_correct']} | "
                f"{prediction_row['net_accuracy_change_examples']} | "
                f"{format_delta(prediction_row['avg_confidence_delta'])} |"
            )
        lines.append("")

        lines.append("## Section Replacement Patterns")
        lines.append("")
        for prediction_row in prediction_rows:
            matching_metric = next(
                row
                for row in metric_rows
                if row["model_path"] == prediction_row["model_path"]
            )
            lines.append(f"### `{matching_metric['short_model']}`")
            lines.append("")
            append_pair_table(
                lines,
                "Top replacement pairs:",
                prediction_row["top_replacement_pairs"],
            )
            append_pair_table(
                lines,
                "Top correct-to-incorrect replacement pairs:",
                prediction_row["top_regression_pairs"],
            )

        lines.append("## Example Regressions")
        lines.append("")
        for prediction_row in prediction_rows:
            matching_metric = next(
                row
                for row in metric_rows
                if row["model_path"] == prediction_row["model_path"]
            )
            lines.append(f"### `{matching_metric['short_model']}`")
            lines.append("")
            if not prediction_row["top_regressions"]:
                lines.append("No correct-to-incorrect regressions were found.")
                lines.append("")
                continue

            lines.append(
                "| Row | True Label | Original Pred | Random Pred | Original Section | Replacement Section | Original Conf | Random Conf | Delta Conf | Headline |"
            )
            lines.append(
                "| ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | --- |"
            )
            for example in prediction_row["top_regressions"]:
                lines.append(
                    f"| {example['index']} | "
                    f"{example['true_label']} | "
                    f"{example['original_prediction']} | "
                    f"{example['random_prediction']} | "
                    f"{escape_md(example['original_section'])} | "
                    f"{escape_md(example['replacement_section'])} | "
                    f"{format_float(example['original_confidence'])} | "
                    f"{format_float(example['random_confidence'])} | "
                    f"{format_delta(example['confidence_delta'])} | "
                    f"{escape_md(example['headline'])} |"
                )
            lines.append("")

    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    original_dir = args.original_dir or find_latest_run_dir(
        args.original_root,
        ORIGINAL_RESULTS_FILENAME,
    )
    random_section_dir = args.random_section_dir or find_latest_run_dir(
        args.random_section_root,
        RANDOM_SECTION_RESULTS_FILENAME,
    )
    output_path = args.output or (
        args.random_section_root / "random_section_comparison_report.md"
    )

    original_results_path = original_dir / ORIGINAL_RESULTS_FILENAME
    original_predictions_path = original_dir / ORIGINAL_PREDICTIONS_FILENAME
    random_results_path = random_section_dir / RANDOM_SECTION_RESULTS_FILENAME
    random_predictions_path = random_section_dir / RANDOM_SECTION_PREDICTIONS_FILENAME

    original_results = load_result_map(original_results_path)
    random_results = load_result_map(random_results_path)
    original_predictions = load_predictions_map(original_predictions_path)
    random_predictions = load_predictions_map(random_predictions_path)

    metric_rows = build_metric_comparison(original_results, random_results)
    prediction_rows = build_prediction_comparison(
        original_predictions,
        random_predictions,
    )
    report = build_markdown(
        original_dir=original_dir,
        random_section_dir=random_section_dir,
        original_results_path=original_results_path,
        random_results_path=random_results_path,
        original_predictions_path=original_predictions_path,
        random_predictions_path=random_predictions_path,
        metric_rows=metric_rows,
        prediction_rows=prediction_rows,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Wrote random-section comparison report to {output_path}")


if __name__ == "__main__":
    main()
