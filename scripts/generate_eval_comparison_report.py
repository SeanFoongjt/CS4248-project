import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path


RESULTS_FILENAMES = {
    "original": "original_test_set_results.csv",
    "shuffled": "experiment_shuffle_description.csv",
}

PREDICTIONS_FILENAMES = {
    "original": "original_test_set_predictions.csv",
    "shuffled": "experiment_shuffle_description_predictions.csv",
}

METRIC_FIELDS = ("accuracy", "precision", "recall", "f1", "loss")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a Markdown comparison report for original vs shuffled evaluation runs."
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs") / "eval_all_models",
        help="Root directory containing original_test/ and shuffled_test/ subfolders.",
    )
    parser.add_argument(
        "--original-dir",
        type=Path,
        default=None,
        help="Specific original_test run directory. Defaults to the latest run with result CSVs.",
    )
    parser.add_argument(
        "--shuffled-dir",
        type=Path,
        default=None,
        help="Specific shuffled_test run directory. Defaults to the latest run with result CSVs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Markdown output path. Defaults to runs/eval_all_models/comparison_report.md.",
    )
    return parser.parse_args()


def find_latest_run_dir(parent: Path, required_filename: str) -> Path:
    candidates = [
        child
        for child in parent.iterdir()
        if child.is_dir() and child.name != "slurm" and (child / required_filename).exists()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No run directory under {parent} contains {required_filename}."
        )
    return max(candidates, key=lambda path: path.name)


def read_csv_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(value: str) -> float:
    return float(value)


def to_bool(value: str) -> bool:
    return str(value).strip().lower() == "true"


def shorten_model_name(row):
    backbone = row["model_type"]
    concept = "with_cn" if to_bool(row["use_conceptnet"]) else "without_cn"
    return f"{backbone}_{concept}"


def load_result_maps(run_dir: Path, kind: str):
    results_path = run_dir / RESULTS_FILENAMES[kind]
    predictions_path = run_dir / PREDICTIONS_FILENAMES[kind]
    results_rows = read_csv_rows(results_path)
    prediction_rows = read_csv_rows(predictions_path)

    results_by_model = {}
    for row in results_rows:
        key = row["model_path"]
        row = dict(row)
        row["short_model"] = shorten_model_name(row)
        for field in METRIC_FIELDS:
            row[field] = to_float(row[field])
        results_by_model[key] = row

    predictions_by_model = defaultdict(list)
    for row in prediction_rows:
        predictions_by_model[row["model_path"]].append(row)

    return results_by_model, predictions_by_model, results_path, predictions_path


def build_metric_comparison(original_results, shuffled_results):
    rows = []
    common_models = sorted(set(original_results) & set(shuffled_results))
    for model_path in common_models:
        original = original_results[model_path]
        shuffled = shuffled_results[model_path]
        row = {
            "model_path": model_path,
            "short_model": original["short_model"],
        }
        for field in METRIC_FIELDS:
            row[f"original_{field}"] = original[field]
            row[f"shuffled_{field}"] = shuffled[field]
            row[f"delta_{field}"] = shuffled[field] - original[field]
        rows.append(row)
    rows.sort(key=lambda row: row["delta_f1"])
    return rows


def build_prediction_comparison(original_predictions, shuffled_predictions):
    rows = []
    common_models = sorted(set(original_predictions) & set(shuffled_predictions))
    for model_path in common_models:
        original_rows = original_predictions[model_path]
        shuffled_rows = shuffled_predictions[model_path]
        if len(original_rows) != len(shuffled_rows):
            raise ValueError(
                f"Prediction row count mismatch for {model_path}: "
                f"{len(original_rows)} original vs {len(shuffled_rows)} shuffled."
            )

        changed_prediction = 0
        changed_correctness = 0
        correct_to_incorrect = 0
        incorrect_to_correct = 0
        unchanged_wrong = 0
        confidence_drop_sum = 0.0
        impactful_examples = []

        for index, (orig, shuf) in enumerate(zip(original_rows, shuffled_rows), start=1):
            orig_pred = int(orig["predicted_label"])
            shuf_pred = int(shuf["predicted_label"])
            orig_correct = to_bool(orig["correct"])
            shuf_correct = to_bool(shuf["correct"])
            orig_conf = to_float(orig["confidence"])
            shuf_conf = to_float(shuf["confidence"])
            confidence_delta = shuf_conf - orig_conf

            if orig_pred != shuf_pred:
                changed_prediction += 1
            if orig_correct != shuf_correct:
                changed_correctness += 1
            if orig_correct and not shuf_correct:
                correct_to_incorrect += 1
            if not orig_correct and shuf_correct:
                incorrect_to_correct += 1
            if (not orig_correct) and (not shuf_correct):
                unchanged_wrong += 1

            confidence_drop_sum += confidence_delta

            if orig_correct and not shuf_correct:
                impactful_examples.append(
                    {
                        "headline": orig["headline"],
                        "true_label": orig["true_label"],
                        "original_prediction": orig["predicted_label"],
                        "shuffled_prediction": shuf["predicted_label"],
                        "original_confidence": orig_conf,
                        "shuffled_confidence": shuf_conf,
                        "confidence_delta": confidence_delta,
                        "index": index,
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
                "unchanged_wrong": unchanged_wrong,
                "avg_confidence_delta": confidence_drop_sum / len(original_rows),
                "top_regressions": impactful_examples[:5],
            }
        )

    rows.sort(key=lambda row: row["correct_to_incorrect"], reverse=True)
    return rows


def format_float(value: float) -> str:
    return f"{value:.4f}"


def format_delta(value: float) -> str:
    return f"{value:+.4f}"


def build_markdown(
    original_dir: Path,
    shuffled_dir: Path,
    original_results_path: Path,
    shuffled_results_path: Path,
    original_predictions_path: Path,
    shuffled_predictions_path: Path,
    metric_rows,
    prediction_rows,
):
    lines = []
    lines.append("# Evaluation Comparison Report")
    lines.append("")
    lines.append(
        f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} from the latest downloaded evaluation artifacts."
    )
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- Original run: `{original_dir}`")
    lines.append(f"- Shuffled run: `{shuffled_dir}`")
    lines.append(f"- Original metrics CSV: `{original_results_path}`")
    lines.append(f"- Shuffled metrics CSV: `{shuffled_results_path}`")
    lines.append(f"- Original predictions CSV: `{original_predictions_path}`")
    lines.append(f"- Shuffled predictions CSV: `{shuffled_predictions_path}`")
    lines.append("")

    if metric_rows:
        best_original = max(metric_rows, key=lambda row: row["original_f1"])
        best_shuffled = max(metric_rows, key=lambda row: row["shuffled_f1"])
        biggest_drop = min(metric_rows, key=lambda row: row["delta_f1"])

        lines.append("## Headline Findings")
        lines.append("")
        lines.append(
            f"- Best original-set F1: `{best_original['short_model']}` at {format_float(best_original['original_f1'])}."
        )
        lines.append(
            f"- Best shuffled-set F1: `{best_shuffled['short_model']}` at {format_float(best_shuffled['shuffled_f1'])}."
        )
        lines.append(
            f"- Largest F1 drop after description shuffling: `{biggest_drop['short_model']}` at {format_delta(biggest_drop['delta_f1'])}."
        )
        lines.append("")

        lines.append("## Metric Comparison")
        lines.append("")
        lines.append(
            "| Model | Original Acc | Shuffled Acc | Delta Acc | Original F1 | Shuffled F1 | Delta F1 | Original Loss | Shuffled Loss | Delta Loss |"
        )
        lines.append(
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        )
        for row in metric_rows:
            lines.append(
                f"| `{row['short_model']}` | "
                f"{format_float(row['original_accuracy'])} | "
                f"{format_float(row['shuffled_accuracy'])} | "
                f"{format_delta(row['delta_accuracy'])} | "
                f"{format_float(row['original_f1'])} | "
                f"{format_float(row['shuffled_f1'])} | "
                f"{format_delta(row['delta_f1'])} | "
                f"{format_float(row['original_loss'])} | "
                f"{format_float(row['shuffled_loss'])} | "
                f"{format_delta(row['delta_loss'])} |"
            )
        lines.append("")

    if prediction_rows:
        lines.append("## Prediction Stability")
        lines.append("")
        lines.append(
            "| Model | Changed Predictions | Changed Pred % | Correct -> Incorrect | Incorrect -> Correct | Net Example Change | Avg Confidence Delta |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for prediction_row in prediction_rows:
            matching_metric = next(
                row
                for row in metric_rows
                if row["model_path"] == prediction_row["model_path"]
            )
            lines.append(
                f"| `{matching_metric['short_model']}` | "
                f"{prediction_row['changed_prediction']} / {prediction_row['total_examples']} | "
                f"{format_float(prediction_row['changed_prediction_rate'])} | "
                f"{prediction_row['correct_to_incorrect']} | "
                f"{prediction_row['incorrect_to_correct']} | "
                f"{prediction_row['net_accuracy_change_examples']} | "
                f"{format_delta(prediction_row['avg_confidence_delta'])} |"
            )
        lines.append("")

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
                "| Row | True Label | Original Pred | Shuffled Pred | Original Conf | Shuffled Conf | Delta Conf | Headline |"
            )
            lines.append(
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"
            )
            for example in prediction_row["top_regressions"]:
                headline = example["headline"].replace("|", "\\|")
                lines.append(
                    f"| {example['index']} | "
                    f"{example['true_label']} | "
                    f"{example['original_prediction']} | "
                    f"{example['shuffled_prediction']} | "
                    f"{format_float(example['original_confidence'])} | "
                    f"{format_float(example['shuffled_confidence'])} | "
                    f"{format_delta(example['confidence_delta'])} | "
                    f"{headline} |"
                )
            lines.append("")

    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    runs_root = args.runs_root
    original_dir = args.original_dir or find_latest_run_dir(
        runs_root / "original_test", RESULTS_FILENAMES["original"]
    )
    shuffled_dir = args.shuffled_dir or find_latest_run_dir(
        runs_root / "shuffled_test", RESULTS_FILENAMES["shuffled"]
    )
    output_path = args.output or (runs_root / "comparison_report.md")

    original_results, original_predictions, original_results_path, original_predictions_path = load_result_maps(
        original_dir, "original"
    )
    shuffled_results, shuffled_predictions, shuffled_results_path, shuffled_predictions_path = load_result_maps(
        shuffled_dir, "shuffled"
    )

    metric_rows = build_metric_comparison(original_results, shuffled_results)
    prediction_rows = build_prediction_comparison(original_predictions, shuffled_predictions)

    report = build_markdown(
        original_dir=original_dir,
        shuffled_dir=shuffled_dir,
        original_results_path=original_results_path,
        shuffled_results_path=shuffled_results_path,
        original_predictions_path=original_predictions_path,
        shuffled_predictions_path=shuffled_predictions_path,
        metric_rows=metric_rows,
        prediction_rows=prediction_rows,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Wrote comparison report to {output_path}")


if __name__ == "__main__":
    main()
