import argparse
import csv
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.experiment_eval import (
    discover_default_all_checkpoints,
    evaluate_and_predict_checkpoint,
    load_samples,
    log_checkpoints,
)
from utils.logger_setup import setup_logger
from utils.section_replacement import load_section_pool, replace_article_sections


PREDICTION_EXTRA_FIELDS = [
    "original_section",
    "replacement_section",
]


def format_result_block(result: dict, seed: int, section_counts_path: str, input_path: str) -> str:
    return (
        "\n=========================================\n"
        "  FINAL RANDOM SECTION REPLACEMENT RESULTS\n"
        "=========================================\n"
        f"Model:          {result['model_path']}\n"
        f"Input:          {input_path}\n"
        f"Section Counts: {section_counts_path}\n"
        f"Seed:           {seed}\n"
        f"Model Type:     {result['model_type']}\n"
        f"Use ConceptNet: {result['use_conceptnet']}\n"
        f"Text Format:    {result['text_format']}\n"
        f"Loss:           {result['loss']:.4f}\n"
        f"Accuracy:       {result['accuracy']:.4f}\n"
        f"Precision:      {result['precision']:.4f}\n"
        f"Recall:         {result['recall']:.4f}\n"
        f"F1-Score:       {result['f1']:.4f}\n"
        "=========================================\n"
    )


def write_results(output_dir: str, results: list[dict], prediction_rows: list[dict], args) -> None:
    summary_path = os.path.join(output_dir, "experiment_random_section_replacement.txt")
    with open(summary_path, "w", encoding="utf-8") as handle:
        for result in results:
            handle.write(
                format_result_block(result, args.seed, args.section_counts, args.input)
            )

    csv_path = os.path.join(output_dir, "experiment_random_section_replacement.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model_path",
                "model_type",
                "pretrained_name",
                "use_conceptnet",
                "text_format",
                "max_length",
                "batch_size",
                "loss",
                "accuracy",
                "precision",
                "recall",
                "f1",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    predictions_csv_path = os.path.join(
        output_dir,
        "experiment_random_section_replacement_predictions.csv",
    )
    with open(predictions_csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model_path",
                "model_type",
                "pretrained_name",
                "use_conceptnet",
                "text_format",
                "max_length",
                "batch_size",
                "headline",
                "true_label",
                "predicted_label",
                "correct",
                "prob_not_sarcastic",
                "prob_sarcastic",
                "confidence",
                "original_section",
                "replacement_section",
            ],
        )
        writer.writeheader()
        writer.writerows(prediction_rows)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained GNN checkpoints on a test split with random section replacements."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/Preprocessed_Article_Section_Description.json",
        help="Path to the dataset file. Supports JSONL and JSON array formats.",
    )
    parser.add_argument(
        "--section-counts",
        type=str,
        default="data/article_section_counts/preprocessed_article_section_counts_by_source.csv",
        help="CSV containing preprocessed article-section counts by source.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="result/experiments",
        help="Directory to save logs and summary outputs.",
    )
    parser.add_argument(
        "--model",
        nargs="*",
        default=None,
        help="Checkpoint path(s). Defaults to the four tuned `all` checkpoints under result/.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used to recover the test split and sample replacement sections.",
    )

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    log_path = os.path.join(
        args.output,
        f"training_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log",
    )
    setup_logger(path=log_path)

    model_paths = args.model if args.model else discover_default_all_checkpoints()
    if not model_paths:
        raise FileNotFoundError("No default `all` checkpoints were found under result/.")
    log_checkpoints(model_paths)

    section_labels, section_weights = load_section_pool(args.section_counts)
    logging.info("Loaded %d non-missing replacement section labels.", len(section_labels))

    samples = load_samples(args.input)
    logging.info("Loaded %d total samples from %s.", len(samples), args.input)

    _, data_test = train_test_split(samples, test_size=0.2, random_state=args.seed)
    logging.info(
        "Recovered test split with %d samples using seed=%d.",
        len(data_test),
        args.seed,
    )

    replaced_test = replace_article_sections(
        data_test,
        section_labels,
        section_weights,
        seed=args.seed,
    )
    replaced_count = sum(
        item.get("original_section") != item.get("replacement_section")
        for item in replaced_test
    )
    eligible_count = sum(
        item.get("original_section") not in ("", "<missing>")
        for item in replaced_test
    )
    logging.info(
        "Randomly replaced %d/%d non-missing article sections.",
        replaced_count,
        eligible_count,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    results = []
    prediction_rows = []

    for model_path in model_paths:
        logging.info("Evaluating random-section test set with checkpoint: %s", model_path)
        result, model_prediction_rows = evaluate_and_predict_checkpoint(
            model_path,
            replaced_test,
            args.output,
            device,
            prediction_extra_fields=PREDICTION_EXTRA_FIELDS,
        )
        results.append(result)
        prediction_rows.extend(model_prediction_rows)
        block = format_result_block(result, args.seed, args.section_counts, args.input)
        logging.info(block)
        print(block)

    write_results(args.output, results, prediction_rows, args)


if __name__ == "__main__":
    main()
