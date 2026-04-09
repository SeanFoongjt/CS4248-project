import argparse
import csv
import logging
import os
from datetime import datetime

import torch
from sklearn.model_selection import train_test_split

from utils.experiment_eval import (
    discover_default_all_checkpoints,
    evaluate_checkpoint,
    load_samples,
    predict_checkpoint,
)
from utils.logger_setup import setup_logger


def format_result_block(result: dict, seed: int, input_path: str) -> str:
    return (
        "\n=========================================\n"
        "        FINAL ORIGINAL TEST RESULTS      \n"
        "=========================================\n"
        f"Model:         {result['model_path']}\n"
        f"Input:         {input_path}\n"
        f"Seed:          {seed}\n"
        f"Model Type:    {result['model_type']}\n"
        f"Use ConceptNet:{result['use_conceptnet']}\n"
        f"Text Format:   {result['text_format']}\n"
        f"Loss:          {result['loss']:.4f}\n"
        f"Accuracy:      {result['accuracy']:.4f}\n"
        f"Precision:     {result['precision']:.4f}\n"
        f"Recall:        {result['recall']:.4f}\n"
        f"F1-Score:      {result['f1']:.4f}\n"
        "=========================================\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained GNN checkpoints on the original unshuffled test split.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/Preprocessed_Article_Section_Description.json",
        help="Path to the dataset file. Supports JSONL and JSON array formats.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="result/original_test_eval",
        help="Directory to save logs and summary outputs.",
    )
    parser.add_argument(
        "--model",
        nargs="*",
        default=None,
        help="Checkpoint path(s). Defaults to the four tuned `all` checkpoints under result/.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed used to recover the test split.")

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    log_path = os.path.join(args.output, f"training_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
    setup_logger(path=log_path)

    model_paths = args.model if args.model else discover_default_all_checkpoints()
    if not model_paths:
        raise FileNotFoundError("No default `all` checkpoints were found under result/.")

    logging.info(f"Evaluating {len(model_paths)} checkpoint(s) on the original test set.")
    for model_path in model_paths:
        logging.info(f" - {model_path}")

    samples = load_samples(args.input)
    logging.info(f"Loaded {len(samples)} total samples from {args.input}.")

    _, data_test = train_test_split(samples, test_size=0.2, random_state=args.seed)
    logging.info(f"Recovered original test split with {len(data_test)} samples using seed={args.seed}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    prediction_rows = []

    for model_path in model_paths:
        logging.info(f"Evaluating original test set with checkpoint: {model_path}")
        result = evaluate_checkpoint(model_path, data_test, args.output, device)
        results.append(result)
        prediction_rows.extend(predict_checkpoint(model_path, data_test, args.output, device))
        block = format_result_block(result, args.seed, args.input)
        logging.info(block)
        print(block)

    summary_path = os.path.join(args.output, "original_test_set_results.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(format_result_block(result, args.seed, args.input))

    csv_path = os.path.join(args.output, "original_test_set_results.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
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

    predictions_csv_path = os.path.join(args.output, "original_test_set_predictions.csv")
    with open(predictions_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_path",
                "model_type",
                "pretrained_name",
                "use_conceptnet",
                "text_format",
                "headline",
                "true_label",
                "predicted_label",
                "correct",
                "prob_not_sarcastic",
                "prob_sarcastic",
                "confidence",
            ],
        )
        writer.writeheader()
        writer.writerows(prediction_rows)


if __name__ == "__main__":
    main()
