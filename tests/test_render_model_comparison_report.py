from __future__ import annotations

import json
from pathlib import Path

from scripts import render_model_comparison_report as report


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_resolve_metrics_path_prefers_best_model_layout(tmp_path):
    study_dir = tmp_path / "roberta_headline_only"
    write_json(
        study_dir / "study_summary.json",
        {
            "study_name": "roberta_headline_only",
            "model": "roberta",
            "recipe": "headline_only",
            "objective_metric": "val.macro_f1",
            "best_trial_id": 7,
            "best_validation_metric": 0.9,
            "corresponding_test_metric": 0.88,
        },
    )
    metrics_path = study_dir / "best_model" / "roberta" / "headline_only" / "metrics.json"
    checkpoint_path = study_dir / "best_model" / "roberta" / "headline_only" / "checkpoint"
    write_json(metrics_path, {"val": {"accuracy": 0.9, "macro_f1": 0.9}, "test": {"accuracy": 0.88, "macro_f1": 0.88}})
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    write_json(
        study_dir / "best_model" / "materialization.json",
        {
            "best_trial_id": 7,
            "best_model_path": str(checkpoint_path),
            "metrics_path": str(metrics_path),
        },
    )

    resolved_metrics, resolved_artifact = report.resolve_metrics_path(study_dir)

    assert resolved_metrics == metrics_path
    assert resolved_artifact == checkpoint_path


def test_build_study_row_falls_back_to_trial_layout(tmp_path):
    study_dir = tmp_path / "tfidf_nb_headline_section"
    write_json(
        study_dir / "study_summary.json",
        {
            "study_name": "tfidf_nb_headline_section",
            "model": "tfidf_nb",
            "recipe": "headline_section",
            "objective_metric": "val.macro_f1",
            "best_trial_id": 3,
            "best_validation_metric": 0.82,
            "corresponding_test_metric": 0.75,
            "artifact_path": "/remote/trial_00003",
        },
    )
    metrics_path = study_dir / "trials" / "trial_00003" / "tfidf_nb" / "headline_section" / "metrics.json"
    write_json(
        metrics_path,
        {
            "val": {"accuracy": 0.81, "macro_f1": 0.82},
            "test": {"accuracy": 0.74, "macro_f1": 0.75},
        },
    )

    row = report.build_study_row(study_dir)

    assert row.study_name == "tfidf_nb_headline_section"
    assert row.val_accuracy == 0.81
    assert row.val_macro_f1 == 0.82
    assert row.test_accuracy == 0.74
    assert row.test_macro_f1 == 0.75
    assert row.metrics_path == str(metrics_path)


def test_render_loss_plot_handles_missing_history(tmp_path):
    out_path = tmp_path / "loss.svg"

    report.render_loss_plot([("tfidf_nb / headline_section", [])], out_path)

    content = out_path.read_text(encoding="utf-8")
    assert "No epoch loss history available" in content


def test_render_loss_plot_pdf_writes_file(tmp_path):
    out_path = tmp_path / "loss.pdf"

    report.render_loss_plot_pdf(
        [
            (
                "roberta / headline_only",
                [
                    {"epoch": 1.0, "train_loss": 0.6, "val_loss": 0.5},
                    {"epoch": 2.0, "train_loss": 0.4, "val_loss": 0.35},
                ],
            )
        ],
        out_path,
    )

    content = out_path.read_bytes()
    assert content.startswith(b"%PDF-1.4")


def test_default_output_dir_uses_common_parent(tmp_path):
    study_dirs = [
        tmp_path / "runs" / "variant2_tuning" / "tfidf_nb_headline_section",
        tmp_path / "runs" / "variant2_tuning" / "roberta_headline_only",
    ]

    out_dir = report.default_output_dir(study_dirs)

    assert out_dir == tmp_path / "runs" / "variant2_tuning"
