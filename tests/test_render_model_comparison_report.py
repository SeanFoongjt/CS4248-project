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


def test_build_variant_summary_pivots_transformer_results():
    rows = [
        report.StudyRow(
            study_name="distilbert_headline_only",
            model="distilbert",
            recipe="headline_only",
            best_trial_id=4,
            objective_metric="val.macro_f1",
            val_accuracy=0.91,
            val_macro_f1=0.90,
            test_accuracy=0.89,
            test_macro_f1=0.88,
            train_loss=0.3,
            val_loss=0.4,
            test_loss=0.5,
            best_epoch=3,
            metrics_path=None,
            artifact_path=None,
        ),
        report.StudyRow(
            study_name="roberta_headline_only",
            model="roberta",
            recipe="headline_only",
            best_trial_id=7,
            objective_metric="val.macro_f1",
            val_accuracy=0.93,
            val_macro_f1=0.92,
            test_accuracy=0.90,
            test_macro_f1=0.89,
            train_loss=0.2,
            val_loss=0.3,
            test_loss=0.4,
            best_epoch=4,
            metrics_path=None,
            artifact_path=None,
        ),
        report.StudyRow(
            study_name="tfidf_nb_headline_only",
            model="tfidf_nb",
            recipe="headline_only",
            best_trial_id=1,
            objective_metric="val.macro_f1",
            val_accuracy=0.7,
            val_macro_f1=0.7,
            test_accuracy=0.69,
            test_macro_f1=0.68,
            train_loss=None,
            val_loss=None,
            test_loss=None,
            best_epoch=None,
            metrics_path=None,
            artifact_path=None,
        ),
    ]

    summary = report.build_variant_summary(rows)

    assert summary == [
        {
            "variant": "headline_only",
            "variant_label": "headline only",
            "distilbert": 0.88,
            "roberta": 0.89,
        }
    ]


def test_render_markdown_includes_variant_summary_table(tmp_path):
    out_path = tmp_path / "report.md"
    rows = [
        report.StudyRow(
            study_name="distilbert_headline_section",
            model="distilbert",
            recipe="headline_section",
            best_trial_id=16,
            objective_metric="val.macro_f1",
            val_accuracy=0.95,
            val_macro_f1=0.9499,
            test_accuracy=0.9467,
            test_macro_f1=0.9467,
            train_loss=0.1,
            val_loss=0.2,
            test_loss=0.3,
            best_epoch=5,
            metrics_path=None,
            artifact_path=None,
        ),
        report.StudyRow(
            study_name="roberta_headline_section",
            model="roberta",
            recipe="headline_section",
            best_trial_id=12,
            objective_metric="val.macro_f1",
            val_accuracy=0.96,
            val_macro_f1=0.9555,
            test_accuracy=0.9512,
            test_macro_f1=0.9512,
            train_loss=0.1,
            val_loss=0.2,
            test_loss=0.3,
            best_epoch=4,
            metrics_path=None,
            artifact_path=None,
        ),
    ]

    report.render_markdown(rows, out_path, "loss.pdf")

    content = out_path.read_text(encoding="utf-8")
    assert "## Best Test Macro-F1 by Data Variant" in content
    assert "| Data Variant | DistilBERT | RoBERTa |" in content
    assert "| headline section | 0.9467 | 0.9512 |" in content


def test_render_markdown_includes_supplemental_results_tables(tmp_path):
    out_path = tmp_path / "report.md"
    rows = [
        report.StudyRow(
            study_name="distilbert_headline_only",
            model="distilbert",
            recipe="headline_only",
            best_trial_id=4,
            objective_metric="val.macro_f1",
            val_accuracy=0.91,
            val_macro_f1=0.90,
            test_accuracy=0.89,
            test_macro_f1=0.88,
            train_loss=0.3,
            val_loss=0.4,
            test_loss=0.5,
            best_epoch=3,
            metrics_path=None,
            artifact_path=None,
        )
    ]
    supplemental = {
        "title": "ConceptNet Feature Ablation on Test Set",
        "description": "Additional project model results.",
        "models": [
            {"id": "distilbert_with", "label": "DistilBERT (With)"},
            {"id": "distilbert_without", "label": "DistilBERT (Without)"},
        ],
        "variants": [
            {"id": "headline", "label": "Headline"},
        ],
        "results": {
            "headline": {
                "distilbert_with": {"loss": 1.1996, "accuracy": 0.9132, "precision": 0.9289, "recall": 0.9120, "f1": 0.9204},
                "distilbert_without": {"loss": 1.1304, "accuracy": 0.9132, "precision": 0.9304, "recall": 0.9103, "f1": 0.9202},
            }
        },
    }

    report.render_markdown(rows, out_path, "loss.pdf", supplemental_results=supplemental)

    content = out_path.read_text(encoding="utf-8")
    assert "## ConceptNet Feature Ablation on Test Set" in content
    assert "### F1-Score" in content
    assert "| Data Variant | DistilBERT (With) | DistilBERT (Without) |" in content
    assert "| Headline | 0.9204 | 0.9202 |" in content
