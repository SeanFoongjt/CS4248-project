from __future__ import annotations

import json
from pathlib import Path

import optuna
import pandas as pd
import pytest
import yaml
from optuna.pruners import MedianPruner, NopPruner
from optuna.trial import FixedTrial

from training import tune_variant2 as tune
from training import variant2_context as v2


def fake_load_preprocessors():
    def prep_sec(x):
        txt = str(x).strip().lower()
        txt = txt.replace("news in brief", "news")
        txt = txt.replace("news in photos", "news")
        return txt.replace("[", "").replace("]", "").replace("'", "")

    def prep_desc(x):
        txt = str(x).strip()
        txt = txt.replace("WASHINGTON—", "")
        txt = txt.replace("The Onion", "[ORG]")
        return txt.strip()

    def prep_bow(x):
        return str(x).lower().replace(":", " ").replace(",", " ").strip()

    return prep_sec, prep_desc, prep_bow


def write_jsonl(tmp_path: Path) -> Path:
    rows = [
        {
            "is_sarcastic": 1,
            "headline": "fun weather saves liar from work",
            "article_section": "['News', 'News In Brief']",
            "description": "WASHINGTON—The Onion says hi",
        },
        {
            "is_sarcastic": 0,
            "headline": "city opens new train line",
            "article_section": "Politics",
            "description": "A normal story about transport",
        },
        {
            "is_sarcastic": 1,
            "headline": "mayor proud of broken bridge",
            "article_section": "Local",
            "description": "Bridge officials reply to critics",
        },
        {
            "is_sarcastic": 0,
            "headline": "team wins final match",
            "article_section": "Sports",
            "description": "Fans celebrate downtown",
        },
        {
            "is_sarcastic": 1,
            "headline": "scientists discover nap better than work",
            "article_section": "Science",
            "description": "Researchers report a very serious finding",
        },
        {
            "is_sarcastic": 0,
            "headline": "school adds new library wing",
            "article_section": "Education",
            "description": "Students get more space for study",
        },
        {
            "is_sarcastic": 1,
            "headline": "boss excited to schedule meeting about meetings",
            "article_section": "Business",
            "description": "Workers react with joy",
        },
        {
            "is_sarcastic": 0,
            "headline": "hospital opens new ward",
            "article_section": "Health",
            "description": "Doctors say the facility will help patients",
        },
        {
            "is_sarcastic": 1,
            "headline": "nation relieved to hear another long speech",
            "article_section": "Politics",
            "description": "Citizens follow the update closely",
        },
        {
            "is_sarcastic": 0,
            "headline": "park adds more trees",
            "article_section": "Environment",
            "description": "Residents welcome the change",
        },
    ]
    path = tmp_path / "toy.jsonl"
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


def write_config(tmp_path: Path, payload: dict) -> Path:
    config_path = tmp_path / "tuning.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


def clear_caches() -> None:
    tune.PREPARED_FRAME_CACHE.clear()
    tune.RECIPE_FRAME_CACHE.clear()


def test_load_tuning_spec_resolves_paths_and_storage(tmp_path):
    data_path = write_jsonl(tmp_path)
    config_path = write_config(
        tmp_path,
        {
            "storage": {"url": "sqlite:///study.db", "study_prefix": "demo"},
            "paths": {"data_path": data_path.name, "output_root": "runs_out"},
            "studies": {
                "nb_recipe": {
                    "model": "tfidf_nb",
                    "recipe": "headline_section",
                    "n_trials": 2,
                    "fixed": {"split": {"random_state": 7}},
                    "search_space": {"model": {"alpha": {"type": "logfloat", "low": 0.1, "high": 1.0}}},
                }
            },
        },
    )

    spec = tune.load_tuning_spec(config_path)

    assert spec.paths.data_path == str(data_path.resolve())
    assert spec.paths.output_root == str((tmp_path / "runs_out").resolve())
    assert spec.storage.url.startswith("sqlite:///")
    assert spec.storage.study_prefix == "demo"


def test_load_tuning_spec_rejects_split_search_space(tmp_path):
    data_path = write_jsonl(tmp_path)
    config_path = write_config(
        tmp_path,
        {
            "storage": {"url": "sqlite:///study.db"},
            "paths": {"data_path": str(data_path)},
            "studies": {
                "bad_study": {
                    "model": "tfidf_nb",
                    "recipe": "headline_section",
                    "n_trials": 1,
                    "search_space": {
                        "split": {"random_state": {"type": "int", "low": 1, "high": 5}},
                    },
                }
            },
        },
    )

    with pytest.raises(ValueError, match="cannot sample split parameters"):
        tune.load_tuning_spec(config_path)


def test_load_tuning_spec_rejects_wrong_model_parameter(tmp_path):
    data_path = write_jsonl(tmp_path)
    config_path = write_config(
        tmp_path,
        {
            "storage": {"url": "sqlite:///study.db"},
            "paths": {"data_path": str(data_path)},
            "studies": {
                "bad_nb": {
                    "model": "tfidf_nb",
                    "recipe": "headline_section",
                    "n_trials": 1,
                    "fixed": {"model": {"C": 1.0}},
                }
            },
        },
    )

    with pytest.raises(ValueError, match="unsupported fixed model keys"):
        tune.load_tuning_spec(config_path)


def test_resolve_trial_parameters_merges_fixed_and_sampled(tmp_path):
    data_path = write_jsonl(tmp_path)
    config_path = write_config(
        tmp_path,
        {
            "storage": {"url": "sqlite:///study.db"},
            "paths": {"data_path": str(data_path)},
            "studies": {
                "nb_recipe": {
                    "model": "tfidf_nb",
                    "recipe": "headline_section",
                    "n_trials": 1,
                    "fixed": {
                        "split": {"random_state": 42},
                        "model": {"max_features": 500},
                    },
                    "search_space": {
                        "model": {
                            "alpha": {"type": "logfloat", "low": 0.1, "high": 1.0},
                            "ngram_range": {"type": "categorical", "choices": [[1, 1], [1, 2]]},
                        }
                    },
                }
            },
        },
    )
    spec = tune.load_tuning_spec(config_path)
    study = spec.studies["nb_recipe"]

    resolved, sampled = tune.resolve_trial_parameters(
        study,
        FixedTrial({"model.alpha": 0.25, "model.ngram_range": "1,2"}),
    )

    assert resolved["split"]["random_state"] == 42
    assert resolved["model"]["alpha"] == 0.25
    assert resolved["model"]["ngram_range"] == (1, 2)
    assert sampled["model.alpha"] == 0.25


def test_build_trial_configs_for_both_families(tmp_path):
    data_path = write_jsonl(tmp_path)
    config_path = write_config(
        tmp_path,
        {
            "storage": {"url": "sqlite:///study.db"},
            "paths": {"data_path": str(data_path)},
            "studies": {
                "nb_recipe": {
                    "model": "tfidf_nb",
                    "recipe": "headline_section",
                    "n_trials": 1,
                    "fixed": {"model": {"alpha": 0.5}},
                },
                "roberta_recipe": {
                    "model": "roberta",
                    "recipe": "headline_only",
                    "n_trials": 1,
                    "fixed": {
                        "run": {"batch_size": 4, "device": "cpu", "seed": 11},
                        "model": {"pretrained_name": "roberta-base", "dropout": 0.2},
                    },
                },
            },
        },
    )
    spec = tune.load_tuning_spec(config_path)

    nb_split, nb_cfg = tune.build_trial_configs(
        spec.studies["nb_recipe"],
        spec,
        spec.studies["nb_recipe"].fixed,
        tmp_path / "trial_nb",
    )
    rb_split, rb_cfg = tune.build_trial_configs(
        spec.studies["roberta_recipe"],
        spec,
        spec.studies["roberta_recipe"].fixed,
        tmp_path / "trial_rb",
    )

    assert isinstance(nb_split, v2.SplitConfig)
    assert nb_cfg.model_params["alpha"] == 0.5
    assert isinstance(rb_split, v2.SplitConfig)
    assert rb_cfg.pretrained_name == "roberta-base"
    assert rb_cfg.batch_size == 4
    assert rb_cfg.dropout == 0.2


def test_roberta_tuning_accepts_headline_only_and_dropout(tmp_path):
    data_path = write_jsonl(tmp_path)
    config_path = write_config(
        tmp_path,
        {
            "storage": {"url": "sqlite:///study.db"},
            "paths": {"data_path": str(data_path)},
            "studies": {
                "rb_recipe": {
                    "model": "roberta",
                    "recipe": "headline_only",
                    "n_trials": 1,
                    "search_space": {
                        "model": {
                            "dropout": {"type": "float", "low": 0.1, "high": 0.3},
                        }
                    },
                }
            },
        },
    )

    spec = tune.load_tuning_spec(config_path)

    assert spec.studies["rb_recipe"].recipe == "headline_only"
    assert "dropout" in spec.studies["rb_recipe"].search_space["model"]


def test_build_pruner_uses_transformer_only(tmp_path):
    data_path = write_jsonl(tmp_path)
    config_path = write_config(
        tmp_path,
        {
            "storage": {"url": "sqlite:///study.db"},
            "paths": {"data_path": str(data_path)},
            "studies": {
                "nb_recipe": {"model": "tfidf_nb", "recipe": "headline_section", "n_trials": 1},
                "rb_recipe": {"model": "roberta", "recipe": "headline_section", "n_trials": 1},
            },
        },
    )
    spec = tune.load_tuning_spec(config_path)

    assert isinstance(tune.build_pruner(spec.studies["nb_recipe"]), NopPruner)
    assert isinstance(tune.build_pruner(spec.studies["rb_recipe"]), MedianPruner)


def test_submit_study_dry_run_writes_slurm_script(tmp_path):
    data_path = write_jsonl(tmp_path)
    config_path = write_config(
        tmp_path,
        {
            "storage": {"url": "sqlite:///study.db"},
            "paths": {"data_path": str(data_path), "output_root": "runs_here"},
            "slurm": {
                "default": {"partition": "cpu", "time": "01:00:00", "cpus_per_task": 2, "mem": "8G"},
                "transformer": {"gpus": "a100-80", "workers": 3},
            },
            "studies": {
                "rb_recipe": {
                    "model": "roberta",
                    "recipe": "headline_section_description",
                    "n_trials": 1,
                }
            },
        },
    )
    spec = tune.load_tuning_spec(config_path)
    result = tune.submit_study(spec, spec.studies["rb_recipe"], dry_run=True)
    script_path = Path(result["script_path"])
    script = script_path.read_text(encoding="utf-8")

    assert result["submitted"] is False
    assert script.splitlines()[0] == "#!/bin/bash"
    assert script.splitlines()[1].startswith("#SBATCH --job-name=")
    assert "#SBATCH --partition=cpu" in script
    assert "#SBATCH --gpus=a100-80" in script
    assert "#SBATCH --gres=" not in script
    assert "#SBATCH --array=0-2" in script
    assert script.index("set -euo pipefail") > script.index("#SBATCH --array=0-2")
    assert 'source .venv/bin/activate' in script
    assert 'torch.cuda.is_available()' in script
    assert "--study rb_recipe" in script


def test_worker_runs_and_resumes_for_classical_study(tmp_path, monkeypatch):
    clear_caches()
    monkeypatch.setattr(v2, "_load_preprocessors", fake_load_preprocessors)
    data_path = write_jsonl(tmp_path)
    config_path = write_config(
        tmp_path,
        {
            "storage": {"url": "sqlite:///optuna_runs.db", "study_prefix": "smoke"},
            "paths": {"data_path": str(data_path), "output_root": "runs_out"},
            "studies": {
                "nb_recipe": {
                    "model": "tfidf_nb",
                    "recipe": "headline_section",
                    "n_trials": 2,
                    "max_rows": 10,
                    "fixed": {
                        "split": {
                            "train_size": 0.6,
                            "val_size": 0.2,
                            "test_size": 0.2,
                            "random_state": 42,
                        },
                        "run": {"seed": 42},
                        "model": {"min_df": 1, "max_df": 1.0},
                    },
                    "search_space": {
                        "model": {
                            "alpha": {"type": "categorical", "choices": [0.5, 1.0]},
                            "ngram_range": {"type": "categorical", "choices": [[1, 1], [1, 2]]},
                        }
                    },
                }
            },
        },
    )
    spec = tune.load_tuning_spec(config_path)
    study_spec = spec.studies["nb_recipe"]

    summary_first = tune.run_worker(spec, study_spec)
    summary_second = tune.run_worker(spec, study_spec)
    out_dir = Path(spec.paths.output_root) / study_spec.name

    assert summary_first["objective_metric"] == "val.macro_f1"
    assert summary_second["n_trials_finished"] == 2
    assert (out_dir / "trials.csv").exists()
    assert (out_dir / "leaderboard.csv").exists()
    assert (out_dir / "best_trial.json").exists()
    assert (out_dir / "study_summary.json").exists()
    assert (out_dir / "trial_00000").exists() is False

    trial_root = out_dir / "trials"
    status_files = sorted(trial_root.glob("trial_*/status.json"))
    params_files = sorted(trial_root.glob("trial_*/params.json"))
    metrics_files = sorted(trial_root.glob("trial_*/metrics_snapshot.json"))
    assert len(status_files) == 2
    assert len(params_files) == 2
    assert len(metrics_files) == 2

    study = optuna.load_study(study_name="smoke.nb_recipe", storage=spec.storage.url)
    assert len([trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]) == 2

    aggregate = tune.summarize_selected_studies(spec, [study_spec])
    assert len(aggregate) == 1
    assert (Path(spec.paths.output_root) / "studies_summary.json").exists()
    assert (Path(spec.paths.output_root) / "studies_summary.csv").exists()
