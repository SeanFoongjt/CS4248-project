from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import optuna
import pandas as pd
import yaml
from optuna.pruners import MedianPruner, NopPruner
from optuna.samplers import TPESampler
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState

from training import variant2_context as v2

VALID_RECIPES = {recipe.name for recipe in v2.VARIANT2_RECIPES}
CLASSICAL_MODELS = {"tfidf_nb", "tfidf_lr"}
TRANSFORMER_MODELS = {"distilbert", "roberta"}
SEARCH_GROUPS = {"run", "model"}
FIXED_GROUPS = {"split", "run", "model"}
SLURM_KEYS = {"partition", "time", "cpus_per_task", "mem", "gres", "workers", "job_name_prefix"}
SPLIT_KEYS = {"train_size", "val_size", "test_size", "random_state"}
CLASSICAL_MODEL_KEYS = {
    "tfidf_nb": {"ngram_range", "min_df", "max_df", "max_features", "sublinear_tf", "norm", "alpha", "lowercase"},
    "tfidf_lr": {
        "ngram_range",
        "min_df",
        "max_df",
        "max_features",
        "sublinear_tf",
        "norm",
        "C",
        "max_iter",
        "solver",
        "class_weight",
        "random_state",
        "lowercase",
    },
}
TRANSFORMER_RUN_KEYS = {
    "batch_size",
    "eval_batch_size",
    "lr",
    "weight_decay",
    "epochs",
    "warmup_ratio",
    "max_length",
    "device",
    "seed",
}
TRANSFORMER_MODEL_KEYS = {
    "distilbert": {"pretrained_name"},
    "roberta": {"pretrained_name", "dropout"},
}
FIXED_ONLY_RUN_KEYS = {"device"}
TRIAL_COUNT_STATES = (TrialState.COMPLETE, TrialState.FAIL, TrialState.PRUNED)
PREPARED_FRAME_CACHE: dict[tuple[str, Optional[int]], pd.DataFrame] = {}
RECIPE_FRAME_CACHE: dict[tuple[str, Optional[int], str, str], pd.DataFrame] = {}


@dataclass
class StorageConfig:
    url: str
    study_prefix: str = "variant2"


@dataclass
class PathsConfig:
    data_path: str
    output_root: str = "runs/variant2_tuning"


@dataclass
class SlurmProfile:
    partition: Optional[str] = None
    time: str = "04:00:00"
    cpus_per_task: int = 4
    mem: str = "16G"
    gres: Optional[str] = None
    workers: int = 1
    job_name_prefix: str = "variant2-tune"


@dataclass
class StudySpec:
    name: str
    model: str
    recipe: str
    n_trials: int
    max_rows: Optional[int] = None
    fixed: dict[str, dict[str, Any]] = field(default_factory=dict)
    search_space: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)
    slurm: dict[str, Any] = field(default_factory=dict)

    @property
    def family(self) -> str:
        if self.model in CLASSICAL_MODELS:
            return "classical"
        return "transformer"


@dataclass
class TuningSpec:
    config_path: str
    storage: StorageConfig
    paths: PathsConfig
    slurm_default: SlurmProfile
    slurm_classical: SlurmProfile
    slurm_transformer: SlurmProfile
    studies: dict[str, StudySpec]


def _ensure_dict(value: Any, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _coerce_value(key: str, value: Any) -> Any:
    if key == "ngram_range":
        if isinstance(value, str):
            parts = [part.strip() for part in value.split(",")]
            if len(parts) == 2:
                return (int(parts[0]), int(parts[1]))
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return (int(value[0]), int(value[1]))
        raise ValueError("ngram_range must be a two-item list or tuple")
    return value


def _resolve_local_path(base_dir: Path, raw_path: str) -> str:
    path = Path(raw_path)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


def _resolve_storage_url(base_dir: Path, raw_url: str) -> str:
    if raw_url.startswith("sqlite:///") and not raw_url.startswith("sqlite:////"):
        rel = raw_url[len("sqlite:///") :]
        path = Path(rel)
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        return f"sqlite:///{path.as_posix()}"
    return raw_url


def _normalize_group_map(
    raw: Any,
    allowed_groups: set[str],
    label: str,
    *,
    coerce_values: bool,
) -> dict[str, dict[str, Any]]:
    raw_dict = _ensure_dict(raw, label)
    unknown_groups = set(raw_dict) - allowed_groups
    if unknown_groups:
        raise ValueError(f"{label} contains unsupported groups: {sorted(unknown_groups)}")
    out = {group: _ensure_dict(raw_dict.get(group), f"{label}.{group}") for group in allowed_groups}
    if coerce_values:
        for group_name, values in out.items():
            out[group_name] = {key: _coerce_value(key, value) for key, value in values.items()}
    return out


def _allowed_run_keys(study: StudySpec) -> set[str]:
    if study.family == "transformer":
        return set(TRANSFORMER_RUN_KEYS)
    return {"seed"}


def _allowed_model_keys(study: StudySpec) -> set[str]:
    if study.family == "classical":
        return set(CLASSICAL_MODEL_KEYS[study.model])
    return set(TRANSFORMER_MODEL_KEYS[study.model])


def _validate_fixed_params(study: StudySpec) -> None:
    split_keys = set(study.fixed["split"])
    if not split_keys.issubset(SPLIT_KEYS):
        raise ValueError(f"Study {study.name} has unsupported fixed split keys: {sorted(split_keys - SPLIT_KEYS)}")
    run_keys = set(study.fixed["run"])
    allowed_run = _allowed_run_keys(study)
    if not run_keys.issubset(allowed_run):
        raise ValueError(f"Study {study.name} has unsupported fixed run keys: {sorted(run_keys - allowed_run)}")
    model_keys = set(study.fixed["model"])
    allowed_model = _allowed_model_keys(study)
    if not model_keys.issubset(allowed_model):
        raise ValueError(f"Study {study.name} has unsupported fixed model keys: {sorted(model_keys - allowed_model)}")
    slurm_keys = set(study.slurm)
    if not slurm_keys.issubset(SLURM_KEYS):
        raise ValueError(f"Study {study.name} has unsupported slurm keys: {sorted(slurm_keys - SLURM_KEYS)}")


def _validate_search_space(study: StudySpec) -> None:
    if study.search_space.get("split"):
        raise ValueError(f"Study {study.name} cannot sample split parameters in v1")
    for key in study.search_space["run"]:
        if key not in _allowed_run_keys(study):
            raise ValueError(f"Study {study.name} has unsupported run search parameter: {key}")
        if key in FIXED_ONLY_RUN_KEYS:
            raise ValueError(f"Study {study.name} cannot sample fixed-only run parameter: {key}")
    for key in study.search_space["model"]:
        if key not in _allowed_model_keys(study):
            raise ValueError(f"Study {study.name} has unsupported model search parameter: {key}")
    for group_name in SEARCH_GROUPS:
        for key, spec in study.search_space[group_name].items():
            _validate_search_param_spec(study.name, group_name, key, spec)


def _validate_search_param_spec(study_name: str, group: str, key: str, spec: Any) -> None:
    if not isinstance(spec, dict):
        raise ValueError(f"Study {study_name} search spec for {group}.{key} must be a mapping")
    kind = spec.get("type")
    if kind not in {"categorical", "int", "float", "logfloat"}:
        raise ValueError(f"Study {study_name} search spec for {group}.{key} has unsupported type: {kind}")
    if kind == "categorical":
        choices = spec.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"Study {study_name} categorical spec for {group}.{key} must define non-empty choices")
        return
    if "low" not in spec or "high" not in spec:
        raise ValueError(f"Study {study_name} search spec for {group}.{key} must define low and high")
    if spec["low"] > spec["high"]:
        raise ValueError(f"Study {study_name} search spec for {group}.{key} has low > high")
    if kind == "int":
        step = spec.get("step", 1)
        if int(step) <= 0:
            raise ValueError(f"Study {study_name} int spec for {group}.{key} must use step >= 1")
    if kind == "float" and "step" in spec and float(spec["step"]) <= 0:
        raise ValueError(f"Study {study_name} float spec for {group}.{key} must use step > 0")


def _load_studies(raw_studies: Any) -> dict[str, StudySpec]:
    studies_dict = _ensure_dict(raw_studies, "studies")
    if not studies_dict:
        raise ValueError("studies must define at least one study")
    studies: dict[str, StudySpec] = {}
    for study_name, raw in studies_dict.items():
        if not isinstance(raw, dict):
            raise ValueError(f"Study {study_name} must be a mapping")
        model = raw.get("model")
        recipe = raw.get("recipe")
        n_trials = raw.get("n_trials")
        if model not in CLASSICAL_MODELS | TRANSFORMER_MODELS:
            raise ValueError(f"Study {study_name} has unsupported model: {model}")
        if recipe not in VALID_RECIPES:
            raise ValueError(f"Study {study_name} has unsupported recipe: {recipe}")
        if not isinstance(n_trials, int) or n_trials <= 0:
            raise ValueError(f"Study {study_name} must define a positive integer n_trials")
        max_rows = raw.get("max_rows")
        if max_rows is not None and (not isinstance(max_rows, int) or max_rows <= 0):
            raise ValueError(f"Study {study_name} max_rows must be a positive integer when provided")
        fixed = _normalize_group_map(
            raw.get("fixed"),
            FIXED_GROUPS,
            f"studies.{study_name}.fixed",
            coerce_values=True,
        )
        search_space = _normalize_group_map(
            raw.get("search_space"),
            FIXED_GROUPS,
            f"studies.{study_name}.search_space",
            coerce_values=False,
        )
        slurm = _ensure_dict(raw.get("slurm"), f"studies.{study_name}.slurm")
        study = StudySpec(
            name=study_name,
            model=model,
            recipe=recipe,
            n_trials=n_trials,
            max_rows=max_rows,
            fixed=fixed,
            search_space=search_space,
            slurm=slurm,
        )
        _validate_fixed_params(study)
        _validate_search_space(study)
        studies[study_name] = study
    return studies


def load_tuning_spec(path: str | Path) -> TuningSpec:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("The tuning config must be a YAML mapping")
    base_dir = path.parent
    storage_raw = _ensure_dict(raw.get("storage"), "storage")
    if "url" not in storage_raw:
        raise ValueError("storage.url is required")
    paths_raw = _ensure_dict(raw.get("paths"), "paths")
    if "data_path" not in paths_raw:
        raise ValueError("paths.data_path is required")
    storage = StorageConfig(
        url=_resolve_storage_url(base_dir, str(storage_raw["url"])),
        study_prefix=str(storage_raw.get("study_prefix", "variant2")),
    )
    paths_cfg = PathsConfig(
        data_path=_resolve_local_path(base_dir, str(paths_raw["data_path"])),
        output_root=_resolve_local_path(base_dir, str(paths_raw.get("output_root", "runs/variant2_tuning"))),
    )
    slurm_raw = _ensure_dict(raw.get("slurm"), "slurm")
    slurm_default = SlurmProfile(**{**asdict(SlurmProfile()), **_ensure_dict(slurm_raw.get("default"), "slurm.default")})
    slurm_classical = SlurmProfile(
        **{**asdict(slurm_default), **_ensure_dict(slurm_raw.get("classical"), "slurm.classical")}
    )
    slurm_transformer = SlurmProfile(
        **{
            **asdict(slurm_default),
            "gres": "gpu:1",
            **_ensure_dict(slurm_raw.get("transformer"), "slurm.transformer"),
        }
    )
    studies = _load_studies(raw.get("studies"))
    return TuningSpec(
        config_path=str(path),
        storage=storage,
        paths=paths_cfg,
        slurm_default=slurm_default,
        slurm_classical=slurm_classical,
        slurm_transformer=slurm_transformer,
        studies=studies,
    )


def _study_output_dir(spec: TuningSpec, study: StudySpec) -> Path:
    return Path(spec.paths.output_root) / study.name


def _trial_output_dir(spec: TuningSpec, study: StudySpec, trial_number: int) -> Path:
    return _study_output_dir(spec, study) / "trials" / f"trial_{trial_number:05d}"


def _full_study_name(spec: TuningSpec, study: StudySpec) -> str:
    prefix = spec.storage.study_prefix.strip()
    return f"{prefix}.{study.name}" if prefix else study.name


def _sample_param(trial: optuna.Trial, param_name: str, spec: dict[str, Any]) -> Any:
    kind = spec["type"]
    if kind == "categorical":
        choices = spec["choices"]
        if param_name.endswith("ngram_range"):
            choices = [
                f"{choice[0]},{choice[1]}" if isinstance(choice, (list, tuple)) and len(choice) == 2 else choice
                for choice in choices
            ]
        return trial.suggest_categorical(param_name, choices)
    if kind == "int":
        return trial.suggest_int(param_name, int(spec["low"]), int(spec["high"]), step=int(spec.get("step", 1)))
    if kind == "float":
        step = spec.get("step")
        kwargs = {"step": float(step)} if step is not None else {}
        return trial.suggest_float(param_name, float(spec["low"]), float(spec["high"]), **kwargs)
    if kind == "logfloat":
        return trial.suggest_float(param_name, float(spec["low"]), float(spec["high"]), log=True)
    raise ValueError(f"Unsupported search type: {kind}")


def resolve_trial_parameters(study: StudySpec, trial: Optional[optuna.Trial] = None) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    resolved = {group: dict(values) for group, values in study.fixed.items()}
    sampled_params: dict[str, Any] = {}
    for group_name in SEARCH_GROUPS:
        for key, param_spec in study.search_space[group_name].items():
            if trial is None:
                raise ValueError("A trial instance is required when the study has sampled parameters")
            param_name = f"{group_name}.{key}"
            value = _sample_param(trial, param_name, param_spec)
            value = _coerce_value(key, value)
            resolved[group_name][key] = value
            sampled_params[param_name] = value
    return resolved, sampled_params


def build_trial_configs(
    study: StudySpec,
    spec: TuningSpec,
    resolved: dict[str, dict[str, Any]],
    trial_dir: Path,
) -> tuple[v2.SplitConfig, v2.ClassicalRunConfig | v2.TransformerRunConfig]:
    split_cfg = v2.SplitConfig(**resolved["split"])
    if study.family == "classical":
        run_cfg = v2.ClassicalRunConfig(
            model_name=study.model,
            output_dir=str(trial_dir),
            model_params=dict(resolved["model"]),
        )
        return split_cfg, run_cfg
    run_kwargs = dict(resolved["run"])
    model_kwargs = dict(resolved["model"])
    run_cfg = v2.default_transformer_run_config(
        model_name=study.model,
        output_dir=str(trial_dir),
        pretrained_name=model_kwargs.get("pretrained_name"),
        dropout=model_kwargs.get("dropout"),
        **run_kwargs,
    )
    return split_cfg, run_cfg


def _prepared_frame(spec: TuningSpec, study: StudySpec) -> pd.DataFrame:
    key = (spec.paths.data_path, study.max_rows)
    if key not in PREPARED_FRAME_CACHE:
        PREPARED_FRAME_CACHE[key] = v2.load_variant2_data(spec.paths.data_path, max_rows=study.max_rows)
    return PREPARED_FRAME_CACHE[key]


def get_cached_recipe_frame(spec: TuningSpec, study: StudySpec) -> pd.DataFrame:
    key = (spec.paths.data_path, study.max_rows, study.recipe, study.family)
    if key not in RECIPE_FRAME_CACHE:
        recipe = v2._recipe_by_name(study.recipe)
        family = "classical" if study.family == "classical" else "transformer"
        RECIPE_FRAME_CACHE[key] = v2.build_recipe_frame(_prepared_frame(spec, study), recipe, family=family)
    return RECIPE_FRAME_CACHE[key].copy()


def build_pruner(study: StudySpec):
    if study.family == "transformer":
        return MedianPruner()
    return NopPruner()


def create_optuna_study(spec: TuningSpec, study: StudySpec):
    sampler_seed = study.fixed["run"].get("seed", 42)
    return optuna.create_study(
        study_name=_full_study_name(spec, study),
        storage=spec.storage.url,
        direction="maximize",
        load_if_exists=True,
        sampler=TPESampler(seed=sampler_seed),
        pruner=build_pruner(study),
    )


def _count_finished_trials(study) -> int:
    return sum(1 for trial in study.trials if trial.state in TRIAL_COUNT_STATES)


def _trial_records(study) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for trial in study.trials:
        row = {
            "number": trial.number,
            "state": trial.state.name,
            "value": trial.value,
            "datetime_start": trial.datetime_start.isoformat() if trial.datetime_start else None,
            "datetime_complete": trial.datetime_complete.isoformat() if trial.datetime_complete else None,
        }
        for key, value in trial.params.items():
            row[f"param.{key}"] = value
        for key, value in trial.user_attrs.items():
            row[f"user_attr.{key}"] = value
        records.append(row)
    return records


def _study_summary(study, study_spec: StudySpec) -> dict[str, Any]:
    summary = {
        "study_name": study_spec.name,
        "model": study_spec.model,
        "recipe": study_spec.recipe,
        "objective_metric": "val.macro_f1",
        "n_trials_target": study_spec.n_trials,
        "n_trials_finished": _count_finished_trials(study),
    }
    completed_trials = [trial for trial in study.trials if trial.state == TrialState.COMPLETE and trial.value is not None]
    if not completed_trials:
        return {
            **summary,
            "best_trial_id": None,
            "best_params": None,
            "best_validation_metric": None,
            "corresponding_test_metric": None,
            "artifact_path": None,
        }
    best_trial = study.best_trial
    return {
        **summary,
        "best_trial_id": best_trial.number,
        "best_params": best_trial.params,
        "best_validation_metric": best_trial.user_attrs.get("val_macro_f1", best_trial.value),
        "corresponding_test_metric": best_trial.user_attrs.get("test_macro_f1"),
        "artifact_path": best_trial.user_attrs.get("artifact_path"),
    }


def _study_export_dict(spec: TuningSpec, study_spec: StudySpec) -> dict[str, Any]:
    return {
        "storage": {"url": spec.storage.url, "study_prefix": spec.storage.study_prefix},
        "paths": {"data_path": spec.paths.data_path, "output_root": spec.paths.output_root},
        "study": {
            "name": study_spec.name,
            "model": study_spec.model,
            "recipe": study_spec.recipe,
            "n_trials": study_spec.n_trials,
            "max_rows": study_spec.max_rows,
            "fixed": study_spec.fixed,
            "search_space": study_spec.search_space,
            "slurm": study_spec.slurm,
        },
    }


def export_study_artifacts(spec: TuningSpec, study_spec: StudySpec, study) -> dict[str, Any]:
    out_dir = _study_output_dir(spec, study_spec)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "study_config.yaml").write_text(
        yaml.safe_dump(_study_export_dict(spec, study_spec), sort_keys=False),
        encoding="utf-8",
    )
    records = _trial_records(study)
    df = pd.DataFrame(records)
    if not df.empty:
        df.to_csv(out_dir / "trials.csv", index=False)
        leaderboard = df[df["state"] == "COMPLETE"].sort_values("value", ascending=False)
        leaderboard.to_csv(out_dir / "leaderboard.csv", index=False)
    else:
        pd.DataFrame().to_csv(out_dir / "trials.csv", index=False)
        pd.DataFrame().to_csv(out_dir / "leaderboard.csv", index=False)
    summary = _study_summary(study, study_spec)
    v2.save_json(out_dir / "study_summary.json", summary)
    v2.save_json(
        out_dir / "study_metadata.json",
        {
            "study_name": _full_study_name(spec, study_spec),
            "model": study_spec.model,
            "recipe": study_spec.recipe,
            "objective_metric": "val.macro_f1",
            "storage_url": spec.storage.url,
            "direction": "maximize",
            "n_trials_finished": _count_finished_trials(study),
        },
    )
    v2.save_json(out_dir / "best_trial.json", summary)
    return summary


def summarize_selected_studies(spec: TuningSpec, studies: list[StudySpec]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for study_spec in studies:
        study = optuna.load_study(study_name=_full_study_name(spec, study_spec), storage=spec.storage.url)
        summaries.append(export_study_artifacts(spec, study_spec, study))
    out_root = Path(spec.paths.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    v2.save_json(out_root / "studies_summary.json", {"studies": summaries})
    pd.DataFrame(summaries).to_csv(out_root / "studies_summary.csv", index=False)
    return summaries


def run_trial(spec: TuningSpec, study_spec: StudySpec, trial: optuna.Trial) -> float:
    trial_dir = _trial_output_dir(spec, study_spec, trial.number)
    trial_dir.mkdir(parents=True, exist_ok=True)
    resolved, sampled_params = resolve_trial_parameters(study_spec, trial)
    recipe_frame = get_cached_recipe_frame(spec, study_spec)
    split_cfg, run_cfg = build_trial_configs(study_spec, spec, resolved, trial_dir)
    v2.save_json(
        trial_dir / "params.json",
        {
            "sampled_params": sampled_params,
            "resolved": resolved,
        },
    )
    trial.set_user_attr("artifact_path", str(trial_dir))
    trial.set_user_attr("model", study_spec.model)
    trial.set_user_attr("recipe", study_spec.recipe)
    trial.set_user_attr("objective_metric", "val.macro_f1")
    try:
        if study_spec.family == "classical":
            metrics = v2.run_classical_recipe(
                recipe_frame,
                v2._recipe_by_name(study_spec.recipe),
                cfg=run_cfg,
                split_cfg=split_cfg,
            )
        else:
            def epoch_callback(epoch: int, row: dict[str, Any]) -> None:
                trial.report(float(row["val_macro_f1"]), step=epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned(f"Pruned at epoch {epoch}")

            metrics = v2.run_transformer_recipe(
                recipe_frame,
                v2._recipe_by_name(study_spec.recipe),
                cfg=run_cfg,
                split_cfg=split_cfg,
                epoch_callback=epoch_callback,
            )
        objective_value = float(metrics["val"]["macro_f1"])
        metrics_snapshot = {
            "objective_value": objective_value,
            "val_metrics": metrics["val"],
            "test_metrics": metrics["test"],
            "metrics_path": str(Path(run_cfg.output_dir) / study_spec.model / study_spec.recipe / "metrics.json"),
        }
        v2.save_json(trial_dir / "metrics_snapshot.json", metrics_snapshot)
        v2.save_json(trial_dir / "status.json", {"status": "complete"})
        trial.set_user_attr("status", "complete")
        trial.set_user_attr("val_macro_f1", objective_value)
        trial.set_user_attr("test_macro_f1", float(metrics["test"]["macro_f1"]))
        trial.set_user_attr("metrics_snapshot_path", str(trial_dir / "metrics_snapshot.json"))
        return objective_value
    except optuna.TrialPruned as exc:
        v2.save_json(trial_dir / "status.json", {"status": "pruned", "reason": str(exc)})
        trial.set_user_attr("status", "pruned")
        raise
    except Exception as exc:
        v2.save_json(trial_dir / "status.json", {"status": "failed", "error": repr(exc)})
        trial.set_user_attr("status", "failed")
        trial.set_user_attr("error", repr(exc))
        raise


def run_worker(spec: TuningSpec, study_spec: StudySpec) -> dict[str, Any]:
    study = create_optuna_study(spec, study_spec)
    study.set_user_attr("model", study_spec.model)
    study.set_user_attr("recipe", study_spec.recipe)
    study.set_user_attr("objective_metric", "val.macro_f1")
    if _count_finished_trials(study) < study_spec.n_trials:
        max_trials_callback = MaxTrialsCallback(study_spec.n_trials, states=TRIAL_COUNT_STATES)
        study.optimize(
            lambda trial: run_trial(spec, study_spec, trial),
            n_trials=None,
            callbacks=[max_trials_callback],
            gc_after_trial=True,
        )
    return export_study_artifacts(spec, study_spec, study)


def _merge_slurm_profile(*profiles: SlurmProfile, overrides: Optional[dict[str, Any]] = None) -> SlurmProfile:
    merged = {}
    for profile in profiles:
        merged.update(asdict(profile))
    if overrides:
        merged.update(overrides)
    return SlurmProfile(**merged)


def resolve_slurm_profile(spec: TuningSpec, study_spec: StudySpec) -> SlurmProfile:
    family_profile = spec.slurm_classical if study_spec.family == "classical" else spec.slurm_transformer
    return _merge_slurm_profile(spec.slurm_default, family_profile, overrides=study_spec.slurm)


def render_slurm_script(spec: TuningSpec, study_spec: StudySpec) -> str:
    profile = resolve_slurm_profile(spec, study_spec)
    study_dir = _study_output_dir(spec, study_spec)
    slurm_dir = study_dir / "slurm"
    out_path = (slurm_dir / "%x_%A_%a.out").as_posix()
    err_path = (slurm_dir / "%x_%A_%a.err").as_posix()
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        f"#SBATCH --job-name={profile.job_name_prefix}-{study_spec.name}",
        f"#SBATCH --time={profile.time}",
        f"#SBATCH --cpus-per-task={profile.cpus_per_task}",
        f"#SBATCH --mem={profile.mem}",
        f"#SBATCH --output={out_path}",
        f"#SBATCH --error={err_path}",
    ]
    if profile.partition:
        lines.append(f"#SBATCH --partition={profile.partition}")
    if profile.gres:
        lines.append(f"#SBATCH --gres={profile.gres}")
    if profile.workers > 1:
        lines.append(f"#SBATCH --array=0-{profile.workers - 1}")
    lines.extend(
        [
            "",
            'cd "${SLURM_SUBMIT_DIR:-.}"',
            f"python -m training.tune_variant2 worker --config {shlex.quote(spec.config_path)} --study {shlex.quote(study_spec.name)}",
            "",
        ]
    )
    return "\n".join(lines)


def write_slurm_script(spec: TuningSpec, study_spec: StudySpec) -> Path:
    study_dir = _study_output_dir(spec, study_spec)
    script_dir = study_dir / "slurm"
    script_dir.mkdir(parents=True, exist_ok=True)
    script_path = script_dir / f"{study_spec.name}.sbatch"
    script_path.write_text(render_slurm_script(spec, study_spec), encoding="utf-8")
    return script_path


def submit_study(spec: TuningSpec, study_spec: StudySpec, dry_run: bool = False) -> dict[str, Any]:
    script_path = write_slurm_script(spec, study_spec)
    result = {"study": study_spec.name, "script_path": str(script_path)}
    if dry_run:
        result["submitted"] = False
        return result
    completed = subprocess.run(["sbatch", str(script_path)], check=True, capture_output=True, text=True)
    result["submitted"] = True
    result["stdout"] = completed.stdout.strip()
    return result


def _select_studies(spec: TuningSpec, study_name: Optional[str], run_all: bool) -> list[StudySpec]:
    if run_all:
        return list(spec.studies.values())
    if not study_name:
        raise ValueError("Provide --study or --all")
    if study_name not in spec.studies:
        raise ValueError(f"Unknown study: {study_name}")
    return [spec.studies[study_name]]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for Variant 2 experiments")
    sub = parser.add_subparsers(dest="command", required=True)

    worker = sub.add_parser("worker", help="Run one Optuna worker for a study")
    worker.add_argument("--config", required=True, help="Path to the YAML tuning config")
    worker.add_argument("--study", required=True, help="Study name to execute")

    submit = sub.add_parser("submit", help="Generate and submit Slurm jobs")
    submit.add_argument("--config", required=True, help="Path to the YAML tuning config")
    submit_group = submit.add_mutually_exclusive_group(required=True)
    submit_group.add_argument("--study", help="Study name to submit")
    submit_group.add_argument("--all", action="store_true", help="Submit all studies")
    submit.add_argument("--dry-run", action="store_true", help="Write sbatch scripts without submitting them")

    summarize = sub.add_parser("summarize", help="Export study summaries and trial tables")
    summarize.add_argument("--config", required=True, help="Path to the YAML tuning config")
    summarize_group = summarize.add_mutually_exclusive_group(required=True)
    summarize_group.add_argument("--study", help="Study name to summarize")
    summarize_group.add_argument("--all", action="store_true", help="Summarize all studies")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    spec = load_tuning_spec(args.config)
    if args.command == "worker":
        if args.study not in spec.studies:
            raise ValueError(f"Unknown study: {args.study}")
        run_worker(spec, spec.studies[args.study])
        return
    selected = _select_studies(spec, getattr(args, "study", None), getattr(args, "all", False))
    if args.command == "submit":
        results = [submit_study(spec, study_spec, dry_run=args.dry_run) for study_spec in selected]
        print(json.dumps({"jobs": results}, indent=2))
        return
    summaries = summarize_selected_studies(spec, selected)
    print(json.dumps({"studies": summaries}, indent=2))


if __name__ == "__main__":
    main()
