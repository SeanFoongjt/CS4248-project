from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
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

def small_df() -> pd.DataFrame:
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
            "description": None,
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
    df = pd.DataFrame(rows)
    df["row_id"] = range(len(df))
    return df


class DummyTextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.items = []
        for text, label in zip(texts, labels):
            token = float(len(text.split()))
            self.items.append(
                {
                    "input_ids": torch.tensor([token], dtype=torch.float32),
                    "attention_mask": torch.tensor([1.0], dtype=torch.float32),
                    "labels": torch.tensor(label, dtype=torch.long),
                }
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class DummyTokenizer:
    def save_pretrained(self, out_dir):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "tokenizer.json").write_text("{}", encoding="utf-8")


class DummyTransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 2)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask=None, labels=None):
        logits = self.linear(input_ids.float())
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return type("DummyOutput", (), {"loss": loss, "logits": logits})


@dataclass
class DummyCfg:
    pretrained_name: str = "dummy-roberta"
    max_length: int = 128
    num_labels: int = 2
    dropout: float = 0.1


class DummyTransformerWrapper:
    def __init__(self):
        self.cfg = DummyCfg()
        self.model = DummyTransformerModel()
        self.tokenizer = DummyTokenizer()
        self.collator = self._collate

    def make_dataset(self, texts, labels=None):
        return DummyTextDataset(texts, labels)

    def _collate(self, batch):
        return {
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
            "labels": torch.stack([item["labels"] for item in batch]),
        }

    def forward_batch(self, batch):
        return self.model(**batch)

    def to(self, device):
        self.model.to(device)
        return self

    def save_pretrained(self, out_dir):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), Path(out_dir) / "model_state.pt")

def test_prepare_variant2_frame_builds_needed_columns(monkeypatch):
    monkeypatch.setattr(v2, "_load_preprocessors", fake_load_preprocessors)
    df = v2.prepare_variant2_frame(small_df())
    needed = {
        "headline_bow",
        "headline_tf",
        "section_raw",
        "section_bow",
        "description_raw",
        "description_bow",
        "label",
    }

    assert needed.issubset(df.columns)
    assert "the onion" not in df.loc[0, "description_raw"].lower()

def test_recipe_registry_matches_supported_variants():
    assert [recipe.name for recipe in v2.VARIANT2_RECIPES] == [
        "headline_only",
        "headline_section",
        "headline_section_description",
    ]


def test_headline_only_recipe_keeps_rows_missing_context(monkeypatch):
    monkeypatch.setattr(v2, "_load_preprocessors", fake_load_preprocessors)
    df = v2.prepare_variant2_frame(small_df())
    recipe = v2.ContextRecipe(name="headline_only")
    sub = v2.subset_for_recipe(df, recipe)

    assert len(sub) == len(df)


def test_recipe_subset_drops_rows_missing_required_context(monkeypatch):
    monkeypatch.setattr(v2, "_load_preprocessors", fake_load_preprocessors)
    df = v2.prepare_variant2_frame(small_df())
    recipe = v2.ContextRecipe(name="headline_section_description", needs_section=True, needs_description=True)
    sub = v2.subset_for_recipe(df, recipe)

    assert len(sub) == len(df) - 1

def test_build_recipe_text_contains_expected_parts(monkeypatch):
    monkeypatch.setattr(v2, "_load_preprocessors", fake_load_preprocessors)
    df = v2.prepare_variant2_frame(small_df())
    recipe = v2.ContextRecipe(name="headline_section_description", needs_section=True, needs_description=True)
    row = v2.subset_for_recipe(df, recipe).iloc[0]
    txt = v2.build_recipe_text(row, recipe, family="classical")

    assert "headline:" in txt
    assert "section:" in txt
    assert "description:" in txt

def test_split_frame_keeps_all_rows_and_all_labels(monkeypatch):
    monkeypatch.setattr(v2, "_load_preprocessors", fake_load_preprocessors)
    df = v2.prepare_variant2_frame(small_df())
    recipe = v2.ContextRecipe(name="headline_section", needs_section=True)
    frame = v2.build_recipe_frame(df, recipe, family="classical")
    split_map = v2.split_frame(frame, v2.SplitConfig())
    total = sum(len(x) for x in split_map.values())

    assert total == len(frame)
    assert set(split_map["train"]["label"].tolist()) == {0, 1}

    merged = pd.concat(list(split_map.values()), ignore_index=True)

    assert sorted(merged["row_id"].tolist()) == sorted(frame["row_id"].tolist())
    assert len(split_map["train"]) == 6
    assert len(split_map["val"]) == 2
    assert len(split_map["test"]) == 2

def test_classical_recipe_runs_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(v2, "_load_preprocessors", fake_load_preprocessors)
    df = v2.prepare_variant2_frame(small_df())
    recipe = v2.ContextRecipe(name="headline_section", needs_section=True)
    frame = v2.build_recipe_frame(df, recipe, family="classical")
    metrics = v2.run_classical_recipe(
        frame,
        recipe,
        cfg=v2.ClassicalRunConfig(model_name="tfidf_nb", output_dir=str(tmp_path)),
        split_cfg=v2.SplitConfig(),
    )

    assert metrics["model"] == "tfidf_nb"
    assert "val" in metrics and "test" in metrics
    assert (tmp_path / "tfidf_nb" / "headline_section" / "metrics.json").exists()


def test_transformer_recipe_records_collaborator_metrics_and_clips_gradients(tmp_path, monkeypatch):
    monkeypatch.setattr(v2, "_load_preprocessors", fake_load_preprocessors)
    monkeypatch.setattr(v2, "build_transformer_wrapper", lambda *args, **kwargs: DummyTransformerWrapper())
    clip_calls = {"count": 0}

    def fake_clip(params, max_norm):
        clip_calls["count"] += 1
        return torch.tensor(0.0)

    monkeypatch.setattr(v2.nn.utils, "clip_grad_norm_", fake_clip)
    df = v2.prepare_variant2_frame(small_df())
    recipe = v2.ContextRecipe(name="headline_only")
    frame = v2.build_recipe_frame(df, recipe, family="transformer")
    metrics = v2.run_transformer_recipe(
        frame,
        recipe,
        cfg=v2.default_transformer_run_config(model_name="roberta", output_dir=str(tmp_path), device="cpu", epochs=2),
        split_cfg=v2.SplitConfig(),
    )

    assert metrics["model"] == "roberta"
    assert metrics["train_loss"] is not None
    assert metrics["val_loss"] is not None
    assert metrics["val_accuracy"] is not None
    assert "macro_f1" in metrics["val"]
    assert "macro_f1" in metrics["test"]
    assert "val_accuracy" in metrics["history"][0]
    assert clip_calls["count"] > 0
    assert (tmp_path / "roberta" / "headline_only" / "checkpoint" / "model_state.pt").exists()
    assert (tmp_path / "roberta" / "headline_only" / "checkpoint" / "tokenizer.json").exists()


def test_load_jsonl_round_trip(tmp_path):
    path = Path(tmp_path) / "toy.jsonl"
    rows = [
        {"is_sarcastic": 0, "headline": "a", "article_section": "news", "description": "x"},
        {"is_sarcastic": 1, "headline": "b", "article_section": "news", "description": "y"},
    ]
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    df = v2.load_jsonl(path)

    assert len(df) == 2
    assert "row_id" in df.columns
