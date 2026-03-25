from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
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

def test_recipe_subset_drops_rows_missing_required_context(monkeypatch):
    monkeypatch.setattr(v2, "_load_preprocessors", fake_load_preprocessors)
    df = v2.prepare_variant2_frame(small_df())
    recipe = v2.ContextRecipe(name="headline_description", needs_description=True)
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