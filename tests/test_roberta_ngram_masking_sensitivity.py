from __future__ import annotations

import json
from pathlib import Path

from scripts import roberta_ngram_masking_sensitivity as masking


def test_resolve_checkpoint_dir_accepts_study_dir(tmp_path):
    study_dir = tmp_path / "roberta_headline_section"
    checkpoint_dir = study_dir / "best_model" / "roberta" / "headline_section" / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "model_state.pt").write_bytes(b"stub")
    (checkpoint_dir / "model_config.json").write_text("{}", encoding="utf-8")
    (study_dir / "best_model" / "materialization.json").write_text(
        json.dumps({"best_model_path": str(checkpoint_dir)}),
        encoding="utf-8",
    )

    resolved = masking.resolve_checkpoint_dir(study_dir)

    assert resolved == checkpoint_dir.resolve()


def test_mask_span_replaces_ngram_with_one_mask():
    tokens = "headline: boss excited to schedule meeting about meetings".split()

    masked = masking.mask_span(tokens, 4, 7, "<mask>")

    assert masked == "headline: boss excited to <mask> meetings"


def test_score_masked_spans_ranks_largest_probability_drop(monkeypatch):
    class DummyTokenizer:
        mask_token = "<mask>"

    class DummyWrapper:
        tokenizer = DummyTokenizer()

    def fake_predict_probabilities_batch(wrapper, texts, device, batch_size=64):
        outputs = []
        for text in texts:
            if text == "headline: boss excited meeting about meetings":
                outputs.append((1, [0.10, 0.90]))
            elif text == "headline: boss excited <mask> meetings":
                outputs.append((1, [0.45, 0.55]))
            else:
                outputs.append((1, [0.20, 0.80]))
        return outputs

    monkeypatch.setattr(masking, "predict_probabilities_batch", fake_predict_probabilities_batch)

    base_pred, base_probs, scores = masking.score_masked_spans(
        DummyWrapper(),
        "headline: boss excited meeting about meetings",
        ngram_sizes=[2],
        device=None,
    )

    assert base_pred == 1
    assert base_probs[1] == 0.90
    assert scores[0].span == "meeting about"
    assert scores[0].sarcastic_delta == 0.35


def test_predict_probabilities_uses_batched_helper(monkeypatch):
    class DummyWrapper:
        pass

    monkeypatch.setattr(
        masking,
        "predict_probabilities_batch",
        lambda wrapper, texts, device, batch_size=64: [(1, [0.2, 0.8]) for _ in texts],
    )

    pred, probs = masking.predict_probabilities(DummyWrapper(), "headline: hello", device=None, batch_size=8)

    assert pred == 1
    assert probs == [0.2, 0.8]


def test_load_input_rows_reads_csv_and_adds_row_id(tmp_path):
    csv_path = tmp_path / "toy.csv"
    csv_path.write_text("headline,article_section,description\nA,Business,Desc\n", encoding="utf-8")

    df = masking.load_input_rows(csv_path)

    assert list(df["headline"]) == ["A"]
    assert list(df["row_id"]) == [0]


def test_aggregate_span_scores_summarizes_across_examples():
    analyses = [
        masking.ExampleAnalysis(
            row_id=0,
            label=1,
            headline="h1",
            section="business",
            description="",
            input_text="headline: h1 section: business",
            predicted_label=1,
            non_sarcastic_prob=0.1,
            sarcastic_prob=0.9,
            top_word="meeting",
            top_word_delta=0.3,
            top_word_masked_sarcastic_prob=0.6,
            top_scores=[
                masking.SpanScore(2, 0, 2, "meeting about", "<mask> business", 0.6, 0.3, 1),
                masking.SpanScore(1, 2, 3, "business", "headline: h1 section: <mask>", 0.7, 0.2, 1),
            ],
        ),
        masking.ExampleAnalysis(
            row_id=1,
            label=1,
            headline="h2",
            section="business",
            description="",
            input_text="headline: h2 section: business",
            predicted_label=1,
            non_sarcastic_prob=0.2,
            sarcastic_prob=0.8,
            top_word="meeting",
            top_word_delta=0.4,
            top_word_masked_sarcastic_prob=0.4,
            top_scores=[
                masking.SpanScore(2, 0, 2, "meeting about", "<mask> business", 0.4, 0.4, 1),
            ],
        ),
    ]

    grouped = masking.aggregate_chosen_words(analyses)

    assert grouped.iloc[0]["word"] == "meeting"
    assert grouped.iloc[0]["count"] == 2
    assert grouped.iloc[0]["mean_sarcastic_delta"] == 0.35
