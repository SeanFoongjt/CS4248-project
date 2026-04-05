from __future__ import annotations
from scripts import context_ablation_analysis as ab

def test_build_recipe_text_fills_missing_fields():
    text = ab.build_recipe_text(
        "headline_section_description",
        "boss excited to schedule meeting",
        section="",
        description="",
    )

    assert "headline: boss excited to schedule meeting" in text
    assert f"section: {ab.EMPTY_TOKEN}" in text
    assert f"description: {ab.EMPTY_TOKEN}" in text

def test_shuffle_words_keeps_same_words():
    rng = __import__("random").Random(7)
    text = "workers react with joy"
    shuffled = ab.shuffle_words(text, rng)

    assert sorted(text.split()) == sorted(shuffled.split())

def test_make_variants_for_full_recipe_contains_main_checks():
    pools = {
        0: {
            "section": [{"row_id": 10, "value": "politics"}],
            "description": [{"row_id": 10, "value": "citizens follow the update closely"}],
        },
        1: {
            "section": [{"row_id": 11, "value": "news"}],
            "description": [{"row_id": 11, "value": "workers react with joy"}],
        },
    }
    rng = __import__("random").Random(1)
    variants = ab.make_variants(
        recipe_name="headline_section_description",
        headline="boss excited to schedule meeting about meetings",
        section="business",
        description="workers react with joy",
        current_row_id=1,
        label=1,
        pools=pools,
        rng=rng,
    )
    names = [name for name, _ in variants]

    assert names[0] == "full"
    assert "no_section" in names
    assert "no_description" in names
    assert "headline_only_template" in names
    assert "context_only" in names
    assert "swap_section_opposite_label" in names
    assert "swap_description_opposite_label" in names

def test_aggregate_variant_metrics_computes_flip_rate():
    analyses = [
        ab.RowAnalysis(
            row_id=0,
            label=1,
            headline="h1",
            section="business",
            description="workers react with joy",
            base_text="headline: h1 section: business description: workers react with joy",
            base_predicted_label=1,
            base_non_sarcastic_prob=0.1,
            base_sarcastic_prob=0.9,
            variants=[
                ab.VariantResult(
                    name="no_description",
                    text="headline: h1 section: business description: [EMPTY]",
                    predicted_label=1,
                    non_sarcastic_prob=0.2,
                    sarcastic_prob=0.8,
                    confidence_delta=0.1,
                    sarcastic_delta=0.1,
                    label_flip=False,
                ),
                ab.VariantResult(
                    name="swap_description_opposite_label",
                    text="headline: h1 section: business description: normal story",
                    predicted_label=0,
                    non_sarcastic_prob=0.7,
                    sarcastic_prob=0.3,
                    confidence_delta=0.6,
                    sarcastic_delta=0.6,
                    label_flip=True,
                ),
            ],
        ),
        ab.RowAnalysis(
            row_id=1,
            label=1,
            headline="h2",
            section="business",
            description="citizens follow the update",
            base_text="headline: h2 section: business description: citizens follow the update",
            base_predicted_label=1,
            base_non_sarcastic_prob=0.2,
            base_sarcastic_prob=0.8,
            variants=[
                ab.VariantResult(
                    name="swap_description_opposite_label",
                    text="headline: h2 section: business description: normal story",
                    predicted_label=0,
                    non_sarcastic_prob=0.8,
                    sarcastic_prob=0.2,
                    confidence_delta=0.6,
                    sarcastic_delta=0.6,
                    label_flip=True,
                )
            ],
        ),
    ]
    summary = ab.aggregate_variant_metrics(analyses)
    row = summary[summary["variant"] == "swap_description_opposite_label"].iloc[0]

    assert row["count"] == 2
    assert abs(row["mean_confidence_delta"] - 0.6) < 1e-9
    assert abs(row["flip_rate"] - 1.0) < 1e-9