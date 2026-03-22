from __future__ import annotations
from models.distilbert import DistilBertConfig, DistilBertSarcasmModel
from models.roberta import RobertaConfig, RobertaSarcasmModel
from models.tfidf_lr import TfidfLogRegModel, TfidfLrConfig
from models.tfidf_nb import TfidfNbConfig, TfidfNbModel

# Small toy data for base-model smoke tests.
CLASSICAL_TEXTS = [
    "former versace store clerk sue secret black code minority shopper",
    "roseanne revival catch thorny political mood better worse",
    "mom start fear son web series closest thing grandchild",
    "boehner want wife listen not come alternative debt reduction idea",
]

CLASSICAL_LABELS = [0, 0, 1, 1]

RAW_TEXTS = [
    "former versace store clerk sues over secret black code for minority shoppers",
    "mom starting to fear son's web series closest thing she will have to grandchild",
]

RAW_LABELS = [0, 1]

def test_tfidf_nb_pipeline_runs():
    wrapper = TfidfNbModel(TfidfNbConfig(min_df=1, max_df=1.0))
    pipe = wrapper.build_pipeline()
    pipe.fit(CLASSICAL_TEXTS, CLASSICAL_LABELS)
    pred = pipe.predict(CLASSICAL_TEXTS)

    assert len(pred) == len(CLASSICAL_TEXTS)

def test_tfidf_lr_pipeline_runs():
    wrapper = TfidfLogRegModel(TfidfLrConfig(min_df=1, max_df=1.0))
    pipe = wrapper.build_pipeline()
    pipe.fit(CLASSICAL_TEXTS, CLASSICAL_LABELS)
    pred = pipe.predict(CLASSICAL_TEXTS)

    assert len(pred) == len(CLASSICAL_TEXTS)

def test_distilbert_forward_batch_runs():
    wrapper = DistilBertSarcasmModel(DistilBertConfig(max_length=64))
    ds = wrapper.make_dataset(RAW_TEXTS, RAW_LABELS)
    batch = wrapper.collator([ds[0], ds[1]])
    out = wrapper.forward_batch(batch)

    assert out.logits.shape == (2, 2)
    assert out.loss is not None

def test_roberta_forward_batch_runs():
    wrapper = RobertaSarcasmModel(RobertaConfig(max_length=64))
    ds = wrapper.make_dataset(RAW_TEXTS, RAW_LABELS)
    batch = wrapper.collator([ds[0], ds[1]])
    out = wrapper.forward_batch(batch)

    assert out.logits.shape == (2, 2)
    assert out.loss is not None