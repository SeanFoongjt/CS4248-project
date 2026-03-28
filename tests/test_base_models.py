from __future__ import annotations
from types import SimpleNamespace

import torch
import torch.nn as nn
import pytest
from transformers.modeling_outputs import BaseModelOutput, SequenceClassifierOutput

from models.distilbert import DistilBertConfig, DistilBertSarcasmModel
import models.transformer_base as transformer_base
from models.roberta import RobertaConfig, RobertaSarcasmModel
import models.roberta as roberta_module
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


class FakeTokenizer:
    def __call__(self, text, truncation=True, max_length=128):
        size = min(len(text.split()), max_length)
        return {
            "input_ids": [1] * max(1, size),
            "attention_mask": [1] * max(1, size),
        }


class FakeCollator:
    def __init__(self, tokenizer=None, return_tensors="pt"):
        self.tokenizer = tokenizer
        self.return_tensors = return_tensors

    def __call__(self, batch):
        max_len = max(len(item["input_ids"]) for item in batch)
        input_ids = []
        attention_mask = []
        labels = []
        for item in batch:
            pad = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [0] * pad)
            attention_mask.append(item["attention_mask"] + [0] * pad)
            if "labels" in item:
                labels.append(item["labels"])
        out = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
        if labels:
            out["labels"] = torch.tensor(labels, dtype=torch.long)
        return out


class FakeAutoSequenceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 2)

    def forward(self, input_ids, attention_mask=None, labels=None):
        pooled = input_ids.float().mean(dim=1, keepdim=True)
        logits = self.linear(pooled)
        loss = nn.CrossEntropyLoss()(logits, labels) if labels is not None else None
        return SequenceClassifierOutput(loss=loss, logits=logits)


class FakeRobertaEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        hidden = torch.ones(batch_size, seq_len, self.config.hidden_size, dtype=torch.float32)
        hidden = hidden * input_ids.unsqueeze(-1).float()
        return BaseModelOutput(last_hidden_state=hidden)


@pytest.fixture
def fake_transformers(monkeypatch):
    monkeypatch.setattr(
        transformer_base.AutoTokenizer,
        "from_pretrained",
        classmethod(lambda cls, *args, **kwargs: FakeTokenizer()),
    )
    monkeypatch.setattr(
        transformer_base.AutoModelForSequenceClassification,
        "from_pretrained",
        classmethod(lambda cls, *args, **kwargs: FakeAutoSequenceModel()),
    )
    monkeypatch.setattr(transformer_base, "DataCollatorWithPadding", FakeCollator)
    monkeypatch.setattr(
        roberta_module.RobertaTokenizer,
        "from_pretrained",
        classmethod(lambda cls, *args, **kwargs: FakeTokenizer()),
    )
    monkeypatch.setattr(roberta_module, "DataCollatorWithPadding", FakeCollator)
    monkeypatch.setattr(
        roberta_module.RobertaModel,
        "from_pretrained",
        classmethod(lambda cls, *args, **kwargs: FakeRobertaEncoder()),
    )

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

def test_distilbert_forward_batch_runs(fake_transformers):
    wrapper = DistilBertSarcasmModel(DistilBertConfig(max_length=64))
    ds = wrapper.make_dataset(RAW_TEXTS, RAW_LABELS)
    batch = wrapper.collator([ds[0], ds[1]])
    out = wrapper.forward_batch(batch)

    assert out.logits.shape == (2, 2)
    assert out.loss is not None

def test_roberta_forward_batch_runs(fake_transformers):
    wrapper = RobertaSarcasmModel(RobertaConfig(max_length=64, dropout=0.2))
    ds = wrapper.make_dataset(RAW_TEXTS, RAW_LABELS)
    batch = wrapper.collator([ds[0], ds[1]])
    out = wrapper.forward_batch(batch)

    assert out.logits.shape == (2, 2)
    assert out.loss is not None
    assert hasattr(wrapper.model, "encoder")
    assert isinstance(wrapper.model.classifier, nn.Sequential)
