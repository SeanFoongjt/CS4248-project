from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

@dataclass
class TransformerConfig:
    """Shared config for transformer sarcasm models."""

    pretrained_name: str
    max_length: int = 128
    num_labels: int = 2


class SarcasmTextDataset(Dataset):
    """Dataset that tokenizes one text input per sample."""

    def __init__(
        self,
        texts: Sequence[str],
        tokenizer,
        max_length: int,
        labels: Optional[Sequence[int]] = None,
    ):
        self.texts = list(texts)
        self.labels = None if labels is None else [int(x) for x in labels]
        self.tokenizer = tokenizer
        self.max_length = max_length

        if self.labels is not None and len(self.texts) != len(self.labels):
            raise ValueError("texts and labels must have the same length")

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        item = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
        )
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


class HfSarcasmModelBase:
    """Thin HF wrapper: tokenizer + dataset + collator + model.
    This class intentionally stops at the model-wrapper level. It does not
    include fit / eval loops, hyperparameter search, or report logic.
    """

    def __init__(self, cfg: TransformerConfig):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.pretrained_name,
            use_fast=True,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            cfg.pretrained_name,
            num_labels=cfg.num_labels,
        )
        self.collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            return_tensors="pt",
        )

    def make_dataset(
        self,
        texts: Sequence[str],
        labels: Optional[Sequence[int]] = None,
    ) -> SarcasmTextDataset:
        """Create a dataset for this model."""

        return SarcasmTextDataset(
            texts=texts,
            tokenizer=self.tokenizer,
            max_length=self.cfg.max_length,
            labels=labels,
        )

    def forward_batch(self, batch: dict):
        """Run one forward pass on a collated batch."""

        return self.model(**batch)

    def to(self, device: torch.device | str) -> "HfSarcasmModelBase":
        """Move the underlying HF model to a device."""

        self.model.to(device)
        return self