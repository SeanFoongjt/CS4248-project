from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn
from transformers import DataCollatorWithPadding, RobertaModel, RobertaTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput

from .transformer_base import SarcasmTextDataset, TransformerConfig


@dataclass
class RobertaConfig(TransformerConfig):
    """Config for the custom RoBERTa sarcasm model."""

    pretrained_name: str = "roberta-base"
    max_length: int = 128
    num_labels: int = 2
    dropout: float = 0.1


class CustomRobertaClassifier(nn.Module):
    """Collaborator-style RoBERTa encoder with a custom MLP head."""

    def __init__(self, cfg: RobertaConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = RobertaModel.from_pretrained(cfg.pretrained_name)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden_size // 2, cfg.num_labels),
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **_: dict,
    ) -> SequenceClassifierOutput:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_repr)
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return SequenceClassifierOutput(loss=loss, logits=logits)


class RobertaSarcasmModel:
    """Custom RoBERTa wrapper compatible with the shared training pipeline."""

    def __init__(self, cfg: Optional[RobertaConfig] = None):
        self.cfg = cfg or RobertaConfig()
        self.tokenizer = RobertaTokenizer.from_pretrained(self.cfg.pretrained_name)
        self.model = CustomRobertaClassifier(self.cfg)
        self.collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            return_tensors="pt",
        )

    def make_dataset(
        self,
        texts: Sequence[str],
        labels: Optional[Sequence[int]] = None,
    ) -> SarcasmTextDataset:
        return SarcasmTextDataset(
            texts=texts,
            tokenizer=self.tokenizer,
            max_length=self.cfg.max_length,
            labels=labels,
        )

    def forward_batch(self, batch: dict) -> SequenceClassifierOutput:
        return self.model(**batch)

    def to(self, device: torch.device | str) -> "RobertaSarcasmModel":
        self.model.to(device)
        return self

    def save_pretrained(self, out_dir: str | Path) -> None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), out_path / "model_state.pt")
        with open(out_path / "model_config.json", "w", encoding="utf-8") as handle:
            json.dump(asdict(self.cfg), handle, ensure_ascii=False, indent=2)
