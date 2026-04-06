from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn
from transformers import AutoTokenizer, DataCollatorWithPadding, RobertaConfig as HfRobertaConfig, RobertaModel, RobertaTokenizer
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

    def __init__(
        self,
        cfg: RobertaConfig,
        *,
        init_encoder_from_pretrained: bool = True,
        encoder_config: Optional[HfRobertaConfig] = None,
    ):
        super().__init__()
        self.cfg = cfg
        if init_encoder_from_pretrained:
            self.encoder = RobertaModel.from_pretrained(cfg.pretrained_name)
        else:
            self.encoder = RobertaModel(encoder_config or _load_fallback_encoder_config(cfg.pretrained_name))
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


def _load_fallback_encoder_config(pretrained_name: str) -> HfRobertaConfig:
    try:
        return HfRobertaConfig.from_pretrained(pretrained_name, local_files_only=True)
    except Exception:
        if pretrained_name == "roberta-base":
            return HfRobertaConfig(
                vocab_size=50265,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                hidden_act="gelu",
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                max_position_embeddings=514,
                type_vocab_size=1,
                initializer_range=0.02,
                layer_norm_eps=1e-5,
                pad_token_id=1,
                bos_token_id=0,
                eos_token_id=2,
                position_embedding_type="absolute",
            )
        raise RuntimeError(
            f"Could not load a local encoder config for {pretrained_name}. "
            "Download the model once or save the encoder config with the checkpoint."
        )


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
        self.model.encoder.config.to_json_file(out_path / "encoder_config.json")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        *,
        device: torch.device | str = "cpu",
    ) -> "RobertaSarcasmModel":
        checkpoint_path = Path(checkpoint_dir)
        cfg = RobertaConfig(**json.loads((checkpoint_path / "model_config.json").read_text(encoding="utf-8")))
        encoder_cfg_path = checkpoint_path / "encoder_config.json"
        if encoder_cfg_path.exists():
            encoder_config = HfRobertaConfig.from_json_file(str(encoder_cfg_path))
        else:
            encoder_config = _load_fallback_encoder_config(cfg.pretrained_name)
        self = cls.__new__(cls)
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, local_files_only=True, use_fast=True)
        self.model = CustomRobertaClassifier(
            cfg,
            init_encoder_from_pretrained=False,
            encoder_config=encoder_config,
        )
        state_dict = torch.load(checkpoint_path / "model_state.pt", map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            return_tensors="pt",
        )
        self.model.to(device)
        return self
