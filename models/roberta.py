from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from .transformer_base import HfSarcasmModelBase, TransformerConfig

@dataclass
class RobertaConfig(TransformerConfig):
    """Config for RoBERTa sarcasm model."""

    pretrained_name: str = "roberta-base"
    max_length: int = 128
    num_labels: int = 2

class RobertaSarcasmModel(HfSarcasmModelBase):
    """Thin RoBERTa base wrapper for sarcasm detection."""

    def __init__(self, cfg: Optional[RobertaConfig] = None):
        super().__init__(cfg or RobertaConfig())