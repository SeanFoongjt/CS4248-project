from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from .transformer_base import HfSarcasmModelBase, TransformerConfig

@dataclass
class DistilBertConfig(TransformerConfig):
    """Config for DistilBERT sarcasm model."""

    pretrained_name: str = "distilbert-base-uncased"
    max_length: int = 128
    num_labels: int = 2

class DistilBertSarcasmModel(HfSarcasmModelBase):
    """Thin DistilBERT base wrapper for sarcasm detection."""

    def __init__(self, cfg: Optional[DistilBertConfig] = None):
        super().__init__(cfg or DistilBertConfig())