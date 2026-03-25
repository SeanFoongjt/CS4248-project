from __future__ import annotations
 
import torch
import json
import random
import sklearn
import numpy as np
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from dataclasses import dataclass
from typing import Optional
import numpy as np
 
 
# ─────────────────────────────────────────────
# 1. CONFIG
# ─────────────────────────────────────────────
 
@dataclass
class RobertaConfig:
    pretrained_name: str = "roberta-base"
    max_length: int = 128
    num_labels: int = 2
    dropout: float = 0.1
    learning_rate: float = 2e-5
    batch_size: int = 32
    num_epochs: int = 3
    warmup_ratio: float = 0.1
 
 
# ─────────────────────────────────────────────
# 2. DATASET  (preprocess step)
# ─────────────────────────────────────────────
 
class SarcasmDataset(Dataset):
    """
    Tokenises raw text and returns tensors ready for RoBERTa.
    Expects a list of (text, label) tuples.
    """
 
    def __init__(
        self,
        samples: list[tuple[str, int]],
        tokenizer: RobertaTokenizer,
        max_length: int,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
 
    def __len__(self) -> int:
        return len(self.samples)
 
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text, label = self.samples[idx]
 
        # Tokenise: pad/truncate to max_length, return attention mask
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
 
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),       # (max_length,)
            "attention_mask": encoding["attention_mask"].squeeze(0),   # (max_length,)
            "label":          torch.tensor(label, dtype=torch.long),
        }
 
 
# ─────────────────────────────────────────────
# 3. MODEL  (RoBERTa encoder + feedforward head)
# ─────────────────────────────────────────────
 
class RobertaSarcasmModel(nn.Module):
    """
    Pipeline:
        tokenised input
            → RoBERTa encoder
            → [CLS] representation   (hidden_size,)
            → dropout
            → linear classifier      (num_labels,)
            → prediction
    """
 
    def __init__(self, cfg: RobertaConfig):
        super().__init__()
        self.cfg = cfg
 
        # Pretrained RoBERTa encoder
        self.encoder = RobertaModel.from_pretrained(cfg.pretrained_name)
        hidden_size = self.encoder.config.hidden_size  # 768 for roberta-base
 
        # Feedforward classification head
        self.classifier = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden_size // 2, cfg.num_labels),
        )
 
    def forward(
        self,
        input_ids: torch.Tensor,       # (batch, seq_len)
        attention_mask: torch.Tensor,  # (batch, seq_len)
    ) -> torch.Tensor:                 # (batch, num_labels) — raw logits
 
        # Transformer output — take [CLS] token (index 0)
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr = outputs.last_hidden_state[:, 0, :]  # (batch, hidden_size)
 
        logits = self.classifier(cls_repr)             # (batch, num_labels)
        return logits
 
 
# ─────────────────────────────────────────────
# 4. TRAINING LOOP
# ─────────────────────────────────────────────
 
def train(
    model: RobertaSarcasmModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: RobertaConfig,
    device: torch.device,
) -> None:
 
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
 
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=0.01)
 
    total_steps = len(train_loader) * cfg.num_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
 
    for epoch in range(cfg.num_epochs):
        # ── train ──
        model.train()
        train_loss = 0.0
 
        for batch in train_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)
 
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss   = loss_fn(logits, labels)
            loss.backward()
 
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
 
            train_loss += loss.item()
 
        # ── validate ──
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
 
        print(
            f"Epoch {epoch + 1}/{cfg.num_epochs} | "
            f"train_loss={train_loss / len(train_loader):.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f}"
        )
 
 
# ─────────────────────────────────────────────
# 5. EVALUATION
# ─────────────────────────────────────────────
 
@torch.no_grad()
def evaluate(
    model: RobertaSarcasmModel,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
 
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
 
    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)
 
        logits = model(input_ids, attention_mask)
        loss   = loss_fn(logits, labels)
 
        total_loss += loss.item()
        preds       = logits.argmax(dim=-1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)
 
    return total_loss / len(loader), correct / total
 
 
# ─────────────────────────────────────────────
# 6. ENTRY POINT
# ─────────────────────────────────────────────
 
def build_pipeline(
    train_samples: list[tuple[str, int]],
    val_samples: list[tuple[str, int]],
    cfg: Optional[RobertaConfig] = None,
) -> None:
    """
    End-to-end pipeline entry point.
 
    Args:
        train_samples: list of (text, label) for training
        val_samples:   list of (text, label) for validation
        cfg:           optional config (uses defaults if None)
    """
    cfg = cfg or RobertaConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    # Tokeniser (shared between train and val)
    tokenizer = RobertaTokenizer.from_pretrained(cfg.pretrained_name)
 
    # Datasets + loaders
    train_ds = SarcasmDataset(train_samples, tokenizer, cfg.max_length)
    val_ds   = SarcasmDataset(val_samples,   tokenizer, cfg.max_length)
 
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size)
 
    # Model
    model = RobertaSarcasmModel(cfg)
 
    # Train
    train(model, train_loader, val_loader, cfg, device)
 
    return model
 
 
# ─────────────────────────────────────────────
# Example usage
# ─────────────────────────────────────────────
 
if __name__ == "__main__":
    samples = []

    with open('Sarcasm_Headlines_Dataset_v2.json') as f:
        for line in f:
            item = json.loads(line)
            headline = item["headline"]
            label = item["is_sarcastic"]
            samples.append((headline, label))

    with open('Sarcasm_Headlines_Dataset.json') as f:
        for line in f:
            item = json.loads(line)
            headline = item["headline"]
            label = item["is_sarcastic"]
            samples.append((headline, label))

    seed = random.randint(1,100)
    print("The seed is " + str(seed))

    data, data_test = train_test_split(samples, test_size=0.2, random_state=seed)
    data_train, data_valid = train_test_split(data, test_size=0.25, random_state=seed)
 
    model = build_pipeline(data_train, data_valid)