from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

@dataclass
class TfidfLrConfig:
    """Config for TF-IDF + Logistic Regression."""

    ngram_range: tuple[int, int] = (1, 2)
    min_df: int | float = 2
    max_df: int | float = 0.98
    max_features: Optional[int] = None
    sublinear_tf: bool = True
    norm: str = "l2"
    C: float = 1.0
    max_iter: int = 2000
    solver: str = "liblinear"
    class_weight: Optional[str] = None
    random_state: int = 42
    lowercase: bool = False

class TfidfLogRegModel:
    """Thin builder for a TF-IDF + Logistic Regression baseline.
    Notes:
        - This wrapper does not own preprocessing. It expects text that is
          already prepared by the upstream preprocessing step.
        - To move from headline-only to context-aware later, only change the
          input text that you feed into the pipeline.
    """

    def __init__(self, cfg: Optional[TfidfLrConfig] = None):
        self.cfg = cfg or TfidfLrConfig()

    def build_vectorizer(self) -> TfidfVectorizer:
        """Build the TF-IDF vectorizer."""

        return TfidfVectorizer(
            ngram_range=self.cfg.ngram_range,
            min_df=self.cfg.min_df,
            max_df=self.cfg.max_df,
            max_features=self.cfg.max_features,
            sublinear_tf=self.cfg.sublinear_tf,
            norm=self.cfg.norm,
            lowercase=self.cfg.lowercase,
        )

    def build_classifier(self) -> LogisticRegression:
        """Build the logistic regression classifier."""
        
        return LogisticRegression(
            C=self.cfg.C,
            max_iter=self.cfg.max_iter,
            solver=self.cfg.solver,
            class_weight=self.cfg.class_weight,
            random_state=self.cfg.random_state,
        )

    def build_pipeline(self) -> Pipeline:
        """Build the full sklearn pipeline."""
        
        return Pipeline(
            steps=[
                ("tfidf", self.build_vectorizer()),
                ("clf", self.build_classifier()),
            ]
        )