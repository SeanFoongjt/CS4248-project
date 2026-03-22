from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

@dataclass
class TfidfNbConfig:
    """Config for TF-IDF + Multinomial Naive Bayes."""

    ngram_range: tuple[int, int] = (1, 2)
    min_df: int | float = 2
    max_df: int | float = 0.98
    max_features: Optional[int] = None
    sublinear_tf: bool = True
    norm: str = "l2"
    alpha: float = 1.0
    lowercase: bool = False

class TfidfNbModel:
    """Thin builder for a TF-IDF + MultinomialNB baseline.
    Notes:
        - This wrapper does not own preprocessing. It expects text that is
          already prepared by the upstream preprocessing step.
        - To move from headline-only to context-aware later, only change the
          input text that you feed into the pipeline.
    """

    def __init__(self, cfg: Optional[TfidfNbConfig] = None):
        self.cfg = cfg or TfidfNbConfig()

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

    def build_classifier(self) -> MultinomialNB:
        """Build the Naive Bayes classifier."""
        
        return MultinomialNB(alpha=self.cfg.alpha)

    def build_pipeline(self) -> Pipeline:
        """Build the full sklearn pipeline."""
        
        return Pipeline(
            steps=[
                ("tfidf", self.build_vectorizer()),
                ("clf", self.build_classifier()),
            ]
        )