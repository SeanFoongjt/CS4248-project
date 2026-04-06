from __future__ import annotations

from typing import Any, Callable, Iterable, Optional

import pandas as pd

from .context_common import ContextRecipe, load_jsonl

PreprocessorLoader = Callable[[], tuple[Callable[..., str], Callable[..., str], Callable[..., str]]]


def load_prepared_data(
    path: str,
    prepare_frame: Callable[[pd.DataFrame], pd.DataFrame],
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Load raw JSONL rows and prepare the dataframe once."""

    raw_df = load_jsonl(path)
    if max_rows is not None:
        raw_df = raw_df.iloc[:max_rows].copy().reset_index(drop=True)
    return prepare_frame(raw_df)


def _is_missing(value: Any) -> bool:
    """Return True when a value should be treated as missing."""

    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    return False


def _to_text(value: Any) -> str:
    """Convert metadata value to a plain string."""

    if _is_missing(value):
        return ""
    if isinstance(value, list):
        return " ".join(str(x).strip() for x in value if str(x).strip())
    return str(value).strip()


def _prep_headline_bow(text: Any, load_preprocessors: PreprocessorLoader) -> str:
    _, _, prep_bow = load_preprocessors()
    return prep_bow(_to_text(text))


def _prep_headline_transformer(text: Any) -> str:
    return _to_text(text)


def _prep_section_raw(text: Any, load_preprocessors: PreprocessorLoader) -> str:
    prep_sec, _, _ = load_preprocessors()
    raw = _to_text(text)
    if not raw:
        return ""
    return prep_sec(raw).strip()


def _prep_section_bow(text: Any, load_preprocessors: PreprocessorLoader) -> str:
    _, _, prep_bow = load_preprocessors()
    out = _prep_section_raw(text, load_preprocessors)
    if not out:
        return ""
    return prep_bow(out)


def _prep_description_raw(text: Any, load_preprocessors: PreprocessorLoader) -> str:
    _, prep_desc, _ = load_preprocessors()
    raw = _to_text(text)
    if not raw:
        return ""
    return prep_desc(raw).strip()


def _prep_description_bow(text: Any, load_preprocessors: PreprocessorLoader) -> str:
    _, _, prep_bow = load_preprocessors()
    out = _prep_description_raw(text, load_preprocessors)
    if not out:
        return ""
    return prep_bow(out)


def prepare_frame(df: pd.DataFrame, load_preprocessors: PreprocessorLoader) -> pd.DataFrame:
    """Build all preprocessed columns needed for the shared input recipes."""

    out = df.copy()
    out["label"] = out["is_sarcastic"].astype(int)
    out["headline_bow"] = out["headline"].map(lambda text: _prep_headline_bow(text, load_preprocessors))
    out["headline_tf"] = out["headline"].map(_prep_headline_transformer)
    out["section_raw"] = out["article_section"].map(lambda text: _prep_section_raw(text, load_preprocessors))
    out["section_bow"] = out["article_section"].map(lambda text: _prep_section_bow(text, load_preprocessors))
    out["description_raw"] = out["description"].map(lambda text: _prep_description_raw(text, load_preprocessors))
    out["description_bow"] = out["description"].map(lambda text: _prep_description_bow(text, load_preprocessors))
    return out


def _join_parts(parts: Iterable[str]) -> str:
    return " ".join(part.strip() for part in parts if part and part.strip()).strip()


def build_recipe_text(row: pd.Series, recipe: ContextRecipe, family: str) -> str:
    """Build one training text for a given model family."""

    if family not in {"classical", "transformer"}:
        raise ValueError(f"Unknown family: {family}")
    if family == "classical":
        headline = row["headline_bow"]
        section = row["section_bow"]
        description = row["description_bow"]
    else:
        headline = row["headline_tf"]
        section = row["section_raw"]
        description = row["description_raw"]
    parts = [f"headline: {headline}"]
    if recipe.needs_section:
        parts.append(f"section: {section}")
    if recipe.needs_description:
        parts.append(f"description: {description}")
    return _join_parts(parts)


def subset_for_recipe(df: pd.DataFrame, recipe: ContextRecipe) -> pd.DataFrame:
    """Keep only rows that truly satisfy the context recipe."""

    mask = pd.Series(True, index=df.index)
    if recipe.needs_section:
        mask &= df["section_raw"].astype(str).str.len() > 0
    if recipe.needs_description:
        mask &= df["description_raw"].astype(str).str.len() > 0
    return df.loc[mask].copy().reset_index(drop=True)


def build_recipe_frame(df: pd.DataFrame, recipe: ContextRecipe, family: str) -> pd.DataFrame:
    """Build the actual text column used by one recipe/model-family pair."""

    sub = subset_for_recipe(df, recipe).copy()
    sub["text"] = sub.apply(build_recipe_text, axis=1, recipe=recipe, family=family)
    return sub[["row_id", "label", "text"]].copy()
