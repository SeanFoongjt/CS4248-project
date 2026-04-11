import csv
import random
from collections import Counter
from pathlib import Path


MISSING_SECTION = "<missing>"


def parse_section_labels(section: str) -> list[str]:
    section = (section or "").strip()
    if not section or section == MISSING_SECTION:
        return []
    return [part.strip() for part in section.split(",") if part.strip()]


def format_section_labels(labels: list[str]) -> str:
    return ", ".join(labels)


def load_section_pool(section_counts_path: str | Path) -> tuple[list[str], list[int]]:
    """Load atomic section labels and empirical frequencies from the counts CSV."""
    section_weights = Counter()
    with Path(section_counts_path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            section = row["section"]
            count = int(row["total_count"])
            for label in parse_section_labels(section):
                if label != MISSING_SECTION:
                    section_weights[label] += count

    if not section_weights:
        raise ValueError(f"No non-missing section labels found in {section_counts_path}.")

    labels, weights = zip(*sorted(section_weights.items()))
    return list(labels), list(weights)


def weighted_sample_without_replacement(
    labels: list[str],
    weights: list[int],
    count: int,
    rng: random.Random,
) -> list[str]:
    if count > len(labels):
        raise ValueError(
            f"Cannot sample {count} unique section labels from a pool of {len(labels)} labels."
        )

    available_labels = list(labels)
    available_weights = list(weights)
    sampled = []

    for _ in range(count):
        choice_index = rng.choices(
            range(len(available_labels)),
            weights=available_weights,
            k=1,
        )[0]
        sampled.append(available_labels.pop(choice_index))
        available_weights.pop(choice_index)

    return sampled


def replace_article_sections(
    samples: list[dict],
    section_labels: list[str],
    section_weights: list[int],
    seed: int,
) -> list[dict]:
    """Replace each known model-facing section with distinct random section labels."""
    rng = random.Random(seed)
    replaced_samples = []

    for sample in samples:
        original_section = (sample.get("section") or "").strip()
        original_labels = parse_section_labels(original_section)

        new_sample = dict(sample)
        new_sample["original_section"] = original_section

        if not original_labels:
            new_sample["replacement_section"] = original_section
            replaced_samples.append(new_sample)
            continue

        original_label_set = set(original_labels)
        candidate_pairs = [
            (label, weight)
            for label, weight in zip(section_labels, section_weights)
            if label not in original_label_set
        ]

        if len(candidate_pairs) >= len(original_labels):
            candidate_labels = [label for label, _ in candidate_pairs]
            candidate_weights = [weight for _, weight in candidate_pairs]
        else:
            candidate_labels = section_labels
            candidate_weights = section_weights

        replacement_labels = weighted_sample_without_replacement(
            candidate_labels,
            candidate_weights,
            len(original_labels),
            rng,
        )
        replacement_section = format_section_labels(replacement_labels)

        new_sample["section"] = replacement_section
        new_sample["replacement_section"] = replacement_section
        replaced_samples.append(new_sample)

    return replaced_samples
