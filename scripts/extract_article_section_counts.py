import argparse
import ast
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path


DEFAULT_INPUT = Path("data") / "Sarcasm_Headlines_Preprocessed.csv"
DEFAULT_OUTPUT_DIR = Path("data") / "article_section_counts"
SOURCE_COLUMNS = ("onion", "huff")
MISSING_SECTION = "<missing>"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract raw and preprocessed article-section counts by source from "
            "Sarcasm_Headlines_Preprocessed.csv."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input preprocessed sarcasm-headlines CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated CSV/JSON section count artifacts.",
    )
    return parser.parse_args()


def normalize_missing(value):
    value = (value or "").strip()
    return value if value else MISSING_SECTION


def parse_raw_article_sections(value):
    """Return one or more raw article_section labels from a CSV cell."""
    value = (value or "").strip()
    if not value:
        return [MISSING_SECTION]

    if value.startswith("[") and value.endswith("]"):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, list):
            sections = [normalize_missing(str(item)) for item in parsed]
            return sections or [MISSING_SECTION]

    return [value]


def increment(counter, source_counter, section, source):
    counter[section] += 1
    source_counter[(section, source)] += 1


def build_count_rows(counter, source_counter):
    rows = []
    for section, total_count in counter.items():
        row = {
            "section": section,
            "total_count": total_count,
            "onion_count": source_counter[(section, "onion")],
            "huff_count": source_counter[(section, "huff")],
        }
        other_count = total_count - sum(row[f"{source}_count"] for source in SOURCE_COLUMNS)
        if other_count:
            row["other_source_count"] = other_count
        rows.append(row)

    return sorted(
        rows,
        key=lambda row: (-row["total_count"], row["section"].casefold()),
    )


def write_csv(path, rows):
    fieldnames = ["section", "total_count", "onion_count", "huff_count"]
    if any("other_source_count" in row for row in rows):
        fieldnames.append("other_source_count")

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    raw_counter = Counter()
    raw_source_counter = Counter()
    preprocessed_counter = Counter()
    preprocessed_source_counter = Counter()
    source_counter = Counter()
    row_count = 0

    with args.input.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_count += 1
            source = (row.get("source") or "").strip().lower()
            source_counter[source] += 1

            for raw_section in parse_raw_article_sections(row.get("article_section")):
                increment(raw_counter, raw_source_counter, raw_section, source)

            preprocessed_section = normalize_missing(row.get("preprocessed_article_section"))
            increment(
                preprocessed_counter,
                preprocessed_source_counter,
                preprocessed_section,
                source,
            )

    raw_rows = build_count_rows(raw_counter, raw_source_counter)
    preprocessed_rows = build_count_rows(
        preprocessed_counter,
        preprocessed_source_counter,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_csv_path = args.output_dir / "raw_article_section_counts_by_source.csv"
    preprocessed_csv_path = (
        args.output_dir / "preprocessed_article_section_counts_by_source.csv"
    )
    json_path = args.output_dir / "article_section_counts_by_source.json"

    write_csv(raw_csv_path, raw_rows)
    write_csv(preprocessed_csv_path, preprocessed_rows)
    json_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "generated_at": datetime.now().isoformat(timespec="seconds"),
                    "input_path": str(args.input),
                    "row_count": row_count,
                    "source_counts": dict(source_counter),
                    "missing_section_label": MISSING_SECTION,
                    "notes": [
                        "raw_article_sections explodes list-like article_section values into one row per section label.",
                        "preprocessed_article_sections is the model-ready candidate set for random section replacement.",
                    ],
                },
                "raw_article_sections": raw_rows,
                "preprocessed_article_sections": preprocessed_rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"Read {row_count} rows from {args.input}")
    print(f"Wrote {len(raw_rows)} raw section rows to {raw_csv_path}")
    print(
        f"Wrote {len(preprocessed_rows)} preprocessed section rows to {preprocessed_csv_path}"
    )
    print(f"Wrote combined JSON artifact to {json_path}")


if __name__ == "__main__":
    main()
