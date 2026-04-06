import argparse
import json
from pathlib import Path
from urllib.parse import urlsplit
from urllib.parse import quote

from url_matching import clean_by_source, extract_id_for_source, is_target_host, parse_cdx_line


def is_valid_clean_article_link(article_link: str, source: str):
    if not isinstance(article_link, str) or not article_link.strip():
        return False

    try:
        parsed = urlsplit(article_link)
    except Exception:
        return False

    path = (parsed.path or "").casefold()
    full = article_link.casefold()

    if source == "huff":
        if not path.startswith("/entry/"):
            return False
        if path.count("/entry/") != 1:
            return False
        tail = full.split("://", 1)[-1]
        if "http:/" in tail or "https:/" in tail:
            return False
        if any(token in full for token in ('"', "%22", "%0a", "\n", "\r", "{", "}", "<", ">")):
            return False

    return True


def load_cdx_file(path: Path, source: str):
    latest_by_article_id = {}
    lines_read = 0
    parse_fail = 0
    non_target_rows = 0
    invalid_article_rows = 0

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            lines_read += 1
            parsed = parse_cdx_line(line)
            if not parsed:
                parse_fail += 1
                continue

            timestamp, original = parsed
            try:
                host = original.split("/", 3)[2]
            except Exception:
                host = ""

            if not is_target_host(host, source):
                non_target_rows += 1
                continue

            article_link = clean_by_source(original, source)
            if not article_link:
                invalid_article_rows += 1
                continue
            if not is_valid_clean_article_link(article_link, source):
                invalid_article_rows += 1
                continue
            article_id = extract_id_for_source(original, source) or extract_id_for_source(article_link, source)
            if not article_id:
                invalid_article_rows += 1
                continue

            row = {
                "article_id": article_id,
                "article_link": article_link,
                "wayback_url": f"https://web.archive.org/web/{timestamp}/{quote(article_link, safe=':/?&=%#')}",
                "wayback_timestamp": timestamp,
            }
            prev = latest_by_article_id.get(article_id)
            if prev is None or timestamp > prev["wayback_timestamp"] or (
                timestamp == prev["wayback_timestamp"] and article_link < prev["article_link"]
            ):
                latest_by_article_id[article_id] = row

    return latest_by_article_id, {
        "lines_read": lines_read,
        "parse_fail": parse_fail,
        "non_target_rows": non_target_rows,
        "invalid_article_rows": invalid_article_rows,
        "unique_articles": len(latest_by_article_id),
    }


def sort_rows(rows: list[dict]):
    return sorted(rows, key=lambda row: (row["article_link"], row["wayback_timestamp"], row.get("article_id") or ""))


def select_balanced_huff_rows(sorted_rows_by_file: list[list[dict]], target_k: int):
    seen_article_ids = set()
    selected_by_file = []

    for rows in sorted_rows_by_file:
        chosen = []
        for row in rows:
            article_id = row["article_id"]
            if article_id in seen_article_ids:
                continue
            chosen.append(row)
            seen_article_ids.add(article_id)
            if len(chosen) == target_k:
                break
        if len(chosen) < target_k:
            return None
        selected_by_file.append(chosen)

    return selected_by_file


def parse_args():
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Build balanced extra Onion/Huff Wayback URL dumps from CDX txt files."
    )
    parser.add_argument(
        "--onion-cdx",
        nargs="+",
        default=[
            str(base_dir / "cdx-results-2010-onwards-new.txt"),
            str(base_dir / "cdx-results-2020-onwards-new.txt"),
        ],
        help="Onion CDX txt files to combine.",
    )
    parser.add_argument(
        "--huff-cdx",
        nargs="+",
        default=[
            str(base_dir / "huff-cdx-results-2020-2026-a-e.txt"),
            str(base_dir / "huff-cdx-results-2020-2026-f-j.txt"),
            str(base_dir / "huff-cdx-results-2020-2026-k-o.txt"),
            str(base_dir / "huff-cdx-results-2020-2026-p-t.txt"),
            str(base_dir / "huff-cdx-results-2020-2026-u-z.txt"),
        ],
        help="Huff CDX txt files to balance across equally.",
    )
    parser.add_argument(
        "--onion-output",
        default=str(base_dir / "extra_onion_article_urls.json"),
        help="Balanced Onion output JSON path.",
    )
    parser.add_argument(
        "--huff-output",
        default=str(base_dir / "extra_huff_article_urls.json"),
        help="Balanced Huff output JSON path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    onion_files = [Path(p) for p in args.onion_cdx]
    huff_files = [Path(p) for p in args.huff_cdx]
    onion_output = Path(args.onion_output)
    huff_output = Path(args.huff_output)

    onion_combined = {}
    onion_stats = []
    for path in onion_files:
        rows_by_article, stats = load_cdx_file(path, "onion")
        onion_stats.append((path, stats))
        for article_id, row in rows_by_article.items():
            prev = onion_combined.get(article_id)
            if prev is None or row["wayback_timestamp"] > prev["wayback_timestamp"] or (
                row["wayback_timestamp"] == prev["wayback_timestamp"] and row["article_link"] < prev["article_link"]
            ):
                onion_combined[article_id] = row

    huff_by_file = []
    for path in huff_files:
        rows_by_article, stats = load_cdx_file(path, "huff")
        huff_by_file.append((path, rows_by_article, stats))

    min_huff_count = min(len(rows_by_article) for _, rows_by_article, _ in huff_by_file)
    onion_capacity = len(onion_combined) // len(huff_by_file)
    max_k = min(min_huff_count, onion_capacity)

    sorted_huff_rows_by_file = [sort_rows(list(rows_by_article.values())) for _, rows_by_article, _ in huff_by_file]
    selected_huff_by_file = None
    k = max_k
    while k > 0:
        selected_huff_by_file = select_balanced_huff_rows(sorted_huff_rows_by_file, k)
        if selected_huff_by_file is not None:
            break
        k -= 1

    if selected_huff_by_file is None:
        raise RuntimeError("Unable to build a globally unique balanced Huff selection")

    selected_huff = [row for rows in selected_huff_by_file for row in rows]
    selected_onion = sort_rows(list(onion_combined.values()))[: len(selected_huff)]

    for row in selected_huff:
        row.pop("article_id", None)
    for row in selected_onion:
        row.pop("article_id", None)

    onion_output.write_text(json.dumps(selected_onion, ensure_ascii=False, indent=2), encoding="utf-8")
    huff_output.write_text(json.dumps(selected_huff, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ONION] files={len(onion_files)} unique_articles={len(onion_combined)}")
    for path, stats in onion_stats:
        print(
            f"[ONION-FILE] path={path.name} lines_read={stats['lines_read']} parse_fail={stats['parse_fail']} "
            f"non_target_rows={stats['non_target_rows']} invalid_article_rows={stats['invalid_article_rows']} "
            f"unique_articles={stats['unique_articles']}"
        )

    print(f"[HUFF] files={len(huff_files)} min_unique_articles_per_file={min_huff_count}")
    for path, rows_by_article, stats in huff_by_file:
        print(
            f"[HUFF-FILE] path={path.name} lines_read={stats['lines_read']} parse_fail={stats['parse_fail']} "
            f"non_target_rows={stats['non_target_rows']} invalid_article_rows={stats['invalid_article_rows']} "
            f"unique_articles={len(rows_by_article)} selected={k}"
        )

    print(f"[BALANCE] k_per_huff_file={k}")
    print(f"[BALANCE] huff_total={len(selected_huff)}")
    print(f"[BALANCE] onion_total={len(selected_onion)}")
    print(f"[OUTPUT] onion_output={onion_output.resolve()}")
    print(f"[OUTPUT] huff_output={huff_output.resolve()}")


if __name__ == "__main__":
    main()
