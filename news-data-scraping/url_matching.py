import argparse
import json
import re
from pathlib import Path
from urllib.parse import parse_qsl, unquote, urlsplit, urlunsplit


ONION_ID_RE = re.compile(r"-(\d{9,12})(?:/amp)?(?:[/?#].*)?$")
ONION_ID_ANYWHERE_RE = re.compile(r"-(\d{9,12})(?:[/?#].*)?$")
HUFF_HEX24_RE = re.compile(r"(?:_us_)([0-9a-fA-F]{24})(?:[/?#_.-]|$)")
HUFF_HEX24_FALLBACK_RE = re.compile(r"(?:^|[/?#_.=-])([0-9a-fA-F]{24})(?:[/?#_.-]|$)")
HUFF_NUMERIC_RE = re.compile(r"(?:^|[-_/])([0-9]{6,})(?:[/?#_.-]|$)")


def host_no_www(host: str) -> str:
    h = (host or "").lower()
    if h.startswith("www."):
        h = h[4:]
    return h


def is_onion_host(host: str) -> bool:
    h = host_no_www(host)
    return h == "theonion.com" or h.endswith(".theonion.com")


def is_huff_host(host: str) -> bool:
    h = host_no_www(host)
    return (
        h == "huffingtonpost.com"
        or h.endswith(".huffingtonpost.com")
        or h == "huffpost.com"
        or h.endswith(".huffpost.com")
    )


def parse_cdx_line(line: str):
    s = line.strip()
    if not s:
        return None
    parts = s.split(maxsplit=1)
    if len(parts) != 2:
        return None
    ts, original = parts
    if not re.fullmatch(r"\d{8,14}", ts):
        return None
    return ts, original.strip()


def decode_for_scan(url: str):
    out = url
    for _ in range(3):
        new = unquote(out)
        if new == out:
            break
        out = new
    return out


def extract_onion_id(value: str):
    if not isinstance(value, str):
        return None
    v = value.strip()
    if not v:
        return None
    m = ONION_ID_RE.search(v)
    if m:
        return m.group(1)
    decoded = decode_for_scan(v)
    m2 = ONION_ID_ANYWHERE_RE.search(decoded)
    if m2:
        return m2.group(1)
    return None


def extract_huff_id(value: str):
    if not isinstance(value, str):
        return None
    decoded = decode_for_scan(value)

    m = HUFF_HEX24_RE.search(decoded)
    if m:
        return m.group(1).lower()

    m2 = HUFF_HEX24_FALLBACK_RE.search(decoded)
    if m2:
        return m2.group(1).lower()

    # Fallback for numeric IDs when no hex id is present
    m3 = HUFF_NUMERIC_RE.search(decoded)
    if m3:
        return f"num:{m3.group(1)}"

    return None


def clean_onion_dataset_article_link(url: str):
    if not isinstance(url, str):
        return None
    s = url.strip()
    if not s:
        return None

    try:
        p = urlsplit(s)
    except Exception:
        return None

    if not p.netloc or not is_onion_host(p.netloc):
        return None

    host = host_no_www(p.netloc)
    path = p.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path[:-1]
    if path.endswith("/amp"):
        path = path[:-4]
    if not path.startswith("/"):
        path = "/" + path

    article_id = extract_onion_id(path)
    if not article_id:
        return None

    return urlunsplit(("https", host, path, "", ""))


def normalize_huff_path(path: str) -> str:
    p = path or "/"
    if not p.startswith("/"):
        p = "/" + p
    while "//" in p:
        p = p.replace("//", "/")
    if len(p) > 1 and p.endswith("/"):
        p = p[:-1]
    if p.endswith("/amp"):
        p = p[:-4]
    return p


def clean_huff_dataset_article_link(url: str):
    if not isinstance(url, str):
        return None
    s = url.strip()
    if not s:
        return None

    try:
        p = urlsplit(s)
    except Exception:
        return None

    if not p.netloc or not is_huff_host(p.netloc):
        return None

    host = "www.huffingtonpost.com"
    path = normalize_huff_path(p.path)
    return urlunsplit(("https", host, path, "", ""))


def normalize_huff_exact(url: str):
    """Host-normalized exact key retaining query for exact-first match."""
    if not isinstance(url, str):
        return None
    s = url.strip()
    if not s:
        return None
    try:
        p = urlsplit(s)
    except Exception:
        return None

    if not p.netloc or not is_huff_host(p.netloc):
        return None

    host = "www.huffingtonpost.com"
    path = normalize_huff_path(p.path)
    # canonicalize query ordering for stable exact key comparison
    query_pairs = parse_qsl(p.query, keep_blank_values=True)
    query_pairs.sort()
    query = "&".join([f"{k}={v}" for k, v in query_pairs])
    return urlunsplit(("https", host, path, query, ""))


def clean_by_source(url: str, source: str):
    if source == "onion":
        return clean_onion_dataset_article_link(url)
    return clean_huff_dataset_article_link(url)


def is_target_host(host: str, source: str):
    if source == "onion":
        return is_onion_host(host)
    return is_huff_host(host)


def extract_id_for_source(value: str, source: str):
    if source == "onion":
        return extract_onion_id(value)
    return extract_huff_id(value)


def build_cdx_maps(cdx_path: Path, source: str):
    by_id = {}
    by_clean_key = {}
    by_exact = {}
    stats = {
        "lines_read": 0,
        "parse_fail": 0,
        "urls_parsed": 0,
        "non_target_rows": 0,
        "id_extracted_rows": 0,
        "clean_key_rows": 0,
        "exact_rows": 0,
    }

    with cdx_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stats["lines_read"] += 1
            parsed = parse_cdx_line(line)
            if not parsed:
                stats["parse_fail"] += 1
                continue

            ts, original = parsed
            stats["urls_parsed"] += 1

            try:
                host = urlsplit(original).netloc
            except Exception:
                host = ""

            if not is_target_host(host, source):
                stats["non_target_rows"] += 1
                continue

            if source == "huff":
                exact_key = normalize_huff_exact(original)
                if exact_key:
                    prev = by_exact.get(exact_key)
                    if prev is None or ts > prev:
                        by_exact[exact_key] = ts
                    stats["exact_rows"] += 1

            clean_key = clean_by_source(original, source)
            if clean_key:
                prev = by_clean_key.get(clean_key)
                if prev is None or ts > prev:
                    by_clean_key[clean_key] = ts
                stats["clean_key_rows"] += 1

            aid = extract_id_for_source(original, source)
            if aid:
                prev2 = by_id.get(aid)
                if prev2 is None or ts > prev2:
                    by_id[aid] = ts
                stats["id_extracted_rows"] += 1

    return by_exact, by_id, by_clean_key, stats


def load_existing_matches(output_path: Path):
    if not output_path.exists():
        return {}, []
    try:
        data = json.loads(output_path.read_text(encoding="utf-8"))
    except Exception:
        return {}, []
    if not isinstance(data, list):
        return {}, []

    by_article_link = {}
    kept = []
    for row in data:
        if not isinstance(row, dict):
            continue
        article_link = str(row.get("article_link", "")).strip()
        wayback_url = str(row.get("wayback_url", "")).strip()
        if not article_link or not wayback_url:
            continue
        if article_link in by_article_link:
            continue
        normalized = {"article_link": article_link, "wayback_url": wayback_url}
        by_article_link[article_link] = normalized
        kept.append(normalized)
    return by_article_link, kept


def match_dataset(
    dataset_path: Path,
    by_exact: dict,
    by_id: dict,
    by_clean_key: dict,
    source: str,
    existing_matches_by_article_link: dict | None = None,
):
    out = []
    seen_article_links = set()
    stats = {
        "total_rows": 0,
        "target_rows": 0,
        "skipped_existing": 0,
        "invalid_clean": 0,
        "cleaned_rows": 0,
        "matched_by_exact": 0,
        "matched_by_id": 0,
        "matched_by_key": 0,
        "unmatched_target_rows": 0,
    }
    reason_counts = {
        "invalid_clean": 0,
        "exact_not_in_cdx": 0,
        "no_id_in_article_link": 0,
        "id_not_in_cdx": 0,
        "clean_key_not_in_cdx": 0,
    }
    unmatched_examples = []

    with dataset_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue

            stats["total_rows"] += 1
            article_link = str(obj.get("article_link", "")).strip()
            if not article_link:
                continue

            try:
                host = urlsplit(article_link).netloc
            except Exception:
                host = ""
            if not is_target_host(host, source):
                continue
            stats["target_rows"] += 1

            if existing_matches_by_article_link and article_link in existing_matches_by_article_link:
                stats["skipped_existing"] += 1
                continue
            if article_link in seen_article_links:
                continue
            seen_article_links.add(article_link)

            clean_link = clean_by_source(article_link, source)
            if not clean_link:
                stats["invalid_clean"] += 1
                reason_counts["invalid_clean"] += 1
                if len(unmatched_examples) < 5:
                    unmatched_examples.append({"article_link": article_link, "reason": "invalid_clean"})
                continue
            stats["cleaned_rows"] += 1

            ts = None

            # Priority 1: exact match (Huff requested behavior)
            if source == "huff":
                exact_key = normalize_huff_exact(article_link)
                if exact_key and exact_key in by_exact:
                    ts = by_exact[exact_key]
                    stats["matched_by_exact"] += 1
                else:
                    reason_counts["exact_not_in_cdx"] += 1

            # Priority 2: id match
            if ts is None:
                aid = extract_id_for_source(clean_link, source)
                if aid and aid in by_id:
                    ts = by_id[aid]
                    stats["matched_by_id"] += 1
                else:
                    if not aid:
                        reason_counts["no_id_in_article_link"] += 1
                    else:
                        reason_counts["id_not_in_cdx"] += 1

            # Priority 3: clean key
            if ts is None:
                ts = by_clean_key.get(clean_link)
                if ts:
                    stats["matched_by_key"] += 1
                else:
                    reason_counts["clean_key_not_in_cdx"] += 1

            if ts:
                wayback_url = f"https://web.archive.org/web/{ts}/{clean_link}"
                out.append({"article_link": article_link, "wayback_url": wayback_url})
            else:
                stats["unmatched_target_rows"] += 1
                if len(unmatched_examples) < 5:
                    unmatched_examples.append({"article_link": article_link, "reason": "no_exact_id_or_key_match"})

    stats["matched_rows"] = len(out)
    return out, stats, reason_counts, unmatched_examples


def main():
    parser = argparse.ArgumentParser(
        description="Match dataset article links to Wayback URLs using CDX timestamps."
    )
    parser.add_argument(
        "--source",
        choices=["onion", "huff"],
        default="onion",
        help="Which source host family to process.",
    )
    parser.add_argument(
        "--cdx-results",
        default="cdx-results-2010-onwards-new.txt",
        help="Path to CDX results file with lines: <timestamp> <original_url>.",
    )
    parser.add_argument(
        "--dataset",
        default=str(Path(__file__).resolve().parent.parent / "Sarcasm_Headlines_Dataset_v2.json"),
        help="Path to dataset JSONL containing article_link.",
    )
    parser.add_argument(
        "--output",
        default="article_link_wayback_matches_robust.json",
        help="Output JSON path (matched rows only).",
    )
    parser.add_argument(
        "--skip-existing-matches",
        action="store_true",
        help="If set, load --output and skip article_link values already matched there.",
    )
    args = parser.parse_args()

    cdx_path = Path(args.cdx_results)
    dataset_path = Path(args.dataset)
    output_path = Path(args.output)
    existing_by_article_link = {}
    existing_rows = []
    if args.skip_existing_matches:
        existing_by_article_link, existing_rows = load_existing_matches(output_path)
        print(f"[COUNT] existing_output_rows_loaded={len(existing_rows)}")

    print(f"[PHASE] 1/3 Build CDX maps for source={args.source}")
    by_exact, by_id, by_clean_key, cdx_stats = build_cdx_maps(cdx_path, args.source)
    print(f"[COUNT] cdx_lines_read={cdx_stats['lines_read']}")
    print(f"[COUNT] cdx_parse_fail={cdx_stats['parse_fail']}")
    print(f"[COUNT] cdx_urls_parsed={cdx_stats['urls_parsed']}")
    print(f"[COUNT] cdx_non_target_rows={cdx_stats['non_target_rows']}")
    print(f"[COUNT] cdx_exact_rows={cdx_stats['exact_rows']}")
    print(f"[COUNT] cdx_id_extracted_rows={cdx_stats['id_extracted_rows']}")
    print(f"[COUNT] cdx_clean_key_rows={cdx_stats['clean_key_rows']}")
    print(f"[COUNT] cdx_unique_exact_keys={len(by_exact)}")
    print(f"[COUNT] cdx_unique_ids={len(by_id)}")
    print(f"[COUNT] cdx_unique_clean_keys={len(by_clean_key)}")

    print(f"[PHASE] 2/3 Match dataset links for source={args.source}")
    matches, match_stats, reason_counts, unmatched_examples = match_dataset(
        dataset_path,
        by_exact,
        by_id,
        by_clean_key,
        args.source,
        existing_matches_by_article_link=existing_by_article_link,
    )
    all_matches = existing_rows + matches
    print(f"[COUNT] dataset_total_rows={match_stats['total_rows']}")
    print(f"[COUNT] dataset_target_rows={match_stats['target_rows']}")
    print(f"[COUNT] skipped_existing={match_stats['skipped_existing']}")
    print(f"[COUNT] dataset_cleaned_rows={match_stats['cleaned_rows']}")
    print(f"[COUNT] dataset_invalid_clean={match_stats['invalid_clean']}")
    print(f"[COUNT] matched_rows={match_stats['matched_rows']}")
    print(f"[COUNT] matched_by_exact={match_stats['matched_by_exact']}")
    print(f"[COUNT] matched_by_id={match_stats['matched_by_id']}")
    print(f"[COUNT] matched_by_key={match_stats['matched_by_key']}")
    print(f"[COUNT] unmatched_target_rows={match_stats['unmatched_target_rows']}")
    print(f"[COUNT] reason_invalid_clean={reason_counts['invalid_clean']}")
    print(f"[COUNT] reason_exact_not_in_cdx={reason_counts['exact_not_in_cdx']}")
    print(f"[COUNT] reason_no_id_in_article_link={reason_counts['no_id_in_article_link']}")
    print(f"[COUNT] reason_id_not_in_cdx={reason_counts['id_not_in_cdx']}")
    print(f"[COUNT] reason_clean_key_not_in_cdx={reason_counts['clean_key_not_in_cdx']}")

    print("[PHASE] 3/3 Write matched-only output")
    output_path.write_text(json.dumps(all_matches, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[COUNT] new_output_rows={len(matches)}")
    print(f"[COUNT] output_rows_total={len(all_matches)}")
    print(f"[COUNT] output_path={output_path.resolve()}")

    print("[SAMPLE] first_5_unmatched_target")
    for ex in unmatched_examples[:5]:
        print(f"[SAMPLE] {ex['reason']} :: {ex['article_link']}")


if __name__ == "__main__":
    main()
