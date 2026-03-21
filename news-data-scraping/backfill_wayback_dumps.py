import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from urllib.parse import quote, urlsplit

from extract_contextual_features import CDX_URL, normalize_dataset_url, safe_get, url_variants
from url_matching import clean_by_source, host_no_www, is_huff_host, is_onion_host


LOG_LOCK = threading.Lock()

SOURCE_CONFIG = {
    "onion": {
        "dump_name": "onion-wayback-urls.json",
        "host_check": is_onion_host,
    },
    "huff": {
        "dump_name": "huff-wayback-urls.json",
        "host_check": is_huff_host,
    },
}


def log(message: str):
    with LOG_LOCK:
        print(message, flush=True)


def source_for_url(article_link: str):
    try:
        host = host_no_www(urlsplit(article_link).netloc)
    except Exception:
        return None
    if is_onion_host(host):
        return "onion"
    if is_huff_host(host):
        return "huff"
    return None


def load_dataset_urls(dataset_path: Path):
    dataset_urls = {"onion": set(), "huff": set()}
    invalid_rows = 0
    total_rows = 0

    with dataset_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_number, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            total_rows += 1
            try:
                obj = json.loads(raw)
            except Exception:
                invalid_rows += 1
                log(f"[DATASET][WARN] line={line_number} invalid JSON; skipping")
                continue

            article_link = str(obj.get("article_link", "")).strip()
            if not article_link:
                invalid_rows += 1
                log(f"[DATASET][WARN] line={line_number} missing article_link; skipping")
                continue

            source = source_for_url(article_link)
            if source in dataset_urls:
                dataset_urls[source].add(article_link)

    return dataset_urls, total_rows, invalid_rows


def load_dump_rows(path: Path):
    if not path.exists():
        return [], {}, 0, 0

    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError(f"Dump at {path} must be a JSON list")

    kept = []
    by_article_link = {}
    invalid_rows = 0
    duplicate_rows = 0

    for idx, row in enumerate(data, start=1):
        if not isinstance(row, dict):
            invalid_rows += 1
            log(f"[DUMP][WARN] path={path.name} row={idx} is not an object; skipping")
            continue

        article_link = str(row.get("article_link", "")).strip()
        wayback_url = str(row.get("wayback_url", "")).strip()
        if not article_link or not wayback_url:
            invalid_rows += 1
            log(f"[DUMP][WARN] path={path.name} row={idx} missing article_link/wayback_url; skipping")
            continue

        if article_link in by_article_link:
            duplicate_rows += 1
            continue

        normalized = {"article_link": article_link, "wayback_url": wayback_url}
        by_article_link[article_link] = normalized
        kept.append(normalized)

    return kept, by_article_link, invalid_rows, duplicate_rows


def build_candidate_urls(article_link: str, source: str):
    candidates = []
    seen = set()

    normalized = normalize_dataset_url(article_link)
    clean_link = clean_by_source(article_link, source)

    for base in [article_link, normalized, clean_link]:
        if not isinstance(base, str) or not base.strip():
            continue
        for variant in url_variants(base):
            if variant not in seen:
                seen.add(variant)
                candidates.append(variant)

    return candidates, clean_link


def query_latest_snapshot(candidate_url: str, from_date: str, to_date: str, retries: int, backoff_factor: float):
    params = [
        ("url", candidate_url),
        ("output", "json"),
        ("fl", "timestamp,original,statuscode,mimetype"),
        ("filter", "statuscode:200"),
        ("filter", "mimetype:text/html"),
        ("collapse", "urlkey"),
        ("from", from_date),
        ("to", to_date),
    ]
    response = safe_get(
        CDX_URL,
        params=params,
        timeout=(10, 30),
        retries=retries,
        backoff_factor=backoff_factor,
    )
    data = response.json()
    if not isinstance(data, list) or len(data) <= 1:
        return None

    row = data[-1]
    timestamp = (row[0] if len(row) > 0 else None) or None
    original = (row[1] if len(row) > 1 else candidate_url) or candidate_url
    status = (row[2] if len(row) > 2 else "200") or "200"
    mimetype = (row[3] if len(row) > 3 else "text/html") or "text/html"
    if not timestamp:
        return None

    return {
        "timestamp": timestamp,
        "original": original,
        "status": status,
        "mimetype": mimetype,
        "wayback_url": f"https://web.archive.org/web/{timestamp}/{quote(original, safe=':/?&=%#')}",
    }


def backfill_single_url(
    article_link: str,
    source: str,
    source_index: int,
    source_total: int,
    from_date: str,
    to_date: str,
    retries: int,
    backoff_factor: float,
    max_attempts_per_url: int,
    retry_backoff_sec: float,
):
    started_at = time.time()
    candidates, clean_link = build_candidate_urls(article_link, source)
    had_request_error = False
    last_error = None
    if not candidates:
        return {
            "article_link": article_link,
            "status": "miss",
            "error": "no_candidates",
            "wayback_url": None,
            "timestamp": None,
            "clean_link": clean_link,
        }

    for candidate_number, candidate in enumerate(candidates, start=1):
        for attempt in range(1, max_attempts_per_url + 1):
            try:
                match = query_latest_snapshot(
                    candidate_url=candidate,
                    from_date=from_date,
                    to_date=to_date,
                    retries=retries,
                    backoff_factor=backoff_factor,
                )
                if match:
                    return {
                        "article_link": article_link,
                        "status": "found",
                        "error": None,
                        "wayback_url": match["wayback_url"],
                        "timestamp": match["timestamp"],
                        "clean_link": clean_link,
                    }

                break
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
                had_request_error = True
                last_error = err
                if attempt < max_attempts_per_url:
                    time.sleep(retry_backoff_sec * attempt)
                else:
                    break

    return {
        "article_link": article_link,
        "status": "miss",
        "error": last_error if had_request_error else "no_snapshot",
        "wayback_url": None,
        "timestamp": None,
        "clean_link": clean_link,
    }


def save_dump(path: Path, rows: list[dict[str, str]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def process_source(
    source: str,
    dataset_urls: set[str],
    dump_path: Path,
    dry_run: bool,
    workers: int,
    retries: int,
    backoff_factor: float,
    max_attempts_per_url: int,
    retry_backoff_sec: float,
    from_date: str,
    to_date: str,
    progress_every: int,
    save_every_hits: int,
):
    start = time.time()
    _, existing_map, invalid_rows, duplicate_rows = load_dump_rows(dump_path)
    existing_count = len(existing_map)
    missing_urls = sorted(url for url in dataset_urls if url not in existing_map)

    log(
        f"[AUDIT] source={source} dataset_urls={len(dataset_urls)} existing_rows={existing_count} "
        f"invalid_rows={invalid_rows} duplicate_rows={duplicate_rows} missing_urls={len(missing_urls)} "
        f"dump_path={dump_path.resolve()}"
    )

    if not missing_urls:
        log(f"[SUMMARY] source={source} nothing to backfill")
        return {
            "source": source,
            "added": 0,
            "missing": 0,
            "errors": 0,
            "duplicates_removed": duplicate_rows,
            "final_rows": existing_count,
            "unresolved_examples": [],
        }

    found = 0
    misses = 0
    errors = 0
    merged = dict(existing_map)
    pending_rows = []
    unresolved_examples = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                backfill_single_url,
                article_link=url,
                source=source,
                source_index=index,
                source_total=len(missing_urls),
                from_date=from_date,
                to_date=to_date,
                retries=retries,
                backoff_factor=backoff_factor,
                max_attempts_per_url=max_attempts_per_url,
                retry_backoff_sec=retry_backoff_sec,
            ): url
            for index, url in enumerate(missing_urls, start=1)
        }

        processed = 0
        for future in as_completed(futures):
            result = future.result()
            processed += 1

            if result["status"] == "found":
                found += 1
                pending_rows.append(
                    {"article_link": result["article_link"], "wayback_url": result["wayback_url"]}
                )
            else:
                misses += 1
                if len(unresolved_examples) < 10:
                    unresolved_examples.append(result["article_link"])
                if result["error"] and result["error"] != "no_snapshot":
                    errors += 1

            if processed % progress_every == 0 or processed == len(missing_urls):
                log(
                    f"[PROGRESS] source={source} completed={processed}/{len(missing_urls)} "
                    f"hits={found} misses={misses} errors={errors}"
                )

            if not dry_run and save_every_hits > 0 and len(pending_rows) >= save_every_hits:
                for row in pending_rows:
                    merged.setdefault(row["article_link"], row)
                final_rows = sorted(merged.values(), key=lambda row: row["article_link"])
                save_dump(dump_path, final_rows)
                log(
                    f"[WRITE][CHECKPOINT] source={source} checkpoint_hits={len(pending_rows)} "
                    f"total_added_so_far={len(merged) - existing_count} final_rows={len(final_rows)} "
                    f"dump_path={dump_path.resolve()}"
                )
                pending_rows = []

    for row in pending_rows:
        merged.setdefault(row["article_link"], row)

    final_rows = sorted(merged.values(), key=lambda row: row["article_link"])
    added = len(final_rows) - existing_count

    if dry_run:
        log(
            f"[WRITE][DRY-RUN] source={source} rows_to_add={added} "
            f"final_rows={len(final_rows)} dump_path={dump_path.resolve()}"
        )
    else:
        if pending_rows or added == 0:
            save_dump(dump_path, final_rows)
            log(
                f"[WRITE][DONE] source={source} rows_added={added} "
                f"final_rows={len(final_rows)} dump_path={dump_path.resolve()}"
            )
        else:
            log(
                f"[WRITE][DONE] source={source} rows_added={added} "
                f"final_rows={len(final_rows)} dump_path={dump_path.resolve()} "
                f"(already checkpointed)"
            )

    elapsed = time.time() - start
    log(
        f"[SUMMARY] source={source} hits={found} misses={misses} errors={errors} "
        f"added={added} final_rows={len(final_rows)} elapsed_s={elapsed:.2f}"
    )

    return {
        "source": source,
        "added": added,
        "missing": len(missing_urls) - found,
        "errors": errors,
        "duplicates_removed": duplicate_rows,
        "final_rows": len(final_rows),
        "unresolved_examples": unresolved_examples,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Backfill missing article_link -> wayback_url mappings by querying the live CDX API."
    )
    default_to_date = date.today().strftime("%Y%m%d")
    base_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--dataset",
        default=str(base_dir.parent / "archive" / "Sarcasm_Headlines_Dataset_v2.json"),
        help="Path to dataset JSONL containing article_link.",
    )
    parser.add_argument(
        "--onion-dump",
        default=str(base_dir / SOURCE_CONFIG["onion"]["dump_name"]),
        help="Path to the Onion wayback dump JSON.",
    )
    parser.add_argument(
        "--huff-dump",
        default=str(base_dir / SOURCE_CONFIG["huff"]["dump_name"]),
        help="Path to the Huff wayback dump JSON.",
    )
    parser.add_argument(
        "--update-target",
        choices=["onion", "huff", "both"],
        default="both",
        help="Which dump(s) to update.",
    )
    parser.add_argument("--workers", type=int, default=8, help="Concurrent workers for CDX lookups.")
    parser.add_argument("--retries", type=int, default=3, help="HTTP retries passed to the shared CDX session.")
    parser.add_argument("--backoff-factor", type=float, default=1.0, help="HTTP backoff factor.")
    parser.add_argument(
        "--max-attempts-per-url",
        type=int,
        default=3,
        help="Maximum retry attempts for each candidate URL query.",
    )
    parser.add_argument(
        "--retry-backoff-sec",
        type=float,
        default=1.0,
        help="Base backoff in seconds between per-URL retry attempts.",
    )
    parser.add_argument(
        "--from-date",
        default="20000101",
        help="CDX from date in YYYYMMDD.",
    )
    parser.add_argument(
        "--to-date",
        default=default_to_date,
        help="CDX to date in YYYYMMDD. Defaults to today's date.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print a progress rollup every N completed URLs.",
    )
    parser.add_argument(
        "--save-every-hits",
        type=int,
        default=100,
        help="Persist the dump after every N successful matches.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and log matches without rewriting the dump files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_path = Path(args.dataset)
    onion_dump_path = Path(args.onion_dump)
    huff_dump_path = Path(args.huff_dump)

    target_sources = ["onion", "huff"] if args.update_target == "both" else [args.update_target]

    log(
        f"[START] dataset={dataset_path.resolve()} update_target={args.update_target} "
        f"workers={args.workers} retries={args.retries} backoff_factor={args.backoff_factor} "
        f"max_attempts_per_url={args.max_attempts_per_url} retry_backoff_sec={args.retry_backoff_sec} "
        f"from_date={args.from_date} to_date={args.to_date} dry_run={args.dry_run}"
    )
    log(
        f"[START] onion_dump={onion_dump_path.resolve()} "
        f"huff_dump={huff_dump_path.resolve()} cdx_url={CDX_URL}"
    )

    dataset_urls, total_rows, invalid_rows = load_dataset_urls(dataset_path)
    log(
        f"[DATASET] total_rows={total_rows} invalid_rows={invalid_rows} "
        f"onion_urls={len(dataset_urls['onion'])} huff_urls={len(dataset_urls['huff'])}"
    )

    dump_paths = {
        "onion": onion_dump_path,
        "huff": huff_dump_path,
    }
    summaries = []

    for source in target_sources:
        summaries.append(
            process_source(
                source=source,
                dataset_urls=dataset_urls[source],
                dump_path=dump_paths[source],
                dry_run=args.dry_run,
                workers=args.workers,
                retries=args.retries,
                backoff_factor=args.backoff_factor,
                max_attempts_per_url=args.max_attempts_per_url,
                retry_backoff_sec=args.retry_backoff_sec,
                from_date=args.from_date,
                to_date=args.to_date,
                progress_every=args.progress_every,
                save_every_hits=args.save_every_hits,
            )
        )

    total_added = sum(item["added"] for item in summaries)
    total_missing = sum(item["missing"] for item in summaries)
    total_errors = sum(item["errors"] for item in summaries)
    total_dupes = sum(item["duplicates_removed"] for item in summaries)

    log(
        f"[DONE] sources={target_sources} total_added={total_added} total_unresolved={total_missing} "
        f"total_errors={total_errors} duplicates_removed={total_dupes}"
    )


if __name__ == "__main__":
    main()
