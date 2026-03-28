import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from backfill_wayback_dumps import build_candidate_urls, query_latest_snapshot
from extract_wayback_article_metadata import (
    RateLimitError,
    detect_source_from_article_link,
    handle_rate_limit_probe,
    clean_text,
    extract_wayback_timestamp,
    load_json_list,
    load_successful_ids,
    log,
    process_row,
    checkpoint_rows,
    write_jsonl_rows,
)
from extract_contextual_features import safe_get


WAYBACK_AVAILABILITY_URL = "https://archive.org/wayback/available"


def load_input_rows(path: Path):
    rows = load_json_list(path)
    if not isinstance(rows, list):
        return []

    deduped = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        article_link = clean_text(row.get("article_link"))
        source = clean_text(row.get("source")) or detect_source_from_article_link(article_link)
        if not article_link or not source or article_link in seen:
            continue
        seen.add(article_link)
        deduped.append(
            {
                "source": source,
                "article_link": article_link,
                "wayback_url": clean_text(row.get("wayback_url")),
                "wayback_timestamp": clean_text(row.get("wayback_timestamp")),
            }
        )
    return deduped


def load_combined_successful_ids(paths: list[Path]):
    combined = set()
    for path in paths:
        combined.update(load_successful_ids(path))
    return combined


def load_recorded_ids(path: Path):
    if not path.exists():
        return set()
    recorded = set()
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            article_link = clean_text(obj.get("article_link"))
            if article_link:
                recorded.add(article_link)
    return recorded


def load_successful_lookup_rows(path: Path):
    if not path.exists():
        return []
    rows = []
    seen = set()
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            article_link = clean_text(obj.get("article_link"))
            fetch_error = clean_text(obj.get("fetch_error"))
            parse_error = clean_text(obj.get("parse_error"))
            if not article_link or fetch_error or parse_error or article_link in seen:
                continue
            seen.add(article_link)
            rows.append(
                {
                    "source": clean_text(obj.get("source")),
                    "article_link": article_link,
                    "wayback_url": clean_text(obj.get("wayback_url")),
                    "wayback_timestamp": clean_text(obj.get("wayback_timestamp")),
                }
            )
    return rows


def huff_url_variants(url: str):
    cleaned = clean_text(url)
    if not cleaned:
        return []
    try:
        parsed = urlsplit(cleaned)
    except Exception:
        return [cleaned]

    host = parsed.netloc
    path = parsed.path or "/"
    query = parsed.query
    candidates = [cleaned]

    if host == "www.huffingtonpost.com":
        candidates.append(urlunsplit((parsed.scheme, "www.huffpost.com", path, query, "")))

    if path.endswith(".html"):
        stripped_path = path[:-5]
        candidates.append(urlunsplit((parsed.scheme, host, stripped_path, query, "")))
        if host == "www.huffingtonpost.com":
            candidates.append(urlunsplit((parsed.scheme, "www.huffpost.com", stripped_path, query, "")))

    deduped = []
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return deduped


def build_robust_candidate_urls(row: dict):
    source = row["source"]
    candidates, _ = build_candidate_urls(row["article_link"], source)
    if source != "huff":
        return candidates

    extra_bases = huff_url_variants(row["article_link"])

    seen = set(candidates)
    robust_candidates = list(candidates)
    for candidate in extra_bases:
        if candidate in seen:
            continue
        seen.add(candidate)
        robust_candidates.append(candidate)
    return robust_candidates


def query_availability_snapshot(candidate_url: str, from_date: str, to_date: str, retries: int, backoff_factor: float):
    response = safe_get(
        WAYBACK_AVAILABILITY_URL,
        params={"url": candidate_url, "timestamp": to_date},
        timeout=(10, 30),
        retries=retries,
        backoff_factor=backoff_factor,
    )
    data = response.json()
    snapshots = data.get("archived_snapshots", {}) if isinstance(data, dict) else {}
    closest = snapshots.get("closest") if isinstance(snapshots, dict) else None
    if not isinstance(closest, dict):
        return None

    available = closest.get("available")
    status = str(closest.get("status") or "")
    archive_url = clean_text(closest.get("url"))
    timestamp = clean_text(closest.get("timestamp"))
    if not available or status != "200" or not archive_url or not timestamp:
        return None
    if timestamp < from_date or timestamp > to_date:
        return None

    return {
        "timestamp": timestamp,
        "wayback_url": archive_url,
    }


def resolve_fresh_wayback_row(
    row: dict,
    from_date: str,
    to_date: str,
    retries: int,
    backoff_factor: float,
    max_attempts_per_url: int,
):
    article_link = row["article_link"]
    source = row["source"]
    candidates = build_robust_candidate_urls(row)
    if not candidates:
        out = dict(row)
        out["fetch_error"] = "No candidate URLs available for CDX lookup"
        out["parse_error"] = None
        return out

    attempted_errors = []
    for candidate in candidates:
        try:
            availability_match = query_availability_snapshot(
                candidate_url=candidate,
                from_date=from_date,
                to_date=to_date,
                retries=retries,
                backoff_factor=backoff_factor,
            )
            if availability_match:
                return {
                    "source": source,
                    "article_link": article_link,
                    "wayback_url": availability_match["wayback_url"],
                    "wayback_timestamp": availability_match["timestamp"],
                }
        except Exception as exc:
            attempted_errors.append(
                f"{candidate} -> availability {type(exc).__name__}: {exc}"
            )

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
                        "source": source,
                        "article_link": article_link,
                        "wayback_url": match["wayback_url"],
                        "wayback_timestamp": match["timestamp"] or extract_wayback_timestamp(match["wayback_url"]),
                    }
                break
            except Exception as exc:
                attempted_errors.append(
                    f"{candidate} -> attempt={attempt} {type(exc).__name__}: {exc}"
                )
                if attempt == max_attempts_per_url:
                    break

    out = dict(row)
    out["fetch_error"] = " | ".join(attempted_errors) if attempted_errors else "No fresh Wayback snapshot found"
    out["parse_error"] = None
    return out


def checkpoint_lookup_rows(output_jsonl: Path, pending_rows: list[dict], processed: int, total: int):
    if not pending_rows:
        return 0
    write_jsonl_rows(output_jsonl, pending_rows)
    log(f"[CHECKPOINT] processed={processed}/{total} wrote={len(pending_rows)} output={output_jsonl.resolve()}")
    return len(pending_rows)


def process_streaming_rows(
    rows: list[dict],
    lookup_output_jsonl: Path,
    output_jsonl: Path,
    bugged_rows_json: Path,
    workers: int,
    retries: int,
    backoff_factor: float,
    save_every_n: int,
    request_wait_sec: float,
    from_date: str,
    to_date: str,
    max_attempts_per_url: int,
    metadata_resume_ids: set[str],
    lookup_resume: bool,
    stop_after_consecutive_429s: bool,
):
    total = len(rows)
    processed = 0
    pending_rows = []
    resolved_rows = []

    if lookup_resume:
        lookup_recorded_ids = load_recorded_ids(lookup_output_jsonl)
        before = len(rows)
        rows = [row for row in rows if row["article_link"] not in lookup_recorded_ids]
        total = len(rows)
        log(
            f"[LOOKUP-RESUME] skipped={before - total} remaining={total} "
            f"lookup_output={lookup_output_jsonl.resolve()}"
        )

    if rows:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    resolve_fresh_wayback_row,
                    row,
                    from_date,
                    to_date,
                    retries,
                    backoff_factor,
                    max_attempts_per_url,
                ): row
                for row in rows
            }

            for future in as_completed(futures):
                base_row = futures[future]
                processed += 1
                try:
                    refreshed_row = future.result()
                except Exception as exc:
                    refreshed_row = {
                        "source": base_row["source"],
                        "article_link": base_row["article_link"],
                        "wayback_url": base_row.get("wayback_url"),
                        "wayback_timestamp": base_row.get("wayback_timestamp"),
                        "fetch_error": f"{type(exc).__name__}: {exc}",
                        "parse_error": None,
                    }

                pending_rows.append(refreshed_row)

                if len(pending_rows) >= save_every_n:
                    checkpoint_lookup_rows(lookup_output_jsonl, pending_rows, processed, total)
                    pending_rows = []

                if processed % 25 == 0 or processed == total:
                    log(f"[PROGRESS] processed={processed}/{total} output={lookup_output_jsonl.resolve()}")

        if pending_rows:
            checkpoint_lookup_rows(lookup_output_jsonl, pending_rows, processed, total)

    resolved_rows = load_successful_lookup_rows(lookup_output_jsonl)

    if metadata_resume_ids:
        before = len(resolved_rows)
        resolved_rows = [row for row in resolved_rows if row["article_link"] not in metadata_resume_ids]
        log(
            f"[METADATA-RESUME] skipped={before - len(resolved_rows)} remaining={len(resolved_rows)} "
            f"lookup_output={lookup_output_jsonl.resolve()}"
        )

    if not resolved_rows:
        log(f"[DONE] no refreshed rows available for metadata extraction output={output_jsonl.resolve()}")
        return

    pending_writes = []
    total_rows = len(resolved_rows)
    written_rows = 0
    fetch_errors = 0
    parse_errors = 0
    attempts = 0
    remaining_rows = list(resolved_rows)

    while remaining_rows:
        attempts += 1
        round_completed = set()
        rate_limit_hit = False
        should_stop = False

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_row, row, retries, backoff_factor, request_wait_sec): row["article_link"]
                for row in remaining_rows
            }
            try:
                for future in as_completed(futures):
                    article_link = futures[future]
                    try:
                        result = future.result()
                    except RateLimitError as exc:
                        rate_limit_hit = True
                        log(
                            f"[RATE-LIMIT] attempt={attempts} completed={written_rows}/{total_rows} "
                            f"status={exc.status_code} article_link={exc.row['article_link']} "
                            f"response_excerpt={exc.response_excerpt}"
                        )
                        for pending_future in futures:
                            if pending_future is not future:
                                pending_future.cancel()
                        skipped_links, pending_writes, written_rows, fetch_errors, parse_errors, should_stop = (
                            handle_rate_limit_probe(
                                remaining_rows=remaining_rows,
                                blocked_article_link=exc.row["article_link"],
                                retries=retries,
                                backoff_factor=backoff_factor,
                                request_wait_sec=request_wait_sec,
                                save_every_n=save_every_n,
                                pending_writes=pending_writes,
                                written_rows=written_rows,
                                total_rows=total_rows,
                                fetch_errors=fetch_errors,
                                parse_errors=parse_errors,
                                output_jsonl=output_jsonl,
                                bugged_rows_json=bugged_rows_json,
                            )
                        )
                        round_completed.update(skipped_links)
                        break

                    round_completed.add(article_link)
                    if result.get("fetch_error"):
                        fetch_errors += 1
                    if result.get("parse_error"):
                        parse_errors += 1
                    pending_writes.append(result)

                    if len(pending_writes) >= save_every_n:
                        written_rows, pending_writes = checkpoint_rows(
                            output_jsonl=output_jsonl,
                            pending_writes=pending_writes,
                            written_rows=written_rows,
                            total_rows=total_rows,
                            fetch_errors=fetch_errors,
                            parse_errors=parse_errors,
                        )
            finally:
                executor.shutdown(wait=False, cancel_futures=True)

        remaining_rows = [row for row in remaining_rows if row["article_link"] not in round_completed]

        if pending_writes and (rate_limit_hit or not remaining_rows):
            written_rows, pending_writes = checkpoint_rows(
                output_jsonl=output_jsonl,
                pending_writes=pending_writes,
                written_rows=written_rows,
                total_rows=total_rows,
                fetch_errors=fetch_errors,
                parse_errors=parse_errors,
            )

        if rate_limit_hit and should_stop and stop_after_consecutive_429s:
            log(
                f"[DONE] stopping_after_consecutive_429s processed={written_rows}/{total_rows} "
                f"remaining={len(remaining_rows)} output={output_jsonl.resolve()}"
            )
            return
        if rate_limit_hit and should_stop and not stop_after_consecutive_429s:
            log(
                f"[RATE-LIMIT] continuing_after_consecutive_429s processed={written_rows}/{total_rows} "
                f"remaining={len(remaining_rows)} bugged_output={bugged_rows_json.resolve()}"
            )

    log(
        f"[DONE] processed={written_rows} fetch_errors={fetch_errors} parse_errors={parse_errors} "
        f"output={output_jsonl.resolve()}"
    )


def parse_args():
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Re-resolve fresh Wayback URLs for input rows and extract article metadata again."
    )
    parser.add_argument(
        "--input-json",
        default=str(base_dir / "wayback_urls_bugged_429.json"),
        help="Input JSON path containing article_link and wayback_url rows to rerun.",
    )
    parser.add_argument(
        "--output-jsonl",
        default=str(base_dir / "wayback_article_metadata_bugged_rerun.jsonl"),
        help="Output JSONL path for rerun metadata extraction.",
    )
    parser.add_argument(
        "--lookup-output-jsonl",
        default=str(base_dir / "wayback_article_metadata_part2_new.jsonl"),
        help="Intermediate JSONL path for fresh Wayback URL lookup results.",
    )
    parser.add_argument(
        "--rerun-bugged-rows-json",
        default=str(base_dir / "wayback_urls_bugged_429_rerun.json"),
        help="Output JSON path for rows that still hit HTTP 429 during rerun extraction.",
    )
    parser.add_argument(
        "--already-success-jsonl",
        action="append",
        default=[],
        help="Additional JSONL file whose successful rows should be skipped before rerun. Can be passed multiple times.",
    )
    parser.add_argument("--workers", type=int, default=4, help="Concurrent workers.")
    parser.add_argument("--retries", type=int, default=2, help="HTTP retries for CDX and archived page fetches.")
    parser.add_argument("--backoff-factor", type=float, default=1.0, help="HTTP retry backoff factor.")
    parser.add_argument("--save-every-n", type=int, default=20, help="Checkpoint every N processed rows.")
    parser.add_argument(
        "--request-wait-sec",
        type=float,
        default=0.0,
        help="Seconds to wait before each archived page request.",
    )
    parser.add_argument(
        "--from-date",
        default="20000101",
        help="CDX from date in YYYYMMDD for fresh Wayback lookup.",
    )
    parser.add_argument(
        "--to-date",
        default=date.today().strftime("%Y%m%d"),
        help="CDX to date in YYYYMMDD for fresh Wayback lookup.",
    )
    parser.add_argument(
        "--max-attempts-per-url",
        type=int,
        default=2,
        help="Maximum retry attempts for each CDX candidate URL.",
    )
    parser.add_argument(
        "--stop-after-consecutive-429s",
        action="store_true",
        help="Stop early if the blocked row and all probe rows hit consecutive HTTP 429 responses.",
    )
    parser.add_argument("--resume", action="store_true", help="Skip rows already successfully rerun in the output.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N bugged rows.")
    return parser.parse_args()


def main():
    args = parse_args()
    input_json = Path(args.input_json)
    output_jsonl = Path(args.output_jsonl)
    lookup_output_jsonl = Path(args.lookup_output_jsonl)
    rerun_bugged_rows_json = Path(args.rerun_bugged_rows_json)

    rows = load_input_rows(input_json)
    if args.limit is not None:
        rows = rows[: args.limit]

    completed_ids = set()
    if args.resume:
        success_paths = [output_jsonl] + [Path(path) for path in args.already_success_jsonl]
        completed_ids = load_combined_successful_ids(success_paths)
        before = len(rows)
        rows = [row for row in rows if row["article_link"] not in completed_ids]
        log(f"[RESUME] skipped={before - len(rows)} remaining={len(rows)} output={output_jsonl.resolve()}")

    if not rows:
        log("[DONE] no bugged rows to rerun")
        return

    process_streaming_rows(
        rows=rows,
        lookup_output_jsonl=lookup_output_jsonl,
        output_jsonl=output_jsonl,
        bugged_rows_json=rerun_bugged_rows_json,
        workers=args.workers,
        retries=args.retries,
        backoff_factor=args.backoff_factor,
        save_every_n=args.save_every_n,
        request_wait_sec=args.request_wait_sec,
        from_date=args.from_date,
        to_date=args.to_date,
        max_attempts_per_url=args.max_attempts_per_url,
        metadata_resume_ids=completed_ids,
        lookup_resume=args.resume,
        stop_after_consecutive_429s=args.stop_after_consecutive_429s,
    )


if __name__ == "__main__":
    main()
