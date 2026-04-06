import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from extract_wayback_article_metadata import detect_source_from_article_link, process_row


FIELDS_TO_CHECK = (
    "description",
    "author_description",
    "author_name",
    "article_section",
)

SUSPICIOUS_SUBSTRINGS = (
    "\u7ab6",
    "\u9083",
    "\u2001",
    "\u2101",
)


def safe_print(message: str):
    text = str(message).encode("ascii", "backslashreplace").decode("ascii")
    print(text, flush=True)


def parse_args():
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Rerun metadata extraction for rows whose stored metadata looks mojibake-corrupted."
    )
    parser.add_argument(
        "--input-jsonl",
        nargs="+",
        default=[
            str(base_dir / "wayback_article_metadata.jsonl"),
            str(base_dir / "wayback_article_metadata_part1.jsonl"),
            str(base_dir / "wayback_article_metadata_part2.jsonl"),
            str(base_dir / "wayback_article_metadata_part3.jsonl"),
        ],
        help="Metadata JSONL file(s) to scan and repair.",
    )
    parser.add_argument(
        "--output-suffix",
        default="_rerun_fixed",
        help="Suffix to append to each repaired output filename.",
    )
    parser.add_argument("--workers", type=int, default=4, help="Concurrent rerun workers for corrupted rows.")
    parser.add_argument("--retries", type=int, default=2, help="HTTP retries for Wayback fetches.")
    parser.add_argument("--backoff-factor", type=float, default=1.0, help="HTTP retry backoff factor.")
    parser.add_argument(
        "--request-wait-sec",
        type=float,
        default=0.0,
        help="Seconds to wait before each rerun request.",
    )
    parser.add_argument(
        "--save-every-n",
        type=int,
        default=20,
        help="Flush repaired output every N written rows.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from an existing repaired output file.")
    parser.add_argument("--limit", type=int, default=None, help="Only inspect the first N input rows per file.")
    return parser.parse_args()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8-sig", errors="ignore") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            yield json.loads(raw)


def jsonl_line_count(path: Path):
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def looks_corrupted_text(text: str):
    if not isinstance(text, str) or not text:
        return False
    if any(token in text for token in SUSPICIOUS_SUBSTRINGS):
        return True
    return any("\u3400" <= ch <= "\u9fff" for ch in text)


def looks_corrupted_value(value):
    if isinstance(value, str):
        return looks_corrupted_text(value)
    if isinstance(value, list):
        return any(looks_corrupted_value(item) for item in value)
    return False


def row_looks_corrupted(row: dict):
    if row.get("fetch_error") or row.get("parse_error"):
        return False
    return any(looks_corrupted_value(row.get(field)) for field in FIELDS_TO_CHECK)


def output_path_for(input_path: Path, suffix: str):
    return input_path.with_name(f"{input_path.stem}{suffix}{input_path.suffix}")


def write_row(handle, row: dict):
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_base_row(row: dict):
    return {
        "source": row.get("source") or detect_source_from_article_link(row.get("article_link")),
        "article_link": row.get("article_link"),
        "wayback_url": row.get("wayback_url"),
        "wayback_timestamp": row.get("wayback_timestamp"),
    }


def flush_ready_rows(out, ready_rows: dict, next_write_index: int, resume_rows: int, save_every_n: int):
    while next_write_index in ready_rows:
        write_row(out, ready_rows.pop(next_write_index))
        next_write_index += 1
        written = next_write_index - resume_rows
        if written % save_every_n == 0:
            out.flush()
    return next_write_index


def repair_file(
    input_path: Path,
    output_path: Path,
    workers: int,
    retries: int,
    backoff_factor: float,
    request_wait_sec: float,
    save_every_n: int,
    resume: bool,
    limit: int | None,
):
    rows = list(iter_jsonl(input_path))
    if limit is not None:
        rows = rows[:limit]

    resume_rows = jsonl_line_count(output_path) if resume else 0
    mode = "a" if resume_rows else "w"
    processed = 0
    repaired = 0
    rerun_failures = 0
    skipped_clean = 0
    skipped_resume = 0
    corrupted_candidates = 0
    ready_rows = {}
    next_write_index = resume_rows
    last_logged_checkpoint = -1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open(mode, encoding="utf-8") as out:
        future_to_index = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for idx, row in enumerate(rows):
                if idx < resume_rows:
                    skipped_resume += 1
                    continue

                processed += 1
                if row_looks_corrupted(row):
                    corrupted_candidates += 1
                    future = executor.submit(
                        process_row,
                        build_base_row(row),
                        retries,
                        backoff_factor,
                        request_wait_sec,
                    )
                    future_to_index[future] = idx
                else:
                    skipped_clean += 1
                    ready_rows[idx] = row

            next_write_index = flush_ready_rows(
                out=out,
                ready_rows=ready_rows,
                next_write_index=next_write_index,
                resume_rows=resume_rows,
                save_every_n=save_every_n,
            )

            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                original_row = rows[idx]
                try:
                    rerun = future.result()
                except Exception:
                    rerun = None

                if rerun and not rerun.get("fetch_error") and not rerun.get("parse_error"):
                    ready_rows[idx] = rerun
                    repaired += 1
                else:
                    ready_rows[idx] = original_row
                    rerun_failures += 1

                next_write_index = flush_ready_rows(
                    out=out,
                    ready_rows=ready_rows,
                    next_write_index=next_write_index,
                    resume_rows=resume_rows,
                    save_every_n=save_every_n,
                )

                written = next_write_index - resume_rows
                if written and written % save_every_n == 0 and written != last_logged_checkpoint:
                    last_logged_checkpoint = written
                    safe_print(
                        f"[CHECKPOINT] file={input_path.name} written={resume_rows + written} "
                        f"repaired={repaired} rerun_failures={rerun_failures} "
                        f"clean_passthrough={skipped_clean}"
                    )

        next_write_index = flush_ready_rows(
            out=out,
            ready_rows=ready_rows,
            next_write_index=next_write_index,
            resume_rows=resume_rows,
            save_every_n=save_every_n,
        )
        out.flush()

    safe_print(
        f"[DONE] file={input_path.name} output={output_path} resume_skipped={skipped_resume} "
        f"written={processed} corrupted_candidates={corrupted_candidates} repaired={repaired} "
        f"rerun_failures={rerun_failures} clean_passthrough={skipped_clean}"
    )


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

    args = parse_args()
    for raw_path in args.input_jsonl:
        input_path = Path(raw_path)
        output_path = output_path_for(input_path, args.output_suffix)
        repair_file(
            input_path=input_path,
            output_path=output_path,
            workers=args.workers,
            retries=args.retries,
            backoff_factor=args.backoff_factor,
            request_wait_sec=args.request_wait_sec,
            save_every_n=args.save_every_n,
            resume=args.resume,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()
