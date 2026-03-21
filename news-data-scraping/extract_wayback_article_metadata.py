import argparse
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit

from bs4 import BeautifulSoup
from requests import HTTPError

from extract_contextual_features import get_session
from url_matching import host_no_www, is_huff_host, is_onion_host


LOG_LOCK = threading.Lock()

OUTPUT_FIELDS = [
    "source",
    "article_link",
    "wayback_url",
    "wayback_timestamp",
    "article_section",
    "news_type",
    "keywords",
    "description",
    "author_description",
    "author_type",
    "author_name",
    "published_date",
    "fetch_error",
    "parse_error",
]

JSON_LD_TYPE_KEYS = {"@type", "type"}
WAYBACK_TS_RE = re.compile(r"/web/(\d{8,14})/")


def log(message: str):
    with LOG_LOCK:
        print(message, flush=True)


class RateLimitError(Exception):
    def __init__(self, row: dict, status_code: int, response_excerpt: str):
        super().__init__(f"Wayback rate limit detected: HTTP {status_code}")
        self.row = row
        self.status_code = status_code
        self.response_excerpt = response_excerpt


RATE_LIMIT_BODY_PATTERNS = [
    "too many requests",
    "rate limit",
    "rate-limited",
    "please try again later",
    "request rate",
    "temporarily blocked",
]


def clean_text(value):
    if value is None:
        return None
    text = " ".join(str(value).split()).strip()
    return text or None


def dedupe_keep_order(values):
    out = []
    seen = set()
    for value in values:
        cleaned = clean_text(value)
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    return out


def load_dump_rows(path: Path, source: str):
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError(f"Dump at {path} must be a JSON list")

    rows = []
    seen_article_links = set()
    duplicate_rows = 0
    invalid_rows = 0

    for idx, row in enumerate(data, start=1):
        if not isinstance(row, dict):
            invalid_rows += 1
            continue

        article_link = clean_text(row.get("article_link"))
        wayback_url = clean_text(row.get("wayback_url"))
        if not article_link or not wayback_url:
            invalid_rows += 1
            continue

        if article_link in seen_article_links:
            duplicate_rows += 1
            continue

        seen_article_links.add(article_link)
        rows.append(
            {
                "source": source,
                "article_link": article_link,
                "wayback_url": wayback_url,
                "wayback_timestamp": extract_wayback_timestamp(wayback_url),
            }
        )

    log(
        f"[LOAD] source={source} rows={len(rows)} duplicate_rows={duplicate_rows} "
        f"invalid_rows={invalid_rows} path={path.resolve()}"
    )
    return rows


def extract_wayback_timestamp(wayback_url: str):
    if not isinstance(wayback_url, str):
        return None
    match = WAYBACK_TS_RE.search(wayback_url)
    if not match:
        return None
    return match.group(1)


def detect_source_from_article_link(article_link: str):
    try:
        host = host_no_www(urlsplit(article_link).netloc)
    except Exception:
        return None
    if is_onion_host(host):
        return "onion"
    if is_huff_host(host):
        return "huff"
    return None


def load_input_rows(onion_dump: Path, huff_dump: Path, input_target: str, limit: int | None):
    rows = []
    if input_target in ("onion", "both"):
        rows.extend(load_dump_rows(onion_dump, "onion"))
    if input_target in ("huff", "both"):
        rows.extend(load_dump_rows(huff_dump, "huff"))
    if limit is not None:
        rows = rows[:limit]
    log(f"[INPUT] target={input_target} rows={len(rows)} limit={limit}")
    return rows


def load_completed_ids(output_jsonl: Path):
    if not output_jsonl.exists():
        return set()
    completed = set()
    with output_jsonl.open("r", encoding="utf-8", errors="ignore") as f:
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
                completed.add(article_link)
    return completed


def meta_content(soup: BeautifulSoup, attr_name: str, attr_value: str):
    tag = soup.find("meta", attrs={attr_name: attr_value})
    if tag and tag.get("content"):
        return clean_text(tag.get("content"))
    return None


def meta_contents(soup: BeautifulSoup, attr_name: str, attr_value: str):
    values = []
    for tag in soup.find_all("meta", attrs={attr_name: attr_value}):
        content = clean_text(tag.get("content"))
        if content:
            values.append(content)
    return dedupe_keep_order(values)


def parse_json_ld_blocks(soup: BeautifulSoup):
    blocks = []
    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = tag.string or tag.get_text(strip=True)
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except Exception:
            continue
        blocks.extend(flatten_json_ld(parsed))
    return blocks


def flatten_json_ld(node):
    out = []
    if isinstance(node, list):
        for item in node:
            out.extend(flatten_json_ld(item))
        return out
    if isinstance(node, dict):
        if "@graph" in node and isinstance(node["@graph"], list):
            for item in node["@graph"]:
                out.extend(flatten_json_ld(item))
        out.append(node)
    return out


def json_ld_types(node):
    raw_type = None
    for key in JSON_LD_TYPE_KEYS:
        if key in node:
            raw_type = node[key]
            break
    if raw_type is None:
        return []
    if isinstance(raw_type, list):
        return [clean_text(x) for x in raw_type if clean_text(x)]
    cleaned = clean_text(raw_type)
    return [cleaned] if cleaned else []


def find_primary_article_json_ld(blocks):
    article_like = {"NewsArticle", "Article", "Report", "BlogPosting", "SatiricalArticle"}
    for block in blocks:
        if any(t in article_like for t in json_ld_types(block)):
            return block
    return blocks[0] if blocks else {}


def json_ld_value(node, *keys):
    current = node
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def normalize_keywords(value):
    if value is None:
        return []
    if isinstance(value, list):
        values = value
    elif isinstance(value, str):
        if "," in value:
            values = [part.strip() for part in value.split(",")]
        else:
            values = [value]
    else:
        values = [str(value)]
    return dedupe_keep_order(values)


def extract_author_nodes(article_block):
    if not isinstance(article_block, dict):
        return []
    author = article_block.get("author")
    if author is None:
        return []
    if isinstance(author, list):
        return author
    return [author]


def extract_author_name(article_block, soup: BeautifulSoup):
    names = []
    for author_node in extract_author_nodes(article_block):
        if isinstance(author_node, dict):
            names.append(author_node.get("name"))
        else:
            names.append(author_node)

    names.extend(meta_contents(soup, "name", "author"))
    names.extend(meta_contents(soup, "property", "author"))

    for selector in [
        "[rel='author']",
        "[itemprop='author']",
        ".byline a",
        ".author a",
        ".author-name",
        ".byline",
    ]:
        for node in soup.select(selector):
            names.append(node.get_text(" ", strip=True))

    cleaned = dedupe_keep_order(names)
    if not cleaned:
        return None
    return cleaned[0] if len(cleaned) == 1 else cleaned


def extract_author_type(article_block):
    types = []
    for author_node in extract_author_nodes(article_block):
        if isinstance(author_node, dict):
            types.extend(json_ld_types(author_node))
    cleaned = dedupe_keep_order(types)
    if not cleaned:
        return None
    return cleaned[0] if len(cleaned) == 1 else cleaned


def extract_author_description(article_block, soup: BeautifulSoup):
    descriptions = []
    for author_node in extract_author_nodes(article_block):
        if isinstance(author_node, dict):
            descriptions.append(author_node.get("description"))

    for selector in [
        ".author-bio",
        ".author__bio",
        ".author-card__bio",
        ".bio",
        "[data-testid='author-bio']",
    ]:
        for node in soup.select(selector):
            descriptions.append(node.get_text(" ", strip=True))

    cleaned = dedupe_keep_order(descriptions)
    if not cleaned:
        return None
    return cleaned[0] if len(cleaned) == 1 else cleaned


def extract_article_section(article_block, soup: BeautifulSoup, source: str):
    values = []
    values.append(article_block.get("articleSection") if isinstance(article_block, dict) else None)
    values.extend(meta_contents(soup, "property", "article:section"))
    values.extend(meta_contents(soup, "name", "article.section"))

    for selector in [
        "[data-testid='rubric']",
        "[data-vars-ga-category]",
        ".rubric",
        ".section",
        ".entry-section",
        ".breadcrumbs a",
        "nav[aria-label='breadcrumb'] a",
    ]:
        for node in soup.select(selector):
            values.append(node.get_text(" ", strip=True))

    cleaned = dedupe_keep_order(values)
    if not cleaned:
        return None

    if source == "onion":
        preferred = [value for value in cleaned if "the onion" not in value.casefold()]
        cleaned = preferred or cleaned

    return cleaned[0]


def extract_news_type(article_block, soup: BeautifulSoup):
    values = []
    if isinstance(article_block, dict):
        values.extend(json_ld_types(article_block))
        values.append(article_block.get("additionalType"))
    values.extend(meta_contents(soup, "property", "og:type"))
    values.extend(meta_contents(soup, "property", "article:content_tier"))
    values.extend(meta_contents(soup, "name", "parsely-type"))

    cleaned = dedupe_keep_order(values)
    return cleaned[0] if cleaned else None


def extract_description(article_block, soup: BeautifulSoup):
    values = []
    if isinstance(article_block, dict):
        values.append(article_block.get("description"))
    values.extend(meta_contents(soup, "name", "description"))
    values.extend(meta_contents(soup, "property", "og:description"))
    values.extend(meta_contents(soup, "name", "twitter:description"))
    cleaned = dedupe_keep_order(values)
    return cleaned[0] if cleaned else None


def extract_keywords(article_block, soup: BeautifulSoup):
    values = []
    if isinstance(article_block, dict):
        values.extend(normalize_keywords(article_block.get("keywords")))
    values.extend(normalize_keywords(meta_content(soup, "name", "keywords")))

    for selector in [
        "[rel='tag']",
        ".tags a",
        ".tag-list a",
        ".topics a",
        "[data-testid='topic-link']",
    ]:
        for node in soup.select(selector):
            values.append(node.get_text(" ", strip=True))

    return dedupe_keep_order(values)


def extract_published_date(article_block, soup: BeautifulSoup):
    values = []
    if isinstance(article_block, dict):
        values.append(article_block.get("datePublished"))
        values.append(article_block.get("dateCreated"))
    values.extend(meta_contents(soup, "property", "article:published_time"))
    values.extend(meta_contents(soup, "name", "pubdate"))
    values.extend(meta_contents(soup, "name", "parsely-pub-date"))

    for node in soup.find_all("time"):
        values.append(node.get("datetime"))
        values.append(node.get_text(" ", strip=True))

    cleaned = dedupe_keep_order(values)
    return cleaned[0] if cleaned else None


def remove_wayback_noise(soup: BeautifulSoup):
    for selector in [
        "#wm-ipp-base",
        "#wm-ipp",
        ".wb-autocomplete-suggestions",
        "script[src*='archive.org/includes/analytics']",
    ]:
        for node in soup.select(selector):
            node.decompose()


def extract_metadata_from_html(html: str, source: str):
    soup = BeautifulSoup(html, "lxml")
    remove_wayback_noise(soup)
    blocks = parse_json_ld_blocks(soup)
    article_block = find_primary_article_json_ld(blocks)

    return {
        "article_section": extract_article_section(article_block, soup, source),
        "news_type": extract_news_type(article_block, soup),
        "keywords": extract_keywords(article_block, soup),
        "description": extract_description(article_block, soup),
        "author_description": extract_author_description(article_block, soup),
        "author_type": extract_author_type(article_block),
        "author_name": extract_author_name(article_block, soup),
        "published_date": extract_published_date(article_block, soup),
        "parse_error": None,
    }


def empty_output_row(base_row):
    row = {field: None for field in OUTPUT_FIELDS}
    row["keywords"] = []
    row.update(
        {
            "source": base_row["source"],
            "article_link": base_row["article_link"],
            "wayback_url": base_row["wayback_url"],
            "wayback_timestamp": base_row.get("wayback_timestamp"),
            "fetch_error": None,
            "parse_error": None,
        }
    )
    return row


def fetch_wayback_response(wayback_url: str, retries: int, backoff_factor: float, request_wait_sec: float):
    if request_wait_sec > 0:
        time.sleep(request_wait_sec)
    response = get_session(retries=retries, backoff_factor=backoff_factor).get(
        wayback_url,
        timeout=(5, 40),
    )
    return response


def response_excerpt(text: str, limit: int = 300):
    return clean_text((text or "")[:limit]) or "<empty body>"


def detect_rate_limit_response(response):
    body_excerpt = response_excerpt(getattr(response, "text", ""))
    haystacks = [
        body_excerpt.casefold(),
        clean_text(getattr(response, "reason", "")) or "",
        clean_text(getattr(response, "url", "")) or "",
    ]
    joined = " ".join(h.casefold() for h in haystacks if h)
    if response.status_code == 429:
        return body_excerpt
    for pattern in RATE_LIMIT_BODY_PATTERNS:
        if pattern in joined:
            return body_excerpt
    return None


def process_row(row, retries: int, backoff_factor: float, request_wait_sec: float):
    out = empty_output_row(row)
    try:
        response = fetch_wayback_response(
            row["wayback_url"],
            retries=retries,
            backoff_factor=backoff_factor,
            request_wait_sec=request_wait_sec,
        )
    except Exception as exc:
        out["fetch_error"] = f"{type(exc).__name__}: {exc}"
        return out

    rate_limit_excerpt = detect_rate_limit_response(response)
    if rate_limit_excerpt is not None:
        raise RateLimitError(out, response.status_code, rate_limit_excerpt)

    try:
        response.raise_for_status()
    except HTTPError as exc:
        out["fetch_error"] = f"HTTPError: status={response.status_code} detail={exc}"
        return out

    try:
        extracted = extract_metadata_from_html(response.text, row["source"])
        out.update(extracted)
    except Exception as exc:
        out["parse_error"] = f"{type(exc).__name__}: {exc}"
    return out


def write_jsonl_rows(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_json_list(path: Path):
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return []
    return data if isinstance(data, list) else []


def append_bugged_rows(path: Path, rows: list[dict]):
    if not rows:
        return
    existing = load_json_list(path)
    seen_links = {
        clean_text(item.get("article_link"))
        for item in existing
        if isinstance(item, dict) and clean_text(item.get("article_link"))
    }
    merged = list(existing)
    added = 0
    for row in rows:
        article_link = clean_text(row.get("article_link"))
        if not article_link or article_link in seen_links:
            continue
        merged.append(row)
        seen_links.add(article_link)
        added += 1
    if added == 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[BUGGED] appended={added} total={len(merged)} output={path.resolve()}")


def checkpoint_rows(
    output_jsonl: Path,
    pending_writes: list[dict],
    written_rows: int,
    total_rows: int,
    fetch_errors: int,
    parse_errors: int,
):
    if not pending_writes:
        return written_rows, []
    write_jsonl_rows(output_jsonl, pending_writes)
    written_rows += len(pending_writes)
    log(
        f"[CHECKPOINT] processed={written_rows}/{total_rows} "
        f"fetch_errors={fetch_errors} parse_errors={parse_errors} output={output_jsonl.resolve()}"
    )
    return written_rows, []


def handle_rate_limit_probe(
    remaining_rows: list[dict],
    blocked_article_link: str,
    retries: int,
    backoff_factor: float,
    request_wait_sec: float,
    save_every_n: int,
    pending_writes: list[dict],
    written_rows: int,
    total_rows: int,
    fetch_errors: int,
    parse_errors: int,
    output_jsonl: Path,
    bugged_rows_json: Path,
):
    blocked_index = next(
        (idx for idx, row in enumerate(remaining_rows) if row["article_link"] == blocked_article_link),
        None,
    )
    if blocked_index is None:
        return set(), pending_writes, written_rows, fetch_errors, parse_errors, True

    skipped_links = {blocked_article_link}
    bugged_rows = []
    probe_rows = remaining_rows[blocked_index + 1 : blocked_index + 4]
    probe_429_count = 0
    probe_success_count = 0
    blocked_row = remaining_rows[blocked_index]
    bugged_rows.append(
        {
            "source": blocked_row["source"],
            "article_link": blocked_row["article_link"],
            "wayback_url": blocked_row["wayback_url"],
            "wayback_timestamp": blocked_row.get("wayback_timestamp"),
            "fetch_error": "HTTP 429 Too Many Requests",
        }
    )

    log(
        f"[RATE-LIMIT] skipping_article_link={blocked_article_link} "
        f"and probing_next={len(probe_rows)} url(s)"
    )

    for probe_row in probe_rows:
        try:
            result = process_row(probe_row, retries, backoff_factor, request_wait_sec)
        except RateLimitError as exc:
            probe_429_count += 1
            skipped_links.add(probe_row["article_link"])
            bugged_rows.append(
                {
                    "source": probe_row["source"],
                    "article_link": probe_row["article_link"],
                    "wayback_url": probe_row["wayback_url"],
                    "wayback_timestamp": probe_row.get("wayback_timestamp"),
                    "fetch_error": "HTTP 429 Too Many Requests",
                }
            )
            log(
                f"[RATE-LIMIT-PROBE] article_link={probe_row['article_link']} "
                f"status={exc.status_code} response_excerpt={exc.response_excerpt}"
            )
            continue

        skipped_links.add(probe_row["article_link"])
        probe_success_count += 1
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

    should_stop = len(probe_rows) > 0 and probe_success_count == 0 and probe_429_count == len(probe_rows)
    append_bugged_rows(bugged_rows_json, bugged_rows)
    return skipped_links, pending_writes, written_rows, fetch_errors, parse_errors, should_stop


def process_rows(
    rows: list[dict],
    output_jsonl: Path,
    bugged_rows_json: Path,
    workers: int,
    retries: int,
    backoff_factor: float,
    save_every_n: int,
    rate_limit_wait_sec: float,
    request_wait_sec: float,
):
    remaining_rows = list(rows)
    pending_writes = []
    total_rows = len(rows)
    written_rows = 0
    fetch_errors = 0
    parse_errors = 0
    attempts = 0

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

                    if written_rows + len(pending_writes) == total_rows or (written_rows + len(pending_writes)) % 25 == 0:
                        log(
                            f"[PROGRESS] processed={written_rows + len(pending_writes)}/{total_rows} "
                            f"fetch_errors={fetch_errors} parse_errors={parse_errors}"
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

        if rate_limit_hit and should_stop:
            log(
                f"[DONE] stopping_after_consecutive_429s processed={written_rows}/{total_rows} "
                f"remaining={len(remaining_rows)} output={output_jsonl.resolve()}"
            )
            return

    log(
        f"[DONE] processed={written_rows} fetch_errors={fetch_errors} parse_errors={parse_errors} "
        f"output={output_jsonl.resolve()}"
    )


def parse_args():
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Extract article metadata fields from Wayback URLs stored in dump files."
    )
    parser.add_argument(
        "--onion-dump",
        default=str(base_dir / "onion-wayback-urls.json"),
        help="Path to Onion dump file.",
    )
    parser.add_argument(
        "--huff-dump",
        default=str(base_dir / "huff-wayback-urls.json"),
        help="Path to Huff dump file.",
    )
    parser.add_argument(
        "--input-target",
        choices=["onion", "huff", "both"],
        default="both",
        help="Which dump(s) to read.",
    )
    parser.add_argument(
        "--output-jsonl",
        default=str(base_dir / f"wayback_article_metadata.jsonl"),
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--bugged-rows-json",
        default=str(base_dir / "wayback_urls_bugged_429.json"),
        help="Output JSON path for rows skipped due to repeated HTTP 429 responses.",
    )
    parser.add_argument("--workers", type=int, default=8, help="Concurrent fetch workers.")
    parser.add_argument("--retries", type=int, default=2, help="HTTP retries for archived page fetches.")
    parser.add_argument("--backoff-factor", type=float, default=1.0, help="HTTP retry backoff factor.")
    parser.add_argument("--save-every-n", type=int, default=20, help="Checkpoint every N processed rows.")
    parser.add_argument(
        "--request-wait-sec",
        type=float,
        default=0.0,
        help="Seconds to wait before each archived page request.",
    )
    parser.add_argument(
        "--rate-limit-wait-sec",
        type=float,
        default=60,
        help="Seconds to wait before automatically retrying unfinished rows after a rate limit.",
    )
    parser.add_argument("--resume", action="store_true", help="Skip rows already present in output JSONL.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N rows.")
    return parser.parse_args()


def main():
    args = parse_args()
    onion_dump = Path(args.onion_dump)
    huff_dump = Path(args.huff_dump)
    output_jsonl = Path(args.output_jsonl)
    bugged_rows_json = Path(args.bugged_rows_json)

    rows = load_input_rows(
        onion_dump=onion_dump,
        huff_dump=huff_dump,
        input_target=args.input_target,
        limit=args.limit,
    )

    if args.resume:
        completed_ids = load_completed_ids(output_jsonl)
        bugged_ids = {
            clean_text(item.get("article_link"))
            for item in load_json_list(bugged_rows_json)
            if isinstance(item, dict) and clean_text(item.get("article_link"))
        }
        before = len(rows)
        rows = [
            row
            for row in rows
            if row["article_link"] not in completed_ids and row["article_link"] not in bugged_ids
        ]
        log(
            f"[RESUME] skipped={before - len(rows)} remaining={len(rows)} output={output_jsonl.resolve()}"
        )

    if not rows:
        log("[DONE] no rows to process")
        return

    process_rows(
        rows=rows,
        output_jsonl=output_jsonl,
        bugged_rows_json=bugged_rows_json,
        workers=args.workers,
        retries=args.retries,
        backoff_factor=args.backoff_factor,
        save_every_n=args.save_every_n,
        rate_limit_wait_sec=args.rate_limit_wait_sec,
        request_wait_sec=args.request_wait_sec,
    )


if __name__ == "__main__":
    main()
