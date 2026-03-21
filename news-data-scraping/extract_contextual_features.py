import argparse
import csv
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit, quote

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

SNAPSHOT_COLS = [
    "wayback_available",
    "wayback_url",
    "wayback_timestamp",
    "wayback_status",
    "wayback_error",
]

CONTENT_COLS = [
    "archived_title",
    "archived_meta_description",
    "archived_article_text",
    "content_error",
]

ALL_COLS = SNAPSHOT_COLS + CONTENT_COLS
_thread_local = threading.local()

CDX_URL = "https://web.archive.org/cdx/search/cdx"

DOMAIN_HOSTS = [
    "www.huffingtonpost.com",
    "www.theonion.com",
    "local.theonion.com",
    "politics.theonion.com",
    "entertainment.theonion.com",
    "sports.theonion.com",
    "ogn.theonion.com",
]

TARGET_HOST_TOKENS = ("huffingtonpost.com", "theonion.com")


def _save_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=True)
    tmp.replace(path)


def _load_json(path: Path):
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _append_error_log(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["article_link", "error"])
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def build_session(retries: int = 5, backoff_factor: float = 1.0):
    s = requests.Session()
    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=50, pool_maxsize=50)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update({"User-Agent": "Mozilla/5.0"})
    return s


def get_session(retries: int = 5, backoff_factor: float = 1.0):
    if (
        not hasattr(_thread_local, "session")
        or getattr(_thread_local, "retries", None) != retries
        or getattr(_thread_local, "backoff_factor", None) != backoff_factor
    ):
        _thread_local.session = build_session(retries=retries, backoff_factor=backoff_factor)
        _thread_local.retries = retries
        _thread_local.backoff_factor = backoff_factor
    return _thread_local.session


def safe_get(url, params=None, timeout=(5, 25), retries: int = 5, backoff_factor: float = 1.0):
    r = get_session(retries=retries, backoff_factor=backoff_factor).get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r


def clean_text(text: str) -> str:
    return " ".join(text.split()) if text else ""


def normalize_dataset_url(url: str):
    if not isinstance(url, str):
        return None
    u = url.strip()
    if not u:
        return None

    bad_prefixes = [
        "https://www.huffingtonpost.comhttp://",
        "https://www.huffingtonpost.comhttps://",
        "http://www.huffingtonpost.comhttp://",
        "http://www.huffingtonpost.comhttps://",
    ]
    for p in bad_prefixes:
        if u.startswith(p):
            u = u.replace("https://www.huffingtonpost.com", "", 1)
            u = u.replace("http://www.huffingtonpost.com", "", 1)
            break

    return u


def canonical_url_key(url: str):
    if not isinstance(url, str) or not url.strip():
        return None
    try:
        p = urlsplit(url.strip())
    except Exception:
        return None

    if not p.netloc:
        return None

    host = p.netloc.lower()
    path = p.path or "/"
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    query = p.query or ""
    return urlunsplit(("https", host, path, query, ""))


def url_variants(url: str):
    if not isinstance(url, str) or not url.strip():
        return []
    u = url.strip()
    p = urlsplit(u)
    netloc = p.netloc.lower()
    path = p.path or "/"
    variants = set()

    for keep_query in [True, False]:
        q = p.query if keep_query else ""
        for scheme in ["https", "http"]:
            variants.add(urlunsplit((scheme, netloc, path, q, "")))
            alt = (path[:-1] or "/") if path.endswith("/") else (path + "/")
            variants.add(urlunsplit((scheme, netloc, alt, q, "")))

    for v in list(variants):
        pv = urlsplit(v)
        n = pv.netloc
        n2 = n[4:] if n.startswith("www.") else "www." + n
        variants.add(urlunsplit((pv.scheme, n2, pv.path, pv.query, "")))

    return [u] + [x for x in variants if x != u]


def verify_dataset_domains(df: pd.DataFrame, url_col: str = "article_link"):
    raw_hosts = {}
    normalized_hosts = {}
    non_target_examples = []

    for raw in df[url_col].dropna().astype(str):
        try:
            raw_h = urlsplit(raw).netloc.lower()
            raw_hosts[raw_h] = raw_hosts.get(raw_h, 0) + 1
        except Exception:
            pass

        nu = normalize_dataset_url(raw)
        key = canonical_url_key(nu) if nu else None
        if not key:
            continue
        nh = urlsplit(key).netloc.lower()
        normalized_hosts[nh] = normalized_hosts.get(nh, 0) + 1

        if not any(tok in nh for tok in TARGET_HOST_TOKENS) and len(non_target_examples) < 10:
            non_target_examples.append(raw)

    only_target_after_normalization = all(any(tok in h for tok in TARGET_HOST_TOKENS) for h in normalized_hosts)
    return {
        "raw_hosts": dict(sorted(raw_hosts.items(), key=lambda x: x[1], reverse=True)),
        "normalized_hosts": dict(sorted(normalized_hosts.items(), key=lambda x: x[1], reverse=True)),
        "only_target_after_normalization": only_target_after_normalization,
        "non_target_examples": non_target_examples,
    }


def empty_snapshot():
    return {
        "wayback_available": False,
        "wayback_url": None,
        "wayback_timestamp": None,
        "wayback_status": None,
        "wayback_error": None,
    }


def empty_content():
    return {
        "archived_title": None,
        "archived_meta_description": None,
        "archived_article_text": None,
        "content_error": None,
    }


def collect_latest_captures_for_domain_window(
    domain_host: str,
    latest_index: dict,
    all_article_links: dict,
    from_date: str,
    to_date: str,
    max_attempts: int = 3,
    retry_backoff_sec: float = 2.0,
):
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            params = [
                ("url", f"{domain_host}/*"),
                ("output", "txt"),
                ("fl", "timestamp original statuscode mimetype"),
                ("filter", "statuscode:200"),
                ("filter", "mimetype:text/html"),
                ("collapse", "urlkey"),
                ("from", from_date),
                ("to", to_date),
            ]

            r = safe_get(
                CDX_URL,
                params=params,
                timeout=(10, 120),
                retries=3,
                backoff_factor=1.0,
            )

            read_count = 0
            kept_count = 0
            for raw_line in r.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue

                parts = raw_line.strip().split(" ", 3)
                if len(parts) < 2:
                    continue

                ts = parts[0]
                original = parts[1]
                status = parts[2] if len(parts) > 2 else "200"
                if len(ts) < 14:
                    ts = ts.ljust(14, "0")

                read_count += 1
                key = canonical_url_key(original)
                if not key:
                    continue

                host = urlsplit(key).netloc.lower()
                if not any(tok in host for tok in TARGET_HOST_TOKENS):
                    continue

                all_article_links[key] = 1
                cur = latest_index.get(key)
                if (cur is None) or ((ts or "") > (cur.get("wayback_timestamp") or "")):
                    wb_original = quote(original, safe=":/?&=%#")
                    latest_index[key] = {
                        "wayback_available": True,
                        "wayback_url": f"https://web.archive.org/web/{ts}/{wb_original}",
                        "wayback_timestamp": ts,
                        "wayback_status": status,
                        "wayback_error": None,
                    }
                    kept_count += 1

            return read_count, kept_count
        except Exception as e:
            last_err = (
                f"CDX query failed for {domain_host} {from_date}-{to_date} "
                f"(attempt {attempt}/{max_attempts}): {type(e).__name__}: {e}"
            )
        if attempt < max_attempts:
            sleep_s = retry_backoff_sec * attempt
            print(f"[WARN] {last_err}; retrying in {sleep_s:.1f}s")
            time.sleep(sleep_s)

    raise RuntimeError(last_err or f"CDX query failed for {domain_host} {from_date}-{to_date}")


def build_domain_latest_index(
    index_cache_path: Path,
    all_links_cache_path: Path,
    domain_hosts=None,
    rebuild=False,
    start_year: int = 2000,
    end_year: int = 2026,
    max_attempts_per_month: int = 3,
    retry_backoff_sec: float = 2.0,
):
    domain_hosts = domain_hosts or DOMAIN_HOSTS
    if index_cache_path.exists() and all_links_cache_path.exists() and not rebuild:
        cached_index = _load_json(index_cache_path)
        cached_all_links = _load_json(all_links_cache_path)
        if cached_index and cached_all_links:
            print(f"[INDEX] loaded existing latest index with {len(cached_index)} canonical URLs")
            print(f"[INDEX] loaded all article links inventory with {len(cached_all_links)} canonical URLs")
            return cached_index, cached_all_links

    latest_index = {}
    all_article_links = {}
    start_dt = date(start_year, 1, 1)
    end_dt = date(end_year, 12, 31)
    for host in domain_hosts:
        print(f"[INDEX] scanning host: {host}")
        pattern_read = 0
        pattern_updates = 0
        cur_dt = start_dt
        while cur_dt <= end_dt:
            from_date = cur_dt.strftime("%Y%m%d")
            to_date = from_date
            day_label = cur_dt.strftime("%Y-%m-%d")
            print(f"[INDEX]   day={day_label} ({from_date} -> {to_date})")
            try:
                read_count, kept_count = collect_latest_captures_for_domain_window(
                    host,
                    latest_index,
                    all_article_links,
                    from_date=from_date,
                    to_date=to_date,
                    max_attempts=max_attempts_per_month,
                    retry_backoff_sec=retry_backoff_sec,
                )
                pattern_read += read_count
                pattern_updates += kept_count
                print(
                    f"[INDEX]   day={day_label} captures_read={read_count} updates={kept_count} "
                    f"unique_now={len(latest_index)}"
                )
            except Exception as e:
                print(f"[WARN] skipping {host} day={day_label} due to error: {type(e).__name__}: {e}")
            cur_dt += timedelta(days=1)

        print(
            f"[INDEX] host done: captures_read={pattern_read} updates={pattern_updates} "
            f"unique_now={len(latest_index)}"
        )

    _save_json(index_cache_path, latest_index)
    _save_json(all_links_cache_path, all_article_links)
    print(f"[INDEX] saved {len(latest_index)} canonical URLs to {index_cache_path}")
    print(f"[INDEX] saved {len(all_article_links)} canonical URLs to {all_links_cache_path}")
    return latest_index, all_article_links


def extract_content(wayback_url: str, retries: int = 5, backoff_factor: float = 1.0, max_chars: int = 4000):
    out = empty_content()
    if not isinstance(wayback_url, str) or not wayback_url.strip():
        out["content_error"] = "Invalid/empty wayback_url"
        return out

    try:
        r = safe_get(wayback_url, timeout=(5, 40), retries=retries, backoff_factor=backoff_factor)
        soup = BeautifulSoup(r.text, "lxml")
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
            tag.decompose()

        out["archived_title"] = clean_text(soup.title.get_text()) if soup.title else None

        meta_desc = ""
        m = soup.find("meta", attrs={"name": "description"})
        if m and m.get("content"):
            meta_desc = clean_text(m["content"])
        out["archived_meta_description"] = meta_desc or None

        selectors = ["article", "[role='main']", ".article", ".entry-content", ".post-content", ".content", "main"]
        candidates = []
        for sel in selectors:
            for node in soup.select(sel):
                txt = clean_text(node.get_text(" ", strip=True))
                if len(txt) > 200:
                    candidates.append(txt)

        if candidates:
            text = max(candidates, key=len)
        else:
            paras = [clean_text(p.get_text(" ", strip=True)) for p in soup.find_all("p")]
            paras = [p for p in paras if len(p) > 40]
            text = " ".join(paras[:20])

        out["archived_article_text"] = text[:max_chars] if text else None
        return out
    except Exception as e:
        out["content_error"] = f"{type(e).__name__}: {e}"
        return out


def build_snapshot_cache_by_matching(df: pd.DataFrame, latest_index: dict, url_col: str = "article_link"):
    snapshot_cache = {}
    unique_urls = [u for u in df[url_col].dropna().astype(str).unique().tolist() if u.strip()]

    matched = 0
    for raw in unique_urls:
        out = empty_snapshot()
        norm = normalize_dataset_url(raw)
        if not norm:
            out["wayback_error"] = "Invalid/empty URL"
            snapshot_cache[raw] = out
            continue

        keys_to_try = []
        for u in url_variants(norm):
            k = canonical_url_key(u)
            if k:
                keys_to_try.append(k)
        keys_to_try = list(dict.fromkeys(keys_to_try))

        found = None
        for k in keys_to_try:
            if k in latest_index:
                found = latest_index[k]
                break

        if found:
            snapshot_cache[raw] = dict(found)
            matched += 1
        else:
            snapshot_cache[raw] = out

    print(f"[MATCH] unique_urls={len(unique_urls)} matched={matched} unmatched={len(unique_urls) - matched}")
    return snapshot_cache


def find_snapshot_for_url_on_day(
    raw_url: str,
    day_yyyymmdd: str,
    retries: int = 3,
    backoff_factor: float = 1.0,
    max_attempts: int = 3,
    retry_backoff_sec: float = 2.0,
):
    out = empty_snapshot()
    normalized = normalize_dataset_url(raw_url)
    if not normalized:
        out["wayback_error"] = "Invalid/empty URL"
        return out

    candidates = []
    for u in url_variants(normalized):
        k = canonical_url_key(u)
        if k:
            candidates.append(k)
    candidates = list(dict.fromkeys(candidates))

    for candidate in candidates:
        last_err = None
        for attempt in range(1, max_attempts + 1):
            try:
                params = [
                    ("url", candidate),
                    ("output", "json"),
                    ("fl", "timestamp,original,statuscode,mimetype"),
                    ("filter", "statuscode:200"),
                    ("filter", "mimetype:text/html"),
                    ("collapse", "urlkey"),
                    ("from", day_yyyymmdd),
                    ("to", day_yyyymmdd),
                ]
                r = safe_get(
                    CDX_URL,
                    params=params,
                    timeout=(10, 30),
                    retries=retries,
                    backoff_factor=backoff_factor,
                )
                data = r.json()
                if isinstance(data, list) and len(data) > 1:
                    row = data[-1]
                    ts = (row[0] if len(row) > 0 else None) or None
                    original = (row[1] if len(row) > 1 else candidate) or candidate
                    status = (row[2] if len(row) > 2 else "200") or "200"
                    out.update(
                        {
                            "wayback_available": True,
                            "wayback_url": f"https://web.archive.org/web/{ts}/{quote(original, safe=':/?&=%#')}",
                            "wayback_timestamp": ts,
                            "wayback_status": status,
                            "wayback_error": None,
                        }
                    )
                    return out
                break
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                if attempt < max_attempts:
                    sleep_s = retry_backoff_sec * attempt
                    time.sleep(sleep_s)

        if last_err:
            out["wayback_error"] = last_err

    return out


def find_snapshot_for_url_in_range(
    raw_url: str,
    from_yyyymmdd: str,
    to_yyyymmdd: str,
    retries: int = 3,
    backoff_factor: float = 1.0,
    max_attempts: int = 3,
    retry_backoff_sec: float = 2.0,
):
    out = empty_snapshot()
    normalized = normalize_dataset_url(raw_url)
    if not normalized:
        out["wayback_error"] = "Invalid/empty URL"
        return out

    candidates = []
    for u in url_variants(normalized):
        k = canonical_url_key(u)
        if k:
            candidates.append(k)
    candidates = list(dict.fromkeys(candidates))

    for candidate in candidates:
        last_err = None
        for attempt in range(1, max_attempts + 1):
            try:
                params = [
                    ("url", candidate),
                    ("output", "json"),
                    ("fl", "timestamp,original,statuscode,mimetype"),
                    ("filter", "statuscode:200"),
                    ("filter", "mimetype:text/html"),
                    ("collapse", "urlkey"),
                    ("from", from_yyyymmdd),
                    ("to", to_yyyymmdd),
                ]
                r = safe_get(
                    CDX_URL,
                    params=params,
                    timeout=(10, 30),
                    retries=retries,
                    backoff_factor=backoff_factor,
                )
                data = r.json()
                if isinstance(data, list) and len(data) > 1:
                    row = data[-1]  # latest capture in range
                    ts = (row[0] if len(row) > 0 else None) or None
                    original = (row[1] if len(row) > 1 else candidate) or candidate
                    status = (row[2] if len(row) > 2 else "200") or "200"
                    out.update(
                        {
                            "wayback_available": True,
                            "wayback_url": f"https://web.archive.org/web/{ts}/{quote(original, safe=':/?&=%#')}",
                            "wayback_timestamp": ts,
                            "wayback_status": status,
                            "wayback_error": None,
                        }
                    )
                    return out
                break
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                if attempt < max_attempts:
                    sleep_s = retry_backoff_sec * attempt
                    time.sleep(sleep_s)
        if last_err:
            out["wayback_error"] = last_err
    return out


def build_snapshot_cache_direct_day(
    df: pd.DataFrame,
    url_col: str = "article_link",
    day_yyyymmdd: str = "20260115",
    workers_snapshot: int = 16,
    save_every_n_urls: int = 500,
    rerun_errors: bool = True,
    rerun_misses: bool = False,
    snapshot_cache: dict | None = None,
    retries: int = 3,
    backoff_factor: float = 1.0,
    max_attempts_per_url: int = 3,
    retry_backoff_sec: float = 2.0,
    snapshot_cache_path: Path | None = None,
    error_log_path: Path | None = None,
):
    unique_urls = [u for u in df[url_col].dropna().astype(str).unique().tolist() if u.strip()]
    snapshot_cache = snapshot_cache or {}
    hit = miss = err = 0

    todo_urls = []
    for u in unique_urls:
        s = snapshot_cache.get(u)
        if s is None:
            todo_urls.append(u)
        elif rerun_errors and s.get("wayback_error"):
            todo_urls.append(u)
        elif rerun_misses and (not s.get("wayback_available")) and (not s.get("wayback_error")):
            todo_urls.append(u)

    print(
        f"[SNAPSHOT] unique_urls={len(unique_urls)} cache_entries={len(snapshot_cache)} "
        f"todo={len(todo_urls)}"
    )

    for i in range(0, len(todo_urls), save_every_n_urls):
        chunk = todo_urls[i : i + save_every_n_urls]
        chunk_error_rows = []
        with ThreadPoolExecutor(max_workers=workers_snapshot) as ex:
            fut_to_url = {
                ex.submit(
                    find_snapshot_for_url_on_day,
                    u,
                    day_yyyymmdd,
                    retries,
                    backoff_factor,
                    max_attempts_per_url,
                    retry_backoff_sec,
                ): u
                for u in chunk
            }
            pbar = tqdm(total=len(chunk), desc=f"snapshot day={day_yyyymmdd} {i}:{i + len(chunk)}")

            for fut in as_completed(fut_to_url):
                u = fut_to_url[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    res = empty_snapshot()
                    res["wayback_error"] = f"{type(e).__name__}: {e}"

                snapshot_cache[u] = res
                if res.get("wayback_error"):
                    err += 1
                    chunk_error_rows.append({"article_link": u, "error": str(res.get("wayback_error"))})
                elif res.get("wayback_available"):
                    hit += 1
                else:
                    miss += 1

                pbar.update(1)
                pbar.set_postfix(hit=hit, miss=miss, err=err, refresh=False)

            pbar.close()

        if snapshot_cache_path is not None:
            _save_json(snapshot_cache_path, snapshot_cache)
            print(f"[SNAPSHOT SAVE] {snapshot_cache_path} | hit={hit} miss={miss} err={err}")
        if error_log_path is not None:
            _append_error_log(error_log_path, chunk_error_rows)

    print(f"[SNAPSHOT DONE] hit={hit} miss={miss} err={err}")
    return snapshot_cache


def build_snapshot_cache_direct_range(
    df: pd.DataFrame,
    url_col: str = "article_link",
    from_yyyymmdd: str = "20250101",
    to_yyyymmdd: str = "20251231",
    workers_snapshot: int = 16,
    save_every_n_urls: int = 500,
    rerun_errors: bool = True,
    rerun_misses: bool = False,
    snapshot_cache: dict | None = None,
    retries: int = 3,
    backoff_factor: float = 1.0,
    max_attempts_per_url: int = 3,
    retry_backoff_sec: float = 2.0,
    snapshot_cache_path: Path | None = None,
    error_log_path: Path | None = None,
):
    unique_urls = [u for u in df[url_col].dropna().astype(str).unique().tolist() if u.strip()]
    snapshot_cache = snapshot_cache or {}
    hit = miss = err = 0

    todo_urls = []
    for u in unique_urls:
        s = snapshot_cache.get(u)
        if s is None:
            todo_urls.append(u)
        elif rerun_errors and s.get("wayback_error"):
            todo_urls.append(u)
        elif rerun_misses and (not s.get("wayback_available")) and (not s.get("wayback_error")):
            todo_urls.append(u)

    print(
        f"[SNAPSHOT] unique_urls={len(unique_urls)} cache_entries={len(snapshot_cache)} "
        f"todo={len(todo_urls)}"
    )

    for i in range(0, len(todo_urls), save_every_n_urls):
        chunk = todo_urls[i : i + save_every_n_urls]
        chunk_error_rows = []
        with ThreadPoolExecutor(max_workers=workers_snapshot) as ex:
            fut_to_url = {
                ex.submit(
                    find_snapshot_for_url_in_range,
                    u,
                    from_yyyymmdd,
                    to_yyyymmdd,
                    retries,
                    backoff_factor,
                    max_attempts_per_url,
                    retry_backoff_sec,
                ): u
                for u in chunk
            }
            pbar = tqdm(total=len(chunk), desc=f"snapshot range={from_yyyymmdd}-{to_yyyymmdd} {i}:{i + len(chunk)}")

            for fut in as_completed(fut_to_url):
                u = fut_to_url[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    res = empty_snapshot()
                    res["wayback_error"] = f"{type(e).__name__}: {e}"

                snapshot_cache[u] = res
                if res.get("wayback_error"):
                    err += 1
                    chunk_error_rows.append({"article_link": u, "error": str(res.get("wayback_error"))})
                elif res.get("wayback_available"):
                    hit += 1
                else:
                    miss += 1

                pbar.update(1)
                pbar.set_postfix(hit=hit, miss=miss, err=err, refresh=False)

            pbar.close()

        if snapshot_cache_path is not None:
            _save_json(snapshot_cache_path, snapshot_cache)
            print(f"[SNAPSHOT SAVE] {snapshot_cache_path} | hit={hit} miss={miss} err={err}")
        if error_log_path is not None:
            _append_error_log(error_log_path, chunk_error_rows)

    print(f"[SNAPSHOT DONE] hit={hit} miss={miss} err={err}")
    return snapshot_cache


def assemble_and_save_output(
    df: pd.DataFrame,
    snapshot_cache: dict,
    content_cache: dict,
    url_col: str,
    checkpoint_csv: str,
):
    rows = []
    for u in df[url_col].fillna("").astype(str).tolist():
        s = snapshot_cache.get(u, empty_snapshot())
        c = content_cache.get(u, empty_content()) if s.get("wayback_available") else empty_content()
        row = {k: s.get(k) for k in SNAPSHOT_COLS}
        row.update({k: c.get(k) for k in CONTENT_COLS})
        rows.append(row)

    enriched = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows, columns=ALL_COLS)], axis=1)
    enriched.to_csv(checkpoint_csv, index=False)

    hits = int(enriched["wayback_available"].fillna(False).sum())
    misses = int((~enriched["wayback_available"].fillna(False)).sum())
    snap_errors = int(enriched["wayback_error"].notna().sum())
    content_errors = int(enriched["content_error"].notna().sum())
    print(f"[DONE] hits={hits} misses={misses} snapshot_errors={snap_errors} content_errors={content_errors}")
    print(f"[DONE] output={Path(checkpoint_csv).resolve()}")
    return enriched


def enrich_with_direct_day_lookup(
    df: pd.DataFrame,
    url_col: str = "article_link",
    day_yyyymmdd: str = "20260115",
    workers_snapshot: int = 16,
    cache_dir: str = "./wayback_cache",
    checkpoint_csv: str = "headline_with_wayback_context.csv",
    save_every_n_urls: int = 500,
    rerun_errors: bool = True,
    rerun_misses: bool = False,
    retries: int = 3,
    backoff_factor: float = 1.0,
    max_attempts_per_url: int = 3,
    retry_backoff_sec: float = 2.0,
):
    if url_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{url_col}'")

    cache_dir = Path(cache_dir)
    snapshot_cache_path = cache_dir / f"snapshot_cache_day_{day_yyyymmdd}.json"
    error_log_path = cache_dir / f"snapshot_errors_day_{day_yyyymmdd}.csv"
    content_cache_path = cache_dir / "content_cache.json"
    snapshot_cache = _load_json(snapshot_cache_path)
    if error_log_path.exists():
        error_log_path.unlink()

    domain_check = verify_dataset_domains(df, url_col=url_col)
    print("[VERIFY] raw hosts (top):", list(domain_check["raw_hosts"].items())[:10])
    print("[VERIFY] normalized hosts (top):", list(domain_check["normalized_hosts"].items())[:10])
    print("[VERIFY] only onion/huffpost after normalization:", domain_check["only_target_after_normalization"])

    snapshot_cache = build_snapshot_cache_direct_day(
        df=df,
        url_col=url_col,
        day_yyyymmdd=day_yyyymmdd,
        workers_snapshot=workers_snapshot,
        save_every_n_urls=save_every_n_urls,
        rerun_errors=rerun_errors,
        rerun_misses=rerun_misses,
        snapshot_cache=snapshot_cache,
        retries=retries,
        backoff_factor=backoff_factor,
        max_attempts_per_url=max_attempts_per_url,
        retry_backoff_sec=retry_backoff_sec,
        snapshot_cache_path=snapshot_cache_path,
        error_log_path=error_log_path,
    )
    print(f"[SNAPSHOT ERRORS] {error_log_path}")

    content_cache = _load_json(content_cache_path)
    return assemble_and_save_output(
        df=df,
        snapshot_cache=snapshot_cache,
        content_cache=content_cache,
        url_col=url_col,
        checkpoint_csv=checkpoint_csv,
    )


def enrich_with_direct_range_lookup(
    df: pd.DataFrame,
    url_col: str = "article_link",
    from_yyyymmdd: str = "20250101",
    to_yyyymmdd: str = "20251231",
    workers_snapshot: int = 16,
    cache_dir: str = "./wayback_cache",
    checkpoint_csv: str = "headline_with_wayback_context.csv",
    save_every_n_urls: int = 500,
    rerun_errors: bool = True,
    rerun_misses: bool = False,
    retries: int = 3,
    backoff_factor: float = 1.0,
    max_attempts_per_url: int = 3,
    retry_backoff_sec: float = 2.0,
):
    if url_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{url_col}'")

    cache_dir = Path(cache_dir)
    snapshot_cache_path = cache_dir / f"snapshot_cache_range_{from_yyyymmdd}_{to_yyyymmdd}.json"
    error_log_path = cache_dir / f"snapshot_errors_range_{from_yyyymmdd}_{to_yyyymmdd}.csv"
    content_cache_path = cache_dir / "content_cache.json"
    snapshot_cache = _load_json(snapshot_cache_path)
    if error_log_path.exists():
        error_log_path.unlink()

    domain_check = verify_dataset_domains(df, url_col=url_col)
    print("[VERIFY] raw hosts (top):", list(domain_check["raw_hosts"].items())[:10])
    print("[VERIFY] normalized hosts (top):", list(domain_check["normalized_hosts"].items())[:10])
    print("[VERIFY] only onion/huffpost after normalization:", domain_check["only_target_after_normalization"])

    snapshot_cache = build_snapshot_cache_direct_range(
        df=df,
        url_col=url_col,
        from_yyyymmdd=from_yyyymmdd,
        to_yyyymmdd=to_yyyymmdd,
        workers_snapshot=workers_snapshot,
        save_every_n_urls=save_every_n_urls,
        rerun_errors=rerun_errors,
        rerun_misses=rerun_misses,
        snapshot_cache=snapshot_cache,
        retries=retries,
        backoff_factor=backoff_factor,
        max_attempts_per_url=max_attempts_per_url,
        retry_backoff_sec=retry_backoff_sec,
        snapshot_cache_path=snapshot_cache_path,
        error_log_path=error_log_path,
    )
    print(f"[SNAPSHOT ERRORS] {error_log_path}")

    content_cache = _load_json(content_cache_path)
    return assemble_and_save_output(
        df=df,
        snapshot_cache=snapshot_cache,
        content_cache=content_cache,
        url_col=url_col,
        checkpoint_csv=checkpoint_csv,
    )


def enrich_with_wayback_domain_index(
    df: pd.DataFrame,
    url_col: str = "article_link",
    workers_content: int = 8,
    retries: int = 5,
    backoff_factor: float = 1.0,
    cache_dir: str = "./wayback_cache",
    checkpoint_csv: str = "headline_with_wayback_context.csv",
    save_every_n_urls: int = 500,
    rerun_errors: bool = True,
    rebuild_index: bool = False,
    start_year: int = 2000,
    end_year: int = 2026,
    max_attempts_per_month: int = 3,
    retry_backoff_sec: float = 2.0,
):
    if url_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{url_col}'")

    cache_dir = Path(cache_dir)
    snapshot_cache_path = cache_dir / "snapshot_cache.json"
    content_cache_path = cache_dir / "content_cache.json"
    index_cache_path = cache_dir / "domain_latest_index.json"
    all_links_cache_path = cache_dir / "domain_all_article_links.json"

    domain_check = verify_dataset_domains(df, url_col=url_col)
    print("[VERIFY] raw hosts (top):", list(domain_check["raw_hosts"].items())[:10])
    print("[VERIFY] normalized hosts (top):", list(domain_check["normalized_hosts"].items())[:10])
    print("[VERIFY] only onion/huffpost after normalization:", domain_check["only_target_after_normalization"])
    if domain_check["non_target_examples"]:
        print("[VERIFY] sample non-target normalized entries:")
        for ex in domain_check["non_target_examples"][:5]:
            print(" -", ex)

    latest_index, all_article_links = build_domain_latest_index(
        index_cache_path=index_cache_path,
        all_links_cache_path=all_links_cache_path,
        rebuild=rebuild_index,
        start_year=start_year,
        end_year=end_year,
        max_attempts_per_month=max_attempts_per_month,
        retry_backoff_sec=retry_backoff_sec,
    )
    print(f"[INDEX] all article links stored: {len(all_article_links)}")

    snapshot_cache = build_snapshot_cache_by_matching(df, latest_index, url_col=url_col)
    _save_json(snapshot_cache_path, snapshot_cache)

    content_cache = _load_json(content_cache_path)
    hit_urls = [u for u, s in snapshot_cache.items() if s.get("wayback_available") and s.get("wayback_url")]
    content_todo = []
    for u in hit_urls:
        c = content_cache.get(u)
        if c is None:
            content_todo.append(u)
        elif rerun_errors and c.get("content_error"):
            content_todo.append(u)

    print(f"[INFO] hit_urls={len(hit_urls)}")
    print(f"[INFO] content_todo={len(content_todo)} (missing + error reruns)")

    content_ok = content_err = 0
    for i in range(0, len(content_todo), save_every_n_urls):
        chunk = content_todo[i : i + save_every_n_urls]

        def _extract_for_url(u):
            wb_url = snapshot_cache[u]["wayback_url"]
            return extract_content(wb_url, retries=retries, backoff_factor=backoff_factor)

        with ThreadPoolExecutor(max_workers=workers_content) as ex:
            fut_to_url = {ex.submit(_extract_for_url, u): u for u in chunk}
            pbar = tqdm(total=len(chunk), desc=f"content {i}:{i + len(chunk)}")

            for fut in as_completed(fut_to_url):
                u = fut_to_url[fut]
                try:
                    c = fut.result()
                except Exception as e:
                    c = empty_content()
                    c["content_error"] = f"{type(e).__name__}: {e}"

                content_cache[u] = c
                if c.get("content_error"):
                    content_err += 1
                else:
                    content_ok += 1

                pbar.update(1)
                pbar.set_postfix(ok=content_ok, err=content_err, refresh=False)

            pbar.close()

        _save_json(content_cache_path, content_cache)
        print(f"[CONTENT SAVE] {content_cache_path} | ok={content_ok} err={content_err}")

    rows = []
    for u in df[url_col].fillna("").astype(str).tolist():
        s = snapshot_cache.get(u, empty_snapshot())
        c = content_cache.get(u, empty_content()) if s.get("wayback_available") else empty_content()
        row = {k: s.get(k) for k in SNAPSHOT_COLS}
        row.update({k: c.get(k) for k in CONTENT_COLS})
        rows.append(row)

    enriched = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows, columns=ALL_COLS)], axis=1)
    enriched.to_csv(checkpoint_csv, index=False)

    hits = int(enriched["wayback_available"].fillna(False).sum())
    misses = int((~enriched["wayback_available"].fillna(False)).sum())
    snap_errors = int(enriched["wayback_error"].notna().sum())
    content_errors = int(enriched["content_error"].notna().sum())

    print(f"[DONE] hits={hits} misses={misses} snapshot_errors={snap_errors} content_errors={content_errors}")
    print(f"[DONE] output={Path(checkpoint_csv).resolve()}")
    return enriched


def main():
    parser = argparse.ArgumentParser(description="Extract contextual features using CDX.")
    parser.add_argument(
        "--input-jsonl",
        default=str(Path("..") / "archive" / "Sarcasm_Headlines_Dataset_v2.json"),
        help="Path to JSONL dataset containing article_link column.",
    )
    parser.add_argument("--url-col", default="article_link", help="URL column name.")
    parser.add_argument(
        "--lookup-mode",
        choices=["direct_day", "direct_range", "domain_index"],
        default="direct_day",
        help="Lookup strategy: direct per-link day query, or domain pre-index crawl.",
    )
    parser.add_argument(
        "--lookup-day",
        default="20260115",
        help="Single-day CDX lookup window in YYYYMMDD (used in direct_day mode).",
    )
    parser.add_argument(
        "--lookup-from",
        default="20250101",
        help="Range start CDX lookup date in YYYYMMDD (used in direct_range mode).",
    )
    parser.add_argument(
        "--lookup-to",
        default="20251231",
        help="Range end CDX lookup date in YYYYMMDD (used in direct_range mode).",
    )
    parser.add_argument("--workers-snapshot", type=int, default=16, help="Thread workers for snapshot lookup.")
    parser.add_argument("--workers-content", type=int, default=8, help="Thread workers for content extraction.")
    parser.add_argument("--retries", type=int, default=2, help="HTTP retries for content fetch.")
    parser.add_argument("--backoff-factor", type=float, default=1.0, help="HTTP backoff factor.")
    parser.add_argument("--cache-dir", default="./wayback_cache", help="Cache directory.")
    parser.add_argument(
        "--checkpoint-csv",
        default="headline_with_wayback_context.csv",
        help="Output CSV path.",
    )
    parser.add_argument("--save-every-n-urls", type=int, default=500, help="Content save chunk size.")
    parser.add_argument(
        "--rerun-errors",
        action="store_true",
        help="Re-run rows with content_error in content cache.",
    )
    parser.add_argument(
        "--rerun-misses",
        action="store_true",
        help="Re-run rows that previously had no snapshot match and no error.",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Force rebuilding Wayback domain index and all-links inventory.",
    )
    parser.add_argument("--start-year", type=int, default=2000, help="Start year for Wayback index crawl.")
    parser.add_argument("--end-year", type=int, default=2026, help="End year for Wayback index crawl.")
    parser.add_argument(
        "--max-attempts-per-month",
        type=int,
        default=2,
        help="Retry attempts per domain/window CDX crawl.",
    )
    parser.add_argument(
        "--retry-backoff-sec",
        type=float,
        default=0.5,
        help="Base backoff seconds between CDX retry attempts.",
    )
    args = parser.parse_args()

    df = pd.read_json(args.input_jsonl, lines=True)
    print(df.head())
    print(df.shape)
    print(df.columns)

    if args.lookup_mode == "direct_day":
        enrich_with_direct_day_lookup(
            df=df,
            url_col=args.url_col,
            day_yyyymmdd=args.lookup_day,
            workers_snapshot=args.workers_snapshot,
            cache_dir=args.cache_dir,
            checkpoint_csv=args.checkpoint_csv,
            save_every_n_urls=args.save_every_n_urls,
            rerun_errors=args.rerun_errors,
            rerun_misses=args.rerun_misses,
            retries=args.retries,
            backoff_factor=args.backoff_factor,
            max_attempts_per_url=args.max_attempts_per_month,
            retry_backoff_sec=args.retry_backoff_sec,
        )
    elif args.lookup_mode == "direct_range":
        enrich_with_direct_range_lookup(
            df=df,
            url_col=args.url_col,
            from_yyyymmdd=args.lookup_from,
            to_yyyymmdd=args.lookup_to,
            workers_snapshot=args.workers_snapshot,
            cache_dir=args.cache_dir,
            checkpoint_csv=args.checkpoint_csv,
            save_every_n_urls=args.save_every_n_urls,
            rerun_errors=args.rerun_errors,
            rerun_misses=args.rerun_misses,
            retries=args.retries,
            backoff_factor=args.backoff_factor,
            max_attempts_per_url=args.max_attempts_per_month,
            retry_backoff_sec=args.retry_backoff_sec,
        )
    else:
        enrich_with_wayback_domain_index(
            df=df,
            url_col=args.url_col,
            workers_content=args.workers_content,
            retries=args.retries,
            backoff_factor=args.backoff_factor,
            cache_dir=args.cache_dir,
            checkpoint_csv=args.checkpoint_csv,
            save_every_n_urls=args.save_every_n_urls,
            rerun_errors=args.rerun_errors,
            rebuild_index=args.rebuild_index,
            start_year=args.start_year,
            end_year=args.end_year,
            max_attempts_per_month=args.max_attempts_per_month,
            retry_backoff_sec=args.retry_backoff_sec,
        )


if __name__ == "__main__":
    main()
