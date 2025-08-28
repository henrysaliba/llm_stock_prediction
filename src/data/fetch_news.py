from __future__ import annotations

import os
import time
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone, date, timedelta

import requests
import pandas as pd

## configuration

# Pull date range from config if available; otherwise default to last 90 days
try:
    from ..utils.config import START_DATE, END_DATE
except Exception:
    # fallback when run standalone
    END_DATE = date.today().isoformat()
    START_DATE = (date.today() - timedelta(days=90)).isoformat()

# Base API endpoint for Finnhub company news
FINNHUB_BASE = "https://finnhub.io/api/v1/company-news"

# Rate limiting and retries
MIN_DELAY_SECONDS = 1.0 # polite delay between calls
TIMEOUT = 20 # request timeout
MAX_RETRIES = 3 # retry attempts on failure
BACKOFF_BASE = 1.8 # exponential backoff factor

# Default API key (overridden by env var if set)
DEFAULT_API_KEY = "d2motp9r01qog444mulgd2motp9r01qog444mum0"

# Request headers (helps avoid blocks)
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/127 Safari/537.36"
    )
}

# Fetch in chunks (Finnhub handles ~30 days well)
CHUNK_DAYS = 30


## helpers
def _get_api_key() -> str:
    # grab API key from environment if available, otherwise fall back to default
    key = os.getenv("FINNHUB_API_KEY", "").strip()
    return key if key else DEFAULT_API_KEY


def _iso_utc_from_unix(ts: Optional[float | int]) -> str:
    # convert Unix timestamp (seconds) -> ISO-8601 UTC string (with 'Z' suffix)
    if ts is None:
        return ""
    try:
        dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return ""


def _request_with_retries(url: str, params: Dict[str, Any]) -> Any:
    # perform a GET request with retries, exponential backoff, and basic 429 handling
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
            if r.status_code == 429:  # API rate limit
                wait = BACKOFF_BASE ** attempt
                logging.warning(f"[WARN] Finnhub rate-limited (429). Sleeping {wait:.1f}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            wait = BACKOFF_BASE ** attempt
            logging.warning(f"[WARN] Finnhub request failed (attempt {attempt}/{MAX_RETRIES}): {e}")
            time.sleep(wait)
    # if all retries exhausted, raise error
    raise RuntimeError(f"Finnhub request failed after {MAX_RETRIES} attempts: {last_err}")


def _date_chunks(start_s: str, end_s: str, span_days: int = CHUNK_DAYS):
    # yield rolling date windows (from, to) between [start_s, end_s]
    start_d = pd.to_datetime(start_s).date()
    end_d = pd.to_datetime(end_s).date()
    cur = start_d
    one = timedelta(days=1)
    while cur <= end_d:
        stop = min(cur + timedelta(days=span_days - 1), end_d)
        yield cur.isoformat(), stop.isoformat()
        cur = stop + one


## fetching
def fetch_news_for_ticker(
    ticker: str,
    start_date: str = START_DATE,
    end_date: str = END_DATE
) -> List[Dict[str, Any]]:
    """
    Fetch company news from Finnhub for a single ticker across [start_date, end_date].
    Returns a list of dicts with schema:
        {"ticker", "title", "link", "pubDate"}
    """
    token = _get_api_key()
    rows: List[Dict[str, Any]] = []

    # break into chunks (API is friendlier with shorter ranges)
    for frm, to in _date_chunks(start_date, end_date, CHUNK_DAYS):
        params = {"symbol": ticker, "from": frm, "to": to, "token": token}
        try:
            data = _request_with_retries(FINNHUB_BASE, params=params) or []
        except Exception as e:
            logging.warning(f"[WARN] Finnhub company-news failed for {ticker} [{frm}..{to}]: {e}")
            continue

        # clean and normalize each returned news item
        for item in data:
            title = (item.get("headline") or "").strip()
            link = (item.get("url") or "").strip()
            pub_iso = _iso_utc_from_unix(item.get("datetime"))
            if not title or not link or not pub_iso:
                continue
            rows.append({"ticker": ticker, "title": title, "link": link, "pubDate": pub_iso})

        # small pause to avoid hammering the API
        time.sleep(MIN_DELAY_SECONDS)

    # deduplicate items (use ticker, title, pubDate as unique key)
    dedup = {(r["ticker"], r["title"], r["pubDate"]): r for r in rows}
    return list(dedup.values())

## normalization

def _normalize_pubdate_series(s: pd.Series) -> pd.Series:
    # force pubDate values into canonical ISO-8601 UTC form with 'Z'
    dt = pd.to_datetime(s, utc=True, errors="coerce")
    return dt.dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def gather_news(tickers: List[str]) -> pd.DataFrame:
    """
    Fetch news for multiple tickers across [START_DATE, END_DATE].
    Returns DataFrame with schema: ticker, title, link, pubDate
    """
    all_rows: List[Dict[str, Any]] = []

    # loop over tickers sequentially (polite delay between requests)
    for i, t in enumerate(tickers):
        if i > 0:
            time.sleep(MIN_DELAY_SECONDS)
        try:
            items = fetch_news_for_ticker(t, START_DATE, END_DATE)
            all_rows.extend(items)
        except Exception as e:
            logging.warning(f"[WARN] Finnhub company-news failed for {t}: {e}")

    df = pd.DataFrame(all_rows, columns=["ticker", "title", "link", "pubDate"])
    if df.empty:
        return pd.DataFrame(columns=["ticker", "title", "link", "pubDate"])

    # normalize pubDate column
    df["pubDate"] = _normalize_pubdate_series(df["pubDate"])
    # drop duplicates and sort by pubDate descending
    df = df.drop_duplicates(subset=["ticker", "title", "pubDate"]).reset_index(drop=True)
    order = pd.to_datetime(df["pubDate"], utc=True, errors="coerce")
    df = df.assign(_dt=order).sort_values("_dt", ascending=False).drop(columns=["_dt"]).reset_index(drop=True)
    return df


def save_news(df: pd.DataFrame, out_path: str) -> None:
    """
    Save news DataFrame to CSV (append-safe).
    Ensures unique (ticker,title,pubDate) across runs.
    Always sorted descending by pubDate.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # if empty, just create an empty file with headers
    if df is None or df.empty:
        pd.DataFrame(columns=["ticker", "title", "link", "pubDate"]).to_csv(out_path, index=False)
        return

    # merge with old file if it exists
    if os.path.exists(out_path):
        try:
            old = pd.read_csv(out_path)
            if "pubDate" in old.columns:
                old["pubDate"] = _normalize_pubdate_series(old["pubDate"])
            df = pd.concat([old, df], ignore_index=True)
        except Exception:
            pass

    # normalize and deduplicate after merge
    df["pubDate"] = _normalize_pubdate_series(df["pubDate"])
    df = df.drop_duplicates(subset=["ticker", "title", "pubDate"]).reset_index(drop=True)

    # sort descending by pubDate and save
    order = pd.to_datetime(df["pubDate"], utc=True, errors="coerce")
    df = df.assign(_dt=order).sort_values("_dt", ascending=False).drop(columns=["_dt"])
    df.to_csv(out_path, index=False)


## entrypoint for manual tests
if __name__ == "__main__":
    test_tickers = ["AAPL", "MSFT"]
    _df = gather_news(test_tickers)
    save_news(_df, "data/raw/news.csv")
    print(f"Saved {len(_df)} rows to data/raw/news.csv")
