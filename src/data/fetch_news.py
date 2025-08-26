import os, time, random
from typing import List, Dict
from urllib.parse import quote_plus
import requests
import pandas as pd
from bs4 import BeautifulSoup

# default headers for requests
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/127 Safari/537.36"
    )
}

def fetch_news(ticker: str) -> List[Dict]:
    # build url
    url = f"https://finviz.com/quote.ashx?t={quote_plus(ticker)}"
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    # find news table
    news_table = soup.find(id="news-table")
    rows = []
    if news_table:
        for row in news_table.find_all("tr"):
            tds = row.find_all("td")
            # skip rows with missing cells
            if len(tds) < 2:
                continue
            # parse time and headline
            time_text = tds[0].get_text(strip=True)
            headline_td = tds[1]
            headline = headline_td.get_text(strip=True)
            link = headline_td.a["href"] if headline_td.a else ""
            # store record
            rows.append({
                "ticker": ticker,
                "title": headline,
                "link": link,
                "pubDate": time_text
            })
    return rows


def fetch_news_for_ticker(ticker: str) -> List[Dict]:
    # try fetching from finviz
    rows = []
    try:
        rows = fetch_news(ticker)
    except Exception as e:
        print(f"[WARN] Finviz failed for {ticker}: {e}")

    return rows


def gather_news(tickers: List[str]) -> pd.DataFrame:
    all_rows = []
    for t in tickers:
        # random delay to avoid hammering
        time.sleep(0.4 + random.uniform(0, 0.4))
        items = fetch_news_for_ticker(t)
        all_rows.extend(items)

    df = pd.DataFrame(all_rows)
    if df.empty:
        # create empty df with expected columns
        df = pd.DataFrame(columns=["ticker", "title", "link", "pubDate"])
    else:
        # drop duplicate headlines for same ticker
        df = df.drop_duplicates(subset=["ticker", "title"])
    return df

def save_news(df: pd.DataFrame, out_path: str):
    # ensure output dir exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # write to csv
    df.to_csv(out_path, index=False)
