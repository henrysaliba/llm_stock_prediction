import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from dateutil import parser as dateparser
from dateutil import tz

# define base directories for project data
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROC_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR = PROC_DIR / "graphs"

# input files
PRICES_CSV = RAW_DIR / "prices.csv"
NEWS_SENT_CSV = PROC_DIR / "news_with_sentiment.csv"

# timezone settings
TZ_NAME = "Australia/Adelaide"
LOCAL_TZ = tz.gettz(TZ_NAME)

# ensure output folder exists
os.makedirs(OUT_DIR, exist_ok=True)


def parse_pubdate(s: str) -> pd.Timestamp | None:
    # parse publication date string into localized timestamp
    if not isinstance(s, str) or not s.strip():
        return None
    try:
        dt = dateparser.parse(s)
        # if missing timezone, assign local tz
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=LOCAL_TZ)
        return pd.Timestamp(dt).tz_convert(LOCAL_TZ)
    except Exception:
        # invalid/unparsable strings -> None
        return None


def load_prices():
    # read raw prices csv
    df = pd.read_csv(PRICES_CSV)
    # parse dates
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    # keep only relevant cols
    return df[["Date", "Ticker", "Close"]].dropna()


def load_news_with_sentiment():
    # read processed news+sentiment file
    df = pd.read_csv(NEWS_SENT_CSV)
    # normalize column names to lowercase
    df.columns = [c.strip().lower() for c in df.columns]
    # drop duplicate headlines for each ticker
    df = df.drop_duplicates(subset=["ticker", "title"])
    # parse and normalize publication dates
    df["pub_dt"] = df["pubdate"].apply(parse_pubdate)
    df = df.dropna(subset=["pub_dt"]).copy()
    df["pub_dt"] = pd.to_datetime(df["pub_dt"]).dt.tz_convert(LOCAL_TZ)
    # add date-only column for grouping
    df["pub_date"] = df["pub_dt"].dt.date
    return df


def figure_daily_sentiment(df_news, ticker, out_path):
    # filter for one ticker
    sub = df_news[df_news["ticker"] == ticker].copy()
    if sub.empty:
        return
    # sentiment index = pos - neg
    sub["sent_index"] = sub["p_positive"] - sub["p_negative"]
    # average sentiment per day
    daily = sub.groupby("pub_date")["sent_index"].mean().sort_index()

    # make line plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(daily.index, daily.values)
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily avg sentiment index")
    ax.set_title(f"Daily sentiment index (p_pos - p_neg) — {ticker}")
    fig.autofmt_xdate()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def figure_sentiment_over_price(df_prices, df_news, ticker, out_path):
    # load prices for ticker and resample to daily
    px = df_prices[df_prices["Ticker"] == ticker].copy()
    if px.empty:
        return
    px = px.sort_values("Date").set_index("Date")["Close"].resample("D").ffill()

    # filter news for ticker
    sub = df_news[df_news["ticker"] == ticker].copy()
    if sub.empty:
        return
    # align pub_date with daily price series
    sub["pub_date"] = pd.to_datetime(sub["pub_date"])
    sub["price_at_news"] = px.reindex(sub["pub_date"]).values

    # plot price curve
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(px.index, px.values, label=f"{ticker} Close")

    # overlay news points, shape depends on sentiment
    markers = {"positive": "o", "neutral": "s", "negative": "^"}
    for cls, mark in markers.items():
        ss = sub[sub["sentiment"] == cls]
        ax.scatter(ss["pub_date"], ss["price_at_news"], marker=mark,
                   alpha=0.7, label=f"News ({cls})")

    ax.set_xlabel("Date")
    ax.set_ylabel("Price (Close)")
    ax.set_title(f"Price with sentiment-marked news — {ticker}")
    ax.legend()
    fig.autofmt_xdate()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    # load input datasets
    prices = load_prices()
    news_s = load_news_with_sentiment()

    # define which tickers to plot
    tickers = "AAPL", "AMZN", "NVDA", "GOOGL", "MSFT"
    for T in tickers:
        # save daily sentiment trend
        figure_daily_sentiment(news_s, T, OUT_DIR / f"daily_sentiment_index_{T}.png")
        # save price with sentiment markers
        figure_sentiment_over_price(prices, news_s, T, OUT_DIR / f"sentiment_over_price_{T}.png")

    print("Graphs saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
