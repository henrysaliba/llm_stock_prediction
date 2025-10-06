from __future__ import annotations
"""
builds a price+sentiment supervised dataset for the hybrid model.
   selected model: RandomForests which uses FinBERT sentiment
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Optional
import numpy as np
import pandas as pd

from src.features.tech_indicators import add_technical_features

# input/output paths
PRICES_CSV = "data/raw/prices.csv"
NEWS_SENT_CSV = "data/processed/news_with_sentiment.csv"
OUT_DATASET = "data/processed/hybrid/model_dataset_hybrid.csv"

def _load_prices(path: str = PRICES_CSV) -> pd.DataFrame:
    # read prices and normalise headers
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    # yfinance multi-index fix
    if "ticker" not in df.columns and "level_1" in df.columns:
        df = df.rename(columns={"level_1": "ticker"})
    if "date" not in df.columns:
        raise ValueError("prices.csv must contain 'Date' (lowercased to 'date').")

    # pick usable close column
    price_col: Optional[str] = None
    for c in ("close", "adj close", "adj_close"):
        if c in df.columns:
            price_col = c; break
    if price_col is None:
        raise ValueError("no close/adj close column found in prices.csv")

    # sort and make simple returns + next-day label
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.dropna(subset=["date"]).sort_values(["ticker","date"]).reset_index(drop=True)
    df["ret_1d"] = df.groupby("ticker")[price_col].pct_change()
    df["fwd_ret_1d"] = df.groupby("ticker")[price_col].shift(-1) / df[price_col] - 1.0

    # standardised columns for downstream joins
    keep = ["date","ticker",price_col,"ret_1d","fwd_ret_1d"]
    if "volume" in df.columns:
        keep.append("volume")
    return df[keep].rename(columns={price_col: "close"})

def _load_news_sent(path: str = NEWS_SENT_CSV) -> pd.DataFrame:
    # read news + sentiment and normalise headers
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    req = {"ticker","pubdate"}
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"news_with_sentiment.csv missing columns: {miss}")

    # timestamps to UTC; drop bad rows
    df["pubdate"] = pd.to_datetime(df["pubdate"], utc=True, errors="coerce")
    df = df.dropna(subset=["pubdate"]).reset_index(drop=True)

    # numeric score: p_pos - p_neg if available; else map labels
    if {"p_positive","p_negative"}.issubset(df.columns):
        df["sentiment_score"] = df["p_positive"].fillna(0.0) - df["p_negative"].fillna(0.0)
    else:
        lbl = {"positive":1.0, "neutral":0.0, "negative":-1.0}
        df["sentiment_score"] = df.get("sentiment", "").map(lbl).fillna(0.0)

    return df[["ticker","pubdate","sentiment_score"]].copy()

def _map_news_to_trade_date_utc(news_ts_utc: pd.Series, trading_days_date: np.ndarray) -> pd.Series:
    # map each news timestamp to the next available trading day (UTC calendar)
    td = np.array(sorted(trading_days_date), dtype="datetime64[D]")
    nd = news_ts_utc.dt.tz_convert("UTC").dt.normalize().values.astype("datetime64[D]")
    idx = np.searchsorted(td, nd, side="left")
    out = np.full(idx.shape, np.datetime64("NaT","D"))
    valid = idx < len(td)
    out[valid] = td[idx[valid]]
    return pd.to_datetime(out).date

def _aggregate_daily_sentiment(news_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
    # aggregate sentiment per (ticker, mapped trade date)
    trading_days = np.array(sorted(prices_df["date"].unique()))
    news_df = news_df.copy()
    news_df["date"] = _map_news_to_trade_date_utc(news_df["pubdate"], trading_days)
    news_df = news_df.dropna(subset=["date"])
    agg = (news_df.groupby(["ticker","date"])
           .agg(sentiment_mean=("sentiment_score","mean"),
                sentiment_sum=("sentiment_score","sum"),
                news_count=("sentiment_score","count"))
           .reset_index())
    return agg

def main():
    # load price data and build technical features (past-only)
    prices = _load_prices(PRICES_CSV)
    feat = add_technical_features(prices)

    # load news and compute daily sentiment aggregates
    news = _load_news_sent(NEWS_SENT_CSV)
    sent_daily = _aggregate_daily_sentiment(news, prices)

    # left-join sentiment onto the full price timeline
    merged = feat.merge(sent_daily, on=["ticker","date"], how="left").sort_values(["ticker","date"])

    # past-only rolling features on sentiment (no leakage)
    for w in [3, 5, 7]:
        merged[f"sent_mean_roll{w}"] = merged.groupby("ticker")["sentiment_mean"].transform(
            lambda s: s.rolling(w, min_periods=1).mean())
        merged[f"sent_sum_roll{w}"] = merged.groupby("ticker")["sentiment_sum"].transform(
            lambda s: s.rolling(w, min_periods=1).sum())
        merged[f"news_count_roll{w}"] = merged.groupby("ticker")["news_count"].transform(
            lambda s: s.rolling(w, min_periods=1).sum())

    # record missingness flags, then fill sentiment NaNs with 0
    for col in ["sentiment_mean","sentiment_sum","news_count",
                "sent_mean_roll3","sent_mean_roll5","sent_mean_roll7",
                "sent_sum_roll3","sent_sum_roll5","sent_sum_roll7",
                "news_count_roll3","news_count_roll5","news_count_roll7"]:
        merged[f"{col}_isna"] = merged[col].isna().astype(int)  # explicit missing indicator
        merged[col] = merged[col].fillna(0.0) # keep timeline continuity

    # assemble final column lists (technical + sentiment)
    base_cols = [
        "date","ticker","close","ret_1d","fwd_ret_1d",
        "ret_lag_1","ret_lag_2","ret_lag_3","ret_lag_5",
        "mom_5","mom_10","mom_20",
        "vol_5","vol_10","vol_20",
        "rsi_14",
    ]
    if "vol_z20" in merged.columns:
        base_cols.append("vol_z20")
    base_cols += [c for c in merged.columns if c.startswith("dow_")]  # day-of-week one-hots

    sent_cols = [
        "sentiment_mean","sentiment_sum","news_count",
        "sent_mean_roll3","sent_mean_roll5","sent_mean_roll7",
        "sent_sum_roll3","sent_sum_roll5","sent_sum_roll7",
        "news_count_roll3","news_count_roll5","news_count_roll7",
    ] + [c for c in merged.columns if c.endswith("_isna")]

    # final dataset; drop rows missing base tech features only
    ds = merged[base_cols + sent_cols + ["fwd_ret_1d","date","ticker"]].dropna(subset=base_cols).reset_index(drop=True)

    # write to disk
    Path(OUT_DATASET).parent.mkdir(parents=True, exist_ok=True)
    ds.to_csv(OUT_DATASET, index=False)
    print(f"Wrote {OUT_DATASET} with {len(ds):,} rows")

if __name__ == "__main__":
    main()
