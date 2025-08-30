#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


## configuration

PRICES_CSV = "data/raw/prices.csv"
NEWS_SENT_CSV = "data/processed/news_with_sentiment.csv"

OUT_PROCESSED_DIR = Path("data/processed")
OUT_FIG_DIR = OUT_PROCESSED_DIR / "graphs"

# ensure output dirs exist
OUT_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)


## io / loading

def load_prices(path: str = PRICES_CSV) -> pd.DataFrame:
    # read prices and normalize column names
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    # yfinance multi-index fix (sometimes 'level_1' carries ticker)
    if "ticker" not in df.columns and "level_1" in df.columns:
        df = df.rename(columns={"level_1": "ticker"})

    if "date" not in df.columns:
        raise ValueError("prices.csv must contain a 'Date' column (lowercased here to 'date').")

    # pick a close column
    price_col: Optional[str] = None
    for c in ("close", "adj close", "adj_close"):
        if c in df.columns:
            price_col = c
            break
    if price_col is None:
        raise ValueError("no close/adj close column found in prices.csv")

    # parse dates and order
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.dropna(subset=["date"]).sort_values(["ticker", "date"]).reset_index(drop=True)

    # same-day and next-day returns
    df["ret_1d"] = df.groupby("ticker")[price_col].pct_change()
    df["fwd_ret_1d"] = df.groupby("ticker")[price_col].shift(-1) / df[price_col] - 1.0

    # standardize output
    keep = ["date", "ticker", price_col, "ret_1d", "fwd_ret_1d"]
    if "volume" in df.columns:
        keep.append("volume")
    return df[keep].rename(columns={price_col: "close"})


def load_news_with_sentiment(path: str = NEWS_SENT_CSV) -> pd.DataFrame:
    # read news + sentiment and normalize columns
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    # required columns
    required = {"ticker", "pubdate", "sentiment"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"news_with_sentiment.csv missing columns: {missing}")

    # parse timestamps to UTC
    df["pubdate"] = pd.to_datetime(df["pubdate"], utc=True, errors="coerce")
    df = df.dropna(subset=["pubdate"]).reset_index(drop=True)

    # numeric score: prefer probs (p_pos - p_neg), else label map
    if {"p_positive", "p_negative"}.issubset(df.columns):
        df["sentiment_score"] = df["p_positive"].fillna(0.0) - df["p_negative"].fillna(0.0)
    else:
        label_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
        df["sentiment_score"] = df["sentiment"].map(label_map).fillna(0.0)

    return df


## alignment helpers

def map_news_to_trading_day_utc(news_ts_utc: pd.Series, trading_days_date: np.ndarray) -> pd.Series:
    """
    map each news timestamp to the NEXT trading day (>= calendar day)

    uses numpy datetime64[D] to avoid tz/NaT issues:
      - convert news timestamps to UTC midnights (dates only)
      - searchsorted against sorted trading-day dates
    """
    trading_days_np = np.array(trading_days_date, dtype="datetime64[D]")
    news_days_np = news_ts_utc.dt.tz_convert("UTC").dt.normalize().values.astype("datetime64[D]")

    # insertion point (first trading day >= news day)
    idx = np.searchsorted(trading_days_np, news_days_np, side="left")

    # safe mapping (guard OOB)
    mapped_np = np.full(idx.shape, np.datetime64("NaT", "D"), dtype="datetime64[D]")
    valid = idx < len(trading_days_np)
    if np.any(valid):
        mapped_np[valid] = trading_days_np[idx[valid]]

    # to Series[python date]
    mapped_dtindex = pd.to_datetime(mapped_np)
    return pd.Series(mapped_dtindex.date)


def aggregate_daily_sentiment(news_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
    # trading-day calendar from price file
    trading_days = np.array(sorted(prices_df["date"].unique()))

    # map each headline to next trading day
    news_df = news_df.copy()
    news_df["trade_date"] = map_news_to_trading_day_utc(news_df["pubdate"], trading_days)
    news_df = news_df.dropna(subset=["trade_date"])

    # aggregate per (ticker, trade_date)
    agg = (
        news_df.groupby(["ticker", "trade_date"])
        .agg(
            sentiment_mean=("sentiment_score", "mean"),
            sentiment_sum=("sentiment_score", "sum"),
            news_count=("sentiment_score", "count"),
            ppos_mean=("p_positive", "mean") if "p_positive" in news_df.columns else ("sentiment_score", "mean"),
            pneg_mean=("p_negative", "mean") if "p_negative" in news_df.columns else ("sentiment_score", "mean"),
        )
        .reset_index()
        .rename(columns={"trade_date": "date"})
    )

    # persist for downstream steps
    agg.to_csv(OUT_PROCESSED_DIR / "sentiment_daily.csv", index=False)
    return agg


## merge + targets

def merge_sentiment_prices(sent_daily: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
    # left-join so we keep the full price timeline
    merged = prices_df.merge(sent_daily, on=["ticker", "date"], how="left")

    # keep NaNs (no-news days) so we don't bias toward zero
    merged = merged.sort_values(["ticker", "date"])

    # 7D rolling mean computed only over existing values
    merged["sentiment_roll7"] = (
    merged.groupby("ticker")["sentiment_mean"]
    .transform(lambda s: s.rolling(window=7, min_periods=1).mean())
    )


    # persist
    merged.to_csv(OUT_PROCESSED_DIR / "sentiment_price_merged.csv", index=False)
    return merged


## plots (saved to data/processed/graphs)

def plot_scatter_signal(merged: pd.DataFrame) -> None:
    # daily sentiment vs next-day returns (all tickers combined)
    df = merged.dropna(subset=["sentiment_mean", "fwd_ret_1d"]).copy()
    if df.empty:
        return

    x = df["sentiment_mean"].values
    y = df["fwd_ret_1d"].values

    # simple linear fit and R^2
    coeffs = np.polyfit(x, y, 1)
    yhat = np.polyval(coeffs, x)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    plt.figure(figsize=(7, 6))
    plt.scatter(x, y, alpha=0.25, s=10)
    xs = np.linspace(x.min(), x.max(), 100)
    plt.plot(xs, np.polyval(coeffs, xs), linewidth=2)
    plt.title(f"Daily sentiment (mean) vs Next-day returns\nLinear fit R²={r2:.4f}")
    plt.xlabel("Daily sentiment (mapped to next trading day)")
    plt.ylabel("Next-day return")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_FIG_DIR / "scatter_sentiment_vs_nextday_return.png", dpi=160)
    plt.close()


def plot_cross_correlation(merged: pd.DataFrame, max_lag: int = 5) -> None:
    # correlation across lags: corr(sent_t, ret_{t+L})
    df = merged.sort_values(["ticker", "date"]).copy()
    df = df[["ticker", "date", "sentiment_mean", "ret_1d"]].dropna(subset=["sentiment_mean", "ret_1d"])
    if df.empty:
        return

    lags = range(-max_lag, max_lag + 1)
    corrs = []
    for L in lags:
        tmp = df.copy()
        tmp["ret_shift"] = tmp.groupby("ticker")["ret_1d"].shift(-L)  # +L means sentiment leads
        corrs.append(tmp["sentiment_mean"].corr(tmp["ret_shift"]))

    plt.figure(figsize=(8, 5))
    plt.bar(list(lags), corrs)
    plt.axhline(0, color="black", linewidth=1)
    plt.title("Cross-correlation: sentiment vs returns  (+lag = sentiment leads)")
    plt.xlabel("Lag (days)")
    plt.ylabel("Correlation")
    plt.tight_layout()
    plt.savefig(OUT_FIG_DIR / "cross_correlation_lag_-5_to_5.png", dpi=160)
    plt.close()


def plot_event_study(merged: pd.DataFrame, horizon: int = 5, pos_q: float = 0.7, neg_q: float = 0.3) -> None:
    # average forward cumulative returns following extreme sentiment days
    df = merged.dropna(subset=["sentiment_mean", "fwd_ret_1d"]).copy()
    if df.empty:
        return

    # label extremes per ticker (prevents one name from dominating)
    def label_events(g: pd.DataFrame) -> pd.DataFrame:
        up = g["sentiment_mean"].quantile(pos_q)
        dn = g["sentiment_mean"].quantile(neg_q)
        g["event_pos"] = g["sentiment_mean"] >= up
        g["event_neg"] = g["sentiment_mean"] <= dn
        return g

    df = df.sort_values(["ticker", "date"])
    df = df.groupby("ticker", group_keys=False).apply(label_events)

    # build forward paths
    def forward_path(sub: pd.DataFrame, mask_col: str) -> np.ndarray:
        idx = sub.index[sub[mask_col]].tolist()
        paths = []
        for i in idx:
            fwd = sub.loc[i : i + horizon - 1, "fwd_ret_1d"].values
            if len(fwd) < horizon:
                continue
            paths.append(np.cumprod(1.0 + fwd) - 1.0)  # cumulative returns
        if not paths:
            return np.array([])
        return np.vstack(paths)

    pos_paths, neg_paths = [], []
    for _, g in df.groupby("ticker"):
        g = g.reset_index(drop=True)
        p = forward_path(g, "event_pos")
        n = forward_path(g, "event_neg")
        if p.size:
            pos_paths.append(p)
        if n.size:
            neg_paths.append(n)

    if not pos_paths and not neg_paths:
        return

    # average paths across tickers
    def avg_path(stacks: list[np.ndarray]) -> np.ndarray | None:
        if not stacks:
            return None
        return np.nanmean(np.vstack(stacks), axis=0)

    pos_avg = avg_path(pos_paths)
    neg_avg = avg_path(neg_paths)

    plt.figure(figsize=(8, 6))
    if pos_avg is not None:
        plt.plot(range(1, horizon + 1), pos_avg, linewidth=2, label="Positive sentiment days")
    if neg_avg is not None:
        plt.plot(range(1, horizon + 1), neg_avg, linewidth=2, label="Negative sentiment days")
    plt.axhline(0, color="black", linewidth=1)
    plt.title(f"Event study: average forward cumulative return (1..{horizon} days)")
    plt.xlabel("Days ahead")
    plt.ylabel("Cumulative return")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_FIG_DIR / "event_study_pos_vs_neg.png", dpi=160)
    plt.close()

def plot_overlays_per_ticker(merged: pd.DataFrame) -> None:
    # Plot price vs sentiment with dual y-axes (per ticker)
    for tkr, g in merged.groupby("ticker"):
        g = g.sort_values("date")
        if g.empty:
            continue

        fig, ax1 = plt.subplots(figsize=(9, 5))

        # Left axis: price
        ax1.plot(g["date"], g["close"], color="tab:blue", linewidth=2, label="Price")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Price", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")

        # Right axis: sentiment (rolling 7-day mean, no rescaling)
        ax2 = ax1.twinx()
        ax2.plot(g["date"], g["sentiment_roll7"], color="tab:orange", linewidth=2, label="7D sentiment")
        ax2.set_ylabel("7D Sentiment (mean)", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")

        # Title and grid
        plt.title(f"{tkr}: Price vs 7D Sentiment")
        fig.tight_layout()
        plt.grid(True, alpha=0.3)

        # Save
        plt.savefig(OUT_FIG_DIR / f"overlay_{tkr}.png", dpi=160)
        plt.close()



## metrics

def compute_metrics(merged: pd.DataFrame) -> pd.DataFrame:
    # quick correlations per ticker (diagnostics, not significance tests)
    rows = []
    for tkr, g in merged.groupby("ticker"):
        g = g.dropna(subset=["sentiment_mean"])
        rows.append({
            "ticker": tkr,
            "corr_same_day": g["sentiment_mean"].corr(g["ret_1d"]),
            "corr_next_day": g["sentiment_mean"].corr(g["fwd_ret_1d"]),
        })

    by_ticker = pd.DataFrame(rows).sort_values("ticker")
    by_ticker.to_csv(OUT_PROCESSED_DIR / "metrics_by_ticker.csv", index=False)

    overall_same = merged["sentiment_mean"].corr(merged["ret_1d"])
    overall_next = merged["sentiment_mean"].corr(merged["fwd_ret_1d"])
    print("\n=== Correlation summary ===")
    print(by_ticker.to_string(index=False))
    print(f"\nOverall corr (sentiment vs SAME-day return): {overall_same:.4f}")
    print(f"Overall corr (sentiment vs NEXT-day return): {overall_next:.4f}\n")

    return by_ticker


## entrypoint

def main() -> None:
    prices = load_prices(PRICES_CSV)
    news = load_news_with_sentiment(NEWS_SENT_CSV)
    sent_daily = aggregate_daily_sentiment(news, prices)
    merged = merge_sentiment_prices(sent_daily, prices)

    plot_scatter_signal(merged)
    plot_cross_correlation(merged, max_lag=5)
    plot_event_study(merged, horizon=5, pos_q=0.7, neg_q=0.3)
    plot_overlays_per_ticker(merged)
    compute_metrics(merged)

    print("Analysis complete")


if __name__ == "__main__":
    main()
