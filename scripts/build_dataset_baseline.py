from __future__ import annotations

"""
Builds a price-only supervised dataset for the baseline model.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pathlib import Path
from typing import Optional
import pandas as pd

from src.features.tech_indicators import add_technical_features

PRICES_CSV = "data/raw/prices.csv"
OUT_DATASET = "data/processed/baseline/model_dataset_baseline.csv"


def _load_prices(path: str = PRICES_CSV) -> pd.DataFrame:
    # read prices.csv and standardize schema.

    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    # yfinance stacked shape sometimes puts ticker in 'level_1'
    if "ticker" not in df.columns and "level_1" in df.columns:
        df = df.rename(columns={"level_1": "ticker"})

    if "date" not in df.columns:
        raise ValueError("prices.csv must contain a 'Date' column (lowercased here to 'date').")

    # choose a close column
    price_col: Optional[str] = None
    for c in ("close", "adj close", "adj_close"):
        if c in df.columns:
            price_col = c
            break
    if price_col is None:
        raise ValueError("no close/adj close column found in prices.csv")

    # normalize types and sort
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.dropna(subset=["date"]).sort_values(["ticker", "date"]).reset_index(drop=True)

    # compute returns used by the model / label
    df["ret_1d"] = df.groupby("ticker")[price_col].pct_change()
    df["fwd_ret_1d"] = df.groupby("ticker")[price_col].shift(-1) / df[price_col] - 1.0

    keep = ["date", "ticker", price_col, "ret_1d", "fwd_ret_1d"]
    if "volume" in df.columns:
        keep.append("volume")

    df = df[keep].rename(columns={price_col: "close"})
    return df


def main():
    # load raw price data
    prices = _load_prices(PRICES_CSV)

    # add strictly "past" technical features (no leakage)
    feat = add_technical_features(prices)

    # choose final columns; drop rows with any missing features/label
    base_cols = [
        "date", "ticker", "close", "ret_1d", "fwd_ret_1d",
        "ret_lag_1", "ret_lag_2", "ret_lag_3", "ret_lag_5",
        "mom_5", "mom_10", "mom_20",
        "vol_5", "vol_10", "vol_20",
        "rsi_14",
    ]
    if "vol_z20" in feat.columns:
        base_cols.append("vol_z20")
    base_cols += [c for c in feat.columns if c.startswith("dow_")]  # include DOW one-hots

    ds = feat[base_cols].dropna().reset_index(drop=True)

    # persist dataset under processed/baseline
    Path(OUT_DATASET).parent.mkdir(parents=True, exist_ok=True)
    ds.to_csv(OUT_DATASET, index=False)
    print(f"Wrote {OUT_DATASET} with {len(ds):,} rows")

if __name__ == "__main__":
    main()
