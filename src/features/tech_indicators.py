from __future__ import annotations

import numpy as np
import pandas as pd

"""
this module creates price-only technical features for the baseline model.
all features are lagged/rolling -> avoids look-ahead leakage into the label.
"""

def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
   # simple RSI. Returned in 0..100; we later divide by 100 for stability
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # rolling means for gains/losses
    avg_gain = gain.rolling(window, min_periods=window).mean()
    avg_loss = loss.rolling(window, min_periods=window).mean()

    # protect division by zero (add small epsilon)
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    # add rolling/lags per ticker. 
      
    df = df.sort_values(["ticker", "date"]).copy()
    out = []

    for tkr, g in df.groupby("ticker", group_keys=False):
        g = g.sort_values("date").copy()

        # past return lags (strictly historical)
        for k in [1, 2, 3, 5]:
            g[f"ret_lag_{k}"] = g["ret_1d"].shift(k)

        # momentum vs SMA (relative distance to moving average)
        for w in [5, 10, 20]:
            sma = g["close"].rolling(w, min_periods=w).mean()
            g[f"mom_{w}"] = (g["close"] / sma) - 1.0

        # realized volatility of daily returns
        for w in [5, 10, 20]:
            g[f"vol_{w}"] = g["ret_1d"].rolling(w, min_periods=w).std()

        # RSI (scale 0..1 so trees don't over-weight it)
        g["rsi_14"] = _rsi(g["close"], 14) / 100.0

        # volume z-score
        if "volume" in g.columns:
            mu = g["volume"].rolling(20, min_periods=10).mean()
            sd = g["volume"].rolling(20, min_periods=10).std()
            g["vol_z20"] = (g["volume"] - mu) / (sd + 1e-9)

        # simple calendar feature: day-of-week one-hot (0=Mon baseline)
        g["dow"] = pd.to_datetime(g["date"]).dt.dayofweek  # 0..6
        g = pd.get_dummies(g, columns=["dow"], prefix="dow", drop_first=True)

        out.append(g)

    df2 = pd.concat(out, axis=0).sort_values(["ticker", "date"]).reset_index(drop=True)

    # warmup drop: require enough history so base features are valid
    needed = ["ret_lag_5", "mom_20", "vol_20", "rsi_14"]
    df2 = df2.dropna(subset=[c for c in needed if c in df2.columns])

    return df2
