import os
from typing import List
import pandas as pd
import yfinance as yf
from ..utils.config import START_DATE, END_DATE


def download_prices(tickers: List[str]) -> pd.DataFrame:
    # fetch daily prices from yfinance
    df = yf.download(
        tickers=tickers,
        start=START_DATE,
        end=END_DATE,
        interval="1d",  # change interval if needed
        group_by="ticker",
        auto_adjust=True,
        progress=False,
    )

    # handle multi-index columns (multiple tickers)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.stack(0).reset_index().rename(columns={"level_1": "ticker"})
    else:
        # single ticker case
        df = df.reset_index()
        df["ticker"] = tickers[0]

    return df


def save_prices(df: pd.DataFrame, out_path: str):
    # ensure output dir exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # write to csv
    df.to_csv(out_path, index=False)


if __name__ == "__main__":
    from ..utils.config import TICKERS

    # download and save prices
    out = download_prices(TICKERS)
    save_prices(out, "data/raw/prices.csv")
    print("saved data/raw/prices.csv")
