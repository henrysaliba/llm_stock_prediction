from __future__ import annotations

"""
Train a baseline (price-only) model and run a walk-forward backtest.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


## paths (baseline-specific folders)

DATASET = "data/processed/baseline/model_dataset_baseline.csv"

OUT_PROC = Path("data/processed/baseline") # all CSVs belong here
OUT_FIG  = Path("data/processed/graphs/baseline") # all baseline PNGs here
for p in (OUT_PROC, OUT_FIG):
    p.mkdir(parents=True, exist_ok=True)


## helpers

def _time_folds(n: int, n_splits: int = 5):
    """Expanding-window folds to respect time ordering and reduce leakage."""
    block = n // (n_splits + 1)
    for k in range(1, n_splits + 1):
        end_train = block * k
        start_test = end_train
        end_test = block * (k + 1) if k < n_splits else n
        if end_test - start_test < 10:  # guard against tiny last block
            continue
        yield np.arange(0, end_train), np.arange(start_test, end_test)


def _equity_curve(df: pd.DataFrame, allow_short: bool, cost_bps: float = 1.0) -> pd.DataFrame:
    """
    portfolio-level curve (equal-weight across tickers).
    df must have: [date, ticker, fwd_ret_1d, pred]
    """

    up_th = 0.001  # +/-10 bps thresholds -> simple & transparent
    dn_th = 0.001

    def to_signal(p):
        if p > up_th: return 1
        if p < -dn_th: return -1 if allow_short else 0
        return 0

    df = df.sort_values(["date","ticker"]).copy()
    df["signal"] = df["pred"].apply(to_signal)

    # transaction costs when signal regime changes
    df["prev_signal"] = df.groupby("ticker")["signal"].shift(1).fillna(0)
    trades = (df["signal"] != df["prev_signal"]).astype(float)
    tc = cost_bps * 1e-4  # 1 bps = 0.0001

    # per-row strategy return
    df["row_ret"] = (df["signal"] * df["fwd_ret_1d"]) - (trades * tc)

    # daily equal-weight across tickers
    daily = df.groupby("date")["row_ret"].mean().reset_index()
    daily["equity"] = (1.0 + daily["row_ret"]).cumprod()
    return daily


def _equity_curve_single_ticker(g: pd.DataFrame, allow_short: bool, cost_bps: float = 1.0) -> pd.DataFrame:
    # same logic as _equity_curve, but for a single ticker slice
    up_th = 0.001
    dn_th = 0.001

    def to_signal(p):
        if p > up_th: return 1
        if p < -dn_th: return -1 if allow_short else 0
        return 0

    g = g.sort_values("date").copy()
    g["signal"] = g["pred"].apply(to_signal)

    g["prev_signal"] = g["signal"].shift(1).fillna(0)
    trades = (g["signal"] != g["prev_signal"]).astype(float)
    tc = cost_bps * 1e-4

    g["row_ret"] = (g["signal"] * g["fwd_ret_1d"]) - (trades * tc)

    daily = g[["date", "row_ret"]].copy()
    daily["equity"] = (1.0 + daily["row_ret"]).cumprod()
    return daily


def _buy_and_hold_curve_single_ticker(g: pd.DataFrame) -> pd.DataFrame:
    # buy-and-hold reference over the same dates as OOF window
    g = g.sort_values("date").copy()
    bh = g[["date"]].copy()
    bh["equity"] = (1.0 + g["fwd_ret_1d"]).cumprod()
    return bh


def _plot_curve(daily: pd.DataFrame, title: str, outpath: Path):
    # save a simple equity curve plot
    plt.figure(figsize=(9, 5))
    plt.plot(daily["date"], daily["equity"], linewidth=2)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Equity (normalised)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def _plot_curve_with_benchmark(curve: pd.DataFrame, bh: pd.DataFrame, title: str, outpath: Path):
    # strategy curve + dashed buy-and-hold for visual context
    plt.figure(figsize=(9, 5))
    plt.plot(curve["date"], curve["equity"], linewidth=2, label="Strategy")
    plt.plot(bh["date"], bh["equity"], linewidth=1.5, linestyle="--", label="Buy & Hold")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Equity (normalised)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


## main 

def main():
    # load dataset and order chronologically
    df = pd.read_csv(DATASET)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    # select features (purely technical; no sentiment here)
    base_feats = [
        "ret_lag_1", "ret_lag_2", "ret_lag_3", "ret_lag_5",
        "mom_5", "mom_10", "mom_20",
        "vol_5", "vol_10", "vol_20",
        "rsi_14",
    ]
    if "vol_z20" in df.columns:
        base_feats.append("vol_z20")
    base_feats += [c for c in df.columns if c.startswith("dow_")]

    X = df[base_feats].values
    y = df["fwd_ret_1d"].values

    # walk-forward CV -> OOF predictions
    oof = np.full(len(df), np.nan)
    fold_rows = []

    for tr_idx, te_idx in _time_folds(len(df), n_splits=5):
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xte, yte = X[te_idx], y[te_idx]

        # RandomForest: robust, fast, no scaling needed
        model = RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        oof[te_idx] = pred

        fold_rows.append({
            "fold_train_rows": int(len(tr_idx)),
            "fold_test_rows": int(len(te_idx)),
            "MAE": float(mean_absolute_error(yte, pred)),
            "R2": float(r2_score(yte, pred)),
        })

    # persist CV diagnostics + headline metrics under processed/baseline/
    cv = pd.DataFrame(fold_rows)
    cv.to_csv(OUT_PROC / "cv_metrics_baseline.csv", index=False)

    mask = ~np.isnan(oof)
    mae = mean_absolute_error(y[mask], oof[mask])
    r2 = r2_score(y[mask], oof[mask])
    pd.DataFrame([{"model": "baseline_rf", "MAE": mae, "R2": r2}]) \
        .to_csv(OUT_PROC / "model_summary_baseline.csv", index=False)

    # build OOF prediction frame for backtests (portfolio + per-ticker)
    pred_df = df.loc[mask, ["date", "ticker", "fwd_ret_1d"]].copy()
    pred_df["pred"] = oof[mask]

    # portfolio-level curves (equal-weight across tickers) 
    for allow_short in [False, True]:
        tag = "long_only" if not allow_short else "long_short"
        curve = _equity_curve(pred_df, allow_short=allow_short, cost_bps=1.0)
        curve.to_csv(OUT_PROC / f"equity_baseline_{tag}.csv", index=False)
        _plot_curve(
            curve,
            title=f"baseline RF | {tag} | walk-forward OOF equity",
            outpath=OUT_FIG / f"equity_baseline_{tag}.png"
        )

    # per-ticker metrics (MAE/R2) 
    rows = []
    for tkr, g in pred_df.groupby("ticker"):
        y_true = g["fwd_ret_1d"].values
        y_pred = g["pred"].values
        rows.append({
            "ticker": tkr,
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "R2": float(r2_score(y_true, y_pred)),
            "n_obs": int(len(g)),
        })
    per_tkr = pd.DataFrame(rows).sort_values("ticker")
    per_tkr.to_csv(OUT_PROC / "per_ticker_metrics_baseline.csv", index=False)

    # per-ticker equity curves + buy-and-hold overlays 
    for tkr, g in pred_df.groupby("ticker"):
        bh = _buy_and_hold_curve_single_ticker(g)

        for allow_short in [False, True]:
            tag = "long_only" if not allow_short else "long_short"
            curve = _equity_curve_single_ticker(g, allow_short=allow_short, cost_bps=1.0)

            # save CSVs under processed/baseline
            curve.to_csv(OUT_PROC / f"equity_baseline_{tkr}_{tag}.csv", index=False)

            # save PNGs under graphs/baseline
            _plot_curve_with_benchmark(
                curve, bh,
                title=f"{tkr} | baseline RF | {tag}",
                outpath=OUT_FIG / f"equity_baseline_{tkr}_{tag}.png"
            )

    print("Finished baseline training + backtest. Graphs formulated.")

if __name__ == "__main__":
    main()
