from __future__ import annotations
"""
Train a hybrid (price + sentiment) model and run a walk-forward backtest.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# input/output paths
DATASET = "data/processed/hybrid/model_dataset_hybrid.csv"

OUT_PROC = Path("data/processed/hybrid") # all hybrid CSVs
OUT_FIG  = Path("data/processed/graphs/hybrid") # all hybrid PNGs
for p in (OUT_PROC, OUT_FIG):
    p.mkdir(parents=True, exist_ok=True)

def _time_folds(n: int, n_splits: int = 5):
    # expanding-window folds to respect time order
    block = n // (n_splits + 1)
    for k in range(1, n_splits + 1):
        end_train = block * k
        start_test = end_train
        end_test = block * (k + 1) if k < n_splits else n
        if end_test - start_test < 10:  # avoid tiny last fold
            continue
        yield np.arange(0, end_train), np.arange(start_test, end_test)

def _equity_curve(df: pd.DataFrame, allow_short: bool, cost_bps: float = 1.0) -> pd.DataFrame:
    # portfolio equity from OOF predictions (equal-weight across tickers)
    up_th = 0.001; dn_th = 0.001  # +/-10 bps thresholds

    def to_signal(p):
        # map regression output to long/short/flat signal
        if p > up_th: return 1
        if p < -dn_th: return -1 if allow_short else 0
        return 0

    df = df.sort_values(["date","ticker"]).copy()
    df["signal"] = df["pred"].apply(to_signal)

    # costs on regime change only
    df["prev_signal"] = df.groupby("ticker")["signal"].shift(1).fillna(0)
    trades = (df["signal"] != df["prev_signal"]).astype(float)
    tc = cost_bps * 1e-4

    # row-level strategy return then aggregate to daily mean
    df["row_ret"] = (df["signal"] * df["fwd_ret_1d"]) - (trades * tc)
    daily = df.groupby("date")["row_ret"].mean().reset_index()
    daily["equity"] = (1.0 + daily["row_ret"]).cumprod()
    return daily

def _plot_curve(curve: pd.DataFrame, title: str, outpath: Path):
    # simple equity plot
    plt.figure(figsize=(9,5))
    plt.plot(curve["date"], curve["equity"], linewidth=2)
    plt.title(title); plt.xlabel("Date"); plt.ylabel("Equity (normalised)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(outpath, dpi=160); plt.close()

def _equity_curve_single_ticker(g, allow_short: bool, cost_bps: float = 1.0):
    # single-ticker equity curve (same logic as portfolio)
    up_th = 0.001; dn_th = 0.001

    def to_signal(p):
        if p > up_th: return 1
        if p < -dn_th: return -1 if allow_short else 0
        return 0

    g = g.sort_values("date").copy()
    g["signal"] = g["pred"].apply(to_signal)

    # costs on regime change
    g["prev_signal"] = g["signal"].shift(1).fillna(0)
    trades = (g["signal"] != g["prev_signal"]).astype(float)
    tc = cost_bps * 1e-4

    # returns and cumulative equity
    g["row_ret"] = (g["signal"] * g["fwd_ret_1d"]) - (trades * tc)
    daily = g[["date","row_ret"]].copy()
    daily["equity"] = (1.0 + daily["row_ret"]).cumprod()
    return daily

def _buy_and_hold_curve_single_ticker(g):
    # long-only buy & hold reference for the same dates
    g = g.sort_values("date").copy()
    bh = g[["date"]].copy()
    bh["equity"] = (1.0 + g["fwd_ret_1d"]).cumprod()
    return bh

def _plot_curve_with_benchmark(curve, bh, title, outpath):
    # strategy vs buy & hold overlay
    plt.figure(figsize=(9,5))
    plt.plot(curve["date"], curve["equity"], linewidth=2, label="Strategy")
    plt.plot(bh["date"], bh["equity"], linewidth=1.5, linestyle="--", label="Buy & Hold")
    plt.title(title); plt.xlabel("Date"); plt.ylabel("Equity (normalised)")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.savefig(outpath, dpi=160); plt.close()

def main():
    # load dataset and sort by time, then ticker
    df = pd.read_csv(DATASET)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date","ticker"]).reset_index(drop=True)

    # technical features (same as baseline)
    base_feats = [
        "ret_lag_1","ret_lag_2","ret_lag_3","ret_lag_5",
        "mom_5","mom_10","mom_20",
        "vol_5","vol_10","vol_20",
        "rsi_14",
    ]
    if "vol_z20" in df.columns:
        base_feats.append("vol_z20")
    base_feats += [c for c in df.columns if c.startswith("dow_")]  # calendar one-hots

    # sentiment features (daily + rolling + missingness flags)
    sent_feats = [
        "sentiment_mean","sentiment_sum","news_count",
        "sent_mean_roll3","sent_mean_roll5","sent_mean_roll7",
        "sent_sum_roll3","sent_sum_roll5","sent_sum_roll7",
        "news_count_roll3","news_count_roll5","news_count_roll7",
    ] + [c for c in df.columns if c.endswith("_isna")]

    # final feature set
    features = base_feats + sent_feats

    X = df[features].values
    y = df["fwd_ret_1d"].values  # next-day return target

    # walk-forward CV -> OOF predictions
    oof = np.full(len(df), np.nan)
    fold_rows = []
    for tr_idx, te_idx in _time_folds(len(df), n_splits=5):
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xte, yte = X[te_idx], y[te_idx]

        # RF regressor; slightly different hyperparams from baseline
        model = RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        oof[te_idx] = pred  # store OOF slice

        # fold diagnostics
        fold_rows.append({
            "fold_train_rows": int(len(tr_idx)),
            "fold_test_rows": int(len(te_idx)),
            "MAE": float(mean_absolute_error(yte, pred)),
            "R2": float(r2_score(yte, pred)),
        })

    # save CV diagnostics
    cv = pd.DataFrame(fold_rows)
    cv.to_csv(OUT_PROC / "cv_metrics_hybrid.csv", index=False)

    # overall OOF metrics
    mask = ~np.isnan(oof)
    mae = mean_absolute_error(y[mask], oof[mask])
    r2 = r2_score(y[mask], oof[mask])
    pd.DataFrame([{"model":"hybrid_rf","MAE":mae,"R2":r2,"n_obs":int(mask.sum())}]).to_csv(
        OUT_PROC / "model_summary_hybrid.csv", index=False)

    # OOF prediction frame for backtests and per-ticker metrics
    pred_df = df.loc[mask, ["date","ticker","fwd_ret_1d"]].copy()
    pred_df["pred"] = oof[mask]

    # portfolio-level curves (equal-weight across tickers)
    for allow_short in [False, True]:
        tag = "long_only" if not allow_short else "long_short"
        curve = _equity_curve(pred_df, allow_short=allow_short, cost_bps=1.0)
        curve.to_csv(OUT_PROC / f"equity_hybrid_{tag}.csv", index=False)
        _plot_curve(curve, f"hybrid RF | {tag} | walk-forward OOF equity",
                    OUT_FIG / f"equity_hybrid_{tag}.png")

    # per-ticker equity curves + buy-and-hold overlays (mirror baseline)
    for tkr, g in pred_df.groupby("ticker"):
        bh = _buy_and_hold_curve_single_ticker(g)
        for allow_short in [False, True]:
            tag = "long_only" if not allow_short else "long_short"
            curve = _equity_curve_single_ticker(g, allow_short=allow_short, cost_bps=1.0)

            # CSVs under processed/hybrid
            curve.to_csv(OUT_PROC / f"equity_hybrid_{tkr}_{tag}.csv", index=False)

            # PNGs under graphs/hybrid
            _plot_curve_with_benchmark(
                curve, bh,
                title=f"{tkr} | hybrid RF | {tag}",
                outpath=OUT_FIG / f"equity_hybrid_{tkr}_{tag}.png"
            )

    # per-ticker diagnostics (OOF MAE/R2 per name)
    rows = []
    for tkr, g in pred_df.groupby("ticker"):
        y_true = g["fwd_ret_1d"].values; y_pred = g["pred"].values
        rows.append({"ticker":tkr,
                     "MAE":float(mean_absolute_error(y_true, y_pred)),
                     "R2":float(r2_score(y_true, y_pred)),
                     "n_obs":int(len(g))})
    pd.DataFrame(rows).sort_values("ticker").to_csv(OUT_PROC / "per_ticker_metrics_hybrid.csv", index=False)

    print("Finished hybrid training + backtest. Graphs formulated.")

if __name__ == "__main__":
    main()
