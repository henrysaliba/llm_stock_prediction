from pathlib import Path
import numpy as np
import pandas as pd

# input/output roots
BASE = Path("data/processed/baseline")
HYB  = Path("data/processed/hybrid")
OUT  = Path("data/processed/hybrid")  # write comparisons next to hybrid outputs

def _end_equity(df):
    # final equity level (normalised PnL proxy)
    return float(df["equity"].iloc[-1]) if len(df) else np.nan

def _daily_vol(df):
    # annualised volatility from daily strategy returns
    r = df["row_ret"].values
    return float(np.std(r, ddof=1) * np.sqrt(252)) if len(r) > 1 else np.nan

def _sharpe(df, rf=0.0):
    # annualised Sharpe using daily returns and rf (per year)
    r = df["row_ret"].values
    mu = np.mean(r) - rf/252
    sd = np.std(r, ddof=1)
    return float(np.sqrt(252) * (mu / (sd + 1e-12))) if len(r) > 1 else np.nan

def _sortino(df, rf=0.0):
    # annualised Sortino (downside risk only)
    r = df["row_ret"].values
    dn = r[r < 0]
    dd = np.std(dn, ddof=1) if len(dn) > 1 else np.nan
    mu = np.mean(r) - rf/252
    return float(np.sqrt(252) * (mu / (dd + 1e-12))) if len(r) > 1 else np.nan

def _max_drawdown(eq):
    # max drawdown from equity curve
    peak = np.maximum.accumulate(eq)
    dd = (eq / peak) - 1.0
    return float(dd.min())

def _cagr(df):
    # CAGR from start to end over trading days
    if len(df) < 2: return np.nan
    eq = df["equity"].values
    days = len(df)
    return float(eq[-1]**(252/days) - 1.0)

def load_curve(path):
    # read equity CSV with parsed dates
    return pd.read_csv(path, parse_dates=["date"])

def summarize_curve(model, tag):
    # compute portfolio metrics for a given model/tag combo
    path = (BASE if model=="baseline" else HYB) / f"equity_{model}_{tag}.csv"
    d = load_curve(path)
    out = {
        "model": model,
        "tag": tag,
        "end_equity": _end_equity(d),
        "CAGR": _cagr(d),
        "ann_vol": _daily_vol(d),
        "Sharpe": _sharpe(d),
        "Sortino": _sortino(d),
        "MaxDD": _max_drawdown(d["equity"].values),
        "n_days": len(d),
    }
    return out

def main():
    # regression summary (OOF MAE/R2) to comparison table
    bsum = pd.read_csv(BASE / "model_summary_baseline.csv")
    hsum = pd.read_csv(HYB  / "model_summary_hybrid.csv")
    reg = pd.concat([bsum.assign(model="baseline"), hsum.assign(model="hybrid")], ignore_index=True)
    reg.to_csv(OUT / "compare_model_summaries.csv", index=False)

    # portfolio metrics for long_only and long_short to comparison table
    rows = []
    for m in ["baseline", "hybrid"]:
        for tag in ["long_only", "long_short"]:
            rows.append(summarize_curve(m, tag))
    perf = pd.DataFrame(rows).sort_values(["tag","model"])
    perf.to_csv(OUT / "compare_portfolio_perf.csv", index=False)

    # pretty print for slide notes
    print("\n=== Regression (OOF) ===")
    print(reg.to_string(index=False))
    print("\n=== Portfolio metrics ===")
    print(perf.to_string(index=False))

if __name__ == "__main__":
    main()
