#!/usr/bin/env python3
"""
Long-only statistical-arbitrage (relative-value mean-reversion) on liquid ETF pairs.

Idea
----
Classic stat-arb trades a cointegrated pair market-neutral: long the cheap leg,
short the rich leg, and profit as the spread reverts to its mean. Shorting is not
wanted here, so we use the *long-only* analogue:

    The spread  s_t = log(A_t) - beta_t * log(B_t)  mean-reverts.
      - When s is unusually LOW  (z << 0): A is cheap relative to B  -> HOLD A.
      - When s is unusually HIGH (z >> 0): B is cheap relative to A  -> HOLD B.
      - When s is back near its mean (|z| small): the edge is gone   -> HOLD CASH.
      - In between (hysteresis band): keep the current position to avoid whipsaw.

So instead of a market-neutral spread, we *rotate* our single long allocation
into whichever leg of the pair is relatively cheap, and step aside to cash once
the dislocation has closed. No leverage, no shorting.

Methodology (look-ahead safe)
-----------------------------
1. Pull daily adjusted closes for several economically-linked ETF groups.
2. FORMATION period (first FORMATION_YEARS): for every intra-group pair, run an
   Engle-Granger cointegration test and a formation-period backtest. Pick the
   cointegrated pair (p < COINT_P) with the best formation Sharpe.
3. TRADING period (everything after formation, fully out-of-sample): trade the
   chosen pair using a ROLLING hedge ratio and ROLLING z-score, both computed
   only from past data and lagged one day before being acted on.
4. Report vs. buy-and-hold each leg and a daily-rebalanced 50/50 basket.

Run:  python statarb_pairs.py
"""

from __future__ import annotations
import itertools
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import coint

warnings.simplefilter("ignore", category=FutureWarning)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
START = "2010-01-01"
END = None  # None = today

# Economically-linked ETF groups. We only test pairs WITHIN a group so that any
# cointegration found has an economic reason (avoids spurious relationships).
GROUPS = {
    "Energy":        ["XLE", "XOP", "VDE"],
    "Tech":          ["XLK", "VGT", "QQQ"],
    "Semis":         ["SMH", "SOXX"],
    "PreciousMetal": ["GLD", "SLV"],
    "GoldMiners":    ["GDX", "GDXJ"],
    "Financials":    ["XLF", "KRE", "KBE"],
    "Homebuilders":  ["XHB", "ITB"],
    "Treasuries":    ["TLT", "IEF"],
    "ConsumerDisc":  ["XLY", "XLP"],
}

FORMATION_YEARS = 4      # in-sample window used ONLY to pick the pair
LOOKBACK = 60            # trading days for rolling hedge ratio + z-score
ENTRY_Z = 1.0            # |z| above which we take the relative-value position
EXIT_Z = 0.25            # |z| below which we flatten to cash (only if ALLOW_CASH)
COST_BPS = 5.0           # one-way transaction cost in basis points per switch

# Long-only mode:
#   ALLOW_CASH = True  -> rotate A / B / CASH (step aside when spread reverts)
#   ALLOW_CASH = False -> stay fully invested, always holding the cheaper leg
#                         (mode "B": capture beta + a relative-value tilt)
ALLOW_CASH = False
COINT_P = 0.10           # max Engle-Granger p-value to consider a pair tradeable
TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def fetch_prices(tickers: list[str]) -> pd.DataFrame:
    """Daily adjusted-close prices, columns = tickers, NaNs dropped."""
    raw = yf.download(tickers, start=START, end=END, auto_adjust=True,
                      progress=False)
    px = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    px = px.dropna(how="all").ffill().dropna()
    return px


# --------------------------------------------------------------------------- #
# Signal construction (all rolling + lagged -> no look-ahead)
# --------------------------------------------------------------------------- #
def rolling_zscore(a: pd.Series, b: pd.Series, lookback: int):
    """
    Rolling hedge ratio beta and rolling z-score of the spread
        spread = log(a) - beta*log(b)
    Everything at time t uses only data up to t (the rolling windows), and the
    caller lags the resulting position by one day before trading on it.
    """
    la, lb = np.log(a), np.log(b)
    # Rolling OLS slope of la on lb:  beta = cov(la,lb)/var(lb)
    cov = la.rolling(lookback).cov(lb)
    var = lb.rolling(lookback).var()
    beta = cov / var
    spread = la - beta * lb
    z = (spread - spread.rolling(lookback).mean()) / spread.rolling(lookback).std()
    return beta, spread, z


def state_machine(z: pd.Series, entry=ENTRY_Z, exit=EXIT_Z,
                  allow_cash=ALLOW_CASH) -> pd.Series:
    """
    Map the spread z-score to a (raw, unshifted) target position with hysteresis.
      position in {+1 = hold A, -1 = hold B, 0 = cash}.
    Decision at close t; callers lag by one day before trading on it.
    """
    pos = np.zeros(len(z))
    state = 0
    zv = z.values
    for i in range(len(zv)):
        zi = zv[i]
        if np.isnan(zi):
            state = 0 if allow_cash else state   # stay invested once started
        elif zi <= -entry:
            state = 1          # A cheap -> hold A
        elif zi >= entry:
            state = -1         # B cheap -> hold B
        elif allow_cash and abs(zi) <= exit:
            state = 0          # spread reverted -> cash
        # else: keep previous state (hysteresis band)
        pos[i] = state
    return pd.Series(pos, index=z.index)


def backtest_pair(a: pd.Series, b: pd.Series,
                  lookback=LOOKBACK, entry=ENTRY_Z, exit=EXIT_Z,
                  cost_bps=COST_BPS, allow_cash=ALLOW_CASH) -> pd.DataFrame:
    """
    Long-only strategy driven by the spread z-score.
      allow_cash=True : rotate between A, B and CASH.
      allow_cash=False: stay fully invested, always holding the cheaper leg.
    Returns a DataFrame with daily strategy returns and the held position.
    """
    _, _, z = rolling_zscore(a, b, lookback)
    ret_a = a.pct_change()
    ret_b = b.pct_change()

    position = state_machine(z, entry, exit, allow_cash)

    # Trade tomorrow on today's signal (lag 1) -> no look-ahead.
    held = position.shift(1).fillna(0)
    asset_ret = np.where(held > 0, ret_a, np.where(held < 0, ret_b, 0.0))
    asset_ret = pd.Series(asset_ret, index=held.index).fillna(0.0)

    # Transaction cost whenever the held instrument changes.
    switches = (held != held.shift(1)).astype(float)
    cost = switches * (cost_bps / 1e4)
    strat_ret = asset_ret - cost

    return pd.DataFrame({"z": z, "position": held,
                         "ret": strat_ret, "switch": switches})


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def metrics(ret: pd.Series) -> dict:
    ret = ret.dropna()
    if ret.std() == 0 or len(ret) == 0:
        return dict(CAGR=0, Vol=0, Sharpe=0, MaxDD=0)
    curve = (1 + ret).cumprod()
    years = len(ret) / TRADING_DAYS
    cagr = curve.iloc[-1] ** (1 / years) - 1
    vol = ret.std() * np.sqrt(TRADING_DAYS)
    sharpe = ret.mean() / ret.std() * np.sqrt(TRADING_DAYS)
    dd = (curve / curve.cummax() - 1).min()
    return dict(CAGR=cagr, Vol=vol, Sharpe=sharpe, MaxDD=dd)


def fmt(m: dict) -> str:
    return (f"CAGR {m['CAGR']*100:6.2f}%  Vol {m['Vol']*100:6.2f}%  "
            f"Sharpe {m['Sharpe']:5.2f}  MaxDD {m['MaxDD']*100:6.2f}%")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    all_tickers = sorted({t for g in GROUPS.values() for t in g})
    print(f"Downloading {len(all_tickers)} ETFs from {START} ...")
    px = fetch_prices(all_tickers)
    print(f"  {px.index[0].date()} -> {px.index[-1].date()}  ({len(px)} days)\n")

    split = px.index[0] + pd.DateOffset(years=FORMATION_YEARS)
    form = px[px.index < split]
    trade = px[px.index >= split]
    print(f"Formation (pair selection): {form.index[0].date()} -> "
          f"{form.index[-1].date()}")
    print(f"Trading   (out-of-sample):  {trade.index[0].date()} -> "
          f"{trade.index[-1].date()}\n")

    # ---- Pair selection on formation data only -------------------------- #
    rows = []
    for group, tickers in GROUPS.items():
        present = [t for t in tickers if t in px.columns]
        for a, b in itertools.combinations(present, 2):
            fa, fb = form[a].dropna(), form[b].dropna()
            idx = fa.index.intersection(fb.index)
            if len(idx) < LOOKBACK * 3:
                continue
            try:
                pval = coint(fa[idx], fb[idx])[1]
            except Exception:
                continue
            bt = backtest_pair(fa[idx], fb[idx])
            m = metrics(bt["ret"])
            rows.append((group, a, b, pval, m["Sharpe"]))

    rank = pd.DataFrame(rows, columns=["group", "A", "B", "coint_p",
                                       "form_sharpe"])
    rank = rank.sort_values("form_sharpe", ascending=False).reset_index(drop=True)

    print("Candidate pairs (formation period):")
    print("  " + rank.to_string(index=False).replace("\n", "\n  "), "\n")

    tradeable = rank[rank["coint_p"] < COINT_P]
    if tradeable.empty:
        print(f"No pair cointegrated at p < {COINT_P}. Aborting.")
        return
    pick = tradeable.iloc[0]
    a_t, b_t = pick["A"], pick["B"]
    print(f"Selected pair: {a_t} / {b_t}  "
          f"(group={pick['group']}, coint p={pick['coint_p']:.4f}, "
          f"formation Sharpe={pick['form_sharpe']:.2f})\n")

    # ---- Out-of-sample trading ----------------------------------------- #
    a, b = trade[a_t].dropna(), trade[b_t].dropna()
    idx = a.index.intersection(b.index)
    a, b = a[idx], b[idx]
    bt = backtest_pair(a, b)

    strat = bt["ret"]
    bh_a = a.pct_change().fillna(0)
    bh_b = b.pct_change().fillna(0)
    bh_5050 = 0.5 * bh_a + 0.5 * bh_b

    n_switch = int(bt["switch"].sum())
    pct_invested = (bt["position"] != 0).mean() * 100

    mode = "rotate A/B/cash" if ALLOW_CASH else "fully-invested (cheaper leg)"
    print(f"--- OUT-OF-SAMPLE RESULTS  ({a.index[0].date()} -> "
          f"{a.index[-1].date()})   mode: {mode} ---")
    print(f"StatArb {a_t}/{b_t}   {fmt(metrics(strat))}")
    print(f"Buy&Hold {a_t:<11} {fmt(metrics(bh_a))}")
    print(f"Buy&Hold {b_t:<11} {fmt(metrics(bh_b))}")
    print(f"50/50 basket       {fmt(metrics(bh_5050))}")
    print(f"\nSwitches: {n_switch}  |  time invested: {pct_invested:.1f}%  "
          f"|  cost/switch: {COST_BPS} bps one-way")

    # ---- Save outputs --------------------------------------------------- #
    out = pd.DataFrame({
        "z": bt["z"], "position": bt["position"],
        "strat_ret": strat,
        "strat_equity": (1 + strat).cumprod(),
        f"bh_{a_t}_equity": (1 + bh_a).cumprod(),
        f"bh_{b_t}_equity": (1 + bh_b).cumprod(),
    })
    csv_path = "equity_statarb/oos_results.csv"
    out.to_csv(csv_path)
    print(f"\nSaved per-day results -> {csv_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                       gridspec_kw={"height_ratios": [3, 1]})
        out["strat_equity"].plot(ax=ax1, lw=2, label=f"StatArb {a_t}/{b_t}")
        out[f"bh_{a_t}_equity"].plot(ax=ax1, alpha=.7, label=f"B&H {a_t}")
        out[f"bh_{b_t}_equity"].plot(ax=ax1, alpha=.7, label=f"B&H {b_t}")
        ax1.set_title(f"Long-only stat-arb (OOS): {a_t}/{b_t}  [{mode}]")
        ax1.set_ylabel("Growth of $1"); ax1.legend(); ax1.grid(alpha=.3)
        out["z"].plot(ax=ax2, color="purple", lw=.8)
        for lvl in (ENTRY_Z, -ENTRY_Z):
            ax2.axhline(lvl, color="r", ls="--", lw=.6)
        for lvl in (EXIT_Z, -EXIT_Z):
            ax2.axhline(lvl, color="g", ls=":", lw=.6)
        ax2.set_ylabel("spread z"); ax2.grid(alpha=.3)
        fig.tight_layout()
        png = "equity_statarb/oos_equity.png"
        fig.savefig(png, dpi=120)
        print(f"Saved chart        -> {png}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
