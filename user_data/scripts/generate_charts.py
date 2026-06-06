#!/usr/bin/env python
"""
generate_charts.py
==================
Generates clean, professional EDUCATIONAL diagrams (one per strategy) that show the
exact indicator interaction that triggers entry/exit, with 🟢 BUY / 🔴 SELL markers.

These are illustrative concept charts (smooth synthetic data) — far clearer for
*learning* a strategy than noisy live data. Run once; the PNGs are committed to the
repo and embedded in the daily email via raw.githubusercontent.com URLs.

Run in the py310 env (has matplotlib + talib):
    conda run -n py310 python user_data/scripts/generate_charts.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import talib

OUT = Path(__file__).resolve().parents[1] / "reports" / "charts"
OUT.mkdir(parents=True, exist_ok=True)

# --- House style -----------------------------------------------------------
PRICE = "#0f172a"      # near-black
C1 = "#2563eb"         # blue
C2 = "#f59e0b"         # amber
GREEN = "#16a34a"
RED = "#dc2626"
GREY = "#94a3b8"
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.edgecolor": "#cbd5e1",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": "#eef2f6",
    "grid.linewidth": 1.0,
    "figure.dpi": 110,
})


def _style(ax, title):
    ax.set_title(title, fontsize=12, fontweight="bold", color=PRICE, loc="left", pad=10)
    ax.tick_params(labelbottom=False, labelleft=False, length=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def _buy(ax, x, y):
    ax.scatter([x], [y], marker="^", s=170, color=GREEN, zorder=6, edgecolors="white", linewidths=1.2)
    ax.annotate("BUY", (x, y), textcoords="offset points", xytext=(0, -22),
                ha="center", color=GREEN, fontweight="bold", fontsize=10)


def _sell(ax, x, y):
    ax.scatter([x], [y], marker="v", s=170, color=RED, zorder=6, edgecolors="white", linewidths=1.2)
    ax.annotate("SELL", (x, y), textcoords="offset points", xytext=(0, 14),
                ha="center", color=RED, fontweight="bold", fontsize=10)


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


def _cross_up(a, b):
    return np.where((a[1:] > b[1:]) & (a[:-1] <= b[:-1]))[0] + 1


def _cross_dn(a, b):
    return np.where((a[1:] < b[1:]) & (a[:-1] >= b[:-1]))[0] + 1


# 1 — EMA crossover --------------------------------------------------------
def ema_cross():
    x = np.arange(140)
    price = (100 + 14 * np.sin(x / 22) + x * 0.05
             + 1.5 * np.sin(x / 4))
    ef = talib.EMA(price, 8)
    es = talib.EMA(price, 21)
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.plot(x, price, color=PRICE, lw=1.2, alpha=0.45, label="Price")
    ax.plot(x, ef, color=C1, lw=2, label="Fast EMA (12)")
    ax.plot(x, es, color=C2, lw=2, label="Slow EMA (26)")
    for i in _cross_up(ef, es):
        if i > 21:
            _buy(ax, x[i], ef[i])
    for i in _cross_dn(ef, es):
        if i > 21:
            _sell(ax, x[i], ef[i])
    _style(ax, "EMA Crossover  —  fast EMA crosses the slow EMA")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    _save(fig, "ema_cross")


# 2 — Donchian breakout ----------------------------------------------------
def donchian():
    rng = np.random.default_rng(1)
    x = np.arange(150)
    price = np.concatenate([
        100 + rng.normal(0, 0.9, 80),                  # sideways range
        100 + np.linspace(0, 26, 70),                  # breakout uptrend
    ])
    upper = np.array([np.nan if i < 20 else price[i - 20:i].max() for i in range(150)])
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.plot(x, price, color=PRICE, lw=1.4, label="Price")
    ax.plot(x, upper, color=C2, lw=2, ls="--", label="Donchian-20 high")
    bo = next((i for i in range(81, 150) if price[i] > upper[i]), None)
    if bo is not None:
        _buy(ax, x[bo], price[bo])
    _style(ax, "Donchian Breakout  —  price closes above the N-bar high")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    _save(fig, "donchian")


# 3 — Bollinger mean reversion --------------------------------------------
def mean_reversion():
    x = np.arange(140)
    price = 100 + 6 * np.sin(x / 12) + np.random.default_rng(2).normal(0, 0.6, 140)
    mid = talib.SMA(price, 20)
    sd = talib.STDDEV(price, 20)
    up, lo = mid + 2 * sd, mid - 2 * sd
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.fill_between(x, lo, up, color=C1, alpha=0.08)
    ax.plot(x, price, color=PRICE, lw=1.4, label="Price")
    ax.plot(x, mid, color=C2, lw=1.6, label="Bollinger mid")
    ax.plot(x, up, color=C1, lw=1.2, alpha=0.7)
    ax.plot(x, lo, color=C1, lw=1.2, alpha=0.7, label="Bollinger bands")
    dips = [i for i in range(21, 140) if price[i] < lo[i]]
    if dips:
        b = dips[len(dips) // 3]
        _buy(ax, x[b], price[b])
        s = next((i for i in range(b + 1, 140) if price[i] >= mid[i]), None)
        if s:
            _sell(ax, x[s], price[s])
    _style(ax, "Mean Reversion  —  buy the dip below the lower band, sell at the mean")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    _save(fig, "mean_reversion")


# 4 — Supertrend -----------------------------------------------------------
def _supertrend_np(high, low, close, period=10, mult=3.0):
    """Self-contained Supertrend (same logic as the strategy) — no freqtrade dep."""
    atr = talib.ATR(high, low, close, timeperiod=period)
    hl2 = (high + low) / 2
    bub, blb = hl2 + mult * atr, hl2 - mult * atr
    n = len(close)
    fub, flb = bub.copy(), blb.copy()
    valid = np.where(~np.isnan(atr))[0]
    first = int(valid[0]) if len(valid) else 0
    st, d = np.full(n, np.nan), np.zeros(n, int)
    for i in range(first + 1, n):
        fub[i] = bub[i] if (bub[i] < fub[i - 1] or close[i - 1] > fub[i - 1]) else fub[i - 1]
        flb[i] = blb[i] if (blb[i] > flb[i - 1] or close[i - 1] < flb[i - 1]) else flb[i - 1]
    st[first], d[first] = fub[first], -1
    for i in range(first + 1, n):
        if st[i - 1] == fub[i - 1]:
            st[i], d[i] = (fub[i], -1) if close[i] <= fub[i] else (flb[i], 1)
        else:
            st[i], d[i] = (flb[i], 1) if close[i] >= flb[i] else (fub[i], -1)
    return st, d


def supertrend_chart():
    rng = np.random.default_rng(5)
    x = np.arange(160)
    close = 100 + 16 * np.sin(x / 30) + x * 0.03 + rng.normal(0, 0.6, 160)
    st, d = _supertrend_np(close + 1.2, close - 1.2, close, 10, 3.0)
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.plot(x, close, color=PRICE, lw=1.4, label="Price")
    up = np.where(d == 1, st, np.nan)
    dn = np.where(d == -1, st, np.nan)
    ax.plot(x, up, color=GREEN, lw=2.2, label="Supertrend (up)")
    ax.plot(x, dn, color=RED, lw=2.2, label="Supertrend (down)")
    for i in range(21, 160):
        if d[i] == 1 and d[i - 1] == -1:
            _buy(ax, x[i], close[i])
        if d[i] == -1 and d[i - 1] == 1:
            _sell(ax, x[i], close[i])
    _style(ax, "Supertrend  —  buy/sell when price flips the ATR trend line")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    _save(fig, "supertrend")


# 5 — Bollinger/Keltner squeeze -------------------------------------------
def squeeze():
    x = np.arange(150)
    vol = np.concatenate([np.linspace(3, 0.6, 75), np.linspace(0.6, 4, 75)])  # contract then expand
    price = 100 + np.cumsum(np.random.default_rng(7).normal(0, 0.15, 150)) + np.linspace(0, 4, 150)
    mid = talib.SMA(price, 20)
    bb_u, bb_l = mid + 2 * vol, mid - 2 * vol
    kc_u, kc_l = mid + 1.5 * 2.2, mid - 1.5 * 2.2  # ~constant Keltner width
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.plot(x, price, color=PRICE, lw=1.3, label="Price")
    ax.plot(x, bb_u, color=C1, lw=1.4, label="Bollinger")
    ax.plot(x, bb_l, color=C1, lw=1.4)
    ax.plot(x, kc_u, color=C2, lw=1.4, ls="--", label="Keltner")
    ax.plot(x, kc_l, color=C2, lw=1.4, ls="--")
    # squeeze = BB inside KC
    on = (bb_u < kc_u) & (bb_l > kc_l)
    if on.any():
        xs = np.where(on)[0]
        ax.axvspan(xs[0], xs[-1], color=GREY, alpha=0.15)
        ax.annotate("SQUEEZE\n(coiled)", (xs[len(xs) // 2], mid[xs[len(xs) // 2]]),
                    ha="center", va="center", color="#475569", fontsize=9, fontweight="bold")
        rel = xs[-1] + 1
        if rel < 150:
            _buy(ax, x[rel], price[rel])
    _style(ax, "Squeeze Breakout  —  Bollinger coils inside Keltner, then expands")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    _save(fig, "squeeze")


# 6 — DCA mean reversion ---------------------------------------------------
def dca():
    x = np.arange(140)
    price = np.concatenate([
        100 - np.linspace(0, 12, 70),                  # stair-step down
        88 + np.linspace(0, 16, 70),                   # recovery
    ]) + np.random.default_rng(9).normal(0, 0.4, 140)
    mid = talib.SMA(price, 20)
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.plot(x, price, color=PRICE, lw=1.4, label="Price")
    ax.plot(x, mid, color=C2, lw=1.6, label="Mean (BB mid)")
    for b in (28, 45, 62):                              # scale-in tranches
        _buy(ax, x[b], price[b])
    s = next((i for i in range(80, 140) if price[i] >= mid[i]), 120)
    _sell(ax, x[s], price[s])
    ax.annotate("average down\n(3 tranches)", (45, price[45]), textcoords="offset points",
                xytext=(10, -40), color=GREEN, fontsize=9)
    _style(ax, "DCA Mean Reversion  —  scale into dips, exit on the bounce")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    _save(fig, "dca")


if __name__ == "__main__":
    ema_cross()
    donchian()
    mean_reversion()
    supertrend_chart()
    squeeze()
    dca()
    print("done — 6 charts in", OUT)
