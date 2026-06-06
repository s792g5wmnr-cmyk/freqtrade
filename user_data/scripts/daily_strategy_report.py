#!/usr/bin/env python
"""
daily_strategy_report.py
========================
Daily comparison of the three BTC strategies. For each run it:
  1. Downloads the latest BTC/USDT data (1h + 4h) via freqtrade.
  2. Backtests all three strategies over a rolling window (default: last 365 days).
  3. Ranks them and picks the "best" (by total return, tie-break profit factor).
  4. Computes the CURRENT trade signal + levels from the latest closed candle.
  5. Writes a dated Markdown report and prints a summary. Optionally emails it.

Run (must use the freqtrade conda env which has freqtrade + ta-lib installed):
    conda run -n freqtrade python user_data/scripts/daily_strategy_report.py

Email (optional): set these env vars to enable SMTP delivery.
    REPORT_EMAIL_TO, SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS
If they are unset, the script just writes the report file (no email).
"""
from __future__ import annotations

import os
import smtplib
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import numpy as np
import pandas as pd
import talib
import talib.abstract as taa
from technical import qtpylib

# --- Paths -----------------------------------------------------------------
REPO = Path(__file__).resolve().parents[2]          # .../freqtrade
CONFIG = REPO / "user_data" / "config.json"
DATA_DIR = REPO / "user_data" / "data" / "binanceus"
REPORTS = REPO / "user_data" / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

# Reuse the Supertrend helper from the strategy module (DRY, identical logic).
sys.path.insert(0, str(REPO / "user_data" / "strategies"))
from MtfSupertrendStrategy import supertrend as _supertrend  # noqa: E402

STRATEGIES = [
    "BtcEmaRsiStrategy", "BtcTrendRiderStrategy", "BtcDipBuyerStrategy",
    "MtfSupertrendStrategy", "SqueezeBreakoutStrategy", "DcaMeanReversionStrategy",
]
# Coins to report on. label = how it's shown; emoji for the section header.
PAIRS = [
    {"pair": "BTC/USDT", "name": "Bitcoin", "emoji": "₿"},
    {"pair": "ETH/USDT", "name": "Ethereum", "emoji": "Ξ"},
]
ALL_PAIRS = [p["pair"] for p in PAIRS]
ROLLING_DAYS = 365


def sh(cmd: list[str]) -> subprocess.CompletedProcess:
    """Run a command from the repo root, raising on failure."""
    return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, check=True)


# --- 1. Download latest data ----------------------------------------------
def download_data() -> str:
    """Download data for every pair and return the backtest timerange (the rolling
    window). The daily (1d) timeframe gets a much longer lead-in so the daily EMA200
    used by MtfSupertrend is fully warmed up at the start of the backtest window."""
    end = datetime.now(timezone.utc).date()
    leads = {"1h": ROLLING_DAYS + 30, "4h": ROLLING_DAYS + 30, "1d": ROLLING_DAYS + 260}
    for tf, lead in leads.items():
        start = end - timedelta(days=lead)
        sh([
            "freqtrade", "download-data", "--config", str(CONFIG),
            "--pairs", *ALL_PAIRS, "--timeframe", tf,
            "--timerange", f"{start.strftime('%Y%m%d')}-{end.strftime('%Y%m%d')}",
        ])
    start = end - timedelta(days=ROLLING_DAYS + 30)
    return f"{start.strftime('%Y%m%d')}-{end.strftime('%Y%m%d')}"


# --- 2. Backtest all strategies, per pair ---------------------------------
def run_backtests(timerange: str) -> dict:
    """Backtest all strategies for each pair separately (so every per-coin metric
    — return, drawdown, profit factor, Sharpe — is complete). Returns
    {pair: {strategy: metrics}}."""
    from freqtrade.data.btanalysis import load_backtest_stats
    out = {}
    for pair in ALL_PAIRS:
        sh([
            "freqtrade", "backtesting", "--config", str(CONFIG),
            "--strategy-list", *STRATEGIES,
            "--pairs", pair, "--timerange", timerange,
            "--export", "trades", "--cache", "none",
        ])
        stats = load_backtest_stats(str(REPO / "user_data" / "backtest_results"))
        out[pair] = {}
        for name, s in stats.get("strategy", {}).items():
            wins = s.get("wins")
            total = s.get("total_trades") or 0
            out[pair][name] = {
                "profit_pct": _pct(s.get("profit_total")),
                "max_dd_pct": _pct(s.get("max_drawdown_account")),
                "profit_factor": s.get("profit_factor"),
                "sharpe": s.get("sharpe"),
                "trades": total,
                "winrate": (wins / total * 100) if (wins is not None and total) else None,
                "market_change_pct": _pct(s.get("market_change")),
            }
    return out


def _pct(v):
    return round(v * 100, 2) if isinstance(v, (int, float)) else None


# --- 3+4. Current signal for each strategy from the latest closed candle ---
def load_ohlcv(tf: str, pair: str = "BTC/USDT") -> pd.DataFrame:
    fname = pair.replace("/", "_")
    df = pd.read_feather(DATA_DIR / f"{fname}-{tf}.feather")
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.set_index("date").sort_index()


def compute_signals(pair: str) -> dict:
    """All six strategies' live signals for one pair."""
    df1h = load_ohlcv("1h", pair)
    df4h = load_ohlcv("4h", pair)
    df1d = load_ohlcv("1d", pair)
    return {
        "BtcEmaRsiStrategy": signal_ema_rsi(df1h),
        "BtcTrendRiderStrategy": signal_trend_rider(df4h),
        "BtcDipBuyerStrategy": signal_dip_buyer(df4h),
        "MtfSupertrendStrategy": signal_mtf_supertrend(df4h, df1d),
        "SqueezeBreakoutStrategy": signal_squeeze(df1h),
        "DcaMeanReversionStrategy": signal_dca(df4h),
    }


def crossed_below(a, b):
    return (a.iloc[-1] < b.iloc[-1]) and (a.iloc[-2] >= b.iloc[-2])


def signal_ema_rsi(df: pd.DataFrame) -> dict:
    c = df["close"].values
    ema_f = talib.EMA(c, 12)
    ema_s = talib.EMA(c, 26)
    ema_t = talib.EMA(c, 200)
    rsi = talib.RSI(c, 14)
    adx = talib.ADX(df["high"].values, df["low"].values, c, 14)
    price = c[-1]
    entry = (ema_f[-1] > ema_s[-1] and ema_f[-2] <= ema_s[-2]
             and price > ema_t[-1] and 50 < rsi[-1] < 70 and adx[-1] > 20)
    exit_ = ((ema_f[-1] < ema_s[-1] and ema_f[-2] >= ema_s[-2])
             or (rsi[-1] > 78 and rsi[-2] <= 78))
    return {
        "timeframe": "1h", "price": price,
        "entry_signal": bool(entry),
        "exit_signal": bool(exit_),
        "entry_trigger": "EMA12 crosses > EMA26, price > EMA200, RSI 50-70, ADX>20",
        "exit_trigger": f"EMA12<EMA26 or RSI>78 (RSI now {rsi[-1]:.0f})",
        "stop_level": price * (1 - 0.10),
        "context": f"EMA12={ema_f[-1]:.0f} EMA26={ema_s[-1]:.0f} EMA200={ema_t[-1]:.0f} ADX={adx[-1]:.0f}",
    }


def signal_trend_rider(df: pd.DataFrame) -> dict:
    c = df["close"].values
    ema_f = talib.EMA(c, 50)
    ema_s = talib.EMA(c, 200)
    adx = talib.ADX(df["high"].values, df["low"].values, c, 14)
    donch = df["high"].rolling(20).max().shift(1).values
    price = c[-1]
    entry = (price > ema_s[-1] and ema_f[-1] > ema_s[-1]
             and price > donch[-1] and adx[-1] > 20)
    exit_ = (c[-1] < ema_f[-1] and c[-2] >= ema_f[-2])  # close crosses below EMA50
    return {
        "timeframe": "4h", "price": price,
        "entry_signal": bool(entry),
        "exit_signal": bool(exit_),
        "entry_trigger": f"close > Donchian-20 high ({donch[-1]:.0f}) in uptrend (price>EMA200)",
        "exit_trigger": f"close crosses below EMA50 ({ema_f[-1]:.0f})",
        "stop_level": price * (1 - 0.15),
        "context": f"EMA50={ema_f[-1]:.0f} EMA200={ema_s[-1]:.0f} ADX={adx[-1]:.0f} Donchian20={donch[-1]:.0f}",
    }


def signal_dip_buyer(df: pd.DataFrame) -> dict:
    c = df["close"].values
    rsi = talib.RSI(c, 14)
    tp = (df["high"] + df["low"] + df["close"]) / 3
    mid = tp.rolling(20).mean()
    std = tp.rolling(20).std()
    lower = (mid - 2 * std).values
    price = c[-1]
    entry = (rsi[-1] < 28 and price < lower[-1])
    exit_ = (rsi[-1] > 55 and rsi[-2] <= 55)  # RSI recovers through 55 (bounce done)
    return {
        "timeframe": "4h", "price": price,
        "entry_signal": bool(entry),
        "exit_signal": bool(exit_),
        "entry_trigger": f"RSI<28 (now {rsi[-1]:.0f}) and close < lower BB ({lower[-1]:.0f})",
        "exit_trigger": "RSI>55 or +6% ROI",
        "stop_level": price * (1 - 0.08),
        "context": f"RSI={rsi[-1]:.0f} BB_lower={lower[-1]:.0f}",
    }


def signal_mtf_supertrend(df4: pd.DataFrame, df1d: pd.DataFrame) -> dict:
    st, direction = _supertrend(df4, 10, 3.0)
    ema50 = taa.EMA(df1d, timeperiod=50)
    ema200 = taa.EMA(df1d, timeperiod=200)
    price = df4["close"].iloc[-1]
    daily_up = bool(df1d["close"].iloc[-1] > ema200.iloc[-1] and ema50.iloc[-1] > ema200.iloc[-1])
    bull = int(direction.iloc[-1]) == 1
    flipped = bull and int(direction.iloc[-2]) == -1
    flipped_bear = (int(direction.iloc[-1]) == -1 and int(direction.iloc[-2]) == 1)
    st_line = float(st.iloc[-1])
    return {
        "timeframe": "4h", "price": price,
        "entry_signal": bool(flipped and daily_up),
        "exit_signal": bool(flipped_bear),
        "entry_trigger": "4h Supertrend flips bullish while daily uptrend (close>EMA200_1d, EMA50_1d>EMA200_1d)",
        "exit_trigger": f"4h Supertrend flips bearish (line ${st_line:,.0f})",
        "stop_level": price * (1 - 0.12),
        "context": f"4h trend={'BULL' if bull else 'BEAR'}, daily_uptrend={daily_up}, ST=${st_line:,.0f}",
    }


def signal_squeeze(df: pd.DataFrame) -> dict:
    bb = qtpylib.bollinger_bands(qtpylib.typical_price(df), window=20, stds=2)
    kc = qtpylib.keltner_channel(df, window=20, atrs=1.5)
    squeeze_on = (bb["upper"] < kc["upper"]) & (bb["lower"] > kc["lower"])
    roc = taa.ROC(df, timeperiod=12)
    ema200 = taa.EMA(df, timeperiod=200)
    price = df["close"].iloc[-1]
    released = bool(squeeze_on.iloc[-2] and not squeeze_on.iloc[-1])
    entry = bool(
        released and price > bb["mid"].iloc[-1] and roc.iloc[-1] > 0 and price > ema200.iloc[-1]
    )
    mid = bb["mid"]
    cl = df["close"]
    exit_ = bool(
        (cl.iloc[-1] < mid.iloc[-1] and cl.iloc[-2] >= mid.iloc[-2])
        or (roc.iloc[-1] < 0 and roc.iloc[-2] >= 0)
    )
    state = "ON (compressed)" if squeeze_on.iloc[-1] else "OFF (released)"
    return {
        "timeframe": "1h", "price": price,
        "entry_signal": entry,
        "exit_signal": exit_,
        "entry_trigger": "squeeze releases + close>BB mid + ROC>0 + price>EMA200",
        "exit_trigger": f"close < BB mid (${bb['mid'].iloc[-1]:,.0f}) or ROC<0",
        "stop_level": price * (1 - 0.06),
        "context": f"squeeze={state}, ROC={roc.iloc[-1]:.2f}, >EMA200={price > ema200.iloc[-1]}",
    }


def signal_dca(df: pd.DataFrame) -> dict:
    rsi = talib.RSI(df["close"].values, 14)
    bb = qtpylib.bollinger_bands(qtpylib.typical_price(df), window=20, stds=2)
    ema200 = taa.EMA(df, timeperiod=200)
    price = df["close"].iloc[-1]
    entry = bool(rsi[-1] < 30 and price < bb["lower"].iloc[-1] and price > ema200.iloc[-1])
    mid = bb["mid"]
    cl = df["close"]
    exit_ = bool(
        (rsi[-1] > 58 and rsi[-2] <= 58)
        or (cl.iloc[-1] >= mid.iloc[-1] and cl.iloc[-2] < mid.iloc[-2])
    )
    return {
        "timeframe": "4h", "price": price,
        "entry_signal": entry,
        "exit_signal": exit_,
        "entry_trigger": f"RSI<30 (now {rsi[-1]:.0f}) + close<BB lower (${bb['lower'].iloc[-1]:,.0f}) + price>EMA200 (uptrend-only DCA)",
        "exit_trigger": f"RSI>58 or close>=BB mid (${bb['mid'].iloc[-1]:,.0f}); scales in on further dips",
        "stop_level": price * (1 - 0.18),
        "context": f"RSI={rsi[-1]:.0f}, >EMA200={price > ema200.iloc[-1]} (no DCA in downtrends)",
    }


# --- Ranking ---------------------------------------------------------------
def pick_best(results: dict) -> str:
    def key(name):
        r = results[name]
        return (r["profit_pct"] if r["profit_pct"] is not None else -999,
                r["profit_factor"] if r["profit_factor"] is not None else -999)
    return max(results, key=key)


# --- Report ----------------------------------------------------------------
def build_report(results: dict, best: dict, signals: dict, timerange: str) -> str:
    """Plain-text/markdown fallback (HTML is the primary email). Multi-coin."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# Daily Crypto Strategy Report — {today}",
        f"\nBacktest window: rolling {ROLLING_DAYS} days.",
    ]
    for p in PAIRS:
        pair = p["pair"]
        pres = results.get(pair, {})
        bname = best[pair]
        mc = next((r["market_change_pct"] for r in pres.values()
                   if r["market_change_pct"] is not None), None)
        lines += [
            f"\n## {p['name']} ({pair}) — buy & hold: "
            f"{_f(mc, '%', '+') if mc is not None else 'n/a'}",
            "\n| Strategy | Return | Max DD | PF | Sharpe | Trades | Win% |",
            "|---|---|---|---|---|---|---|",
        ]
        for name in STRATEGIES:
            r = pres.get(name, {})
            tag = "**" + name + "** 🏆" if name == bname else name
            lines.append(
                f"| {tag} | {_f(r.get('profit_pct'),'%','+')} | {_f(r.get('max_dd_pct'),'%')} "
                f"| {_f(r.get('profit_factor'))} | {_f(r.get('sharpe'))} "
                f"| {r.get('trades','-')} | {_f(r.get('winrate'),'%')} |"
            )
        s = signals[pair][bname]
        state = "ENTRY SIGNAL ACTIVE" if s["entry_signal"] else "No entry (hold/wait)"
        lines += [
            f"\nBest today: **{bname}** — {state}",
            f"- Price: ${s['price']:,.0f} ({s['timeframe']})",
            f"- Entry: {s['entry_trigger']}",
            f"- Exit: {s['exit_trigger']}",
            f"- Stop-loss: ${s['stop_level']:,.0f}",
            f"- Expected return*: {_f(pres[bname].get('profit_pct'),'%','+')} | "
            f"Max DD*: {_f(pres[bname].get('max_dd_pct'),'%')}",
        ]
    lines += [
        "\n---",
        "_*Historical backtest figures, not a forecast. Educational only — not financial advice._",
    ]
    return "\n".join(L for L in lines if L != "")


def _f(v, suffix="", sign=""):
    if v is None:
        return "n/a"
    if sign == "+":
        return f"{v:+.2f}{suffix}"
    return f"{v:.2f}{suffix}"


def _color(v):
    """Green for non-negative, red for negative, grey for missing."""
    if v is None:
        return "#64748b"
    return "#16a34a" if v >= 0 else "#dc2626"


def _coin_section(p: dict, pres: dict, bname: str, psignals: dict) -> str:
    """HTML block for one coin: section header + best-strategy card + ranked table."""
    s = psignals[bname]
    br = pres[bname]
    mc = next((r["market_change_pct"] for r in pres.values()
               if r["market_change_pct"] is not None), None)
    cur = f"${s['price']:,.0f}"

    if s["entry_signal"]:
        badge = ('<span style="background:#16a34a;color:#ffffff;padding:5px 14px;'
                 'border-radius:999px;font-size:13px;font-weight:600;">● ENTRY SIGNAL ACTIVE</span>')
    else:
        badge = ('<span style="background:#e2e8f0;color:#475569;padding:5px 14px;'
                 'border-radius:999px;font-size:13px;font-weight:600;">● No entry — hold / wait</span>')

    def fact(label, value):
        return ('<tr>'
                f'<td style="padding:7px 0;color:#64748b;font-size:13px;width:150px;vertical-align:top;">{label}</td>'
                f'<td style="padding:7px 0;color:#0f172a;font-size:13px;font-weight:500;">{value}</td></tr>')

    facts = (
        fact("Price", f"{cur} &nbsp;<span style='color:#94a3b8;'>({s['timeframe']} timeframe)</span>")
        + fact("Entry point", s["entry_trigger"])
        + fact("Exit point", s["exit_trigger"])
        + fact("Stop-loss", f"${s['stop_level']:,.0f}")
        + fact("Expected return*", f'<span style="color:{_color(br.get("profit_pct"))};font-weight:600;">{_f(br.get("profit_pct"), "%", "+")}</span>')
        + fact("Max drawdown*", _f(br.get("max_dd_pct"), "%"))
        + fact("Indicators", f"<span style='color:#475569;'>{s['context']}</span>")
    )

    rows = ""
    for i, name in enumerate(STRATEGIES):
        r = pres.get(name, {})
        is_best = name == bname
        ret = r.get("profit_pct")
        bg = "#ecfdf5" if is_best else ("#ffffff" if i % 2 == 0 else "#f8fafc")
        label = ("🏆 " if is_best else "") + name.replace("Strategy", "")
        weight = "700" if is_best else "400"
        cell = "padding:10px 12px;border-bottom:1px solid #e2e8f0;"
        rows += (
            f'<tr style="background:{bg};">'
            f'<td style="{cell}font-weight:{weight};color:#0f172a;">{label}</td>'
            f'<td style="{cell}text-align:right;font-weight:600;color:{_color(ret)};">{_f(ret, "%", "+")}</td>'
            f'<td style="{cell}text-align:right;color:#334155;">{_f(r.get("max_dd_pct"), "%")}</td>'
            f'<td style="{cell}text-align:right;color:#334155;">{_f(r.get("profit_factor"))}</td>'
            f'<td style="{cell}text-align:right;color:#334155;">{r.get("trades", "–")}</td>'
            f'<td style="{cell}text-align:right;color:#334155;">{_f(r.get("winrate"), "%")}</td></tr>'
        )

    mc_html = (f'<strong style="color:{_color(mc)};">{_f(mc, "%", "+")}</strong>'
               if mc is not None else "n/a")

    return f"""
  <tr><td style="padding:20px 28px 0;">
    <div style="background:#0f172a;border-radius:8px;padding:12px 16px;">
      <span style="color:#ffffff;font-size:17px;font-weight:700;">{p['emoji']} {p['name']}</span>
      <span style="color:#94a3b8;font-size:12px;"> &nbsp;{p['pair']} · buy &amp; hold {mc_html}</span>
    </div>
  </td></tr>
  <tr><td style="padding:16px 28px 4px;">
    <div style="font-size:11px;letter-spacing:1px;color:#64748b;text-transform:uppercase;">Top strategy</div>
    <div style="font-size:20px;font-weight:700;color:#0f172a;margin:4px 0 10px;">{bname.replace('Strategy', '')}</div>
    <div style="margin-bottom:16px;">{badge}</div>
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0">{facts}</table>
  </td></tr>
  <tr><td style="padding:8px 28px 20px;">
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;font-size:13px;border:1px solid #e2e8f0;">
      <tr style="background:#1e293b;">
        <th style="padding:10px 12px;text-align:left;color:#cbd5e1;font-weight:600;">Strategy</th>
        <th style="padding:10px 12px;text-align:right;color:#cbd5e1;font-weight:600;">Return</th>
        <th style="padding:10px 12px;text-align:right;color:#cbd5e1;font-weight:600;">Max&nbsp;DD</th>
        <th style="padding:10px 12px;text-align:right;color:#cbd5e1;font-weight:600;">PF</th>
        <th style="padding:10px 12px;text-align:right;color:#cbd5e1;font-weight:600;">Trades</th>
        <th style="padding:10px 12px;text-align:right;color:#cbd5e1;font-weight:600;">Win%</th>
      </tr>
      {rows}
    </table>
  </td></tr>"""


def build_html(results: dict, best: dict, signals: dict, timerange: str) -> str:
    """A clean, email-client-safe HTML report (inline styles, table layout). Multi-coin."""
    today = datetime.now(timezone.utc).strftime("%b %d, %Y · %H:%M UTC")
    coins = "".join(_coin_section(p, results[p["pair"]], best[p["pair"]], signals[p["pair"]])
                    for p in PAIRS)
    return f"""<!DOCTYPE html>
<html><body style="margin:0;padding:0;background:#eef2f6;">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#eef2f6;padding:24px 0;">
<tr><td align="center">
<table role="presentation" width="640" cellpadding="0" cellspacing="0" style="max-width:640px;width:100%;background:#ffffff;border-radius:12px;overflow:hidden;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;box-shadow:0 1px 3px rgba(0,0,0,0.08);">
  <tr><td style="background:#0f172a;padding:24px 28px;">
    <div style="color:#ffffff;font-size:20px;font-weight:700;">📈 Daily Crypto Strategy Report</div>
    <div style="color:#94a3b8;font-size:13px;margin-top:4px;">{today} &nbsp;·&nbsp; rolling {ROLLING_DAYS}-day backtest</div>
  </td></tr>
  {coins}
  {build_strategy_guide()}
  <tr><td style="padding:18px 28px 26px;background:#f8fafc;border-top:1px solid #e2e8f0;">
    <div style="font-size:11px;color:#94a3b8;line-height:1.6;">
      * Expected return &amp; max drawdown are the strategy's historical backtest figures over the window above — not a forecast.
      Trend-followers profit in trends and chop in ranges; mean-reversion is the opposite.
      Past performance does not guarantee future results. <strong>Educational only — not financial advice.</strong>
    </div>
  </td></tr>
</table>
</td></tr>
</table>
</body></html>"""


# Static educational guide appended to every report. Each strategy: how it works,
# entry/exit rules, and the math behind its indicators (monospace = email-safe).
_GUIDE = [
    {
        "title": "1 · BtcEmaRsi — EMA Crossover + RSI Momentum",
        "tf": "1h",
        "concept": "Classic momentum. A fast EMA crossing above a slow EMA marks a shift to "
                   "upward momentum. RSI confirms momentum is present but not exhausted; ADX "
                   "confirms a real trend (filters out chop); the 200-EMA keeps trades aligned "
                   "with the larger trend.",
        "entry": "EMA(12) crosses above EMA(26) · price &gt; EMA(200) · 50 &lt; RSI &lt; 70 · ADX &gt; 20",
        "exit": "EMA(12) crosses below EMA(26), or RSI &gt; 78 (overbought)",
        "formulas": [
            "EMAₜ = α·Priceₜ + (1−α)·EMAₜ₋₁ ,   α = 2 / (N + 1)",
            "RSI = 100 − 100 / (1 + RS) ,   RS = (avg gain) / (avg loss) over 14",
        ],
        "diagram": "",
    },
    {
        "title": "2 · BtcTrendRider — Donchian Breakout (trend-rider)",
        "tf": "4h",
        "concept": "Pure trend-following. Buys a fresh N-bar high (a breakout) only inside a "
                   "confirmed uptrend, then lets the winner run with a trailing stop and exits "
                   "when the trend weakens. The point is to capture large moves, not scalp.",
        "entry": "close &gt; Donchian-20 high · EMA(50) &gt; EMA(200) · ADX &gt; 20",
        "exit": "close crosses below EMA(50); trailing stop locks in gains",
        "formulas": [
            "Donchian upper(N) = max(High over last N bars)",
            "Breakout: Closeₜ &gt; max(Highₜ₋₁ … Highₜ₋ₙ)",
        ],
        "diagram": "",
    },
    {
        "title": "3 · BtcDipBuyer — RSI + Bollinger Mean-Reversion",
        "tf": "4h",
        "concept": "Mean reversion. When price gets stretched far below its average (a panic "
                   "dip under the lower Bollinger Band while RSI is oversold), it tends to snap "
                   "back. Buys the dip, takes a quick profit on the bounce.",
        "entry": "RSI &lt; 28 · close &lt; lower Bollinger Band",
        "exit": "RSI &gt; 55, or price reverts to the band middle / ROI",
        "formulas": [
            "Mid = SMA(typical price, 20) ,   typical price = (H+L+C)/3",
            "Upper / Lower = Mid ± 2·σ   (σ = std-dev over 20 bars)",
        ],
        "diagram": "",
    },
    {
        "title": "4 · MtfSupertrend — Multi-Timeframe Supertrend",
        "tf": "4h + daily",
        "concept": "An ATR-based trend line (Supertrend) that sits below price in uptrends and "
                   "above it in downtrends; price closing across it flips the trend. The 4h "
                   "Supertrend gives entries/exits, but they're only taken when the DAILY trend "
                   "agrees — a higher-timeframe filter that cuts whipsaws.",
        "entry": "4h Supertrend flips bullish  AND  daily close &gt; daily EMA(200) &amp; daily EMA(50) &gt; EMA(200)",
        "exit": "4h Supertrend flips bearish",
        "formulas": [
            "TR = max(Hₜ−Lₜ , |Hₜ−Cₜ₋₁| , |Lₜ−Cₜ₋₁|) ;   ATR = average(TR, N)",
            "Basic bands = (H+L)/2 ± multiplier·ATR  →  trend flips when close crosses the active band",
        ],
        "diagram": "",
    },
    {
        "title": "5 · SqueezeBreakout — Bollinger/Keltner Volatility Squeeze",
        "tf": "1h",
        "concept": "A volatility play (TTM-squeeze). When the Bollinger Bands contract INSIDE "
                   "the Keltner Channel, volatility is compressed — a big move usually follows. "
                   "We wait for the squeeze to release and enter in the breakout direction.",
        "entry": "squeeze releases (BB exits KC) · close &gt; BB mid · ROC &gt; 0 · price &gt; EMA(200)",
        "exit": "close &lt; BB mid, or ROC &lt; 0 (momentum fades)",
        "formulas": [
            "Bollinger = SMA ± 2·σ      Keltner = EMA ± 1.5·ATR",
            "SQUEEZE ON  ⇔  BB_upper &lt; KC_upper  AND  BB_lower &gt; KC_lower",
        ],
        "diagram": ("Keltner   |———————————————————|\n"
                    "Bollinger      |—————————|          ← BB inside KC = SQUEEZE (coiled)\n"
                    "release        |—————————————————|  ← BB expands out  →  trade the breakout"),
    },
    {
        "title": "6 · DcaMeanReversion — Dollar-Cost-Averaging Dip Buyer",
        "tf": "4h",
        "concept": "Mean reversion with position scaling. Buys an oversold dip, and if price "
                   "falls further, adds tranches to improve the average entry (DCA), then exits "
                   "on reversion. A strict rule — only DCA when price is ABOVE the 200-EMA — "
                   "stops it from averaging down into a falling-knife bear market.",
        "entry": "RSI &lt; 30 · close &lt; lower BB · price &gt; EMA(200)  (uptrend-only)",
        "exit": "RSI &gt; 58 or price reverts to BB mid; scales in (max 3 safety orders) on deeper dips",
        "formulas": [
            "Average entry = Σ(qtyᵢ × priceᵢ) / Σ qtyᵢ   (improves as you add lower)",
            "Safety order i fires when profit &lt; −3.5% × (entries so far)",
        ],
        "diagram": "",
    },
]


def build_strategy_guide() -> str:
    cards = ""
    fbox = ("margin:6px 0 2px;padding:8px 10px;background:#f1f5f9;border-radius:6px;"
            "font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;"
            "font-size:12px;color:#0f172a;white-space:pre-wrap;line-height:1.7;")
    for g in _GUIDE:
        formulas = "".join(f'<div style="{fbox}">{f}</div>' for f in g["formulas"])
        diagram = (f'<div style="{fbox}background:#0f172a;color:#e2e8f0;">{g["diagram"]}</div>'
                   if g["diagram"] else "")
        cards += f"""
      <div style="border:1px solid #e2e8f0;border-radius:8px;padding:14px 16px;margin-bottom:12px;">
        <div style="font-size:14px;font-weight:700;color:#0f172a;">{g['title']}
          <span style="font-size:11px;font-weight:500;color:#64748b;">· {g['tf']}</span></div>
        <div style="font-size:13px;color:#475569;line-height:1.6;margin:8px 0;">{g['concept']}</div>
        <div style="font-size:12px;color:#0f172a;margin:4px 0;"><strong style="color:#16a34a;">Entry:</strong> {g['entry']}</div>
        <div style="font-size:12px;color:#0f172a;margin:4px 0;"><strong style="color:#dc2626;">Exit:</strong> {g['exit']}</div>
        {formulas}{diagram}
      </div>"""
    return f"""<tr><td style="padding:8px 28px 16px;">
    <div style="font-size:11px;letter-spacing:1px;color:#64748b;text-transform:uppercase;margin:14px 0 10px;">📚 Strategy reference guide — how each one works</div>
    {cards}
  </td></tr>"""


def maybe_email(subject: str, html_body: str, text_body: str) -> str:
    to = os.environ.get("REPORT_EMAIL_TO")
    if not to:
        return "email skipped (REPORT_EMAIL_TO not set)"
    recipients = [a.strip() for a in to.split(",") if a.strip()]

    # Preferred: Resend HTTPS API (port 443). Use this on hosts that block
    # outbound SMTP ports (e.g. DigitalOcean droplets).
    resend_key = os.environ.get("RESEND_API_KEY")
    if resend_key:
        import json
        import urllib.error
        import urllib.request
        sender = os.environ.get("REPORT_EMAIL_FROM", "onboarding@resend.dev")
        payload = json.dumps(
            {"from": sender, "to": recipients, "subject": subject,
             "html": html_body, "text": text_body}
        ).encode()
        req = urllib.request.Request(
            "https://api.resend.com/emails",
            data=payload,
            headers={
                "Authorization": f"Bearer {resend_key}",
                "Content-Type": "application/json",
                # A real User-Agent avoids Cloudflare bot-blocking (error 1010)
                # that rejects the default urllib User-Agent in front of the API.
                "User-Agent": "Mozilla/5.0 (compatible; freqtrade-report/1.0)",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return f"email sent via Resend to {to} (HTTP {r.status})"
        except urllib.error.HTTPError as e:
            return f"Resend API error {e.code}: {e.read().decode()[:300]}"

    # Fallback: SMTP (works on machines that allow outbound SMTP, e.g. a Mac).
    host = os.environ.get("SMTP_HOST")
    if not host:
        return "email skipped (set RESEND_API_KEY, or SMTP_HOST for SMTP delivery)"
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = os.environ.get("SMTP_USER", "freqtrade-bot")
    msg["To"] = to
    msg.attach(MIMEText(text_body, "plain", "utf-8"))
    msg.attach(MIMEText(html_body, "html", "utf-8"))
    with smtplib.SMTP(host, int(os.environ.get("SMTP_PORT", "587"))) as srv:
        srv.starttls()
        if os.environ.get("SMTP_USER"):
            srv.login(os.environ["SMTP_USER"], os.environ["SMTP_PASS"])
        srv.send_message(msg)
    return f"email sent via SMTP to {to}"


def main() -> int:
    end = datetime.now(timezone.utc).date()
    print("[1/4] Downloading latest data ...")
    timerange = download_data()
    print(f"[2/4] Backtesting strategies for {len(ALL_PAIRS)} pairs ...")
    results = run_backtests(timerange)
    print("[3/4] Computing current signals ...")
    signals = {pair: compute_signals(pair) for pair in ALL_PAIRS}
    best = {pair: pick_best(results[pair]) for pair in ALL_PAIRS}
    report_md = build_report(results, best, signals, timerange)
    report_html = build_html(results, best, signals, timerange)

    out_file = REPORTS / f"{end.strftime('%Y-%m-%d')}.md"
    out_file.write_text(report_md, encoding="utf-8")
    html_file = REPORTS / f"{end.strftime('%Y-%m-%d')}.html"
    html_file.write_text(report_html, encoding="utf-8")
    print(f"[4/4] Report written: {out_file}")
    tops = " · ".join(f"{p['emoji']} {best[p['pair']].replace('Strategy', '')}" for p in PAIRS)
    subject = f"📈 Crypto Strategy Report {end} — {tops}"
    print(maybe_email(subject, report_html, report_md))
    print("\n" + report_md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
