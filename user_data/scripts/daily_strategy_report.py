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
from email.mime.text import MIMEText
from pathlib import Path

import numpy as np
import pandas as pd
import talib

# --- Paths -----------------------------------------------------------------
REPO = Path(__file__).resolve().parents[2]          # .../freqtrade
CONFIG = REPO / "user_data" / "config.json"
DATA_DIR = REPO / "user_data" / "data" / "binanceus"
REPORTS = REPO / "user_data" / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

STRATEGIES = ["BtcEmaRsiStrategy", "BtcTrendRiderStrategy", "BtcDipBuyerStrategy"]
PAIR = "BTC/USDT"
ROLLING_DAYS = 365


def sh(cmd: list[str]) -> subprocess.CompletedProcess:
    """Run a command from the repo root, raising on failure."""
    return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, check=True)


# --- 1. Download latest data ----------------------------------------------
def download_data(timerange: str) -> None:
    for tf in ("1h", "4h"):
        sh([
            "freqtrade", "download-data", "--config", str(CONFIG),
            "--pairs", PAIR, "--timeframe", tf, "--timerange", timerange,
        ])


# --- 2. Backtest all strategies -------------------------------------------
def run_backtests(timerange: str) -> dict:
    """Backtest all strategies in one call and load the stats via freqtrade's API."""
    sh([
        "freqtrade", "backtesting", "--config", str(CONFIG),
        "--strategy-list", *STRATEGIES,
        "--pairs", PAIR, "--timerange", timerange,
        "--export", "trades", "--cache", "none",
    ])
    from freqtrade.data.btanalysis import load_backtest_stats
    stats = load_backtest_stats(str(REPO / "user_data" / "backtest_results"))
    out = {}
    for name, s in stats.get("strategy", {}).items():
        wins = s.get("wins")
        total = s.get("total_trades") or 0
        out[name] = {
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
def load_ohlcv(tf: str) -> pd.DataFrame:
    df = pd.read_feather(DATA_DIR / f"BTC_USDT-{tf}.feather")
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.set_index("date").sort_index()


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
    return {
        "timeframe": "1h", "price": price,
        "entry_signal": bool(entry),
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
    return {
        "timeframe": "4h", "price": price,
        "entry_signal": bool(entry),
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
    return {
        "timeframe": "4h", "price": price,
        "entry_signal": bool(entry),
        "entry_trigger": f"RSI<28 (now {rsi[-1]:.0f}) and close < lower BB ({lower[-1]:.0f})",
        "exit_trigger": "RSI>55 or +6% ROI",
        "stop_level": price * (1 - 0.08),
        "context": f"RSI={rsi[-1]:.0f} BB_lower={lower[-1]:.0f}",
    }


SIGNAL_FN = {
    "BtcEmaRsiStrategy": signal_ema_rsi,
    "BtcTrendRiderStrategy": signal_trend_rider,
    "BtcDipBuyerStrategy": signal_dip_buyer,
}


# --- Ranking ---------------------------------------------------------------
def pick_best(results: dict) -> str:
    def key(name):
        r = results[name]
        return (r["profit_pct"] if r["profit_pct"] is not None else -999,
                r["profit_factor"] if r["profit_factor"] is not None else -999)
    return max(results, key=key)


# --- Report ----------------------------------------------------------------
def build_report(results: dict, best: str, signals: dict, timerange: str) -> str:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    mc = next((r["market_change_pct"] for r in results.values()
               if r["market_change_pct"] is not None), None)
    lines = [
        f"# Daily BTC Strategy Report — {today}",
        f"\nBacktest window: **{timerange}** (rolling {ROLLING_DAYS} days) | "
        f"Buy & Hold over window: **{mc:+.2f}%**" if mc is not None else "",
        "\n## Strategy comparison\n",
        "| Strategy | Return | Max DD | Profit Factor | Sharpe | Trades | Win% |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in STRATEGIES:
        r = results.get(name, {})
        lines.append(
            f"| {'**'+name+'** 🏆' if name == best else name} "
            f"| {_f(r.get('profit_pct'),'%','+')} | {_f(r.get('max_dd_pct'),'%')} "
            f"| {_f(r.get('profit_factor'))} | {_f(r.get('sharpe'))} "
            f"| {r.get('trades','-')} | {_f(r.get('winrate'),'%')} |"
        )
    s = signals[best]
    state = "🟢 ENTRY SIGNAL ACTIVE" if s["entry_signal"] else "⚪ No entry signal (wait / hold)"
    lines += [
        f"\n## Best strategy today: **{best}**\n",
        f"- **Current signal:** {state}",
        f"- **Timeframe:** {s['timeframe']}  |  **BTC price:** ${s['price']:,.0f}",
        f"- **Entry point:** {s['entry_trigger']}",
        f"- **Exit point:** {s['exit_trigger']}",
        f"- **Stop-loss level:** ${s['stop_level']:,.0f}",
        f"- **Expected profit (backtest return):** {_f(results[best].get('profit_pct'),'%','+')}",
        f"- **Potential max drawdown:** {_f(results[best].get('max_dd_pct'),'%')}",
        f"- **Indicator context:** {s['context']}",
        "\n---",
        "_Backtest metrics are historical and not a forecast. Trend-followers profit in "
        "trends and chop in ranges; mean-reversion is the opposite. Past performance does "
        "not guarantee future results. Educational, not financial advice._",
    ]
    return "\n".join(L for L in lines if L != "")


def _f(v, suffix="", sign=""):
    if v is None:
        return "n/a"
    if sign == "+":
        return f"{v:+.2f}{suffix}"
    return f"{v:.2f}{suffix}"


def maybe_email(subject: str, body_md: str) -> str:
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
            {"from": sender, "to": recipients, "subject": subject, "text": body_md}
        ).encode()
        req = urllib.request.Request(
            "https://api.resend.com/emails",
            data=payload,
            headers={
                "Authorization": f"Bearer {resend_key}",
                "Content-Type": "application/json",
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
    msg = MIMEText(body_md, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = os.environ.get("SMTP_USER", "freqtrade-bot")
    msg["To"] = to
    with smtplib.SMTP(host, int(os.environ.get("SMTP_PORT", "587"))) as srv:
        srv.starttls()
        if os.environ.get("SMTP_USER"):
            srv.login(os.environ["SMTP_USER"], os.environ["SMTP_PASS"])
        srv.send_message(msg)
    return f"email sent via SMTP to {to}"


def main() -> int:
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=ROLLING_DAYS + 30)  # +30 for indicator warmup
    timerange = f"{start.strftime('%Y%m%d')}-{end.strftime('%Y%m%d')}"

    print(f"[1/4] Downloading latest data ({timerange}) ...")
    download_data(timerange)
    print("[2/4] Backtesting strategies ...")
    results = run_backtests(timerange)
    print("[3/4] Computing current signals ...")
    df1h, df4h = load_ohlcv("1h"), load_ohlcv("4h")
    signals = {
        "BtcEmaRsiStrategy": signal_ema_rsi(df1h),
        "BtcTrendRiderStrategy": signal_trend_rider(df4h),
        "BtcDipBuyerStrategy": signal_dip_buyer(df4h),
    }
    best = pick_best(results)
    report = build_report(results, best, signals, timerange)

    out_file = REPORTS / f"{end.strftime('%Y-%m-%d')}.md"
    out_file.write_text(report, encoding="utf-8")
    print(f"[4/4] Report written: {out_file}")
    print(maybe_email(f"BTC Strategy Report {end} — best: {best}", report))
    print("\n" + report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
