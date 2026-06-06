#!/usr/bin/env python
"""
signal_alerts.py
================
Event-driven companion to daily_strategy_report.py. Meant to run HOURLY.

Each run it computes the current live signal for all strategies and compares to the
previous run's state (persisted in a small JSON file). It emails an alert ONLY when
a strategy newly flips to an ENTRY signal — so you get timely "buy signal fired"
notifications without a flood of identical digests. Silent when nothing changes.

Why hourly: the fastest strategies use 1h candles, so signals can't change more often
than once per hour. Running more frequently would add no information.

Run (freqtrade conda env / venv):
    python user_data/scripts/signal_alerts.py
Email is sent via the same REPORT_EMAIL_TO / RESEND_API_KEY (or SMTP_*) env vars.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Reuse everything from the daily report module (signal fns, data IO, emailer).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import daily_strategy_report as dr  # noqa: E402

STATE_FILE = dr.REPO / "user_data" / "reports" / ".signal_state.json"


def current_signals() -> dict:
    """Refresh data (append latest candles) and compute each strategy's live signal."""
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=300)  # covers the daily EMA200; appends to existing data
    dr.download_data(f"{start.strftime('%Y%m%d')}-{end.strftime('%Y%m%d')}")
    df1h, df4h, df1d = dr.load_ohlcv("1h"), dr.load_ohlcv("4h"), dr.load_ohlcv("1d")
    return {
        "BtcEmaRsiStrategy": dr.signal_ema_rsi(df1h),
        "BtcTrendRiderStrategy": dr.signal_trend_rider(df4h),
        "BtcDipBuyerStrategy": dr.signal_dip_buyer(df4h),
        "MtfSupertrendStrategy": dr.signal_mtf_supertrend(df4h, df1d),
        "SqueezeBreakoutStrategy": dr.signal_squeeze(df1h),
        "DcaMeanReversionStrategy": dr.signal_dca(df4h),
    }


def build_alert(entries: list) -> tuple[str, str]:
    now = datetime.now(timezone.utc).strftime("%b %d, %Y · %H:%M UTC")
    cards = ""
    text = [f"NEW BTC ENTRY SIGNAL(S) — {now}", ""]
    fact_css = "padding:5px 0;font-size:13px;"
    for name, s in entries:
        rows = "".join(
            f'<tr><td style="{fact_css}color:#64748b;width:120px;">{k}</td>'
            f'<td style="{fact_css}color:#0f172a;font-weight:500;">{v}</td></tr>'
            for k, v in [
                ("BTC price", f"${s['price']:,.0f}  ({s['timeframe']})"),
                ("Entry", s["entry_trigger"]),
                ("Exit plan", s["exit_trigger"]),
                ("Stop-loss", f"${s['stop_level']:,.0f}"),
                ("Indicators", s["context"]),
            ]
        )
        cards += (
            '<div style="border:1px solid #16a34a;border-left:4px solid #16a34a;'
            'border-radius:8px;padding:14px 16px;margin-bottom:12px;">'
            f'<div style="font-size:16px;font-weight:700;color:#0f172a;">🟢 {name.replace("Strategy", "")}'
            f' <span style="font-size:11px;font-weight:500;color:#64748b;">· {s["timeframe"]}</span></div>'
            f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="margin-top:8px;">{rows}</table>'
            "</div>"
        )
        text += [f"{name}: ENTRY", f"  price ${s['price']:,.0f}",
                 f"  entry: {s['entry_trigger']}", f"  exit: {s['exit_trigger']}",
                 f"  stop ${s['stop_level']:,.0f}", ""]

    html = f"""<!DOCTYPE html>
<html><body style="margin:0;padding:0;background:#eef2f6;">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#eef2f6;padding:24px 0;"><tr><td align="center">
<table role="presentation" width="600" cellpadding="0" cellspacing="0" style="max-width:600px;width:100%;background:#ffffff;border-radius:12px;overflow:hidden;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;box-shadow:0 1px 3px rgba(0,0,0,0.08);">
  <tr><td style="background:#16a34a;padding:20px 24px;">
    <div style="color:#ffffff;font-size:19px;font-weight:700;">🟢 BTC Entry Signal Alert</div>
    <div style="color:#dcfce7;font-size:13px;margin-top:4px;">{now}</div>
  </td></tr>
  <tr><td style="padding:20px 24px;">
    <div style="font-size:13px;color:#475569;margin-bottom:14px;">
      {len(entries)} strategy(ies) just flipped to an <strong style="color:#16a34a;">ENTRY</strong> signal:
    </div>
    {cards}
  </td></tr>
  <tr><td style="padding:16px 24px;background:#f8fafc;border-top:1px solid #e2e8f0;">
    <div style="font-size:11px;color:#94a3b8;line-height:1.6;">
      Event-driven alert (checked hourly; sent only on a new signal). The full daily digest still
      arrives once a day. <strong>Educational only — not financial advice.</strong>
    </div>
  </td></tr>
</table></td></tr></table></body></html>"""
    return html, "\n".join(text)


def main() -> int:
    print("[alerts] computing current signals ...")
    sigs = current_signals()
    prev = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}

    new_entries = []
    state = {}
    for name, s in sigs.items():
        st = "ENTRY" if s["entry_signal"] else "NEUTRAL"
        state[name] = st
        if st == "ENTRY" and prev.get(name) != "ENTRY":
            new_entries.append((name, s))

    STATE_FILE.write_text(json.dumps(state))

    if not new_entries:
        active = [n for n, v in state.items() if v == "ENTRY"]
        print(f"[alerts] no NEW entry signals (currently active: {active or 'none'}); no email sent")
        return 0

    html, text = build_alert(new_entries)
    subject = f"🟢 BTC Entry Alert — {len(new_entries)} new signal(s): " + ", ".join(
        n.replace("Strategy", "") for n, _ in new_entries
    )
    print(dr.maybe_email(subject, html, text))
    return 0


if __name__ == "__main__":
    sys.exit(main())
