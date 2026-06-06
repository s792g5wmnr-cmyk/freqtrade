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
    """Refresh data (append latest candles) and compute live signals for every
    (pair, strategy). Returns {pair: {strategy: signal}}."""
    dr.download_data()  # appends latest candles; ensures daily EMA200 lead-in
    return {pair: dr.compute_signals(pair) for pair in dr.ALL_PAIRS}


_PAIR_META = {p["pair"]: p for p in dr.PAIRS}


def build_alert(entries: list) -> tuple[str, str]:
    now = datetime.now(timezone.utc).strftime("%b %d, %Y · %H:%M UTC")
    cards = ""
    text = [f"NEW CRYPTO SIGNAL(S) — {now}", ""]
    fact_css = "padding:5px 0;font-size:13px;"
    n_buy = sum(1 for e in entries if e[2] == "BUY")
    n_sell = len(entries) - n_buy
    for pair, name, side, s in entries:
        meta = _PAIR_META.get(pair, {"name": pair, "emoji": ""})
        coin = f"{meta['emoji']} {meta['name']}"
        is_buy = side == "BUY"
        accent = "#16a34a" if is_buy else "#dc2626"
        icon = "🟢 BUY" if is_buy else "🔴 SELL"
        trigger_label, trigger_val = (
            ("Entry", s["entry_trigger"]) if is_buy else ("Exit reason", s["exit_trigger"])
        )
        facts = [("Price", f"${s['price']:,.0f}  ({s['timeframe']})"),
                 (trigger_label, trigger_val)]
        if is_buy:
            facts.append(("Stop-loss", f"${s['stop_level']:,.0f}"))
        facts.append(("Indicators", s["context"]))
        rows = "".join(
            f'<tr><td style="{fact_css}color:#64748b;width:120px;">{k}</td>'
            f'<td style="{fact_css}color:#0f172a;font-weight:500;">{v}</td></tr>'
            for k, v in facts
        )
        cards += (
            f'<div style="border:1px solid {accent};border-left:4px solid {accent};'
            'border-radius:8px;padding:14px 16px;margin-bottom:12px;">'
            f'<div style="font-size:16px;font-weight:700;color:#0f172a;">{icon} · {coin} — {name.replace("Strategy", "")}'
            f' <span style="font-size:11px;font-weight:500;color:#64748b;">· {s["timeframe"]}</span></div>'
            f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="margin-top:8px;">{rows}</table>'
            "</div>"
        )
        text += [f"{icon} · {coin} — {name}", f"  price ${s['price']:,.0f}",
                 f"  {trigger_label.lower()}: {trigger_val}", ""]

    summary = " · ".join(p for p in [f"{n_buy} BUY" if n_buy else "", f"{n_sell} SELL" if n_sell else ""] if p)
    html = f"""<!DOCTYPE html>
<html><body style="margin:0;padding:0;background:#eef2f6;">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#eef2f6;padding:24px 0;"><tr><td align="center">
<table role="presentation" width="600" cellpadding="0" cellspacing="0" style="max-width:600px;width:100%;background:#ffffff;border-radius:12px;overflow:hidden;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;box-shadow:0 1px 3px rgba(0,0,0,0.08);">
  <tr><td style="background:#0f172a;padding:20px 24px;">
    <div style="color:#ffffff;font-size:19px;font-weight:700;">⚡ Crypto Signal Alert</div>
    <div style="color:#94a3b8;font-size:13px;margin-top:4px;">{now}</div>
  </td></tr>
  <tr><td style="padding:20px 24px;">
    <div style="font-size:13px;color:#475569;margin-bottom:14px;">
      {summary} signal(s) just fired (🟢 buy = entry setup · 🔴 sell = exit condition):
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
    sigs = current_signals()  # {pair: {strategy: signal}}
    prev = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}
    # Only alert once we have a comparable previous state in the current format;
    # otherwise (first run / format change) just seed state silently.
    seeding = not (prev and all(isinstance(v, dict) for v in prev.values()))

    fired = []  # (pair, name, side, signal)
    state = {}
    for pair, strat_sigs in sigs.items():
        for name, s in strat_sigs.items():
            key = f"{pair}|{name}"
            e_now = "Y" if s["entry_signal"] else "N"
            x_now = "Y" if s["exit_signal"] else "N"
            state[key] = {"entry": e_now, "exit": x_now}
            pk = prev.get(key)
            pk = pk if isinstance(pk, dict) else {}
            if e_now == "Y" and pk.get("entry") != "Y":
                fired.append((pair, name, "BUY", s))
            if x_now == "Y" and pk.get("exit") != "Y":
                fired.append((pair, name, "SELL", s))

    STATE_FILE.write_text(json.dumps(state))

    if seeding:
        print("[alerts] seeded state (first run / format change); no email this run")
        return 0
    if not fired:
        print("[alerts] no new BUY/SELL signals; no email sent")
        return 0

    html, text = build_alert(fired)
    n_buy = sum(1 for f in fired if f[2] == "BUY")
    n_sell = len(fired) - n_buy
    bits = " · ".join(p for p in [f"{n_buy} BUY" if n_buy else "", f"{n_sell} SELL" if n_sell else ""] if p)
    subject = f"⚡ Crypto Signal Alert — {bits}: " + ", ".join(
        f"{_PAIR_META.get(p, {}).get('emoji', '')}{n.replace('Strategy', '')}({sd[0]})"
        for p, n, sd, _ in fired
    )
    print(dr.maybe_email(subject, html, text))
    return 0


if __name__ == "__main__":
    sys.exit(main())
