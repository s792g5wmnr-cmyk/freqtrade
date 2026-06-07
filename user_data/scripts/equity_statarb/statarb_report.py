#!/usr/bin/env python3
"""
statarb_report.py
=================
A professional, emailed report for the long-only ETF statistical-arbitrage model
(see statarb_pairs.py). Mirrors the BTC freqtrade report's look & delivery.

For each run it:
  1. Pulls daily ETF data and tests every intra-group pair for cointegration
     (Engle-Granger) on a formation window.
  2. Backtests each pair OUT-OF-SAMPLE (long-only, fully-invested "cheaper leg"
     mode) and computes CAGR / Sharpe / Max-DD vs. buy-and-hold.
  3. Ranks the pairs and picks the best cointegrated one.
  4. Computes the CURRENT entry point: which leg to hold now (from the latest
     spread z-score) and whether the signal just flipped today.
  5. Renders a clean HTML email (+ Markdown fallback) and optionally sends it via
     Resend (HTTPS) or SMTP -- same env contract as the BTC report.

Run:    python statarb_report.py
Email:  set REPORT_EMAIL_TO and either RESEND_API_KEY (+REPORT_EMAIL_FROM) or
        SMTP_HOST/SMTP_PORT/SMTP_USER/SMTP_PASS.
"""
from __future__ import annotations

import itertools
import os
import smtplib
import sys
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import pandas as pd
from statsmodels.tsa.stattools import coint

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from statarb_pairs import (  # noqa: E402  (reuse the model, single source of truth)
    GROUPS, START, FORMATION_YEARS, LOOKBACK, ENTRY_Z, EXIT_Z, COINT_P,
    fetch_prices, rolling_zscore, state_machine, backtest_pair, metrics,
)

REPORTS = HERE / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)
ALLOW_CASH = False  # report mode B: fully invested, always hold the cheaper leg


# --------------------------------------------------------------------------- #
# Current entry point (latest closed candle, unshifted)
# --------------------------------------------------------------------------- #
def current_signal(a: pd.Series, b: pd.Series) -> dict:
    """Which leg to hold *now*, the spread z-score, and whether it flipped today."""
    beta, _, z = rolling_zscore(a, b, LOOKBACK)
    raw = state_machine(z, ENTRY_Z, EXIT_Z, ALLOW_CASH)
    target, prev = int(raw.iloc[-1]), int(raw.iloc[-2])
    return {
        "target": target,                 # +1 hold A, -1 hold B, 0 cash
        "switched": target != prev,
        "z": float(z.iloc[-1]),
        "beta": float(beta.iloc[-1]),
        "asof": z.index[-1].date(),
        "price_a": float(a.iloc[-1]),
        "price_b": float(b.iloc[-1]),
    }


# --------------------------------------------------------------------------- #
# Build per-pair results (OOS metrics + cointegration + live signal)
# --------------------------------------------------------------------------- #
def analyse(px: pd.DataFrame) -> list[dict]:
    split = px.index[0] + pd.DateOffset(years=FORMATION_YEARS)
    form, trade = px[px.index < split], px[px.index >= split]
    rows = []
    for group, tickers in GROUPS.items():
        present = [t for t in tickers if t in px.columns]
        for A, B in itertools.combinations(present, 2):
            fa, fb = form[A].dropna(), form[B].dropna()
            fidx = fa.index.intersection(fb.index)
            if len(fidx) < LOOKBACK * 3:
                continue
            try:
                pval = coint(fa[fidx], fb[fidx])[1]
            except Exception:
                continue

            ta, tb = trade[A].dropna(), trade[B].dropna()
            tidx = ta.index.intersection(tb.index)
            ta, tb = ta[tidx], tb[tidx]
            bt = backtest_pair(ta, tb, allow_cash=ALLOW_CASH)
            m = metrics(bt["ret"])
            bh_a = metrics(ta.pct_change().fillna(0))
            bh_b = metrics(tb.pct_change().fillna(0))
            bh_5050 = metrics(0.5 * ta.pct_change().fillna(0)
                              + 0.5 * tb.pct_change().fillna(0))

            full_a, full_b = px[A].dropna(), px[B].dropna()
            fidx2 = full_a.index.intersection(full_b.index)
            sig = current_signal(full_a[fidx2], full_b[fidx2])

            rows.append({
                "group": group, "A": A, "B": B, "coint_p": pval,
                "cagr": m["CAGR"], "sharpe": m["Sharpe"], "maxdd": m["MaxDD"],
                "bh_a": bh_a, "bh_b": bh_b, "bh_5050": bh_5050,
                "switches": int(bt["switch"].sum()),
                "n_days": len(bt), "sig": sig,
                "oos_start": ta.index[0].date(), "oos_end": ta.index[-1].date(),
            })
    rows.sort(key=lambda r: (r["sharpe"] if r["sharpe"] is not None else -9),
              reverse=True)
    return rows


def hold_label(r: dict) -> str:
    t = r["sig"]["target"]
    return r["A"] if t > 0 else (r["B"] if t < 0 else "Cash")


def cheap_leg_text(r: dict) -> str:
    z, t = r["sig"]["z"], r["sig"]["target"]
    if t > 0:
        return f"{r['A']} is cheap vs {r['B']} (z = {z:+.2f})"
    if t < 0:
        return f"{r['B']} is cheap vs {r['A']} (z = {z:+.2f})"
    return f"spread near fair value (z = {z:+.2f}) — in cash"


# --------------------------------------------------------------------------- #
# Formatting helpers (shared style with the BTC report)
# --------------------------------------------------------------------------- #
def _pf(v, suffix="%", sign=""):
    if v is None:
        return "n/a"
    v = v * 100 if suffix == "%" else v
    return f"{v:+.2f}{suffix}" if sign == "+" else f"{v:.2f}{suffix}"


def _color(v):
    if v is None:
        return "#64748b"
    return "#16a34a" if v >= 0 else "#dc2626"


# --------------------------------------------------------------------------- #
# Markdown fallback
# --------------------------------------------------------------------------- #
def build_md(rows: list[dict], best: dict) -> str:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    L = [f"# Equity Stat-Arb Report — {today}",
         f"\nLong-only relative-value mean-reversion on cointegrated ETF pairs.",
         f"Out-of-sample {best['oos_start']} → {best['oos_end']}.\n",
         "## Pair ranking (by out-of-sample Sharpe)\n",
         "| Pair | Group | Coint p | CAGR | Sharpe | Max DD | Now hold |",
         "|---|---|---|---|---|---|---|"]
    for r in rows:
        tag = f"**{r['A']}/{r['B']}** 🏆" if r is best else f"{r['A']}/{r['B']}"
        L.append(f"| {tag} | {r['group']} | {r['coint_p']:.3f} | "
                 f"{_pf(r['cagr'],'%','+')} | {_pf(r['sharpe'],'')} | "
                 f"{_pf(r['maxdd'],'%')} | {hold_label(r)} |")
    s = best["sig"]
    action = (f"SWITCH → hold {hold_label(best)} (entry today)" if s["switched"]
              else f"Hold {hold_label(best)}")
    L += [f"\n## Best pair: {best['A']}/{best['B']} — {action}",
          f"- Signal: {cheap_leg_text(best)}",
          f"- Hedge ratio β: {s['beta']:.3f}  ·  entry |z|≥{ENTRY_Z}",
          f"- OOS: CAGR {_pf(best['cagr'],'%','+')} · Sharpe {_pf(best['sharpe'],'')}"
          f" · Max DD {_pf(best['maxdd'],'%')}",
          f"- Buy&Hold {best['A']}: Sharpe {_pf(best['bh_a']['Sharpe'],'')} · "
          f"{best['B']}: Sharpe {_pf(best['bh_b']['Sharpe'],'')} · "
          f"50/50: Sharpe {_pf(best['bh_5050']['Sharpe'],'')}",
          "\n---",
          "_Historical backtest figures, not a forecast. Educational only — not "
          "financial advice._"]
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# HTML report (email-client-safe, inline styles) — matches the BTC report
# --------------------------------------------------------------------------- #
def _badge(best: dict) -> str:
    s = best["sig"]
    if s["target"] == 0:
        txt, bg, fg = "● No position — spread near fair value", "#e2e8f0", "#475569"
    elif s["switched"]:
        txt, bg, fg = f"● SWITCH → BUY {hold_label(best)} (entry today)", "#16a34a", "#ffffff"
    else:
        txt, bg, fg = f"● HOLD {hold_label(best)}", "#16a34a", "#ffffff"
    return (f'<span style="background:{bg};color:{fg};padding:5px 14px;'
            f'border-radius:999px;font-size:13px;font-weight:600;">{txt}</span>')


def _best_card(best: dict) -> str:
    s = best["sig"]

    def fact(label, value):
        return ('<tr>'
                f'<td style="padding:7px 0;color:#64748b;font-size:13px;width:160px;vertical-align:top;">{label}</td>'
                f'<td style="padding:7px 0;color:#0f172a;font-size:13px;font-weight:500;">{value}</td></tr>')

    cagr = f'<span style="color:{_color(best["cagr"])};font-weight:600;">{_pf(best["cagr"], "%", "+")}</span>'
    facts = (
        fact("Entry point", f'<strong>{cheap_leg_text(best)}</strong>')
        + fact("Action now", f"Hold <strong>{hold_label(best)}</strong>"
               + (" &nbsp;<span style='color:#16a34a;'>(fresh switch today)</span>" if s["switched"] else ""))
        + fact("Spread z-score", f"{s['z']:+.2f} &nbsp;<span style='color:#94a3b8;'>(enter when |z| ≥ {ENTRY_Z})</span>")
        + fact("Hedge ratio β", f"{s['beta']:.3f}")
        + fact("Latest prices", f"{best['A']} ${s['price_a']:,.2f} &nbsp;·&nbsp; {best['B']} ${s['price_b']:,.2f}")
        + fact("OOS return (CAGR)*", cagr)
        + fact("OOS Sharpe*", f"{_pf(best['sharpe'], '')}")
        + fact("OOS max drawdown*", _pf(best["maxdd"], "%"))
        + fact("Cointegration p", f"{best['coint_p']:.4f} &nbsp;<span style='color:#94a3b8;'>(formation window)</span>")
        + fact("vs Buy &amp; Hold",
               f"{best['A']} Sharpe {_pf(best['bh_a']['Sharpe'],'')} · "
               f"{best['B']} Sharpe {_pf(best['bh_b']['Sharpe'],'')} · "
               f"50/50 {_pf(best['bh_5050']['Sharpe'],'')}")
    )
    return f"""
  <tr><td style="padding:20px 28px 0;">
    <div style="background:#0f172a;border-radius:8px;padding:12px 16px;">
      <span style="color:#ffffff;font-size:17px;font-weight:700;">🥇 Top pair · {best['A']} / {best['B']}</span>
      <span style="color:#94a3b8;font-size:12px;"> &nbsp;{best['group']} sector · long-only relative value</span>
    </div>
  </td></tr>
  <tr><td style="padding:16px 28px 4px;">
    <div style="margin-bottom:16px;">{_badge(best)}</div>
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0">{facts}</table>
  </td></tr>"""


def _ranking_table(rows: list[dict], best: dict) -> str:
    cell = "padding:10px 12px;border-bottom:1px solid #e2e8f0;"
    body = ""
    for i, r in enumerate(rows):
        is_best = r is best
        coint = r["coint_p"] < COINT_P
        bg = "#ecfdf5" if is_best else ("#ffffff" if i % 2 == 0 else "#f8fafc")
        weight = "700" if is_best else "400"
        pair = ("🏆 " if is_best else "") + f"{r['A']}/{r['B']}"
        pmark = (f'<span style="color:{"#16a34a" if coint else "#94a3b8"};">{r["coint_p"]:.3f}</span>'
                 + ("" if coint else " ·"))
        body += (
            f'<tr style="background:{bg};">'
            f'<td style="{cell}font-weight:{weight};color:#0f172a;">{pair}'
            f'<div style="font-size:11px;color:#94a3b8;font-weight:400;">{r["group"]}</div></td>'
            f'<td style="{cell}text-align:right;">{pmark}</td>'
            f'<td style="{cell}text-align:right;font-weight:600;color:{_color(r["cagr"])};">{_pf(r["cagr"], "%", "+")}</td>'
            f'<td style="{cell}text-align:right;color:#334155;">{_pf(r["sharpe"], "")}</td>'
            f'<td style="{cell}text-align:right;color:#334155;">{_pf(r["maxdd"], "%")}</td>'
            f'<td style="{cell}text-align:right;color:#0f172a;font-weight:500;">{hold_label(r)}</td></tr>'
        )
    return f"""
  <tr><td style="padding:8px 28px 20px;">
    <div style="font-size:11px;letter-spacing:1px;color:#64748b;text-transform:uppercase;margin:6px 0 10px;">All pairs · ranked by out-of-sample Sharpe</div>
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;font-size:13px;border:1px solid #e2e8f0;">
      <tr style="background:#1e293b;">
        <th style="padding:10px 12px;text-align:left;color:#cbd5e1;font-weight:600;">Pair</th>
        <th style="padding:10px 12px;text-align:right;color:#cbd5e1;font-weight:600;">Coint&nbsp;p</th>
        <th style="padding:10px 12px;text-align:right;color:#cbd5e1;font-weight:600;">CAGR</th>
        <th style="padding:10px 12px;text-align:right;color:#cbd5e1;font-weight:600;">Sharpe</th>
        <th style="padding:10px 12px;text-align:right;color:#cbd5e1;font-weight:600;">Max&nbsp;DD</th>
        <th style="padding:10px 12px;text-align:right;color:#cbd5e1;font-weight:600;">Now&nbsp;hold</th>
      </tr>
      {body}
    </table>
    <div style="font-size:11px;color:#94a3b8;margin-top:8px;">Green coint&nbsp;p = cointegrated at p&lt;{COINT_P} (tradeable). The 🏆 pair is the top-ranked cointegrated one.</div>
  </td></tr>"""


def _guide() -> str:
    fbox = ("margin:6px 0 2px;padding:8px 10px;background:#f1f5f9;border-radius:6px;"
            "font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;"
            "font-size:12px;color:#0f172a;white-space:pre-wrap;line-height:1.7;")
    formulas = "".join(f'<div style="{fbox}">{f}</div>' for f in [
        "spreadₜ = log(Aₜ) − βₜ · log(Bₜ) ,   βₜ = cov(logA, logB) / var(logB)   (rolling)",
        "zₜ = (spreadₜ − mean(spread)) / std(spread)   over the rolling lookback",
        "z ≤ −entry → hold A (A cheap) ;  z ≥ +entry → hold B (B cheap) ;  else keep position",
    ])
    diagram = ("z  +entry ─────────●────────  B cheap → rotate into B\n"
               "    0     ~~~~~~~~/~~~~~~~~~~  fair value (hold current leg)\n"
               "   −entry ──●───────────────  A cheap → rotate into A")
    return f"""<tr><td style="padding:8px 28px 16px;">
    <div style="font-size:11px;letter-spacing:1px;color:#64748b;text-transform:uppercase;margin:14px 0 10px;">📚 How this model works</div>
    <div style="border:1px solid #e2e8f0;border-radius:8px;padding:14px 16px;margin-bottom:14px;">
      <div style="font-size:14px;font-weight:700;color:#0f172a;">Long-only relative-value mean-reversion <span style="font-size:11px;font-weight:500;color:#64748b;">· daily</span></div>
      <div style="font-size:13px;color:#475569;line-height:1.6;margin:8px 0;">
        Two economically-linked ETFs that are <em>cointegrated</em> move together over the long run, so the
        gap (spread) between them mean-reverts. Classic stat-arb trades this gap market-neutral (long the cheap
        leg, short the rich one). Long-only here means we simply <strong>hold whichever leg is currently cheap</strong>
        and rotate when the spread flips — capturing the relative-value tilt with no shorting.
      </div>
      <div style="{fbox}">{diagram}</div>
      <div style="font-size:12px;color:#0f172a;margin:8px 0 2px;"><strong style="color:#16a34a;">Entry:</strong> spread z-score reaches an extreme — buy the relatively cheap leg.</div>
      <div style="font-size:12px;color:#0f172a;margin:2px 0;"><strong style="color:#dc2626;">Exit / switch:</strong> spread reverts past the opposite threshold — rotate into the other leg.</div>
      {formulas}
    </div>
  </td></tr>"""


def build_html(rows: list[dict], best: dict) -> str:
    today = datetime.now(timezone.utc).strftime("%b %d, %Y · %H:%M UTC")
    return f"""<!DOCTYPE html>
<html><body style="margin:0;padding:0;background:#eef2f6;">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#eef2f6;padding:24px 0;">
<tr><td align="center">
<table role="presentation" width="640" cellpadding="0" cellspacing="0" style="max-width:640px;width:100%;background:#ffffff;border-radius:12px;overflow:hidden;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;box-shadow:0 1px 3px rgba(0,0,0,0.08);">
  <tr><td style="background:#0f172a;padding:24px 28px;">
    <div style="color:#ffffff;font-size:20px;font-weight:700;">📊 Equity Stat-Arb Strategy Report</div>
    <div style="color:#94a3b8;font-size:13px;margin-top:4px;">{today} &nbsp;·&nbsp; long-only ETF pairs &nbsp;·&nbsp; OOS {best['oos_start']} → {best['oos_end']}</div>
  </td></tr>
  {_best_card(best)}
  {_ranking_table(rows, best)}
  {_guide()}
  <tr><td style="padding:18px 28px 26px;background:#f8fafc;border-top:1px solid #e2e8f0;">
    <div style="font-size:11px;color:#94a3b8;line-height:1.6;">
      * CAGR, Sharpe and max drawdown are out-of-sample backtest figures over the window above — not a forecast.
      The relative-value edge between near-identical ETFs is thin; this model rarely beats simply holding the
      stronger leg. Past performance does not guarantee future results. <strong>Educational only — not financial advice.</strong>
    </div>
  </td></tr>
</table>
</td></tr>
</table>
</body></html>"""


# --------------------------------------------------------------------------- #
# Email delivery (identical contract to the BTC report)
# --------------------------------------------------------------------------- #
def maybe_email(subject: str, html_body: str, text_body: str) -> str:
    to = os.environ.get("REPORT_EMAIL_TO")
    if not to:
        return "email skipped (REPORT_EMAIL_TO not set)"
    recipients = [a.strip() for a in to.split(",") if a.strip()]

    resend_key = os.environ.get("RESEND_API_KEY")
    if resend_key:
        import json
        import urllib.error
        import urllib.request
        sender = os.environ.get("REPORT_EMAIL_FROM", "onboarding@resend.dev")
        payload = json.dumps({"from": sender, "to": recipients, "subject": subject,
                              "html": html_body, "text": text_body}).encode()
        req = urllib.request.Request(
            "https://api.resend.com/emails", data=payload,
            headers={"Authorization": f"Bearer {resend_key}",
                     "Content-Type": "application/json",
                     "User-Agent": "Mozilla/5.0 (compatible; statarb-report/1.0)"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return f"email sent via Resend to {to} (HTTP {r.status})"
        except urllib.error.HTTPError as e:
            return f"Resend API error {e.code}: {e.read().decode()[:300]}"

    host = os.environ.get("SMTP_HOST")
    if not host:
        return "email skipped (set RESEND_API_KEY, or SMTP_HOST for SMTP delivery)"
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = os.environ.get("SMTP_USER", "statarb-bot")
    msg["To"] = to
    msg.attach(MIMEText(text_body, "plain", "utf-8"))
    msg.attach(MIMEText(html_body, "html", "utf-8"))
    with smtplib.SMTP(host, int(os.environ.get("SMTP_PORT", "587"))) as srv:
        srv.starttls()
        if os.environ.get("SMTP_USER"):
            srv.login(os.environ["SMTP_USER"], os.environ["SMTP_PASS"])
        srv.send_message(msg)
    return f"email sent via SMTP to {to}"


# --------------------------------------------------------------------------- #
def main() -> int:
    all_tickers = sorted({t for g in GROUPS.values() for t in g})
    print(f"[1/3] Downloading {len(all_tickers)} ETFs ...")
    px = fetch_prices(all_tickers)
    print(f"[2/3] Analysing pairs ({px.index[0].date()} → {px.index[-1].date()}) ...")
    rows = analyse(px)
    if not rows:
        print("No pairs analysed."); return 1
    tradeable = [r for r in rows if r["coint_p"] < COINT_P]
    best = (tradeable or rows)[0]

    html = build_html(rows, best)
    md = build_md(rows, best)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    (REPORTS / f"{today}.html").write_text(html, encoding="utf-8")
    (REPORTS / f"{today}.md").write_text(md, encoding="utf-8")
    print(f"[3/3] Report written to {REPORTS}/{today}.html")

    action = (f"SWITCH→{hold_label(best)}" if best["sig"]["switched"]
              else f"hold {hold_label(best)}")
    subject = f"📊 Equity Stat-Arb {today} — {best['A']}/{best['B']}: {action}"
    print(maybe_email(subject, html, md))
    print("\n" + md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
