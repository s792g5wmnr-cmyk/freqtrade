#!/usr/bin/env bash
#
# run_equity_report.sh — daily long-only ETF statistical-arbitrage report.
# Mirrors run_report.sh: load the shared secrets, activate the venv, run the script.
# Installed at /opt/freqtrade-report/run_equity_report.sh; called by cron.
#
set -euo pipefail
ENV_FILE="${ENV_FILE:-/etc/freqtrade-report.env}"
INSTALL_DIR="${INSTALL_DIR:-/opt/freqtrade-report}"
set -a; source "$ENV_FILE"; set +a
cd "$INSTALL_DIR"
source "$INSTALL_DIR/.venv/bin/activate"
exec python user_data/scripts/equity_statarb/statarb_report.py
