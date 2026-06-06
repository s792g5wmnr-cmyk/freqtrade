#!/usr/bin/env bash
#
# vps_setup.sh — turn a fresh Ubuntu VPS into a daily BTC-strategy-report machine.
#
# Run as root on a fresh Ubuntu 22.04/24.04 server:
#     curl -fsSL https://raw.githubusercontent.com/s792g5wmnr-cmyk/freqtrade/develop/user_data/scripts/vps_setup.sh -o vps_setup.sh
#     sudo bash vps_setup.sh
#
# Then edit the secrets file it creates (add your Gmail app password) and run a test.
#
set -euo pipefail

REPO_URL="https://github.com/s792g5wmnr-cmyk/freqtrade.git"
BRANCH="develop"
INSTALL_DIR="${INSTALL_DIR:-/opt/freqtrade-report}"
ENV_FILE="${ENV_FILE:-/etc/freqtrade-report.env}"
RUN_HOUR_UTC="${RUN_HOUR_UTC:-13}"   # 13:00 UTC = 9:00 AM America/New_York (EDT)

echo "==> [1/8] Installing system packages ..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y python3 python3-venv python3-dev build-essential git wget curl pkg-config cron
systemctl enable --now cron 2>/dev/null || true

echo "==> [2/8] Cloning the repo to $INSTALL_DIR ..."
if [ -d "$INSTALL_DIR/.git" ]; then
    git -C "$INSTALL_DIR" fetch --depth 1 origin "$BRANCH"
    git -C "$INSTALL_DIR" reset --hard "origin/$BRANCH"
else
    git clone --branch "$BRANCH" --depth 1 "$REPO_URL" "$INSTALL_DIR"
fi
cd "$INSTALL_DIR"

echo "==> [3/8] Creating Python venv and installing freqtrade (several minutes) ..."
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -U pip wheel >/dev/null
pip install -e .

echo "==> [4/8] Ensuring TA-Lib is importable ..."
if ! python -c "import talib" 2>/dev/null; then
    echo "    TA-Lib wheel not usable; building the C library from source ..."
    cd /tmp
    wget -q -O ta-lib.tar.gz https://github.com/TA-Lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz
    tar xzf ta-lib.tar.gz
    cd ta-lib-0.6.4/
    ./configure --prefix=/usr >/dev/null
    make >/dev/null
    make install
    ldconfig
    cd "$INSTALL_DIR"
    pip install --no-cache-dir --force-reinstall TA-Lib
fi
python -c "import talib, freqtrade, pyarrow; print('    deps OK')"

echo "==> [5/8] Auto-detecting a reachable exchange for data ..."
START=$(date -u -d '6 days ago' +%Y%m%d); END=$(date -u +%Y%m%d)
PICKED=""
for EX in binanceus binance coinbase kucoin; do
    sed -i 's/"name": *"[^"]*"/"name": "'"$EX"'"/' user_data/config.json
    if freqtrade download-data --config user_data/config.json --pairs BTC/USDT \
            --timeframe 1h --timerange "${START}-${END}" >/tmp/dl.log 2>&1; then
        PICKED="$EX"; echo "    Using exchange: $EX"; break
    fi
done
if [ -z "$PICKED" ]; then
    echo "    !! No exchange was reachable from this VPS. Check the network / try another region." >&2
    exit 1
fi

echo "==> [6/8] Writing secrets file $ENV_FILE (if absent) ..."
if [ ! -f "$ENV_FILE" ]; then
    cat > "$ENV_FILE" <<'EOF'
# Daily report email settings. Fill in SMTP_PASS with your Gmail APP PASSWORD
# (16 chars, from https://myaccount.google.com/apppasswords). chmod 600.
REPORT_EMAIL_TO="owen19910930@gmail.com, owen19910930@hotmail.com"
SMTP_HOST="smtp.gmail.com"
SMTP_PORT="587"
SMTP_USER="owen19910930@gmail.com"
SMTP_PASS="PUT_YOUR_GMAIL_APP_PASSWORD_HERE"
EOF
    chmod 600 "$ENV_FILE"
    echo "    Created $ENV_FILE — you MUST edit it to set SMTP_PASS."
else
    echo "    $ENV_FILE already exists; leaving it untouched."
fi

echo "==> [7/8] Writing wrapper $INSTALL_DIR/run_report.sh ..."
cat > "$INSTALL_DIR/run_report.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
set -a; source "$ENV_FILE"; set +a
cd "$INSTALL_DIR"
source "$INSTALL_DIR/.venv/bin/activate"
exec python user_data/scripts/daily_strategy_report.py
EOF
chmod +x "$INSTALL_DIR/run_report.sh"

echo "==> [8/8] Installing daily cron job at ${RUN_HOUR_UTC}:00 UTC ..."
CRON_LINE="0 ${RUN_HOUR_UTC} * * * $INSTALL_DIR/run_report.sh >> /var/log/freqtrade-report.log 2>&1"
EXISTING_CRON="$(crontab -l 2>/dev/null | grep -v 'run_report.sh' || true)"
printf '%s\n%s\n' "$EXISTING_CRON" "$CRON_LINE" | crontab -

cat <<EOF

============================================================
 SETUP COMPLETE.
 Exchange:    $PICKED
 Install dir: $INSTALL_DIR
 Secrets:     $ENV_FILE   (edit this to add your Gmail app password)
 Cron:        daily at ${RUN_HOUR_UTC}:00 UTC  (= 9 AM ET in summer)
 Logs:        /var/log/freqtrade-report.log

 NEXT STEPS:
   1. nano $ENV_FILE        # set SMTP_PASS to your 16-char Gmail app password
   2. $INSTALL_DIR/run_report.sh   # send a test report now
============================================================
EOF
