# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
"""
BtcEmaRsiStrategy
=================
A simple, readable sample strategy for trading Bitcoin (BTC/USDT) on freqtrade.

Idea (trend-following + momentum confirmation):
  ENTRY (long): fast EMA crosses ABOVE slow EMA  (uptrend begins)
                AND RSI is in a healthy-momentum band (not oversold, not overbought)
                AND ADX confirms the trend has real strength
                AND there is volume.
  EXIT  (long): fast EMA crosses BELOW slow EMA   (trend reverses)
                OR  RSI becomes overbought (take profit into strength).

This is long-only and meant as an educational starting point — NOT financial
advice. Always backtest and dry-run before risking real funds.

Backtest example (from the freqtrade repo root, inside your freqtrade env):
    freqtrade download-data --exchange binance --pairs BTC/USDT \
        --timeframe 1h --timerange 20230101-20240101
    freqtrade backtesting --strategy BtcEmaRsiStrategy --pairs BTC/USDT \
        --timeframe 1h --timerange 20230101-20240101
"""
from datetime import datetime
from pandas import DataFrame

from freqtrade.strategy import (
    IStrategy,
    IntParameter,
    DecimalParameter,
)

import talib.abstract as ta
from technical import qtpylib


class BtcEmaRsiStrategy(IStrategy):
    # Interface version — keep at 3 for current freqtrade.
    INTERFACE_VERSION = 3

    # Long-only strategy (set True only for margin/futures with shorting enabled).
    can_short: bool = False

    # --- Core risk settings -------------------------------------------------
    # Timeframe: 1h is a good balance for BTC — fewer false signals than 5m,
    # still responsive enough to ride multi-day trends.
    timeframe = "1h"

    # Take-profit ladder (minutes -> required profit fraction).
    # Exit at 4% immediately if hit; accept smaller gains the longer a trade is open.
    minimal_roi = {
        "0": 0.06,     # 6% take profit at any time
        "240": 0.03,   # after 4h, accept 3%
        "480": 0.015,  # after 8h, accept 1.5%
        "720": 0,      # after 12h, exit at break-even
    }

    # Hard stoploss: cut a losing trade at -8%.
    stoploss = -0.08

    # Trailing stop: once a trade is +2% in profit, trail by 1.5% to lock gains.
    trailing_stop = True
    trailing_stop_positive = 0.015
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    # Only recompute indicators on a new candle (faster, avoids repainting).
    process_only_new_candles = True

    # Use our exit signals in addition to ROI / stoploss.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # We need enough history for the 200-period trend filter + slow EMA.
    startup_candle_count: int = 200

    # --- Hyperoptable parameters -------------------------------------------
    # These can be tuned with `freqtrade hyperopt`. Defaults are sensible.
    ema_fast = IntParameter(8, 30, default=12, space="buy", optimize=True)
    ema_slow = IntParameter(20, 60, default=26, space="buy", optimize=True)
    rsi_entry_min = IntParameter(40, 60, default=50, space="buy", optimize=True)
    rsi_entry_max = IntParameter(60, 80, default=70, space="buy", optimize=True)
    adx_min = IntParameter(15, 40, default=20, space="buy", optimize=True)
    rsi_exit = IntParameter(70, 90, default=78, space="sell", optimize=True)

    # Order execution
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    # Chart plotting config (used by `freqtrade plot-dataframe`).
    plot_config = {
        "main_plot": {
            "ema_fast": {"color": "cyan"},
            "ema_slow": {"color": "orange"},
            "ema_trend": {"color": "grey"},
        },
        "subplots": {
            "RSI": {"rsi": {"color": "red"}},
            "ADX": {"adx": {"color": "blue"}},
        },
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Trend EMAs
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=int(self.ema_fast.value))
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=int(self.ema_slow.value))
        # Long-term trend filter: only go long when price is above the 200 EMA.
        dataframe["ema_trend"] = ta.EMA(dataframe, timeperiod=200)

        # Momentum
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        # Trend strength
        dataframe["adx"] = ta.ADX(dataframe)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # Fast EMA crosses above slow EMA -> fresh bullish momentum
                qtpylib.crossed_above(dataframe["ema_fast"], dataframe["ema_slow"])
                # Only trade in line with the broader uptrend
                & (dataframe["close"] > dataframe["ema_trend"])
                # RSI in a healthy band: momentum present but not exhausted
                & (dataframe["rsi"] > self.rsi_entry_min.value)
                & (dataframe["rsi"] < self.rsi_entry_max.value)
                # Trend has real strength
                & (dataframe["adx"] > self.adx_min.value)
                # Never trade a dead candle
                & (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (
                    # Trend reversal: fast EMA crosses back below slow EMA
                    qtpylib.crossed_below(dataframe["ema_fast"], dataframe["ema_slow"])
                    # OR momentum is overbought -> take profit into strength
                    | (dataframe["rsi"] > self.rsi_exit.value)
                )
                & (dataframe["volume"] > 0)
            ),
            "exit_long",
        ] = 1

        return dataframe
