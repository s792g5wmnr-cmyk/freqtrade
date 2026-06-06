# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
"""
BtcTrendRiderStrategy
=====================
A long-only TREND-RIDING strategy for Bitcoin (BTC/USDT, 4h).

WHY THIS DESIGN (and why it differs from BtcEmaRsiStrategy):
  Our earlier strategy *scalped out* of every move (tight ROI ladder + RSI-overbought
  exit), so it never captured a trend and lost to buy-and-hold by a mile.
  This one does the opposite — the two rules every trend-follower lives by:
      1. Cut losers fast.
      2. LET WINNERS RUN.
  So we DISABLE the take-profit ladder and only exit when the trend actually dies.

LOGIC
  Regime filter : only go long when price is above the 200-EMA (a real uptrend).
  Entry         : a Donchian breakout — close makes a new `donchian` -bar high,
                  with EMA50 > EMA200 and ADX confirming trend strength.
  Exit          : trend death — price closes back below the 50-EMA.
  Risk          : wide stop so the trend can breathe, plus a trailing stop that
                  locks in profit only after a healthy gain (so winners can run).

NOTE: This is a trend-follower. It is EXPECTED to shine in trending/bull markets
and to chop in sideways markets. A great single-year backtest is NOT proof it will
work in the future — always check it across multiple regimes (see the out-of-sample
results we generated). Educational, not financial advice.
"""
from datetime import datetime
from pandas import DataFrame

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter

import talib.abstract as ta
from technical import qtpylib


class BtcTrendRiderStrategy(IStrategy):
    INTERFACE_VERSION = 3
    can_short: bool = False

    timeframe = "4h"

    # --- Let winners run: effectively DISABLE the ROI take-profit. ---
    # We never want ROI to cut a trending winner short; exits are signal/stop driven.
    minimal_roi = {"0": 100.0}

    # Wide hard stop so normal volatility doesn't shake us out of a good trend.
    stoploss = -0.15

    # Trailing stop: do nothing until we're +12% in profit, then trail by 5%.
    # This lets a trade run a long way before the trail engages, then protects gains.
    trailing_stop = True
    trailing_stop_positive = 0.05
    trailing_stop_positive_offset = 0.12
    trailing_only_offset_is_reached = True

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # 200-period EMA + Donchian need history before signals are valid.
    startup_candle_count: int = 210

    # --- Hyperoptable parameters (sensible trend-following defaults) ---
    ema_fast = IntParameter(20, 80, default=50, space="buy", optimize=True)
    ema_slow = IntParameter(120, 250, default=200, space="buy", optimize=True)
    donchian = IntParameter(10, 40, default=20, space="buy", optimize=True)
    adx_min = IntParameter(15, 35, default=20, space="buy", optimize=True)

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    plot_config = {
        "main_plot": {
            "ema_fast": {"color": "cyan"},
            "ema_slow": {"color": "orange"},
            "donchian_high": {"color": "green"},
        },
        "subplots": {
            "ADX": {"adx": {"color": "blue"}},
        },
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=int(self.ema_fast.value))
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=int(self.ema_slow.value))
        dataframe["adx"] = ta.ADX(dataframe)

        # Donchian channel upper band = highest high of the last N bars (excluding
        # the current bar, via shift) -> a close above it is a fresh breakout.
        dataframe["donchian_high"] = (
            dataframe["high"].rolling(int(self.donchian.value)).max().shift(1)
        )
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # Regime: only trade long inside a confirmed uptrend
                (dataframe["close"] > dataframe["ema_slow"])
                & (dataframe["ema_fast"] > dataframe["ema_slow"])
                # Momentum breakout: new N-bar high
                & (dataframe["close"] > dataframe["donchian_high"])
                # Trend has real strength
                & (dataframe["adx"] > self.adx_min.value)
                & (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # Trend death: price closes back below the fast EMA.
                qtpylib.crossed_below(dataframe["close"], dataframe["ema_fast"])
                & (dataframe["volume"] > 0)
            ),
            "exit_long",
        ] = 1
        return dataframe
