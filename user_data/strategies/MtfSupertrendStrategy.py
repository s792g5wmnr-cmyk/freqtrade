# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
"""
MtfSupertrendStrategy
=====================
Multi-Timeframe Supertrend trend-follower (long-only, BTC/USDT, 4h).

Edge: trend-following with a HIGHER-TIMEFRAME regime filter. The 4h Supertrend
gives entries/exits; the DAILY trend (via the @informative decorator) gates them,
so we only take longs when the bigger picture is bullish. This is the fix for the
chop that hurt the simple EMA-cross strategy.

  Entry (long): 4h Supertrend flips bullish  AND  daily close > daily EMA200
                AND daily EMA50 > daily EMA200 (confirmed daily uptrend).
  Exit:         4h Supertrend flips bearish.

Showcases: the @informative('1d') multi-timeframe decorator. Educational only.
"""
import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, informative

import talib.abstract as ta


def supertrend(dataframe: DataFrame, period: int = 10, multiplier: float = 3.0):
    """Return (supertrend_line, direction) where direction is +1 bullish / -1 bearish."""
    atr = ta.ATR(dataframe, timeperiod=period).values
    hl2 = (dataframe["high"] + dataframe["low"]) / 2
    basic_ub = (hl2 + multiplier * atr).values
    basic_lb = (hl2 - multiplier * atr).values
    close = dataframe["close"].values
    n = len(dataframe)
    final_ub = basic_ub.copy()
    final_lb = basic_lb.copy()
    # Seed the recursion at the first index where ATR is valid, otherwise the
    # NaN warmup propagates forever and the trend direction never flips.
    valid = np.where(~np.isnan(atr))[0]
    first = int(valid[0]) if len(valid) else 0
    st = np.full(n, np.nan)
    direction = np.zeros(n, dtype=int)
    for i in range(first + 1, n):
        final_ub[i] = basic_ub[i] if (basic_ub[i] < final_ub[i - 1] or close[i - 1] > final_ub[i - 1]) else final_ub[i - 1]
        final_lb[i] = basic_lb[i] if (basic_lb[i] > final_lb[i - 1] or close[i - 1] < final_lb[i - 1]) else final_lb[i - 1]
    st[first] = final_ub[first]
    direction[first] = -1
    for i in range(first + 1, n):
        if st[i - 1] == final_ub[i - 1]:
            if close[i] <= final_ub[i]:
                st[i] = final_ub[i]; direction[i] = -1
            else:
                st[i] = final_lb[i]; direction[i] = 1
        else:
            if close[i] >= final_lb[i]:
                st[i] = final_lb[i]; direction[i] = 1
            else:
                st[i] = final_ub[i]; direction[i] = -1
    return pd.Series(st, index=dataframe.index), pd.Series(direction, index=dataframe.index)


class MtfSupertrendStrategy(IStrategy):
    INTERFACE_VERSION = 3
    can_short: bool = False

    timeframe = "4h"

    # Let winners run; exits are driven by the Supertrend flip + stoploss.
    minimal_roi = {"0": 100.0}
    stoploss = -0.12

    trailing_stop = True
    trailing_stop_positive = 0.04
    trailing_stop_positive_offset = 0.10
    trailing_only_offset_is_reached = True

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False

    # Need ~200 daily candles for the daily EMA200 regime filter.
    startup_candle_count: int = 200

    st_period = IntParameter(7, 20, default=10, space="buy", optimize=True)
    st_mult = DecimalParameter(2.0, 4.0, default=3.0, decimals=1, space="buy", optimize=True)

    order_types = {
        "entry": "limit", "exit": "limit",
        "stoploss": "market", "stoploss_on_exchange": False,
    }
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    @informative("1d")
    def populate_indicators_1d(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema200"] = ta.EMA(dataframe, timeperiod=200)
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        st, direction = supertrend(dataframe, int(self.st_period.value), float(self.st_mult.value))
        dataframe["supertrend"] = st
        dataframe["st_dir"] = direction
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # 4h Supertrend flips bullish on this candle
                (dataframe["st_dir"] == 1)
                & (dataframe["st_dir"].shift(1) == -1)
                # Daily regime: confirmed uptrend (columns merged with _1d suffix)
                & (dataframe["close"] > dataframe["ema200_1d"])
                & (dataframe["ema50_1d"] > dataframe["ema200_1d"])
                & (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["st_dir"] == -1)
                & (dataframe["st_dir"].shift(1) == 1)
                & (dataframe["volume"] > 0)
            ),
            "exit_long",
        ] = 1
        return dataframe
