# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
"""
SqueezeBreakoutStrategy
=======================
Bollinger/Keltner "squeeze" volatility-breakout strategy (long-only, BTC/USDT, 1h).

Edge (TTM Squeeze): when the Bollinger Bands contract INSIDE the Keltner Channel,
volatility is compressed and a large directional move often follows. We wait for the
squeeze to RELEASE and enter in the direction of momentum.

  Squeeze ON : BB upper < KC upper AND BB lower > KC lower (bands compressed inside KC)
  Entry(long): squeeze was ON last bar and is now OFF (released)  AND  close > BB mid
               AND momentum positive (ROC > 0)  AND  price > EMA200 (long-only trend guard)
  Exit:        close < BB mid  OR  ROC < 0 (momentum fades).

Showcases: combining two volatility envelopes (Bollinger + Keltner). Educational only.
"""
from pandas import DataFrame

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter

import talib.abstract as ta
from technical import qtpylib


class SqueezeBreakoutStrategy(IStrategy):
    INTERFACE_VERSION = 3
    can_short: bool = False

    timeframe = "1h"

    minimal_roi = {"0": 0.10, "360": 0.05, "720": 0.0}
    stoploss = -0.06

    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.04
    trailing_only_offset_is_reached = True

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False

    startup_candle_count: int = 200

    bb_window = IntParameter(15, 30, default=20, space="buy", optimize=True)
    roc_period = IntParameter(6, 20, default=12, space="buy", optimize=True)

    order_types = {
        "entry": "limit", "exit": "limit",
        "stoploss": "market", "stoploss_on_exchange": False,
    }
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        win = int(self.bb_window.value)
        bb = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=win, stds=2)
        dataframe["bb_upper"] = bb["upper"]
        dataframe["bb_mid"] = bb["mid"]
        dataframe["bb_lower"] = bb["lower"]

        kc = qtpylib.keltner_channel(dataframe, window=win, atrs=1.5)
        dataframe["kc_upper"] = kc["upper"]
        dataframe["kc_lower"] = kc["lower"]

        # Squeeze ON when Bollinger Bands are inside the Keltner Channel
        dataframe["squeeze_on"] = (
            (dataframe["bb_upper"] < dataframe["kc_upper"])
            & (dataframe["bb_lower"] > dataframe["kc_lower"])
        )

        dataframe["roc"] = ta.ROC(dataframe, timeperiod=int(self.roc_period.value))
        dataframe["ema200"] = ta.EMA(dataframe, timeperiod=200)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # Squeeze just released (was ON last bar, OFF now)
                (dataframe["squeeze_on"].shift(1))
                & (~dataframe["squeeze_on"])
                # Break to the upside with momentum
                & (dataframe["close"] > dataframe["bb_mid"])
                & (dataframe["roc"] > 0)
                # Long-only trend guard
                & (dataframe["close"] > dataframe["ema200"])
                & (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                ((dataframe["close"] < dataframe["bb_mid"]) | (dataframe["roc"] < 0))
                & (dataframe["volume"] > 0)
            ),
            "exit_long",
        ] = 1
        return dataframe
