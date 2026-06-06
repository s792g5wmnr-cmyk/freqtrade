# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
"""
BtcDipBuyerStrategy
===================
A long-only MEAN-REVERSION strategy for Bitcoin (BTC/USDT, 4h).

WHY: In a falling/choppy market a trend-follower has no trend to ride. The other
long-only edge is mean reversion: buy capitulation (deeply oversold) dips and sell
the bounce. We take quick profits (mean reversion is short-lived) and cut losers.

LOGIC
  Entry : RSI < oversold (panic) AND price closed below the lower Bollinger Band
          (a statistically stretched move) -> buy the dip.
  Exit  : RSI recovers above `rsi_exit` (the bounce has played out),
          OR the ROI ladder takes a quick profit, OR the stop cuts a loser.

NOTE: Mean reversion is the mirror risk of trend-following — it profits in chop and
gets run over by strong sustained trends (it keeps "buying the dip" all the way
down). One good backtest in one regime proves nothing. Educational, not advice.
"""
from pandas import DataFrame
from freqtrade.strategy import IStrategy, IntParameter
import talib.abstract as ta
from technical import qtpylib


class BtcDipBuyerStrategy(IStrategy):
    INTERFACE_VERSION = 3
    can_short: bool = False

    timeframe = "4h"

    # Take quick profits — mean-reversion bounces don't last.
    minimal_roi = {"0": 0.06, "240": 0.03, "720": 0.0}
    stoploss = -0.08

    trailing_stop = False
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False

    startup_candle_count: int = 50

    rsi_entry = IntParameter(15, 35, default=28, space="buy", optimize=True)
    rsi_exit = IntParameter(45, 65, default=55, space="sell", optimize=True)

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        bb = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe["bb_lower"] = bb["lower"]
        dataframe["bb_mid"] = bb["mid"]
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["rsi"] < self.rsi_entry.value)          # deeply oversold
                & (dataframe["close"] < dataframe["bb_lower"])     # stretched below band
                & (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["rsi"] > self.rsi_exit.value)           # bounce played out
                & (dataframe["volume"] > 0)
            ),
            "exit_long",
        ] = 1
        return dataframe
