# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
"""
DcaMeanReversionStrategy
========================
Mean-reversion with DCA position scaling (long-only, BTC/USDT, 4h).

Edge: buy oversold dips, and if price keeps falling, SCALE IN with additional
tranches to improve the average entry, then exit on reversion. Uses freqtrade's
adjust_trade_position() (position_adjustment_enable) — its signature feature.

  Initial entry: RSI < oversold AND close < lower Bollinger Band
                 AND close > EMA200  <-- KEY GUARD: only DCA in an uptrend, never
                 average down into a sustained bear market (falling-knife protection).
  Scale-in:      add a tranche each time the trade is a further step underwater,
                 capped at max_entry_position_adjustment safety orders.
  Exit:          RSI recovers / price reverts to the BB mid, or ROI/stoploss.

Showcases: adjust_trade_position() DCA. ⚠ Mean reversion + DCA is risky; the EMA200
guard + a hard cap on safety orders + a wide stop bound the downside. Educational only.
"""
from datetime import datetime
from typing import Optional

from pandas import DataFrame

from freqtrade.strategy import IStrategy, IntParameter
from freqtrade.persistence import Trade

import talib.abstract as ta
from technical import qtpylib


class DcaMeanReversionStrategy(IStrategy):
    INTERFACE_VERSION = 3
    can_short: bool = False

    timeframe = "4h"

    minimal_roi = {"0": 0.05, "480": 0.025, "1440": 0.0}
    stoploss = -0.18  # wide: we average down, so the per-entry stop must be loose

    trailing_stop = False
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False

    # DCA settings
    position_adjustment_enable = True
    max_entry_position_adjustment = 3      # up to 3 safety orders after the initial buy
    max_dca_multiplier = 4                 # total exposure cap = 1 + 3 tranches

    startup_candle_count: int = 200

    rsi_entry = IntParameter(20, 35, default=30, space="buy", optimize=True)
    rsi_exit = IntParameter(50, 65, default=58, space="sell", optimize=True)

    order_types = {
        "entry": "limit", "exit": "limit",
        "stoploss": "market", "stoploss_on_exchange": False,
    }
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        bb = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe["bb_lower"] = bb["lower"]
        dataframe["bb_mid"] = bb["mid"]
        dataframe["ema200"] = ta.EMA(dataframe, timeperiod=200)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["rsi"] < self.rsi_entry.value)
                & (dataframe["close"] < dataframe["bb_lower"])
                & (dataframe["close"] > dataframe["ema200"])   # uptrend-only DCA guard
                & (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                ((dataframe["rsi"] > self.rsi_exit.value) | (dataframe["close"] >= dataframe["bb_mid"]))
                & (dataframe["volume"] > 0)
            ),
            "exit_long",
        ] = 1
        return dataframe

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str, **kwargs) -> float:
        # Reserve room for the safety orders: open with a fraction of the budget.
        return proposed_stake / self.max_dca_multiplier

    def adjust_trade_position(self, trade: Trade, current_time: datetime, current_rate: float,
                              current_profit: float, min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> Optional[float]:
        count = trade.nr_of_successful_entries
        # Add the next safety order only once we're a further step underwater.
        if current_profit > -0.035 * count:
            return None
        if count > self.max_entry_position_adjustment:
            return None
        # Add a tranche the size of the first entry.
        try:
            first_stake = trade.select_filled_orders(trade.entry_side)[0].cost
        except Exception:
            return None
        stake = first_stake
        if min_stake is not None:
            stake = max(stake, min_stake)
        return min(stake, max_stake) if max_stake else stake
