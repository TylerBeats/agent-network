"""
Signal evaluation — maps strategy entry/exit configs to indicator checks.

Entry triggers fire when the named condition becomes True at bar i.
Confirmations are persistent conditions that must also be True.
"""
from __future__ import annotations

import math

import numpy as np

from backtesting.indicators import (
    adx,
    atr,
    bollinger_bands,
    cci,
    donchian_channels,
    ema,
    ichimoku,
    keltner_channels,
    macd,
    obv,
    roc,
    rsi,
    sma,
    stochastic,
    volume_sma,
    williams_r,
)
from backtesting.models import Strategy
from data.candles import Candle

# ── Direction helpers ────────────────────────────────────────────────────────

_SHORT_TRIGGERS: frozenset[str] = frozenset({
    "RSI_OVERBOUGHT", "PRICE_BELOW_EMA", "MACD_CROSS_BELOW",
    "BB_UPPER_TOUCH", "STOCH_OVERBOUGHT", "CCI_OVERBOUGHT",
    "WILLR_OVERBOUGHT", "ROC_CROSS_BELOW", "DC_LOWER_BREAK", "KB_UPPER_TOUCH",
    "KC_UPPER_TOUCH", "PRICE_BELOW_CLOUD",
})


def is_short_trigger(trigger_name: str) -> bool:
    """Return True if the trigger implies a short (sell) entry."""
    return trigger_name.upper() in _SHORT_TRIGGERS


# ── Entry trigger evaluators ─────────────────────────────────────────────────
# Each is called as fn(indicators_dict, bar_index) -> bool

ENTRY_TRIGGERS: dict[str, callable] = {
    "RSI_OVERSOLD":     lambda ind, i: (
        not math.isnan(ind["rsi"][i]) and not math.isnan(ind["rsi"][i - 1])
        and ind["rsi"][i] < 30 and ind["rsi"][i - 1] >= 30
    ),
    "RSI_OVERBOUGHT":   lambda ind, i: (
        not math.isnan(ind["rsi"][i]) and not math.isnan(ind["rsi"][i - 1])
        and ind["rsi"][i] > 70 and ind["rsi"][i - 1] <= 70
    ),
    "PRICE_ABOVE_EMA":  lambda ind, i: (
        not math.isnan(ind["ema_primary"][i]) and not math.isnan(ind["ema_primary"][i - 1])
        and ind["close"][i - 1] <= ind["ema_primary"][i - 1]
        and ind["close"][i] > ind["ema_primary"][i]
    ),
    "PRICE_BELOW_EMA":  lambda ind, i: (
        not math.isnan(ind["ema_primary"][i]) and not math.isnan(ind["ema_primary"][i - 1])
        and ind["close"][i - 1] >= ind["ema_primary"][i - 1]
        and ind["close"][i] < ind["ema_primary"][i]
    ),
    "MACD_CROSS_ABOVE": lambda ind, i: (
        not math.isnan(ind["macd_line"][i]) and not math.isnan(ind["macd_sig"][i])
        and not math.isnan(ind["macd_line"][i - 1]) and not math.isnan(ind["macd_sig"][i - 1])
        and ind["macd_line"][i] > ind["macd_sig"][i]
        and ind["macd_line"][i - 1] <= ind["macd_sig"][i - 1]
    ),
    "MACD_CROSS_BELOW": lambda ind, i: (
        not math.isnan(ind["macd_line"][i]) and not math.isnan(ind["macd_sig"][i])
        and not math.isnan(ind["macd_line"][i - 1]) and not math.isnan(ind["macd_sig"][i - 1])
        and ind["macd_line"][i] < ind["macd_sig"][i]
        and ind["macd_line"][i - 1] >= ind["macd_sig"][i - 1]
    ),
    "BB_LOWER_TOUCH":   lambda ind, i: (
        not math.isnan(ind["bb_lower"][i])
        and ind["close"][i] <= ind["bb_lower"][i]
    ),
    "BB_UPPER_TOUCH":   lambda ind, i: (
        not math.isnan(ind["bb_upper"][i])
        and ind["close"][i] >= ind["bb_upper"][i]
    ),
    "STOCH_OVERSOLD":   lambda ind, i: (
        not math.isnan(ind["stoch_k"][i])
        and ind["stoch_k"][i] < 20
    ),
    "STOCH_OVERBOUGHT": lambda ind, i: (
        not math.isnan(ind["stoch_k"][i])
        and ind["stoch_k"][i] > 80
    ),
    "CCI_OVERSOLD":     lambda ind, i: (
        not math.isnan(ind["cci"][i]) and not math.isnan(ind["cci"][i - 1])
        and ind["cci"][i] > -100 and ind["cci"][i - 1] <= -100
    ),
    "CCI_OVERBOUGHT":   lambda ind, i: (
        not math.isnan(ind["cci"][i]) and not math.isnan(ind["cci"][i - 1])
        and ind["cci"][i] < 100 and ind["cci"][i - 1] >= 100
    ),
    "WILLR_OVERSOLD":   lambda ind, i: (
        not math.isnan(ind["willr"][i]) and not math.isnan(ind["willr"][i - 1])
        and ind["willr"][i] > -80 and ind["willr"][i - 1] <= -80
    ),
    "WILLR_OVERBOUGHT": lambda ind, i: (
        not math.isnan(ind["willr"][i]) and not math.isnan(ind["willr"][i - 1])
        and ind["willr"][i] < -20 and ind["willr"][i - 1] >= -20
    ),
    "ROC_CROSS_ABOVE":  lambda ind, i: (
        not math.isnan(ind["roc"][i]) and not math.isnan(ind["roc"][i - 1])
        and ind["roc"][i] > 0 and ind["roc"][i - 1] <= 0
    ),
    "ROC_CROSS_BELOW":  lambda ind, i: (
        not math.isnan(ind["roc"][i]) and not math.isnan(ind["roc"][i - 1])
        and ind["roc"][i] < 0 and ind["roc"][i - 1] >= 0
    ),
    "DC_UPPER_BREAK":   lambda ind, i: (
        not math.isnan(ind["dc_upper"][i]) and not math.isnan(ind["dc_upper"][i - 1])
        and ind["close"][i] > ind["dc_upper"][i - 1]
    ),
    "DC_LOWER_BREAK":   lambda ind, i: (
        not math.isnan(ind["dc_lower"][i]) and not math.isnan(ind["dc_lower"][i - 1])
        and ind["close"][i] < ind["dc_lower"][i - 1]
    ),
    "KB_LOWER_TOUCH":   lambda ind, i: (
        not math.isnan(ind["kb_lower"][i])
        and ind["close"][i] <= ind["kb_lower"][i]
    ),
    "KB_UPPER_TOUCH":   lambda ind, i: (
        not math.isnan(ind["kb_upper"][i])
        and ind["close"][i] >= ind["kb_upper"][i]
    ),
    # Keltner Channel aliases (KC_ prefix = standard naming; same computation as KB_)
    "KC_LOWER_TOUCH":   lambda ind, i: (
        not math.isnan(ind["kb_lower"][i])
        and ind["close"][i] <= ind["kb_lower"][i]
    ),
    "KC_UPPER_TOUCH":   lambda ind, i: (
        not math.isnan(ind["kb_upper"][i])
        and ind["close"][i] >= ind["kb_upper"][i]
    ),
    # Ichimoku Cloud triggers — crossover only (not persistent, to avoid excessive signals)
    "PRICE_ABOVE_CLOUD": lambda ind, i: (
        not math.isnan(ind["cloud_upper"][i]) and not math.isnan(ind["cloud_upper"][i - 1])
        and ind["close"][i - 1] <= ind["cloud_upper"][i - 1]
        and ind["close"][i] > ind["cloud_upper"][i]
    ),
    "PRICE_BELOW_CLOUD": lambda ind, i: (
        not math.isnan(ind["cloud_lower"][i]) and not math.isnan(ind["cloud_lower"][i - 1])
        and ind["close"][i - 1] >= ind["cloud_lower"][i - 1]
        and ind["close"][i] < ind["cloud_lower"][i]
    ),
}

# ── Confirmation evaluators ───────────────────────────────────────────────────

CONFIRMATIONS: dict[str, callable] = {
    "PRICE_ABOVE_EMA":  lambda ind, i: (
        not math.isnan(ind["ema_confirmation"][i])
        and ind["close"][i] > ind["ema_confirmation"][i]
    ),
    "PRICE_BELOW_EMA":  lambda ind, i: (
        not math.isnan(ind["ema_confirmation"][i])
        and ind["close"][i] < ind["ema_confirmation"][i]
    ),
    "VOLUME_ABOVE_SMA": lambda ind, i: (
        not math.isnan(ind["volume_sma"][i])
        and ind["volume"][i] > ind["volume_sma"][i]
    ),
    "ATR_EXPANDING":    lambda ind, i: (
        not math.isnan(ind["atr"][i]) and not math.isnan(ind["atr"][i - 1])
        and ind["atr"][i] > ind["atr"][i - 1]
    ),
    "ADX_TRENDING":     lambda ind, i: (
        not math.isnan(ind["adx"][i])
        and ind["adx"][i] > 25
    ),
    "OBV_RISING":       lambda ind, i: (
        not math.isnan(ind["obv"][i]) and not math.isnan(ind["obv"][i - 1])
        and ind["obv"][i] > ind["obv"][i - 1]
    ),
    "NONE":             lambda ind, i: True,
}


# ── Indicator pre-computation ─────────────────────────────────────────────────

def compute_indicators(strategy: Strategy, candles: list[Candle]) -> dict:
    """
    Compute all indicators required by the strategy.
    Returns a dict of named numpy arrays aligned to the candle series.
    """
    n = len(candles)
    close  = np.array([c.close  for c in candles])
    high   = np.array([c.high   for c in candles])
    low    = np.array([c.low    for c in candles])
    volume = np.array([c.volume for c in candles])

    ind: dict[str, np.ndarray] = {
        "close":  close,
        "high":   high,
        "low":    low,
        "volume": volume,
        # 200-period EMA always computed for regime classification in engine
        "ema200": ema(close, 200),
    }

    # ── Primary indicator ──────────────────────────────────────────────────
    p_type   = strategy.primary_indicator["type"].upper()
    p_params = strategy.primary_indicator.get("params", {})

    if p_type == "RSI":
        ind["rsi"] = rsi(close, p_params.get("period", 14))
    elif p_type in ("EMA", "SMA", "WMA"):
        period = p_params.get("period", 20)
        if p_type == "EMA":
            from backtesting.indicators import ema as _ema_fn
            ind["ema_primary"] = _ema_fn(close, period)
        elif p_type == "WMA":
            from backtesting.indicators import wma as _wma_fn
            ind["ema_primary"] = _wma_fn(close, period)
        else:
            ind["ema_primary"] = sma(close, period)
    elif p_type == "MACD":
        ml, sig, hist = macd(
            close,
            p_params.get("fast", 12),
            p_params.get("slow", 26),
            p_params.get("signal", 9),
        )
        ind["macd_line"] = ml
        ind["macd_sig"]  = sig
        ind["macd_hist"] = hist
    elif p_type in ("BOLLINGER", "BB"):
        upper, middle, lower = bollinger_bands(
            close,
            p_params.get("period", 20),
            p_params.get("std_dev", 2.0),
        )
        ind["bb_upper"]  = upper
        ind["bb_middle"] = middle
        ind["bb_lower"]  = lower
    elif p_type == "STOCH":
        k, d = stochastic(
            high, low, close,
            p_params.get("k_period", 14),
            p_params.get("d_period", 3),
        )
        ind["stoch_k"] = k
        ind["stoch_d"] = d
    elif p_type == "ATR":
        ind["atr"] = atr(high, low, close, p_params.get("period", 14))
    elif p_type == "ADX":
        adx_line, pdi, mdi = adx(high, low, close, p_params.get("period", 14))
        ind["adx"]     = adx_line
        ind["adx_pdi"] = pdi
        ind["adx_mdi"] = mdi
    elif p_type == "CCI":
        ind["cci"] = cci(high, low, close, p_params.get("period", 20))
    elif p_type == "WILLR":
        ind["willr"] = williams_r(high, low, close, p_params.get("period", 14))
    elif p_type == "ROC":
        ind["roc"] = roc(close, p_params.get("period", 12))
    elif p_type == "KELTNER":
        upper, middle, lower = keltner_channels(
            high, low, close,
            p_params.get("ema_period", 20),
            p_params.get("atr_period", 10),
            p_params.get("multiplier", 2.0),
        )
        ind["kb_upper"]  = upper
        ind["kb_middle"] = middle
        ind["kb_lower"]  = lower
    elif p_type == "DONCHIAN":
        upper, middle, lower = donchian_channels(
            high, low,
            p_params.get("period", 20),
        )
        ind["dc_upper"]  = upper
        ind["dc_middle"] = middle
        ind["dc_lower"]  = lower
    elif p_type == "ICHIMOKU":
        cloud_upper, cloud_lower = ichimoku(
            high, low,
            p_params.get("tenkan_period", 9),
            p_params.get("kijun_period", 26),
            p_params.get("senkou_b_period", 52),
        )
        ind["cloud_upper"] = cloud_upper
        ind["cloud_lower"] = cloud_lower
    elif p_type == "OBV":
        ind["obv"] = obv(close, volume)
    else:
        # Fallback: compute RSI with default params
        ind["rsi"] = rsi(close, 14)

    # ── Confirmation indicator ─────────────────────────────────────────────
    c_type   = strategy.confirmation_indicator["type"].upper()
    c_params = strategy.confirmation_indicator.get("params", {})

    if c_type in ("EMA", "SMA", "WMA"):
        period = c_params.get("period", 50)
        if c_type == "WMA":
            from backtesting.indicators import wma as _wma_fn
            ind["ema_confirmation"] = _wma_fn(close, period)
        else:
            fn = ema if c_type == "EMA" else sma
            ind["ema_confirmation"] = fn(close, period)
    elif c_type == "VOLUME":
        ind["volume_sma"] = volume_sma(volume, c_params.get("period", 20))
    elif c_type == "ATR":
        if "atr" not in ind:
            ind["atr"] = atr(high, low, close, c_params.get("period", 14))
    elif c_type == "ADX":
        if "adx" not in ind:
            adx_line, pdi, mdi = adx(high, low, close, c_params.get("period", 14))
            ind["adx"]     = adx_line
            ind["adx_pdi"] = pdi
            ind["adx_mdi"] = mdi
    elif c_type == "OBV":
        if "obv" not in ind:
            ind["obv"] = obv(close, volume)
    else:
        # Default: volume SMA as confirmation
        ind["volume_sma"] = volume_sma(volume, 20)

    # Ensure ATR is always available (used for stop-loss calculations)
    if "atr" not in ind:
        ind["atr"] = atr(high, low, close, 14)

    # Ensure volume_sma is always available
    if "volume_sma" not in ind:
        ind["volume_sma"] = volume_sma(volume, 20)

    # Ensure ema_primary and ema_confirmation exist (guards for signal lambdas)
    if "ema_primary" not in ind:
        ind["ema_primary"] = np.full(n, np.nan)
    if "ema_confirmation" not in ind:
        ind["ema_confirmation"] = np.full(n, np.nan)
    if "rsi" not in ind:
        ind["rsi"] = np.full(n, np.nan)
    if "macd_line" not in ind:
        ind["macd_line"] = np.full(n, np.nan)
        ind["macd_sig"]  = np.full(n, np.nan)
    if "bb_upper" not in ind:
        ind["bb_upper"] = np.full(n, np.nan)
        ind["bb_lower"] = np.full(n, np.nan)
    if "stoch_k" not in ind:
        ind["stoch_k"] = np.full(n, np.nan)
    # New indicator guards
    if "cci" not in ind:
        ind["cci"] = np.full(n, np.nan)
    if "willr" not in ind:
        ind["willr"] = np.full(n, np.nan)
    if "roc" not in ind:
        ind["roc"] = np.full(n, np.nan)
    if "adx" not in ind:
        ind["adx"] = np.full(n, np.nan)
    if "obv" not in ind:
        ind["obv"] = np.full(n, np.nan)
    if "dc_upper" not in ind:
        ind["dc_upper"] = np.full(n, np.nan)
        ind["dc_lower"] = np.full(n, np.nan)
    if "kb_upper" not in ind:
        ind["kb_upper"] = np.full(n, np.nan)
        ind["kb_lower"] = np.full(n, np.nan)
    if "cloud_upper" not in ind:
        ind["cloud_upper"] = np.full(n, np.nan)
        ind["cloud_lower"] = np.full(n, np.nan)

    return ind


# ── Signal evaluation ────────────────────────────────────────────────────────

def check_entry(strategy: Strategy, indicators: dict, bar: int) -> bool:
    """Return True if both the entry trigger and confirmation fire at this bar."""
    if bar < 1:
        return False
    trigger_name = strategy.entry.get("trigger", "NONE")
    confirm_name = strategy.entry.get("filter", "NONE")

    trigger_fn = ENTRY_TRIGGERS.get(trigger_name)
    confirm_fn = CONFIRMATIONS.get(confirm_name, CONFIRMATIONS["NONE"])

    if trigger_fn is None:
        return False
    try:
        return trigger_fn(indicators, bar) and confirm_fn(indicators, bar)
    except (IndexError, ZeroDivisionError):
        return False


def calc_stop_price(
    strategy: Strategy,
    entry_price: float,
    indicators: dict,
    bar: int,
    short: bool | None = None,
) -> float:
    """Calculate stop-loss price from strategy exit config.
    For shorts: stop is ABOVE entry. For longs: stop is BELOW entry.
    `short` overrides trigger-based direction when bidirectional testing forces a direction.
    """
    sl = strategy.exit.get("stop_loss", {})
    sl_type  = sl.get("type", "atr_multiple")
    sl_value = float(sl.get("value", 2.0))
    if short is None:
        short = is_short_trigger(strategy.entry.get("trigger", ""))

    if sl_type == "atr_multiple":
        atr_val = indicators["atr"][bar]
        if math.isnan(atr_val) or atr_val <= 0:
            atr_val = entry_price * 0.02  # fallback: 2% of price
        return entry_price + sl_value * atr_val if short else entry_price - sl_value * atr_val
    else:  # fixed_pct
        return entry_price * (1.0 + sl_value / 100.0) if short else entry_price * (1.0 - sl_value / 100.0)


def calc_take_profit_price(
    strategy: Strategy,
    entry_price: float,
    stop_price: float,
    short: bool | None = None,
) -> float:
    """Calculate take-profit price from strategy exit config.
    For shorts: TP is BELOW entry. For longs: TP is ABOVE entry.
    `short` overrides trigger-based direction when bidirectional testing forces a direction.
    """
    tp = strategy.exit.get("take_profit", {})
    tp_type  = tp.get("type", "r_multiple")
    tp_value = float(tp.get("value", 2.0))
    if short is None:
        short = is_short_trigger(strategy.entry.get("trigger", ""))

    risk = abs(entry_price - stop_price)
    if tp_type == "r_multiple":
        return entry_price - risk * tp_value if short else entry_price + risk * tp_value
    else:  # fixed_pct
        return entry_price * (1.0 - tp_value / 100.0) if short else entry_price * (1.0 + tp_value / 100.0)
