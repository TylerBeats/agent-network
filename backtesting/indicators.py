"""
Technical indicator functions — all operate on numpy arrays.

Each function returns an array of the same length as the input,
with NaN values filling the warmup period where the indicator
cannot yet be computed.
"""
from __future__ import annotations

import numpy as np


def sma(prices: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average."""
    result = np.full(len(prices), np.nan)
    for i in range(period - 1, len(prices)):
        result[i] = np.mean(prices[i - period + 1 : i + 1])
    return result


def ema(prices: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average — standard multiplier 2/(period+1)."""
    result = np.full(len(prices), np.nan)
    if len(prices) < period:
        return result
    k = 2.0 / (period + 1)
    # Seed with SMA of first `period` values
    result[period - 1] = np.mean(prices[:period])
    for i in range(period, len(prices)):
        result[i] = prices[i] * k + result[i - 1] * (1 - k)
    return result


def rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Relative Strength Index using Wilder smoothing.
    RSI = 100 - 100 / (1 + avg_gain / avg_loss)
    """
    result = np.full(len(prices), np.nan)
    if len(prices) < period + 1:
        return result

    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Seed with simple average of first period
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, len(prices) - 1):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rs = avg_gain / avg_loss if avg_loss != 0 else float("inf")
        result[i + 1] = 100.0 - 100.0 / (1.0 + rs)

    # Fill the first valid value (at index `period`)
    rs0 = avg_gain / avg_loss if avg_loss != 0 else float("inf")
    result[period] = 100.0 - 100.0 / (1.0 + rs0)

    return result


def macd(
    prices: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    MACD indicator.
    Returns (macd_line, signal_line, histogram).
    """
    ema_fast = ema(prices, fast)
    ema_slow = ema(prices, slow)
    macd_line = ema_fast - ema_slow

    # Signal is EMA of macd_line (skip NaN warmup)
    signal_line = np.full(len(prices), np.nan)
    first_valid = slow - 1  # first index where macd_line is valid
    if len(prices) > first_valid + signal_period:
        valid_macd = macd_line[first_valid:]
        sig = ema(valid_macd, signal_period)
        signal_line[first_valid:] = sig

    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(
    prices: np.ndarray,
    period: int = 20,
    std_dev: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bollinger Bands.
    Returns (upper, middle, lower).
    """
    middle = sma(prices, period)
    upper = np.full(len(prices), np.nan)
    lower = np.full(len(prices), np.nan)
    for i in range(period - 1, len(prices)):
        std = np.std(prices[i - period + 1 : i + 1], ddof=0)
        upper[i] = middle[i] + std_dev * std
        lower[i] = middle[i] - std_dev * std
    return upper, middle, lower


def atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """
    Average True Range using Wilder smoothing.
    True Range = max(high-low, |high-prev_close|, |low-prev_close|)
    """
    result = np.full(len(close), np.nan)
    if len(close) < 2:
        return result

    tr = np.full(len(close), np.nan)
    tr[0] = high[0] - low[0]
    for i in range(1, len(close)):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )

    if len(close) < period:
        return result

    # Seed with simple average of first `period` TR values
    result[period - 1] = np.mean(tr[:period])
    for i in range(period, len(close)):
        result[i] = (result[i - 1] * (period - 1) + tr[i]) / period

    return result


def stochastic(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    k_period: int = 14,
    d_period: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Stochastic Oscillator.
    %K = (close - lowest_low) / (highest_high - lowest_low) * 100
    %D = SMA(%K, d_period)
    Returns (k, d).
    """
    k = np.full(len(close), np.nan)
    for i in range(k_period - 1, len(close)):
        hh = np.max(high[i - k_period + 1 : i + 1])
        ll = np.min(low[i - k_period + 1 : i + 1])
        denom = hh - ll
        k[i] = (close[i] - ll) / denom * 100 if denom != 0 else 50.0

    d = sma(k, d_period)
    return k, d


def volume_sma(volume: np.ndarray, period: int = 20) -> np.ndarray:
    """Simple Moving Average of volume."""
    return sma(volume, period)


def wma(prices: np.ndarray, period: int) -> np.ndarray:
    """Weighted Moving Average — linearly weighted (most recent bar = highest weight)."""
    result = np.full(len(prices), np.nan)
    weights = np.arange(1, period + 1, dtype=float)
    weight_sum = weights.sum()
    for i in range(period - 1, len(prices)):
        result[i] = np.dot(prices[i - period + 1 : i + 1], weights) / weight_sum
    return result


def adx(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Average Directional Index using Wilder smoothing.
    Returns (adx_line, plus_di, minus_di).
    ADX > 25 indicates a trending market.
    """
    n = len(close)
    result_adx = np.full(n, np.nan)
    result_pdi = np.full(n, np.nan)
    result_mdi = np.full(n, np.nan)

    if n < period + 1:
        return result_adx, result_pdi, result_mdi

    tr      = np.full(n, 0.0)
    dm_plus = np.full(n, 0.0)
    dm_minus = np.full(n, 0.0)

    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        up   = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        dm_plus[i]  = up   if up > down and up > 0   else 0.0
        dm_minus[i] = down if down > up and down > 0 else 0.0

    # Wilder smooth TR, +DM, -DM
    atr_s = np.full(n, 0.0)
    pdi_s = np.full(n, 0.0)
    mdi_s = np.full(n, 0.0)

    atr_s[period] = np.sum(tr[1 : period + 1])
    pdi_s[period] = np.sum(dm_plus[1 : period + 1])
    mdi_s[period] = np.sum(dm_minus[1 : period + 1])

    for i in range(period + 1, n):
        atr_s[i] = atr_s[i - 1] - atr_s[i - 1] / period + tr[i]
        pdi_s[i] = pdi_s[i - 1] - pdi_s[i - 1] / period + dm_plus[i]
        mdi_s[i] = mdi_s[i - 1] - mdi_s[i - 1] / period + dm_minus[i]

    # DX values from period onwards
    dx_arr = np.full(n, np.nan)
    for i in range(period, n):
        if atr_s[i] > 0:
            pdi = 100.0 * pdi_s[i] / atr_s[i]
            mdi = 100.0 * mdi_s[i] / atr_s[i]
            result_pdi[i] = pdi
            result_mdi[i] = mdi
            denom = pdi + mdi
            dx_arr[i] = abs(pdi - mdi) / denom * 100 if denom > 0 else 0.0

    # Seed ADX at 2*period with mean of DX[period:2*period], then Wilder smooth
    if n > 2 * period:
        valid_dx = [dx_arr[i] for i in range(period, 2 * period) if not np.isnan(dx_arr[i])]
        if valid_dx:
            result_adx[2 * period - 1] = np.mean(valid_dx)
            for i in range(2 * period, n):
                if not np.isnan(dx_arr[i]):
                    result_adx[i] = (result_adx[i - 1] * (period - 1) + dx_arr[i]) / period

    return result_adx, result_pdi, result_mdi


def cci(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 20,
) -> np.ndarray:
    """
    Commodity Channel Index.
    CCI = (Typical Price - SMA(Typical Price)) / (0.015 * Mean Absolute Deviation)
    Overbought > +100, oversold < -100.
    """
    result = np.full(len(close), np.nan)
    typical = (high + low + close) / 3.0
    for i in range(period - 1, len(close)):
        window   = typical[i - period + 1 : i + 1]
        mean_tp  = np.mean(window)
        mean_dev = np.mean(np.abs(window - mean_tp))
        result[i] = (typical[i] - mean_tp) / (0.015 * mean_dev) if mean_dev != 0 else 0.0
    return result


def williams_r(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """
    Williams %R — oscillator ranging from -100 to 0.
    Oversold < -80, overbought > -20.
    """
    result = np.full(len(close), np.nan)
    for i in range(period - 1, len(close)):
        hh = np.max(high[i - period + 1 : i + 1])
        ll = np.min(low[i - period + 1 : i + 1])
        denom = hh - ll
        result[i] = (hh - close[i]) / denom * -100 if denom != 0 else -50.0
    return result


def roc(prices: np.ndarray, period: int = 12) -> np.ndarray:
    """
    Rate of Change — percentage change over `period` bars.
    Positive ROC = upward momentum; negative = downward.
    """
    result = np.full(len(prices), np.nan)
    for i in range(period, len(prices)):
        prev = prices[i - period]
        result[i] = (prices[i] - prev) / prev * 100 if prev != 0 else 0.0
    return result


def keltner_channels(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    ema_period: int = 20,
    atr_period: int = 10,
    multiplier: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Keltner Channels.
    middle = EMA(close, ema_period)
    upper  = middle + multiplier * ATR(atr_period)
    lower  = middle - multiplier * ATR(atr_period)
    Returns (upper, middle, lower).
    """
    middle  = ema(close, ema_period)
    atr_val = atr(high, low, close, atr_period)
    upper   = middle + multiplier * atr_val
    lower   = middle - multiplier * atr_val
    return upper, middle, lower


def donchian_channels(
    high: np.ndarray,
    low: np.ndarray,
    period: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Donchian Channels.
    upper  = highest high over `period` bars
    lower  = lowest low over `period` bars
    middle = (upper + lower) / 2
    Returns (upper, middle, lower).
    """
    upper = np.full(len(high), np.nan)
    lower = np.full(len(low), np.nan)
    for i in range(period - 1, len(high)):
        upper[i] = np.max(high[i - period + 1 : i + 1])
        lower[i] = np.min(low[i - period + 1 : i + 1])
    middle = (upper + lower) / 2.0
    return upper, middle, lower


def obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """
    On-Balance Volume — cumulative volume indicator.
    OBV rises when close > prev close; falls when close < prev close.
    """
    result = np.full(len(close), np.nan)
    if len(close) == 0:
        return result
    result[0] = volume[0]
    for i in range(1, len(close)):
        if close[i] > close[i - 1]:
            result[i] = result[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            result[i] = result[i - 1] - volume[i]
        else:
            result[i] = result[i - 1]
    return result
