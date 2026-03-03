"""
Yahoo Finance data source — free OHLCV for equities, ETFs, commodities, and indices.

No API key required. Uses the yfinance library.

Supported tickers (examples):
  GLD        — SPDR Gold Trust ETF (~$200 USD/share, highly liquid)
  SPY        — SPDR S&P 500 ETF (most liquid ETF in existence)
  QQQ        — Invesco Nasdaq-100 ETF
  ^GSPC      — S&P 500 index (price-only, no real volume)
  ^NDX       — Nasdaq-100 index (price-only)
  BTC-USD    — Bitcoin/USD via Yahoo
  AAPL, MSFT — Individual US equities

Volume semantics differ from crypto subgraphs:
  - ETFs/equities: shares traded (relative comparisons are valid for VOLUME_ABOVE_SMA)
  - Futures (GC=F): contracts traded
  - Indices (^GSPC): synthetic volume — prefer ETF equivalents (SPY)

Close prices are split- and dividend-adjusted by default (auto_adjust=True),
which makes long-run returns comparable across the full history.
"""
from __future__ import annotations

import logging

import yfinance as yf

from data.candles import Candle
from data.quality import DataQualityResult, analyse_data_quality

logger = logging.getLogger(__name__)

# Maximum history periods per yfinance interval
_INTERVAL_MAP: list[tuple[int, str, str]] = [
    # (max_bucket_seconds, yf_interval, yf_period)
    (3_600,  "1h",  "730d"),   # ≤1h  → hourly, 2 years max
    (14_400, "1h",  "730d"),   # ≤4h  → hourly (then aggregated), 2 years max
    (86_400, "1d",  "max"),    # ≤1d  → daily, full history
    (999_999,"1wk", "max"),    # >1d  → weekly, full history
]


def fetch(
    symbol: str,
    bucket_seconds: int = 86400,
) -> tuple[list[Candle], DataQualityResult]:
    """
    Download OHLCV data from Yahoo Finance and return as internal Candle objects.

    Args:
        symbol:         Yahoo Finance ticker string (e.g. "SPY", "GLD", "QQQ").
        bucket_seconds: Desired candle width in seconds. Hourly bars are
                        aggregated if bucket_seconds > 3600. Daily is default
                        and recommended for traditional markets.

    Returns:
        (candles, DataQualityResult) — candles sorted oldest → newest.

    Raises:
        RuntimeError if the symbol is not found or returns no data.
    """
    yf_interval, yf_period = _resolve_interval(bucket_seconds)
    logger.info(
        "yahoo.fetch: symbol=%s interval=%s period=%s bucket=%ds",
        symbol, yf_interval, yf_period, bucket_seconds,
    )

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=yf_period, interval=yf_interval, auto_adjust=True)
    except Exception as exc:
        raise RuntimeError(f"Yahoo Finance fetch failed for '{symbol}': {exc}") from exc

    if df is None or df.empty:
        raise RuntimeError(
            f"Yahoo Finance returned no data for '{symbol}'. "
            "Check the ticker symbol is correct (e.g. 'SPY', 'GLD', 'QQQ')."
        )

    # Aggregate sub-daily bars into the requested bucket width
    if bucket_seconds > 3600 and yf_interval == "1h":
        df = _aggregate_hours(df, bucket_seconds // 3600)

    candles: list[Candle] = []
    for ts_raw, row in df.iterrows():
        ts_ms = _to_ms(ts_raw)
        o = float(row.get("Open",  0) or 0)
        h = float(row.get("High",  0) or 0)
        l = float(row.get("Low",   0) or 0)
        c = float(row.get("Close", 0) or 0)
        v = float(row.get("Volume", 0) or 0)

        if o <= 0 or h <= 0 or l <= 0 or c <= 0:
            continue

        candles.append(Candle(
            ts=ts_ms,
            open=o,
            high=h,
            low=l,
            close=c,
            volume=v,
            swap_count=0,   # no individual swap records from Yahoo Finance
        ))

    if not candles:
        raise RuntimeError(
            f"No valid candles built from Yahoo Finance data for '{symbol}'. "
            "All rows had zero or invalid prices."
        )

    candles.sort(key=lambda c: c.ts)
    logger.info(
        "yahoo.fetch: %d candles for %s  [%s → %s]",
        len(candles), symbol,
        _fmt_ts(candles[0].ts), _fmt_ts(candles[-1].ts),
    )

    quality = analyse_data_quality(candles, source="yahoo")
    return candles, quality


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_interval(bucket_seconds: int) -> tuple[str, str]:
    for max_bucket, interval, period in _INTERVAL_MAP:
        if bucket_seconds <= max_bucket:
            return interval, period
    return "1wk", "max"


def _to_ms(ts_raw) -> int:
    """Convert a pandas Timestamp (or similar) to Unix milliseconds."""
    try:
        return int(ts_raw.timestamp() * 1000)
    except AttributeError:
        # Fallback for integer nanosecond timestamps
        return int(ts_raw) // 1_000_000


def _fmt_ts(ts_ms: int) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def _aggregate_hours(df, hours_per_bucket: int):
    """
    Resample an hourly DataFrame into N-hour OHLCV buckets.
    Used to produce 4h candles from 1h yfinance data.
    """
    rule = f"{hours_per_bucket}h"
    agg = {
        "Open":   "first",
        "High":   "max",
        "Low":    "min",
        "Close":  "last",
        "Volume": "sum",
    }
    return df.resample(rule).agg(agg).dropna(subset=["Open", "Close"])
