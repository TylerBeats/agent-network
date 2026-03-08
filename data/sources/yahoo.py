"""
Yahoo Finance data source — free OHLCV for equities, ETFs, commodities, and indices.

No API key required. Uses the yfinance library.

OFFLINE MODE: Reads from data/local_cache/{asset}_{timeframe}.csv if available.
Falls back to live yfinance download if local file is not found.

Asset → file mapping:
  spx500 / SPY  → spx500_1m.csv, spx500_5m.csv, ... spx500_1d.csv
  nas100 / QQQ  → nas100_1m.csv, nas100_5m.csv, ... nas100_1d.csv
  us30   / DIA  → us30_1m.csv,   us30_5m.csv,   ... us30_1d.csv
  xauusd / GLD  → xauusd_1m.csv, xauusd_5m.csv, ... xauusd_1d.csv

Run download_market_data.py once while online to populate the cache.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from data.candles import Candle
from data.quality import DataQualityResult, analyse_data_quality

logger = logging.getLogger(__name__)

# ── Local cache directory ─────────────────────────────────────────────────
_CACHE_DIR = Path(__file__).parent.parent / "local_cache"

# ── Symbol → asset name mapping (for cache file lookup) ──────────────────
_SYMBOL_TO_ASSET: dict[str, str] = {
    "SPY":   "spx500",
    "QQQ":   "nas100",
    "DIA":   "us30",
    "GLD":   "xauusd",
    # Index fallbacks
    "^GSPC": "spx500",
    "^NDX":  "nas100",
    "^DJI":  "us30",
    "GC=F":  "xauusd",
}

# ── bucket_seconds → timeframe label ─────────────────────────────────────
_BUCKET_TO_LABEL: dict[int, str] = {
    60:    "1m",
    300:   "5m",
    900:   "15m",
    1800:  "30m",
    3600:  "1h",
    14400: "4h",
    86400: "1d",
}

# ── yfinance interval config (used for live fallback) ────────────────────
_INTERVAL_MAP: list[tuple[int, str, str]] = [
    (3_600,   "1h",  "730d"),
    (14_400,  "1h",  "730d"),
    (86_400,  "1d",  "max"),
    (999_999, "1wk", "max"),
]


def fetch(
    symbol: str,
    bucket_seconds: int = 86400,
) -> tuple[list[Candle], DataQualityResult]:
    """
    Return OHLCV candles for the given symbol and bucket size.
    Tries local cache first; falls back to live yfinance download.
    """
    # ── Try local cache ───────────────────────────────────────────────────
    candles = _try_load_local(symbol, bucket_seconds)
    if candles:
        logger.info(
            "yahoo.fetch [LOCAL]: %d candles for %s %ds  [%s → %s]",
            len(candles), symbol, bucket_seconds,
            _fmt_ts(candles[0].ts), _fmt_ts(candles[-1].ts),
        )
        quality = analyse_data_quality(candles, source="yahoo")
        return candles, quality

    # ── Fall back to live yfinance ────────────────────────────────────────
    logger.warning(
        "yahoo.fetch: no local cache for %s %ds — downloading live (requires internet)",
        symbol, bucket_seconds,
    )
    return _fetch_live(symbol, bucket_seconds)


def _try_load_local(symbol: str, bucket_seconds: int) -> list[Candle] | None:
    """
    Attempt to load candles from a local CSV file.
    Returns None if no suitable file found.
    """
    asset = _SYMBOL_TO_ASSET.get(symbol.upper())
    label = _BUCKET_TO_LABEL.get(bucket_seconds)

    if not asset or not label:
        # Unknown symbol or non-standard bucket — skip local cache
        logger.debug(
            "yahoo.fetch: no cache mapping for symbol=%s bucket=%ds",
            symbol, bucket_seconds,
        )
        return None

    cache_file = _CACHE_DIR / f"{asset}_{label}.csv"

    if not cache_file.exists():
        logger.info(
            "yahoo.fetch: local cache miss — %s not found", cache_file
        )
        return None

    try:
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)

        # Handle multi-level column headers from yfinance CSV exports
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        # Normalise column names to Title case
        df.columns = [c.strip().title() for c in df.columns]

        # Must have OHLCV
        required = {"Open", "High", "Low", "Close"}
        if not required.issubset(set(df.columns)):
            logger.warning(
                "yahoo.fetch: cache file %s missing columns %s — found %s",
                cache_file.name, required, list(df.columns),
            )
            return None

        # Drop invalid rows
        df = df[(df["Open"] > 0) & (df["High"] > 0) & (df["Low"] > 0) & (df["Close"] > 0)]

        if df.empty:
            logger.warning("yahoo.fetch: cache file %s has no valid rows", cache_file.name)
            return None

        candles: list[Candle] = []
        for ts_raw, row in df.iterrows():
            ts_ms = _to_ms(ts_raw)
            candles.append(Candle(
                ts=ts_ms,
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=float(row.get("Volume", 0) or 0),
                swap_count=0,
            ))

        candles.sort(key=lambda c: c.ts)
        logger.info(
            "yahoo.fetch [LOCAL]: loaded %d candles from %s",
            len(candles), cache_file.name,
        )
        return candles if candles else None

    except Exception as exc:
        logger.error(
            "yahoo.fetch: failed to read local cache %s — %s",
            cache_file, exc,
        )
        return None


def _fetch_live(
    symbol: str,
    bucket_seconds: int = 86400,
) -> tuple[list[Candle], DataQualityResult]:
    """Live yfinance download — original behaviour."""
    import yfinance as yf

    yf_interval, yf_period = _resolve_interval(bucket_seconds)
    logger.info(
        "yahoo.fetch [LIVE]: symbol=%s interval=%s period=%s bucket=%ds",
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
            "Check the ticker symbol and run download_market_data.py while online."
        )

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
        candles.append(Candle(ts=ts_ms, open=o, high=h, low=l, close=c, volume=v, swap_count=0))

    if not candles:
        raise RuntimeError(f"No valid candles built from Yahoo Finance data for '{symbol}'.")

    candles.sort(key=lambda c: c.ts)
    quality = analyse_data_quality(candles, source="yahoo")
    return candles, quality


# ── Also add nas100, us30, xauusd to chains.py ────────────────────────────
# Add these entries to CHAIN_CONFIG in data/chains.py if not already present:
#
# "nas100": {
#     "src": "yahoo", "label": "Nasdaq-100 (QQQ ETF)",
#     "symbol": "QQQ", "default_bucket": 86400,
# },
# "us30": {
#     "src": "yahoo", "label": "Dow Jones (DIA ETF)",
#     "symbol": "DIA", "default_bucket": 86400,
# },
# "xauusd": {
#     "src": "yahoo", "label": "Gold (GLD ETF)",
#     "symbol": "GLD", "default_bucket": 86400,
# },


# ── Helpers ───────────────────────────────────────────────────────────────

def _resolve_interval(bucket_seconds: int) -> tuple[str, str]:
    for max_bucket, interval, period in _INTERVAL_MAP:
        if bucket_seconds <= max_bucket:
            return interval, period
    return "1wk", "max"


def _to_ms(ts_raw) -> int:
    try:
        return int(ts_raw.timestamp() * 1000)
    except AttributeError:
        return int(ts_raw) // 1_000_000


def _fmt_ts(ts_ms: int) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def _aggregate_hours(df, hours_per_bucket: int):
    rule = f"{hours_per_bucket}h"
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    return df.resample(rule).agg(agg).dropna(subset=["Open", "Close"])
