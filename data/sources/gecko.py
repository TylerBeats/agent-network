"""
GeckoTerminal OHLCV API fallback.

Used for Solana and any chain without a subgraph.
Provides pre-aggregated candle data (~3 months of hourly history max).
No API key required.
Note: VWAP cannot be applied — swap_count is 0 for all candles.
"""
from __future__ import annotations

import logging
import time

import requests

from data.candles import Candle
from data.quality import DataQualityResult, analyse_data_quality

logger = logging.getLogger(__name__)

GECKO_BASE = "https://api.geckoterminal.com/api/v2"
PAGE_SIZE = 1000
DELAY_BETWEEN_PAGES = 0.5  # seconds — GeckoTerminal enforces stricter rate limits


def fetch(
    network: str,
    pair_address: str,
    timeframe: str = "hour",
    limit: int = 1000,
) -> tuple[list[Candle], DataQualityResult]:
    """
    Fetch OHLCV candles from GeckoTerminal.

    Args:
        network:      GeckoTerminal network slug (e.g. "solana", "pulsechain", "eth").
        pair_address: Pool/pair contract address.
        timeframe:    "minute", "hour", or "day".
        limit:        Total number of candles to request (max ~3 months for hourly).

    Returns:
        (candles, quality_result)
    """
    aggregate = 15 if timeframe == "minute" else 1
    all_raw: list[list] = []
    before_timestamp: int | None = None
    total_pages = max(1, -(-limit // PAGE_SIZE))  # ceiling division

    for page in range(1, total_pages + 1):
        remaining = limit - len(all_raw)
        fetch_count = min(remaining, PAGE_SIZE)

        url = (
            f"{GECKO_BASE}/networks/{network}/pools/{pair_address}"
            f"/ohlcv/{timeframe}?limit={fetch_count}&aggregate={aggregate}&currency=usd"
        )
        if before_timestamp is not None:
            url += f"&before_timestamp={before_timestamp}"

        logger.debug("GeckoTerminal page %d/%d — %d candles so far", page, total_pages, len(all_raw))

        resp = requests.get(
            url,
            headers={"Accept": "application/json"},
            timeout=30,
        )
        if not resp.ok:
            raise RuntimeError(f"GeckoTerminal HTTP {resp.status_code}: {resp.text[:200]}")

        raw = resp.json().get("data", {}).get("attributes", {}).get("ohlcv_list") or []
        if not raw:
            logger.info("GeckoTerminal ran out of data at page %d", page)
            break

        all_raw.extend(raw)
        before_timestamp = int(min(c[0] for c in raw))

        if len(raw) < fetch_count:
            break
        if page < total_pages:
            time.sleep(DELAY_BETWEEN_PAGES)

    if not all_raw:
        raise ValueError(
            f"No OHLCV data returned from GeckoTerminal for {network}/{pair_address}. "
            "Check the pair address and network slug."
        )

    # Deduplicate by timestamp and sort ascending
    seen: set[int] = set()
    candles: list[Candle] = []
    for row in sorted(all_raw, key=lambda c: c[0]):
        ts_ms = int(row[0]) * 1000
        if ts_ms in seen:
            continue
        seen.add(ts_ms)
        candles.append(Candle(
            ts=ts_ms,
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
            volume=float(row[5]),
            swap_count=0,  # GeckoTerminal does not expose per-swap data
        ))

    quality = analyse_data_quality(candles, "gecko")
    logger.info(
        "GeckoTerminal: %d candles — quality: %s (%d warning(s))",
        len(candles), quality.confidence, len(quality.warnings),
    )
    return candles, quality
