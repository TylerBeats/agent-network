from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Candle:
    ts: int          # bucket start — Unix milliseconds
    open: float      # first swap's raw price in the bucket
    high: float      # highest raw swap price seen in the bucket
    low: float       # lowest raw swap price seen in the bucket
    close: float     # VWAP for subgraph data; last price for GeckoTerminal
    volume: float    # sum of amountUSD across all swaps in the bucket
    swap_count: int  # number of individual swaps; 0 for GeckoTerminal candles


def build_candles_from_swaps(
    swaps: list[dict],
    bucket_seconds: int,
    is_token0: bool,
    is_v2: bool,
) -> list[Candle]:
    """
    Build VWAP candles from a list of raw swap dicts.

    For PulseX v2 (is_v2=True):
        amountUSD counts both sides of the swap, so raw_price = (amtUSD / 2) / tokenVol

    For Uniswap v3 (is_v2=False):
        amountUSD is one side only, so raw_price = amtUSD / tokenVol
        amount0 / amount1 are signed — use absolute value for token volume.

    VWAP close = Σ(raw_price × amtUSD) / Σ(amtUSD)
    Large trades dominate; thin-liquidity outlier swaps are suppressed.
    """
    bucket_ms = bucket_seconds * 1000
    buckets: dict[int, dict] = {}

    for swap in swaps:
        ts = int(swap["timestamp"]) * 1000
        bucket_key = (ts // bucket_ms) * bucket_ms

        amt_usd = float(swap.get("amountUSD", 0))
        if not amt_usd or amt_usd <= 0:
            continue

        if is_v2:
            # PulseX v2: amount0In/amount0Out/amount1In/amount1Out are all non-negative
            if is_token0:
                token_vol = float(swap.get("amount0In", 0)) + float(swap.get("amount0Out", 0))
            else:
                token_vol = float(swap.get("amount1In", 0)) + float(swap.get("amount1Out", 0))
            # amountUSD counts both sides — divide by 2 to get one side's USD value
            raw_price = (amt_usd / 2) / token_vol if token_vol > 0 else 0
        else:
            # Uniswap v3: amount0/amount1 are signed
            if is_token0:
                token_vol = abs(float(swap.get("amount0", 0)))
            else:
                token_vol = abs(float(swap.get("amount1", 0)))
            # amountUSD is one side only
            raw_price = amt_usd / token_vol if token_vol > 0 else 0

        if not raw_price or not math.isfinite(raw_price) or raw_price <= 0:
            continue

        weight = amt_usd  # larger trades exert more influence on VWAP close

        if bucket_key not in buckets:
            buckets[bucket_key] = {
                "ts":              bucket_key,
                "open":            raw_price,
                "high":            raw_price,
                "low":             raw_price,
                "volume":          amt_usd,
                "vwap_numerator":  raw_price * weight,
                "vwap_denominator": weight,
                "swap_count":      1,
            }
        else:
            b = buckets[bucket_key]
            b["high"]             = max(b["high"], raw_price)
            b["low"]              = min(b["low"], raw_price)
            b["volume"]          += amt_usd
            b["vwap_numerator"]  += raw_price * weight
            b["vwap_denominator"] += weight
            b["swap_count"]      += 1

    candles = [
        Candle(
            ts=b["ts"],
            open=b["open"],
            high=b["high"],
            low=b["low"],
            close=b["vwap_numerator"] / b["vwap_denominator"],
            volume=b["volume"],
            swap_count=b["swap_count"],
        )
        for b in sorted(buckets.values(), key=lambda x: x["ts"])
    ]

    return candles
