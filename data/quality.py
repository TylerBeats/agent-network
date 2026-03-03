from __future__ import annotations

import logging
from dataclasses import dataclass, field

from data.candles import Candle

logger = logging.getLogger(__name__)

# Thresholds — match the JS backtester exactly
SPIKE_RATIO = 5     # high/low ratio within a single candle
JUMP_RATIO = 10     # close-to-close ratio between consecutive candles
SINGLE_SWAP_PCT = 30  # % of single-swap candles that triggers a warning


@dataclass
class DataQualityResult:
    passed: bool
    warnings: list[str] = field(default_factory=list)
    confidence: str = "high"  # "high" | "medium" | "low"


def analyse_data_quality(candles: list[Candle], source: str) -> DataQualityResult:
    """
    Run the four data quality checks ported from the JS backtester.

    confidence:
      "high"   — no warnings
      "medium" — 1 warning
      "low"    — 2 or more warnings
    """
    if not candles or len(candles) < 2:
        return DataQualityResult(passed=True, warnings=[], confidence="high")

    warnings: list[str] = []

    # ── Check 1: intra-candle high/low ratio ─────────────────────────────────
    # A spike ratio >= SPIKE_RATIO within a single candle indicates thin-liquidity
    # price distortion (e.g. a V3 out-of-range dust swap moving through empty ticks).
    noisy = [c for c in candles if c.low > 0 and (c.high / c.low) >= SPIKE_RATIO]
    if noisy:
        noisy_pct = len(noisy) / len(candles) * 100
        worst = max(c.high / c.low for c in noisy)
        warnings.append(
            f"{len(noisy)} candle(s) ({noisy_pct:.1f}%) have a high/low ratio "
            f">= {SPIKE_RATIO}x (worst: {worst:.0f}x) — likely thin-liquidity price "
            f"spikes. VWAP suppresses these in the close price but high/low extremes "
            f"are preserved for trigger detection."
        )

    # ── Check 2: candle-to-candle close jumps ────────────────────────────────
    # A jump >= JUMP_RATIO between consecutive VWAP closes signals discontinuities
    # that can produce false ATH readings and premature indicator triggers.
    big_jumps = 0
    max_jump = 0.0
    for i in range(1, len(candles)):
        prev, curr = candles[i - 1].close, candles[i].close
        if prev > 0 and curr > 0:
            ratio = max(curr / prev, prev / curr)
            if ratio >= JUMP_RATIO:
                big_jumps += 1
                max_jump = max(max_jump, ratio)
    if big_jumps:
        warnings.append(
            f"{big_jumps} candle-to-candle VWAP jump(s) >= {JUMP_RATIO}x "
            f"(largest: {max_jump:.0f}x) — close price has discontinuities that may "
            f"create false ATH readings and premature Fibonacci triggers."
        )

    # ── Check 3: single-swap candle density (subgraph only) ──────────────────
    # When > SINGLE_SWAP_PCT% of candles contain only one swap, VWAP provides no
    # smoothing — each "weighted average" is just a single raw price.
    if source == "subgraph":
        single = sum(1 for c in candles if c.swap_count == 1)
        single_pct = single / len(candles) * 100
        if single_pct > SINGLE_SWAP_PCT:
            warnings.append(
                f"{single_pct:.0f}% of candles contain only 1 swap — VWAP provides "
                f"no smoothing for these. Consider a larger bucket size to aggregate "
                f"more swaps per candle."
            )

    # ── Check 4: GeckoTerminal notice ────────────────────────────────────────
    if source == "gecko":
        warnings.append(
            "GeckoTerminal pre-aggregates candles — VWAP cannot be applied (no "
            "per-swap data). High/low extremes may still include thin-liquidity spikes."
        )

    passed = len(warnings) == 0
    if len(warnings) == 0:
        confidence = "high"
    elif len(warnings) == 1:
        confidence = "medium"
    else:
        confidence = "low"

    return DataQualityResult(passed=passed, warnings=warnings, confidence=confidence)


def trim_low_liquidity_candles(
    candles: list[Candle],
    min_daily_volume_usd: float = 50_000.0,
    bucket_seconds: int = 14400,
) -> list[Candle]:
    """
    Genesis liquidity filter — removes leading candles from before the token
    reached a minimum sustainable liquidity level.

    Scans forward until a rolling 30-day window of candles has an average daily
    volume >= min_daily_volume_usd.  Returns the candle list starting at the
    first bar of the first qualifying window.

    If no window ever qualifies (e.g. the token never reached the threshold),
    returns the most recent 30 days of candles so backtesting still has data.

    Args:
        candles:              Full candle list sorted oldest → newest.
        min_daily_volume_usd: Minimum required average daily volume in USD.
        bucket_seconds:       Candle width in seconds (determines candles/day).
    """
    if not candles:
        return candles

    candles_per_day = max(1, 86400 // bucket_seconds)
    lookback = candles_per_day * 30  # 30-day rolling window

    if len(candles) <= lookback:
        return candles

    for i in range(len(candles) - lookback):
        window = candles[i : i + lookback]
        daily_avg = sum(c.volume for c in window) / 30.0
        if daily_avg >= min_daily_volume_usd:
            trimmed = len(candles) - len(candles[i:])
            if trimmed > 0:
                logger.info(
                    "Genesis filter: trimmed %d early low-liquidity candles "
                    "(avg daily vol was below $%.0f)",
                    trimmed,
                    min_daily_volume_usd,
                )
            return candles[i:]

    # No window met the threshold — return the last 30 days (best available)
    logger.warning(
        "Genesis filter: no window met $%.0f avg daily volume; "
        "returning last %d candles",
        min_daily_volume_usd,
        lookback,
    )
    return candles[-lookback:]
