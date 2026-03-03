"""
Feedback report construction — pure data assembly.

The TradingCoachAgent calls build_feedback_report() to assemble structured
data from the monthly review and performance history, then uses the LLM to
generate the next_cycle_hypothesis narrative.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from performance.tracker import MonthlyReview, WeeklyReview


@dataclass
class FeedbackReport:
    cycle: int
    best_indicator_types: list[str]    # e.g. ["RSI", "MACD"] — from winning strategy history
    best_rr_ratio: float               # best average R achieved across history
    failing_regimes: list[str]         # regimes where current strategy underperforms
    next_cycle_hypothesis: str         # filled by LLM in TradingCoachAgent
    retired_patterns: list[dict]       # [{"primary_indicator_type": "RSI", "entry_trigger": "..."}]
    risk_adjustment_for_trader: float  # global position size multiplier (fallback)
    health_tiers: dict = field(default_factory=dict)              # {strategy_id: "Good"|"Marginal"|"Poor"}
    per_strategy_adjustments: dict = field(default_factory=dict)  # {strategy_id: float multiplier}
    compounding_modes: dict = field(default_factory=dict)         # {strategy_id: "none"|"monthly"|"per_trade"}


def build_feedback_report(
    monthly_review: MonthlyReview,
    history: list[dict],
    cycle: int,
    per_strategy_reviews: dict[str, WeeklyReview] | None = None,
    months_at_good_tier: dict[str, int] | None = None,
    per_strategy_compounding: dict[str, str] | None = None,
) -> FeedbackReport:
    """
    Assemble a FeedbackReport from quantitative data.

    Args:
        monthly_review:           Most recent MonthlyReview.
        history:                  List of past winner dicts (from backtester output).
        cycle:                    Current cycle number.
        per_strategy_reviews:     Optional dict of {strategy_id: WeeklyReview} for
                                  per-strategy health and risk adjustment.
        months_at_good_tier:      Optional dict of {strategy_id: int} tracking how many
                                  months each strategy has held the Good tier.
        per_strategy_compounding: Optional dict of {strategy_id: mode} for current
                                  compounding mode per strategy.

    Returns:
        FeedbackReport with next_cycle_hypothesis left as empty string
        (the TradingCoachAgent LLM fills it in).
    """
    per_strategy_reviews  = per_strategy_reviews or {}
    months_at_good_tier   = months_at_good_tier or {}
    per_strategy_compounding = per_strategy_compounding or {}

    best_indicator_types = _extract_best_indicators(history)
    best_rr_ratio        = _extract_best_rr(history)
    failing_regimes      = _find_failing_regimes(history)

    retired_patterns: list[dict] = []
    if monthly_review.retire and history:
        last = history[-1]
        schema = last.get("strategy_schema", {})
        retired_patterns.append({
            "primary_indicator_type": schema.get("primary_indicator", {}).get("type", ""),
            "entry_trigger":          schema.get("entry", {}).get("trigger", ""),
        })

    # Build per-strategy health tiers, risk multipliers, and compounding modes
    health_tiers: dict[str, str] = {}
    per_adjustments: dict[str, float] = {}
    new_compounding: dict[str, str] = {}

    for sid, review in per_strategy_reviews.items():
        tier = review.health_tier
        health_tiers[sid] = tier
        per_adjustments[sid] = _strategy_risk_multiplier(tier, monthly_review)
        months_good = months_at_good_tier.get(sid, 0)
        current_mode = per_strategy_compounding.get(sid, "none")
        new_compounding[sid] = _resolve_compounding_mode(sid, tier, months_good, current_mode)

    return FeedbackReport(
        cycle=cycle,
        best_indicator_types=best_indicator_types,
        best_rr_ratio=best_rr_ratio,
        failing_regimes=failing_regimes,
        next_cycle_hypothesis="",   # LLM fills this in TradingCoachAgent
        retired_patterns=retired_patterns,
        risk_adjustment_for_trader=monthly_review.risk_adjustment,
        health_tiers=health_tiers,
        per_strategy_adjustments=per_adjustments,
        compounding_modes=new_compounding,
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _strategy_risk_multiplier(tier: str, monthly: MonthlyReview) -> float:
    """
    Derive per-strategy position size multiplier from health tier.
    Poor strategies get an additional penalty regardless of monthly P&L.
    """
    if tier == "Poor":
        return 0.5
    if tier == "Marginal":
        return 0.75
    return monthly.risk_adjustment   # Good: use monthly P&L-based adjustment (0.5–1.25)


def _resolve_compounding_mode(
    strategy_id: str,  # noqa: ARG001 — reserved for future per-strategy logic
    tier: str,
    months_good: int,
    current_mode: str,
) -> str:
    """
    Determine the compounding mode for a strategy based on its health and tenure.

    Unlock rules:
    - No upgrades while degraded (Marginal or Poor).
    - After 6 months at Good tier: upgrade to "per_trade".
    - After 3 months at Good tier: upgrade from "none" to "monthly".
    """
    if tier != "Good":
        return current_mode   # never upgrade a degraded strategy
    if months_good >= 6:
        return "per_trade"
    if months_good >= 3 and current_mode == "none":
        return "monthly"
    return current_mode


def _extract_best_indicators(history: list[dict]) -> list[str]:
    """Find the primary indicator types from the top-scoring historical strategies."""
    seen: set[str] = set()
    result: list[str] = []
    # Iterate most recent first
    for entry in reversed(history):
        schema = entry.get("strategy_schema", {})
        ind_type = schema.get("primary_indicator", {}).get("type", "")
        if ind_type and ind_type.upper() not in seen:
            seen.add(ind_type.upper())
            result.append(ind_type.upper())
        if len(result) >= 3:
            break
    return result or ["RSI"]


def _extract_best_rr(history: list[dict]) -> float:
    """Return the best avg_r_multiple seen across all historical entries."""
    r_values = [
        float(entry.get("avg_r", 0.0))
        for entry in history
        if entry.get("avg_r") is not None
    ]
    return max(r_values) if r_values else 0.0


def _find_failing_regimes(history: list[dict]) -> list[str]:
    """
    Identify market regimes where the most recent strategy underperformed.
    A regime is 'failing' if its PnL was negative in the most recent cycle.
    """
    if not history:
        return []
    last = history[-1]
    failing = []
    regime_keys = {
        "bull":     "bull_pnl_pct",
        "bear":     "bear_pnl_pct",
        "sideways": "sideways_pnl_pct",
    }
    for regime, key in regime_keys.items():
        val = last.get(key)
        if val is not None and float(val) < 0:
            failing.append(regime)
    return failing
