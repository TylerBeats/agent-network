"""
Hard elimination filters.

Any strategy failing even one of these rules is immediately disqualified,
regardless of its scores on other dimensions.
"""
from __future__ import annotations

from backtesting.models import BacktestResult, FilterResult, MetricsResult, MonteCarloResult


def _streak_limit(bucket_seconds: int) -> int:
    """
    Scale the maximum allowed consecutive-loss streak by bar size.

    Longer bars naturally produce longer losing streaks because each bar
    represents more calendar time and trend-following strategies can sit
    in drawdown for many bars during a regime transition.

      hourly  (<=3600s)  -> 8  losses
      4-hour  (<=14400s) -> 10 losses
      daily   (<=86400s) -> 15 losses
      weekly+ (>86400s)  -> 10 losses  (fewer bars total; stricter is fine)
    """
    if bucket_seconds <= 3_600:
        return 8
    if bucket_seconds <= 14_400:
        return 10
    if bucket_seconds <= 86_400:
        return 15
    return 10


def apply_filters(
    metrics: MetricsResult,
    backtest: BacktestResult,
    mc_result: MonteCarloResult | None = None,
    bucket_seconds: int = 14_400,
) -> FilterResult:
    """
    Apply all hard elimination filters.
    Returns the first failure encountered, or passed=True if all filters clear.

    Args:
        metrics:        Computed trade metrics for the strategy.
        backtest:       Backtest result (risk level, OOS metrics, flags).
        mc_result:      Monte Carlo drawdown result (optional). When provided,
                        adds two additional MC-based filters.
        bucket_seconds: Candle width -- used to scale the consecutive-loss limit.
    """
    streak_limit = _streak_limit(bucket_seconds)

    filters: list[tuple[bool, str]] = [
        (
            metrics.max_drawdown_pct > 0.20,
            f"Max drawdown {metrics.max_drawdown_pct:.1%} exceeds 20%",
        ),
        (
            metrics.avg_r_multiple <= 0.0,
            f"Negative expected value (EV = {metrics.avg_r_multiple:.2f}R per trade)",
        ),
        (
            metrics.win_rate < 0.35,
            f"Win rate {metrics.win_rate:.1%} below minimum 35%",
        ),
        (
            metrics.trade_count < 30,
            f"Only {metrics.trade_count} trades -- insufficient sample size (minimum 30)",
        ),
        (
            metrics.worst_trade_pct > 0.05,
            f"Single trade loss {metrics.worst_trade_pct:.1%} exceeds 5% of account equity",
        ),
        (
            metrics.max_consecutive_losses > streak_limit,
            f"Consecutive losing streak of {metrics.max_consecutive_losses} exceeds limit of {streak_limit}",
        ),
        (
            metrics.sharpe < 0.5,
            f"Sharpe Ratio {metrics.sharpe:.2f} below minimum 0.5",
        ),
        (
            backtest.only_works_at_max_risk,
            "Strategy only produces positive risk-adjusted return at the 2% risk level",
        ),
    ]

    for failed, reason in filters:
        if failed:
            return FilterResult(passed=False, failure_reason=reason)

    # -- Monte Carlo filters (applied only when MC data is available) ----------
    if mc_result is not None:
        if mc_result.risk_class == "extreme":
            return FilterResult(
                passed=False,
                failure_reason=(
                    f"Monte Carlo 95th pct drawdown {mc_result.p95_drawdown:.1%} "
                    f"is Extreme risk (>12%) -- unacceptable real-world risk profile"
                ),
            )
        if not mc_result.prop_firm_8pct:
            return FilterResult(
                passed=False,
                failure_reason=(
                    f"Fails 8% prop firm rule: MC 95th pct drawdown = {mc_result.p95_drawdown:.1%} "
                    f"(must be below 8%)"
                ),
            )

    return FilterResult(passed=True, failure_reason=None)
