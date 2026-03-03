"""
6-dimension weighted scoring model.

Each dimension is normalised to 0-100, then multiplied by its weight.
Weights reflect the EV-first, risk-first philosophy of the network.
"""
from __future__ import annotations

from backtesting.models import MetricsResult, MonteCarloResult, ScoreResult

WEIGHTS: dict[str, float] = {
    "ev_score":               0.25,  # Expected value per trade (avg R-multiple based)
    "drawdown_profile":       0.25,  # MC 95th pct DD if available, else historical DD
    "risk_adjusted_return":   0.20,  # Sharpe + Sortino
    "consistency":            0.15,  # Monthly return variance + regime stability
    "profit_factor_score":    0.10,  # Gross profit / gross loss
    "statistical_confidence": 0.05,  # Trade count relative to minimum sample size
}


def score_strategy(
    metrics: MetricsResult,
    strategy_id: str,
    mc_result: MonteCarloResult | None = None,
) -> ScoreResult:
    """
    Score a strategy across 6 dimensions.
    Returns a ScoreResult with individual dimension scores and the weighted total.

    Args:
        metrics:     Computed performance metrics from the backtest.
        strategy_id: Strategy identifier for the result.
        mc_result:   Monte Carlo result -- when provided, the drawdown dimension
                     uses the 95th percentile MC drawdown instead of historical max.
    """
    ev   = _score_ev(metrics)
    dd   = _score_drawdown_profile(metrics, mc_result)
    rar  = _score_risk_adjusted_return(metrics)
    cons = _score_consistency(metrics)
    pf   = _score_profit_factor(metrics)
    sc   = _score_statistical_confidence(metrics)

    total = (
        ev   * WEIGHTS["ev_score"]               +
        dd   * WEIGHTS["drawdown_profile"]        +
        rar  * WEIGHTS["risk_adjusted_return"]    +
        cons * WEIGHTS["consistency"]             +
        pf   * WEIGHTS["profit_factor_score"]     +
        sc   * WEIGHTS["statistical_confidence"]
    )

    return ScoreResult(
        strategy_id=strategy_id,
        total=round(total, 4),
        ev_score=round(ev, 4),
        drawdown_profile=round(dd, 4),
        risk_adjusted_return=round(rar, 4),
        consistency=round(cons, 4),
        profit_factor_score=round(pf, 4),
        statistical_confidence=round(sc, 4),
    )


# -- Dimension scoring functions -----------------------------------------------

def _score_ev(m: MetricsResult) -> float:
    """
    Score based on expected value per trade in R units.

    avg_r_multiple is mathematically equivalent to EV per trade:
      EV = (win_rate * avg_win_R) - (loss_rate * 1.0)
         = sum(r_multiples) / n_trades

    Minimum threshold: +0.20R (20% of risk per trade) to score above zero.
    Scale: 0.20R -> 0 pts, 1.0R -> 100 pts (linear, capped).
    Weighted by profit factor to favour strategies where the edge is backed
    by real profit distribution, not a few lucky outliers.
    """
    ev_r = m.avg_r_multiple
    if ev_r <= 0.20:
        return 0.0

    raw_score = min((ev_r - 0.20) / 0.80, 1.0) * 100.0  # 0.20R->0, 1.0R->100

    # Profit factor quality weight: PF=1.5->0.5, PF=3.0->1.0 (capped)
    pf_weight = min(max((m.profit_factor - 1.5) / 1.5, 0.0), 1.0)
    # Blend: 60% raw EV + 40% PF-weighted EV (ensures low-PF strategies score lower)
    blended = raw_score * (0.60 + 0.40 * pf_weight)
    return round(min(blended, 100.0), 4)


def _score_drawdown_profile(
    m: MetricsResult,
    mc_result: MonteCarloResult | None = None,
) -> float:
    """
    Lower drawdown = higher score.

    When Monte Carlo data is available, uses the 95th percentile drawdown
    (more reliable real-world risk estimate) scaled to the 12% extreme threshold.
    Without MC, falls back to historical max drawdown scaled to the 20% hard limit.

    Also penalises long duration and slow recovery.
    """
    if mc_result is not None:
        # MC path: 0% -> 100 pts, 12% (extreme threshold) -> 0 pts
        dd_score = max(0.0, 1.0 - (mc_result.p95_drawdown / 0.12)) * 80
    else:
        # Historical path: 0% -> 100 pts, 20% (hard limit) -> 0 pts
        dd_score = max(0.0, 1.0 - (m.max_drawdown_pct / 0.20)) * 80

    # Duration penalty: each 100 bars of DD duration loses 5 pts (max 10 pts penalty)
    dur_penalty = min(m.max_drawdown_duration_bars / 100.0 * 5.0, 10.0)

    # Recovery penalty: each 200 bars to recover loses 5 pts (max 10 pts penalty)
    rec_penalty = min(m.max_drawdown_recovery_bars / 200.0 * 5.0, 10.0)

    return max(0.0, dd_score - dur_penalty - rec_penalty)


def _score_risk_adjusted_return(m: MetricsResult) -> float:
    """
    Average of Sharpe score and Sortino score.
    Sharpe: 0 -> 0 pts, 3.0 -> 100 pts (capped).
    Sortino: 0 -> 0 pts, 4.0 -> 100 pts (capped).
    """
    sharpe_score  = min(max(m.sharpe / 3.0, 0.0), 1.0) * 100
    sortino_val   = min(m.sortino, 4.0) if m.sortino != float("inf") else 4.0
    sortino_score = min(max(sortino_val / 4.0, 0.0), 1.0) * 100
    return (sharpe_score + sortino_score) / 2.0


def _score_consistency(m: MetricsResult) -> float:
    """
    Low monthly return variance = higher score.
    Variance of 0% -> 100 pts; 0.10 (10%) -> 0 pts.
    Bonus for positive performance across all three regimes.
    """
    var_score = max(0.0, 1.0 - (m.monthly_return_variance / 0.10)) * 80

    # Regime bonus: up to 20 pts if all three regimes are positive
    regime_bonus = 0.0
    if m.bull_pnl_pct > 0:
        regime_bonus += 6.67
    if m.bear_pnl_pct > 0:
        regime_bonus += 6.67
    if m.sideways_pnl_pct > 0:
        regime_bonus += 6.67

    return min(100.0, var_score + regime_bonus)


def _score_profit_factor(m: MetricsResult) -> float:
    """
    Profit factor below 1.5 -> 0 pts.
    1.5 -> 0 pts; 4.0 -> 100 pts (linearly, capped).
    """
    if m.profit_factor < 1.5:
        return 0.0
    if m.profit_factor == float("inf"):
        return 100.0
    return min((m.profit_factor - 1.5) / 2.5, 1.0) * 100


def _score_statistical_confidence(m: MetricsResult) -> float:
    """
    Score based on trade count as a proxy for statistical reliability.
    Fewer than 30 trades -> 0 pts (should have been filtered already).
    30 -> 0 pts, 100 -> 70 pts, 200+ -> 100 pts, >500 penalised (overtrading).
    """
    n = m.trade_count
    if n < 30:
        return 0.0
    if n <= 100:
        return (n - 30) / 70.0 * 70.0
    if n <= 200:
        return 70.0 + (n - 100) / 100.0 * 30.0
    if n <= 500:
        # Linear decay from 100 -> 50 as frequency goes from 200 -> 500
        excess = (n - 200) / 300.0
        return 100.0 - excess * 50.0
    return 50.0
