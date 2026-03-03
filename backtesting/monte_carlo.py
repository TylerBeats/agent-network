"""
Monte Carlo drawdown analysis.

Runs 800 independent simulations by resampling trade outcomes (with replacement)
to estimate the 95th percentile maximum drawdown -- the statistical worst-case
rather than just the single historical path.

This is more reliable than historical max DD for risk classification because it
accounts for the full range of possible trade sequences the strategy could produce.
"""
from __future__ import annotations

import numpy as np

from backtesting.models import MonteCarloResult, Trade


# Risk classification thresholds (95th pct drawdown)
_LOW_THRESHOLD      = 0.04   # < 4%   -> Low
_MODERATE_THRESHOLD = 0.07   # < 7%   -> Moderate
_HIGH_THRESHOLD     = 0.12   # < 12%  -> High
                              # >= 12% -> Extreme


def run_monte_carlo(
    trades: list[Trade],
    risk_pct: float,
    n_sims: int = 800,
    trades_per_sim: int = 100,
) -> MonteCarloResult:
    """
    Run Monte Carlo drawdown simulations for a strategy.

    Args:
        trades:         Completed trades from the IS backtest.
        risk_pct:       Risk per trade as a percentage (e.g. 1.0 = 1%).
        n_sims:         Number of independent simulations (default 800).
        trades_per_sim: Trades resampled per simulation (default 100).

    Returns:
        MonteCarloResult with 95th percentile statistics and risk classification.
    """
    if len(trades) < 10:
        # Not enough real trades to resample meaningfully -- worst-case classification
        return MonteCarloResult(
            p95_drawdown=1.0,
            p95_consecutive_losses=99,
            risk_class="extreme",
            prop_firm_5pct=False,
            prop_firm_8pct=False,
            prop_firm_10pct=False,
            n_sims=n_sims,
            n_trades=trades_per_sim,
        )

    # Extract R multiples from completed trades
    r_multiples = np.array([t.r_multiple for t in trades], dtype=float)
    risk_fraction = risk_pct / 100.0  # convert % to fraction for equity calculation

    rng = np.random.default_rng(seed=42)  # reproducible results

    sim_drawdowns: list[float] = []
    sim_streaks: list[int] = []

    for _ in range(n_sims):
        # Resample trades_per_sim R multiples with replacement
        sample = rng.choice(r_multiples, size=trades_per_sim, replace=True)

        # Build equity curve: equity[t+1] = equity[t] * (1 + r * risk_fraction)
        equity = 1.0
        peak = 1.0
        max_dd = 0.0
        streak = 0
        max_streak = 0

        for r in sample:
            equity *= (1.0 + r * risk_fraction)
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
            # Track consecutive losses
            if r < 0:
                streak += 1
                if streak > max_streak:
                    max_streak = streak
            else:
                streak = 0

        sim_drawdowns.append(max_dd)
        sim_streaks.append(max_streak)

    sim_drawdowns.sort()
    sim_streaks.sort()

    # 95th percentile: index 760 of 800 sorted values (0-indexed)
    p95_idx = int(n_sims * 0.95)
    p95_dd = sim_drawdowns[min(p95_idx, len(sim_drawdowns) - 1)]
    p95_streak = sim_streaks[min(p95_idx, len(sim_streaks) - 1)]

    risk_class = _classify(p95_dd)

    p95_dd_f = float(p95_dd)   # ensure plain Python float (not numpy.float64)
    return MonteCarloResult(
        p95_drawdown=round(p95_dd_f, 4),
        p95_consecutive_losses=int(p95_streak),
        risk_class=risk_class,
        prop_firm_5pct=bool(p95_dd_f < 0.05),
        prop_firm_8pct=bool(p95_dd_f < 0.08),
        prop_firm_10pct=bool(p95_dd_f < 0.10),
        n_sims=n_sims,
        n_trades=trades_per_sim,
    )


def _classify(p95_dd: float) -> str:
    if p95_dd < _LOW_THRESHOLD:
        return "low"
    if p95_dd < _MODERATE_THRESHOLD:
        return "moderate"
    if p95_dd < _HIGH_THRESHOLD:
        return "high"
    return "extreme"
