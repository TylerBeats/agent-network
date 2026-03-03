"""
Weekly and monthly performance review computation.

Operates on the raw trade_log produced by DryRunBroker (list of dicts),
filtering by ISO week/month and computing all relevant metrics.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class WeeklyReview:
    week: str               # ISO week string, e.g. "2026-W03"
    pnl_usd: float
    pnl_pct: float          # pnl_usd / starting_equity
    win_rate: float         # fraction of winning exit trades
    avg_r_multiple: float
    max_drawdown_pct: float
    trade_count: int        # number of exit trades in this week
    health_tier: str = "Good"         # "Good" | "Marginal" | "Poor"
    ev_drift_r: float = 0.0           # live avg_r - projected avg_r
    win_rate_drift: float = 0.0       # live win_rate - projected win_rate (fraction)
    live_dd_vs_mc_ratio: float = 0.0  # live max_dd / mc_p95_dd (>1.0 = exceeded)


@dataclass
class MonthlyReview:
    month: str              # "2026-01"
    pnl_pct: float          # total monthly P&L as fraction
    live_sharpe: float      # annualised Sharpe from weekly returns
    backtest_sharpe: float  # from the original backtest result
    sharpe_decay: float     # live_sharpe - backtest_sharpe (negative = decaying)
    max_drawdown_pct: float # worst weekly drawdown within this month
    consecutive_losing_weeks: int
    risk_adjustment: float  # position size multiplier: 0.5 | 0.75 | 1.0 | 1.25
    retire: bool            # True if 3+ consecutive losing weeks
    health_tier: str = "Good"  # dominant tier across the month's weekly reviews


def compute_weekly_review(
    trades: list[dict],
    week_str: str,
    starting_equity: float = 10_000.0,
    projected_avg_r: float = 0.0,
    projected_win_rate: float = 0.0,
    mc_p95_dd: float | None = None,
) -> WeeklyReview:
    """
    Compute a WeeklyReview from a list of exit trade dicts.

    Args:
        trades:             Full trade_log from DryRunBroker (entry + exit records).
        week_str:           ISO week string to filter by, e.g. "2026-W03".
        starting_equity:    Account equity at start of week (used for pnl_pct denominator).
        projected_avg_r:    Expected avg R-multiple per trade from the backtest.
        projected_win_rate: Expected win rate from the backtest (fraction).
        mc_p95_dd:          Monte Carlo 95th-pct drawdown from backtest (fraction, e.g. 0.07).

    Returns:
        WeeklyReview for the specified week with health tier and drift metrics.
    """
    exit_trades = [
        t for t in trades
        if t.get("type") == "exit" and _iso_week(t.get("timestamp", 0)) == week_str
    ]

    trade_count = len(exit_trades)
    if trade_count == 0:
        return WeeklyReview(
            week=week_str,
            pnl_usd=0.0,
            pnl_pct=0.0,
            win_rate=0.0,
            avg_r_multiple=0.0,
            max_drawdown_pct=0.0,
            trade_count=0,
            health_tier="Good",
            ev_drift_r=0.0,
            win_rate_drift=0.0,
            live_dd_vs_mc_ratio=0.0,
        )

    pnl_usd     = sum(t.get("pnl", 0.0) for t in exit_trades)
    pnl_pct     = pnl_usd / starting_equity if starting_equity > 0 else 0.0
    wins        = sum(1 for t in exit_trades if t.get("pnl", 0.0) > 0)
    win_rate    = wins / trade_count
    r_multiples = [t.get("r_multiple", 0.0) for t in exit_trades]
    avg_r       = sum(r_multiples) / len(r_multiples)

    # Max drawdown from sequential equity curve within the week
    max_dd = _week_max_drawdown(exit_trades, starting_equity)

    # Health drift metrics
    ev_drift_r     = avg_r - projected_avg_r
    win_rate_drift = win_rate - projected_win_rate
    live_dd_ratio  = (max_dd / mc_p95_dd) if (mc_p95_dd and mc_p95_dd > 0) else 0.0
    health_tier    = _classify_health_tier(ev_drift_r, win_rate_drift, live_dd_ratio)

    return WeeklyReview(
        week=week_str,
        pnl_usd=pnl_usd,
        pnl_pct=pnl_pct,
        win_rate=win_rate,
        avg_r_multiple=avg_r,
        max_drawdown_pct=max_dd,
        trade_count=trade_count,
        health_tier=health_tier,
        ev_drift_r=round(ev_drift_r, 4),
        win_rate_drift=round(win_rate_drift, 4),
        live_dd_vs_mc_ratio=round(live_dd_ratio, 4),
    )


def compute_monthly_review(
    weekly_reviews: list[WeeklyReview],
    backtest_sharpe: float,
) -> MonthlyReview:
    """
    Aggregate weekly reviews into a monthly summary and apply risk adjustment rules.

    Risk adjustment rules (applied in priority order):
      - Monthly max drawdown > 15%           → risk_adjustment = 0.50
      - Monthly PnL < -5% (losing month)     → risk_adjustment = 0.75
      - Monthly PnL > 15% (strong month)     → risk_adjustment = 1.25
      - Otherwise                             → risk_adjustment = 1.00

    Retirement: retire = True if consecutive_losing_weeks >= 3.

    Health tier: any "Poor" week → "Poor"; any "Marginal" → "Marginal"; else "Good".
    """
    month = datetime.now(tz=timezone.utc).strftime("%Y-%m")

    if not weekly_reviews:
        return MonthlyReview(
            month=month,
            pnl_pct=0.0,
            live_sharpe=0.0,
            backtest_sharpe=backtest_sharpe,
            sharpe_decay=-backtest_sharpe,
            max_drawdown_pct=0.0,
            consecutive_losing_weeks=0,
            risk_adjustment=1.0,
            retire=False,
            health_tier="Good",
        )

    weekly_returns   = [w.pnl_pct for w in weekly_reviews]
    total_pnl_pct    = sum(weekly_returns)
    max_drawdown_pct = max(w.max_drawdown_pct for w in weekly_reviews)

    # Annualised Sharpe from weekly returns (need >= 2 weeks for meaningful std)
    if len(weekly_returns) >= 2:
        mean_r = sum(weekly_returns) / len(weekly_returns)
        variance = sum((r - mean_r) ** 2 for r in weekly_returns) / (len(weekly_returns) - 1)
        std_r = math.sqrt(variance) if variance > 0 else 0.0
        live_sharpe = (mean_r / std_r * math.sqrt(52)) if std_r > 0 else 0.0
    else:
        live_sharpe = 0.0

    sharpe_decay = live_sharpe - backtest_sharpe

    # Count consecutive losing weeks (trailing streak)
    consecutive_losing = 0
    for w in reversed(weekly_reviews):
        if w.pnl_pct < 0:
            consecutive_losing += 1
        else:
            break

    retire = consecutive_losing >= 3

    # Risk adjustment (priority: drawdown > loss > gain > default)
    if max_drawdown_pct > 0.15:
        risk_adjustment = 0.50
    elif total_pnl_pct < -0.05:
        risk_adjustment = 0.75
    elif total_pnl_pct > 0.15:
        risk_adjustment = 1.25
    else:
        risk_adjustment = 1.00

    # Health tier from constituent weekly tiers
    tiers = [w.health_tier for w in weekly_reviews]
    if "Poor" in tiers:
        health_tier = "Poor"
    elif "Marginal" in tiers:
        health_tier = "Marginal"
    else:
        health_tier = "Good"

    return MonthlyReview(
        month=month,
        pnl_pct=total_pnl_pct,
        live_sharpe=round(live_sharpe, 4),
        backtest_sharpe=backtest_sharpe,
        sharpe_decay=round(sharpe_decay, 4),
        max_drawdown_pct=max_drawdown_pct,
        consecutive_losing_weeks=consecutive_losing,
        risk_adjustment=risk_adjustment,
        retire=retire,
        health_tier=health_tier,
    )


def compute_per_strategy_reviews(
    trades: list[dict],
    week_str: str,
    starting_equity: float,
    projected_avg_r: float,
    projected_win_rate: float,
    mc_p95_dd_by_strategy: dict[str, float],
) -> dict[str, WeeklyReview]:
    """
    Group exit trades by strategy_id and compute a WeeklyReview for each.

    Returns a dict mapping strategy_id -> WeeklyReview.
    Strategies with no trades in the given week are omitted.
    """
    by_strategy: dict[str, list[dict]] = {}
    for t in trades:
        if t.get("type") != "exit":
            continue
        if _iso_week(t.get("timestamp", 0)) != week_str:
            continue
        sid = t.get("strategy_id", "unknown")
        by_strategy.setdefault(sid, []).append(t)

    result: dict[str, WeeklyReview] = {}
    for sid, strat_trades in by_strategy.items():
        # Wrap in full trade list format that compute_weekly_review expects
        wrapped = [dict(t) for t in strat_trades]
        result[sid] = compute_weekly_review(
            wrapped, week_str, starting_equity,
            projected_avg_r, projected_win_rate,
            mc_p95_dd_by_strategy.get(sid),
        )
    return result


# ── Internal helpers ──────────────────────────────────────────────────────────

def _classify_health_tier(
    ev_drift_r: float,
    win_rate_drift: float,
    live_dd_vs_mc_ratio: float,
) -> str:
    """
    Classify strategy health based on live vs backtest drift metrics.

    Poor:    any severe degradation (EV dropped >0.20R, WR down >10pp, or live DD
             exceeded MC p95 by more than 20%)
    Good:    all metrics within tolerance
    Marginal: everything in between
    """
    if ev_drift_r < -0.20 or win_rate_drift < -0.10 or live_dd_vs_mc_ratio > 1.2:
        return "Poor"
    if ev_drift_r >= -0.10 and win_rate_drift >= -0.05 and live_dd_vs_mc_ratio <= 1.0:
        return "Good"
    return "Marginal"


def _iso_week(timestamp_ms: int) -> str:
    """Convert Unix milliseconds to ISO week string, e.g. '2026-W03'."""
    dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
    year, week, _ = dt.isocalendar()
    return f"{year}-W{week:02d}"


def _week_max_drawdown(exit_trades: list[dict], starting_equity: float) -> float:
    """Compute maximum drawdown fraction from sequential trade pnls within a week."""
    equity = starting_equity
    peak   = equity
    max_dd = 0.0
    for t in sorted(exit_trades, key=lambda x: x.get("timestamp", 0)):
        equity += t.get("pnl", 0.0)
        if equity > peak:
            peak = equity
        if peak > 0:
            dd = (peak - equity) / peak
            if dd > max_dd:
                max_dd = dd
    return max_dd
