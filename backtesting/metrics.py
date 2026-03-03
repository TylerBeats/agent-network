"""
Performance metric computation.

All metrics are derived from the equity curve and trade list
produced by the backtesting engine.
"""
from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

from backtesting.models import BacktestResult, MetricsResult, Trade
from data.candles import Candle

# Annualisation factor — 365 trading days for crypto (24/7 markets)
_ANNUALISE = math.sqrt(365)


def compute_metrics(result: BacktestResult, candles: list[Candle]) -> MetricsResult:
    """Compute all performance metrics from a BacktestResult."""
    trades      = result.trades
    equity_curve = result.equity_curve
    initial     = equity_curve[0] if equity_curve else 10_000.0

    if not trades:
        return _zero_metrics(result.best_risk_pct)

    # ── Total P&L ─────────────────────────────────────────────────────────────
    final_equity  = equity_curve[-1] if equity_curve else initial
    total_pnl_usd = final_equity - initial
    total_pnl_pct = total_pnl_usd / initial if initial > 0 else 0.0

    # ── Sharpe & Sortino ─────────────────────────────────────────────────────
    sharpe  = _sharpe_from_equity(equity_curve, candles)
    sortino = _sortino_from_equity(equity_curve, candles)

    # ── Drawdown ─────────────────────────────────────────────────────────────
    dd_pct, dd_dur, dd_rec = _max_drawdown(equity_curve)

    # ── Win rate & profit factor ──────────────────────────────────────────────
    wins   = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    win_rate = len(wins) / len(trades)

    gross_profit = sum(t.pnl for t in wins)
    gross_loss   = abs(sum(t.pnl for t in losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    # ── R-multiple ────────────────────────────────────────────────────────────
    avg_r = sum(t.r_multiple for t in trades) / len(trades)

    # ── Consecutive losses ────────────────────────────────────────────────────
    max_consec = _max_consecutive_losses(trades)

    # ── Worst single trade ────────────────────────────────────────────────────
    # Express as fraction of initial capital (not per-trade equity — simpler & consistent)
    worst_pct = abs(min(t.pnl for t in losses)) / initial if losses else 0.0

    # ── Monthly return variance ───────────────────────────────────────────────
    monthly_var = _monthly_return_variance(equity_curve, candles)

    # ── Regime P&L ────────────────────────────────────────────────────────────
    bull_pnl, bear_pnl, sideways_pnl = _regime_pnl(trades, initial)

    return MetricsResult(
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown_pct=dd_pct,
        max_drawdown_duration_bars=dd_dur,
        max_drawdown_recovery_bars=dd_rec,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_r_multiple=avg_r,
        trade_count=len(trades),
        max_consecutive_losses=max_consec,
        worst_trade_pct=worst_pct,
        monthly_return_variance=monthly_var,
        bull_pnl_pct=bull_pnl,
        bear_pnl_pct=bear_pnl,
        sideways_pnl_pct=sideways_pnl,
        best_risk_pct=result.best_risk_pct,
        total_pnl_pct=total_pnl_pct,
        total_pnl_usd=total_pnl_usd,
    )


# ── Helper functions ──────────────────────────────────────────────────────────

def _daily_returns(equity_curve: list[float], candles: list[Candle]) -> np.ndarray:
    """
    Sample equity at each calendar-day boundary and compute log returns.
    Falls back to per-bar returns if candle timestamps are not available.
    """
    if not equity_curve or len(equity_curve) < 2:
        return np.array([])

    # Group bars by day using candle timestamps
    day_equity: dict[str, float] = {}
    for i, c in enumerate(candles):
        if i >= len(equity_curve):
            break
        day = datetime.fromtimestamp(c.ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        day_equity[day] = equity_curve[i]  # last equity value of each day

    if len(day_equity) < 2:
        # Not enough days — use bar-level returns
        eq = np.array(equity_curve)
        return np.diff(eq) / eq[:-1]

    sorted_vals = [v for _, v in sorted(day_equity.items())]
    eq = np.array(sorted_vals)
    returns = np.diff(eq) / eq[:-1]
    return returns[np.isfinite(returns)]


def _sharpe_from_equity(equity_curve: list[float], candles: list[Candle]) -> float:
    """Annualised Sharpe Ratio (risk-free rate = 0)."""
    returns = _daily_returns(equity_curve, candles)
    if len(returns) < 2:
        return 0.0
    std = np.std(returns, ddof=1)
    if std == 0:
        return 0.0
    return float(np.mean(returns) / std * _ANNUALISE)


def _sortino_from_equity(equity_curve: list[float], candles: list[Candle]) -> float:
    """Annualised Sortino Ratio (downside deviation only)."""
    returns = _daily_returns(equity_curve, candles)
    if len(returns) < 2:
        return 0.0
    downside = returns[returns < 0]
    if len(downside) == 0:
        return float("inf")
    downside_std = np.std(downside, ddof=1)
    if downside_std == 0:
        return 0.0
    return float(np.mean(returns) / downside_std * _ANNUALISE)


def _max_drawdown(equity_curve: list[float]) -> tuple[float, int, int]:
    """
    Returns (max_drawdown_pct, duration_bars, recovery_bars).
    max_drawdown_pct is expressed as a positive fraction (e.g. 0.15 = 15% drawdown).
    """
    if not equity_curve:
        return 0.0, 0, 0

    eq = np.array(equity_curve)
    peak = eq[0]
    peak_idx = 0
    max_dd = 0.0
    max_dd_peak_idx = 0
    max_dd_trough_idx = 0

    for i, val in enumerate(eq):
        if val > peak:
            peak = val
            peak_idx = i
        dd = (peak - val) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
            max_dd_peak_idx = peak_idx
            max_dd_trough_idx = i

    duration = max_dd_trough_idx - max_dd_peak_idx

    # Recovery: bars from trough back to previous peak level
    recovery = 0
    trough_val = eq[max_dd_trough_idx]
    peak_val   = eq[max_dd_peak_idx]
    for i in range(max_dd_trough_idx, len(eq)):
        if eq[i] >= peak_val:
            recovery = i - max_dd_trough_idx
            break
    else:
        recovery = len(eq) - max_dd_trough_idx  # not recovered within data

    return max_dd, duration, recovery


def _max_consecutive_losses(trades: list[Trade]) -> int:
    max_streak = 0
    streak = 0
    for t in trades:
        if t.pnl <= 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak


def _monthly_return_variance(equity_curve: list[float], candles: list[Candle]) -> float:
    """Standard deviation of monthly returns."""
    if not equity_curve or not candles:
        return 0.0

    month_equity: dict[str, list[float]] = defaultdict(list)
    for i, c in enumerate(candles):
        if i >= len(equity_curve):
            break
        month = datetime.fromtimestamp(c.ts / 1000, tz=timezone.utc).strftime("%Y-%m")
        month_equity[month].append(equity_curve[i])

    if len(month_equity) < 2:
        return 0.0

    monthly_returns = []
    sorted_months = sorted(month_equity.keys())
    for j in range(1, len(sorted_months)):
        prev = month_equity[sorted_months[j - 1]][-1]
        curr = month_equity[sorted_months[j]][-1]
        if prev > 0:
            monthly_returns.append((curr - prev) / prev)

    if len(monthly_returns) < 2:
        return 0.0
    return float(np.std(monthly_returns, ddof=1))


def _regime_pnl(trades: list[Trade], initial_capital: float) -> tuple[float, float, float]:
    """Sum P&L by regime, expressed as fraction of initial capital."""
    bull = sum(t.pnl for t in trades if t.regime == "bull")
    bear = sum(t.pnl for t in trades if t.regime == "bear")
    side = sum(t.pnl for t in trades if t.regime == "sideways")
    return bull / initial_capital, bear / initial_capital, side / initial_capital


def _zero_metrics(risk_pct: float) -> MetricsResult:
    return MetricsResult(
        sharpe=0.0, sortino=0.0,
        max_drawdown_pct=0.0, max_drawdown_duration_bars=0, max_drawdown_recovery_bars=0,
        win_rate=0.0, profit_factor=0.0, avg_r_multiple=0.0, trade_count=0,
        max_consecutive_losses=0, worst_trade_pct=0.0, monthly_return_variance=0.0,
        bull_pnl_pct=0.0, bear_pnl_pct=0.0, sideways_pnl_pct=0.0,
        best_risk_pct=risk_pct,
    )
