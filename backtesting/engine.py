"""
Bar-by-bar backtesting simulation engine.

Simulates a strategy against a candle series, tracking trades,
equity curve, and regime labels. Runs at each risk level and
selects the best-performing one.
"""
from __future__ import annotations

import logging
import math

import numpy as np

logger = logging.getLogger(__name__)

from backtesting.indicators import ema as _ema
from backtesting.models import BacktestResult, RiskLevelResult, Strategy, Trade
from backtesting.signals import (
    calc_stop_price,
    calc_take_profit_price,
    check_entry,
    compute_indicators,
    is_short_trigger,
)
from data.candles import Candle

# ── Constants ─────────────────────────────────────────────────────────────────

INITIAL_CAPITAL: float = 10_000.0
SLIPPAGE_PCT:    float = 0.001    # 0.1% per side
COMMISSION_FLAT: float = 1.0      # $1 minimum per trade
COMMISSION_PCT:  float = 0.0005   # 0.05% of trade value
WARMUP_BARS:     int   = 200      # bars needed for 200-period EMA
RISK_LEVELS:     list[float] = [0.25, 0.5, 1.0, 1.5, 2.0]


# ── Public API ────────────────────────────────────────────────────────────────

def run_all_risk_levels(
    strategy: Strategy,
    candles: list[Candle],
    initial_capital: float = INITIAL_CAPITAL,
    force_short: bool | None = None,
) -> list[RiskLevelResult]:
    """
    Run the strategy at each risk level.
    force_short: True=short, False=long, None=use trigger direction.
    Returns results for all levels (even if some have zero trades).
    """
    from backtesting.metrics import _sharpe_from_equity  # avoid circular at module level
    from backtesting.signals import ENTRY_TRIGGERS

    results = []
    indicators = compute_indicators(strategy, candles)

    # Trigger frequency diagnostic — warn if signal fires rarely (independent of direction)
    trigger_name = strategy.entry.get("trigger", "NONE")
    trigger_fn = ENTRY_TRIGGERS.get(trigger_name)
    if trigger_fn is not None and force_short is not False:
        # Only run diagnostic once (on the first call, i.e. long pass)
        fires = 0
        usable = max(len(candles) - WARMUP_BARS, 1)
        for _i in range(WARMUP_BARS, len(candles)):
            try:
                if trigger_fn(indicators, _i):
                    fires += 1
            except Exception:
                pass
        if fires < 50:
            logger.warning(
                "Low trigger frequency: '%s' (%s) fired only %d/%d IS bars -- may produce too few trades",
                strategy.name[:30], trigger_name, fires, usable,
            )

    for risk_pct in RISK_LEVELS:
        trades, equity_curve = _simulate(
            strategy, candles, indicators, risk_pct, initial_capital, force_short=force_short
        )
        sharpe = _sharpe_from_equity(equity_curve, candles)
        results.append(RiskLevelResult(
            risk_pct=risk_pct,
            trades=trades,
            equity_curve=equity_curve,
            sharpe=sharpe,
        ))

    return results


def run_bidirectional(
    strategy: Strategy,
    candles: list[Candle],
    n_total_bars: int,
    initial_capital: float = INITIAL_CAPITAL,
) -> tuple[list[RiskLevelResult], BacktestResult, str]:
    """
    Run the strategy in both long and short directions independently.
    Selects whichever direction produces the higher Sharpe at its best risk level.
    Returns (risk_results, best_backtest, direction_str).
    """
    long_results  = run_all_risk_levels(strategy, candles, initial_capital, force_short=False)
    short_results = run_all_risk_levels(strategy, candles, initial_capital, force_short=True)

    best_long  = pick_best_risk_level(long_results,  n_total_bars)
    best_short = pick_best_risk_level(short_results, n_total_bars)

    long_sharpe  = next((r.sharpe for r in long_results  if r.risk_pct == best_long.best_risk_pct),  0.0)
    short_sharpe = next((r.sharpe for r in short_results if r.risk_pct == best_short.best_risk_pct), 0.0)

    if short_sharpe > long_sharpe:
        best_short.direction = "short"
        return short_results, best_short, "short"
    else:
        best_long.direction = "long"
        return long_results, best_long, "long"


def pick_best_risk_level(
    risk_results: list[RiskLevelResult],
    n_total_bars: int = 0,
) -> BacktestResult:
    """
    Select the risk level with the best Sharpe Ratio.
    Sets only_works_at_max_risk=True if only the 2% level produces a positive Sharpe.
    n_total_bars: total candle count (IS + OOS) used to compute split sizes.
    """
    valid = [r for r in risk_results if len(r.trades) > 0]
    if not valid:
        # No trades at any risk level — return the first result as placeholder
        r = risk_results[0]
        split = int(n_total_bars * 0.8) if n_total_bars > 0 else 0
        return BacktestResult(
            strategy_id="unknown",
            best_risk_pct=r.risk_pct,
            trades=r.trades,
            equity_curve=r.equity_curve,
            in_sample_bars=split,
            out_of_sample_bars=max(0, n_total_bars - split),
            only_works_at_max_risk=False,
        )

    best = max(valid, key=lambda r: r.sharpe)

    # Check if only the maximum risk level produces a positive Sharpe
    non_max_positive = any(
        r.sharpe > 0 for r in risk_results if r.risk_pct < 2.0
    )
    only_max = (best.risk_pct == 2.0) and not non_max_positive

    split = int(n_total_bars * 0.8) if n_total_bars > 0 else int(len(best.equity_curve) * 0.8)

    return BacktestResult(
        strategy_id="unknown",  # caller fills this in
        best_risk_pct=best.risk_pct,
        trades=best.trades,
        equity_curve=best.equity_curve,
        in_sample_bars=split,
        out_of_sample_bars=max(0, n_total_bars - split),
        only_works_at_max_risk=only_max,
    )


def run_oos_validation(
    strategy: Strategy,
    oos_candles: list[Candle],
    risk_pct: float,
    initial_capital: float = INITIAL_CAPITAL,
    force_short: bool | None = None,
) -> tuple[float, int, float]:
    """
    Run the strategy on OOS candles at the chosen risk level.
    force_short should match the direction selected during IS testing.
    Returns (oos_sharpe, oos_trade_count, oos_win_rate).
    """
    from backtesting.metrics import _sharpe_from_equity  # avoid circular at module level

    if len(oos_candles) < WARMUP_BARS + 2:
        return 0.0, 0, 0.0

    indicators = compute_indicators(strategy, oos_candles)
    trades, equity_curve = _simulate(
        strategy, oos_candles, indicators, risk_pct, initial_capital, force_short=force_short
    )

    oos_sharpe = _sharpe_from_equity(equity_curve, oos_candles)
    oos_trade_count = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    oos_win_rate = wins / oos_trade_count if oos_trade_count > 0 else 0.0

    return oos_sharpe, oos_trade_count, oos_win_rate


def run_recent_window(
    strategy: Strategy,
    candles: list[Candle],
    risk_pct: float,
    force_short: bool,
    years: float = 5.0,
    initial_capital: float = INITIAL_CAPITAL,
) -> dict:
    """
    Run the strategy on the most recent `years` of candle data.
    Returns a summary dict (empty dict if insufficient data).
    Display-only — not used for filtering or scoring.
    """
    from backtesting.metrics import _sharpe_from_equity, _max_drawdown  # avoid circular

    ms_per_year = 365.25 * 24 * 3600 * 1000
    cutoff_ts   = candles[-1].ts - years * ms_per_year
    recent      = [c for c in candles if c.ts >= cutoff_ts]

    if len(recent) < WARMUP_BARS + 10:
        return {}

    indicators  = compute_indicators(strategy, recent)
    trades, equity_curve = _simulate(
        strategy, recent, indicators, risk_pct, initial_capital, force_short=force_short
    )

    if not trades:
        return {}

    initial    = equity_curve[0] if equity_curve else initial_capital
    total_ret  = (equity_curve[-1] - initial) / initial if initial > 0 else 0.0
    actual_yrs = (candles[-1].ts - recent[0].ts) / ms_per_year
    annual_ret = ((1 + total_ret) ** (1 / max(actual_yrs, 0.01))) - 1 if total_ret > -1.0 else -1.0

    sharpe     = _sharpe_from_equity(equity_curve, recent)
    dd_pct, _, _ = _max_drawdown(equity_curve)
    wins       = sum(1 for t in trades if t.pnl > 0)

    return {
        "years":             round(actual_yrs, 1),
        "total_return_pct":  round(total_ret * 100, 2),
        "annual_return_pct": round(annual_ret * 100, 2),
        "max_drawdown_pct":  round(dd_pct * 100, 2),
        "sharpe":            round(sharpe, 3),
        "trade_count":       len(trades),
        "win_rate":          round(wins / len(trades) * 100, 2),
    }


# ── Simulation loop ───────────────────────────────────────────────────────────

def _simulate(
    strategy: Strategy,
    candles: list[Candle],
    indicators: dict,
    risk_pct: float,
    initial_capital: float,
    force_short: bool | None = None,
) -> tuple[list[Trade], list[float]]:
    """
    Bar-by-bar simulation.

    Equity curve has one entry per bar (the equity value at close of that bar).
    Trades record exact entry/exit details.
    """
    n = len(candles)
    if n < WARMUP_BARS + 2:
        return [], [initial_capital] * n

    equity = initial_capital
    equity_curve = [initial_capital] * n

    max_positions = int(strategy.risk.get("max_open_positions", 3))
    time_exit_bars = strategy.exit.get("time_exit_bars")
    trailing_atr_mult = strategy.exit.get("trailing_stop_atr")

    short = force_short if force_short is not None else is_short_trigger(strategy.entry.get("trigger", ""))
    direction = "short" if short else "long"

    # Open positions list — each entry is a dict with position state
    open_positions: list[dict] = []
    completed_trades: list[Trade] = []

    regime_arr = _compute_regimes(indicators["ema200"], np.array([c.close for c in candles]))

    for i in range(WARMUP_BARS, n):
        bar_close = candles[i].close

        # ── Check exits for all open positions ────────────────────────
        still_open = []
        for pos in open_positions:
            trade = _check_exit(pos, candles, indicators, i, time_exit_bars, trailing_atr_mult, equity)
            if trade is not None:
                equity += trade.pnl
                completed_trades.append(trade)
            else:
                # Update trailing stop if applicable
                if trailing_atr_mult and not math.isnan(indicators["atr"][i]):
                    atr_val = indicators["atr"][i]
                    if short:
                        new_trail = bar_close + trailing_atr_mult * atr_val
                        pos["trailing_stop"] = min(pos.get("trailing_stop", math.inf), new_trail)
                    else:
                        new_trail = bar_close - trailing_atr_mult * atr_val
                        pos["trailing_stop"] = max(pos.get("trailing_stop", 0), new_trail)
                still_open.append(pos)

        open_positions = still_open

        # ── Check entry ───────────────────────────────────────────────
        if len(open_positions) < max_positions and check_entry(strategy, indicators, i):
            # Short fills slightly below current price; long slightly above
            entry_price = bar_close * (1 - SLIPPAGE_PCT) if short else bar_close * (1 + SLIPPAGE_PCT)
            stop_price  = calc_stop_price(strategy, entry_price, indicators, i, short=short)

            # For longs: stop must be below entry. For shorts: stop must be above entry.
            invalid_stop = (stop_price <= entry_price) if short else (stop_price >= entry_price)
            if invalid_stop or stop_price <= 0:
                equity_curve[i] = equity
                continue

            risk_usd  = equity * (risk_pct / 100.0)
            risk_dist = abs(entry_price - stop_price)
            size      = risk_usd / risk_dist
            cost      = _commission(entry_price * size)
            equity   -= cost

            tp_price = calc_take_profit_price(strategy, entry_price, stop_price, short=short)
            if trailing_atr_mult and not math.isnan(indicators["atr"][i]):
                atr_val = indicators["atr"][i]
                trail_stop = (entry_price + trailing_atr_mult * atr_val if short
                              else entry_price - trailing_atr_mult * atr_val)
            else:
                trail_stop = None

            open_positions.append({
                "entry_bar":     i,
                "entry_price":   entry_price,
                "stop_price":    stop_price,
                "tp_price":      tp_price,
                "trailing_stop": trail_stop,
                "size":          size,
                "risk_usd":      risk_usd,
                "regime":        regime_arr[i],
                "entry_ts":      candles[i].ts,
                "direction":     direction,
            })

        equity_curve[i] = equity

    # ── Force-close any positions still open at end of data ──────────
    for pos in open_positions:
        pos_short = pos.get("direction") == "short"
        exit_price = candles[-1].close * (1 + SLIPPAGE_PCT if pos_short else 1 - SLIPPAGE_PCT)
        cost       = _commission(exit_price * pos["size"])
        gross_pnl  = ((pos["entry_price"] - exit_price) * pos["size"] if pos_short
                      else (exit_price - pos["entry_price"]) * pos["size"])
        net_pnl    = gross_pnl - cost
        equity    += net_pnl
        equity_curve[-1] = equity
        completed_trades.append(Trade(
            entry_bar=pos["entry_bar"],
            exit_bar=n - 1,
            entry_price=pos["entry_price"],
            exit_price=exit_price,
            size=pos["size"],
            direction=pos.get("direction", "long"),
            pnl=net_pnl,
            pnl_pct=net_pnl / (pos["entry_price"] * pos["size"]),
            r_multiple=net_pnl / pos["risk_usd"] if pos["risk_usd"] > 0 else 0,
            exit_reason="end_of_data",
            entry_ts=pos["entry_ts"],
            exit_ts=candles[-1].ts,
            regime=pos["regime"],
        ))

    return completed_trades, equity_curve


def _check_exit(
    pos: dict,
    candles: list[Candle],
    indicators: dict,
    i: int,
    time_exit_bars,
    trailing_atr_mult,
    equity: float,
) -> Trade | None:
    """
    Check whether an open position should exit at bar i.
    Returns a completed Trade if exiting, None to keep holding.
    """
    low   = candles[i].low
    high  = candles[i].high
    close = candles[i].close
    short = pos.get("direction") == "short"

    exit_price = None
    exit_reason = None

    if short:
        # Short: stop hit when price rises to/above stop; TP hit when price falls to/below TP
        if high >= pos["stop_price"]:
            exit_price  = pos["stop_price"] * (1 + SLIPPAGE_PCT)
            exit_reason = "stop_loss"
        elif low <= pos["tp_price"]:
            exit_price  = pos["tp_price"] * (1 + SLIPPAGE_PCT)
            exit_reason = "take_profit"
        elif pos.get("trailing_stop") and high >= pos["trailing_stop"]:
            exit_price  = pos["trailing_stop"] * (1 + SLIPPAGE_PCT)
            exit_reason = "trailing_stop"
        elif time_exit_bars and (i - pos["entry_bar"]) >= time_exit_bars:
            exit_price  = close * (1 + SLIPPAGE_PCT)
            exit_reason = "time_exit"
    else:
        # Long: stop hit when price falls to/below stop; TP hit when price rises to/above TP
        if low <= pos["stop_price"]:
            exit_price  = pos["stop_price"] * (1 - SLIPPAGE_PCT)
            exit_reason = "stop_loss"
        elif high >= pos["tp_price"]:
            exit_price  = pos["tp_price"] * (1 - SLIPPAGE_PCT)
            exit_reason = "take_profit"
        elif pos.get("trailing_stop") and low <= pos["trailing_stop"]:
            exit_price  = pos["trailing_stop"] * (1 - SLIPPAGE_PCT)
            exit_reason = "trailing_stop"
        elif time_exit_bars and (i - pos["entry_bar"]) >= time_exit_bars:
            exit_price  = close * (1 - SLIPPAGE_PCT)
            exit_reason = "time_exit"

    if exit_price is None:
        return None

    cost      = _commission(exit_price * pos["size"])
    gross_pnl = ((pos["entry_price"] - exit_price) * pos["size"] if short
                 else (exit_price - pos["entry_price"]) * pos["size"])
    net_pnl   = gross_pnl - cost

    return Trade(
        entry_bar=pos["entry_bar"],
        exit_bar=i,
        entry_price=pos["entry_price"],
        exit_price=exit_price,
        size=pos["size"],
        direction=pos.get("direction", "long"),
        pnl=net_pnl,
        pnl_pct=net_pnl / (pos["entry_price"] * pos["size"]),
        r_multiple=net_pnl / pos["risk_usd"] if pos["risk_usd"] > 0 else 0,
        exit_reason=exit_reason,
        entry_ts=pos["entry_ts"],
        exit_ts=candles[i].ts,
        regime=pos["regime"],
    )


def _commission(trade_value: float) -> float:
    return max(COMMISSION_FLAT, trade_value * COMMISSION_PCT)


def _compute_regimes(ema200: np.ndarray, close: np.ndarray) -> list[str]:
    """
    Classify each bar as 'bull', 'bear', or 'sideways'.
    Bull:     close > EMA200 and EMA200 is rising (vs 5 bars ago)
    Bear:     close < EMA200 and EMA200 is falling
    Sideways: everything else
    """
    n = len(close)
    regimes = ["sideways"] * n
    for i in range(5, n):
        e = ema200[i]
        if math.isnan(e):
            continue
        e_prev = ema200[i - 5]
        if math.isnan(e_prev):
            continue
        if close[i] > e and e > e_prev:
            regimes[i] = "bull"
        elif close[i] < e and e < e_prev:
            regimes[i] = "bear"
    return regimes
