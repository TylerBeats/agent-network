"""
Bar-by-bar backtesting simulation engine.

Simulates a strategy against a candle series, tracking trades,
equity curve, and regime labels. Runs at each risk level and
selects the best-performing one.
"""
from __future__ import annotations

import math

import numpy as np

from backtesting.indicators import ema as _ema
from backtesting.models import BacktestResult, RiskLevelResult, Strategy, Trade
from backtesting.signals import (
    calc_stop_price,
    calc_take_profit_price,
    check_entry,
    compute_indicators,
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
) -> list[RiskLevelResult]:
    """
    Run the strategy at each of the four risk levels.
    Returns results for all four levels (even if some have zero trades).
    """
    from backtesting.metrics import _sharpe_from_equity  # avoid circular at module level

    results = []
    indicators = compute_indicators(strategy, candles)

    for risk_pct in RISK_LEVELS:
        trades, equity_curve = _simulate(
            strategy, candles, indicators, risk_pct, initial_capital
        )
        sharpe = _sharpe_from_equity(equity_curve, candles)
        results.append(RiskLevelResult(
            risk_pct=risk_pct,
            trades=trades,
            equity_curve=equity_curve,
            sharpe=sharpe,
        ))

    return results


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
) -> tuple[float, int, float]:
    """
    Run the strategy on OOS candles at the chosen risk level.
    Returns (oos_sharpe, oos_trade_count, oos_win_rate).
    """
    from backtesting.metrics import _sharpe_from_equity  # avoid circular at module level

    if len(oos_candles) < WARMUP_BARS + 2:
        return 0.0, 0, 0.0

    indicators = compute_indicators(strategy, oos_candles)
    trades, equity_curve = _simulate(strategy, oos_candles, indicators, risk_pct, initial_capital)

    oos_sharpe = _sharpe_from_equity(equity_curve, oos_candles)
    oos_trade_count = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    oos_win_rate = wins / oos_trade_count if oos_trade_count > 0 else 0.0

    return oos_sharpe, oos_trade_count, oos_win_rate


# ── Simulation loop ───────────────────────────────────────────────────────────

def _simulate(
    strategy: Strategy,
    candles: list[Candle],
    indicators: dict,
    risk_pct: float,
    initial_capital: float,
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
                    new_trail = bar_close - trailing_atr_mult * indicators["atr"][i]
                    pos["trailing_stop"] = max(pos.get("trailing_stop", 0), new_trail)
                still_open.append(pos)

        open_positions = still_open

        # ── Check entry ───────────────────────────────────────────────
        if len(open_positions) < max_positions and check_entry(strategy, indicators, i):
            entry_price = bar_close * (1 + SLIPPAGE_PCT)
            stop_price  = calc_stop_price(strategy, entry_price, indicators, i)

            if stop_price >= entry_price or stop_price <= 0:
                # Invalid stop — skip this entry
                equity_curve[i] = equity
                continue

            risk_usd  = equity * (risk_pct / 100.0)
            risk_dist = entry_price - stop_price
            size      = risk_usd / risk_dist
            cost      = _commission(entry_price * size)
            equity   -= cost

            tp_price = calc_take_profit_price(strategy, entry_price, stop_price)
            trail_stop = (
                entry_price - trailing_atr_mult * indicators["atr"][i]
                if trailing_atr_mult and not math.isnan(indicators["atr"][i])
                else None
            )

            open_positions.append({
                "entry_bar":    i,
                "entry_price":  entry_price,
                "stop_price":   stop_price,
                "tp_price":     tp_price,
                "trailing_stop": trail_stop,
                "size":         size,
                "risk_usd":     risk_usd,
                "regime":       regime_arr[i],
                "entry_ts":     candles[i].ts,
            })

        equity_curve[i] = equity

    # ── Force-close any positions still open at end of data ──────────
    for pos in open_positions:
        exit_price = candles[-1].close * (1 - SLIPPAGE_PCT)
        cost       = _commission(exit_price * pos["size"])
        gross_pnl  = (exit_price - pos["entry_price"]) * pos["size"]
        net_pnl    = gross_pnl - cost
        equity    += net_pnl
        equity_curve[-1] = equity
        completed_trades.append(Trade(
            entry_bar=pos["entry_bar"],
            exit_bar=n - 1,
            entry_price=pos["entry_price"],
            exit_price=exit_price,
            size=pos["size"],
            direction="long",
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

    exit_price = None
    exit_reason = None

    # Stop loss hit (check against bar's low)
    if low <= pos["stop_price"]:
        exit_price  = pos["stop_price"] * (1 - SLIPPAGE_PCT)
        exit_reason = "stop_loss"

    # Take profit hit (check against bar's high)
    elif high >= pos["tp_price"]:
        exit_price  = pos["tp_price"] * (1 - SLIPPAGE_PCT)
        exit_reason = "take_profit"

    # Trailing stop hit
    elif pos.get("trailing_stop") and low <= pos["trailing_stop"]:
        exit_price  = pos["trailing_stop"] * (1 - SLIPPAGE_PCT)
        exit_reason = "trailing_stop"

    # Time exit
    elif time_exit_bars and (i - pos["entry_bar"]) >= time_exit_bars:
        exit_price  = close * (1 - SLIPPAGE_PCT)
        exit_reason = "time_exit"

    if exit_price is None:
        return None

    cost      = _commission(exit_price * pos["size"])
    gross_pnl = (exit_price - pos["entry_price"]) * pos["size"]
    net_pnl   = gross_pnl - cost

    return Trade(
        entry_bar=pos["entry_bar"],
        exit_bar=i,
        entry_price=pos["entry_price"],
        exit_price=exit_price,
        size=pos["size"],
        direction="long",
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
