"""
DryRunBroker — simulates trade execution using real candle prices.
State is persisted to state/virtual_account.json between runs.
"""
from __future__ import annotations

import json
import os
import time
import uuid

from execution.broker import BaseBroker, FillResult, TradeOrder


class DryRunBroker(BaseBroker):
    """
    Paper-trading broker that simulates fills at limit_price + SLIPPAGE_PCT.
    Commissions use a tiered model: flat $1 minimum or 0.05% of trade value.
    All state is persisted to a JSON file so equity and positions survive
    between process invocations.
    """

    INITIAL_EQUITY   = 10_000.0
    SLIPPAGE_PCT     = 0.001    # 0.1% slippage on entry
    COMMISSION_FLAT  = 1.0      # minimum commission per trade
    COMMISSION_PCT   = 0.0005   # 0.05% of trade value

    def __init__(self, state_path: str = "state/virtual_account.json"):
        self._path = state_path
        self._state = self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> dict:
        if os.path.exists(self._path):
            with open(self._path) as f:
                return json.load(f)
        return {
            "equity":                self.INITIAL_EQUITY,
            "start_of_day_equity":   self.INITIAL_EQUITY,
            "start_of_week_equity":  self.INITIAL_EQUITY,
            "positions":             [],
            "trade_log":             [],
            "daily_pnl":             0.0,
            "weekly_pnl":            0.0,
            "strategy_daily_pnl":    {},   # {strategy_id: daily_pnl_float}
        }

    def save(self) -> None:
        parent = os.path.dirname(self._path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(self._state, f, indent=2)

    # ── BaseBroker interface ──────────────────────────────────────────────────

    def get_equity(self) -> float:
        return self._state["equity"]

    def get_daily_pnl(self) -> float:
        return self._state["daily_pnl"]

    def get_open_positions(self) -> list[dict]:
        return list(self._state["positions"])

    def place_order(self, order: TradeOrder) -> FillResult:
        """Fill at limit_price * (1 + SLIPPAGE_PCT), deduct commission."""
        base_price = order.limit_price if order.limit_price is not None else order.stop_price
        filled_price = base_price * (1.0 + self.SLIPPAGE_PCT)
        trade_value  = filled_price * order.qty
        commission   = max(self.COMMISSION_FLAT, trade_value * self.COMMISSION_PCT)

        self._state["equity"]                          -= commission
        self._state["daily_pnl"]                       -= commission
        self._state["weekly_pnl"]                       = self._state.get("weekly_pnl", 0.0) - commission
        sdp = self._state.setdefault("strategy_daily_pnl", {})
        sdp[order.strategy_id] = sdp.get(order.strategy_id, 0.0) - commission

        position = {
            "id":           str(uuid.uuid4()),
            "strategy_id":  order.strategy_id,
            "symbol":       order.symbol,
            "side":         order.side,
            "qty":          order.qty,
            "entry_price":  filled_price,
            "stop_price":   order.stop_price,
            "take_profit":  order.take_profit,
            "risk_usd":     order.risk_usd,
        }
        self._state["positions"].append(position)

        fill = FillResult(
            order_id=position["id"],
            filled_price=filled_price,
            filled_qty=order.qty,
            commission=commission,
            timestamp=int(time.time() * 1000),
            status="filled",
        )
        self._state["trade_log"].append({
            "order_id":     fill.order_id,
            "type":         "entry",
            "side":         order.side,
            "filled_price": fill.filled_price,
            "qty":          fill.filled_qty,
            "commission":   fill.commission,
            "risk_usd":     order.risk_usd,
            "timestamp":    fill.timestamp,
        })
        self.save()
        return fill

    def check_exits(self, current_price: float) -> list[FillResult]:
        """
        Check all open positions against current_price.
        Closes any that hit their stop-loss or take-profit level.
        Returns FillResult for each closed position.
        """
        fills: list[FillResult] = []
        remaining: list[dict] = []

        for pos in self._state["positions"]:
            hit_stop = current_price <= pos["stop_price"]
            hit_tp   = current_price >= pos["take_profit"]

            if hit_stop or hit_tp:
                exit_price  = pos["stop_price"] if hit_stop else pos["take_profit"]
                gross_pnl   = (exit_price - pos["entry_price"]) * pos["qty"]
                commission  = max(self.COMMISSION_FLAT, exit_price * pos["qty"] * self.COMMISSION_PCT)
                net_pnl     = gross_pnl - commission
                risk_usd    = pos.get("risk_usd", 0.0)
                r_multiple  = (net_pnl / risk_usd) if risk_usd > 0 else 0.0

                self._state["equity"]     += net_pnl
                self._state["daily_pnl"]  += net_pnl
                self._state["weekly_pnl"]  = self._state.get("weekly_pnl", 0.0) + net_pnl
                sdp = self._state.setdefault("strategy_daily_pnl", {})
                strat_id = pos.get("strategy_id", "")
                sdp[strat_id] = sdp.get(strat_id, 0.0) + net_pnl

                fill = FillResult(
                    order_id=pos["id"],
                    filled_price=exit_price,
                    filled_qty=pos["qty"],
                    commission=commission,
                    timestamp=int(time.time() * 1000),
                    status="filled",
                )
                self._state["trade_log"].append({
                    "order_id":     fill.order_id,
                    "type":         "exit",
                    "strategy_id":  pos.get("strategy_id", ""),
                    "reason":       "stop_loss" if hit_stop else "take_profit",
                    "filled_price": exit_price,
                    "qty":          pos["qty"],
                    "pnl":          net_pnl,
                    "commission":   commission,
                    "r_multiple":   r_multiple,
                    "timestamp":    fill.timestamp,
                })
                fills.append(fill)
            else:
                remaining.append(pos)

        self._state["positions"] = remaining
        if fills:
            self.save()
        return fills

    def close_position(self, position_id: str) -> FillResult:
        """Force-close a position by id (used for emergency exits)."""
        for i, pos in enumerate(self._state["positions"]):
            if pos["id"] == position_id:
                self._state["positions"].pop(i)
                fill = FillResult(
                    order_id=position_id,
                    filled_price=pos["entry_price"],
                    filled_qty=pos["qty"],
                    commission=self.COMMISSION_FLAT,
                    timestamp=int(time.time() * 1000),
                    status="filled",
                )
                self.save()
                return fill
        raise ValueError(f"Position {position_id} not found")

    # ── Safety helpers ────────────────────────────────────────────────────────

    def is_daily_halt(self) -> bool:
        """True if daily PnL is worse than -5% of start-of-day equity."""
        sod = self._state.get("start_of_day_equity", self.INITIAL_EQUITY)
        if sod <= 0:
            return False
        return (self._state["daily_pnl"] / sod) < -0.05

    def is_weekly_reduction(self) -> bool:
        """True if weekly PnL is worse than -8% of start-of-week equity."""
        sow = self._state.get("start_of_week_equity", self.INITIAL_EQUITY)
        if sow <= 0:
            return False
        return (self._state.get("weekly_pnl", 0.0) / sow) < -0.08

    def get_strategy_daily_pnl(self, strategy_id: str) -> float:
        """Return today's P&L for a specific strategy (0.0 if no trades yet)."""
        return self._state.get("strategy_daily_pnl", {}).get(strategy_id, 0.0)

    def check_strategy_daily_halt(self, strategy_id: str, equity: float) -> bool:
        """True if this strategy's daily loss exceeds 3% of current equity."""
        if equity <= 0:
            return False
        return (self.get_strategy_daily_pnl(strategy_id) / equity) < -0.03

    def get_combined_open_risk(self, equity: float) -> float:
        """Return sum of all open positions' risk_usd as a percentage of equity."""
        if equity <= 0:
            return 0.0
        total_risk = sum(p.get("risk_usd", 0.0) for p in self._state["positions"])
        return (total_risk / equity) * 100.0

    def check_combined_exposure(self, new_risk_usd: float, equity: float) -> bool:
        """True if adding new_risk_usd would push combined open risk above 6% of equity."""
        if equity <= 0:
            return True
        current_pct = self.get_combined_open_risk(equity)
        new_pct = (new_risk_usd / equity) * 100.0
        return (current_pct + new_pct) > 6.0

    def reset_daily(self) -> None:
        """Call at start of each trading day to reset the daily counters."""
        self._state["daily_pnl"]           = 0.0
        self._state["start_of_day_equity"] = self._state["equity"]
        self._state["strategy_daily_pnl"]  = {}
        self.save()

    def reset_weekly(self) -> None:
        """Call at start of each trading week to reset the weekly counters."""
        self._state["weekly_pnl"]            = 0.0
        self._state["start_of_week_equity"]  = self._state["equity"]
        self.save()

    def get_trade_log(self) -> list[dict]:
        """Return full trade log (entry + exit records)."""
        return list(self._state["trade_log"])
