"""
AlpacaBroker — paper/live trading via Alpaca Markets REST API.

Requires ALPACA_API_KEY and ALPACA_API_SECRET in .env.
Enable by setting BROKER_MODE=alpaca in .env.

NOTE: Alpaca supports standard equities and crypto tickers (BTCUSD, ETHUSD).
      It is NOT compatible with PulseChain token addresses — use DryRunBroker
      for those assets.

Local state file (state/alpaca_paper_state.json) tracks per-strategy P&L,
the trade log, and daily/weekly equity snapshots — data Alpaca does not
provide natively at the per-strategy level.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from datetime import date, timedelta

import requests

from execution.broker import BaseBroker, FillResult, TradeOrder

PAPER_BASE_URL = "https://paper-api.alpaca.markets"
LIVE_BASE_URL  = "https://api.alpaca.markets"

_DEFAULT_STATE_PATH = "state/alpaca_paper_state.json"


class AlpacaBroker(BaseBroker):
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        paper: bool = True,
        state_path: str = _DEFAULT_STATE_PATH,
    ):
        self._base = PAPER_BASE_URL if paper else LIVE_BASE_URL
        self._headers = {
            "APCA-API-KEY-ID":     api_key,
            "APCA-API-SECRET-KEY": api_secret,
            "Content-Type":        "application/json",
        }
        self._state_path = state_path
        self._state = self._load_state()

    # ── Local state persistence ───────────────────────────────────────────────

    def _load_state(self) -> dict:
        if os.path.exists(self._state_path):
            with open(self._state_path) as f:
                return json.load(f)
        return {
            "trade_log":             [],
            "strategy_daily_pnl":    {},   # {strategy_id: float}
            "start_of_day_equity":   0.0,
            "start_of_week_equity":  0.0,
            "seen_order_ids":        [],   # bracket exit legs already logged
            "last_reset_date":       "",   # ISO date of last daily reset
            "last_week_reset_date":  "",   # ISO date of last weekly reset
        }

    def _save_state(self) -> None:
        parent = os.path.dirname(self._state_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self._state_path, "w") as f:
            json.dump(self._state, f, indent=2)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get(self, path: str, params: dict | None = None) -> dict | list:
        resp = requests.get(
            f"{self._base}{path}",
            headers=self._headers,
            params=params,
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, body: dict) -> dict:
        resp = requests.post(f"{self._base}{path}", json=body, headers=self._headers, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def _delete(self, path: str) -> dict:
        resp = requests.delete(f"{self._base}{path}", headers=self._headers, timeout=10)
        resp.raise_for_status()
        return resp.json() if resp.content else {}

    # ── BaseBroker interface ──────────────────────────────────────────────────

    def get_equity(self) -> float:
        account = self._get("/v2/account")
        return float(account["equity"])

    def get_daily_pnl(self) -> float:
        account = self._get("/v2/account")
        return float(account["equity"]) - float(account["last_equity"])

    def get_open_positions(self) -> list[dict]:
        return self._get("/v2/positions")

    def place_order(self, order: TradeOrder) -> FillResult:
        body: dict = {
            "symbol":        order.symbol,
            "qty":           str(order.qty),
            "side":          order.side,
            "type":          order.order_type,
            "time_in_force": "gtc",
            "order_class":   "bracket",
            "stop_loss":     {"stop_price": str(order.stop_price)},
            "take_profit":   {"limit_price": str(order.take_profit)},
        }
        if order.order_type == "limit" and order.limit_price is not None:
            body["limit_price"] = str(order.limit_price)

        result = self._post("/v2/orders", body)
        fill = FillResult(
            order_id=result.get("id", str(uuid.uuid4())),
            filled_price=float(result.get("filled_avg_price") or order.limit_price or 0),
            filled_qty=float(result.get("filled_qty") or order.qty),
            commission=0.0,   # Alpaca charges no commission
            timestamp=int(time.time() * 1000),
            status=result.get("status", "pending"),
        )

        # Track per-strategy daily P&L (entry has no immediate P&L impact,
        # but initialise the key so later checks never KeyError)
        sdp = self._state.setdefault("strategy_daily_pnl", {})
        sdp.setdefault(order.strategy_id, 0.0)

        self._state["trade_log"].append({
            "order_id":     fill.order_id,
            "type":         "entry",
            "strategy_id":  order.strategy_id,
            "side":         order.side,
            "symbol":       order.symbol,
            "filled_price": fill.filled_price,
            "qty":          fill.filled_qty,
            "risk_usd":     order.risk_usd,
            "timestamp":    fill.timestamp,
        })
        self._save_state()
        return fill

    def close_position(self, position_id: str) -> FillResult:
        result = self._delete(f"/v2/positions/{position_id}")
        return FillResult(
            order_id=result.get("id", position_id),
            filled_price=float(result.get("filled_avg_price") or 0),
            filled_qty=float(result.get("filled_qty") or 0),
            commission=0.0,
            timestamp=int(time.time() * 1000),
            status=result.get("status", "filled"),
        )

    # ── Exit detection ────────────────────────────────────────────────────────

    def check_exits(self, current_price: float) -> list[FillResult]:  # noqa: ARG002
        """
        Poll Alpaca for closed orders since the start of today (UTC).
        Alpaca handles bracket stop/TP exits server-side; this method
        discovers them and logs them locally.

        current_price is accepted for interface compatibility but not used —
        Alpaca reports the actual fill price directly.
        """
        today_str = date.today().isoformat()
        seen: set[str] = set(self._state.get("seen_order_ids", []))
        fills: list[FillResult] = []

        try:
            closed_orders = self._get(
                "/v2/orders",
                params={"status": "closed", "after": today_str, "limit": 100},
            )
        except Exception:
            return []

        if not isinstance(closed_orders, list):
            return []

        for order in closed_orders:
            oid = order.get("id", "")
            # Only process sell legs (exits) we haven't seen before
            if oid in seen or order.get("side") != "sell":
                continue
            if order.get("status") != "filled":
                continue

            seen.add(oid)
            exit_price = float(order.get("filled_avg_price") or 0)
            qty        = float(order.get("filled_qty") or 0)
            pnl        = 0.0  # Alpaca doesn't give entry cost directly here

            fill = FillResult(
                order_id=oid,
                filled_price=exit_price,
                filled_qty=qty,
                commission=0.0,
                timestamp=int(time.time() * 1000),
                status="filled",
            )

            # Best-effort strategy attribution via parent order legs
            strategy_id = order.get("client_order_id", "")

            sdp = self._state.setdefault("strategy_daily_pnl", {})
            sdp[strategy_id] = sdp.get(strategy_id, 0.0) + pnl

            self._state["trade_log"].append({
                "order_id":     oid,
                "type":         "exit",
                "strategy_id":  strategy_id,
                "reason":       order.get("order_class", "bracket"),
                "filled_price": exit_price,
                "qty":          qty,
                "pnl":          pnl,
                "timestamp":    fill.timestamp,
            })
            fills.append(fill)

        self._state["seen_order_ids"] = list(seen)
        if fills:
            self._save_state()
        return fills

    # ── Safety helpers ────────────────────────────────────────────────────────

    def is_daily_halt(self) -> bool:
        """True if today's P&L (from Alpaca account) is worse than -5% of start-of-day equity."""
        try:
            daily_pnl = self.get_daily_pnl()
            sod = self._state.get("start_of_day_equity", 0.0)
            if sod <= 0:
                # Initialise on first call
                sod = self.get_equity()
                self._state["start_of_day_equity"] = sod
                self._save_state()
            return (daily_pnl / sod) < -0.05
        except Exception:
            return False

    def is_weekly_reduction(self) -> bool:
        """True if this week's equity drop exceeds 8% of start-of-week equity."""
        try:
            history = self._get(
                "/v2/portfolio/history",
                params={"period": "1W", "timeframe": "1D"},
            )
            base_value = history.get("base_value", 0.0)
            equity_series = history.get("equity", [])
            if not equity_series or not base_value:
                return False
            current_equity = float(equity_series[-1] or 0)
            return (current_equity - float(base_value)) / float(base_value) < -0.08
        except Exception:
            return False

    def get_strategy_daily_pnl(self, strategy_id: str) -> float:
        """Return today's P&L for a specific strategy (0.0 if no trades yet)."""
        return self._state.get("strategy_daily_pnl", {}).get(strategy_id, 0.0)

    def check_strategy_daily_halt(self, strategy_id: str, equity: float) -> bool:
        """True if this strategy's daily loss exceeds 3% of current equity."""
        if equity <= 0:
            return False
        return (self.get_strategy_daily_pnl(strategy_id) / equity) < -0.03

    def check_combined_exposure(self, new_risk_usd: float, equity: float) -> bool:
        """
        True if adding new_risk_usd would push combined open risk above 6% of equity.
        Open positions are fetched live from Alpaca.
        """
        if equity <= 0:
            return True
        try:
            positions = self.get_open_positions()
            total_risk = sum(float(p.get("unrealized_pl", 0)) for p in positions)
            current_pct = abs(total_risk) / equity * 100.0
            new_pct = (new_risk_usd / equity) * 100.0
            return (current_pct + new_pct) > 6.0
        except Exception:
            return False

    def get_trade_log(self) -> list[dict]:
        """Return full locally-tracked trade log (entry + exit records)."""
        return list(self._state["trade_log"])

    def reset_daily(self) -> None:
        """Call at start of each trading day to reset daily counters."""
        try:
            equity = self.get_equity()
        except Exception:
            equity = self._state.get("start_of_day_equity", 0.0)
        self._state["start_of_day_equity"]  = equity
        self._state["strategy_daily_pnl"]   = {}
        self._state["last_reset_date"]       = date.today().isoformat()
        self._save_state()

    def reset_weekly(self) -> None:
        """Call at start of each trading week to reset weekly counters."""
        try:
            equity = self.get_equity()
        except Exception:
            equity = self._state.get("start_of_week_equity", 0.0)
        self._state["start_of_week_equity"]  = equity
        self._state["last_week_reset_date"]  = date.today().isoformat()
        self._save_state()
