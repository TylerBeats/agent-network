"""
AlpacaBroker — paper/live trading via Alpaca Markets REST API.

Requires ALPACA_API_KEY and ALPACA_API_SECRET in .env.
Enable by setting BROKER_MODE=alpaca in .env.

NOTE: Alpaca supports standard equities and crypto tickers (BTCUSD, ETHUSD).
      It is NOT compatible with PulseChain token addresses — use DryRunBroker
      for those assets.
"""
from __future__ import annotations

import time
import uuid

import requests

from execution.broker import BaseBroker, FillResult, TradeOrder

PAPER_BASE_URL = "https://paper-api.alpaca.markets"
LIVE_BASE_URL  = "https://api.alpaca.markets"


class AlpacaBroker(BaseBroker):
    def __init__(self, api_key: str, api_secret: str, paper: bool = True):
        self._base = PAPER_BASE_URL if paper else LIVE_BASE_URL
        self._headers = {
            "APCA-API-KEY-ID":     api_key,
            "APCA-API-SECRET-KEY": api_secret,
            "Content-Type":        "application/json",
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get(self, path: str) -> dict:
        resp = requests.get(f"{self._base}{path}", headers=self._headers, timeout=10)
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
        return FillResult(
            order_id=result.get("id", str(uuid.uuid4())),
            filled_price=float(result.get("filled_avg_price") or order.limit_price or 0),
            filled_qty=float(result.get("filled_qty") or order.qty),
            commission=0.0,   # Alpaca charges no commission
            timestamp=int(time.time() * 1000),
            status=result.get("status", "pending"),
        )

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
