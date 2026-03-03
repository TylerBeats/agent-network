"""
Base broker abstractions — shared dataclasses and ABC.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class TradeOrder:
    strategy_id: str
    symbol: str            # token address or ticker
    side: str              # "buy" | "sell"
    qty: float             # units
    order_type: str        # "limit" | "market"
    limit_price: float | None
    stop_price: float      # stop-loss level placed immediately
    take_profit: float     # take-profit level placed immediately
    risk_usd: float        # USD amount risked on this trade


@dataclass
class FillResult:
    order_id: str
    filled_price: float
    filled_qty: float
    commission: float
    timestamp: int         # Unix ms
    status: str            # "filled" | "rejected" | "partial"


class BaseBroker(ABC):
    @abstractmethod
    def get_equity(self) -> float: ...

    @abstractmethod
    def place_order(self, order: TradeOrder) -> FillResult: ...

    @abstractmethod
    def get_open_positions(self) -> list[dict]: ...

    @abstractmethod
    def close_position(self, position_id: str) -> FillResult: ...

    @abstractmethod
    def get_daily_pnl(self) -> float: ...
