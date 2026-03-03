"""
Unit tests for the DryRunBroker execution layer.
All tests use a temporary file so they never touch state/virtual_account.json.
"""
from __future__ import annotations

import pytest

from execution.broker import TradeOrder
from execution.dry_run import DryRunBroker


# ── Helpers ───────────────────────────────────────────────────────────────────

def _broker(tmp_path) -> DryRunBroker:
    return DryRunBroker(state_path=str(tmp_path / "account.json"))


def _order(
    limit_price: float = 1.0,
    stop_price:  float = 0.95,
    take_profit: float = 1.10,
    qty:         float = 100.0,
    risk_usd:    float = 5.0,
) -> TradeOrder:
    return TradeOrder(
        strategy_id="test",
        symbol="TEST",
        side="buy",
        qty=qty,
        order_type="limit",
        limit_price=limit_price,
        stop_price=stop_price,
        take_profit=take_profit,
        risk_usd=risk_usd,
    )


# ── Fill price tests ──────────────────────────────────────────────────────────

def test_dry_run_fill_at_slippage_price(tmp_path):
    """Filled price must be limit_price * (1 + SLIPPAGE_PCT)."""
    broker = _broker(tmp_path)
    fill = broker.place_order(_order(limit_price=1.0))
    expected = 1.0 * (1.0 + DryRunBroker.SLIPPAGE_PCT)
    assert abs(fill.filled_price - expected) < 1e-9
    assert fill.status == "filled"


# ── Commission tests ──────────────────────────────────────────────────────────

def test_dry_run_commission_minimum(tmp_path):
    """Very small trades must pay the flat $1 minimum commission."""
    broker = _broker(tmp_path)
    # qty=0.001, price≈1.0 → trade_value≈0.001 → pct_commission≈0.0000005 → flat wins
    fill = broker.place_order(_order(limit_price=1.0, qty=0.001))
    assert fill.commission == pytest.approx(DryRunBroker.COMMISSION_FLAT, rel=1e-6)


def test_dry_run_commission_percentage(tmp_path):
    """Large trades must pay the percentage commission (0.05% of value)."""
    broker = _broker(tmp_path)
    # qty=10_000, price=1.0 → value=10_000 → pct_commission=$5 > flat $1
    fill = broker.place_order(_order(limit_price=1.0, qty=10_000))
    expected_value  = 1.0 * (1 + DryRunBroker.SLIPPAGE_PCT) * 10_000
    expected_comm   = expected_value * DryRunBroker.COMMISSION_PCT
    assert expected_comm > DryRunBroker.COMMISSION_FLAT  # confirm test is meaningful
    assert fill.commission == pytest.approx(expected_comm, rel=1e-6)


# ── Exit tests ────────────────────────────────────────────────────────────────

def test_dry_run_stop_loss_triggers(tmp_path):
    """A price below the stop level must close the position."""
    broker = _broker(tmp_path)
    broker.place_order(_order(stop_price=0.95, take_profit=1.10))
    assert len(broker.get_open_positions()) == 1

    fills = broker.check_exits(current_price=0.90)   # below stop
    assert len(fills) == 1
    assert fills[0].filled_price == pytest.approx(0.95)
    assert len(broker.get_open_positions()) == 0


def test_dry_run_take_profit_triggers(tmp_path):
    """A price above the take-profit level must close the position."""
    broker = _broker(tmp_path)
    broker.place_order(_order(stop_price=0.95, take_profit=1.10))

    fills = broker.check_exits(current_price=1.15)   # above TP
    assert len(fills) == 1
    assert fills[0].filled_price == pytest.approx(1.10)
    assert len(broker.get_open_positions()) == 0


# ── Equity tests ──────────────────────────────────────────────────────────────

def test_dry_run_equity_reduces_on_loss(tmp_path):
    """Equity must decrease after a stop-loss exit."""
    broker    = _broker(tmp_path)
    start_eq  = broker.get_equity()
    broker.place_order(_order(limit_price=1.0, stop_price=0.90, take_profit=1.20, qty=100.0))
    broker.check_exits(current_price=0.85)   # triggers stop at 0.90
    # PnL = (0.90 - entry) * qty = negative
    assert broker.get_equity() < start_eq


# ── Daily halt tests ──────────────────────────────────────────────────────────

def test_dry_run_daily_halt_check(tmp_path):
    """is_daily_halt() must return True when daily_pnl is worse than -5% of start equity."""
    broker = _broker(tmp_path)
    # Manually inject a large daily loss
    broker._state["daily_pnl"]          = -600.0   # 6% of 10_000
    broker._state["start_of_day_equity"] = 10_000.0
    assert broker.is_daily_halt() is True


def test_dry_run_no_halt_on_small_loss(tmp_path):
    """is_daily_halt() must return False when daily_pnl is within tolerance."""
    broker = _broker(tmp_path)
    broker._state["daily_pnl"]          = -400.0   # 4% — below the 5% threshold
    broker._state["start_of_day_equity"] = 10_000.0
    assert broker.is_daily_halt() is False


# ── Per-strategy safety tests ─────────────────────────────────────────────────

def test_dry_run_strategy_daily_halt_at_3pct(tmp_path):
    """check_strategy_daily_halt() returns True when that strategy is down >3% of equity."""
    broker = _broker(tmp_path)
    equity = 10_000.0
    broker._state["equity"] = equity
    # Inject a per-strategy loss of 3.1% of equity = $310
    broker._state.setdefault("strategy_daily_pnl", {})["strat_A"] = -310.0
    assert broker.check_strategy_daily_halt("strat_A", equity) is True


def test_dry_run_strategy_daily_halt_not_triggered(tmp_path):
    """check_strategy_daily_halt() returns False when strategy loss is below 3%."""
    broker = _broker(tmp_path)
    equity = 10_000.0
    broker._state["equity"] = equity
    broker._state.setdefault("strategy_daily_pnl", {})["strat_A"] = -200.0  # 2% -- under limit
    assert broker.check_strategy_daily_halt("strat_A", equity) is False


def test_dry_run_combined_exposure_blocks_when_over_6pct(tmp_path):
    """check_combined_exposure() returns True when total open risk would exceed 6%."""
    broker = _broker(tmp_path)
    equity = 10_000.0
    # Two existing positions each risking $250 = 2.5% each -> 5% combined open
    broker._state["positions"] = [
        {"strategy_id": "s1", "risk_usd": 250.0, "symbol": "T", "qty": 1, "entry_price": 1,
         "stop_price": 0.9, "take_profit": 1.2, "side": "buy"},
        {"strategy_id": "s2", "risk_usd": 250.0, "symbol": "T", "qty": 1, "entry_price": 1,
         "stop_price": 0.9, "take_profit": 1.2, "side": "buy"},
    ]
    # Adding another $200 (2%) would push total to 7% -> should be blocked
    assert broker.check_combined_exposure(new_risk_usd=200.0, equity=equity) is True


def test_dry_run_combined_exposure_allows_when_under_6pct(tmp_path):
    """check_combined_exposure() returns False when total open risk stays under 6%."""
    broker = _broker(tmp_path)
    equity = 10_000.0
    # One position at $200 (2% of equity)
    broker._state["positions"] = [
        {"strategy_id": "s1", "risk_usd": 200.0, "symbol": "T", "qty": 1, "entry_price": 1,
         "stop_price": 0.9, "take_profit": 1.2, "side": "buy"},
    ]
    # Adding $300 (3%) -> total 5% -> under 6% -> allowed
    assert broker.check_combined_exposure(new_risk_usd=300.0, equity=equity) is False


def test_dry_run_strategy_pnl_tracked_on_entry(tmp_path):
    """Per-strategy daily P&L should decrease by commission on entry."""
    broker = _broker(tmp_path)
    order = TradeOrder(
        strategy_id="strat_X", symbol="T", side="buy", qty=100.0,
        order_type="limit", limit_price=1.0,
        stop_price=0.95, take_profit=1.10, risk_usd=5.0,
    )
    broker.place_order(order)
    pnl = broker.get_strategy_daily_pnl("strat_X")
    assert pnl < 0.0  # commission deducted


def test_dry_run_reset_daily_clears_strategy_pnl(tmp_path):
    """reset_daily() should zero out per-strategy P&L tracking."""
    broker = _broker(tmp_path)
    broker._state.setdefault("strategy_daily_pnl", {})["s1"] = -500.0
    broker.reset_daily()
    assert broker.get_strategy_daily_pnl("s1") == 0.0


# ── Exit log strategy_id test ─────────────────────────────────────────────────

def test_dry_run_exit_log_has_strategy_id(tmp_path):
    """Exit trade log entries must include strategy_id so the Coach can split by strategy."""
    broker = _broker(tmp_path)
    broker.place_order(_order(stop_price=0.95, take_profit=1.10))
    broker.check_exits(current_price=0.90)   # triggers stop
    exits = [t for t in broker.get_trade_log() if t.get("type") == "exit"]
    assert len(exits) == 1
    assert "strategy_id" in exits[0]
