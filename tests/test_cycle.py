"""
Unit tests for cycle state persistence (load/save round-trip).
"""
from __future__ import annotations

import pytest

from agents.asset_profile import build_asset_key, get_asset_class, get_asset_hints, summarise_strategy_log
from cycle.state import CycleState, load_state, save_state, state_path_for


def test_cycle_state_default_values():
    """A freshly constructed CycleState must have zero/empty/default values."""
    state = CycleState()
    assert state.cycle_number == 0
    assert state.risk_multiplier == 1.0
    assert state.active_strategy is None
    assert state.trade_log == []
    assert state.retired_patterns == []
    assert state.last_feedback is None
    assert state.asset_key == ""
    assert state.asset_strategy_log == []


def test_cycle_state_roundtrip(tmp_path):
    """Saving and reloading a CycleState must reproduce all fields exactly."""
    path = str(tmp_path / "cycle_state.json")

    original = CycleState(
        cycle_number=7,
        active_strategy={"strategy_id": "s1", "sharpe": 1.8},
        trade_log=[{"type": "exit", "pnl": 120.0}],
        weekly_reviews=[{"week": "2026-W03", "pnl_pct": 0.02}],
        monthly_reviews=[{"month": "2026-01", "risk_adjustment": 1.25}],
        performance_history=[{"strategy_id": "s1", "score": 72.4}],
        retired_patterns=[{"primary_indicator_type": "RSI", "entry_trigger": "RSI_OVERSOLD"}],
        last_feedback={"best_indicator_types": ["MACD"]},
        risk_multiplier=0.75,
        asset_key="gold",
        asset_strategy_log=[
            {"cycle": 1, "primary_indicator": "EMA", "entry_trigger": "PRICE_ABOVE_EMA",
             "result": "passed", "score": 72.4},
            {"cycle": 1, "primary_indicator": "RSI", "entry_trigger": "RSI_OVERSOLD",
             "result": "failed", "failure_reason": "Win rate 33% below minimum"},
        ],
    )

    save_state(original, path)
    loaded = load_state(path)

    assert loaded.cycle_number == original.cycle_number
    assert loaded.risk_multiplier == pytest.approx(original.risk_multiplier)
    assert loaded.active_strategy == original.active_strategy
    assert loaded.trade_log == original.trade_log
    assert loaded.weekly_reviews == original.weekly_reviews
    assert loaded.monthly_reviews == original.monthly_reviews
    assert loaded.performance_history == original.performance_history
    assert loaded.retired_patterns == original.retired_patterns
    assert loaded.last_feedback == original.last_feedback
    assert loaded.asset_key == "gold"
    assert len(loaded.asset_strategy_log) == 2
    assert loaded.asset_strategy_log[0]["primary_indicator"] == "EMA"
    assert loaded.asset_strategy_log[1]["result"] == "failed"


def test_load_state_missing_file(tmp_path):
    """load_state() must return a fresh CycleState if the file does not exist."""
    path = str(tmp_path / "nonexistent.json")
    state = load_state(path)
    assert isinstance(state, CycleState)
    assert state.cycle_number == 0


# ── asset_profile tests ───────────────────────────────────────────────────────

def test_asset_key_fixed_symbol_chain():
    """Chains with no token address should produce just the chain name."""
    assert build_asset_key("gold", "") == "gold"
    assert build_asset_key("spx500", "") == "spx500"
    assert build_asset_key("nasdaq", "") == "nasdaq"


def test_asset_key_with_contract_address():
    """Chains with a hex address should produce chain + first 10 chars of address."""
    key = build_asset_key("pulsechain", "0x2b591e99afe9f32eaa6214f7b7629768c40eeb39")
    assert key == "pulsechain_0x2b591e99"


def test_asset_key_with_ticker():
    """Chains with a user-supplied ticker (stocks) should embed the ticker."""
    key = build_asset_key("stocks", "AAPL")
    assert key == "stocks_AAPL"


def test_state_path_for_asset():
    """state_path_for() should produce a path under the state/ directory."""
    assert state_path_for("gold") == "state/gold_cycle_state.json"
    assert state_path_for("spx500_full") == "state/spx500_full_cycle_state.json"


def test_asset_class_mapping():
    """Known chains must map to their expected asset classes."""
    assert get_asset_class("gold")         == "commodity"
    assert get_asset_class("spx500")       == "equity_etf"
    assert get_asset_class("spx500_full")  == "equity_index"
    assert get_asset_class("pulsechain")   == "crypto"
    assert get_asset_class("ethereum")     == "crypto"
    assert get_asset_class("nasdaq")       == "equity_etf"
    assert get_asset_class("unknown_chain") == "crypto"   # fallback


def test_asset_hints_commodity_avoids_mean_reversion():
    """Gold hints must list mean-reversion triggers in avoid_triggers."""
    hints = get_asset_hints("gold")
    assert "RSI_OVERSOLD" in hints["avoid_triggers"]
    assert "BB_LOWER_TOUCH" in hints["avoid_triggers"]
    assert "CCI_OVERSOLD" in hints["avoid_triggers"]
    assert "EMA" in hints["preferred_indicators"]


def test_asset_hints_equity_index_volume_warning():
    """^GSPC hints must warn about synthetic volume."""
    hints = get_asset_hints("spx500_full")
    assert "synthetic" in hints["volume_note"].lower()


def test_summarise_strategy_log_empty():
    """Empty log should return an empty string."""
    assert summarise_strategy_log([]) == ""


def test_summarise_strategy_log_counts():
    """summarise_strategy_log should correctly tally pass/fail counts."""
    log = [
        {"primary_indicator": "EMA", "entry_trigger": "PRICE_ABOVE_EMA", "result": "passed", "score": 70},
        {"primary_indicator": "EMA", "entry_trigger": "PRICE_ABOVE_EMA", "result": "passed", "score": 80},
        {"primary_indicator": "RSI", "entry_trigger": "RSI_OVERSOLD",    "result": "failed",
         "failure_reason": "Win rate too low", "score": 0},
    ]
    summary = summarise_strategy_log(log)
    assert "EMA" in summary
    assert "RSI" in summary
    assert "2/2" in summary   # EMA passed 2/2
    assert "0/1" in summary   # RSI passed 0/1
