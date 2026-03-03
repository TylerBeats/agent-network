"""
Cycle state — persistent JSON snapshot of the autonomous trading network's progress.

Saved to state/cycle_state.json after every cycle so the system can resume
after a restart without losing its performance history.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

STATE_DIR  = "state"
STATE_FILE = "state/cycle_state.json"


@dataclass
class CycleState:
    cycle_number: int = 0

    # Most recent winning strategy dict from the Backtester
    active_strategy: dict | None = None

    # All exit trade records from DryRunBroker (accumulated across cycles)
    trade_log: list[dict] = field(default_factory=list)

    # Historical weekly reviews (serialised WeeklyReview dicts)
    weekly_reviews: list[dict] = field(default_factory=list)

    # Historical monthly reviews (serialised MonthlyReview dicts)
    monthly_reviews: list[dict] = field(default_factory=list)

    # Full backtester winner dicts from all previous cycles
    performance_history: list[dict] = field(default_factory=list)

    # Patterns retired by the Trading Coach (avoid regenerating)
    retired_patterns: list[dict] = field(default_factory=list)

    # Latest FeedbackReport from the Trading Coach
    last_feedback: dict | None = None

    # Current position-size multiplier (set by Trading Coach monthly review)
    risk_multiplier: float = 1.0

    # Short key identifying which asset this state belongs to (e.g. "gold", "pulsechain_0x2b591e")
    asset_key: str = ""

    # Full record of every strategy evaluated by the Backtester for this asset,
    # across all cycles — used by the Strategy Builder to learn what works.
    asset_strategy_log: list[dict] = field(default_factory=list)

    # Top-3 winner dicts from the most recent Backtester run (ordered by score)
    active_strategies: list[dict] = field(default_factory=list)

    # Projected benchmarks stored from the most recent Backtester winner
    projected_avg_r: float = 0.0          # avg R-multiple the winning strategy expects per trade
    projected_win_rate: float = 0.0       # win rate the winning strategy achieved in backtest

    # MC 95th-pct drawdown per strategy_id (populated from backtester mc_p95_dd)
    mc_p95_dd_by_strategy: dict = field(default_factory=dict)

    # Compounding mode management
    months_at_good_tier: dict = field(default_factory=dict)      # {strategy_id: int}
    per_strategy_compounding: dict = field(default_factory=dict) # {strategy_id: "none"|"monthly"|"per_trade"}


def state_path_for(asset_key: str) -> str:
    """Return the JSON file path for the given asset key."""
    safe = asset_key.replace("/", "-").replace("\\", "-") or "default"
    return f"{STATE_DIR}/{safe}_cycle_state.json"


def load_state(path: str = STATE_FILE) -> CycleState:
    """
    Load CycleState from JSON file.
    Returns a fresh CycleState if the file does not exist.
    """
    if not os.path.exists(path):
        return CycleState()
    with open(path) as f:
        data = json.load(f)
    return CycleState(
        cycle_number=data.get("cycle_number", 0),
        active_strategy=data.get("active_strategy"),
        trade_log=data.get("trade_log", []),
        weekly_reviews=data.get("weekly_reviews", []),
        monthly_reviews=data.get("monthly_reviews", []),
        performance_history=data.get("performance_history", []),
        retired_patterns=data.get("retired_patterns", []),
        last_feedback=data.get("last_feedback"),
        risk_multiplier=float(data.get("risk_multiplier", 1.0)),
        asset_key=data.get("asset_key", ""),
        asset_strategy_log=data.get("asset_strategy_log", []),
        active_strategies=data.get("active_strategies", []),
        projected_avg_r=float(data.get("projected_avg_r", 0.0)),
        projected_win_rate=float(data.get("projected_win_rate", 0.0)),
        mc_p95_dd_by_strategy=data.get("mc_p95_dd_by_strategy", {}),
        months_at_good_tier=data.get("months_at_good_tier", {}),
        per_strategy_compounding=data.get("per_strategy_compounding", {}),
    )


def save_state(state: CycleState, path: str = STATE_FILE) -> None:
    """Serialise CycleState to JSON, creating the state/ directory if needed."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload = {
        "cycle_number":        state.cycle_number,
        "active_strategy":     state.active_strategy,
        "trade_log":           state.trade_log,
        "weekly_reviews":      state.weekly_reviews,
        "monthly_reviews":     state.monthly_reviews,
        "performance_history": state.performance_history,
        "retired_patterns":    state.retired_patterns,
        "last_feedback":       state.last_feedback,
        "risk_multiplier":     state.risk_multiplier,
        "asset_key":           state.asset_key,
        "asset_strategy_log":  state.asset_strategy_log,
        "active_strategies":        state.active_strategies,
        "projected_avg_r":          state.projected_avg_r,
        "projected_win_rate":       state.projected_win_rate,
        "mc_p95_dd_by_strategy":    state.mc_p95_dd_by_strategy,
        "months_at_good_tier":      state.months_at_good_tier,
        "per_strategy_compounding": state.per_strategy_compounding,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
