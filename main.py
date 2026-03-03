import logging

from anthropic import Anthropic

from agents.backtester import BacktesterAgent
from agents.coordinator import CoordinatorAgent
from agents.strategy_builder import StrategyBuilderAgent
from agents.trader import TraderAgent
from agents.trading_coach import TradingCoachAgent
from config.settings import (
    ANTHROPIC_API_KEY,
    ASSET_BUCKET_SECONDS,
    ASSET_CHAIN,
    ASSET_TOKEN_ADDRESS,
    LOG_LEVEL,
)
from core.network import AgentNetwork
from cycle.orchestrator import run_cycle
from agents.asset_profile import build_asset_key
from cycle.state import load_state, save_state, state_path_for

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(levelname)s %(name)s: %(message)s",
)


def build_network() -> AgentNetwork:
    client  = Anthropic(api_key=ANTHROPIC_API_KEY)
    network = AgentNetwork()

    agents = [
        StrategyBuilderAgent(client),
        BacktesterAgent(client),
        TraderAgent(client),
        TradingCoachAgent(client),
    ]
    coordinator = CoordinatorAgent(client, worker_names=[a.name for a in agents])

    network.register(coordinator)
    for agent in agents:
        network.register(agent)

    return network


def main():
    # ── Asset configuration ───────────────────────────────────────────────────
    asset_key  = build_asset_key(ASSET_CHAIN, ASSET_TOKEN_ADDRESS)
    s_path     = state_path_for(asset_key)

    asset = {
        "chain":          ASSET_CHAIN,
        "token_address":  ASSET_TOKEN_ADDRESS,
        "bucket_seconds": ASSET_BUCKET_SECONDS,
    }

    # ── Load persistent state (asset-specific file) ───────────────────────────
    state = load_state(s_path)
    state.asset_key = asset_key   # ensure field is set on fresh states

    # ── Build the agent network ───────────────────────────────────────────────
    network = build_network()

    # ── Run one complete autonomous cycle ─────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Autonomous Trading Network — Cycle {state.cycle_number + 1}")
    print(f"  Asset: {ASSET_CHAIN} / {ASSET_TOKEN_ADDRESS or '(not configured)'}")
    print(f"  State: {s_path}")
    print(f"{'='*60}\n")

    state = run_cycle(state, network, asset, state_path=s_path)

    print(f"\n{'='*60}")
    print(f"  Cycle {state.cycle_number} complete")
    print(f"  Asset: {asset_key}  |  Strategy log: {len(state.asset_strategy_log)} entries")
    print(f"  Risk multiplier: {state.risk_multiplier:.2f}x")
    print(f"  Retired patterns: {len(state.retired_patterns)}")
    print(f"  Total trades logged: {len(state.trade_log)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
