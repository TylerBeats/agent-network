import logging
import os
from agents.backtester import BacktesterAgent
from agents.coordinator import CoordinatorAgent
from agents.strategy_builder import StrategyBuilderAgent
from agents.trader import TraderAgent
from agents.trading_coach import TradingCoachAgent
from config.settings import ANTHROPIC_API_KEY, ASSET_BUCKET_SECONDS, ASSET_CHAIN, ASSET_TOKEN_ADDRESS, LOG_LEVEL
from core.network import AgentNetwork
from cycle.orchestrator import run_cycle
from agents.asset_profile import build_asset_key
from cycle.state import load_state, save_state, state_path_for

logging.basicConfig(level=getattr(logging, LOG_LEVEL), format="%(levelname)s %(name)s: %(message)s")

def build_network():
    if os.getenv("LOCAL_LLM", "false").lower() == "true":
        from openai import OpenAI
        client = OpenAI(base_url=os.getenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:8080/v1"), api_key="local", timeout=600)
        print("[LOCAL LLM] Zenith Connected")
    else:
        from anthropic import Anthropic
        client = Anthropic(api_key=ANTHROPIC_API_KEY)
        print("[CLOUD] Using Anthropic API")
    network = AgentNetwork()
    agents = [StrategyBuilderAgent(client), BacktesterAgent(client), TraderAgent(client), TradingCoachAgent(client)]
    coordinator = CoordinatorAgent(client, worker_names=[a.name for a in agents])
    network.register(coordinator)
    for agent in agents:
        network.register(agent)
    return network

def main():
    asset_key = build_asset_key(ASSET_CHAIN, ASSET_TOKEN_ADDRESS)
    s_path = state_path_for(asset_key)
    asset = {"chain": ASSET_CHAIN, "token_address": ASSET_TOKEN_ADDRESS, "bucket_seconds": ASSET_BUCKET_SECONDS}
    state = load_state(s_path)
    state.asset_key = asset_key
    network = build_network()
    print(f"\n{'='*60}\n  Cycle {state.cycle_number + 1} | {ASSET_CHAIN}\n{'='*60}\n")
    state = run_cycle(state, network, asset, state_path=s_path)
    print(f"\n{'='*60}\n  Cycle {state.cycle_number} complete\n{'='*60}\n")

if __name__ == "__main__":
    main()
