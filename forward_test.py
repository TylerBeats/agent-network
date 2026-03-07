"""
forward_test.py — Trader-only forward testing runner.

Runs the Trader agent against live/paper market prices using the active
strategies selected by the most recent full cycle (main.py).

Usage:
    python forward_test.py

Schedule this to run on each new candle close, matching the strategy's
candle period:
  - 4h candles  → run every 4 hours (e.g. via cron or Task Scheduler)
  - daily bars  → run once daily after market close

Prerequisites:
    Run `python main.py` at least once to populate active_strategies.
    The broker is selected via BROKER_MODE in .env:
        BROKER_MODE=dryrun   (default — DryRunBroker paper simulation)
        BROKER_MODE=alpaca   (AlpacaBroker — Alpaca paper account)
"""
from __future__ import annotations

import json
import logging
from datetime import date, timedelta

from anthropic import Anthropic

from agents.trader import TraderAgent
from agents.asset_profile import build_asset_key
from config.settings import (
    ANTHROPIC_API_KEY,
    ASSET_BUCKET_SECONDS,
    ASSET_CHAIN,
    ASSET_TOKEN_ADDRESS,
    LOG_LEVEL,
)
from core.message import Message, MessageType
from cycle.state import CycleState, load_state, save_state, state_path_for
from execution.dry_run import DryRunBroker

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _call_trader(trader: TraderAgent, payload: dict) -> dict:
    """Send a TASK to the Trader and return the parsed result dict."""
    msg = Message(
        sender="forward_test",
        recipient="trader",
        type=MessageType.TASK,
        content=json.dumps(payload),
        metadata={},
    )
    trader.receive(msg)
    responses = trader.process()
    for resp in responses:
        if resp.type == MessageType.RESULT:
            return json.loads(resp.content)
        if resp.type == MessageType.ERROR:
            logger.error("Trader returned error: %s", resp.content)
            return {}
    return {}


def _is_new_day(state: CycleState) -> bool:
    """True if today's UTC date is different from the last forward-test run date."""
    today = date.today().isoformat()
    return state.last_forward_test_date != today


def _is_new_week(state: CycleState) -> bool:
    """True if today is Monday and the last forward-test date was in a previous week."""
    today = date.today()
    if today.weekday() != 0:   # 0 = Monday
        return False
    if not state.last_forward_test_date:
        return True
    last = date.fromisoformat(state.last_forward_test_date)
    return (today - last) >= timedelta(days=1)


def main() -> None:
    asset_key = build_asset_key(ASSET_CHAIN, ASSET_TOKEN_ADDRESS)
    s_path    = state_path_for(asset_key)
    state     = load_state(s_path)

    print(f"\n{'='*60}")
    print(f"  Forward Test Runner")
    print(f"  Asset: {ASSET_CHAIN} / {ASSET_TOKEN_ADDRESS or '(not configured)'}")
    print(f"  State: {s_path}")
    print(f"{'='*60}\n")

    # -- Guard: require active strategies from a completed main.py cycle ---------
    if not state.active_strategies:
        print(
            "[forward_test] No active_strategies found in state.\n"
            "  Run `python main.py` first to select strategies via backtesting.\n"
            "  Exiting."
        )
        return

    # -- Daily / weekly broker resets --------------------------------------------
    from execution.dry_run import DryRunBroker as _DryRun
    from config.settings import BROKER_MODE, ALPACA_API_KEY, ALPACA_API_SECRET

    def _make_broker():
        if BROKER_MODE == "alpaca":
            from execution.alpaca import AlpacaBroker
            return AlpacaBroker(api_key=ALPACA_API_KEY, api_secret=ALPACA_API_SECRET, paper=True)
        return _DryRun()

    broker = _make_broker()

    if _is_new_week(state):
        broker.reset_weekly()
        print("[forward_test] Weekly counters reset (new week)")

    if _is_new_day(state):
        broker.reset_daily()
        print("[forward_test] Daily counters reset (new day)")

    # -- Build asset config dict -------------------------------------------------
    asset = {
        "chain":          ASSET_CHAIN,
        "token_address":  ASSET_TOKEN_ADDRESS,
        "bucket_seconds": ASSET_BUCKET_SECONDS,
    }

    # -- Build Trader payload (mirrors orchestrator Step 4) ----------------------
    trader_payload = {
        "strategies":               state.active_strategies,
        "strategy":                 state.active_strategies[0],   # backward compat
        "asset":                    asset,
        "risk_multiplier":          state.risk_multiplier,
        "per_strategy_adjustments": {},
        "per_strategy_compounding": state.per_strategy_compounding,
    }

    # -- Run Trader agent --------------------------------------------------------
    client  = Anthropic(api_key=ANTHROPIC_API_KEY)
    trader  = TraderAgent(client)

    print(f"[forward_test] Running Trader on {len(state.active_strategies)} active strateg"
          f"{'y' if len(state.active_strategies) == 1 else 'ies'}...")

    result = _call_trader(trader, trader_payload)

    if not result:
        print("[forward_test] Trader returned no result. Check logs for errors.")
        return

    # -- Accumulate exit trades into CycleState ----------------------------------
    new_exits = [t for t in result.get("trade_log", []) if t.get("type") == "exit"]
    state.trade_log.extend(new_exits)

    # -- Update last-run date and persist ----------------------------------------
    state.last_forward_test_date = date.today().isoformat()
    save_state(state, s_path)

    # -- Print summary -----------------------------------------------------------
    actions      = result.get("actions", [])
    equity       = result.get("equity", 0.0)
    daily_pnl    = result.get("daily_pnl", 0.0)
    open_pos     = len(result.get("active_positions", []))
    weekly_mult  = result.get("weekly_risk_multiplier", 1.0)

    pnl_sign     = "+" if daily_pnl >= 0 else ""
    action_str   = ", ".join(str(a) for a in actions) if actions else "none"

    print(f"\n  Actions:         {action_str}")
    print(f"  Equity:          ${equity:,.2f}")
    print(f"  Daily P&L:       {pnl_sign}${daily_pnl:,.2f}")
    print(f"  Open positions:  {open_pos}")
    print(f"  Exits this run:  {len(new_exits)}")
    if weekly_mult < 1.0:
        print(f"  [!] Weekly reduction active — sizing at {weekly_mult:.0%}")

    if new_exits:
        print("\n  Closed trades:")
        for t in new_exits:
            pnl    = t.get("pnl", 0.0)
            rm     = t.get("r_multiple", 0.0)
            reason = t.get("reason", "?")
            strat  = t.get("strategy_id", "")
            sign   = "+" if pnl >= 0 else ""
            strat_tag = f" [{strat[:8]}]" if strat else ""
            print(f"    {reason:<12}{strat_tag}  P&L: {sign}${pnl:,.2f}  R: {rm:+.2f}")

    print(f"\n  Total trade log entries: {len(state.trade_log)}")
    print(f"  State saved to:          {s_path}")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
