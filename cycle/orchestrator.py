"""
Cycle orchestrator -- drives all 4 agents in sequence for one autonomous cycle.

Agents are called directly (bypassing the coordinator) via their _handle()
methods so the orchestrator can pass structured payloads and collect typed
results without going through the LLM router.
"""
from __future__ import annotations

import json
import logging

from core.message import Message, MessageType
from core.network import AgentNetwork
from cycle.state import CycleState, save_state

logger = logging.getLogger(__name__)


def run_cycle(
    state: CycleState,
    network: AgentNetwork,
    asset: dict,
    state_path: str = "state/cycle_state.json",
) -> CycleState:
    """
    Run one complete autonomous trading cycle.

    Sequence:
      1. Trading Coach  -- review last cycle, produce feedback + risk adjustment
      2. Strategy Builder -- generate 10 strategies using Coach feedback
      3. Backtester     -- evaluate strategies, select winner
      4. Trader         -- check entry/exit conditions, place orders if signalled
      5. Persist updated state

    Args:
        state:      Current CycleState (modified in place and returned).
        network:    AgentNetwork with all 4 agents registered.
        asset:      Asset config dict: {"chain", "token_address", "bucket_seconds", ...}.
        state_path: Where to save the updated state.

    Returns:
        Updated CycleState.
    """
    state.cycle_number += 1
    logger.info("=== Cycle %d starting ===", state.cycle_number)
    results: dict[str, str] = {}
    _cycle_tok_in:  int = 0
    _cycle_tok_out: int = 0

    # -- Step 1: Trading Coach -------------------------------------------------
    coach_data: dict = {}
    if state.trade_log or state.active_strategy:
        coach_input = json.dumps({
            "trade_log":               state.trade_log,
            "backtest_sharpe":         _last_backtest_sharpe(state),
            "cycle":                   state.cycle_number - 1,
            "performance_history":     state.performance_history,
            # Projected benchmarks from last cycle's backtester winners
            "projected_avg_r":         state.projected_avg_r,
            "projected_win_rate":      state.projected_win_rate,
            "mc_p95_dd_by_strategy":                state.mc_p95_dd_by_strategy,
            "months_at_good_tier":                  state.months_at_good_tier,
            "per_strategy_compounding":             state.per_strategy_compounding,
            "projected_monthly_pnl_pct_by_strategy": state.projected_monthly_pnl_pct_by_strategy,
            "months_below_projection":              state.months_below_projection,
        })
        coach_result, _ti, _to = _call_agent("trading_coach", coach_input, {}, network)
        _cycle_tok_in += _ti; _cycle_tok_out += _to
        results["coach"] = coach_result
        print(f"[Trading Coach] Cycle {state.cycle_number - 1} review complete")

        try:
            coach_data = json.loads(coach_result)
            state.last_feedback  = coach_data.get("feedback")
            monthly = coach_data.get("monthly_review", {})
            state.risk_multiplier = float(monthly.get("risk_adjustment", state.risk_multiplier))

            # Accumulate monthly/weekly reviews in state history
            if monthly:
                state.monthly_reviews.append(monthly)
            weekly = coach_data.get("weekly_review", {})
            if weekly:
                state.weekly_reviews.append(weekly)

            # Update per-strategy compounding modes and Good-tier tenure
            new_modes = coach_data.get("compounding_modes", {})
            state.per_strategy_compounding.update(new_modes)
            new_months = coach_data.get("months_at_good_tier", {})
            state.months_at_good_tier.update(new_months)
            state.months_below_projection.update(
                coach_data.get("months_below_projection", {})
            )

            # Accumulate new retired patterns (avoid duplicates)
            for pattern in (coach_data.get("feedback") or {}).get("retired_patterns", []):
                if pattern not in state.retired_patterns:
                    state.retired_patterns.append(pattern)

            health = monthly.get("health_tier", "Good")
            print(
                f"  Risk multiplier: {state.risk_multiplier:.2f}x  |  "
                f"Monthly health: {health}  |  "
                f"Retire flag: {monthly.get('retire', False)}"
            )
        except (json.JSONDecodeError, TypeError, KeyError):
            pass
    else:
        print(f"[Trading Coach] Skipped -- no prior trade history")

    # -- Step 2: Strategy Builder ----------------------------------------------
    builder_input = json.dumps({
        "asset":               asset,
        "coach_feedback":      state.last_feedback,
        "retired_patterns":    state.retired_patterns,
        "cycle":               state.cycle_number,
        "asset_strategy_log":  state.asset_strategy_log,
    })
    builder_result, _ti, _to = _call_agent("strategy_builder", builder_input, {}, network)
    _cycle_tok_in += _ti; _cycle_tok_out += _to
    results["builder"] = builder_result
    strategies, diversity_score = _parse_strategies(builder_result)
    try:
        _bdata = json.loads(builder_result)
        _dev_rate = float(_bdata.get("combo_deviation_rate", 0.0))
    except (json.JSONDecodeError, TypeError, ValueError):
        _dev_rate = 0.0
    print(f"[Strategy Builder] Generated {len(strategies)} strategies  "
          f"(diversity score: {diversity_score:.1f}/100  |  combo deviation: {_dev_rate:.1f}%)")

    # -- Step 3: Backtester ----------------------------------------------------
    backtester_input = json.dumps({
        "strategies": strategies,
        "asset":      asset,
    })
    backtester_result, _ti, _to = _call_agent(
        "backtester",
        backtester_input,
        {"past_failures": state.retired_patterns},
        network,
    )
    _cycle_tok_in += _ti; _cycle_tok_out += _to
    results["backtester"] = backtester_result

    try:
        bt_data  = json.loads(backtester_result)
        winner   = bt_data.get("winner")
        winners  = bt_data.get("winners", [winner] if winner else [])
        _print_scorecard(bt_data)
        if winners:
            state.active_strategy   = winners[0]          # primary winner (backward compat)
            state.active_strategies = winners             # full top-3 list
            for w in winners:
                state.performance_history.append(w)

            # Store projected benchmarks for use by the Coach next cycle
            w0 = winners[0]
            state.projected_avg_r    = float(w0.get("avg_r", 0.0))
            state.projected_win_rate = float(w0.get("win_rate", 0.0))
            for w in winners:
                sid   = w.get("strategy_id", "")
                mc_dd = w.get("mc_p95_dd")
                if sid and mc_dd is not None:
                    state.mc_p95_dd_by_strategy[sid] = float(mc_dd)
                proj_pct = w.get("projected_monthly_pnl_pct")
                if sid and proj_pct is not None:
                    state.projected_monthly_pnl_pct_by_strategy[sid] = float(proj_pct)
                proj_table = w.get("return_projections")
                if sid and proj_table:
                    state.return_projections_by_strategy[sid] = proj_table
        else:
            print("[Backtester] No winners selected -- all strategies failed filters")

        # Accumulate every evaluated strategy into the per-asset log
        cycle_n = state.cycle_number
        for s in bt_data.get("all_scores", []):
            state.asset_strategy_log.append({
                "cycle":             cycle_n,
                "strategy_name":     s.get("name", ""),
                "primary_indicator": s.get("primary_indicator", ""),
                "entry_trigger":     s.get("entry_trigger", ""),
                "confirmation":      s.get("confirmation", ""),
                "result":            "passed" if s.get("passed") else "failed",
                "failure_reason":    s.get("failure", ""),
                "score":             s.get("score", 0),
                "sharpe":            s.get("sharpe", 0),
                "oos_sharpe":        s.get("oos_sharpe", 0),
                "win_rate":          s.get("win_rate", 0),
                "max_drawdown_pct":  s.get("max_drawdown_pct", 0),
                "total_pnl_pct":     s.get("total_pnl_pct", 0),
                "mc_p95_dd":         s.get("mc_p95_dd"),
                "mc_risk_class":     s.get("mc_risk_class"),
            })
    except (json.JSONDecodeError, TypeError):
        print("[Backtester] Could not parse result")
        winners = []

    # -- Step 4: Trader --------------------------------------------------------
    active_strats = getattr(state, "active_strategies", None) or (
        [state.active_strategy] if state.active_strategy else []
    )
    if active_strats:
        trader_input = json.dumps({
            "strategies":                active_strats,
            "strategy":                  active_strats[0],  # backward compat
            "asset":                     asset,
            "risk_multiplier":           state.risk_multiplier,
            "per_strategy_adjustments":  coach_data.get("per_strategy_adjustments", {}),
            "per_strategy_compounding":  state.per_strategy_compounding,
        })
        trader_result, _ti, _to = _call_agent("trader", trader_input, {}, network)
        _cycle_tok_in += _ti; _cycle_tok_out += _to
        results["trader"] = trader_result

        try:
            trader_data = json.loads(trader_result)
            new_trades  = [
                t for t in trader_data.get("trade_log", [])
                if t.get("type") == "exit"
            ]
            state.trade_log.extend(new_trades)
            # Aggregate across all strategies
            actions      = trader_data.get("actions", [trader_data.get("entry", {}).get("action", "no_signal")])
            equity       = trader_data.get("equity", 0.0)
            daily_pnl    = trader_data.get("daily_pnl", 0.0)
            weekly_mult  = trader_data.get("weekly_risk_multiplier", 1.0)
            open_pos     = len(trader_data.get("active_positions", []))
            n_strats     = len(active_strats)

            pnl_sign = "+" if daily_pnl >= 0 else ""
            action_summary = ", ".join(str(a) for a in actions) if isinstance(actions, list) else str(actions)
            print(
                f"[Trader] Strategies: {n_strats}  |  Actions: {action_summary}  |  "
                f"Equity: ${equity:,.2f}  |  "
                f"Daily P&L: {pnl_sign}${daily_pnl:,.2f}  |  "
                f"Open positions: {open_pos}  |  "
                f"New exits: {len(new_trades)}"
                + (f"  |  [!] weekly reduction active ({weekly_mult:.1f}x)" if weekly_mult < 1.0 else "")
            )

            if new_trades:
                print("  Closed trades this cycle:")
                for t in new_trades:
                    pnl    = t.get("pnl", 0.0)
                    rm     = t.get("r_multiple", 0.0)
                    reason = t.get("reason", "?")
                    strat  = t.get("strategy_id", "")
                    sign   = "+" if pnl >= 0 else ""
                    strat_tag = f" [{strat[:8]}]" if strat else ""
                    print(f"    {reason:<12}{strat_tag} P&L: {sign}${pnl:,.2f}  R-multiple: {rm:+.2f}")

        except (json.JSONDecodeError, TypeError):
            print("[Trader] Could not parse result")
    else:
        print("[Trader] Skipped -- no active strategy")

    # -- Step 5: Token cost summary --------------------------------------------
    _print_cycle_cost(_cycle_tok_in, _cycle_tok_out, state.cycle_number)

    # -- Step 6: Persist state -------------------------------------------------
    save_state(state, state_path)
    print(f"State saved to {state_path}")

    return state


# -- Helpers -------------------------------------------------------------------

def _call_agent(
    agent_name: str,
    content: str,
    metadata: dict,
    network: AgentNetwork,
) -> tuple[str, int, int]:
    """
    Send a TASK message directly to a named agent and return (content, input_tokens, output_tokens).
    Bypasses the coordinator and message bus -- calls the agent's methods directly.
    Token counts are the delta for this call (handles agents that make multiple LLM calls).
    """
    agent = network._agents.get(agent_name)
    if agent is None:
        raise RuntimeError(f"Agent '{agent_name}' not registered in network")

    tok_in_before  = getattr(agent, "_total_input_tokens",  0)
    tok_out_before = getattr(agent, "_total_output_tokens", 0)

    msg = Message(
        sender="orchestrator",
        recipient=agent_name,
        type=MessageType.TASK,
        content=content,
        metadata=metadata,
    )
    agent.receive(msg)
    responses = agent.process()

    tok_in  = getattr(agent, "_total_input_tokens",  0) - tok_in_before
    tok_out = getattr(agent, "_total_output_tokens", 0) - tok_out_before

    for resp in responses:
        if resp.type in (MessageType.RESULT, MessageType.ERROR):
            if resp.type == MessageType.ERROR:
                logger.error("Agent '%s' returned ERROR: %s", agent_name, resp.content[:200])
            return resp.content, tok_in, tok_out

    raise RuntimeError(f"No RESULT received from agent '{agent_name}'")


# Anthropic Sonnet pricing (USD per million tokens, as of 2025)
_COST_PER_M_INPUT  = 3.00
_COST_PER_M_OUTPUT = 15.00


def _print_cycle_cost(input_tokens: int, output_tokens: int, cycle: int) -> None:
    cost_in  = input_tokens  / 1_000_000 * _COST_PER_M_INPUT
    cost_out = output_tokens / 1_000_000 * _COST_PER_M_OUTPUT
    total    = cost_in + cost_out
    print(
        f"[Token Usage] Cycle {cycle} -- "
        f"Input: {input_tokens:,} tok (${cost_in:.4f})  |  "
        f"Output: {output_tokens:,} tok (${cost_out:.4f})  |  "
        f"Estimated cost: ${total:.4f}"
    )


def _print_scorecard(bt_data: dict) -> None:
    """
    Print a formatted scorecard table of all evaluated strategies.
    Passed strategies show full metrics; failed strategies show why they were eliminated.
    Top-3 winners are marked with *, **, *** by rank.
    """
    all_scores = bt_data.get("all_scores", [])
    winner     = bt_data.get("winner", {})
    winners    = bt_data.get("winners", [winner] if winner else [])
    quality    = bt_data.get("data_quality", {})
    period     = bt_data.get("period", {})
    # Ordered list of winner strategy IDs (index 0 = rank 1)
    winner_ids = [w.get("strategy_id") for w in winners if w]

    passed = [s for s in all_scores if s.get("passed")]
    failed = [s for s in all_scores if not s.get("passed")]

    # Sort passed by score desc, then failed by max_drawdown asc (least bad first)
    passed.sort(key=lambda s: s.get("score") or 0, reverse=True)
    failed.sort(key=lambda s: s.get("max_drawdown_pct") or 1)

    W = 132
    div = "-" * W

    print(f"\n[Backtester] Evaluated {len(all_scores)} strategies -- "
          f"{len(passed)} passed filters, {len(failed)} eliminated")
    print(f"  Data quality: {quality.get('confidence', '?').upper()}  "
          f"| Candle warnings: {len(quality.get('warnings', []))}")

    # -- Backtest period summary -----------------------------------------------
    if period:
        chain  = period.get("chain", "?")
        symbol = period.get("symbol", "")
        label  = f"{chain.upper()}" + (f" / {symbol}" if symbol else "")
        bucket = period.get("bucket_seconds", 0)
        candle_label = period.get("candle_label", "bars")
        bucket_str = f"{bucket:,}s" if bucket else "?"
        print(f"  Asset:           {label}")
        print(f"  Candle size:     {candle_label} ({bucket_str} per bar)")
        print(
            f"  Backtest period: {period.get('is_start')} -> {period.get('oos_end')}  "
            f"({period.get('total_candles'):,} {candle_label}  "
            f"~ {period.get('years_covered', '?')} years)"
        )
        print(
            f"  In-sample  (IS): {period.get('is_start')} -> {period.get('is_end')}  "
            f"({period.get('is_candles'):,} bars -- strategy selection)"
        )
        print(
            f"  Out-of-sample:   {period.get('oos_start')} -> {period.get('oos_end')}  "
            f"({period.get('oos_candles'):,} bars -- held-out validation)"
        )
    print(div)

    # Header (marker col is 3 chars wide)
    print(
        f"  {'':3} {'#':<3} {'Strategy':<40} {'St':<5} {'D':<1}  "
        f"{'Score':>5}  {'Return':>7}  {'MaxDD':>6}  "
        f"{'WinRate':>7}  {'PF':>5}  {'AvgR':>5}  "
        f"{'Sharpe':>6}  {'OOS Sh':>6}  {'MC95':>6}  {'MCCls':<5}  "
        f"{'Trades':>6}  {'Risk':>5}  {'Conf':<6}"
    )
    print(div)

    rank_markers = {0: "*  ", 1: "** ", 2: "***"}

    row = 0
    for s in passed:
        row += 1
        sid = s.get("id", "")
        if sid in winner_ids:
            marker = rank_markers.get(winner_ids.index(sid), "   ")
        else:
            marker = "   "

        mc_dd_val = s.get("mc_p95_dd")
        mc_rc_val = s.get("mc_risk_class") or ""
        mc_dd_str = f"{mc_dd_val * 100:>4.1f}%" if mc_dd_val is not None else "  -- "
        mc_rc_str = (mc_rc_val[:4] if mc_rc_val else "--").ljust(5)

        annual_pct = s.get("annualised_return_pct", 0.0) or 0.0
        direction  = s.get("direction", "long")
        dir_char   = "S" if direction == "short" else "L"
        print(
            f"  {marker} {row:<3} {s['name'][:40]:<40} {'PASS':<5} {dir_char:<1}  "
            f"{s['score']:>5.1f}  "
            f"{s['total_pnl_pct']*100:>+7.1f}%  "
            f"{s['max_drawdown_pct']*100:>5.1f}%  "
            f"{s['win_rate']*100:>6.1f}%  "
            f"{s['profit_factor']:>5.2f}  "
            f"{s['avg_r']:>5.2f}  "
            f"{s['sharpe']:>6.2f}  "
            f"{s['oos_sharpe']:>6.2f}  "
            f"{mc_dd_str:>6}  "
            f"{mc_rc_str:<5}  "
            f"{s['trade_count']:>6}  "
            f"{s['best_risk_pct']:>4.1f}%  "
            f"{s.get('confidence','?'):<6}  "
            f"({annual_pct:>+.1f}%/yr)"
        )
        rw = s.get("recent_window") or {}
        if rw:
            rw_warn = "  [!] Sharpe degraded" if (rw.get("sharpe", 0) < s.get("sharpe", 0) - 0.2) else ""
            print(
                f"           Recent {rw.get('years','?')}yr: "
                f"Return: {rw['total_return_pct']:>+6.1f}% total / {rw['annual_return_pct']:>+5.1f}% annual | "
                f"MaxDD: {rw['max_drawdown_pct']:>5.1f}% | "
                f"Sharpe: {rw['sharpe']:>5.2f} | "
                f"Trades: {rw['trade_count']:>3} | "
                f"WinRate: {rw['win_rate']:>5.1f}%"
                + rw_warn
            )
        for rl in (s.get("risk_breakdown") or []):
            mc_rl = rl.get("mc_p95_dd")
            mc_rl_str = f"{mc_rl:>5.1f}%" if mc_rl is not None else "  --  "
            print(
                f"           Risk {rl['risk_pct']:.2f}% | "
                f"Return: {rl['total_return_pct']:>+6.1f}% total / {rl['annual_return_pct']:>+5.1f}% annual | "
                f"MaxDD: {rl['max_drawdown_pct']:>5.1f}% | "
                f"Sharpe: {rl['sharpe']:>5.2f} | "
                f"MC95: {mc_rl_str}"
            )

    if failed:
        print(f"  {'-'*127}")
        for s in failed:
            row += 1
            reason   = (s.get("failure") or "")[:50]
            risk_val = s.get("best_risk_pct")
            risk_str = f"{risk_val:>4.1f}%" if risk_val is not None else " --  "
            direction = s.get("direction", "long")
            dir_char  = "S" if direction == "short" else "L"
            print(
                f"     {row:<3} {s['name'][:40]:<40} {'FAIL':<5} {dir_char:<1}  "
                f"{'--':>5}  "
                f"{s['total_pnl_pct']*100:>+7.1f}%  "
                f"{s['max_drawdown_pct']*100:>5.1f}%  "
                f"{s['win_rate']*100:>6.1f}%  "
                f"{'--':>5}  "
                f"{'--':>5}  "
                f"{s['sharpe']:>6.2f}  "
                f"{'--':>6}  "
                f"{'--':>6}  "
                f"{'--':<5}  "
                f"{s['trade_count']:>6}  "
                f"{risk_str:>5}  "
                f"  > {reason}"
            )

    print(div)

    # Winner summary block: show all top-3 winners
    rank_labels = {1: "WINNER", 2: "RUNNER-UP", 3: "3RD PLACE"}
    for rank_idx, w in enumerate(winners, 1):
        if not w:
            continue
        mark         = "*" * rank_idx
        label        = rank_labels.get(rank_idx, f"#{rank_idx}")
        w_name       = w.get("strategy_name", "?")
        w_score      = w.get("score", 0)
        w_ret        = w.get("total_pnl_pct", 0) * 100
        w_dd         = w.get("max_drawdown", 0) * 100
        w_sh         = w.get("sharpe", 0)
        w_oos        = w.get("oos_sharpe", 0)
        w_conf       = w.get("confidence_rating", "?")
        w_risk       = w.get("best_risk_pct", 0)
        w_trades_is  = w.get("trade_count", 0)
        w_trades_oos = w.get("oos_trade_count", 0)
        mc_p95       = w.get("mc_p95_dd")
        mc_cls       = w.get("mc_risk_class", "?")
        mc_str       = f"MC95: {mc_p95*100:.1f}% ({mc_cls})" if mc_p95 is not None else ""
        w_annual     = w.get("annualised_return_pct", 0.0) or 0.0
        w_dir        = w.get("direction", "long").capitalize()
        w_rb         = w.get("risk_breakdown") or []
        w_rw         = w.get("recent_window") or {}
        # Period strings from the shared period block
        is_start  = period.get("is_start", "?")   if period else "?"
        oos_end   = period.get("oos_end", "?")     if period else "?"
        years     = period.get("years_covered", "?") if period else "?"
        print(
            f"  {mark} {label}: {w_name}\n"
            f"    Score: {w_score:.1f}/100  |  Direction: {w_dir}  |  "
            f"Return: {w_ret:+.1f}% total / {w_annual:+.1f}% annual  |  "
            f"MaxDD: {w_dd:.1f}%  |  Sharpe IS/OOS: {w_sh:.2f}/{w_oos:.2f}  |  "
            f"Risk: {w_risk}%  |  Confidence: {w_conf}"
            + (f"  |  {mc_str}" if mc_str else "")
            + f"\n    Trades: {w_trades_is} IS + {w_trades_oos} OOS"
            f"  |  Period: {is_start} -> {oos_end} ({years} yrs)"
        )
        if w_rw:
            rw_warn = "  [!] Sharpe degraded" if (w_rw.get("sharpe", 0) < w_sh - 0.2) else ""
            print(
                f"    Recent {w_rw.get('years','?')}yr: "
                f"Return: {w_rw['total_return_pct']:>+6.1f}% total / {w_rw['annual_return_pct']:>+5.1f}% annual | "
                f"MaxDD: {w_rw['max_drawdown_pct']:>5.1f}% | "
                f"Sharpe: {w_rw['sharpe']:>5.2f} | "
                f"Trades: {w_rw['trade_count']:>3} | "
                f"WinRate: {w_rw['win_rate']:>5.1f}%"
                + rw_warn
            )
        if w_rb:
            print("    Risk Breakdown:")
            for rl in w_rb:
                mc_rl = rl.get("mc_p95_dd")
                mc_rl_str = f"{mc_rl:>5.1f}%" if mc_rl is not None else "  --  "
                print(
                    f"      Risk {rl['risk_pct']:.2f}% | "
                    f"Return: {rl['total_return_pct']:>+6.1f}% total / {rl['annual_return_pct']:>+5.1f}% annual | "
                    f"MaxDD: {rl['max_drawdown_pct']:>5.1f}% | "
                    f"Sharpe: {rl['sharpe']:>5.2f} | "
                    f"MC95: {mc_rl_str}"
                )
    if winners:
        print(div)
        print()


def _parse_strategies(raw: str) -> tuple[list[dict], float]:
    """
    Extract a list of strategy dicts and diversity score from the Strategy Builder output.
    Returns (strategies, diversity_score).
    """
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data, 0.0
        if isinstance(data, dict) and "strategies" in data:
            return data["strategies"], float(data.get("diversity_score", 0.0))
    except (json.JSONDecodeError, TypeError):
        pass

    # Try to find a JSON array in the text (strip markdown fences)
    import re
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0)), 0.0
        except json.JSONDecodeError:
            pass

    logger.warning("Could not parse strategy list from builder output; using empty list")
    return [], 0.0


def _last_backtest_sharpe(state: CycleState) -> float:
    """Return the Sharpe ratio from the most recent winning strategy."""
    if state.active_strategy:
        return float(state.active_strategy.get("sharpe", 1.0))
    return 1.0
