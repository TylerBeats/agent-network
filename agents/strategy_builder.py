import json
import logging
import re
import time

from anthropic import Anthropic

from agents.asset_profile import get_asset_hints, summarise_strategy_log
from agents.worker import WorkerAgent
from core.message import Message, MessageType

logger = logging.getLogger(__name__)

# -- Valid strategy vocabulary -------------------------------------------------

VALID_INDICATORS    = [
    "RSI", "EMA", "SMA", "MACD", "BOLLINGER", "ATR", "STOCH",
    "WMA", "ADX", "CCI", "WILLR", "ROC", "KELTNER", "DONCHIAN", "OBV",
]
VALID_TRIGGERS      = [
    "RSI_OVERSOLD", "RSI_OVERBOUGHT",
    "PRICE_ABOVE_EMA", "PRICE_BELOW_EMA",
    "MACD_CROSS_ABOVE", "MACD_CROSS_BELOW",
    "BB_LOWER_TOUCH", "BB_UPPER_TOUCH",
    "STOCH_OVERSOLD", "STOCH_OVERBOUGHT",
    "CCI_OVERSOLD", "CCI_OVERBOUGHT",
    "WILLR_OVERSOLD", "WILLR_OVERBOUGHT",
    "ROC_CROSS_ABOVE", "ROC_CROSS_BELOW",
    "DC_UPPER_BREAK", "DC_LOWER_BREAK",
    "KB_LOWER_TOUCH", "KB_UPPER_TOUCH",
]
VALID_CONFIRMATIONS = [
    "PRICE_ABOVE_EMA", "PRICE_BELOW_EMA",
    "VOLUME_ABOVE_SMA", "ATR_EXPANDING",
    "ADX_TRENDING", "OBV_RISING", "NONE",
]

_STRATEGY_SCHEMA = """
{
  "id": "<unique_id>",
  "name": "<descriptive name>",
  "primary_indicator": {
    "type": "<RSI|EMA|SMA|WMA|MACD|BOLLINGER|ATR|STOCH|ADX|CCI|WILLR|ROC|KELTNER|DONCHIAN|OBV>",
    "params": { "period": <int> }
  },
  "confirmation_indicator": {
    "type": "<EMA|SMA|WMA|ATR|VOLUME|ADX|OBV|NONE>",
    "params": { "period": <int> }
  },
  "entry": {
    "trigger": "<one of the valid triggers>",
    "filter": "<one of the valid confirmations>"
  },
  "exit": {
    "stop_loss":         { "type": "atr_multiple", "value": <float> },
    "take_profit":       { "type": "r_multiple",   "value": <float> },
    "trailing_stop_atr": null,
    "time_exit_bars":    null
  },
  "risk": { "max_open_positions": <1-5> },
  "metadata": { "timeframe": "4h", "market_type": "crypto" }
}
"""


class StrategyBuilderAgent(WorkerAgent):
    """Generates candidate trading strategies for backtesting evaluation."""

    def __init__(self, client: Anthropic):
        super().__init__(
            name="strategy_builder",
            role=(
                "creative trading strategy generator for an autonomous trading network. "
                "Your sole purpose is generating well-structured, testable trading strategies "
                "by combining technical indicators, risk/reward parameters, entry/exit rules, "
                "and market context filters. You generate batches of 10 candidate strategies "
                "per cycle, ensuring no two strategies are too correlated -- diversification is "
                "mandatory. Each strategy is output as a structured JSON object containing: "
                "primary signal indicator (trend, momentum, volatility, volume, or "
                "support/resistance category), at least one confirmation indicator, entry "
                "trigger (exact condition that initiates the trade), entry filter (secondary "
                "confirmation), stop loss (ATR multiple or fixed percentage), take profit "
                "target (R-multiple e.g. 2R or 3R, or indicator-based), optional trailing "
                "stop, time exit (maximum hold duration), risk per trade (proposed at 0.5%, "
                "1%, 1.5%, or 2% -- the Backtester selects the best level; 2% is the absolute "
                "cap), max open positions (default 3, never exceeds 5), and metadata tags "
                "for market type, timeframe, indicator set, and risk profile. "
                "You incorporate Trading Coach feedback each cycle to refine future generation "
                "and maintain a strategy memory log to avoid regenerating underperforming "
                "approaches. You do not evaluate strategies -- that is the Backtester's job. "
                "You produce, diversify, and innovate."
            ),
            client=client,
        )

    N_TOTAL    = 100   # target strategies per cycle
    BATCH_SIZE = 10    # strategies per LLM call

    # Anthropic rate limit: 30,000 input tokens per minute.
    # Pacing: after each batch, sleep long enough so the tokens consumed
    # by that call would have cleared the rolling 60-second window.
    # min_gap_seconds = tokens_used / (30_000 / 60) = tokens_used / 500
    _RATE_LIMIT_TOKENS_PER_MIN: int = 30_000

    def _handle(self, message: Message) -> Message | None:
        if message.type != MessageType.TASK:
            return None

        recipient = message.metadata.get("original_sender", message.sender)

        # -- Try structured generation request ---------------------------------
        try:
            request           = json.loads(message.content)
            coach_feedback    = request.get("coach_feedback")
            retired_patterns  = request.get("retired_patterns", [])
            asset             = request.get("asset", {})
            cycle             = int(request.get("cycle", 1))
            asset_strategy_log = request.get("asset_strategy_log", [])
        except (json.JSONDecodeError, TypeError):
            # Fallback: plain LLM response for unstructured tasks
            result = self._call_llm(message.content)
            return Message(
                sender=self.name,
                recipient=recipient,
                type=MessageType.RESULT,
                content=result,
            )

        # -- Batched generation: N_BATCHES x BATCH_SIZE = N_TOTAL strategies ---
        # Dedup key is a 3-tuple (indicator, trigger, confirmation) so that
        # RSI+OVERSOLD+EMA and RSI+OVERSOLD+ATR are both accepted as unique.
        # This gives ~14 × 20 × 7 = 1,960 possible slots for 100 strategies.
        n_batches     = self.N_TOTAL // self.BATCH_SIZE
        all_strategies: list[dict] = []
        seen_combos: set[tuple[str, str, str]] = set()
        _rate_per_sec = self._RATE_LIMIT_TOKENS_PER_MIN / 60.0   # 500 tok/s

        for batch_idx in range(n_batches):
            prompt = _build_prompt(
                coach_feedback, retired_patterns, asset, cycle,
                asset_strategy_log, seen_combos, batch_idx,
            )
            call_start = time.monotonic()

            # Retry once on rate-limit as a safety net
            for attempt in range(2):
                try:
                    raw = self._call_llm(prompt)
                    break
                except Exception as exc:
                    if attempt == 0 and "rate_limit" in str(exc).lower():
                        logger.warning(
                            "StrategyBuilderAgent: rate limit on batch %d, sleeping 65s",
                            batch_idx,
                        )
                        time.sleep(65)
                    else:
                        raise

            # Token-based pacing: enforce minimum wall-clock gap between calls
            # so the rate of input tokens never exceeds 30k/minute.
            # min_gap = tokens_used / (30_000 tok/min ÷ 60 s/min) = tok / 500
            min_gap = self._last_input_tokens / _rate_per_sec
            elapsed = time.monotonic() - call_start
            if elapsed < min_gap:
                sleep_for = min_gap - elapsed
                logger.debug(
                    "StrategyBuilderAgent: batch %d used %d input tokens — pacing %.1fs",
                    batch_idx, self._last_input_tokens, sleep_for,
                )
                time.sleep(sleep_for)

            batch = _extract_and_sanitise(raw, cycle, offset=len(all_strategies))
            for s in batch:
                combo = (
                    s.get("primary_indicator", {}).get("type", ""),
                    s.get("entry", {}).get("trigger", ""),
                    s.get("entry", {}).get("filter", "NONE"),
                )
                if combo not in seen_combos:
                    seen_combos.add(combo)
                    all_strategies.append(s)

        diversity = _compute_diversity_score(all_strategies)

        logger.info(
            "StrategyBuilderAgent: cycle %d generated %d strategies (diversity: %.1f/100)",
            cycle, len(all_strategies), diversity,
        )

        output = {
            "strategies":      all_strategies,
            "diversity_score": diversity,
        }

        return Message(
            sender=self.name,
            recipient=recipient,
            type=MessageType.RESULT,
            content=json.dumps(output, indent=2),
        )


# -- Helpers -------------------------------------------------------------------

def _build_prompt(
    coach_feedback: dict | None,
    retired_patterns: list[dict],
    asset: dict,
    cycle: int,
    asset_strategy_log: list[dict] | None = None,
    seen_combos: set[tuple[str, str, str]] | None = None,
    batch_idx: int = 0,
) -> str:
    chain          = asset.get("chain", "pulsechain")
    bucket_seconds = asset.get("bucket_seconds", 14400)
    bar_label      = "daily bars" if bucket_seconds >= 86400 else f"{bucket_seconds // 3600}h bars"
    hints          = get_asset_hints(chain)

    # -- Asset class context block ----------------------------------------------
    asset_section = (
        f"\nASSET PROFILE:\n"
        f"  Class:    {hints['class_label']}\n"
        f"  Timeframe: {bar_label}\n"
        f"  Regime note: {hints['regime_note']}\n"
        f"  Strategy focus: {hints['strategy_focus']}\n"
        f"  Volume note: {hints['volume_note']}\n"
        f"  PREFERRED indicators: {', '.join(hints['preferred_indicators'])}\n"
        f"  PREFERRED triggers:   {', '.join(hints['preferred_triggers'])}\n"
        f"  PREFERRED confirmations: {', '.join(hints['preferred_confirmations'])}\n"
    )
    if hints["avoid_triggers"]:
        asset_section += (
            f"  AVOID these triggers -- they consistently fail on this asset class:\n"
            f"    {', '.join(hints['avoid_triggers'])}\n"
            f"  Reason: {hints['avoid_reason']}\n"
        )

    # -- Per-asset historical strategy log -------------------------------------
    log_section = ""
    if asset_strategy_log:
        log_section = "\n" + summarise_strategy_log(asset_strategy_log) + "\n"

    # -- Trading Coach feedback -------------------------------------------------
    feedback_section = ""
    if coach_feedback:
        hypothesis = coach_feedback.get("next_cycle_hypothesis", "")
        best_inds  = ", ".join(coach_feedback.get("best_indicator_types", []))
        failing    = ", ".join(coach_feedback.get("failing_regimes", []))
        best_rr    = coach_feedback.get("best_rr_ratio", 0.0)
        feedback_section = (
            f"\nTRADING COACH FEEDBACK (incorporate this into your strategies):\n"
            f"  Best indicator types to explore: {best_inds or 'any'}\n"
            f"  Best R/R ratio achieved: {best_rr:.2f}\n"
            f"  Failing regimes (focus area): {failing or 'none identified'}\n"
            f"  Next-cycle hypothesis: {hypothesis}\n"
        )

    # -- Retired patterns -------------------------------------------------------
    retired_section = ""
    if retired_patterns:
        lines = "\n".join(
            f"  - indicator={p.get('primary_indicator_type','')} "
            f"trigger={p.get('entry_trigger','')}"
            for p in retired_patterns
        )
        retired_section = f"\nAVOID these retired patterns (do NOT regenerate them):\n{lines}\n"

    # Already-generated combinations to avoid in this batch
    already_used = ""
    if seen_combos:
        combos_str = ", ".join(
            f"{ind}+{trig}+{conf}" for ind, trig, conf in sorted(seen_combos)
        )
        already_used = (
            f"\nALREADY GENERATED (do NOT repeat these indicator+trigger+confirmation combos):\n"
            f"  {combos_str}\n"
        )

    return (
        f"You are the Strategy Builder for an autonomous trading network. Cycle: {cycle}, "
        f"batch {batch_idx + 1} of 10 (generating strategies #{batch_idx * 10 + 1}-{(batch_idx + 1) * 10}).\n"
        f"Asset: chain={chain}, bucket_seconds={bucket_seconds} ({bar_label}).\n"
        f"{asset_section}"
        f"{log_section}"
        f"{feedback_section}"
        f"{retired_section}"
        f"{already_used}"
        f"\nGenerate exactly 10 diverse candidate trading strategies as a JSON ARRAY.\n"
        f"IMPORTANT: Bias your generation strongly toward the PREFERRED indicators and triggers "
        f"listed above. Avoid the explicitly listed AVOID triggers. "
        f"Ensure all 10 strategies in this batch use DIFFERENT indicator+trigger+confirmation "
        f"combinations from each other AND from the already-generated list above. "
        f"Vary the 'filter' field (confirmation) freely — the same indicator+trigger pair "
        f"with a different confirmation counts as a unique strategy.\n\n"
        f"Valid primary indicator types: {' | '.join(VALID_INDICATORS)}\n"
        f"Valid entry triggers: {' | '.join(VALID_TRIGGERS)}\n"
        f"Valid confirmation filters: {' | '.join(VALID_CONFIRMATIONS)}\n\n"
        f"Each strategy must follow this exact schema:\n{_STRATEGY_SCHEMA}\n"
        f"Reply with ONLY the JSON array -- no commentary, no markdown fences."
    )


def _extract_and_sanitise(raw: str, cycle: int, offset: int = 0) -> list[dict]:
    """Parse the LLM response and normalise each strategy dict."""
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?", "", raw).strip()

    strategies: list[dict] = []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            strategies = parsed
        elif isinstance(parsed, dict) and "strategies" in parsed:
            strategies = parsed["strategies"]
    except json.JSONDecodeError:
        # Try to extract the first JSON array from the text
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                strategies = json.loads(match.group(0))
            except json.JSONDecodeError:
                logger.error("StrategyBuilderAgent: could not parse JSON from LLM output")
                return []

    result = []
    for i, s in enumerate(strategies[:10]):
        result.append(_sanitise(s, offset + i, cycle))
    return result


def _compute_diversity_score(strategies: list[dict]) -> float:
    """
    Score 0–100: how diverse the batch of strategies is.
    Measures uniqueness of primary indicator types and entry triggers.
    100 = every strategy has a unique indicator AND a unique trigger.
    """
    if not strategies:
        return 0.0
    n = len(strategies)
    indicators = [s.get("primary_indicator", {}).get("type", "") for s in strategies]
    triggers   = [s.get("entry", {}).get("trigger", "") for s in strategies]
    unique_ind  = len(set(indicators)) / n
    unique_trig = len(set(triggers)) / n
    return round((unique_ind + unique_trig) / 2 * 100, 1)


def _sanitise(raw: dict, index: int, cycle: int) -> dict:
    """Fill missing or invalid fields with safe defaults."""
    p_type = raw.get("primary_indicator", {}).get("type", "RSI").upper()
    if p_type not in VALID_INDICATORS:
        p_type = "RSI"

    trigger = raw.get("entry", {}).get("trigger", "RSI_OVERSOLD").upper()
    if trigger not in VALID_TRIGGERS:
        trigger = "RSI_OVERSOLD"

    confirm = raw.get("entry", {}).get("filter", "NONE").upper()
    if confirm not in VALID_CONFIRMATIONS:
        confirm = "NONE"

    return {
        "id":   raw.get("id", f"c{cycle}_s{index+1:02d}"),
        "name": raw.get("name", f"Strategy {index + 1}"),
        "primary_indicator": {
            "type":   p_type,
            "params": raw.get("primary_indicator", {}).get("params", {"period": 14}),
        },
        "confirmation_indicator": raw.get("confirmation_indicator", {
            "type": "EMA", "params": {"period": 50},
        }),
        "entry": {"trigger": trigger, "filter": confirm},
        "exit":  raw.get("exit", {
            "stop_loss":         {"type": "atr_multiple", "value": 2.0},
            "take_profit":       {"type": "r_multiple",   "value": 2.0},
            "trailing_stop_atr": None,
            "time_exit_bars":    None,
        }),
        "risk":     raw.get("risk", {"max_open_positions": 3}),
        "metadata": raw.get("metadata", {"timeframe": "4h", "market_type": "crypto"}),
    }
