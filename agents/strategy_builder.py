import json
import logging
import os
import random
import re
import time

from anthropic import Anthropic

from agents.asset_profile import get_asset_hints, summarise_strategy_log
from agents.worker import WorkerAgent
from config.settings import LOCAL_LLM
from core.message import Message, MessageType

logger = logging.getLogger(__name__)

# -- Valid strategy vocabulary -------------------------------------------------

# ATR, ADX, OBV have no entry triggers — they are confirmation-only.
# Only indicators with matching entry triggers may be used as primary.
VALID_PRIMARY_INDICATORS = [
    "RSI", "EMA", "SMA", "WMA", "MACD", "BOLLINGER",
    "STOCH", "CCI", "WILLR", "ROC", "KELTNER", "DONCHIAN", "ICHIMOKU",
]
# Legacy name kept for backward compatibility (e.g. existing tests)
VALID_INDICATORS = VALID_PRIMARY_INDICATORS + ["ATR", "ADX", "OBV"]

# Each primary indicator maps to exactly the triggers it can produce data for.
_PRIMARY_TRIGGER_MAP: dict[str, list[str]] = {
    "RSI":      ["RSI_OVERSOLD", "RSI_OVERBOUGHT"],
    "EMA":      ["PRICE_ABOVE_EMA", "PRICE_BELOW_EMA"],
    "SMA":      ["PRICE_ABOVE_EMA", "PRICE_BELOW_EMA"],
    "WMA":      ["PRICE_ABOVE_EMA", "PRICE_BELOW_EMA"],
    "MACD":     ["MACD_CROSS_ABOVE", "MACD_CROSS_BELOW"],
    "BOLLINGER":["BB_LOWER_TOUCH", "BB_UPPER_TOUCH"],
    "STOCH":    ["STOCH_OVERSOLD", "STOCH_OVERBOUGHT"],
    "CCI":      ["CCI_OVERSOLD", "CCI_OVERBOUGHT"],
    "WILLR":    ["WILLR_OVERSOLD", "WILLR_OVERBOUGHT"],
    "ROC":      ["ROC_CROSS_ABOVE", "ROC_CROSS_BELOW"],
    "DONCHIAN": ["DC_UPPER_BREAK", "DC_LOWER_BREAK"],
    "KELTNER":  ["KB_LOWER_TOUCH", "KB_UPPER_TOUCH", "KC_LOWER_TOUCH", "KC_UPPER_TOUCH"],
    "ICHIMOKU": ["PRICE_ABOVE_CLOUD", "PRICE_BELOW_CLOUD"],
}

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
    "KC_LOWER_TOUCH", "KC_UPPER_TOUCH",
    "PRICE_ABOVE_CLOUD", "PRICE_BELOW_CLOUD",
]
VALID_CONFIRMATIONS = [
    "PRICE_ABOVE_EMA", "PRICE_BELOW_EMA",
    "VOLUME_ABOVE_SMA", "ATR_EXPANDING",
    "ADX_TRENDING", "OBV_RISING", "NONE",
]

# ── Indicator family grouping (controls batch assignment) ─────────────────────
# 4 batches × 25 slots = 100 strategies, one group per batch.
# Explicit slot allocation guarantees all indicator families are represented.
_COMBO_GROUPS: list[tuple[str, set[str]]] = [
    ("Trend (EMA / SMA / WMA)",                          {"EMA", "SMA", "WMA"}),
    ("Channel Breakout (Donchian / Keltner)",             {"DONCHIAN", "KELTNER"}),
    ("Momentum (RSI / ROC / CCI / Stochastic / WillR)",  {"RSI", "ROC", "CCI", "STOCH", "WILLR"}),
    ("Volatility & MACD (Bollinger / Ichimoku / MACD)",  {"BOLLINGER", "ICHIMOKU", "MACD"}),
]
_GROUP_SLOTS = [25, 25, 25, 25]  # 100 total; one group per batch

_BATCH_FAMILY_LABELS: dict[int, str] = {
    i: label for i, (label, _) in enumerate(_COMBO_GROUPS)
}


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
    BATCH_SIZE = 25    # strategies per LLM call (4 batches × 25 = 100 total)

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

        # -- Combo matrix: pre-assign all (indicator, trigger, confirmation) slots --
        # This guarantees diversity by construction and removes the ever-growing
        # "ALREADY GENERATED" list from the prompt (saving input tokens each batch).
        # Cache persists across cycles so the LLM explores new combos first.
        combo_cache = _load_combo_cache()
        all_combos = _enumerate_all_combos()   # 168 valid triplets

        hints = get_asset_hints(asset.get("chain", "pulsechain"))
        preferred_inds   = set(hints.get("preferred_indicators", []))
        avoid_triggers   = set(hints.get("avoid_triggers", []))

        # Build per-group slot allocations so all indicator families are represented.
        # Each of the 4 groups gets exactly 25 slots (one per batch).
        # Within each group: avoid-trigger combos last, unseen first, preferred first.
        rng = random.Random(cycle)
        cycle_combos = _build_cycle_combos(all_combos, combo_cache, preferred_inds, avoid_triggers, rng)

        # -- Batched generation: 4 × 25 = 100 strategies -----------------------
        n_batches        = self.N_TOTAL // self.BATCH_SIZE   # 4
        all_strategies: list[dict] = []
        seen_combos: set[tuple[str, str, str]] = set()    # cross-batch dedup safety net
        total_deviations = 0
        total_generated  = 0
        _rate_per_sec    = self._RATE_LIMIT_TOKENS_PER_MIN / 60.0   # 500 tok/s

        for batch_idx in range(n_batches):
            batch_combos = cycle_combos[
                batch_idx * self.BATCH_SIZE : (batch_idx + 1) * self.BATCH_SIZE
            ]
            prompt = _build_prompt(
                coach_feedback, retired_patterns, asset, cycle,
                asset_strategy_log, batch_combos, batch_idx,
            )
            call_start = time.monotonic()

            # Retry once on rate-limit or (LOCAL_LLM) truncated response
            for attempt in range(2):
                try:
                    raw = self._call_llm(prompt)
                except Exception as exc:
                    if attempt == 0 and "rate_limit" in str(exc).lower():
                        logger.warning(
                            "StrategyBuilderAgent: rate limit on batch %d, sleeping 65s",
                            batch_idx,
                        )
                        time.sleep(65)
                        continue
                    raise

                if LOCAL_LLM and attempt == 0 and _looks_truncated(raw):
                    logger.warning(
                        "StrategyBuilderAgent: truncated response on batch %d, retrying",
                        batch_idx,
                    )
                    # Roll back the memory entries added by _call_llm so the retry
                    # starts from a clean state (no partial assistant message in context).
                    if len(self.memory) >= 2:
                        self.memory.pop()   # truncated assistant response
                        self.memory.pop()   # user prompt
                    continue
                break

            # Token-based pacing: enforce minimum wall-clock gap between calls
            # so the rate of input tokens never exceeds 30k/minute.
            # min_gap = tokens_used / (30_000 tok/min ÷ 60 s/min) = tok / 500
            try:
                min_gap = int(self._last_input_tokens) / _rate_per_sec
            except (TypeError, ValueError):
                min_gap = 0.0
            elapsed = time.monotonic() - call_start
            if elapsed < min_gap:
                sleep_for = min_gap - elapsed
                logger.debug(
                    "StrategyBuilderAgent: batch %d used %d input tokens — pacing %.1fs",
                    batch_idx, self._last_input_tokens, sleep_for,
                )
                time.sleep(sleep_for)

            batch, devs = _extract_and_sanitise(
                raw, cycle,
                offset=len(all_strategies),
                limit=self.BATCH_SIZE,
                assigned_combos=batch_combos,
            )
            total_deviations += devs
            total_generated  += len(batch)
            if devs > 0:
                logger.info(
                    "StrategyBuilderAgent: batch %d -- %d/%d strategies deviated from "
                    "assigned combos (corrected)",
                    batch_idx, devs, len(batch),
                )
            for s in batch:
                combo = (
                    s.get("primary_indicator", {}).get("type", ""),
                    s.get("entry", {}).get("trigger", ""),
                    s.get("entry", {}).get("filter", "NONE"),
                )
                if combo not in seen_combos:
                    seen_combos.add(combo)
                    all_strategies.append(s)

        # Update combo cache with everything generated this cycle
        _save_combo_cache(combo_cache | seen_combos)

        diversity = _compute_diversity_score(all_strategies)
        combo_deviation_rate = round(
            total_deviations / total_generated * 100, 1
        ) if total_generated > 0 else 0.0

        logger.info(
            "StrategyBuilderAgent: cycle %d generated %d strategies "
            "(diversity: %.1f/100, combo deviation: %.1f%%)",
            cycle, len(all_strategies), diversity, combo_deviation_rate,
        )

        output = {
            "strategies":           all_strategies,
            "diversity_score":      diversity,
            "combo_deviation_rate": combo_deviation_rate,
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
    assigned_combos: list[tuple[str, str, str]] | None = None,
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

    # -- Assigned combo slots for this batch -----------------------------------
    # Hard constraints: LLM MUST use the exact primary_indicator+trigger per slot.
    # Deviations are detected post-generation and corrected automatically.
    n_slots      = len(assigned_combos) if assigned_combos else 0
    family_label = _BATCH_FAMILY_LABELS.get(batch_idx, "Mixed")
    slot_lines   = ""
    if assigned_combos:
        slot_lines = "\n".join(
            f"  Slot {i+1:2d}: primary_indicator.type = \"{ind}\"  |  "
            f"entry.trigger = \"{trig}\"  |  entry.filter = \"{conf}\""
            for i, (ind, trig, conf) in enumerate(assigned_combos)
        )

    n_batches_total = 4
    return (
        f"You are the Strategy Builder for an autonomous trading network. Cycle: {cycle}, "
        f"batch {batch_idx + 1} of {n_batches_total} "
        f"(strategies #{batch_idx * n_slots + 1}-{(batch_idx + 1) * n_slots}).\n"
        f"Batch focus: {family_label}\n"
        f"Asset: chain={chain}, bucket_seconds={bucket_seconds} ({bar_label}).\n"
        f"{asset_section}"
        f"{log_section}"
        f"{feedback_section}"
        f"{retired_section}"
        + (
            f"\n{'='*70}\n"
            f"MANDATORY COMBO CONSTRAINTS — NO SUBSTITUTIONS ALLOWED\n"
            f"{'='*70}\n"
            f"Each slot below specifies an EXACT primary_indicator.type and entry.trigger.\n"
            f"You MUST use these values verbatim. Do NOT substitute a different indicator\n"
            f"or trigger — even if you think another would work better.\n"
            f"Your ONLY creative freedom is:\n"
            f"  - indicator period (params.period)\n"
            f"  - stop_loss ATR multiple (value)\n"
            f"  - take_profit R-multiple (value)\n"
            f"  - confirmation indicator type and period\n"
            f"  - max_open_positions (1-5)\n"
            f"  - strategy name and metadata\n"
            f"\nSlot assignments ({n_slots} strategies, one per slot):\n"
            f"{slot_lines}\n"
            f"{'='*70}\n"
            if assigned_combos else
            f"\nGenerate exactly {n_slots or 25} diverse candidate trading strategies.\n"
        )
        + f"\nGenerate a JSON ARRAY of exactly {n_slots or 25} strategy objects.\n"
        f"Object i MUST use the primary_indicator and trigger from Slot i above.\n"
        f"Each strategy must follow this exact schema:\n{_STRATEGY_SCHEMA}\n"
        f"Reply with ONLY the JSON array -- no commentary, no markdown fences."
    )


def _enumerate_all_combos() -> list[tuple[str, str, str]]:
    """Enumerate every valid (primary_indicator, trigger, confirmation) triplet.
    With 12 primary indicators × 2 triggers each × 7 confirmations = 168 total slots.
    """
    combos: list[tuple[str, str, str]] = []
    for ind, triggers in _PRIMARY_TRIGGER_MAP.items():
        for trig in triggers:
            for conf in VALID_CONFIRMATIONS:
                combos.append((ind, trig, conf))
    return combos


def _build_cycle_combos(
    all_combos: list[tuple[str, str, str]],
    combo_cache: set[tuple[str, str, str]],
    preferred_inds: set[str],
    avoid_triggers: set[str],
    rng: random.Random,
) -> list[tuple[str, str, str]]:
    """
    Select exactly N_TOTAL (100) combos with explicit per-group slot allocation.
    Each of the 4 indicator family groups contributes exactly 25 slots (one batch).
    Within each group, combos are ordered: avoid-trigger last, unseen before seen,
    preferred indicators first.  Any shortfall (small groups) is padded by repeating
    the last combo so batch size is always exactly BATCH_SIZE.
    """
    cycle_combos: list[tuple[str, str, str]] = []
    for (_, group_inds), n_slots in zip(_COMBO_GROUPS, _GROUP_SLOTS):
        group = [c for c in all_combos if c[0] in group_inds]
        rng.shuffle(group)
        group.sort(key=lambda c: (
            c[1] in avoid_triggers,      # avoid-trigger combos pushed to back
            c in combo_cache,            # unseen combos before seen
            c[0] not in preferred_inds,  # preferred indicators float to top
        ))
        # Pad with last combo if group has fewer combos than needed
        if group and len(group) < n_slots:
            group = group + [group[-1]] * (n_slots - len(group))
        cycle_combos.extend(group[:n_slots])
    return cycle_combos


_COMBO_CACHE_PATH = "state/combo_cache.json"


def _load_combo_cache() -> set[tuple[str, str, str]]:
    """Load the set of (indicator, trigger, confirmation) combos tried in previous cycles."""
    if not os.path.exists(_COMBO_CACHE_PATH):
        return set()
    try:
        with open(_COMBO_CACHE_PATH) as f:
            return {tuple(c) for c in json.load(f)}
    except Exception:
        return set()


def _save_combo_cache(combos: set[tuple[str, str, str]]) -> None:
    """Persist the updated combo cache to disk."""
    os.makedirs(os.path.dirname(_COMBO_CACHE_PATH), exist_ok=True)
    try:
        with open(_COMBO_CACHE_PATH, "w") as f:
            json.dump([list(c) for c in sorted(combos)], f, indent=2)
    except Exception as exc:
        logger.warning("StrategyBuilderAgent: could not save combo cache: %s", exc)


def _looks_truncated(text: str) -> bool:
    """Return True if the response looks like a cut-off JSON array.
    Used in LOCAL_LLM mode where output token limits are common.
    """
    stripped = re.sub(r"```(?:json)?", "", text).strip()
    return "[" in stripped and not stripped.endswith("]")


def normalize_strategy_json(raw: dict) -> dict:
    """Map alternative field names from local LLMs to our required schema.

    Handles these common deviations:
    - "strategy_name" → "name"
    - "entry_trigger" / "entry_condition" → entry.trigger
    - "confirmation_filter" / "entry_filter" → entry.filter
    - indicators[0] / indicators[1] → primary_indicator / confirmation_indicator
      (both str shorthand like "RSI" and full dicts are accepted)
    - "exit_condition" is discarded (we use stop_loss/take_profit)
    - String shorthand for indicator dicts ("RSI" → {"type": "RSI", "params": {"period": 14}})
    """
    out = dict(raw)

    # name
    if "name" not in out and "strategy_name" in out:
        out["name"] = out.pop("strategy_name")

    # indicators list → primary / confirmation
    indicators_list = out.pop("indicators", None)
    if isinstance(indicators_list, list):
        if len(indicators_list) >= 1 and "primary_indicator" not in out:
            ind = indicators_list[0]
            out["primary_indicator"] = (
                {"type": ind.upper(), "params": {"period": 14}} if isinstance(ind, str) else ind
            )
        if len(indicators_list) >= 2 and "confirmation_indicator" not in out:
            ind = indicators_list[1]
            out["confirmation_indicator"] = (
                {"type": ind.upper(), "params": {"period": 14}} if isinstance(ind, str) else ind
            )

    # entry.trigger from top-level entry_trigger / entry_condition
    flat_trigger = out.pop("entry_trigger", None) or out.pop("entry_condition", None)
    if flat_trigger is not None:
        entry = out.get("entry") if isinstance(out.get("entry"), dict) else {}
        if "trigger" not in entry:
            entry["trigger"] = flat_trigger
        out["entry"] = entry

    # entry.filter from top-level confirmation_filter / entry_filter
    flat_filter = out.pop("confirmation_filter", None) or out.pop("entry_filter", None)
    if flat_filter is not None:
        entry = out.get("entry") if isinstance(out.get("entry"), dict) else {}
        if "filter" not in entry:
            entry["filter"] = flat_filter
        out["entry"] = entry

    # exit_condition is ignored — we use stop_loss/take_profit
    out.pop("exit_condition", None)

    # Coerce string shorthand → indicator dicts
    for key in ("primary_indicator", "confirmation_indicator"):
        val = out.get(key)
        if isinstance(val, str):
            out[key] = {"type": val.upper(), "params": {"period": 14}}

    return out


def _extract_and_sanitise(
    raw: str,
    cycle: int,
    offset: int = 0,
    limit: int = 25,
    assigned_combos: list[tuple[str, str, str]] | None = None,
) -> tuple[list[dict], int]:
    """
    Parse the LLM response, normalise each strategy dict, and enforce assigned combos.

    Returns (strategies, deviation_count) where deviation_count is how many strategies
    had their primary_indicator or trigger corrected to match the assigned combo.
    Missing slots are filled with defaults using the assigned combo.
    """
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
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                strategies = json.loads(match.group(0))
            except json.JSONDecodeError:
                logger.error("StrategyBuilderAgent: could not parse JSON from LLM output")

    result: list[dict] = []
    deviations = 0

    for i, s in enumerate(strategies[:limit]):
        if LOCAL_LLM:
            s = normalize_strategy_json(s)
        sanitised = _sanitise(s, offset + i, cycle)

        if assigned_combos and i < len(assigned_combos):
            a_ind, a_trig, a_conf = assigned_combos[i]
            actual_ind  = sanitised["primary_indicator"]["type"]
            actual_trig = sanitised["entry"]["trigger"]
            if actual_ind != a_ind or actual_trig != a_trig:
                deviations += 1
                # Preserve LLM's parameters but enforce the assigned combo
                override = dict(sanitised)
                override["primary_indicator"] = {
                    "type":   a_ind,
                    "params": sanitised["primary_indicator"].get("params", {"period": 14}),
                }
                override["entry"] = {"trigger": a_trig, "filter": a_conf}
                sanitised = _sanitise(override, offset + i, cycle)

        result.append(sanitised)

    # Fill any slots the LLM left empty with sanitised defaults
    if assigned_combos:
        for i in range(len(result), min(limit, len(assigned_combos))):
            a_ind, a_trig, a_conf = assigned_combos[i]
            default = {
                "primary_indicator": {"type": a_ind, "params": {"period": 14}},
                "entry": {"trigger": a_trig, "filter": a_conf},
            }
            result.append(_sanitise(default, offset + i, cycle))

    return result, deviations


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
    if p_type not in VALID_PRIMARY_INDICATORS:
        p_type = "RSI"

    trigger = raw.get("entry", {}).get("trigger", "RSI_OVERSOLD").upper()
    # If the trigger is not compatible with the primary, pick the first valid one.
    valid_triggers_for_primary = _PRIMARY_TRIGGER_MAP.get(p_type, VALID_TRIGGERS)
    if trigger not in valid_triggers_for_primary:
        trigger = valid_triggers_for_primary[0]

    confirm = raw.get("entry", {}).get("filter", "NONE").upper()
    if confirm not in VALID_CONFIRMATIONS:
        confirm = "NONE"

    # Strip directional labels — backtester determines direction empirically
    raw_name = raw.get("name", f"Strategy {index + 1}")
    clean_name = re.sub(r'\b(Long|Short|Bull(?:ish)?|Bear(?:ish)?)\s*', '', raw_name, flags=re.IGNORECASE).strip(" -–")
    name = clean_name if clean_name else raw_name

    return {
        "id":   raw.get("id", f"c{cycle}_s{index+1:02d}"),
        "name": name,
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
