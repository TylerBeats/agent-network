"""
Strategy selection logic.

select_top_3(): Primary function -- selects up to 3 decorrelated winners.
select_winner(): Legacy single-winner wrapper (kept for backward compat in tests).
"""
from __future__ import annotations

from backtesting.models import EvaluatedStrategy, StrategySelection


def select_top_3(
    evaluated: list[EvaluatedStrategy],
    past_failures: list[dict] | None = None,
) -> list[EvaluatedStrategy]:
    """
    Select up to 3 passing strategies, ensuring low inter-strategy correlation.

    Selection rules:
    1. Only strategies that passed all hard filters are considered.
    2. Ranked by total score descending.
    3. Strategies materially similar to past deployed underperformers are skipped.
    4. Strategies correlated with an already-selected winner are skipped.
    5. Returns 0-3 strategies (fewer if fewer pass filters).

    Two strategies are considered correlated if they share the same
    primary indicator type AND the same entry trigger.

    Args:
        evaluated:     All evaluated strategies from the current cycle.
        past_failures: Previously deployed strategies that underperformed.
                       Each dict has keys "primary_indicator_type" and "entry_trigger".

    Returns:
        Ordered list of up to 3 EvaluatedStrategy (highest score first).
    """
    past_failures = past_failures or []

    passing = sorted(
        [e for e in evaluated if e.filter_result.passed and e.score is not None],
        key=lambda e: e.score.total,
        reverse=True,
    )

    selected: list[EvaluatedStrategy] = []
    for candidate in passing:
        if len(selected) >= 3:
            break
        if _is_similar_to_past_failure(candidate, past_failures):
            continue
        if any(_are_correlated(candidate, s) for s in selected):
            continue
        selected.append(candidate)

    return selected


def select_winner(
    evaluated: list[EvaluatedStrategy],
    past_failures: list[dict] | None = None,
) -> StrategySelection:
    """
    Legacy single-winner interface. Returns a StrategySelection wrapping select_top_3.
    Raises ValueError if no strategy passes the hard filters (matches original behaviour).
    """
    past_failures = past_failures or []
    winners = select_top_3(evaluated, past_failures)

    if not winners:
        eliminated = [(e.strategy.name, e.filter_result.failure_reason) for e in evaluated]
        reasons = "; ".join(f"{n}: {r}" for n, r in eliminated)
        raise ValueError(
            f"No strategies survived the hard elimination filters. "
            f"Elimination reasons: {reasons}"
        )

    parts = []
    for i, w in enumerate(winners, 1):
        parts.append(
            f"#{i} '{w.strategy.name}' (score {w.score.total:.1f}, "
            f"Sharpe {w.metrics.sharpe:.2f}, risk {w.backtest.best_risk_pct}%)"
        )
    note = "Selected: " + " | ".join(parts)

    return StrategySelection(
        winners=winners,
        all_evaluated=evaluated,
        selection_note=note,
    )


def _are_correlated(a: EvaluatedStrategy, b: EvaluatedStrategy) -> bool:
    """
    Two strategies are correlated if they share the same primary indicator type
    AND the same entry trigger -- they would produce nearly identical signals.
    """
    return (
        a.strategy.primary_indicator.get("type", "").upper()
        == b.strategy.primary_indicator.get("type", "").upper()
        and a.strategy.entry.get("trigger", "").upper()
        == b.strategy.entry.get("trigger", "").upper()
    )


def _is_similar_to_past_failure(
    candidate: EvaluatedStrategy,
    past_failures: list[dict],
) -> bool:
    """
    A candidate is 'materially similar' to a past failure if it shares both
    the same primary indicator type AND the same entry trigger.
    """
    c_indicator = candidate.strategy.primary_indicator.get("type", "").upper()
    c_trigger   = candidate.strategy.entry.get("trigger", "").upper()

    for failure in past_failures:
        f_indicator = str(failure.get("primary_indicator_type", "")).upper()
        f_trigger   = str(failure.get("entry_trigger", "")).upper()
        if c_indicator == f_indicator and c_trigger == f_trigger:
            return True
    return False
