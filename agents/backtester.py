import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone

from anthropic import Anthropic

from agents.worker import WorkerAgent
from backtesting.engine import WARMUP_BARS, run_bidirectional, run_oos_validation, run_recent_window
from backtesting.filters import apply_filters
from backtesting.metrics import compute_metrics, project_returns
from backtesting.models import (
    EvaluatedStrategy,
    Strategy,
    StrategySelection,
)
from backtesting.monte_carlo import run_monte_carlo
from backtesting.scorer import score_strategy
from backtesting.selector import select_top_3
from config.settings import GRAPH_API_KEY
from core.message import Message, MessageType
from data.pipeline import fetch_candles

logger = logging.getLogger(__name__)


class BacktesterAgent(WorkerAgent):
    """Evaluates candidate strategies and selects the single best performer."""

    def __init__(self, client: Anthropic):
        super().__init__(
            name="backtester",
            role=(
                "rigorous quality gate for an autonomous trading network. "
                "You receive 10 candidate strategy JSON objects from the Strategy Builder, "
                "run each against a minimum of 2 years of historical OHLCV data (sourced from "
                "Yahoo Finance or Alpha Vantage, covering at least one bull, bear, and sideways "
                "period, with the final 20% reserved as an out-of-sample test set), apply hard "
                "elimination filters, score surviving strategies across six weighted dimensions, "
                "and select exactly one winner per cycle. "
                "Hard elimination filters -- any failure immediately disqualifies a strategy: "
                "max drawdown exceeds 20% of starting capital; win rate below 35%; fewer than "
                "30 trades in the backtest period; any single trade loss exceeds 5% of account "
                "equity; consecutive losing streak exceeds 8 trades; Sharpe Ratio below 0.5; "
                "strategy only performs well at the 2% risk level. "
                "Scoring dimensions (weights reflect risk-first philosophy): "
                "Risk-Adjusted Return 30% (Sharpe + Sortino Ratio), "
                "Drawdown Profile 25% (depth, duration, recovery time), "
                "Consistency 20% (monthly return variance and win rate stability across regimes), "
                "Profit Factor 15% (gross profit / gross loss -- minimum 1.5 required to score), "
                "Trade Frequency 5% (statistically meaningful without overtrading), "
                "Market Regime Performance 5% (works across bull, bear, and sideways markets). "
                "The highest-scoring survivor is selected, subject to a final check: it must not "
                "be materially similar to a previously deployed underperformer -- if it is, the "
                "second-ranked strategy is chosen instead. "
                "You produce: a full scorecard for all 10 strategies (shared with the Trading "
                "Coach), the winning strategy JSON with complete performance metrics, an "
                "elimination report explaining each rejection, and a confidence rating "
                "(High / Medium / Low) for the selected strategy."
            ),
            client=client,
        )

    def _handle(self, message: Message) -> Message | None:
        if message.type != MessageType.TASK:
            return None

        recipient = message.metadata.get("original_sender", message.sender)

        # -- Try structured backtesting request --------------------------------
        try:
            request       = json.loads(message.content)
            strategies_raw = request["strategies"]
            asset          = request["asset"]
        except (json.JSONDecodeError, KeyError, TypeError):
            # Fallback: plain LLM response for unstructured tasks
            result = self._call_llm(message.content)
            return Message(
                sender=self.name,
                recipient=recipient,
                type=MessageType.RESULT,
                content=result,
            )

        # -- Fetch candle data -------------------------------------------------
        try:
            candles, quality = fetch_candles(
                chain=asset["chain"],
                token_address=asset.get("token_address", ""),
                bucket_seconds=asset.get("bucket_seconds", 3600),
                graph_api_key=GRAPH_API_KEY,
                gecko_pair_address=asset.get("gecko_pair_address", ""),
            )
        except Exception as exc:
            logger.error("BacktesterAgent: data fetch failed -- %s", exc)
            return Message(
                sender=self.name,
                recipient=recipient,
                type=MessageType.ERROR,
                content=f"Data fetch failed: {exc}",
            )

        if len(candles) < 230:
            return Message(
                sender=self.name,
                recipient=recipient,
                type=MessageType.ERROR,
                content=(
                    f"Insufficient data: only {len(candles)} candles available. "
                    "A minimum of 230 bars is required for backtesting."
                ),
            )

        # -- IS / OOS split (80% in-sample, 20% out-of-sample) ----------------
        n_total = len(candles)
        split   = int(n_total * 0.8)
        is_candles  = candles[:split]
        # OOS window includes WARMUP_BARS lookback so indicators can warm up
        oos_candles = candles[max(0, split - WARMUP_BARS):]

        # -- Compute date range strings for logging and result -----------------
        period = _build_period_info(candles, split, asset)
        logger.info(
            "Backtest period  : %s -> %s  (%d bars, %s)",
            period["is_start"], period["oos_end"],
            n_total, period["candle_label"],
        )
        logger.info(
            "In-sample        : %s -> %s  (%d bars) - strategy selection",
            period["is_start"], period["is_end"], split,
        )
        logger.info(
            "Out-of-sample    : %s -> %s  (%d bars) - held-out validation",
            period["oos_start"], period["oos_end"], n_total - split,
        )

        # -- Run engine for each strategy (IS candles only for selection) ------
        evaluated: list[EvaluatedStrategy] = []
        for raw in strategies_raw:
            try:
                strategy = _parse_strategy(raw)
            except Exception as exc:
                logger.warning("Skipping malformed strategy: %s", exc)
                continue

            # Bidirectional: run both long and short, pick the better direction by Sharpe
            risk_results, best, direction = run_bidirectional(
                strategy, is_candles, n_total_bars=n_total
            )
            best.strategy_id = strategy.id
            force_short = (direction == "short")

            # OOS validation in the same direction as IS selection
            oos_sharpe, oos_count, oos_wr = run_oos_validation(
                strategy, oos_candles, best.best_risk_pct, force_short=force_short
            )

            # Recent 5yr window (display-only, uses full candle series)
            recent = run_recent_window(strategy, candles, best.best_risk_pct, force_short=force_short)
            best.oos_sharpe      = oos_sharpe
            best.oos_trade_count = oos_count
            best.oos_win_rate    = oos_wr

            metrics   = compute_metrics(best, is_candles)

            # Monte Carlo drawdown analysis (800 simulations at selected risk level)
            mc_result = run_monte_carlo(best.trades, best.best_risk_pct)

            f_result  = apply_filters(
                metrics, best,
                mc_result=mc_result,
                bucket_seconds=asset.get("bucket_seconds", 14400),
            )
            score     = score_strategy(metrics, strategy.id, mc_result=mc_result) if f_result.passed else None

            # Structured confidence rating
            if f_result.passed and score:
                best.confidence_rating = _compute_confidence(
                    is_sharpe=metrics.sharpe,
                    oos_sharpe=oos_sharpe,
                    only_works_at_max_risk=best.only_works_at_max_risk,
                    data_quality_confidence=quality.confidence,
                )

            evaluated.append(EvaluatedStrategy(
                strategy=strategy,
                backtest=best,
                metrics=metrics,
                filter_result=f_result,
                score=score,
                mc_result=mc_result,
                all_risk_results=risk_results,
                recent_window=recent or None,
            ))
            if f_result.passed:
                logger.info(
                    "  PASS  %-42s | score=%5.1f | return=%+7.1f%% | maxDD=%5.1f%% "
                    "| wr=%4.1f%% | PF=%4.2f | Sharpe(IS/OOS)=%5.2f/%5.2f | trades=%d",
                    strategy.name[:42],
                    score.total if score else 0.0,
                    metrics.total_pnl_pct * 100,
                    metrics.max_drawdown_pct * 100,
                    metrics.win_rate * 100,
                    metrics.profit_factor,
                    metrics.sharpe,
                    oos_sharpe,
                    metrics.trade_count,
                )
            else:
                logger.info(
                    "  FAIL  %-42s | maxDD=%5.1f%% | wr=%4.1f%% | Sharpe=%5.2f | trades=%d"
                    " | reason: %s",
                    strategy.name[:42],
                    metrics.max_drawdown_pct * 100,
                    metrics.win_rate * 100,
                    metrics.sharpe,
                    metrics.trade_count,
                    f_result.failure_reason,
                )

        # -- Select top 3 winners -----------------------------------------------
        past_failures = message.metadata.get("past_failures", [])
        winners = select_top_3(evaluated, past_failures)
        selection = StrategySelection(
            winners=winners,
            all_evaluated=evaluated,
            selection_note=_build_selection_note(winners),
        ) if winners else None

        # -- Deterministic narrative report (no LLM needed — all data is computed) --
        narrative = _build_narrative_report(selection, quality) if selection is not None else ""

        is_years = period.get("years_covered", 1.0)

        # Precompute per-risk-level breakdown for passing strategies only
        _breakdown_cache: dict[str, list[dict]] = {}
        for e in evaluated:
            if e.filter_result.passed and e.all_risk_results:
                _breakdown_cache[e.strategy.id] = _risk_breakdown(e.all_risk_results, is_years)

        result = {
            "winner":             _safe_to_dict(selection.winner, is_years, _breakdown_cache) if selection else None,
            "winners":            [d for d in (_safe_to_dict(w, is_years, _breakdown_cache) for w in winners) if d is not None],
            "runner_up":          _safe_to_dict(selection.runner_up, is_years, _breakdown_cache) if (selection and selection.runner_up) else None,
            "period":             period,
            "elimination_report": _build_elimination_report(evaluated),
            "all_scores":         [
                {
                    "id":              e.strategy.id,
                    "name":            e.strategy.name,
                    "passed":          e.filter_result.passed,
                    "failure":         e.filter_result.failure_reason,
                    "score":           round(e.score.total, 2) if e.score else None,
                    "total_pnl_pct":   round(e.metrics.total_pnl_pct, 4),
                    "total_pnl_usd":   round(e.metrics.total_pnl_usd, 2),
                    "annualised_return_pct": round(_annualise(e.metrics.total_pnl_pct, is_years) * 100, 2),
                    "max_drawdown_pct": round(e.metrics.max_drawdown_pct, 4),
                    "win_rate":        round(e.metrics.win_rate, 4),
                    "profit_factor":   round(e.metrics.profit_factor, 3),
                    "sharpe":          round(e.metrics.sharpe, 3),
                    "oos_sharpe":      round(e.backtest.oos_sharpe, 3),
                    "trade_count":     e.metrics.trade_count,
                    "avg_r":           round(e.metrics.avg_r_multiple, 3),
                    "best_risk_pct":   e.backtest.best_risk_pct,
                    "confidence":      e.backtest.confidence_rating,
                    "mc_p95_dd":       round(e.mc_result.p95_drawdown, 4) if e.mc_result else None,
                    "mc_risk_class":   e.mc_result.risk_class if e.mc_result else None,
                    "prop_firm_8pct":  e.mc_result.prop_firm_8pct if e.mc_result else None,
                    "primary_indicator": e.strategy.primary_indicator.get("type", ""),
                    "entry_trigger":   e.strategy.entry.get("trigger", ""),
                    "confirmation":    e.strategy.entry.get("filter", "NONE"),
                    "direction":       e.backtest.direction,
                    "recent_window":   e.recent_window,
                    "risk_breakdown":  _breakdown_cache.get(e.strategy.id),
                }
                for e in evaluated
            ],
            "data_quality":       asdict(quality),
            "selection_note":     selection.selection_note if selection else "No strategies survived hard filters.",
            "narrative":          narrative,
        }

        return Message(
            sender=self.name,
            recipient=recipient,
            type=MessageType.RESULT,
            content=json.dumps(result, indent=2),
        )


# -- Helpers -------------------------------------------------------------------

def _parse_strategy(raw: dict) -> Strategy:
    return Strategy(
        id=raw.get("id", "unknown"),
        name=raw.get("name", "Unnamed Strategy"),
        primary_indicator=raw.get("primary_indicator", {"type": "RSI", "params": {"period": 14}}),
        confirmation_indicator=raw.get("confirmation_indicator", {"type": "EMA", "params": {"period": 50}}),
        entry=raw.get("entry", {"trigger": "RSI_OVERSOLD", "filter": "NONE"}),
        exit=raw.get("exit", {
            "stop_loss":        {"type": "atr_multiple", "value": 2.0},
            "take_profit":      {"type": "r_multiple", "value": 2.0},
            "trailing_stop_atr": None,
            "time_exit_bars":   None,
        }),
        risk=raw.get("risk", {"max_open_positions": 3}),
        metadata=raw.get("metadata", {}),
    )


def _evaluated_to_dict(
    e: EvaluatedStrategy,
    is_years: float = 1.0,
    risk_breakdown: list | None = None,
) -> dict:
    trades_per_month = e.metrics.trade_count / max(is_years * 12, 1)
    projections = project_returns(
        avg_r=e.metrics.avg_r_multiple,
        risk_pct=e.backtest.best_risk_pct,
        trades_per_month=trades_per_month,
    )
    total_ret = e.metrics.total_pnl_pct
    annual_ret = _annualise(total_ret, is_years)
    return {
        "strategy_id":           e.strategy.id,
        "strategy_name":         e.strategy.name,
        "best_risk_pct":         e.backtest.best_risk_pct,
        "trade_count":           e.metrics.trade_count,
        "sharpe":                round(e.metrics.sharpe, 3),
        "sortino":               round(e.metrics.sortino, 3),
        "max_drawdown":          round(e.metrics.max_drawdown_pct, 4),
        "win_rate":              round(e.metrics.win_rate, 4),
        "profit_factor":         round(e.metrics.profit_factor, 3),
        "avg_r":                 round(e.metrics.avg_r_multiple, 3),
        "total_pnl_pct":         round(total_ret, 4),
        "total_pnl_usd":         round(e.metrics.total_pnl_usd, 2),
        "annualised_return_pct": round(annual_ret * 100, 2),
        "direction":             e.backtest.direction,
        "recent_window":         e.recent_window,
        "score":                 e.score.total if e.score else None,
        "score_detail":          asdict(e.score) if e.score else None,
        # OOS validation metrics
        "oos_sharpe":            round(e.backtest.oos_sharpe, 3),
        "oos_trade_count":       e.backtest.oos_trade_count,
        "oos_win_rate":          round(e.backtest.oos_win_rate, 4),
        "confidence_rating":     e.backtest.confidence_rating,
        # Full strategy schema so the Trader can reconstruct the Strategy object
        "strategy_schema":       asdict(e.strategy),
        # Monte Carlo 95th-pct drawdown (used by Coach for live DD comparison)
        "mc_p95_dd":             round(e.mc_result.p95_drawdown, 4) if e.mc_result else None,
        # Per-risk-level breakdown (all tested levels, not just best)
        "risk_breakdown":        risk_breakdown or [],
        # Return projections at 1/3/6/12/24 months under 3 compounding scenarios
        "projected_monthly_pnl_pct": projections["1m"]["none"],
        "return_projections":        projections,
    }


def _safe_to_dict(
    e: EvaluatedStrategy,
    is_years: float,
    breakdown_cache: dict,
) -> dict | None:
    """Wrap _evaluated_to_dict in a try/except so one failing winner doesn't drop others."""
    try:
        return _evaluated_to_dict(e, is_years, breakdown_cache.get(e.strategy.id))
    except Exception as exc:
        logger.error("Failed to serialise winner %s: %s", e.strategy.id, exc)
        return None


def _annualise(total_ret: float, is_years: float) -> float:
    """Annualise a total return fraction. Returns -1.0 if total loss >= 100%."""
    if total_ret <= -1.0:
        return -1.0
    return ((1 + total_ret) ** (1 / max(is_years, 0.01))) - 1


def _risk_breakdown(risk_results: list, is_years: float) -> list[dict]:
    """Compute per-risk-level return, max DD, Sharpe, and MC95 for display."""
    from backtesting.monte_carlo import run_monte_carlo
    breakdown = []
    for r in risk_results:
        eq = r.equity_curve
        if not eq:
            continue
        initial = eq[0]
        total_ret = (eq[-1] - initial) / initial if initial > 0 else 0.0
        annual_ret = _annualise(total_ret, is_years)
        peak, max_dd = initial, 0.0
        for v in eq:
            if v > peak:
                peak = v
            if peak > 0:
                dd = (peak - v) / peak
                if dd > max_dd:
                    max_dd = dd
        mc = run_monte_carlo(r.trades, r.risk_pct) if r.trades else None
        breakdown.append({
            "risk_pct":          r.risk_pct,
            "total_return_pct":  round(total_ret * 100, 2),
            "annual_return_pct": round(annual_ret * 100, 2),
            "max_drawdown_pct":  round(max_dd * 100, 2),
            "sharpe":            round(r.sharpe, 3),
            "mc_p95_dd":         round(mc.p95_drawdown * 100, 2) if mc else None,
        })
    return breakdown


def _ts_to_date(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def _build_period_info(candles: list, split: int, asset: dict) -> dict:
    """Build a human-readable period summary for display in scorecard and logs."""
    bucket = asset.get("bucket_seconds", 14400)
    if bucket >= 86400:
        candle_label = "daily bars"
    elif bucket >= 14400:
        candle_label = "4-hour bars"
    elif bucket >= 3600:
        candle_label = "1-hour bars"
    else:
        candle_label = f"{bucket}s bars"

    # Duration in years (approx, based on timestamp delta)
    ts_start = candles[0].ts
    ts_end   = candles[-1].ts
    years    = (ts_end - ts_start) / (1000 * 86400 * 365.25)

    return {
        "is_start":     _ts_to_date(candles[0].ts),
        "is_end":       _ts_to_date(candles[split - 1].ts),
        "oos_start":    _ts_to_date(candles[split].ts),
        "oos_end":      _ts_to_date(candles[-1].ts),
        "total_candles": len(candles),
        "is_candles":   split,
        "oos_candles":  len(candles) - split,
        "bucket_seconds": bucket,
        "candle_label": candle_label,
        "years_covered": round(years, 1),
        "chain":        asset.get("chain", "?"),
        "symbol":       asset.get("token_address", ""),
    }


def _compute_confidence(
    is_sharpe: float,
    oos_sharpe: float,
    only_works_at_max_risk: bool,
    data_quality_confidence: str,
) -> str:
    """
    Assign a structured confidence rating for the selected strategy.
    High:   data quality high, OOS/IS Sharpe >= 0.7, not only_max_risk
    Low:    data quality low OR OOS/IS < 0.4 OR only_max_risk
    Medium: everything else
    """
    if only_works_at_max_risk:
        return "Low"
    if data_quality_confidence == "low":
        return "Low"
    oos_is_ratio = oos_sharpe / is_sharpe if is_sharpe > 0 else 0.0
    if oos_is_ratio < 0.4:
        return "Low"
    if data_quality_confidence == "high" and oos_is_ratio >= 0.7:
        return "High"
    return "Medium"


def _build_selection_note(winners: list[EvaluatedStrategy]) -> str:
    if not winners:
        return "No strategies survived hard filters."
    parts = []
    for i, w in enumerate(winners, 1):
        parts.append(
            f"#{i} '{w.strategy.name}' (score {w.score.total:.1f}, "
            f"Sharpe {w.metrics.sharpe:.2f}, risk {w.backtest.best_risk_pct}%)"
        )
    return "Selected: " + " | ".join(parts)


def _build_elimination_report(evaluated: list[EvaluatedStrategy]) -> list[dict]:
    return [
        {
            "strategy_id":   e.strategy.id,
            "strategy_name": e.strategy.name,
            "passed":        e.filter_result.passed,
            "reason":        e.filter_result.failure_reason,
        }
        for e in evaluated
        if not e.filter_result.passed
    ]


def _build_narrative_report(selection: StrategySelection, quality) -> str:
    """
    Build a concise evaluation report using only computed data — no LLM call required.
    All metrics are deterministic; no reasoning is needed here.
    """
    top = selection.winner
    eliminated = [e for e in selection.all_evaluated if not e.filter_result.passed]

    oos_is_pct = (
        top.backtest.oos_sharpe / top.metrics.sharpe * 100
        if top.metrics.sharpe > 0 else 0.0
    )
    robustness = (
        "strong OOS robustness"          if oos_is_pct >= 70
        else "acceptable OOS degradation" if oos_is_pct >= 40
        else "significant OOS degradation — deploy cautiously"
    )

    mc_note = ""
    if top.mc_result:
        prop = "passing" if top.mc_result.prop_firm_8pct else "failing"
        mc_note = (
            f"Monte Carlo 95th-pct DD {top.mc_result.p95_drawdown:.1%} "
            f"({top.mc_result.risk_class} risk, {prop} 8% prop rule). "
        )

    # Tally most common elimination reason
    reason_counts: dict[str, int] = {}
    for e in eliminated:
        key = (e.filter_result.failure_reason or "unknown").split(":")[0].strip()
        reason_counts[key] = reason_counts.get(key, 0) + 1
    top_reason = max(reason_counts, key=reason_counts.get) if reason_counts else "various"

    n_winners = len(selection.winners)
    winner_names = ", ".join(f"'{w.strategy.name}'" for w in selection.winners)

    sentences = [
        f"Selected {n_winners} winner{'s' if n_winners > 1 else ''}: {winner_names}.",
        (
            f"Top-ranked '{top.strategy.name}' scored {top.score.total:.1f}/100 with EV "
            f"{top.metrics.avg_r_multiple:.2f}R/trade, Sharpe {top.metrics.sharpe:.2f} IS / "
            f"{top.backtest.oos_sharpe:.2f} OOS ({oos_is_pct:.0f}%) — {robustness}."
        ),
        f"{mc_note}Win rate {top.metrics.win_rate:.1%}, PF {top.metrics.profit_factor:.2f}, "
        f"max DD {top.metrics.max_drawdown_pct:.1%}, {top.metrics.trade_count} IS trades.",
        f"{len(eliminated)} strategies eliminated; most common failure: {top_reason}.",
        f"Confidence: {top.backtest.confidence_rating} | Data quality: {quality.confidence}.",
    ]
    return " ".join(s for s in sentences if s.strip())