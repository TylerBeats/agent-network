import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone

from anthropic import Anthropic

from agents.worker import WorkerAgent
from backtesting.engine import WARMUP_BARS, pick_best_risk_level, run_all_risk_levels, run_oos_validation
from backtesting.filters import apply_filters
from backtesting.metrics import compute_metrics
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

            risk_results = run_all_risk_levels(strategy, is_candles)
            best         = pick_best_risk_level(risk_results, n_total_bars=n_total)
            best.strategy_id = strategy.id

            # OOS validation at the selected risk level
            oos_sharpe, oos_count, oos_wr = run_oos_validation(
                strategy, oos_candles, best.best_risk_pct
            )
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

        # -- LLM generates the narrative report (only when there is a winner) --
        narrative = ""
        if selection is not None:
            llm_prompt = _build_llm_prompt(selection, quality)
            narrative  = self._call_llm(llm_prompt)

        result = {
            "winner":             _evaluated_to_dict(selection.winner) if selection else None,
            "winners":            [_evaluated_to_dict(w) for w in winners],
            "runner_up":          _evaluated_to_dict(selection.runner_up) if (selection and selection.runner_up) else None,
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


def _evaluated_to_dict(e: EvaluatedStrategy) -> dict:
    return {
        "strategy_id":       e.strategy.id,
        "strategy_name":     e.strategy.name,
        "best_risk_pct":     e.backtest.best_risk_pct,
        "trade_count":       e.metrics.trade_count,
        "sharpe":            round(e.metrics.sharpe, 3),
        "sortino":           round(e.metrics.sortino, 3),
        "max_drawdown":      round(e.metrics.max_drawdown_pct, 4),
        "win_rate":          round(e.metrics.win_rate, 4),
        "profit_factor":     round(e.metrics.profit_factor, 3),
        "avg_r":             round(e.metrics.avg_r_multiple, 3),
        "total_pnl_pct":     round(e.metrics.total_pnl_pct, 4),
        "total_pnl_usd":     round(e.metrics.total_pnl_usd, 2),
        "score":             e.score.total if e.score else None,
        "score_detail":      asdict(e.score) if e.score else None,
        # OOS validation metrics
        "oos_sharpe":        round(e.backtest.oos_sharpe, 3),
        "oos_trade_count":   e.backtest.oos_trade_count,
        "oos_win_rate":      round(e.backtest.oos_win_rate, 4),
        "confidence_rating": e.backtest.confidence_rating,
        # Full strategy schema so the Trader can reconstruct the Strategy object
        "strategy_schema":   asdict(e.strategy),
        # Monte Carlo 95th-pct drawdown (used by Coach for live DD comparison)
        "mc_p95_dd":         round(e.mc_result.p95_drawdown, 4) if e.mc_result else None,
    }


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


def _build_llm_prompt(selection: StrategySelection, quality) -> str:
    eliminated = [e for e in selection.all_evaluated if not e.filter_result.passed]
    winner_lines = []
    for i, w in enumerate(selection.winners, 1):
        oos_is_pct = (
            f"{(w.backtest.oos_sharpe / w.metrics.sharpe * 100):.0f}%"
            if w.metrics.sharpe > 0 else "n/a"
        )
        mc_note = ""
        if w.mc_result:
            mc_note = (
                f"  MC 95th pct DD: {w.mc_result.p95_drawdown:.1%}  "
                f"Risk class: {w.mc_result.risk_class}  "
                f"8% prop rule: {'PASS' if w.mc_result.prop_firm_8pct else 'FAIL'}\n"
            )
        winner_lines.append(
            f"WINNER #{i}: {w.strategy.name}\n"
            f"  Score: {w.score.total:.1f}/100  EV: {w.metrics.avg_r_multiple:.2f}R/trade\n"
            f"  Risk level: {w.backtest.best_risk_pct}%\n"
            f"  IS Sharpe: {w.metrics.sharpe:.2f}  IS Sortino: {w.metrics.sortino:.2f}\n"
            f"  OOS Sharpe: {w.backtest.oos_sharpe:.2f}  OOS/IS: {oos_is_pct}\n"
            f"  Max drawdown: {w.metrics.max_drawdown_pct:.1%}  "
            f"Win rate: {w.metrics.win_rate:.1%}  PF: {w.metrics.profit_factor:.2f}\n"
            f"  Trades: {w.metrics.trade_count}  Confidence: {w.backtest.confidence_rating}\n"
            f"{mc_note}"
        )
    top = selection.winner
    elim_lines = "\n".join(
        f"  - {e.strategy.name}: {e.filter_result.failure_reason}" for e in eliminated
    )
    return (
        "You are the Backtester agent. Produce a concise evaluation report.\n\n"
        + "\n".join(winner_lines)
        + f"\nELIMINATED ({len(eliminated)} strategies):\n"
        + elim_lines
        + f"\n\nDATA QUALITY: confidence={quality.confidence}, warnings={quality.warnings}\n\n"
        "Write a 3-5 sentence evaluation report explaining: why the top-ranked strategy was "
        "selected, what its key strengths are (EV per trade and Monte Carlo risk class), "
        "how OOS performance compares to IS, and the main elimination reasons. "
        f"The confidence rating for the primary winner is '{top.backtest.confidence_rating}' -- "
        "confirm or briefly explain this rating."
    )