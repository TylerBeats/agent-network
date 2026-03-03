import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone

from anthropic import Anthropic

from agents.worker import WorkerAgent
from core.message import Message, MessageType
from performance.feedback import build_feedback_report
from performance.tracker import (
    WeeklyReview,
    compute_monthly_review,
    compute_per_strategy_reviews,
    compute_weekly_review,
)

logger = logging.getLogger(__name__)


class TradingCoachAgent(WorkerAgent):
    """Monitors performance, manages risk exposure, and drives continuous improvement."""

    def __init__(self, client: Anthropic):
        super().__init__(
            name="trading_coach",
            role=(
                "strategic intelligence layer of an autonomous trading network -- the closest "
                "thing to a human fund manager in the system. You step back and ask: is this "
                "working? You review performance across all active strategies weekly and monthly, "
                "identify what is working and what is not, adjust risk exposure dynamically, and "
                "feed actionable insights back to the Strategy Builder to improve the next "
                "generation of strategies. You monitor for strategy decay -- a previously good "
                "strategy losing its edge over time -- and maintain the performance history log "
                "used by all other agents. "
                "Weekly review metrics: weekly P&L (absolute and percentage), win rate vs "
                "backtest expectation, average R-multiple vs projected EV, max weekly drawdown "
                "vs MC 95th-pct drawdown, actual trade count vs expected frequency (divergence "
                "signals regime change). "
                "Monthly review metrics: monthly P&L with benchmark comparison, live Sharpe "
                "vs backtest Sharpe (significant gap signals decay), health tier classification "
                "(Good/Marginal/Poor) per strategy, drawdown analysis vs MC p95 drawdown, and "
                "compounding mode management (no-compounding → monthly after 3 months Good-tier "
                "→ per-trade after 6 months Good-tier). "
                "Risk exposure adjustment rules: per-strategy multipliers based on health tier "
                "(Good: monthly P&L-based; Marginal: 0.75x; Poor: 0.5x); combined 5% daily halt; "
                "retire strategy after 3 consecutive losing weeks. "
                "Structured feedback to Strategy Builder each cycle must contain: best-performing "
                "indicator combinations, best R/R ratios, failing regimes, health tier summary, "
                "compounding mode updates, and a specific hypothesis for the next batch."
            ),
            client=client,
        )

    def _handle(self, message: Message) -> Message | None:
        if message.type != MessageType.TASK:
            return None

        recipient = message.metadata.get("original_sender", message.sender)

        # -- Try structured review request -------------------------------------
        try:
            request             = json.loads(message.content)
            trade_log           = request.get("trade_log", [])
            backtest_sharpe     = float(request.get("backtest_sharpe", 1.0))
            cycle               = int(request.get("cycle", 0))
            performance_history = request.get("performance_history", [])
            projected_avg_r     = float(request.get("projected_avg_r", 0.0))
            projected_win_rate  = float(request.get("projected_win_rate", 0.0))
            mc_p95_dd_by_strategy  = request.get("mc_p95_dd_by_strategy", {})
            months_at_good_tier    = request.get("months_at_good_tier", {})
            per_strategy_compounding = request.get("per_strategy_compounding", {})
        except (json.JSONDecodeError, TypeError):
            # Fallback: plain LLM response for unstructured tasks
            result = self._call_llm(message.content)
            return Message(
                sender=self.name,
                recipient=recipient,
                type=MessageType.RESULT,
                content=result,
            )

        # -- Compute current week/month strings --------------------------------
        now = datetime.now(tz=timezone.utc)
        year, week, _ = now.isocalendar()
        week_str = f"{year}-W{week:02d}"

        # -- Compute aggregate weekly review -----------------------------------
        # Use the best winner's projected metrics as the benchmark
        weekly_review = compute_weekly_review(
            trade_log, week_str,
            projected_avg_r=projected_avg_r,
            projected_win_rate=projected_win_rate,
            mc_p95_dd=next(iter(mc_p95_dd_by_strategy.values()), None),
        )

        # -- Compute per-strategy weekly reviews -------------------------------
        per_strategy_reviews = compute_per_strategy_reviews(
            trade_log, week_str,
            starting_equity=10_000.0,
            projected_avg_r=projected_avg_r,
            projected_win_rate=projected_win_rate,
            mc_p95_dd_by_strategy=mc_p95_dd_by_strategy,
        )

        # -- Increment months_at_good_tier for strategies at Good tier ---------
        updated_months = dict(months_at_good_tier)
        for sid, review in per_strategy_reviews.items():
            if review.health_tier == "Good":
                updated_months[sid] = updated_months.get(sid, 0) + 1
            else:
                # reset tenure — must re-earn Good tier continuously
                updated_months[sid] = 0

        # -- Compute monthly review --------------------------------------------
        prior_weekly: list[WeeklyReview] = []
        for h in performance_history:
            wr = h.get("weekly_review")
            if wr:
                try:
                    prior_weekly.append(WeeklyReview(**wr))
                except TypeError:
                    pass
        all_weekly = prior_weekly + [weekly_review]
        monthly_review = compute_monthly_review(all_weekly, backtest_sharpe)

        # -- Build quantitative feedback report --------------------------------
        feedback = build_feedback_report(
            monthly_review, performance_history, cycle,
            per_strategy_reviews=per_strategy_reviews,
            months_at_good_tier=updated_months,
            per_strategy_compounding=per_strategy_compounding,
        )

        # -- LLM generates the narrative and next-cycle hypothesis -------------
        llm_prompt = _build_coach_prompt(
            weekly_review, monthly_review, feedback,
            performance_history, per_strategy_reviews,
        )
        narrative = self._call_llm(llm_prompt)

        feedback.next_cycle_hypothesis = narrative

        result = {
            "feedback":                 asdict(feedback),
            "monthly_review":           asdict(monthly_review),
            "weekly_review":            asdict(weekly_review),
            "per_strategy_reviews":     {sid: asdict(r) for sid, r in per_strategy_reviews.items()},
            "health_tiers":             feedback.health_tiers,
            "per_strategy_adjustments": feedback.per_strategy_adjustments,
            "compounding_modes":        feedback.compounding_modes,
            "months_at_good_tier":      updated_months,
            "narrative":                narrative,
        }

        logger.info(
            "TradingCoachAgent: cycle %d -- weekly PnL=%.2f%%, monthly_tier=%s, "
            "risk_adj=%.2f, retire=%s, per_strategy=%d",
            cycle,
            weekly_review.pnl_pct * 100,
            monthly_review.health_tier,
            monthly_review.risk_adjustment,
            monthly_review.retire,
            len(per_strategy_reviews),
        )

        return Message(
            sender=self.name,
            recipient=recipient,
            type=MessageType.RESULT,
            content=json.dumps(result, indent=2),
        )


# -- Helpers -------------------------------------------------------------------

def _build_coach_prompt(
    weekly_review: WeeklyReview,
    monthly_review,
    feedback,
    performance_history: list[dict],
    per_strategy_reviews: dict[str, WeeklyReview] | None = None,
) -> str:
    prior_cycles = len(performance_history)
    per_strategy_reviews = per_strategy_reviews or {}

    # Build per-strategy health section
    strat_lines = ""
    if per_strategy_reviews:
        lines = []
        for sid, r in per_strategy_reviews.items():
            lines.append(
                f"  {sid}: tier={r.health_tier}, "
                f"EV drift={r.ev_drift_r:+.2f}R, "
                f"WR drift={r.win_rate_drift:+.1%}, "
                f"DD/MC={r.live_dd_vs_mc_ratio:.2f}x"
            )
        strat_lines = "PER-STRATEGY HEALTH:\n" + "\n".join(lines) + "\n\n"

    return (
        f"You are the Trading Coach. Produce a concise performance review and strategy hypothesis.\n\n"
        f"WEEKLY REVIEW (week {weekly_review.week}):\n"
        f"  Trades: {weekly_review.trade_count}  |  "
        f"PnL: {weekly_review.pnl_usd:+.2f} USD ({weekly_review.pnl_pct:.1%})  |  "
        f"Win rate: {weekly_review.win_rate:.1%}  |  Avg R: {weekly_review.avg_r_multiple:.2f}\n"
        f"  Max drawdown: {weekly_review.max_drawdown_pct:.1%}  |  "
        f"EV drift: {weekly_review.ev_drift_r:+.2f}R  |  "
        f"WR drift: {weekly_review.win_rate_drift:+.1%}  |  "
        f"Health: {weekly_review.health_tier}\n\n"
        f"MONTHLY REVIEW ({monthly_review.month}):\n"
        f"  Total PnL: {monthly_review.pnl_pct:.1%}  |  "
        f"Live Sharpe: {monthly_review.live_sharpe:.2f}  |  "
        f"Backtest Sharpe: {monthly_review.backtest_sharpe:.2f}\n"
        f"  Sharpe decay: {monthly_review.sharpe_decay:.2f}  |  "
        f"Max DD: {monthly_review.max_drawdown_pct:.1%}  |  "
        f"Health: {monthly_review.health_tier}\n"
        f"  Consecutive losing weeks: {monthly_review.consecutive_losing_weeks}  |  "
        f"Risk adjustment: {monthly_review.risk_adjustment:.2f}x  |  "
        f"Retire flag: {monthly_review.retire}\n\n"
        f"{strat_lines}"
        f"FEEDBACK SUMMARY:\n"
        f"  Best indicator types so far: {', '.join(feedback.best_indicator_types) or 'none yet'}\n"
        f"  Best R/R ratio achieved: {feedback.best_rr_ratio:.2f}\n"
        f"  Failing regimes: {', '.join(feedback.failing_regimes) or 'none identified'}\n"
        f"  Retired patterns: {len(feedback.retired_patterns)}\n"
        f"  Prior cycles in history: {prior_cycles}\n\n"
        f"Write 3–5 sentences covering:\n"
        f"1. What worked and what did not this cycle, referencing health tier if degraded\n"
        f"2. A specific hypothesis for the next strategy batch (which indicator type, "
        f"timeframe, or market regime to focus on)\n"
        f"3. Any risk management observations, including compounding mode or sizing changes\n"
        f"Be concrete and actionable. The Strategy Builder will use your hypothesis directly."
    )
