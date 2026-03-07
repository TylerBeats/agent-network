import json
import logging

from anthropic import Anthropic

from agents.worker import WorkerAgent
from backtesting.models import Strategy
from backtesting.signals import (
    calc_stop_price,
    calc_take_profit_price,
    check_entry,
    compute_indicators,
    is_short_trigger,
)
from config.settings import (
    ALPACA_API_KEY,
    ALPACA_API_SECRET,
    ASSET_BUCKET_SECONDS,
    BROKER_MODE,
    GRAPH_API_KEY,
)
from core.message import Message, MessageType
from data.pipeline import fetch_candles
from execution.broker import TradeOrder
from execution.dry_run import DryRunBroker

logger = logging.getLogger(__name__)


class TraderAgent(WorkerAgent):
    """Executes approved strategies in live or paper markets with precision."""

    def __init__(self, client: Anthropic):
        super().__init__(
            name="trader",
            role=(
                "precise execution arm of an autonomous trading network. "
                "You receive the Backtester's selected strategy and implement it faithfully in "
                "real or simulated markets (Alpaca Markets for development, Interactive Brokers "
                "for professional deployments). You do not deviate from the strategy, do not "
                "make discretionary decisions, and do not override its risk parameters -- doing "
                "so would invalidate the backtest entirely. Your job is flawless execution, "
                "complete trade logging, and surfacing real-world performance data for the "
                "Trading Coach. "
                "Hard safety overrides you enforce regardless of strategy instruction: "
                "daily halt if account equity drops 5% in a single day (no exceptions); "
                "reduce position sizes by 50% for the remainder of the week if weekly drawdown "
                "reaches 8%; no single trade may risk more than 2% of account equity; "
                "pause all new entries if the Trading Coach flags a major market regime shift; "
                "close open positions if possible and halt all activity if broker connection is lost. "
                "Pre-trade checklist before any execution: all strategy parameters confirmed and "
                "loaded; daily loss limit not reached; open position count below maximum; new "
                "trade is not highly correlated with existing open positions; market is open with "
                "sufficient liquidity; no entry within 30 minutes of major economic announcements. "
                "Execution rules: entry orders are limit orders by default (market orders only if "
                "price moves away within 5 seconds); stop loss placed immediately upon entry -- "
                "never a mental stop; take profit target set immediately upon entry; position size "
                "calculated precisely from the risk per trade parameter, never rounded up; partial "
                "exits permitted if the strategy defines scaling-out rules. "
                "You produce: a complete trade log with full entry/exit details for every trade, "
                "a daily P&L summary, a weekly performance report delivered to the Trading Coach, "
                "and anomaly alerts for human review when conditions warrant."
            ),
            client=client,
        )

    def _handle(self, message: Message) -> Message | None:
        if message.type != MessageType.TASK:
            return None

        recipient = message.metadata.get("original_sender", message.sender)

        # -- Parse request ---------------------------------------------------------
        try:
            request = json.loads(message.content)
            # Accept list of strategies (new) or single strategy (backward compat)
            strategies_list = request.get("strategies") or []
            if not strategies_list and "strategy" in request:
                strategies_list = [request["strategy"]]
            strategies_list = [s for s in strategies_list if s]
            asset = request["asset"]
            risk_multiplier = float(request.get("risk_multiplier", 1.0))
            if not strategies_list:
                raise KeyError("No strategies provided")
        except (json.JSONDecodeError, KeyError, TypeError):
            return Message(
                sender=self.name,
                recipient=recipient,
                type=MessageType.ERROR,
                content="TraderAgent: malformed request — expected JSON with 'strategies' and 'asset' keys.",
            )

        # -- Initialise broker -----------------------------------------------------
        broker = _make_broker()

        # -- Combined daily halt (5% account loss -> halt everything) --------------
        if broker.is_daily_halt():
            logger.warning("TraderAgent: daily halt triggered -- account down >5%%")
            return Message(
                sender=self.name,
                recipient=recipient,
                type=MessageType.ERROR,
                content="Daily halt: account down >5% today. No new trades placed.",
            )

        # -- Weekly drawdown reduction (>=8% weekly loss -> 50% size) --------------
        weekly_risk_multiplier = 1.0
        if hasattr(broker, "is_weekly_reduction") and broker.is_weekly_reduction():
            weekly_risk_multiplier = 0.5
            logger.warning(
                "TraderAgent: weekly drawdown >=8%% -- position size reduced by 50%% for remainder of week"
            )

        # -- Fetch latest candles (shared across all strategies) -------------------
        try:
            candles, _ = fetch_candles(
                chain=asset.get("chain", "pulsechain"),
                token_address=asset.get("token_address", ""),
                bucket_seconds=asset.get("bucket_seconds", ASSET_BUCKET_SECONDS),
                graph_api_key=GRAPH_API_KEY,
                gecko_pair_address=asset.get("gecko_pair_address", ""),
            )
        except Exception as exc:
            logger.error("TraderAgent: data fetch failed -- %s", exc)
            return Message(
                sender=self.name,
                recipient=recipient,
                type=MessageType.ERROR,
                content=f"Data fetch failed: {exc}",
            )

        current_price = candles[-1].close if candles else 0.0

        # -- Check exits on all open positions (once, before entries) --------------
        exit_fills = []
        if candles:
            try:
                exit_fills = broker.check_exits(current_price)
                if exit_fills:
                    logger.info("TraderAgent: closed %d position(s) at %.6f", len(exit_fills), current_price)
            except Exception as exc:
                logger.warning("TraderAgent: exit check failed -- %s", exc)

        # Refresh equity after exits so exposure checks are current
        equity = broker.get_equity()

        # -- Per-strategy entry loop -----------------------------------------------
        actions: list[str] = []

        for strategy_dict in strategies_list:
            strat_id = strategy_dict.get("strategy_id", "unknown")

            # Per-strategy daily halt: >3% loss on this strategy today
            if hasattr(broker, "check_strategy_daily_halt") and \
                    broker.check_strategy_daily_halt(strat_id, equity):
                logger.warning("TraderAgent: strategy %s halted (>3%% daily loss)", strat_id)
                actions.append("halted")
                continue

            if len(candles) < 50:
                actions.append("insufficient_data")
                continue

            try:
                strategy   = _parse_strategy(strategy_dict)
                indicators = compute_indicators(strategy, candles)
                bar        = len(candles) - 1

                if not check_entry(strategy, indicators, bar):
                    actions.append("no_signal")
                    continue

                short = is_short_trigger(strategy.entry.get("trigger", ""))
                side  = "sell" if short else "buy"
                # Short fills slightly below market (receive less); long fills slightly above (pay more)
                entry_price   = current_price * (1.0 - DryRunBroker.SLIPPAGE_PCT if short
                                                 else 1.0 + DryRunBroker.SLIPPAGE_PCT)
                stop_price    = calc_stop_price(strategy, entry_price, indicators, bar)
                tp_price      = calc_take_profit_price(strategy, entry_price, stop_price)
                risk_per_unit = abs(entry_price - stop_price)

                if risk_per_unit <= 0:
                    actions.append("skipped:zero_risk")
                    continue

                # Apply per-strategy multiplier, global multiplier, weekly reduction, and hard 2% cap
                base_risk       = float(strategy_dict.get("best_risk_pct", 1.0))
                strat_mult      = float(
                    request.get("per_strategy_adjustments", {}).get(strat_id, 1.0)
                )
                effective_risk_pct = min(
                    base_risk * strat_mult * risk_multiplier * weekly_risk_multiplier, 2.0
                )
                # Compounding mode: "none" uses fixed INITIAL_EQUITY as sizing base;
                # "monthly" / "per_trade" use current equity (compound growth)
                compounding_mode = request.get("per_strategy_compounding", {}).get(strat_id, "none")
                base_equity = equity if compounding_mode != "none" else DryRunBroker.INITIAL_EQUITY
                risk_usd    = base_equity * (effective_risk_pct / 100.0)

                # Combined exposure cap: total open risk must not exceed 6% of equity
                if hasattr(broker, "check_combined_exposure") and \
                        broker.check_combined_exposure(risk_usd, equity):
                    actions.append("skipped:exposure_limit")
                    continue

                # Max open positions (strategy-level limit)
                max_positions = strategy_dict.get("strategy_schema", {}).get(
                    "risk", {}
                ).get("max_open_positions", 3)
                if len(broker.get_open_positions()) >= max_positions:
                    actions.append("skipped:max_positions")
                    continue

                qty    = risk_usd / risk_per_unit
                symbol = asset.get("token_address", asset.get("chain", "unknown"))
                order = TradeOrder(
                    strategy_id=strat_id,
                    symbol=symbol,
                    side=side,
                    qty=qty,
                    order_type="limit",
                    limit_price=entry_price,
                    stop_price=stop_price,
                    take_profit=tp_price,
                    risk_usd=risk_usd,
                )

                # Directional correlation check — log same-symbol same-direction positions.
                # The combined exposure cap above is the hard enforcement; this is informational.
                open_positions = broker.get_open_positions()
                same_dir = [
                    p for p in open_positions
                    if p.get("symbol") == symbol and p.get("side") == order.side
                ]
                if same_dir:
                    logger.info(
                        "TraderAgent [%s]: %d correlated same-direction position(s) open "
                        "in %s — proceeding (exposure cap enforced above)",
                        strat_id, len(same_dir), symbol,
                    )

                fill = broker.place_order(order)
                actions.append("entry")
                logger.info(
                    "TraderAgent [%s]: entered at %.6f (stop=%.6f, tp=%.6f, risk=$%.2f)",
                    strat_id, fill.filled_price, stop_price, tp_price, risk_usd,
                )

            except Exception as exc:
                logger.error("TraderAgent [%s]: entry evaluation failed -- %s", strat_id, exc)
                actions.append("error")

        result = {
            "trade_log":              broker.get_trade_log(),
            "exit_count":             len(exit_fills),
            "actions":                actions,
            # backward-compat single-strategy key
            "entry":                  {"action": actions[0] if actions else "no_signal"},
            "daily_pnl":              broker.get_daily_pnl(),
            "equity":                 broker.get_equity(),
            "active_positions":       broker.get_open_positions(),
            "weekly_risk_multiplier": weekly_risk_multiplier,
        }

        return Message(
            sender=self.name,
            recipient=recipient,
            type=MessageType.RESULT,
            content=json.dumps(result, indent=2),
        )


# -- Helpers -------------------------------------------------------------------

def _make_broker():
    """Return the configured broker instance."""
    if BROKER_MODE == "alpaca":
        from execution.alpaca import AlpacaBroker
        from config.settings import ALPACA_API_SECRET as _SECRET
        return AlpacaBroker(api_key=ALPACA_API_KEY, api_secret=_SECRET, paper=True)
    return DryRunBroker()


def _parse_strategy(d: dict) -> Strategy:
    """Reconstruct a Strategy object from the winner dict's strategy_schema."""
    schema = d.get("strategy_schema", d)
    return Strategy(
        id=schema.get("id", d.get("strategy_id", "unknown")),
        name=schema.get("name", d.get("strategy_name", "Unknown")),
        primary_indicator=schema.get("primary_indicator", {"type": "RSI", "params": {"period": 14}}),
        confirmation_indicator=schema.get("confirmation_indicator", {"type": "EMA", "params": {"period": 50}}),
        entry=schema.get("entry", {"trigger": "RSI_OVERSOLD", "filter": "NONE"}),
        exit=schema.get("exit", {
            "stop_loss":         {"type": "atr_multiple", "value": 2.0},
            "take_profit":       {"type": "r_multiple",   "value": 2.0},
            "trailing_stop_atr": None,
            "time_exit_bars":    None,
        }),
        risk=schema.get("risk", {"max_open_positions": 3}),
        metadata=schema.get("metadata", {}),
    )
