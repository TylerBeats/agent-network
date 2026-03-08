"""
zenith_orchestrator.py — AlgoMesh V3 Real Pipeline
====================================================
Every log line produced inside StrategyBuilderAgent and BacktesterAgent is
forwarded live to the relay (port 8081) so the dashboard updates in real time
as each batch and each strategy is processed.

Startup order:
  1. python zenith_log_relay.py       (port 8081)
  2. python zenith_orchestrator.py    (port 8000)
  3. Open Dashboard.html
"""

import datetime
import json
import logging
import os
import sys
import threading

import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Repo root on sys.path
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from agents.strategy_builder import StrategyBuilderAgent
from agents.backtester import BacktesterAgent
from agents.asset_profile import build_asset_key
from core.message import Message, MessageType
from cycle.state import CycleState, load_state, save_state, state_path_for

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RELAY_URL     = "http://127.0.0.1:8081/log"
APP_PORT      = 8000

ASSET_CONFIG: dict[str, dict] = {
    "SPX500": {"chain": "spx500",  "token_address": "", "bucket_seconds": 86400},
    "NAS100": {"chain": "nas100",  "token_address": "", "bucket_seconds": 86400},
    "US30":   {"chain": "us30",    "token_address": "", "bucket_seconds": 86400},
    "XAUUSD": {"chain": "xauusd", "token_address": "", "bucket_seconds": 86400},
}
DEFAULT_ASSET = "SPX500"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("zenith_orchestrator")

# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RunRequest(BaseModel):
    cycle: int = 1
    asset: str = DEFAULT_ASSET


# ---------------------------------------------------------------------------
# Relay logging helper
# ---------------------------------------------------------------------------
def ui_log(msg: str, level: str = "INFO", src: str = "SYSTEM", progress: int | None = None):
    payload = {
        "ts":    datetime.datetime.now().strftime("%H:%M:%S"),
        "level": level,
        "src":   src,
        "msg":   msg,
    }
    if progress is not None:
        payload["progress"] = progress
    try:
        requests.post(RELAY_URL, json=payload, timeout=1.0)
    except Exception:
        pass
    logger.info("[%s] %s", src, msg)


# ---------------------------------------------------------------------------
# Live relay handler
# Attaches to agent loggers so every internal logger.info/warning/error call
# inside strategy_builder.py and backtester.py is forwarded to the dashboard
# in real time — no agent file changes required.
# ---------------------------------------------------------------------------
class _RelayHandler(logging.Handler):

    _LOGGER_SRC: dict[str, str] = {
        "agents.strategy_builder": "BUILDER",
        "agents.backtester":       "BACKTESTER",
        "backtesting.engine":      "BACKTESTER",
        "backtesting.filters":     "BACKTESTER",
        "backtesting.monte_carlo": "BACKTESTER",
        "backtesting.scorer":      "BACKTESTER",
        "backtesting.selector":    "BACKTESTER",
        "data.pipeline":           "SYSTEM",
        "data.sources.yahoo":      "SYSTEM",
    }

    _LEVEL_MAP: dict[int, str] = {
        logging.DEBUG:    "INFO",
        logging.INFO:     "INFO",
        logging.WARNING:  "WARN",
        logging.ERROR:    "ERROR",
        logging.CRITICAL: "ERROR",
    }

    def emit(self, record: logging.LogRecord):
        src   = self._LOGGER_SRC.get(record.name, "SYSTEM")
        level = self._LEVEL_MAP.get(record.levelno, "INFO")
        msg   = self.format(record)
        payload = {
            "ts":    datetime.datetime.now().strftime("%H:%M:%S"),
            "level": level,
            "src":   src,
            "msg":   msg,
        }
        try:
            requests.post(RELAY_URL, json=payload, timeout=0.8)
        except Exception:
            pass


def _install_relay_handlers():
    """Install the relay handler on every agent and backtesting logger."""
    handler = _RelayHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    for name in (
        "agents.strategy_builder",
        "agents.backtester",
        "backtesting.engine",
        "backtesting.filters",
        "backtesting.monte_carlo",
        "backtesting.scorer",
        "backtesting.selector",
        "data.pipeline",
        "data.sources.yahoo",
    ):
        lg = logging.getLogger(name)
        lg.setLevel(logging.DEBUG)
        if not any(isinstance(h, _RelayHandler) for h in lg.handlers):
            lg.addHandler(handler)


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------
def _build_client():
    from openai import OpenAI
    base_url = os.getenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:8080/v1")
    return OpenAI(base_url=base_url, api_key="local", timeout=600)


# ---------------------------------------------------------------------------
# Coach feedback from CycleState
# ---------------------------------------------------------------------------
def _build_coach_feedback(state: CycleState) -> dict | None:
    history = state.performance_history
    if len(history) < 5:
        return None

    from collections import Counter
    recent     = history[-20:]
    ind_counts = Counter(
        w.get("strategy_schema", {}).get("primary_indicator", {}).get("type", "")
        for w in recent
    )
    best_ind = ind_counts.most_common(1)[0][0] if ind_counts else ""
    avg_rr   = sum(w.get("avg_r", 0.0) for w in recent) / len(recent)

    return {
        "best_indicator_types":  [best_ind] if best_ind else [],
        "best_rr_ratio":         round(avg_rr, 2),
        "failing_regimes":       [],
        "next_cycle_hypothesis": (
            f"Recent winners favour {best_ind} indicators with avg R:R {avg_rr:.2f}. "
            "Prioritise similar setups while exploring new indicator families."
        ),
    }


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------
def run_agentic_workflow(cycle_num: int, asset_key: str):
    # Patch relay handler onto all agent loggers before anything runs
    _install_relay_handlers()

    asset_cfg  = ASSET_CONFIG.get(asset_key.upper(), ASSET_CONFIG[DEFAULT_ASSET])
    chain      = asset_cfg["chain"]
    ak         = build_asset_key(chain, asset_cfg.get("token_address", ""))
    state_path = state_path_for(ak)
    state      = load_state(state_path)
    state.asset_key = ak

    if state.cycle_number > 0:
        cycle_num = state.cycle_number + 1

    ui_log(
        f"INITIATING CYCLE #{cycle_num} | ASSET: {asset_key.upper()}",
        level="SYSTEM", src="ORCHESTRATOR",
    )

    # ── STEP 1: Coach ─────────────────────────────────────────────────────
    ui_log("Analyzing performance history for pattern biasing...", src="COACH", progress=10)
    coach_feedback = _build_coach_feedback(state)

    if coach_feedback:
        best_ind = ", ".join(coach_feedback["best_indicator_types"]) or "any"
        ui_log(
            f"Guidance: Bias toward {best_ind} | avg R:R {coach_feedback['best_rr_ratio']:.2f} "
            f"(based on {len(state.performance_history)} historical winners)",
            src="COACH",
        )
    else:
        remaining = max(0, 5 - len(state.performance_history))
        ui_log(
            f"No bias applied — need {remaining} more winning cycle(s) to activate pattern guidance.",
            src="COACH",
        )

    # ── STEP 2: Strategy Builder ───────────────────────────────────────────
    # Milestone anchors set progress on the bar.
    # The _RelayHandler streams every internal logger line live between them.
    ui_log(
        "Starting Strategy Builder — 4 batches × 25 strategies = 100 total.",
        src="BUILDER", progress=18,
    )

    client  = _build_client()
    builder = StrategyBuilderAgent(client)

    builder_request = json.dumps({
        "cycle":              cycle_num,
        "asset":              asset_cfg,
        "coach_feedback":     coach_feedback,
        "retired_patterns":   state.retired_patterns,
        "asset_strategy_log": state.asset_strategy_log,
    })

    builder_msg = Message(
        sender="orchestrator",
        recipient="strategy_builder",
        type=MessageType.TASK,
        content=builder_request,
        metadata={"original_sender": "orchestrator"},
    )

    builder.receive(builder_msg)

    ui_log("Batch 1/4 (Trend: EMA / SMA / WMA) — sending to LLM...",          src="BUILDER", progress=20)
    ui_log("Batch 2/4 (Channel: Donchian / Keltner) — queued...",              src="BUILDER", progress=35)
    ui_log("Batch 3/4 (Momentum: RSI / ROC / CCI / Stoch / WillR) — queued...", src="BUILDER", progress=50)
    ui_log("Batch 4/4 (Volatility & MACD: BB / Ichimoku / MACD) — queued...", src="BUILDER", progress=65)

    # Blocks here; _RelayHandler pushes per-batch lines from inside _handle()
    try:
        builder_responses = builder.process()
    except Exception as exc:
        import traceback
        ui_log(f"Builder exception: {exc}", src="BUILDER", level="ERROR")
        ui_log(traceback.format_exc()[-600:], src="BUILDER", level="ERROR")
        return

    if not builder_responses:
        ui_log(
            "Builder returned no response — is the LLM server running on port 8080?",
            src="BUILDER", level="ERROR",
        )
        return

    builder_result_msg = builder_responses[0]

    if builder_result_msg.type == MessageType.ERROR:
        ui_log(f"Builder error: {builder_result_msg.content[:400]}", src="BUILDER", level="ERROR")
        return

    try:
        builder_output = json.loads(builder_result_msg.content)
        strategies     = builder_output.get("strategies", [])
        diversity      = builder_output.get("diversity_score", 0.0)
        dev_rate       = builder_output.get("combo_deviation_rate", 0.0)
    except (json.JSONDecodeError, AttributeError) as exc:
        ui_log(f"Could not parse builder output: {exc}", src="BUILDER", level="ERROR")
        return

    ui_log(
        f"{len(strategies)} strategies generated | diversity {diversity}/100 | "
        f"combo deviation {dev_rate}%",
        src="BUILDER", level="ZENITH",
    )
    ui_log("builder→backtester", src="BUILDER", level="ZENITH")

    if not strategies:
        ui_log("Builder produced 0 strategies — aborting cycle.", src="BUILDER", level="ERROR")
        return

    # ── STEP 3: Backtester ────────────────────────────────────────────────
    # _RelayHandler pushes every PASS / FAIL line from inside _handle() live.
    ui_log(
        f"Backtester starting — evaluating {len(strategies)} strategies...",
        src="BACKTESTER", progress=70,
    )
    ui_log("Fetching local candle data from cache...", src="BACKTESTER")
    ui_log(
        "Running IS/OOS split (80/20) + 800 Monte Carlo sims + hard filters per strategy.",
        src="BACKTESTER", progress=72,
    )

    backtester = BacktesterAgent(client)

    backtester_msg = Message(
        sender="orchestrator",
        recipient="backtester",
        type=MessageType.TASK,
        content=json.dumps({"strategies": strategies, "asset": asset_cfg}),
        metadata={
            "original_sender": "orchestrator",
            "past_failures":   state.retired_patterns,
        },
    )

    backtester.receive(backtester_msg)

    try:
        backtester_responses = backtester.process()
    except Exception as exc:
        import traceback
        ui_log(f"Backtester exception: {exc}", src="BACKTESTER", level="ERROR")
        ui_log(traceback.format_exc()[-600:], src="BACKTESTER", level="ERROR")
        return

    if not backtester_responses:
        ui_log("Backtester returned no response.", src="BACKTESTER", level="ERROR")
        return

    backtester_result_msg = backtester_responses[0]

    if backtester_result_msg.type == MessageType.ERROR:
        ui_log(
            f"Backtester error: {backtester_result_msg.content[:400]}",
            src="BACKTESTER", level="ERROR",
        )
        return

    try:
        bt_output = json.loads(backtester_result_msg.content)
    except json.JSONDecodeError as exc:
        ui_log(f"Could not parse backtester output: {exc}", src="BACKTESTER", level="ERROR")
        return

    all_scores = bt_output.get("all_scores", [])
    winners    = bt_output.get("winners", [])
    passed     = [s for s in all_scores if s.get("passed")]
    failed     = [s for s in all_scores if not s.get("passed")]

    ui_log(
        f"Evaluation complete — {len(passed)} passed all filters, {len(failed)} eliminated.",
        src="BACKTESTER", progress=98,
    )

    # Surface most common failure reason
    if failed:
        from collections import Counter
        reasons    = [s.get("failure", "unknown").split(":")[0].strip() for s in failed]
        top_reason = Counter(reasons).most_common(1)[0]
        ui_log(
            f"Most common elimination: '{top_reason[0]}' ({top_reason[1]} strategies)",
            src="BACKTESTER",
        )

    # Update asset_strategy_log
    for s in all_scores:
        state.asset_strategy_log.append({
            "cycle":             cycle_num,
            "primary_indicator": s.get("primary_indicator", ""),
            "entry_trigger":     s.get("entry_trigger", ""),
            "result":            "passed" if s.get("passed") else "failed",
            "failure_reason":    s.get("failure", ""),
            "score":             s.get("score") or 0,
        })

    if not winners:
        ui_log("No strategies survived hard filters this cycle.", src="BACKTESTER", level="WARN")
        state.cycle_number = cycle_num
        save_state(state, state_path)
        ui_log(f"Cycle {cycle_num} complete — no winners. State saved.", src="SYSTEM", level="TIMING")
        return

    # Log each winner
    for i, w in enumerate(winners, 1):
        ui_log(
            f"Winner #{i}: '{w['strategy_name']}' | "
            f"Score {w.get('score', 0):.1f} | "
            f"Sharpe {w.get('sharpe', 0):.2f} IS / {w.get('oos_sharpe', 0):.2f} OOS | "
            f"WR {w.get('win_rate', 0)*100:.1f}% | "
            f"MaxDD {w.get('max_drawdown', 0)*100:.1f}% | "
            f"EV {w.get('avg_r', 0):.2f}R | "
            f"Trades {w.get('trade_count', 0)} | "
            f"Risk {w.get('best_risk_pct', 0)}% | "
            f"Confidence: {w.get('confidence_rating', '?')} | "
            f"MC p95 DD: {w.get('mc_p95_dd', 0)*100:.1f}%",
            src="BACKTESTER", level="ZENITH",
        )

    ui_log("backtester→coach", src="BACKTESTER", level="ZENITH")

    # ── STEP 4: Persist to CycleState ──────────────────────────────────────
    state.cycle_number      = cycle_num
    state.active_strategy   = winners[0] if winners else None
    state.active_strategies = winners
    state.performance_history.extend(winners)

    top = winners[0]
    state.projected_avg_r    = top.get("avg_r", 0.0)
    state.projected_win_rate = top.get("win_rate", 0.0)
    for w in winners:
        sid = w.get("strategy_id", "")
        if sid and w.get("mc_p95_dd") is not None:
            state.mc_p95_dd_by_strategy[sid]                 = w["mc_p95_dd"]
            state.projected_monthly_pnl_pct_by_strategy[sid] = w.get("projected_monthly_pnl_pct", 0.0)
            state.return_projections_by_strategy[sid]         = w.get("return_projections", {})

    save_state(state, state_path)
    ui_log(
        f"State saved → {os.path.basename(state_path)} | "
        f"Cycle {cycle_num} | {len(state.performance_history)} total winners in history",
        src="SYSTEM",
    )

    # ── STEP 5: Coach summary ──────────────────────────────────────────────
    narrative = bt_output.get("narrative", "")
    if narrative:
        ui_log(narrative[:500], src="COACH", progress=100)
    else:
        ui_log(
            f"Cycle {cycle_num} complete — {len(winners)} winner(s) saved to state.",
            src="COACH", progress=100,
        )

    period = bt_output.get("period", {})
    if period:
        ui_log(
            f"Backtest period: {period.get('is_start')} → {period.get('oos_end')} "
            f"({period.get('years_covered', '?')} yrs, {period.get('total_candles', '?')} bars, "
            f"{period.get('candle_label', '')})",
            src="COACH",
        )

    dq = bt_output.get("data_quality", {})
    if dq:
        warnings = dq.get("warnings", [])
        ui_log(
            f"Data quality: {dq.get('confidence', '?')}"
            + (f" | warnings: {'; '.join(warnings)}" if warnings else ""),
            src="COACH",
        )

    ui_log("coach→complete", src="COACH", level="ZENITH")
    ui_log(
        f"Strategies evaluated. Cycle {cycle_num} complete.",
        src="SYSTEM", level="TIMING",
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/run")
async def trigger_run(req: RunRequest):
    threading.Thread(
        target=run_agentic_workflow,
        args=(req.cycle, req.asset),
        daemon=True,
    ).start()
    return {"status": "started", "cycle": req.cycle, "asset": req.asset}


@app.get("/health")
async def health():
    return {"status": "online"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=APP_PORT)