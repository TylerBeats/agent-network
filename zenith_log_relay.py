"""
zenith_log_relay.py
───────────────────
Drop this file in C:\\Users\\Tyler\\agent-network\\
Run it with: python zenith_log_relay.py

UPDATED: 
1. Includes do_POST to capture Orchestrator logs directly.
2. Enhanced monkey-patching for BaseAgent and Orchestrator.
3. CORS headers for cross-port communication (8000 -> 8081).
"""

import json
import logging
import os
import sys
import time
import threading
import traceback
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────
LOG_FILE    = Path("zenith_logs.jsonl")   # one JSON object per line
RELAY_PORT = 8081
MAX_ENTRIES = 2000                        # keep last N entries in memory

# ── In-memory log buffer ──────────────────────────────────────────────────
_lock    = threading.Lock()
_entries = []          # list of dicts
_last_id = 0

def _push(level: str, source: str, msg: str, extra: dict | None = None):
    global _last_id
    entry = {
        "id":    _last_id,
        "ts":    datetime.now().strftime("%H:%M:%S.%f")[:-3],
        "level": level,   # INFO | WARN | ERROR | SYSTEM | AGENT | TIMING | ZENITH
        "src":    source,
        "msg":    msg,
    }
    if extra:
        entry.update(extra)
    with _lock:
        _last_id += 1
        _entries.append(entry)
        if len(_entries) > MAX_ENTRIES:
            _entries.pop(0)
    # Also append to disk
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass

# ── Custom logging handler ─────────────────────────────────────────────────
class RelayHandler(logging.Handler):
    SOURCE_MAP = {
        "cycle.orchestrator": "ORCHESTRATOR",
        "agents.base":        "BASE_AGENT",
        "agents.strategy_builder": "BUILDER",
        "agents.backtester":       "BACKTESTER",
        "agents.trader":           "TRADER",
        "agents.trading_coach":    "COACH",
        "agents.coordinator":      "COORDINATOR",
        "core.network":            "NETWORK",
        "httpx":                   "HTTP",
        "openai":                  "OPENAI",
    }

    def emit(self, record: logging.LogRecord):
        src = self.SOURCE_MAP.get(record.name, record.name.upper())
        level = {
            logging.DEBUG:    "DEBUG",
            logging.INFO:     "INFO",
            logging.WARNING:  "WARN",
            logging.ERROR:    "ERROR",
            logging.CRITICAL: "ERROR",
        }.get(record.levelno, "INFO")

        extra = {}
        if record.exc_info:
            extra["traceback"] = traceback.format_exception(*record.exc_info)
            level = "ERROR"

        _push(level, src, self.format(record), extra if extra else None)

# ── Monkey-patch BaseAgent to capture token stats & timing ────────────────
def patch_base_agent():
    try:
        from agents.base import BaseAgent
        original_call = BaseAgent._call_llm

        def patched_call(self, prompt: str) -> str:
            t0 = time.time()
            agent_name = getattr(self, "name", "unknown")
            prompt_tokens_est = len(prompt.split())

            _push("AGENT", agent_name.upper(), f"→ LLM call started | prompt ~{prompt_tokens_est} words", {
                "prompt_preview": prompt[:200] + "..." if len(prompt) > 200 else prompt
            })

            try:
                result = original_call(self, prompt)
                elapsed = time.time() - t0

                # Read token stats that base.py already tracks
                in_tok  = getattr(self, "_last_input_tokens",  0)
                out_tok = getattr(self, "_last_output_tokens", 0)
                tps     = out_tok / elapsed if elapsed > 0 and out_tok > 0 else 0

                _push("ZENITH", agent_name.upper(),
                    f"← LLM response | {in_tok} in / {out_tok} out tokens | "
                    f"{tps:.1f} t/s | {elapsed:.1f}s", {
                        "input_tokens":  in_tok,
                        "output_tokens": out_tok,
                        "elapsed_s":      round(elapsed, 2),
                        "tokens_per_s":  round(tps, 1),
                        "response_preview": result[:300] + "..." if len(result) > 300 else result,
                    })
                return result

            except Exception as e:
                elapsed = time.time() - t0
                _push("ERROR", agent_name.upper(),
                    f"✗ LLM call FAILED after {elapsed:.1f}s: {type(e).__name__}: {e}", {
                        "elapsed_s": round(elapsed, 2),
                        "traceback": traceback.format_exc(),
                    })
                raise

        BaseAgent._call_llm = patched_call
        _push("SYSTEM", "RELAY", "✓ BaseAgent._call_llm patched — token stats & timing active")

    except ImportError as e:
        _push("WARN", "RELAY", f"Could not patch BaseAgent: {e}")

# ── Monkey-patch orchestrator to capture cycle timing ─────────────────────
def patch_orchestrator():
    try:
        import cycle.orchestrator as orch
        original_run = orch.run_cycle

        def patched_run(state, network, asset, **kwargs):
            cycle_num = getattr(state, "cycle_number", "?") + 1
            t0 = time.time()
            _push("SYSTEM", "ORCHESTRATOR", f"━━━ CYCLE {cycle_num} START ━━━")

            agent_timings = {}
            original_call_agent = orch._call_agent

            def timed_call_agent(agent_name, *args, **kw):
                t_agent = time.time()
                _push("AGENT", agent_name.upper(), f"▶ Agent starting")
                try:
                    result = original_call_agent(agent_name, *args, **kw)
                    elapsed = time.time() - t_agent
                    agent_timings[agent_name] = round(elapsed, 2)
                    _push("TIMING", agent_name.upper(),
                        f"✓ Agent complete in {elapsed:.1f}s", {"elapsed_s": round(elapsed, 2)})
                    return result
                except Exception as e:
                    elapsed = time.time() - t_agent
                    _push("ERROR", agent_name.upper(),
                        f"✗ Agent FAILED after {elapsed:.1f}s: {type(e).__name__}: {e}", {
                            "elapsed_s": round(elapsed, 2),
                            "traceback": traceback.format_exc(),
                        })
                    raise

            orch._call_agent = timed_call_agent
            try:
                result = original_run(state, network, asset, **kwargs)
            finally:
                orch._call_agent = original_call_agent

            total = time.time() - t0
            _push("TIMING", "ORCHESTRATOR",
                f"━━━ CYCLE {cycle_num} COMPLETE in {total:.1f}s ━━━ | timings: {agent_timings}",
                {"total_s": round(total, 2), "agent_timings": agent_timings})
            return result

        orch.run_cycle = patched_run
        _push("SYSTEM", "RELAY", "✓ orchestrator.run_cycle patched — cycle timing active")

    except ImportError as e:
        _push("WARN", "RELAY", f"Could not patch orchestrator: {e}")

# ── HTTP relay server ──────────────────────────────────────────────────────
class RelayHTTPHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        """Allows Orchestrator to send logs directly to the Relay via POST"""
        if self.path == "/log":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            try:
                entry = json.loads(post_data.decode('utf-8'))
                # Store the log sent from the Orchestrator
                _push(entry.get("level", "INFO"), 
                      entry.get("src", "SYSTEM"), 
                      entry.get("msg", ""), 
                      entry)
                
                self.send_response(200)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(b"ok")
            except Exception as e:
                self.send_response(400)
                self.end_headers()

    def do_GET(self):
        if self.path.startswith("/logs"):
            # Dashboard polls here: ?since=<id> returns new entries
            since = 0
            if "since=" in self.path:
                try:
                    since = int(self.path.split("since=")[1].split("&")[0])
                except Exception:
                    pass

            with _lock:
                entries = [e for e in _entries if e["id"] > since]

            body = json.dumps(entries).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)

        elif self.path == "/status":
            body = json.dumps({
                "relay": "online",
                "entries": len(_entries),
                "last_id": _last_id,
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        elif self.path == "/clear":
            with _lock:
                _entries.clear()
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(b"cleared")

        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        """Handle CORS pre-flight requests"""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, fmt, *args):
        pass  # silence HTTP access logs


def start_relay_server():
    server = HTTPServer(("127.0.0.1", RELAY_PORT), RelayHTTPHandler)
    _push("SYSTEM", "RELAY", f"✓ Log relay server listening on http://127.0.0.1:{RELAY_PORT}")
    print(f"[RELAY] Log server on http://127.0.0.1:{RELAY_PORT}")
    server.serve_forever()


# ── Main entry ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Install logging handler
    root_logger = logging.getLogger()
    relay_handler = RelayHandler()
    relay_handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(relay_handler)

    _push("SYSTEM", "RELAY", "═══════════════════════════════════")
    _push("SYSTEM", "RELAY", "  Zenith Log Relay starting up")
    _push("SYSTEM", "RELAY", "═══════════════════════════════════")

    # Start HTTP server in background thread
    t = threading.Thread(target=start_relay_server, daemon=True)
    t.start()
    time.sleep(0.5)

    # Apply patches (Must be done BEFORE starting Orchestrator)
    patch_base_agent()
    patch_orchestrator()

    _push("SYSTEM", "RELAY", "✓ All patches applied — ready")
    print("[RELAY] Ready. Now run: python zenith_orchestrator.py in another window.")
    print("[RELAY] Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[RELAY] Stopped.")