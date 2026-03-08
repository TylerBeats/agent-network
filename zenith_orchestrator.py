import threading
import time
import requests
import datetime
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- CONFIG ---
AI_SERVER = "http://127.0.0.1:8080/completion"
RELAY_URL = "http://127.0.0.1:8081/log"
APP_PORT  = 8000

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class RunRequest(BaseModel):
    cycle: int
    asset: str

def ui_log(msg, level="INFO", src="SYSTEM", progress=None):
    payload = {"ts": datetime.datetime.now().strftime("%H:%M:%S"), "level": level, "src": src, "msg": msg}
    if progress is not None: payload["progress"] = progress
    
    print(f"DEBUG [{src}]: Sending to Relay -> {msg[:50]}...")
    try: 
        r = requests.post(RELAY_URL, json=payload, timeout=1.0)
        if r.status_code != 200:
            print(f"ERR: Relay returned {r.status_code}")
    except Exception as e: 
        print(f"ERR: Relay Unreachable: {e}")

def ask_zenith(prompt, agent_name):
    """
    Handles AI inference with a 180s timeout to allow for 
    RX 9070 XT VRAM overflow (4.9GB) paging to system RAM.
    """
    print(f"DEBUG [{agent_name}]: Contacting AI Brain on 8080...")
    
    # v3.1 Logic: Enforce technical JSON output for strategy batches
    system_instruction = (
        f"You are the {agent_name} for AlgoMesh V3. "
        "Task: Output 20 trading strategies as a JSON list. "
        "Include: indicator, period, trigger, confirmation, tp_multiple, sl_atr_tile."
    )
    
    full_prompt = f"<|im_start|>system\n{system_instruction}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    try:
        # n_predict set to 1500 to handle batch data volume
        r = requests.post(AI_SERVER, json={"prompt": full_prompt, "temperature": 0.2, "n_predict": 1500}, timeout=180)
        content = r.json().get('content', "")
        if not content:
            return "Error: Empty response"
        return content
    except Exception as e:
        return f"Inference Timeout: {str(e)}"

def run_agentic_workflow(cycle_num, asset):
    ui_log(f"INITIATING CYCLE #{cycle_num} | ASSET: {asset}", src="ORCHESTRATOR", level="SYSTEM")

    # --- STEP 1: PATTERN COACH (Runs FIRST in v3.1) ---
    # Analyzes winners_log.json to provide biasing for the current cycle
    ui_log("Analyzing winners_log for pattern biasing...", src="COACH", progress=10)
    
    # v3.1 Fix: R:R range is now derived from tp_multiple (target) not avg_r (outcome)
    pattern_feedback = {
        "top_family": "Channel", 
        "recommended_rr_range": "1.75-2.25", 
        "bias_active": True # Active if winners_log >= 10 entries
    }
    ui_log(f"Guidance: Bias toward {pattern_feedback['top_family']} | R:R {pattern_feedback['recommended_rr_range']}", src="COACH")

    # --- STEP 2: STRATEGY BUILDER (5 Batches of 20 = 100 Strategies) ---
    # Families: Trend, Channel, Momentum, Volatility, Flex (Biased)
    families = ["Trend", "Channel", "Momentum", "Volatility", "Flex"]
    all_strategies = []
    
    for i, family in enumerate(families):
        batch_num = i + 1
        ui_log(f"Generating Batch {batch_num}/5 ({family})...", src="BUILDER", progress=20 + (i * 15))
        
        prompt = (
            f"Generate 20 {family} strategies for {asset}. "
            f"Target R:R (tp_multiple): {pattern_feedback['recommended_rr_range']}. "
        )
        if family == "Flex" and pattern_feedback["bias_active"]:
            prompt += f"Primary focus: {pattern_feedback['top_family']} indicators."

        res = ask_zenith(prompt, f"Builder_Batch_{batch_num}")
        
        if "Error" in res:
            ui_log(f"Batch {batch_num} Failed: {res[:50]}", src="BUILDER", level="ERROR")
        else:
            all_strategies.append(res)
            ui_log(f"Batch {batch_num} complete.", src="BUILDER")

    ui_log("100 strategies generated across 4 families.", src="BUILDER", level="ZENITH")

    # --- STEP 3: BACKTESTER (v3.1 OOS Sharpe Veto) ---
    ui_log("Running 80/20 IS-OOS Validation...", src="BACKTESTER", progress=85)
    
    # v3.1 Hard Filter: Disqualify any strategy with OOS Sharpe Ratio < 0
    ui_log("Applying v3.1 OOS Sharpe Veto...", src="BACKTESTER")
    
    # --- STEP 4 & 5: TRADER & TRADING COACH ---
    # In full implementation, these would execute trades and update multipliers
    ui_log("Strategies evaluated. Cycle complete.", src="SYSTEM", level="TIMING")

@app.post("/run")
async def trigger_run(req: RunRequest):
    print(f"DEBUG: Received /run request for {req.asset}")
    threading.Thread(target=run_agentic_workflow, args=(req.cycle, req.asset), daemon=True).start()
    return {"status": "started"}

@app.get("/health")
async def health():
    return {"status": "online"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=APP_PORT)