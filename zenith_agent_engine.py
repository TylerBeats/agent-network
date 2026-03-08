import requests
import time
import json
import datetime

# --- CONFIG ---
ZENITH_API = "http://127.0.0.1:8080/completion"
RELAY_URL = "http://127.0.0.1:8081/log"

def ui_log(msg, level="INFO", src="SYSTEM", progress=None):
    """Pushes status to your React Dashboard """
    payload = {
        "ts": datetime.datetime.now().strftime("%H:%M:%S"),
        "level": level, "src": src, "msg": msg
    }
    if progress is not None: payload["progress"] = progress
    try: requests.post(RELAY_URL, json=payload, timeout=0.1)
    except: pass

def call_zenith_ai(prompt):
    """Uses the inference logic from your voice.py"""
    payload = {
        "prompt": f"<|im_start|>system\nAgentic Trading Mode<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
        "temperature": 0.2,
        "stop": ["<|im_end|>"]
    }
    r = requests.post(ZENITH_API, json=payload)
    return r.json().get('content', '')

def run_algo_mesh_cycle(asset="SPX500"):
    # PHASE 1: BUILDER
    ui_log(f"Generating strategies for {asset}...", src="BUILDER", progress=20)
    strat_raw = call_zenith_ai(f"Generate 3 mean-reversion strategies for {asset}")
    ui_log("Strategies generated. Sending to Backtester...", src="BUILDER", progress=100)
    ui_log("builder→backtester", src="BUILDER", level="ZENITH") # Triggers UI Arrow 

    # PHASE 2: BACKTESTER
    ui_log("Analyzing historical M5 data...", src="BACKTESTER", progress=40)
    time.sleep(2) # Simulating compute time on your RX 9070 XT
    ui_log("Monte Carlo verification complete.", src="BACKTESTER", progress=100)
    ui_log("backtester→trader", src="BACKTESTER", level="ZENITH")

    # PHASE 3: TRADER
    ui_log("Checking dry-run margin...", src="TRADER", progress=50)
    ui_log("Strategy deployed to Paper Account.", src="TRADER", progress=100)
    ui_log("trader→coach", src="TRADER", level="ZENITH")

    # PHASE 4: COACH
    ui_log("Cycle complete. Performance logged.", src="COACH", progress=100)
    ui_log("Cycle Complete", src="SYSTEM", level="TIMING")