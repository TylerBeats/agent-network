import { useState, useEffect, useRef } from "react";

// ─── Config ───────────────────────────────────────────────────────────────
const ZENITH_BASE = "http://127.0.0.1:8000"; // Point to Orchestrator
const RELAY_BASE  = "http://127.0.0.1:8081"; // Point to Log Relay
const POLL_MS     = 1500;                        // how often to poll relay

// ─── Agent definitions ────────────────────────────────────────────────────
const AGENTS = {
  builder:    { id: "builder",    label: "STRATEGY BUILDER",    short: "Builder",    role: "Generates candidate strategies per cycle",              color: "#f59e0b", icon: "◈" },
  backtester: { id: "backtester", label: "STRATEGY BACKTESTER", short: "Backtester", role: "Evaluates candidates · selects winner via Monte Carlo",  color: "#06b6d4", icon: "⬡" },
  trader:     { id: "trader",     label: "TRADER",              short: "Trader",     role: "Executes approved strategy in dry-run mode",             color: "#10b981", icon: "▶" },
  coach:      { id: "coach",      label: "TRADING COACH",       short: "Coach",      role: "Monitors performance · adjusts risk · feeds Builder",    color: "#a78bfa", icon: "◎" },
};
const AGENT_ORDER = ["builder", "backtester", "trader", "coach"];
const TIER_COLOR  = { GOOD: "#10b981", MARGINAL: "#f59e0b", POOR: "#ef4444" };
const TIER_BG     = { GOOD: "#10b98115", MARGINAL: "#f59e0b15", POOR: "#ef444415" };

// ─── Log level colors ─────────────────────────────────────────────────────
const LEVEL_COLOR = {
  SYSTEM: "#06b6d4",
  AGENT:  "#f59e0b",
  ZENITH: "#a78bfa",
  TIMING: "#10b981",
  INFO:   "#475569",
  WARN:   "#f59e0b",
  ERROR:  "#ef4444",
  DEBUG:  "#334155",
  HTTP:   "#334155",
  OPENAI: "#64748b",
};

const LEVEL_BG = {
  ERROR:  "#ef444408",
  WARN:   "#f59e0b08",
  ZENITH: "#a78bfa08",
  TIMING: "#10b98108",
};

// ─── Source badge colors ──────────────────────────────────────────────────
const SRC_COLOR = {
  ORCHESTRATOR: "#06b6d4", BUILDER: "#f59e0b", BACKTESTER: "#06b6d4",
  TRADER: "#10b981", COACH: "#a78bfa", COORDINATOR: "#94a3b8",
  NETWORK: "#64748b", HTTP: "#334155", OPENAI: "#475569",
  BASE_AGENT: "#94a3b8", RELAY: "#10b981",
};

function loadCycles() {
  try { return JSON.parse(localStorage.getItem("algomesh_cycles") || "[]"); } catch { return []; }
}
function saveCycles(c) {
  try { localStorage.setItem("algomesh_cycles", JSON.stringify(c.slice(-50))); } catch {}
}

let gid = 0;

export default function App() {
  const [view, setView]             = useState("log");   // default to log for setup
  const [agentStates, setAgentStates] = useState(() =>
    Object.fromEntries(AGENT_ORDER.map(id => [id, { status: "idle", logs: [], progress: 0 }]))
  );
  const [isRunning, setIsRunning]   = useState(false);
  const [activeAgent, setActiveAgent] = useState(null);
  const [cycleNum, setCycleNum]     = useState(1);
  const [loopPhase, setLoopPhase]   = useState(null);
  const [selectedCycle, setSelectedCycle] = useState(null);
  const [cycles, setCycles]         = useState(loadCycles);

  // ── Live log state ──
  const [logEntries, setLogEntries] = useState([]);
  const [lastId, setLastId]         = useState(-1);
  const [relayOnline, setRelayOnline] = useState(false);
  const [zenithOnline, setZenithOnline] = useState(null);
  const [filterLevel, setFilterLevel] = useState("ALL");
  const [filterSrc, setFilterSrc]   = useState("ALL");
  const [expandedId, setExpandedId] = useState(null);
  const [autoScroll, setAutoScroll] = useState(true);
  const [showTraceback, setShowTraceback] = useState(true);

  const logEndRef   = useRef(null);
  const runRef      = useRef(false);
  const pollRef     = useRef(null);

  useEffect(() => { saveCycles(cycles); }, [cycles]);

  // ── Auto-scroll ──
  useEffect(() => {
    if (autoScroll) logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logEntries, autoScroll]);

  // ── Poll relay for logs ──
  useEffect(() => {
    const poll = async () => {
      try {
        const r = await fetch(`${RELAY_BASE}/logs?since=${lastId}`, { signal: AbortSignal.timeout(2000) });
        if (r.ok) {
          const data = await r.json();
          setRelayOnline(true);
          if (data.length > 0) {
            setLogEntries(prev => [...prev.slice(-1000), ...data]);
            setLastId(data[data.length - 1].id);
          }
        }
      } catch {
        setRelayOnline(false);
      }
    };
    poll();
    pollRef.current = setInterval(poll, POLL_MS);
    return () => clearInterval(pollRef.current);
  }, [lastId]);

  // ── Poll Zenith health ──
  useEffect(() => {
    const check = async () => {
      try {
        const r = await fetch(`${ZENITH_BASE}/health`, { signal: AbortSignal.timeout(2000) });
        setZenithOnline(r.ok);
      } catch { setZenithOnline(false); }
    };
    check();
    const t = setInterval(check, 10000);
    return () => clearInterval(t);
  }, []);

  const sleep = ms => new Promise(r => setTimeout(r, ms));

  // ── Animated cycle run (visual only - real run is python main.py) ──
  const runCycle = async () => {
    if (isRunning) return;
    setIsRunning(true);
    runRef.current = true;
    const thisNum = cycleNum;
    setCycleNum(c => c + 1);
    setAgentStates(Object.fromEntries(AGENT_ORDER.map(id => [id, { status: "idle", logs: [], progress: 0 }])));

    const connections = [null, "builder→backtester", "backtester→trader", "trader→coach"];
    const cycleStart  = Date.now();
    let halted = false;

    for (let i = 0; i < AGENT_ORDER.length; i++) {
      if (!runRef.current) { halted = true; break; }
      const agentId = AGENT_ORDER[i];
      const agent   = AGENTS[agentId];
      if (connections[i]) { setLoopPhase(connections[i]); await sleep(400); }
      setActiveAgent(agentId);
      setLoopPhase(null);
      setAgentStates(prev => ({ ...prev, [agentId]: { status: "running", logs: [], progress: 0 } }));

      const demoLogs = {
        builder:    ["Loading Coach feedback...", "Generating strategies via Zenith...", "Diversification check complete ✓", "Strategies serialised → Backtester"],
        backtester: ["Received strategies from Builder", "Loading historical OHLCV data...", "Running elimination filters...", "Monte Carlo simulations complete", "Winner selected → Trader"],
        trader:     ["Loading approved strategy...", "Pre-trade checklist passed ✓", "Monitoring entry triggers...", "Session report → Coach"],
        coach:      ["Reviewing cycle performance...", "Win rate drift analysis complete", "Risk adjustments applied", "Feedback → Builder ✓"],
      };

      const lines = demoLogs[agentId];
      for (let j = 0; j < lines.length; j++) {
        if (!runRef.current) { halted = true; break; }
        await sleep(400 + Math.random() * 300);
        setAgentStates(prev => ({
          ...prev, [agentId]: {
            ...prev[agentId],
            logs: [...prev[agentId].logs, lines[j]],
            progress: Math.round(((j + 1) / lines.length) * 100),
          }
        }));
      }
      if (halted) break;
      setAgentStates(prev => ({ ...prev, [agentId]: { ...prev[agentId], status: "success", progress: 100 } }));
      await sleep(300);
    }

    if (!halted) {
      setLoopPhase("coach→builder");
      await sleep(900);
      setLoopPhase(null);
      const elapsed = ((Date.now() - cycleStart) / 1000).toFixed(0);
      const cy = {
        id: Date.now(), cycle: thisNum,
        completedAt: new Date().toLocaleTimeString("en-US", { hour12: false }),
        duration: `${Math.floor(elapsed/60)}m ${elapsed%60}s`,
        strategies: [{ id: `#${String(thisNum).padStart(3,"0")}`, ev: 35+Math.floor(Math.random()*15), liveEv: 28+Math.floor(Math.random()*12), winRate: 44+Math.floor(Math.random()*8), liveWinRate: 42+Math.floor(Math.random()*8), dd: (3+Math.random()*4).toFixed(1), sharpe: (0.9+Math.random()*0.6).toFixed(2), pnl: Math.floor(Math.random()*2000)-200, tier: "GOOD", propFirm: "✓ All pass", rr: "2R" }],
        totalPnl: 0, builderBatch: 10, surviving: Math.floor(Math.random()*5)+3,
        coachAction: "Cycle complete. Feedback dispatched.", regime: ["TRENDING","SIDEWAYS","VOLATILE"][Math.floor(Math.random()*3)], riskLevel: "1%",
      };
      cy.totalPnl = cy.strategies.reduce((s, st) => s + st.pnl, 0);
      setCycles(prev => [cy, ...prev]);
    }

    setActiveAgent(null); setIsRunning(false); runRef.current = false;
    setTimeout(() => setAgentStates(prev => Object.fromEntries(Object.entries(prev).map(([k,v]) => [k, {...v, status: "idle"}]))), 4000);
  };

  const stopCycle = () => { runRef.current = false; setLoopPhase(null); };
  const clearLogs = () => { setLogEntries([]); setLastId(-1); fetch(`${RELAY_BASE}/clear`).catch(() => {}); };

  // ── Filter logic ──
  const allSources  = ["ALL", ...new Set(logEntries.map(e => e.src))];
  const allLevels   = ["ALL", "SYSTEM", "AGENT", "ZENITH", "TIMING", "INFO", "WARN", "ERROR"];
  const filtered    = logEntries.filter(e => {
    if (filterLevel !== "ALL" && e.level !== filterLevel) return false;
    if (filterSrc   !== "ALL" && e.src   !== filterSrc)   return false;
    return true;
  });

  const totalPnl  = cycles.reduce((s, c) => s + c.totalPnl, 0);
  const allStrats = cycles.flatMap(c => c.strategies);
  const avgEv     = allStrats.length ? allStrats.reduce((s, st) => s + st.liveEv, 0) / allStrats.length : 0;

  const zenithColor = zenithOnline === null ? "#334155" : zenithOnline ? "#10b981" : "#ef4444";
  const relayColor  = relayOnline ? "#10b981" : "#ef4444";

  return (
    <div style={{ minHeight: "100vh", background: "#060d1a", color: "#cbd5e1", fontFamily: "'IBM Plex Mono', monospace" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=Bebas+Neue&display=swap');
        *{box-sizing:border-box;margin:0;padding:0}
        ::-webkit-scrollbar{width:4px;height:4px}
        ::-webkit-scrollbar-track{background:#0a1628}
        ::-webkit-scrollbar-thumb{background:#1e3a5f;border-radius:2px}
        @keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
        @keyframes fadein{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:translateY(0)}}
        @keyframes blink{0%,100%{opacity:1}50%{opacity:0}}
        .fadein{animation:fadein .25s ease both}
        .hov-row:hover{background:#0d1e35!important;cursor:pointer}
        .log-row:hover{background:#0a1628!important;cursor:pointer}
        .tab{cursor:pointer;transition:color .18s;background:none;border:none;font-family:'IBM Plex Mono',monospace}
        .tab:hover{color:#e2e8f0!important}
        .runbtn{cursor:pointer;transition:all .2s;font-family:'IBM Plex Mono',monospace}
        .runbtn:hover{filter:brightness(1.2)}
        .chip{cursor:pointer;border:none;font-family:'IBM Plex Mono',monospace;transition:all .15s}
        .chip:hover{filter:brightness(1.3)}
        .agent-card{transition:border-color .3s,box-shadow .3s}
      `}</style>

      {/* ── Header ── */}
      <div style={{ padding: "12px 24px", borderBottom: "1px solid #0f2040", display: "flex", alignItems: "center", justifyContent: "space-between", background: "#04091a", position: "sticky", top: 0, zIndex: 200 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <div style={{ fontFamily: "Bebas Neue", fontSize: 20, letterSpacing: 4, color: "#f59e0b" }}>ALGO<span style={{ color: "#06b6d4" }}>MESH</span></div>
          <div style={{ width: 1, height: 18, background: "#1e3a5f" }} />
          <div style={{ fontSize: 9, color: "#334155", letterSpacing: 2 }}>ZENITH AGENTIC NETWORK v2</div>
          <div style={{ width: 1, height: 18, background: "#1e3a5f" }} />
          {/* Status dots */}
          {[
            { label: "ZENITH :8080", color: zenithColor, pulse: zenithOnline },
            { label: "RELAY :8081",  color: relayColor,  pulse: relayOnline  },
          ].map(s => (
            <div key={s.label} style={{ display: "flex", alignItems: "center", gap: 5 }}>
              <div style={{ width: 6, height: 6, borderRadius: "50%", background: s.color, boxShadow: s.pulse ? `0 0 6px ${s.color}` : "none", animation: s.pulse ? "pulse 2s infinite" : "none" }} />
              <div style={{ fontSize: 9, color: s.color, letterSpacing: 1 }}>{s.label}</div>
            </div>
          ))}
          {!relayOnline && (
            <div style={{ fontSize: 9, color: "#ef4444", letterSpacing: 1 }}>
              ⚠ run: python zenith_log_relay.py
            </div>
          )}
        </div>
        <div style={{ display: "flex", gap: 24, alignItems: "center" }}>
          {[
            { l: "P&L",     v: `${totalPnl >= 0 ? "+" : ""}$${Math.abs(totalPnl).toLocaleString()}`, c: totalPnl >= 0 ? "#10b981" : "#ef4444" },
            { l: "AVG EV",  v: allStrats.length ? `+${avgEv.toFixed(1)}%` : "—", c: "#f59e0b" },
            { l: "CYCLE",   v: `#${cycleNum}`, c: "#06b6d4" },
            { l: "LOG ENTRIES", v: logEntries.length, c: relayOnline ? "#10b981" : "#334155" },
          ].map(s => (
            <div key={s.l} style={{ textAlign: "right" }}>
              <div style={{ fontSize: 8, color: "#334155", letterSpacing: 1.5 }}>{s.l}</div>
              <div style={{ fontSize: 14, color: s.c, fontFamily: "Bebas Neue", letterSpacing: 1 }}>{s.v}</div>
            </div>
          ))}
        </div>
      </div>

      {/* ── Nav ── */}
      <div style={{ display: "flex", padding: "0 24px", borderBottom: "1px solid #0f2040", background: "#04091a" }}>
        {[["log","◎ LIVE LOG"],["cycle","◈ CYCLE VIEW"],["reports","⬡ REPORTS"]].map(([v,l]) => (
          <button key={v} className="tab" onClick={() => setView(v)} style={{
            padding: "10px 16px", fontSize: 10, letterSpacing: 2,
            color: view === v ? "#f59e0b" : "#475569", cursor: "pointer",
            borderBottom: view === v ? "2px solid #f59e0b" : "2px solid transparent",
          }}>{l}</button>
        ))}
        <div style={{ flex: 1 }} />
        <div style={{ display: "flex", gap: 8, alignItems: "center", padding: "7px 0" }}>
          <button className="runbtn" onClick={isRunning ? stopCycle : runCycle} style={{
            padding: "6px 16px", border: `1px solid ${isRunning ? "#ef4444" : zenithOnline ? "#10b981" : "#334155"}`,
            background: isRunning ? "#ef444410" : zenithOnline ? "#10b98110" : "#33415510",
            borderRadius: 3, color: isRunning ? "#ef4444" : zenithOnline ? "#10b981" : "#334155",
            fontSize: 10, letterSpacing: 2, fontWeight: 600,
          }}>{isRunning ? "◼ STOP" : "▶ RUN CYCLE"}</button>
        </div>
      </div>

      <div style={{ padding: "16px 24px" }}>

        {/* ══════════════════════ LIVE LOG ══════════════════════ */}
        {view === "log" && (
          <div>
            {/* Setup instructions if relay offline */}
            {!relayOnline && (
              <div className="fadein" style={{ marginBottom: 14, padding: "14px 18px", background: "#0a0f1e", border: "1px solid #ef444433", borderRadius: 6 }}>
                <div style={{ fontSize: 10, color: "#ef4444", letterSpacing: 1, marginBottom: 8 }}>⚠ LOG RELAY OFFLINE — SETUP REQUIRED</div>
                <div style={{ fontSize: 10, color: "#475569", lineHeight: 2 }}>
                  <span style={{ color: "#94a3b8" }}>1.</span> Copy <span style={{ color: "#f59e0b" }}>zenith_log_relay.py</span> to <span style={{ color: "#06b6d4" }}>C:\Users\Tyler\agent-network\</span><br />
                  <span style={{ color: "#94a3b8" }}>2.</span> Open a new PowerShell → <span style={{ color: "#10b981" }}>cd C:\Users\Tyler\agent-network</span><br />
                  <span style={{ color: "#94a3b8" }}>3.</span> Run: <span style={{ color: "#10b981" }}>python zenith_log_relay.py</span><br />
                  <span style={{ color: "#94a3b8" }}>4.</span> In another PowerShell: <span style={{ color: "#10b981" }}>python main.py</span><br />
                  <span style={{ color: "#94a3b8" }}>5.</span> Logs will appear here in real time with full detail
                </div>
              </div>
            )}

            {/* Filter bar */}
            <div style={{ display: "flex", gap: 8, marginBottom: 12, flexWrap: "wrap", alignItems: "center" }}>
              <div style={{ fontSize: 9, color: "#334155", letterSpacing: 1, marginRight: 4 }}>LEVEL:</div>
              {allLevels.map(l => (
                <button key={l} className="chip" onClick={() => setFilterLevel(l)} style={{
                  padding: "3px 9px", borderRadius: 3, fontSize: 9, letterSpacing: 1,
                  background: filterLevel === l ? (LEVEL_COLOR[l] || "#334155") + "22" : "#070f1e",
                  color: filterLevel === l ? (LEVEL_COLOR[l] || "#94a3b8") : "#334155",
                  border: `1px solid ${filterLevel === l ? (LEVEL_COLOR[l] || "#334155") + "55" : "#0f2040"}`,
                }}>{l}</button>
              ))}
              <div style={{ width: 1, height: 16, background: "#1e3a5f", margin: "0 4px" }} />
              <div style={{ fontSize: 9, color: "#334155", letterSpacing: 1, marginRight: 4 }}>SOURCE:</div>
              {allSources.slice(0, 10).map(s => (
                <button key={s} className="chip" onClick={() => setFilterSrc(s)} style={{
                  padding: "3px 9px", borderRadius: 3, fontSize: 9, letterSpacing: 1,
                  background: filterSrc === s ? (SRC_COLOR[s] || "#94a3b8") + "22" : "#070f1e",
                  color: filterSrc === s ? (SRC_COLOR[s] || "#94a3b8") : "#334155",
                  border: `1px solid ${filterSrc === s ? (SRC_COLOR[s] || "#94a3b8") + "55" : "#0f2040"}`,
                }}>{s}</button>
              ))}
              <div style={{ flex: 1 }} />
              <button className="chip" onClick={() => setShowTraceback(v => !v)} style={{
                padding: "3px 9px", borderRadius: 3, fontSize: 9, letterSpacing: 1,
                background: showTraceback ? "#ef444415" : "#070f1e",
                color: showTraceback ? "#ef4444" : "#334155",
                border: `1px solid ${showTraceback ? "#ef444433" : "#0f2040"}`,
              }}>TRACEBACKS {showTraceback ? "ON" : "OFF"}</button>
              <button className="chip" onClick={() => setAutoScroll(v => !v)} style={{
                padding: "3px 9px", borderRadius: 3, fontSize: 9, letterSpacing: 1,
                background: autoScroll ? "#10b98115" : "#070f1e",
                color: autoScroll ? "#10b981" : "#334155",
                border: `1px solid ${autoScroll ? "#10b98133" : "#0f2040"}`,
              }}>AUTO-SCROLL {autoScroll ? "ON" : "OFF"}</button>
              <button className="chip" onClick={clearLogs} style={{
                padding: "3px 9px", borderRadius: 3, fontSize: 9, letterSpacing: 1,
                background: "#070f1e", color: "#475569", border: "1px solid #0f2040",
              }}>CLEAR</button>
              <div style={{ fontSize: 9, color: "#334155" }}>{filtered.length} / {logEntries.length} entries</div>
            </div>

            {/* Log output */}
            <div style={{ background: "#040c19", borderRadius: 6, border: "1px solid #0f2040", padding: "12px 0", minHeight: "60vh", maxHeight: "70vh", overflowY: "auto" }}>
              {filtered.length === 0 ? (
                <div style={{ padding: "40px 20px", textAlign: "center", color: "#1e3a5f", fontSize: 11 }}>
                  {relayOnline ? "No log entries match filters." : "Waiting for relay connection..."}
                </div>
              ) : filtered.map(e => {
                const levelColor = LEVEL_COLOR[e.level] || "#475569";
                const rowBg      = LEVEL_BG[e.level] || "transparent";
                const isExpanded = expandedId === e.id;
                const hasExtra   = e.traceback || e.prompt_preview || e.response_preview || e.agent_timings;

                return (
                  <div key={e.id} className="log-row fadein" onClick={() => hasExtra && setExpandedId(isExpanded ? null : e.id)}
                    style={{ padding: "3px 16px", background: isExpanded ? "#0a1628" : rowBg, borderBottom: "1px solid #0a162833", cursor: hasExtra ? "pointer" : "default" }}>
                    <div style={{ display: "flex", gap: 10, alignItems: "baseline", fontSize: 10 }}>
                      {/* Timestamp */}
                      <span style={{ color: "#1e3a5f", flexShrink: 0, fontSize: 9, fontVariantNumeric: "tabular-nums" }}>{e.ts}</span>
                      {/* Level badge */}
                      <span style={{ fontSize: 8, padding: "1px 5px", borderRadius: 2, background: levelColor + "18", color: levelColor, letterSpacing: 1, flexShrink: 0, fontWeight: 600 }}>{e.level}</span>
                      {/* Source badge */}
                      <span style={{ fontSize: 8, padding: "1px 5px", borderRadius: 2, background: (SRC_COLOR[e.src] || "#64748b") + "15", color: SRC_COLOR[e.src] || "#64748b", letterSpacing: 1, flexShrink: 0 }}>{e.src}</span>
                      {/* Message */}
                      <span style={{ color: levelColor, lineHeight: 1.7, flex: 1 }}>{e.msg}</span>
                      {/* Expand indicator */}
                      {hasExtra && <span style={{ color: "#1e3a5f", fontSize: 9, flexShrink: 0 }}>{isExpanded ? "▲" : "▼"}</span>}
                    </div>

                    {/* Expanded detail */}
                    {isExpanded && (
                      <div className="fadein" style={{ marginTop: 6, marginLeft: 160, marginBottom: 6 }}>
                        {/* Token stats */}
                        {(e.input_tokens !== undefined) && (
                          <div style={{ display: "flex", gap: 20, marginBottom: 8 }}>
                            {[
                              { l: "INPUT TOKENS",  v: e.input_tokens },
                              { l: "OUTPUT TOKENS", v: e.output_tokens },
                              { l: "TOKENS/SEC",    v: `${e.tokens_per_s} t/s` },
                              { l: "ELAPSED",       v: `${e.elapsed_s}s` },
                            ].map(m => (
                              <div key={m.l} style={{ padding: "6px 10px", background: "#070f1e", borderRadius: 3, border: "1px solid #0f2040" }}>
                                <div style={{ fontSize: 8, color: "#334155", letterSpacing: 1 }}>{m.l}</div>
                                <div style={{ fontSize: 13, color: "#a78bfa", fontFamily: "Bebas Neue", letterSpacing: 1 }}>{m.v}</div>
                              </div>
                            ))}
                          </div>
                        )}
                        {/* Agent timings */}
                        {e.agent_timings && (
                          <div style={{ display: "flex", gap: 10, marginBottom: 8, flexWrap: "wrap" }}>
                            {Object.entries(e.agent_timings).map(([agent, secs]) => (
                              <div key={agent} style={{ padding: "6px 10px", background: "#070f1e", borderRadius: 3, border: "1px solid #0f2040" }}>
                                <div style={{ fontSize: 8, color: "#334155", letterSpacing: 1 }}>{agent.toUpperCase()}</div>
                                <div style={{ fontSize: 13, color: "#10b981", fontFamily: "Bebas Neue", letterSpacing: 1 }}>{secs}s</div>
                              </div>
                            ))}
                          </div>
                        )}
                        {/* Prompt preview */}
                        {e.prompt_preview && (
                          <div style={{ marginBottom: 8 }}>
                            <div style={{ fontSize: 8, color: "#334155", letterSpacing: 1, marginBottom: 4 }}>PROMPT PREVIEW</div>
                            <div style={{ padding: "8px 10px", background: "#070f1e", borderRadius: 3, border: "1px solid #0f2040", fontSize: 9, color: "#475569", lineHeight: 1.8, whiteSpace: "pre-wrap", maxHeight: 120, overflowY: "auto" }}>{e.prompt_preview}</div>
                          </div>
                        )}
                        {/* Response preview */}
                        {e.response_preview && (
                          <div style={{ marginBottom: 8 }}>
                            <div style={{ fontSize: 8, color: "#334155", letterSpacing: 1, marginBottom: 4 }}>RESPONSE PREVIEW</div>
                            <div style={{ padding: "8px 10px", background: "#070f1e", borderRadius: 3, border: "1px solid #a78bfa22", fontSize: 9, color: "#64748b", lineHeight: 1.8, whiteSpace: "pre-wrap", maxHeight: 160, overflowY: "auto" }}>{e.response_preview}</div>
                          </div>
                        )}
                        {/* Traceback */}
                        {showTraceback && e.traceback && (
                          <div>
                            <div style={{ fontSize: 8, color: "#ef4444", letterSpacing: 1, marginBottom: 4 }}>FULL TRACEBACK</div>
                            <div style={{ padding: "8px 10px", background: "#0a0000", borderRadius: 3, border: "1px solid #ef444422", fontSize: 9, color: "#ef4444", lineHeight: 1.8, whiteSpace: "pre-wrap", maxHeight: 300, overflowY: "auto", fontFamily: "monospace" }}>
                              {Array.isArray(e.traceback) ? e.traceback.join("") : e.traceback}
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
              <div ref={logEndRef} />
            </div>
          </div>
        )}

        {/* ══════════════════════ CYCLE VIEW ══════════════════════ */}
        {view === "cycle" && (
          <div>
            <FeedbackLoop agents={AGENTS} agentStates={agentStates} activeAgent={activeAgent} loopPhase={loopPhase} />
            {/* Hardware info bar */}
            <div style={{ marginTop: 14, marginBottom: 14, padding: "10px 16px", background: "#070f1e", border: "1px solid #0f2040", borderRadius: 6, display: "flex", gap: 28, flexWrap: "wrap" }}>
              {[
                { l: "ZENITH SERVER", v: "http://127.0.0.1:8080", c: zenithColor },
                { l: "MODEL",         v: "Qwen3-VL-30B-A3B Q4_K_M", c: "#94a3b8" },
                { l: "GPU",           v: "RX 9070 XT · 16GB Vulkan", c: "#94a3b8" },
                { l: "CONTEXT",       v: "16384 · Flash Attn ✓", c: "#94a3b8" },
                { l: "ASSET",         v: "SPX500 · DRY RUN", c: "#f59e0b" },
                { l: "LOG RELAY",     v: relayOnline ? "ONLINE :8081" : "OFFLINE", c: relayColor },
              ].map(s => (
                <div key={s.l}>
                  <div style={{ fontSize: 9, color: "#334155", letterSpacing: 1.5 }}>{s.l}</div>
                  <div style={{ fontSize: 11, color: s.c, fontFamily: "Bebas Neue", letterSpacing: 1 }}>{s.v}</div>
                </div>
              ))}
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: 14 }}>
              {AGENT_ORDER.map((id, idx) => {
                const ag = AGENTS[id];
                const st = agentStates[id];
                const isActive = st.status === "running";
                return (
                  <div key={id} className="agent-card fadein" style={{ background: "#070f1e", borderRadius: 6, overflow: "hidden", border: `1px solid ${isActive ? ag.color + "55" : "#0f2040"}`, boxShadow: isActive ? `0 0 20px ${ag.color}22` : "none", animationDelay: `${idx * 0.07}s` }}>
                    <div style={{ padding: "12px 14px", borderBottom: "1px solid #0f2040", display: "flex", alignItems: "center", gap: 10, background: isActive ? `${ag.color}08` : "transparent" }}>
                      <span style={{ fontSize: 16, color: ag.color }}>{ag.icon}</span>
                      <div style={{ flex: 1 }}>
                        <div style={{ fontFamily: "Bebas Neue", fontSize: 15, letterSpacing: 2, color: ag.color }}>{ag.label}</div>
                        <div style={{ fontSize: 9, color: "#334155", marginTop: 1 }}>{ag.role}</div>
                      </div>
                      <StatusBadge status={st.status} />
                    </div>
                    <div style={{ height: 2, background: "#0a1628" }}>
                      <div style={{ height: "100%", width: `${st.progress}%`, transition: "width .4s", background: st.status === "success" ? "#10b981" : st.status === "error" ? "#ef4444" : `linear-gradient(90deg, ${ag.color}, ${ag.color}88)` }} />
                    </div>
                    <div style={{ padding: "10px 14px", minHeight: 100, maxHeight: 130, overflowY: "auto" }}>
                      {st.logs.length === 0
                        ? <div style={{ color: "#1e3a5f", fontSize: 10, fontStyle: "italic" }}>Awaiting cycle start...</div>
                        : st.logs.map((line, i) => (
                          <div key={i} style={{ fontSize: 10, lineHeight: 1.85, color: i === st.logs.length-1 ? "#94a3b8" : "#334155" }}>
                            <span style={{ color: "#1e3a5f" }}>›</span> {line}
                            {i === st.logs.length-1 && isActive && <span style={{ animation: "blink 1s infinite", marginLeft: 2, color: ag.color }}>█</span>}
                          </div>
                        ))
                      }
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* ══════════════════════ REPORTS ══════════════════════ */}
        {view === "reports" && (
          <div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 12, marginBottom: 20 }}>
              {[
                { l: "CYCLES RUN",    v: cycles.length, c: "#06b6d4" },
                { l: "SESSION P&L",   v: `${totalPnl >= 0 ? "+" : ""}$${Math.abs(totalPnl).toLocaleString()}`, c: totalPnl >= 0 ? "#10b981" : "#ef4444" },
                { l: "AVG LIVE EV",   v: allStrats.length ? `+${avgEv.toFixed(1)}%` : "—", c: "#f59e0b" },
                { l: "GOOD STRATS",   v: allStrats.filter(s => s.tier === "GOOD").length, c: "#10b981" },
                { l: "RETIRED",       v: allStrats.filter(s => s.tier === "POOR").length, c: "#ef4444" },
              ].map(s => (
                <div key={s.l} style={{ padding: "14px 16px", background: "#070f1e", borderRadius: 6, border: "1px solid #0f2040" }}>
                  <div style={{ fontSize: 9, color: "#334155", letterSpacing: 1.5, marginBottom: 5 }}>{s.l}</div>
                  <div style={{ fontFamily: "Bebas Neue", fontSize: 22, letterSpacing: 1, color: s.c }}>{s.v}</div>
                </div>
              ))}
            </div>
            {cycles.length === 0 ? (
              <div style={{ padding: "60px 20px", textAlign: "center", color: "#1e3a5f", fontSize: 11 }}>No cycles completed yet.</div>
            ) : (
              <div style={{ display: "grid", gridTemplateColumns: selectedCycle ? "1fr 380px" : "1fr", gap: 16 }}>
                <div style={{ background: "#070f1e", borderRadius: 6, border: "1px solid #0f2040", overflow: "hidden" }}>
                  <div style={{ display: "grid", gridTemplateColumns: "70px 90px 80px 80px 70px 1fr 80px", padding: "9px 14px", borderBottom: "1px solid #0f2040", fontSize: 9, color: "#334155", letterSpacing: 1.5 }}>
                    {["CYCLE","TIME","DURATION","REGIME","RISK","COACH ACTION","P&L"].map(h => <div key={h}>{h}</div>)}
                  </div>
                  {cycles.map(cy => (
                    <div key={cy.id} className="hov-row" onClick={() => setSelectedCycle(selectedCycle?.id === cy.id ? null : cy)}
                      style={{ display: "grid", gridTemplateColumns: "70px 90px 80px 80px 70px 1fr 80px", padding: "11px 14px", borderBottom: "1px solid #0a1628", background: selectedCycle?.id === cy.id ? "#0d1e35" : "transparent", fontSize: 10 }}>
                      <div style={{ color: "#06b6d4", fontFamily: "Bebas Neue", fontSize: 14 }}>#{cy.cycle}</div>
                      <div style={{ color: "#475569" }}>{cy.completedAt}</div>
                      <div style={{ color: "#475569" }}>{cy.duration}</div>
                      <div style={{ color: cy.regime === "VOLATILE" ? "#ef4444" : cy.regime === "TRENDING" ? "#10b981" : "#f59e0b", fontSize: 9 }}>{cy.regime}</div>
                      <div style={{ color: "#475569" }}>{cy.riskLevel}</div>
                      <div style={{ color: "#475569", fontSize: 9, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", paddingRight: 8 }}>{cy.coachAction}</div>
                      <div style={{ color: cy.totalPnl >= 0 ? "#10b981" : "#ef4444", fontWeight: 600 }}>{cy.totalPnl >= 0 ? "+" : ""}${Math.abs(cy.totalPnl).toLocaleString()}</div>
                    </div>
                  ))}
                </div>
                {selectedCycle && (
                  <div className="fadein" style={{ background: "#070f1e", borderRadius: 6, border: "1px solid #0f2040", padding: 18, alignSelf: "start", position: "sticky", top: 80 }}>
                    <div style={{ fontFamily: "Bebas Neue", fontSize: 18, letterSpacing: 2, color: "#06b6d4", marginBottom: 4 }}>CYCLE #{selectedCycle.cycle}</div>
                    <div style={{ fontSize: 9, color: "#334155", marginBottom: 14 }}>{selectedCycle.completedAt} · {selectedCycle.duration} · {selectedCycle.builderBatch} candidates · {selectedCycle.surviving} surviving</div>
                    {selectedCycle.strategies.map(st => (
                      <div key={st.id} style={{ marginBottom: 10, padding: "10px 12px", background: "#040c19", borderRadius: 4, borderLeft: `2px solid ${TIER_COLOR[st.tier]}` }}>
                        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
                          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                            <span style={{ fontFamily: "Bebas Neue", fontSize: 16, color: "#e2e8f0" }}>{st.id}</span>
                            <span style={{ fontSize: 9, padding: "2px 6px", borderRadius: 2, color: TIER_COLOR[st.tier], background: TIER_BG[st.tier] }}>{st.tier}</span>
                          </div>
                          <div style={{ fontSize: 12, color: st.pnl >= 0 ? "#10b981" : "#ef4444", fontWeight: 600 }}>{st.pnl >= 0 ? "+" : ""}${Math.abs(st.pnl).toLocaleString()}</div>
                        </div>
                        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
                          {[
                            { l: "LIVE EV",  v: `+${st.liveEv}%`,    sub: `proj +${st.ev}%`,   c: st.liveEv/st.ev >= 0.8 ? "#10b981" : "#f59e0b" },
                            { l: "WIN RATE", v: `${st.liveWinRate}%`, sub: `bt ${st.winRate}%`, c: Math.abs(st.liveWinRate-st.winRate) <= 10 ? "#10b981" : "#f59e0b" },
                            { l: "95th DD",  v: `${st.dd}%`,          sub: st.propFirm,         c: st.dd < 5 ? "#10b981" : "#f59e0b" },
                            { l: "SHARPE",   v: st.sharpe,             sub: `R:R ${st.rr}`,      c: st.sharpe >= 1 ? "#10b981" : "#f59e0b" },
                          ].map(m => (
                            <div key={m.l} style={{ padding: "6px 8px", background: "#070f1e", borderRadius: 3 }}>
                              <div style={{ fontSize: 8, color: "#334155" }}>{m.l}</div>
                              <div style={{ fontSize: 13, color: m.c, fontFamily: "Bebas Neue" }}>{m.v}</div>
                              <div style={{ fontSize: 8, color: "#475569" }}>{m.sub}</div>
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                    <div style={{ marginTop: 10, padding: "9px 12px", background: "#040c19", borderRadius: 4, borderLeft: "2px solid #a78bfa" }}>
                      <div style={{ fontSize: 9, color: "#a78bfa", marginBottom: 4 }}>COACH ACTION</div>
                      <div style={{ fontSize: 10, color: "#64748b", lineHeight: 1.7 }}>{selectedCycle.coachAction}</div>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Feedback Loop SVG ────────────────────────────────────────────────────
function FeedbackLoop({ agents, agentStates, activeAgent, loopPhase }) {
  const nodes = [
    { id: "builder",    x: 100, y: 80, label: "STRATEGY\nBUILDER" },
    { id: "backtester", x: 340, y: 80, label: "STRATEGY\nBACKTESTER" },
    { id: "trader",     x: 580, y: 80, label: "TRADER" },
    { id: "coach",      x: 820, y: 80, label: "TRADING\nCOACH" },
  ];
  const arrows = [
    { id: "builder→backtester", x1: 162, x2: 280 },
    { id: "backtester→trader",  x1: 402, x2: 520 },
    { id: "trader→coach",       x1: 642, x2: 760 },
  ];
  const nodeColor = id => {
    const s = agentStates[id].status;
    if (s === "running") return agents[id].color;
    if (s === "success") return "#10b981";
    if (s === "error")   return "#ef4444";
    return "#1e3a5f";
  };
  const isFB = loopPhase === "coach→builder";

  return (
    <div style={{ background: "#070f1e", border: "1px solid #0f2040", borderRadius: 6, padding: "16px 20px" }}>
      <div style={{ fontSize: 9, color: "#334155", letterSpacing: 2, marginBottom: 10 }}>AGENT PIPELINE · ZENITH LOCAL AI · CONTINUOUS FEEDBACK LOOP</div>
      <svg width="100%" viewBox="0 0 960 200" style={{ overflow: "visible" }}>
        <defs>
          <marker id="a"  markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L0,6 L6,3 z" fill="#1e3a5f" /></marker>
          <marker id="al" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L0,6 L6,3 z" fill="#06b6d4" /></marker>
          <marker id="af" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L0,6 L6,3 z" fill="#a78bfa" /></marker>
        </defs>
        {arrows.map(a => {
          const lit = loopPhase === a.id;
          return (
            <g key={a.id}>
              <line x1={a.x1} y1={84} x2={a.x2} y2={84} stroke={lit ? "#06b6d4" : "#1e3a5f"} strokeWidth={lit ? 2 : 1} markerEnd={lit ? "url(#al)" : "url(#a)"} style={lit ? { filter: "drop-shadow(0 0 4px #06b6d4)" } : {}} />
              {lit && <circle r="3" fill="#06b6d4"><animateMotion dur="0.5s" repeatCount="1" path={`M${a.x1},84 L${a.x2},84`} /></circle>}
            </g>
          );
        })}
        <path d="M 882,110 C 882,170 100,170 100,110" fill="none" stroke={isFB ? "#a78bfa" : "#1a2744"} strokeWidth={isFB ? 2 : 1} strokeDasharray={isFB ? "none" : "4,4"} markerEnd={isFB ? "url(#af)" : "url(#a)"} style={isFB ? { filter: "drop-shadow(0 0 5px #a78bfa)" } : {}} />
        {isFB && <circle r="3" fill="#a78bfa"><animateMotion dur="0.9s" repeatCount="1" path="M 882,110 C 882,170 100,170 100,110" /></circle>}
        <text x="490" y="185" textAnchor="middle" fontSize="8" fill={isFB ? "#a78bfa" : "#1e3a5f"} letterSpacing="1.5">COACH → BUILDER FEEDBACK LOOP</text>
        {nodes.map(n => {
          const color = nodeColor(n.id);
          const isAct = activeAgent === n.id;
          return (
            <g key={n.id}>
              <rect x={n.x-62} y={n.y-34} width={124} height={68} rx={5} fill="#070f1e" stroke={color} strokeWidth={isAct ? 2 : 1} style={isAct ? { filter: `drop-shadow(0 0 8px ${color})` } : {}} />
              <circle cx={n.x+50} cy={n.y-22} r={4} fill={color} style={isAct ? { animation: "pulse 1s infinite" } : {}} />
              <text x={n.x} y={n.y-10} textAnchor="middle" fontSize="14" fill={color}>{agents[n.id].icon}</text>
              {n.label.split("\n").map((line, i) => (
                <text key={i} x={n.x} y={n.y+12+i*13} textAnchor="middle" fontSize="8.5" fontFamily="Bebas Neue" letterSpacing="1.5" fill={isAct ? color : "#64748b"}>{line}</text>
              ))}
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function StatusBadge({ status }) {
  const cfg = { idle: { label: "IDLE", color: "#334155", bg: "#33415515" }, running: { label: "RUNNING", color: "#0ea5e9", bg: "#0ea5e915" }, success: { label: "DONE", color: "#10b981", bg: "#10b98115" }, error: { label: "ERROR", color: "#ef4444", bg: "#ef444415" } };
  const c = cfg[status] || cfg.idle;
  return (
    <div style={{ fontSize: 9, padding: "3px 8px", borderRadius: 3, color: c.color, background: c.bg, border: `1px solid ${c.color}33`, letterSpacing: 1, fontWeight: 600 }}>
      {status === "running" && <span style={{ animation: "pulse 1s infinite", marginRight: 4 }}>●</span>}
      {c.label}
    </div>
  );
}
