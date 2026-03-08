@echo off
title ZENITH AGENTIC STACK
set "PY_PATH=C:\Users\Tyler\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\python.exe"
set "BASE_DIR=C:\Users\Tyler\agent-network"

echo ==================================================
echo   ALGO-MESH V3 : STARTING SYSTEM STACK
echo ==================================================

:: 1. Start Log Relay
echo [1/3] Starting Log Relay (Port 8081)...
start "RELAY" "%PY_PATH%" "%BASE_DIR%\zenith_log_relay.py"
timeout /t 3 >nul

:: 2. Start Orchestrator
echo [2/3] Starting Orchestrator (Port 8000)...
start "ORCHESTRATOR" "%PY_PATH%" "%BASE_DIR%\zenith_orchestrator.py"
timeout /t 5 >nul

:: 3. Start Frontend UI
echo [3/3] Launching React Development Server...
echo (Significant delay added to prevent Connection Refused)
:: We use 'start /min' to run the npm process in the background
start /min cmd /c "cd /d %BASE_DIR% && npm start"

:: 15-second grace period for React to compile
timeout /t 15 >nul

echo [FINAL] Opening Dashboard...
start http://localhost:3000

echo.
echo ==================================================
echo   ALL SYSTEMS TRIGGERED. 
echo   Check 'ORCHESTRATOR' window for Port 8000 status.
echo ==================================================
pause