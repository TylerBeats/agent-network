"""
download_market_data.py
───────────────────────
Run this ONCE while online to download all market data locally.
Drop in C:\\Users\\Tyler\\agent-network\\
Run: python download_market_data.py

Downloads all timeframes for all 4 assets and saves to data/local_cache/
After this runs, yahoo.py reads from local files — no internet needed.

Yahoo Finance limits:
  1m  → 7 days max
  5m  → 60 days max
  15m → 60 days max
  30m → 60 days max
  1h  → 730 days max
  4h  → 730 days (aggregated from 1h)
  1d  → full history (decades)
"""

import os
import time
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime

# ── Output directory ──────────────────────────────────────────────────────
CACHE_DIR = Path(__file__).parent / "data" / "local_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Assets ────────────────────────────────────────────────────────────────
ASSETS = {
    "spx500":  "SPY",       # S&P 500 — SPY ETF (real volume)
    "nas100":  "QQQ",       # Nasdaq-100 — QQQ ETF
    "us30":    "DIA",       # Dow Jones — DIA ETF
    "xauusd":  "GLD",       # Gold — GLD ETF (real volume, tracks spot)
}

# ── Timeframes ────────────────────────────────────────────────────────────
# (label, yf_interval, yf_period, bucket_seconds)
TIMEFRAMES = [
    ("1m",   "1m",  "7d",    60),
    ("5m",   "5m",  "60d",   300),
    ("15m",  "15m", "60d",   900),
    ("30m",  "30m", "60d",   1800),
    ("1h",   "1h",  "730d",  3600),
    ("4h",   "1h",  "730d",  14400),   # downloaded as 1h, aggregated to 4h
    ("1d",   "1d",  "max",   86400),
]


def download_asset(name: str, symbol: str):
    print(f"\n{'='*50}")
    print(f"  {name.upper()} ({symbol})")
    print(f"{'='*50}")

    for label, interval, period, bucket_seconds in TIMEFRAMES:
        out_path = CACHE_DIR / f"{name}_{label}.csv"

        print(f"  [{label}] Downloading {symbol} {interval} {period}...", end=" ", flush=True)

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval, auto_adjust=True)

            if df is None or df.empty:
                print(f"✗ No data returned")
                continue

            # Aggregate 1h → 4h
            if label == "4h":
                agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
                df = df.resample("4h").agg(agg).dropna(subset=["Open", "Close"])

            # Drop rows with zero/invalid prices
            df = df[(df["Open"] > 0) & (df["Close"] > 0) & (df["High"] > 0) & (df["Low"] > 0)]

            if df.empty:
                print(f"✗ All rows invalid after filtering")
                continue

            df.to_csv(out_path)
            size_kb = out_path.stat().st_size / 1024
            print(f"✓ {len(df)} bars → {out_path.name} ({size_kb:.1f} KB)")

        except Exception as e:
            print(f"✗ Failed: {e}")

        # Small delay to avoid rate limiting
        time.sleep(0.5)


def main():
    print(f"\nMarket Data Downloader")
    print(f"Output: {CACHE_DIR.absolute()}")
    print(f"Assets: {', '.join(ASSETS.keys())}")
    print(f"Timeframes: {', '.join(t[0] for t in TIMEFRAMES)}")
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}")

    for name, symbol in ASSETS.items():
        download_asset(name, symbol)

    print(f"\n{'='*50}")
    print(f"  Download complete: {datetime.now().strftime('%H:%M:%S')}")
    print(f"  Files saved to: {CACHE_DIR.absolute()}")

    # Print summary
    print(f"\n  Summary:")
    total_size = 0
    for f in sorted(CACHE_DIR.glob("*.csv")):
        size_kb = f.stat().st_size / 1024
        total_size += size_kb
        rows = sum(1 for _ in open(f)) - 1
        print(f"    {f.name:<30} {rows:>5} bars  {size_kb:>7.1f} KB")
    print(f"\n  Total: {total_size/1024:.1f} MB")
    print(f"\n  You can now run python main.py offline.")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
