"""
collect_prices.py
-----------------
Pulls silver spot prices and macro covariates at mixed frequencies.

Data sources:
  - yfinance     : Silver spot (SI=F), Gold (GC=F), USD index (DX-Y.NYB), S&P500 (^GSPC)  [daily]
  - FRED (fredapi): CPI, Fed Funds Rate, Industrial Production, M2                          [monthly]
  - EIA API      : U.S. silver industrial demand (optional, requires free API key)          [annual/quarterly]

Install:
  pip install yfinance fredapi pandas requests
"""

import os
import pandas as pd
import yfinance as yf
from fredapi import Fred
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# ── CONFIG ────────────────────────────────────────────────────────────────────

# Date range for the entire study — 10 years of data
START = "2015-01-01"
END   = "2024-12-31"

# Always save relative to this script's location, regardless of where you run it from
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

# FRED is the Federal Reserve's free economic data platform (~800k series).
# Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html
FRED_API_KEY = os.getenv("FRED_API_KEY", "YOUR_FRED_KEY_HERE")


# ── 1. DAILY PRICE DATA (yfinance) ────────────────────────────────────────────

# These are the market variables we'll use as daily covariates in ARIMAX and LSTM.
# Each one was chosen because it has a known economic relationship with silver:
#   - gold:      silver tracks gold closely (both are safe-haven assets)
#   - usd_index: dollar up → silver down (priced in USD, inverse relationship)
#   - sp500:     risk appetite proxy — when stocks fall, silver sometimes rises
#   - copper:    industrial demand signal (both copper and silver are used in electronics)
#   - tip_etf:   inflation expectations — silver is an inflation hedge
TICKERS = {
    "silver":    "SI=F",      # Silver futures front month — best free proxy for spot price
    "gold":      "GC=F",
    "usd_index": "DX-Y.NYB",
    "sp500":     "^GSPC",
    "tip_etf":   "TIP",
    "copper":    "HG=F",
}

def fetch_daily_prices(tickers: dict, start: str, end: str) -> pd.DataFrame:
    # Download closing price for each ticker and stack into one DataFrame
    frames = {}
    for name, ticker in tickers.items():
        print(f"  Fetching {name} ({ticker})...")
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        # Newer yfinance returns a MultiIndex — squeeze() flattens it to a plain Series
        close = df["Close"].squeeze()
        close.name = name
        frames[name] = close
    combined = pd.concat(frames.values(), axis=1)
    combined.index = pd.to_datetime(combined.index)
    return combined  # shape: (trading days, n_tickers)


# ── 2. MONTHLY MACRO DATA (FRED) ──────────────────────────────────────────────

# These are the slow-moving economic variables — published monthly by government agencies.
# This is the "mixed-frequency" part of the thesis: these arrive at a different speed
# than the daily prices above, which is exactly what MIDAS is designed to handle.
#   - cpi:        inflation — silver is an inflation hedge, so this matters a lot
#   - fed_funds:  interest rates — higher rates strengthen the dollar, pressure silver
#   - ind_prod:   industrial output — proxy for physical silver demand (electronics, solar)
#   - m2:         money supply — loose monetary policy historically lifts commodity prices
#   - real_rates: 10Y real rate — arguably the single most important macro driver of silver
FRED_SERIES = {
    "cpi":               "CPIAUCSL",            # Consumer Price Index
    "fed_funds":         "FEDFUNDS",            # Federal Funds Rate
    "ind_prod":          "INDPRO",              # Industrial Production Index
    "m2":                "M2SL",                # M2 money supply
    "usd_dxy":           "DTWEXBGS",            # Broad dollar index (daily on FRED)
    "real_rates":        "REAINTRATREARAT10Y",  # 10Y real interest rate
    "silver_price_fred": "SLVPRUSD",            # FRED's own monthly silver price (sanity check)
}

def fetch_fred_series(series: dict, start: str, end: str, api_key: str) -> pd.DataFrame:
    fred = Fred(api_key=api_key)
    frames = {}
    for name, series_id in series.items():
        print(f"  Fetching FRED: {name} ({series_id})...")
        try:
            s = fred.get_series(series_id, observation_start=start, observation_end=end)
            frames[name] = s.rename(name)
        except Exception as e:
            # Non-fatal — skip the series and continue with the rest
            print(f"  Warning: could not fetch {series_id}: {e}")
    return pd.concat(frames.values(), axis=1)  # shape: (months, n_series)


# ── 3. SAVE ───────────────────────────────────────────────────────────────────
def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    print("Fetching daily price data...")
    prices = fetch_daily_prices(TICKERS, START, END)
    out = f"{RAW_DIR}/daily_prices.csv"
    prices.to_csv(out)
    print(f"  Saved {prices.shape} -> {out}\n")

    # Note: macro data stays at monthly frequency intentionally.
    # We do NOT upsample it here — that happens in 02_features.ipynb via forward-fill,
    # and the raw monthly version is kept for MIDAS which needs the original frequency.
    print("Fetching monthly FRED macro data...")
    macro = fetch_fred_series(FRED_SERIES, START, END, FRED_API_KEY)
    out = f"{RAW_DIR}/monthly_macro.csv"
    macro.to_csv(out)
    print(f"  Saved {macro.shape} -> {out}\n")

    print("Done. Next: run collect_reddit.py and collect_news.py")


if __name__ == "__main__":
    main()
