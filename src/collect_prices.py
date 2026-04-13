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

# ── CONFIG ────────────────────────────────────────────────────────────────────
START = "2015-01-01"
END   = "2024-12-31"
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

FRED_API_KEY = os.getenv("FRED_API_KEY", "YOUR_FRED_KEY_HERE")  # https://fred.stlouisfed.org/docs/api/api_key.html

# ── 1. DAILY PRICE DATA (yfinance) ────────────────────────────────────────────
TICKERS = {
    "silver":    "SI=F",      # Silver futures (front month) — best free proxy for spot
    "gold":      "GC=F",      # Gold/silver ratio covariate
    "usd_index": "DX-Y.NYB",  # Dollar strength — strong inverse relationship with silver
    "sp500":     "^GSPC",     # Risk-on/off proxy
    "tip_etf":   "TIP",       # Inflation expectations proxy
    "copper":    "HG=F",      # Industrial demand proxy
}

def fetch_daily_prices(tickers: dict, start: str, end: str) -> pd.DataFrame:
    frames = {}
    for name, ticker in tickers.items():
        print(f"  Fetching {name} ({ticker})...")
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        close = df["Close"].squeeze()  # flatten MultiIndex to Series
        close.name = name
        frames[name] = close
    combined = pd.concat(frames.values(), axis=1)
    combined.index = pd.to_datetime(combined.index)
    return combined


# ── 2. MONTHLY MACRO DATA (FRED) ──────────────────────────────────────────────
FRED_SERIES = {
    "cpi":          "CPIAUCSL",   # Consumer Price Index (inflation)
    "fed_funds":    "FEDFUNDS",   # Federal Funds Rate
    "ind_prod":     "INDPRO",     # Industrial Production Index
    "m2":           "M2SL",       # M2 money supply
    "usd_dxy":      "DTWEXBGS",   # Broad dollar index (daily on FRED)
    "real_rates":   "REAINTRATREARAT10Y",  # 10Y real interest rate
    "silver_price_fred": "SLVPRUSD",       # FRED's own silver price series (monthly)
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
            print(f"  Warning: could not fetch {series_id}: {e}")
    return pd.concat(frames.values(), axis=1)


# ── 3. SAVE ───────────────────────────────────────────────────────────────────
def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    print("Fetching daily price data...")
    prices = fetch_daily_prices(TICKERS, START, END)
    out = f"{RAW_DIR}/daily_prices.csv"
    prices.to_csv(out)
    print(f"  Saved {prices.shape} -> {out}\n")

    print("Fetching monthly FRED macro data...")
    macro = fetch_fred_series(FRED_SERIES, START, END, FRED_API_KEY)
    out = f"{RAW_DIR}/monthly_macro.csv"
    macro.to_csv(out)
    print(f"  Saved {macro.shape} -> {out}\n")

    print("Done. Next: run collect_reddit.py and collect_news.py")


if __name__ == "__main__":
    main()
