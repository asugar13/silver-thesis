"""
collect_prices.py
-----------------
Pulls silver spot prices and macro covariates at mixed frequencies.

Data sources & outputs:
  - yfinance       : Silver (SI=F), Gold (GC=F), USD index (DX-Y.NYB), S&P500 (^GSPC),
                     copper (HG=F), VIX (^VIX), oil (CL=F)
                       → data/raw/daily_prices.csv   (yfinance trading-day index)
  - FRED (fredapi) : DFII10, T10YIE, ICSA
                       → data/raw/daily_fred.csv     (raw FRED dates — Sat-stamped
                       ICSA + daily-business-day DFII10/T10YIE; consumers align via
                       `reindex(prices.index, method='ffill')`)
                     CPI, FedFunds, INDPRO, M2
                       → data/raw/monthly_macro.csv

The yfinance + FRED-daily split keeps raw data raw: each file matches its source's
native calendar, alignment decisions live at the feature-engineering layer
(02_features.ipynb / 01_eda.ipynb) where they're visible to anyone reading the code.

Install:
  pip install yfinance fredapi pandas python-dotenv
"""

import os
import pandas as pd
import yfinance as yf
from fredapi import Fred
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# ── CONFIG ────────────────────────────────────────────────────────────────────

from config import START_DATE as START, END_DATE as END  # single source of truth

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
#   - vix:       fear index — risk-off spikes can drive safe-haven silver demand
#   - oil:       WTI crude — input cost for mining, industrial demand proxy
TICKERS = {
    "silver":    "SI=F",      # Silver futures front month, best free proxy for spot price
    "gold":      "GC=F",      # Gold futures, moves closely with silver and usually leads it
    "usd_index": "DX-Y.NYB",  # ICE Dollar Index, what every paper means by "DXY". A stronger dollar raises the price for non-USD buyers and tends to depress demand and the nominal price
    "sp500":     "^GSPC",     # S&P500 index, risk appetite proxy. When stocks fall, silver sometimes rises as a safe haven
    "copper":    "HG=F",      # Copper futures front month, industrial demand signal. Around half of silver demand is industrial, so copper tracks the part gold misses
    "vix":       "^VIX",      # Fear index, risk-off spikes can drive safe-haven silver demand. A statistic computed from option prices, which go up because people want to hedge when the S&P falls
    "oil":       "CL=F",      # WTI crude, mostly an inflation proxy. Also a small cost input for mining
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

# ── 2. DAILY FRED DATA ────────────────────────────────────────────────────────

# Higher-frequency FRED series (daily or weekly), merged onto the daily price
# grid in main() via reindex + ffill. These sit alongside the yfinance daily
# columns in daily_prices.csv.
#
# Background you need for the comments below:
#   A "Treasury" is a loan you make to the US government by buying a bond. The
#   "yield" is the annual return it pays you.
#   TIPS = Treasury Inflation-Protected Securities. A special government bond
#   whose value rises with inflation, so its yield tells you the return AFTER
#   stripping inflation out. That stripped-out number is the "real" rate: what
#   you actually earn in purchasing power.
#   A normal (non-TIPS) bond yield is "nominal": it includes inflation.
#   So:  nominal yield  =  real yield  +  expected inflation
#   Rearranged, the inflation the market is expecting = nominal − TIPS. That gap
#   is called the "breakeven".

FRED_DAILY = {
    # The return on a 10-year government bond after inflation is removed (the
    # "real" rate). This is probably the single most important driver of silver:
    # silver pays you nothing to hold it, so when real returns elsewhere rise,
    # holding metal looks worse and silver tends to fall.
    "real_rates_10y": "DFII10",

    # The inflation the market expects over the next 10 years, backed out from
    # bond prices (nominal yield minus the TIPS yield). Silver is bought as an
    # inflation hedge, and it reacts to what people EXPECT inflation to be, not
    # just to the official inflation numbers after they come out.
    "breakeven_10y":  "T10YIE",

    # Number of people who filed for unemployment benefits that week. A fast,
    # weekly read on whether the economy is weakening. It is the closest free
    # stand-in for the PMI factory surveys, which are timely but behind a paywall.
    # Note: stamped to the Saturday of its week but only published the following
    # Thursday, so lag it before use or you are leaking future information.
    "jobless_claims": "ICSA",
}

# ── 3. MONTHLY MACRO DATA (FRED) ──────────────────────────────────────────────

# These are the slow-moving economic variables — published monthly by government agencies.
# This is the "mixed-frequency" part of the thesis: these arrive at a different speed
# than the daily prices above, which is exactly what MIDAS is designed to handle.
#   - cpi:       inflation — silver is an inflation hedge, so this matters a lot
#   - fed_funds: interest rates — higher rates strengthen the dollar, pressure silver
#   - ind_prod:  industrial output — proxy for physical silver demand (electronics, solar)
#   - m2:        money supply — loose monetary policy historically lifts commodity prices
FRED_MONTHLY = {
    "cpi":       "CPIAUCSL",  # Consumer Price Index (price level of fixed consumption basket) — realised inflation
    "fed_funds": "FEDFUNDS",  # Federal Funds Rate (overnight interbank lending rate, set by the FOMC Federal Open Market Committee)
    "ind_prod":  "INDPRO",    # Industrial Production Index (Federal Reserve index measuring real output of US manufacturing, mining, and utilities, normalised to 100 in 2017)
    "m2":        "M2SL",      # M2 Broad money supply (rapid M2 growth signals future inflationary pressure, which reprices real assets upward)
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
    return pd.concat(frames.values(), axis=1)


# ── 4. SAVE ───────────────────────────────────────────────────────────────────
def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    print("Fetching daily price data...")
    prices = fetch_daily_prices(TICKERS, START, END)
    out = f"{RAW_DIR}/daily_prices.csv"
    prices.to_csv(out)
    print(f"  Saved {prices.shape} -> {out}\n")

    # FRED daily kept in its own file at native FRED dates — no alignment baked
    # into the collector. Consumers (02_features.ipynb, 01_eda.ipynb) align via
    # `reindex(prices.index, method='ffill')`. method='ffill' (not reindex().ffill())
    # is critical for ICSA: Sat-stamped values would otherwise be dropped at reindex
    # before any ffill could rescue them. Publication-lag note for ICSA: Sat-stamped,
    # published ~5 business days later; the 1-week lag applied in feature-building
    # (.shift(1) on W-FRI aggregation) clears it.
    print("Fetching daily FRED series...")
    fred_daily = fetch_fred_series(FRED_DAILY, START, END, FRED_API_KEY)
    out = f"{RAW_DIR}/daily_fred.csv"
    fred_daily.to_csv(out)
    print(f"  Saved {fred_daily.shape} -> {out}\n")

    # Note: macro data stays at monthly frequency intentionally.
    # We do NOT upsample it here — that happens in 02_features.ipynb via forward-fill,
    # and the raw monthly version is kept for MIDAS which needs the original frequency.
    print("Fetching monthly FRED macro data...")
    macro = fetch_fred_series(FRED_MONTHLY, START, END, FRED_API_KEY)
    out = f"{RAW_DIR}/monthly_macro.csv"
    macro.to_csv(out)
    print(f"  Saved {macro.shape} -> {out}\n")

    print("Done. Next: run collect_reddit.py, collect_news.py, collect_cot.py")


if __name__ == "__main__":
    main()


# ── CHANGELOG ────────────────────────────────────────────────────────────────
# 2026-05-28 — variable cleanup + new daily FRED series.
#   Added — new file data/raw/daily_fred.csv (FRED_DAILY group):
#     • DFII10 (real_rates_10y) — 10Y TIPS yield. The canonical real-rate driver
#       of gold/silver; market-traded so no model-revision risk.
#     • T10YIE (breakeven_10y) — market-implied 10Y expected inflation. Silver
#       reacts to expectations, not just realised CPI prints — needs both.
#     • ICSA (jobless_claims) — weekly initial claims. Closest free substitute
#       for ISM PMI's release-timing edge (ISM PMI is paywalled on FRED).
#   Removed:
#     • TICKERS["tip_etf"] (TIP ETF) — superseded by DFII10. The ETF price is a
#       leveraged exposure to the rate with duration + MTM noise on top; DFII10
#       is the rate directly.
#     • FRED_SERIES["usd_dxy"] (DTWEXBGS) — already covered by DX-Y.NYB via
#       yfinance. Tradeoff: kept market-weighted DXY (6 currencies, finance-
#       standard reference) over FRED's trade-weighted broad dollar.
#     • FRED_SERIES["real_rates"] (REAINTRATREARAT10Y) — Cleveland Fed's model-
#       derived monthly real rate. DFII10 is the daily market-priced version;
#       single canonical real-rate feature is cleaner than carrying both.
#     • FRED_SERIES["silver_price_fred"] (SLVPRUSD) — was only a sanity check vs
#       yfinance silver, never consumed downstream.
#     • EIA API line from the docstring — was never implemented; the available
#       silver industrial-demand sources (Silver Institute, USGS) are annual at
#       best, not compatible with this weekly-forecast pipeline.
#   Renamed:
#     • FRED_SERIES → FRED_MONTHLY now that FRED_DAILY also exists.
