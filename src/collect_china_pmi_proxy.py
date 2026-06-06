"""
collect_china_pmi_proxy.py
--------------------------
Fetches a monthly China manufacturing-confidence / PMI proxy from FRED and saves
it to data/raw/monthly_china_pmi_proxy.csv.

FRED series: CHNBSCICP02STSAM
  - Full name : Business Tendency Surveys (Manufacturing): Confidence
                Indicators: Composite Indicators: National Indicator for China
  - Source    : OECD Main Economic Indicators (sourced from NBS China)
  - Freq      : Monthly, seasonally adjusted
  - Units     : Balance of opinion (% reporting improvement minus
                % reporting deterioration). This is a proxy, not the
                50-centred headline NBS Manufacturing PMI from the NBS
                press releases.
  - Coverage  : ~1996 onwards

Timing / publication lag:
  This is a monthly variable. Treat the FRED/OECD date as a period stamp, not
  as the date the value became observable. The official China PMI is typically
  released around month-end / the next calendar day, while this FRED/OECD proxy
  can update later in FRED. If this series is added to 02_features.ipynb, it should be
  availability-lagged like the other monthly macro variables; do not simply
  forward-fill the raw monthly stamp into the reference month.

  Current feature-code convention uses days from a month-start FRED stamp:
      china_pmi_proxy: 31  # public NBS PMI availability, month-end / next-day

  A longer lag would only be needed for a strict FRED-vintage exercise that treats
  the FRED/OECD update date, rather than the public NBS release, as availability.

  If you need exact point-in-time availability, use the NBS release calendar or
  FRED/ALFRED vintages instead of a fixed lag.

Output:
  data/raw/monthly_china_pmi_proxy.csv  — columns: [date, china_pmi_proxy]
    date            : FRED monthly period stamp, not publication date
    china_pmi_proxy : float, balance-of-opinion %

Pipeline fit, if added to 02_features.ipynb:
    pmi = pd.read_csv("data/raw/monthly_china_pmi_proxy.csv",
                      index_col="date", parse_dates=True)
    # Build publication-availability lags first, then join/broadcast.

Install — same deps as collect_prices.py, nothing new needed:
    pip install fredapi pandas python-dotenv
"""

import os
import sys
import pandas as pd
from fredapi import Fred
from dotenv import load_dotenv

# ── CONFIG ────────────────────────────────────────────────────────────────────

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

try:
    from config import START_DATE as START, END_DATE as END
except ImportError:
    START = "2015-01-01"
    END   = "2026-05-31"

RAW_DIR      = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
FRED_API_KEY = os.getenv("FRED_API_KEY", "YOUR_FRED_KEY_HERE")
FRED_SERIES  = "CHNBSCICP02STSAM"   # confirmed live on FRED as of 2026
OUTPUT_FILE  = os.path.join(RAW_DIR, "monthly_china_pmi_proxy.csv")


# ── FETCH ─────────────────────────────────────────────────────────────────────

def fetch_china_pmi(series_id: str, start: str, end: str, api_key: str) -> pd.DataFrame:
    if api_key == "YOUR_FRED_KEY_HERE":
        sys.exit(
            "[ERROR] No FRED API key found.\n"
            "  Set FRED_API_KEY in your .env file or as an environment variable.\n"
            "  Free key: https://fred.stlouisfed.org/docs/api/api_key.html"
        )

    print(f"  Fetching FRED: china_pmi_proxy ({series_id})...")
    fred = Fred(api_key=api_key)
    s = fred.get_series(series_id, observation_start=start, observation_end=end)
    s.index = pd.to_datetime(s.index)
    s.name  = "china_pmi_proxy"

    df = s.to_frame().reset_index().rename(columns={"index": "date"})
    df = df.dropna(subset=["china_pmi_proxy"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    df = fetch_china_pmi(FRED_SERIES, START, END, FRED_API_KEY)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"  Saved {df.shape} -> {OUTPUT_FILE}")
    print(f"  Date range : {df['date'].min().date()} -> {df['date'].max().date()}")
    latest = df.iloc[-1]
    direction = "positive" if latest["china_pmi_proxy"] >= 0 else "negative"
    print(f"  Latest     : {latest['china_pmi_proxy']:.2f} ({direction} proxy reading)")
    print("\nDone. Next: run collect_reddit.py, collect_news.py, collect_cot.py")


if __name__ == "__main__":
    main()
