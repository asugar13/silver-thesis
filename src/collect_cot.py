"""One-shot collector for the COMEX silver Commitments-of-Traders disaggregated report.

Pulls history from `cot_hist` (covers through 2016) and then `cot_year(yyyy)` for each
subsequent year, filters to the silver row, keeps a small set of position columns,
and writes `data/raw/cot_silver.csv` (date-indexed).

Run once:  python src/collect_cot.py
"""
from __future__ import annotations

import os
from pathlib import Path

import cot_reports as ct
import pandas as pd

SILVER_LABEL = "SILVER - COMMODITY EXCHANGE INC."
COT_TYPE     = "disaggregated_fut"
YEARS_AFTER_HIST = list(range(2017, 2027))   # cot_hist covers through 2016; backfill rest year-by-year

KEEP_COLS = [
    "Report_Date_as_YYYY-MM-DD",
    "Open_Interest_All",
    "Prod_Merc_Positions_Long_All",
    "Prod_Merc_Positions_Short_All",
    "Swap_Positions_Long_All",
    "Swap__Positions_Short_All",
    "M_Money_Positions_Long_All",
    "M_Money_Positions_Short_All",
    "Other_Rept_Positions_Long_All",
    "Other_Rept_Positions_Short_All",
    "NonRept_Positions_Long_All",
    "NonRept_Positions_Short_All",
]

OUT = Path(__file__).resolve().parent.parent / "data" / "raw" / "cot_silver.csv"


def filter_silver(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["Market_and_Exchange_Names"] == SILVER_LABEL].copy()
    keep = [c for c in KEEP_COLS if c in df.columns]
    df = df[keep]
    df = df.rename(columns={"Report_Date_as_YYYY-MM-DD": "report_date"})
    df["report_date"] = pd.to_datetime(df["report_date"])
    for c in df.columns:
        if c == "report_date":
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("report_date").reset_index(drop=True)


def main() -> None:
    print(f"COT report type: {COT_TYPE}")
    print("Fetching historical archive (1986 → 2016)...")
    hist = ct.cot_hist(cot_report_type=COT_TYPE, store_txt=False, verbose=False)
    hist = filter_silver(hist)
    print(f"  hist rows: {len(hist)}")

    pieces = [hist]
    for y in YEARS_AFTER_HIST:
        try:
            yr = ct.cot_year(year=y, cot_report_type=COT_TYPE, store_txt=False, verbose=False)
        except Exception as exc:
            print(f"  {y}: skipped ({exc})")
            continue
        yr = filter_silver(yr)
        print(f"  {y}: {len(yr)} rows")
        pieces.append(yr)

    out = pd.concat(pieces, ignore_index=True).drop_duplicates("report_date").sort_values("report_date")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"\nWrote {OUT} — {len(out)} rows, {out['report_date'].min().date()} → {out['report_date'].max().date()}")


if __name__ == "__main__":
    main()
