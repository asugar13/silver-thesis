"""
collect_news_gkg.py
-------------------
Full-window silver news sentiment from the GDELT GKG via BigQuery, using the
GKG's own per-article tone as the signal.

Why tone, not FinBERT:
  The GKG-on-BigQuery `Extras` field has no usable <PAGE_TITLE>, so there are no
  headlines to FinBERT (see the abandoned collect_news_gkg_backfill.py). GKG's
  V2Tone is computed on the full (machine-translated-to-English) article text, so
  it needs no title and is valid across languages. We pull the whole 2015–2026
  window so the news-sentiment series is one homogeneous, content-based measure —
  and so it overlaps the FinBERT series (2017+) for an apples-to-apples check
  before deciding which to use downstream.

Relevance:
  collect_news.py's DOC keyword matched article *content*; URL-slug matching on the
  GKG can't, so 'silver' alone pulls in Adam Silver (NBA), Silverstone, Silver
  Spring, etc. The fix: require the article to also carry an economic theme
  (V2Themes LIKE ECON_/WB_), which the sports/crime/entertainment junk lacks. This
  SQL filter is deliberately *loose* — the raw V2Themes column is kept in the
  output so relevance can be tightened locally (in 03_sentiment.ipynb) without
  re-running the query.

Output:
  data/raw/news_gkg.csv  — [seendate, url, domain, language, tone, v2themes]
    Raw tone (≈ −10…+10); the tanh rescale to [−1, 1] lives in 03_sentiment.ipynb.

Cost:
  Queries the daily-partitioned `gdelt-bq.gdeltv2.gkg_partitioned` and filters on
  _PARTITIONTIME so only the in-window partitions are scanned. Dry-runs first and
  prints the estimated scan before running for real.

Auth:
  gcloud auth application-default login   (or GOOGLE_APPLICATION_CREDENTIALS=key.json)
  Billing project via GCP_PROJECT / GOOGLE_CLOUD_PROJECT in .env, else the ADC default.

Install:
  pip install google-cloud-bigquery db-dtypes pandas python-dotenv
"""

import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from config import START_DATE as START, END_DATE as END  # single source of truth

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT_FILE = os.path.join(RAW_DIR, "news_gkg.csv")

PROJECT = os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")

# Loose economic-theme prefixes — drops sports/crime/entertainment 'silver' noise
# while keeping recall high. Tighten downstream against the saved V2Themes column.
ECON_THEME_PREFIXES = ["ECON_", "WB_"]
_THEME_CLAUSE = " OR ".join(f"V2Themes LIKE '%{p}%'" for p in ECON_THEME_PREFIXES)

QUERY = f"""
SELECT
  DATE,                       -- INT64, YYYYMMDDHHMMSS
  DocumentIdentifier  AS url,
  SourceCommonName    AS domain,
  V2Tone,                     -- comma-sep; field 0 = average document tone
  V2Themes,                   -- kept raw for local relevance tightening
  TranslationInfo             -- empty ⇒ English-origin; populated ⇒ translated
FROM `gdelt-bq.gdeltv2.gkg_partitioned`
WHERE _PARTITIONTIME >= @start AND _PARTITIONTIME < @end
  AND LOWER(DocumentIdentifier) LIKE '%silver%'
  AND ({_THEME_CLAUSE})
"""


def _language(translation_info: str) -> str:
    """English-origin records leave TranslationInfo empty; translated ones tag the
    source language. Binary label — tone is valid regardless, this is audit-only."""
    return "Other" if isinstance(translation_info, str) and translation_info.strip() else "English"


def _tone(v2tone: str) -> float:
    """First field of V2Tone = average tone (negative ↔ positive sentiment)."""
    if not isinstance(v2tone, str) or not v2tone:
        return float("nan")
    try:
        return float(v2tone.split(",")[0])
    except (ValueError, IndexError):
        return float("nan")


def fetch_gkg(start: str, end: str) -> pd.DataFrame:
    """Run the GKG tone query for [start, end) and shape the output frame."""
    try:
        from google.cloud import bigquery
    except ImportError:
        raise SystemExit(
            "google-cloud-bigquery not installed — run:\n"
            "  pip install google-cloud-bigquery db-dtypes"
        )

    client = bigquery.Client(project=PROJECT)
    params = [
        bigquery.ScalarQueryParameter("start", "TIMESTAMP", f"{start} 00:00:00"),
        bigquery.ScalarQueryParameter("end",   "TIMESTAMP", f"{end} 00:00:00"),
    ]

    # Dry run first — never run a costly query blind.
    dry = client.query(
        QUERY,
        job_config=bigquery.QueryJobConfig(
            query_parameters=params, dry_run=True, use_query_cache=False
        ),
    )
    print(f"  Estimated scan: {dry.total_bytes_processed / 1e9:.2f} GB")

    rows = client.query(
        QUERY, job_config=bigquery.QueryJobConfig(query_parameters=params)
    ).to_dataframe()
    if rows.empty:
        return rows

    dt = pd.to_datetime(rows["DATE"].astype("int64").astype(str),
                        format="%Y%m%d%H%M%S", errors="coerce")
    out = pd.DataFrame({
        "seendate": dt.dt.strftime("%Y%m%dT%H%M%SZ"),
        "url":      rows["url"],
        "domain":   rows["domain"],
        "language": rows["TranslationInfo"].map(_language),
        "tone":     rows["V2Tone"].map(_tone),
        "v2themes": rows["V2Themes"],
    })
    return out.dropna(subset=["seendate"]).reset_index(drop=True)


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    print(f"Fetching GKG silver tone ({START} -> {END})...")

    df = fetch_gkg(START, END)
    if df.empty:
        print("  No rows returned — nothing to save.")
        return

    df = df.sort_values("seendate").reset_index(drop=True)
    df.to_csv(OUT_FILE, index=False)
    print(f"  Saved {df.shape} -> {OUT_FILE}")
    print(f"  Date span: {df['seendate'].min()} -> {df['seendate'].max()}")
    print(f"  Languages: {df['language'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()


# ── CHANGELOG ────────────────────────────────────────────────────────────────
# 2026-05-31 — replaces collect_news_gkg_backfill.py.
#   The backfill aimed to FinBERT GKG titles, but GKG-on-BigQuery has no titles.
#   Pivoted to GKG's own tone over the full 2015–2026 window (homogeneous, no title
#   needed, overlaps FinBERT for comparison) with an economic-theme relevance
#   filter replacing the noisy URL-slug-only match.
