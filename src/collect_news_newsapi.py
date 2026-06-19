"""
collect_news_newsapi.py
-----------------------
Collect silver news articles from NewsAPI.ai (Event Registry).

Why this exists: the GDELT feed (news_gdelt.csv) is title-only, ~86% non-English, and
starts only 2017-10. NewsAPI.ai returns full article bodies + per-article sentiment from
a curated, reputable English source set, back to 2015 — a single consistent news source
across the whole 2015-2026 study window.

Token economics ($90 / "5K" plan = 5,000 tokens/month):
  - 1 request = up to 100 articles = 1 page.
  - Recent (< ~30 days): 1 token/page.  Historical: 5 tokens per *searched calendar year*.
  - => query ONE calendar year per run so every page costs 5 tokens, never more.
    (Never pass a multi-year range — that multiplies the per-page cost.)

Usage:
  # 1. add your key to .env:  NEWSAPI_AI_KEY=...
  python src/collect_news_newsapi.py --validate     # cheap recent slice (1 tok/page), no write
  python src/collect_news_newsapi.py --year 2015     # one historical year -> appends to CSV
  python src/collect_news_newsapi.py --year 2016     # rerun with the next year, and so on

Install: pip install requests pandas python-dotenv
"""

import os
import sys
import json
import time
import argparse
from datetime import date, timedelta

import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

RAW_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT_CSV  = os.path.join(RAW_DIR, "news_newsapi.csv")
API_URL  = "https://eventregistry.org/api/v1/article/getArticles"
API_KEY  = os.getenv("NEWSAPI_AI_KEY")

# ── query knobs (tune after the validation run) ───────────────────────────────
KEYWORD       = "silver"
KEYWORD_LOC   = "title"   # precision-first; flip to "body,title" if a thin year misses weeks
LANG          = "eng"
# Reputable-source WHITELIST — not source rank. Rank selects *high-traffic* sources (Indian
# retail price pages, aggregators), which is the wrong axis for "reputable", so we name the
# sources explicitly. Tilted toward high-frequency metals/markets desks (kitco/investing/
# fxstreet/marketwatch) so even the quiet 2015-16 silver bear market clears the ≥1-article/week
# bar. Re-validate after editing: a wrong/unknown URI simply returns 0 rows for that source.
SOURCE_WHITELIST = [
    # high-frequency metals & markets desks — the weekly-coverage backbone
    "kitco.com", "investing.com", "fxstreet.com", "marketwatch.com",
    "barchart.com", "mining.com", "finance.yahoo.com",
    # prestige financial press — quality
    "reuters.com", "bloomberg.com", "cnbc.com", "ft.com", "wsj.com",
    "forbes.com", "businessinsider.com", "seekingalpha.com",
]
PAGE_SIZE     = 100       # API hard max
MAX_PAGES     = 300       # runaway backstop (300 pages * 5 tok = 1500 tok); a real year is far less
SLEEP_S       = 1.0       # polite pause between pages


def _build_query(date_start: str, date_end: str) -> dict:
    """Flat getArticles query for one date window."""
    q = {
        "apiKey":                   API_KEY,
        "keyword":                  KEYWORD,
        "keywordLoc":               KEYWORD_LOC,
        "lang":                     LANG,
        "dateStart":                date_start,
        "dateEnd":                  date_end,
        "dataType":                 "news",
        "isDuplicateFilter":        "skipDuplicates",   # drop syndicated copies
        "sourceUri":                SOURCE_WHITELIST,    # reputable-source whitelist (OR-ed)
        "sourceOper":               "or",
        "resultType":               "articles",
        "articlesSortBy":           "date",
        "articlesCount":            PAGE_SIZE,
        "includeArticleSentiment":  True,
        "includeArticleCategories": False,
        "includeArticleConcepts":   False,
    }
    return q


def _fetch_page(date_start: str, date_end: str, page: int) -> dict:
    """One getArticles call. Returns the parsed `articles` object. Retries on transient errors."""
    q = _build_query(date_start, date_end)
    q["articlesPage"] = page
    for attempt in range(4):
        try:
            resp = requests.post(API_URL, json=q, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict) and data.get("error"):
                raise RuntimeError(f"API error: {data['error']}")
            return data.get("articles", {}) or {}
        except Exception as e:
            wait = 8 * (attempt + 1)
            print(f"  page {page} error (attempt {attempt+1}/4): {type(e).__name__} — {e}; waiting {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"page {page}: all retries failed")


def _row(a: dict) -> dict:
    """Extract the fields we keep. Defensive .get() — the --validate dump confirms the real schema."""
    src = a.get("source") or {}
    return {
        "dateTime":     a.get("dateTime") or a.get("date"),
        "date":         a.get("date"),
        "title":        a.get("title"),
        "body":         a.get("body"),
        "url":          a.get("url"),
        "source_uri":   src.get("uri"),
        "source_title": src.get("title"),
        "sentiment":    a.get("sentiment"),
        "lang":         a.get("lang"),
    }


def _collect_window(date_start: str, date_end: str, tok_per_page: int,
                    max_pages: int = MAX_PAGES) -> pd.DataFrame:
    """Page through a date window, returning all matched articles as a DataFrame."""
    rows, page, pages = [], 1, None
    while page <= max_pages:
        art = _fetch_page(date_start, date_end, page)
        results = art.get("results", []) or []
        if pages is None:
            pages = art.get("pages", 1)
            total = art.get("totalResults", "?")
            print(f"  {date_start} → {date_end}: {total} articles across {pages} page(s) "
                  f"≈ {min(pages, max_pages) * tok_per_page} tokens")
        if not results:
            break
        rows.extend(_row(a) for a in results)
        print(f"  page {page}/{pages}  (+{len(results)} → {len(rows)} total, "
              f"≈ {page * tok_per_page} tokens spent)")
        if page >= (pages or 1):
            break
        page += 1
        time.sleep(SLEEP_S)
    if page > max_pages:
        print(f"  WARNING: hit MAX_PAGES={max_pages} cap — window may be truncated")
    return pd.DataFrame(rows)


def _weekly_coverage(df: pd.DataFrame, label: str):
    """Report W-FRI weeks with zero articles — directly checks the ≥1-article/week requirement."""
    if df.empty:
        print(f"  coverage [{label}]: no articles"); return
    s = pd.to_datetime(df["dateTime"], errors="coerce", utc=True).dt.tz_localize(None).dropna()
    counts = s.dt.to_period("W-FRI").value_counts().sort_index()
    full   = pd.period_range(counts.index.min(), counts.index.max(), freq="W-FRI")
    counts = counts.reindex(full, fill_value=0)
    zero   = counts[counts == 0]
    print(f"  coverage [{label}]: {len(counts)} weeks spanned, "
          f"{len(zero)} with 0 articles, median {int(counts.median())}/wk")
    if len(zero):
        gaps = ", ".join(str(p) for p in zero.index[:12])
        print(f"    ⚠ gap weeks: {gaps}{' …' if len(zero) > 12 else ''}")
        print("    → loosen: flip KEYWORD_LOC to 'body,title' or add a high-frequency source")


def _require_key():
    if not API_KEY:
        sys.exit("ERROR: NEWSAPI_AI_KEY not set. Add it to .env:  NEWSAPI_AI_KEY=your_key_here")


# ── modes ──────────────────────────────────────────────────────────────────────
def validate():
    """Cheap recent slice (last 30 days = 1 token/page). Read-only: dumps schema + a sample,
    so we confirm field names and source quality before spending on the historical archive."""
    _require_key()
    end   = date.today()
    start = end - timedelta(days=30)
    print(f"VALIDATE: recent slice {start} → {end} (1 token/page), max 2 pages\n")

    art = _fetch_page(start.isoformat(), end.isoformat(), page=1)
    results = art.get("results", []) or []
    print(f"totalResults={art.get('totalResults')}  pages={art.get('pages')}\n")
    if not results:
        print("No results — loosen KEYWORD_LOC to 'body,title' or raise SRC_RANK_END, then retry.")
        return

    print("=== raw keys on first article (confirm schema) ===")
    print(json.dumps(sorted(results[0].keys()), indent=2))
    print("\n=== first article (truncated body) ===")
    a0 = dict(results[0])
    if isinstance(a0.get("body"), str):
        a0["body"] = a0["body"][:300] + " …"
    print(json.dumps(a0, indent=2, ensure_ascii=False)[:1500])

    df = pd.DataFrame(_row(a) for a in results)
    print(f"\n=== sample of {len(df)} articles (date / source / sentiment / title) ===")
    show = df[["date", "source_title", "sentiment", "title"]].head(15)
    with pd.option_context("display.max_colwidth", 70, "display.width", 160):
        print(show.to_string(index=False))
    print(f"\nUnique sources in slice: {df['source_uri'].nunique()}")
    print(df["source_uri"].value_counts().to_string())
    missing = [s for s in SOURCE_WHITELIST if s not in set(df["source_uri"])]
    if missing:
        print(f"\nWhitelisted sources with 0 rows in this 30-day slice "
              f"(URI may be wrong, or just quiet — confirm against the year pull): {missing}")
    print("\nValidation only — nothing written. If sources/relevance look right, run --year 2015.")


def collect_year(year: int):
    """Pull one calendar year (5 tokens/page) and merge into OUT_CSV, refreshing that year."""
    _require_key()
    start = f"{year}-01-01"
    end   = min(f"{year}-12-31", date.today().isoformat())   # never query the future
    print(f"YEAR {year}: {start} → {end} (5 tokens/page)\n")

    new = _collect_window(start, end, tok_per_page=5)
    if new.empty:
        print(f"  no articles for {year} — nothing written")
        return
    new["year"] = year

    os.makedirs(RAW_DIR, exist_ok=True)
    if os.path.exists(OUT_CSV):
        prior = pd.read_csv(OUT_CSV)
        prior = prior[prior.get("year") != year]          # refresh this year cleanly
        combined = pd.concat([prior, new], ignore_index=True)
    else:
        combined = new
    combined = (combined.drop_duplicates(subset="url")
                        .sort_values("dateTime")
                        .reset_index(drop=True))
    combined.to_csv(OUT_CSV, index=False)
    print(f"\n  Saved {new.shape[0]} new {year} rows → {combined.shape[0]} total in {OUT_CSV}")
    _weekly_coverage(new, str(year))   # did this year clear the ≥1-article/week bar?


def main():
    ap = argparse.ArgumentParser(description="Collect silver news from NewsAPI.ai, one year per run.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--validate", action="store_true", help="cheap recent-slice probe, no write")
    g.add_argument("--year", type=int, help="calendar year to collect (e.g. 2015)")
    args = ap.parse_args()
    if args.validate:
        validate()
    else:
        collect_year(args.year)


if __name__ == "__main__":
    main()
