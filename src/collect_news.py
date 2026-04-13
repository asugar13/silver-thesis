"""
collect_news.py
---------------
Collects financial news headlines for silver sentiment analysis.

Sources:
  1. GDELT Project     — free, full historical coverage back to 1979, no key needed
  2. NewsAPI           — free tier: 1 month history, 100 req/day (good for recent data)
  3. Alpha Vantage     — free tier includes news sentiment API with silver ticker filter

Install:
  pip install requests pandas gdeltdoc
"""

import os
import time
import requests
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

# ── 1. GDELT (recommended for historical depth) ────────────────────────────────
# GDELT indexes global news and is completely free.
# GKG (Global Knowledge Graph) table has tone/sentiment scores built in.

from gdeltdoc import GdeltDoc, Filters

def fetch_gdelt_headlines(keywords: list, start: str, end: str) -> pd.DataFrame:
    """
    Searches GDELT for articles matching any of the keywords.
    start/end format: "YYYY-MM-DD"

    Returns DataFrame with columns: datetime, title, url, domain, tone
    Note: GDELT free API returns max 250 articles per query window.
    Run in rolling monthly windows for full coverage.
    """
    gd = GdeltDoc()
    f = Filters(
        keyword   = " OR ".join(keywords),
        start_date = start,
        end_date   = end,
    )
    try:
        articles = gd.article_search(f)
        return articles
    except Exception as e:
        print(f"  GDELT error: {e}")
        return pd.DataFrame()


def fetch_gdelt_full(keywords: list, start: str, end: str) -> pd.DataFrame:
    """Pages through GDELT in 2-week windows."""
    from datetime import datetime, timedelta
    frames   = []
    start_dt = datetime.fromisoformat(start)
    end_dt   = datetime.fromisoformat(end)
    current  = start_dt

    while current < end_dt:
        next_dt = min(current + timedelta(days=14), end_dt)
        print(f"  GDELT: {current.date()} -> {next_dt.date()}")
        df = fetch_gdelt_headlines(keywords,
                                   current.strftime("%Y-%m-%d"),
                                   next_dt.strftime("%Y-%m-%d"))
        if not df.empty:
            frames.append(df)
        current = next_dt
        time.sleep(0.5)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ── 2. NewsAPI (good for the last 12 months, easy to use) ─────────────────────
# Sign up at https://newsapi.org — free plan gives 1 month history

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "YOUR_NEWSAPI_KEY")

def fetch_newsapi(query: str, from_date: str, to_date: str,
                  page_size: int = 100) -> pd.DataFrame:
    url = "https://newsapi.org/v2/everything"
    params = {
        "q":        query,
        "from":     from_date,
        "to":       to_date,
        "language": "en",
        "sortBy":   "publishedAt",
        "pageSize": page_size,
        "apiKey":   NEWSAPI_KEY,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    articles = resp.json().get("articles", [])
    rows = [{
        "datetime": a["publishedAt"],
        "source":   a["source"]["name"],
        "title":    a["title"],
        "description": a.get("description", ""),
        "url":      a["url"],
    } for a in articles]
    df = pd.DataFrame(rows)
    if not df.empty:
        df["datetime"] = pd.to_datetime(df["datetime"])
    return df


# ── 3. Alpha Vantage News Sentiment ───────────────────────────────────────────
# Free API key at https://www.alphavantage.co/support/#api-key
# Returns articles with pre-computed sentiment scores per ticker.

AV_KEY = os.getenv("ALPHA_VANTAGE_KEY", "YOUR_AV_KEY")

def fetch_av_news(tickers: str = "SILVER", limit: int = 200) -> pd.DataFrame:
    """
    tickers: comma-separated. Use 'SILVER' or 'SLV' for silver ETF coverage.
    Note: free tier is rate-limited to 25 req/day.
    """
    url = "https://www.alphavantage.co/query"
    params = {
        "function":  "NEWS_SENTIMENT",
        "tickers":   tickers,
        "limit":     limit,
        "apikey":    AV_KEY,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    feed = resp.json().get("feed", [])
    rows = []
    for article in feed:
        for ts in article.get("ticker_sentiment", []):
            if ts["ticker"] in tickers.split(","):
                rows.append({
                    "datetime":         article["time_published"],
                    "title":            article["title"],
                    "source":           article["source"],
                    "overall_sentiment_score": article["overall_sentiment_score"],
                    "overall_sentiment_label": article["overall_sentiment_label"],
                    "ticker_sentiment_score":  ts["ticker_sentiment_score"],
                    "ticker_relevance_score":  ts["relevance_score"],
                })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["datetime"] = pd.to_datetime(df["datetime"], format="%Y%m%dT%H%M%S")
    return df


# ── 4. MAIN ───────────────────────────────────────────────────────────────────
SILVER_KEYWORDS = ["silver", "silver price", "XAG", "silver squeeze",
                   "silver futures", "COMEX silver"]

def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    # GDELT — full historical coverage
    print("Fetching GDELT news (2015-2024)...")
    gdelt_df = fetch_gdelt_full(SILVER_KEYWORDS, "2020-01-01", "2024-12-31")
    if not gdelt_df.empty:
        out = f"{RAW_DIR}/news_gdelt.csv"
        gdelt_df.to_csv(out, index=False)
        print(f"  Saved {gdelt_df.shape} -> {out}\n")

    # Alpha Vantage — comes with pre-computed sentiment (saves FinBERT time on news)
    print("Fetching Alpha Vantage news sentiment...")
    av_df = fetch_av_news(tickers="SLV,PSLV")
    if not av_df.empty:
        out = f"{RAW_DIR}/news_alphavantage.csv"
        av_df.to_csv(out, index=False)
        print(f"  Saved {av_df.shape} -> {out}\n")

    # NewsAPI — recent headlines
    print("Fetching NewsAPI headlines...")
    news_df = fetch_newsapi("silver price OR silver squeeze OR XAG",
                            from_date="2024-01-01", to_date="2024-12-31")
    if not news_df.empty:
        out = f"{RAW_DIR}/news_newsapi.csv"
        news_df.to_csv(out, index=False)
        print(f"  Saved {news_df.shape} -> {out}")


if __name__ == "__main__":
    main()
