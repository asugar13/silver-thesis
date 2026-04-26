"""
collect_reddit.py
-----------------
Scrapes posts and comments from silver-related subreddits using PRAW.

Subreddits:
  - r/WallStreetSilver  (retail squeeze community, very relevant for 2021 event)
  - r/Silverbugs        (stackers/long-term retail holders)
  - r/Gold              (precious metals overlap)
  - r/investing         (broader retail sentiment on silver)

Setup:
  1. Create a Reddit app at https://www.reddit.com/prefs/apps (select "script")
  2. Set env vars: REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT
  3. pip install praw pandas

Note on history depth:
  PRAW only returns ~1000 newest posts per subreddit. For full history back to 2015
  use Pushshift via the Arctic Shift mirror (see ALTERNATIVE below).
"""

import os
import time
import pandas as pd
import praw
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# ── CONFIG ────────────────────────────────────────────────────────────────────
SUBREDDITS = ["WallStreetSilver", "Silverbugs", "Gold", "investing"]
POST_LIMIT  = 1000          # max per subreddit in one PRAW call
RAW_DIR     = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

reddit = praw.Reddit(
    client_id     = os.getenv("REDDIT_CLIENT_ID",     "YOUR_CLIENT_ID"),
    client_secret = os.getenv("REDDIT_CLIENT_SECRET", "YOUR_CLIENT_SECRET"),
    user_agent    = os.getenv("REDDIT_USER_AGENT",    "silver-thesis-scraper/0.1"),
)


# ── 1. PRAW SCRAPER ───────────────────────────────────────────────────────────
def scrape_subreddit(name: str, limit: int) -> pd.DataFrame:
    sub = reddit.subreddit(name)
    rows = []
    for post in sub.new(limit=limit):
        rows.append({
            "id":         post.id,
            "subreddit":  name,
            "created_utc": pd.to_datetime(post.created_utc, unit="s"),
            "title":      post.title,
            "selftext":   post.selftext,
            "score":      post.score,
            "num_comments": post.num_comments,
            "upvote_ratio": post.upvote_ratio,
            "url":        post.url,
        })
    return pd.DataFrame(rows)


def scrape_all(subreddits: list, limit: int) -> pd.DataFrame:
    frames = []
    for name in subreddits:
        print(f"  Scraping r/{name}...")
        try:
            df = scrape_subreddit(name, limit)
            frames.append(df)
            print(f"    {len(df)} posts")
            time.sleep(1)   # be polite to the API
        except Exception as e:
            print(f"    Error: {e}")
    if not frames:
        print("  No posts collected — check your Reddit API credentials.")
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ── 2. ALTERNATIVE: Arctic Shift (Pushshift mirror) for full history ───────────
# Arctic Shift provides full Reddit history via HTTP.
# Endpoint: https://arctic-shift.photon-reddit.com/api/posts/search
# No auth required, but rate-limit to ~1 req/sec.

import requests

def fetch_pushshift_posts(subreddit: str, start_epoch: int, end_epoch: int,
                          size: int = 100) -> pd.DataFrame:
    """
    Pulls historical posts from Arctic Shift (Pushshift mirror).
    Use this for data going back to 2015.
    """
    url = "https://arctic-shift.photon-reddit.com/api/posts/search"
    params = {
        "subreddit": subreddit,
        "after":     start_epoch,
        "before":    end_epoch,
        "limit":     size,
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"    HTTP error: {e} — skipping window")
        return pd.DataFrame()

    data = resp.json().get("data", [])
    if not data:
        return pd.DataFrame()

    # Keep only columns that exist in the response
    available_cols = ["id", "subreddit", "created_utc", "title",
                      "selftext", "score", "num_comments", "upvote_ratio"]
    df = pd.DataFrame(data)
    cols = [c for c in available_cols if c in df.columns]
    df = df[cols]
    df["created_utc"] = pd.to_datetime(df["created_utc"], unit="s")
    return df


def fetch_full_history(subreddit: str, start: str, end: str,
                       window_days: int = 30) -> pd.DataFrame:
    """
    Pages through Arctic Shift in monthly windows to get full history.
    Empty windows at the start are fine (subreddit may not exist yet).
    Empty windows AFTER data has been found raise an error — likely an API problem.
    """
    import time as _time
    from datetime import datetime, timedelta

    start_dt    = datetime.fromisoformat(start)
    end_dt      = datetime.fromisoformat(end)
    frames      = []
    seen_data   = False   # flips to True once we receive the first non-empty window

    current = start_dt
    while current < end_dt:
        next_dt = min(current + timedelta(days=window_days), end_dt)
        print(f"  {subreddit}: {current.date()} -> {next_dt.date()}")

        # Single request per window — pagination not used because Arctic Shift's
        # cursor doesn't work reliably. Instead we use small windows (2 days)
        # to stay naturally under the 100-post cap.
        window_df = fetch_pushshift_posts(
            subreddit,
            int(current.timestamp()),
            int(next_dt.timestamp()),
            size=100,
        )

        # Warn if a 2-day window still hits 100 — means we're missing posts
        if len(window_df) == 100:
            print(f"    WARNING: hit 100-post cap on {current.date()} -> {next_dt.date()} — some posts may be missing")

        if window_df.empty:
            if seen_data:
                raise RuntimeError(
                    f"Unexpected empty window for r/{subreddit} "
                    f"({current.date()} -> {next_dt.date()}) after data was already found. "
                    f"Possible API issue or rate limit. Stopping to avoid a gap in data."
                )
            else:
                print(f"    No posts yet — subreddit may not exist for this period, skipping.")
        else:
            seen_data = True
            print(f"    {len(window_df)} posts")
            frames.append(window_df)

        current = next_dt
        _time.sleep(1)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ── 3. MAIN ───────────────────────────────────────────────────────────────────
def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    # Option A: recent posts via PRAW (commented out — needs Reddit API credentials)
    # Uncomment once you have REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env
    # print("Scraping recent posts via PRAW...")
    # recent = scrape_all(SUBREDDITS, POST_LIMIT)
    # if not recent.empty:
    #     out = f"{RAW_DIR}/reddit_recent.csv"
    #     recent.to_csv(out, index=False)
    #     print(f"  Saved {recent.shape} -> {out}\n")

    # Option B: full history via Arctic Shift (recommended for the thesis)
    print("Fetching full history via Arctic Shift...")
    TARGET_SUBS = ["WallStreetSilver", "Silverbugs"]
    hist_frames = []
    for sub in TARGET_SUBS:
        # 2-day windows: keeps each request well under 100 posts so we don't
        # rely on pagination (which Arctic Shift doesn't support reliably)
        df = fetch_full_history(sub, start="2015-01-01", end="2024-12-31", window_days=2)
        hist_frames.append(df)

    if hist_frames:
        history = pd.concat(hist_frames, ignore_index=True)
        out = f"{RAW_DIR}/reddit_history.csv"
        history.to_csv(out, index=False)
        print(f"  Saved {history.shape} -> {out}")


if __name__ == "__main__":
    main()
