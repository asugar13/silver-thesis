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

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

# ── CONFIG ────────────────────────────────────────────────────────────────────
SUBREDDITS = ["WallStreetSilver", "Silverbugs", "Gold", "investing"]
POST_LIMIT  = 1000          # max per subreddit in one PRAW call
RAW_DIR     = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")

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
                          size: int = 100, max_retries: int = 5) -> pd.DataFrame:
    """
    Pulls historical posts from Arctic Shift (Pushshift mirror).
    Use this for data going back to 2015.

    Retries on transient errors (connection reset, timeout, 5xx) with
    exponential backoff: 2s, 4s, 8s, 16s, 32s. HTTP 4xx (e.g. 400 bad request,
    404) is treated as a real failure and returns immediately.
    """
    url = "https://arctic-shift.photon-reddit.com/api/posts/search"
    params = {
        "subreddit": subreddit,
        "after":     start_epoch,
        "before":    end_epoch,
        "limit":     size,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            break   # success
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status is not None and 400 <= status < 500:
                print(f"    HTTP {status}: {e} — skipping window")
                return pd.DataFrame()
            # 5xx — transient, retry
            print(f"    HTTP {status} on attempt {attempt+1}/{max_retries}: {e}")
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError) as e:
            print(f"    network error on attempt {attempt+1}/{max_retries}: {type(e).__name__}")

        if attempt == max_retries - 1:
            print(f"    giving up after {max_retries} attempts — skipping window")
            return pd.DataFrame()
        sleep_for = 2 ** (attempt + 1)
        print(f"    retrying in {sleep_for}s...")
        time.sleep(sleep_for)

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


def fetch_window_complete(subreddit: str, start_ts: int, end_ts: int,
                          size: int = 100, min_window_seconds: int = 3600,
                          depth: int = 0) -> pd.DataFrame:
    """
    Fetch all posts in [start_ts, end_ts). When the API hits its per-call cap
    (`size` posts returned), recursively split the window in half until each
    sub-window returns fewer than `size` posts (i.e. complete coverage).

    `min_window_seconds` caps recursion: if a window of that length still hits
    the cap, we accept the partial result and emit a warning (avoids infinite
    splitting in pathologically dense periods).
    """
    import time as _time

    df = fetch_pushshift_posts(subreddit, start_ts, end_ts, size=size)
    _time.sleep(1)

    if len(df) < size:
        return df

    duration = end_ts - start_ts
    indent   = "    " + "  " * depth
    if duration <= min_window_seconds:
        print(f"{indent}WARNING: {duration}s window still hit {size}-post cap — accepting partial data")
        return df

    mid_ts = start_ts + duration // 2
    print(f"{indent}cap hit on {duration//3600}h window — splitting")
    left  = fetch_window_complete(subreddit, start_ts, mid_ts, size, min_window_seconds, depth + 1)
    right = fetch_window_complete(subreddit, mid_ts,   end_ts, size, min_window_seconds, depth + 1)
    return pd.concat([left, right], ignore_index=True)


def _checkpoint_path(subreddit: str) -> str:
    """Per-subreddit partial-progress file. Deleted once the subreddit finishes."""
    return os.path.join(RAW_DIR, f".reddit_partial_{subreddit}.csv")


def _save_checkpoint(path: str, frames: list) -> None:
    """Write accumulated frames to disk, deduped, so a later run can resume."""
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True)
    if "id" in df.columns:
        df = df.drop_duplicates(subset="id", keep="first").reset_index(drop=True)
    df.to_csv(path, index=False)


def fetch_full_history(subreddit: str, start: str, end: str,
                       window_days: int = 2) -> pd.DataFrame:
    """
    Pages through Arctic Shift to get full post history.

    Uses large 90-day windows while the subreddit has no data yet (fast skip
    over years before the subreddit existed), then switches to weekly windows
    once data appears. When a weekly window hits the API cap, the fetch is
    recursively split (see `fetch_window_complete`) so dense periods like the
    2021 WallStreetSilver squeeze are fully captured.

    **Resume**: after every successful non-empty window, accumulated rows are
    written to a per-subreddit checkpoint CSV. If the script is killed and
    re-run, the checkpoint is loaded and fetching resumes from the day after
    the last saved post. The checkpoint is deleted once the subreddit finishes.

    Empty windows at the start are fine — subreddit may not exist yet.
    Empty windows AFTER data has been found raise an error (likely an API gap),
    but the checkpoint is preserved so the next run picks up where this one stopped.
    """
    import time as _time
    from datetime import datetime, timedelta

    PRESCAN_DAYS = 90   # large window while subreddit not yet active — skips fast
    ACTIVE_DAYS  = 7    # weekly window once posts exist (split further if cap hit)

    start_dt  = datetime.fromisoformat(start)
    end_dt    = datetime.fromisoformat(end)
    frames    = []
    seen_data = False   # flips to True once we receive the first non-empty window

    # Try to resume from a checkpoint
    ckpt_path = _checkpoint_path(subreddit)
    if os.path.exists(ckpt_path):
        try:
            ckpt = pd.read_csv(ckpt_path, parse_dates=["created_utc"])
        except Exception as e:
            print(f"  [resume] checkpoint at {ckpt_path} unreadable ({e}); starting fresh")
            ckpt = pd.DataFrame()
        if not ckpt.empty:
            last_ts = pd.to_datetime(ckpt["created_utc"]).max()
            # Resume from the day after the last saved post (truncate to date).
            resume_from = datetime.combine(
                (last_ts + pd.Timedelta(days=1)).date(),
                datetime.min.time(),
            )
            if resume_from > start_dt:
                print(f"  [resume] r/{subreddit}: checkpoint has {len(ckpt)} rows "
                      f"through {last_ts.date()}; resuming from {resume_from.date()}")
                start_dt = resume_from
                frames.append(ckpt)
                seen_data = True

    current = start_dt
    while current < end_dt:
        step    = ACTIVE_DAYS if seen_data else PRESCAN_DAYS
        next_dt = min(current + timedelta(days=step), end_dt)
        print(f"  {subreddit}: {current.date()} -> {next_dt.date()}")

        if seen_data:
            # Active phase — use recursive splitting so we never truncate at the cap
            window_df = fetch_window_complete(
                subreddit,
                int(current.timestamp()),
                int(next_dt.timestamp()),
                size=100,
            )
        else:
            # Prescan phase — single call, cap not a concern (no posts yet)
            window_df = fetch_pushshift_posts(
                subreddit,
                int(current.timestamp()),
                int(next_dt.timestamp()),
                size=100,
            )
            _time.sleep(1)

        if window_df.empty:
            if seen_data:
                # Preserve checkpoint so the next run resumes from here.
                _save_checkpoint(ckpt_path, frames)
                raise RuntimeError(
                    f"Unexpected empty window for r/{subreddit} "
                    f"({current.date()} -> {next_dt.date()}) after data was already found. "
                    f"Possible API issue or rate limit. Stopping to avoid a gap in data. "
                    f"Checkpoint preserved at {ckpt_path} — re-run to resume."
                )
            else:
                print(f"    No posts yet — skipping.")
        else:
            seen_data = True
            print(f"    {len(window_df)} posts")
            frames.append(window_df)
            # Persist progress so a crash mid-run doesn't lose this window.
            _save_checkpoint(ckpt_path, frames)

        current = next_dt

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    # Recursive splits may produce duplicates at window boundaries — dedupe by post id.
    if "id" in out.columns:
        before = len(out)
        out = out.drop_duplicates(subset="id", keep="first").reset_index(drop=True)
        if before != len(out):
            print(f"  Dedup: {before} -> {len(out)} rows")

    # Subreddit fully fetched — checkpoint no longer needed.
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
        print(f"  [done] removed checkpoint {ckpt_path}")
    return out


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
    # WallStreetSilver was created in Jan 2021 — start from 2020-01-01 so the
    # prescan phase only burns ~4 windows before finding the first posts.
    # Silverbugs dates back to ~2012 so 2015-01-01 is fine.
    from config import END_DATE
    SUB_STARTS = {
        "WallStreetSilver": "2020-01-01",
        "Silverbugs":       "2015-01-01",
    }
    hist_frames = []
    for sub, sub_start in SUB_STARTS.items():
        # Per-subreddit final file — if it exists, the subreddit is fully fetched
        # already and can be skipped on re-run (resume across subreddit boundary).
        sub_final = os.path.join(RAW_DIR, f"reddit_history_{sub}.csv")
        if os.path.exists(sub_final):
            print(f"  [skip] r/{sub}: complete file exists at {sub_final}")
            hist_frames.append(pd.read_csv(sub_final))
            continue

        df = fetch_full_history(sub, start=sub_start, end=END_DATE)
        df.to_csv(sub_final, index=False)
        print(f"  Saved {df.shape} -> {sub_final}")
        hist_frames.append(df)

    if hist_frames:
        history = pd.concat(hist_frames, ignore_index=True)
        out = f"{RAW_DIR}/reddit_history.csv"
        history.to_csv(out, index=False)
        print(f"  Saved combined {history.shape} -> {out}")


if __name__ == "__main__":
    main()
