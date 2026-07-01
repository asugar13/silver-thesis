"""
collect_trends.py
-----------------
Fetches weekly Google Trends data for silver-related search terms.

Google Trends returns a 0-100 index relative to the peak within each
requested window. For long time series (>5 years) it downgrades to monthly,
so we fetch overlapping 5-year windows at weekly granularity and stitch them
together by rescaling on the overlap period.

Reference: Da, Engelberg & Gao (2011) "In Search of Attention" — Google
Trends as a retail investor attention proxy (Journal of Finance).

Install:
    pip install pytrends
"""

import time
import pandas as pd
import numpy as np
from pytrends.request import TrendReq
import os

RAW_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw')

# Search terms — averaged into one composite attention index.
# "silver" captures broad interest; "buy silver" and "silver price" capture
# intent more directly linked to retail trading activity.
KEYWORDS = ['silver', 'buy silver', 'silver price']

from config import END_DATE

# Overlapping 5-year windows — each returns weekly data.
# Overlap of ~1 year is used to align the 0-100 scales across windows.
WINDOWS = [
    ('2015-01-01', '2019-12-31'),   # 5 years → weekly
    ('2018-01-01', '2022-12-31'),   # 5 years → weekly
    ('2022-01-01',  END_DATE),      # ~4 years → weekly
]

OVERLAP_DAYS = 365   # minimum overlap used for rescaling


def fetch_window(pytrends_obj, keywords, start, end, retries=3):
    """Fetch one window of weekly Google Trends data."""
    timeframe = f'{start} {end}'
    for attempt in range(retries):
        try:
            pytrends_obj.build_payload(keywords, timeframe=timeframe, geo='')
            df = pytrends_obj.interest_over_time()
            if df.empty:
                return pd.DataFrame()
            if 'isPartial' in df.columns:
                df = df.drop(columns=['isPartial'])
            df.index = pd.to_datetime(df.index).tz_localize(None)
            return df
        except Exception as e:
            wait = 30 * (attempt + 1)
            print(f'  Error (attempt {attempt+1}): {e} — waiting {wait}s')
            time.sleep(wait)
    return pd.DataFrame()


def stitch_windows(frames):
    """
    Rescale and concatenate overlapping Trends windows into one continuous series.

    Each frame is on a 0-100 scale relative to its own peak. We align them
    sequentially: for each pair of consecutive windows, compute the ratio of
    their means over the shared overlap period and rescale the earlier window.
    """
    if len(frames) == 1:
        return frames[0]

    result = frames[0].copy()

    for next_frame in frames[1:]:
        overlap_start = next_frame.index[0]
        overlap_end   = result.index[-1]

        if overlap_start >= overlap_end:
            # No overlap — just concatenate (can't rescale)
            print('  Warning: no overlap between windows — concatenating without rescaling')
            result = pd.concat([result, next_frame[next_frame.index > overlap_end]])
            continue

        overlap_result = result.loc[overlap_start:overlap_end]
        overlap_next   = next_frame.loc[overlap_start:overlap_end]

        # Compute per-column scaling factor from overlap means
        scale = {}
        for col in result.columns:
            if col in next_frame.columns:
                mean_result = overlap_result[col].mean()
                mean_next   = overlap_next[col].mean()
                # Scale the next window to match the result window's level
                scale[col] = mean_result / mean_next if mean_next > 0 else 1.0

        # Apply scaling to the non-overlapping part of next_frame
        new_part = next_frame[next_frame.index > overlap_end].copy()
        for col, factor in scale.items():
            if col in new_part.columns:
                new_part[col] = new_part[col] * factor

        result = pd.concat([result, new_part])

    return result.sort_index()


def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    pt = TrendReq(hl='en-US', tz=0, timeout=(10, 30))

    frames = []
    for start, end in WINDOWS:
        print(f'Fetching {start} -> {end}...')
        df = fetch_window(pt, KEYWORDS, start, end)
        if not df.empty:
            frames.append(df)
            print(f'  {len(df)} weekly rows, peak: {df.max().to_dict()}')
        else:
            print(f'  No data returned')
        time.sleep(10)   # be polite — Google rate-limits aggressively

    if not frames:
        print('No data fetched. Check your internet connection or try again later.')
        return

    print('\nStitching windows...')
    combined = stitch_windows(frames)

    # Composite index: weighted mean — 'silver' dominates (buy silver / silver price
    # are an order of magnitude lower in raw scale and contribute negligible signal)
    combined['trends_silver'] = (
        combined['silver'] * 0.6 +
        combined['silver price'] * 0.3 +
        combined['buy silver'] * 0.1
    )

    # Clip to 0-100 after rescaling (rescaling can push slightly outside)
    combined['trends_silver'] = combined['trends_silver'].clip(0, 100)

    out = os.path.join(RAW_DIR, 'google_trends.csv')
    combined.to_csv(out)
    print(f'\nSaved {combined.shape} -> {out}')
    print(f'Date range: {combined.index[0].date()} -> {combined.index[-1].date()}')
    print(f'\nPeak attention weeks (trends_silver > 80):')
    print(combined[combined['trends_silver'] > 80][['trends_silver']].to_string())


if __name__ == '__main__':
    main()
