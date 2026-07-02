# Mean-Variance Dynamics and Market Efficiency in Silver: A Multi-Model Forecasting Framework

Master in Business Analytics and Data Science — IE School of Science and Technology (2025/2026)
**Student:** Asier Ugarteche Perez
**Supervisor:** Prof. Dae-Jin Lee

## Research question

Are weekly silver **returns** forecastable from past prices and public information (cross-asset
returns, macro, positioning, news/Reddit sentiment) — i.e. do weak- and semi-strong-form market
efficiency hold — and is the conditional **variance** (realised volatility) predictable even
where the mean is not?

Two parallel chapters answer this: a **returns** chapter (ARIMA/ARIMAX, VAR, MIDAS, Random
Forest, XGBoost, LSTM — each with sentiment and feature ablations, tested against a
random-walk-with-drift floor) and a **volatility** chapter (GARCH, HAR-RV, RF, XGBoost on weekly
realised volatility, capped by a VaR backtest for economic significance).

## Project structure

```
thesis/
├── data/
│   ├── raw/                     # collected data — written only by src/collection/ scripts
│   └── processed/               # splits, feature frames, model outputs (metrics/preds/period CSVs)
├── notebooks/
│   ├── preparation/             # shared upstream of both chapters
│   │   ├── 01_eda.ipynb         # EDA + white-noise/MDS diagnostics of the return series
│   │   ├── 02_features.ipynb    # daily train/val/test splits + weekly feature frame + groups
│   │   └── 03_sentiment.ipynb   # FinBERT (news) + Twitter-RoBERTa (Reddit) scoring
│   ├── exploratory/
│   │   └── technical_features_weekly.ipynb   # feature-selection study (RF importance / LASSO)
│   ├── returns/                 # returns chapter
│   │   ├── models/              # 01_arima 02_var 03_midas 04_random_forest 05_xgboost 06_lstm
│   │   │   └── daily_inputs/    # midas_daily, lstm_daily — daily inputs, weekly target
│   │   ├── regime_efficiency.ipynb   # adaptive-markets capstone
│   │   ├── evaluation.ipynb     # cross-model comparison + Diebold-Mariano battery
│   │   └── notes.md
│   └── volatility/              # volatility chapter
│       ├── 00_features.ipynb    # weekly realised-volatility target + HAR/EXOG/sentiment features
│       ├── models/              # 01_garch 02_har 03_random_forest 04_xgboost
│       ├── var_backtest.ipynb   # VaR backtest capstone (Kupiec / Christoffersen / pinball / ES)
│       ├── evaluation.ipynb
│       └── notes.md
├── src/
│   ├── collection/              # data-collection scripts (collect_*.py) + config.py
│   ├── eval_utils.py            # shared metrics + Diebold-Mariano / Pesaran-Timmermann / OOS R²
│   ├── vol_utils.py             # volatility metrics + QLIKE Diebold-Mariano
│   └── eda_utils.py
├── docs/                        # proposal and supplementary PDFs
├── images/                      # figures consumed by thesis.tex
├── thesis.tex                   # the thesis document (latexmk)
└── CLAUDE.md                    # detailed methodology reference (per-notebook conventions)
```

## Workflow

**1. Collect data** (optional — `data/raw/` ships with the repo archive):

```bash
export FRED_API_KEY=...          # prices/macro/PMI; Reddit needs REDDIT_CLIENT_ID/SECRET
python src/collection/collect_prices.py
python src/collection/collect_reddit.py   # etc. — one script per source
```

**2. Preparation** (order matters): `03_sentiment` → `02_features` (the feature frame folds
sentiment in); `01_eda` any time after the splits exist.

**3. Returns chapter**: run `returns/models/*` in any order, then `regime_efficiency`, then
`returns/evaluation.ipynb` (reads every model's metrics/preds CSVs).

**4. Volatility chapter**: `00_features` → `models/*` (any order) → `var_backtest` →
`volatility/evaluation.ipynb`.

Every notebook reads/writes `data/processed/` via relative paths, so run them from their own
directory (as Jupyter does by default).

## Evaluation

- **Returns** — RMSE/MAE + directional accuracy (DA/WDA); the efficiency verdict is the
  Diebold-Mariano test against a random-walk-with-drift floor (Campbell-Thompson OOS R² for
  effect size, Pesaran-Timmermann for direction, ex-2025 robustness).
- **Volatility** — RMSE/MAE/R² + direction-of-change accuracy vs a naïve RV(t−1) floor, with
  QLIKE-loss Diebold-Mariano as the primary test; economic significance via the VaR backtest.

## Data sources

| Source | Data | Access |
|---|---|---|
| yfinance | silver, gold, copper, USD index, S&P500, VIX, oil (daily) | free |
| FRED | daily real rates; monthly CPI, Fed Funds, M2, industrial production; China PMI proxy | free API key |
| CFTC | Commitments of Traders (silver, managed money) | free |
| Arctic Shift / PRAW | Reddit post history (r/WallStreetSilver, r/Silverbugs) | free |
| GDELT + GDELT-GKG | news headlines / URL metadata | free |
| NewsAPI.ai (Event Registry) | full-body EN silver news, 2015+ | paid key |
| Google Trends | "silver" search interest | free |

## Installation

```bash
pip install -r requirements.txt
```

Python 3.11. Tested on macOS Apple Silicon (`tf` conda environment); the LSTMs use MPS when
available.
