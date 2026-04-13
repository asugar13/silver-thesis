# Forecasting Silver Prices with Mixed-Frequency Models, LSTMs, and Sentiment Analysis

Master in Business Analytics and Data Science — IE School of Science and Technology (2025/2026)  
**Student:** Asier Ugarteche Perez  
**Supervisor:** Prof. Dae-Jin Lee

---

## Research Question

Do retail trader sentiment signals (Reddit, news) materially improve short-term silver price forecasts, and does deep learning outperform classical econometric models (ARIMAX, MIDAS) for this task?

---

## Project Structure

```
thesis/
├── data/
│   ├── raw/                    # Original collected data (never modified)
│   │   ├── daily_prices.csv    # Silver, gold, copper, USD index, S&P500 (yfinance)
│   │   ├── monthly_macro.csv   # CPI, Fed Funds Rate, M2, Industrial Production (FRED)
│   │   ├── reddit_history.csv  # Posts from r/WallStreetSilver, r/Silverbugs
│   │   ├── reddit_recent.csv   # Recent posts via PRAW
│   │   ├── news_gdelt.csv      # News headlines via GDELT
│   │   └── news_alphavantage.csv
│   └── processed/              # Cleaned, merged, model-ready files
│       ├── features.csv        # Full aligned feature matrix (daily)
│       ├── train/val/test.csv  # Split: 2015–2021 / 2022 / 2023–2024
│       ├── daily_sentiment.csv # FinBERT-scored daily sentiment index
│       ├── metrics_arima.csv   # ARIMA / ARIMAX evaluation results
│       ├── metrics_midas.csv   # MIDAS evaluation results
│       ├── metrics_lstm.csv    # LSTM evaluation results
│       ├── results_table.tex   # LaTeX table for thesis
│       └── model_comparison.png
│
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory data analysis
│   ├── 02_features.ipynb       # Feature engineering & frequency alignment
│   ├── 03_arima_baseline.ipynb # ARIMA / ARIMAX modelling
│   ├── 04_midas.ipynb          # MIDAS regression
│   ├── 05_lstm.ipynb           # LSTM deep learning model
│   ├── 06_sentiment.ipynb      # FinBERT sentiment scoring
│   └── 07_evaluation.ipynb     # Final comparison & Diebold-Mariano test
│
├── src/
│   ├── collect_prices.py       # Pull price & macro data
│   ├── collect_reddit.py       # Scrape Reddit posts
│   ├── collect_news.py         # Scrape news headlines
│   └── eda_utils.py            # Reusable EDA function
│
├── requirements.txt
└── README.md
```

---

## Workflow

Run in this order:

### Step 1 — Collect data

```bash
# Set API keys first
export FRED_API_KEY="your_key"          # https://fred.stlouisfed.org/docs/api/api_key.html
export REDDIT_CLIENT_ID="your_id"       # https://www.reddit.com/prefs/apps
export REDDIT_CLIENT_SECRET="your_secret"

python src/collect_prices.py    # → data/raw/daily_prices.csv, monthly_macro.csv
python src/collect_reddit.py    # → data/raw/reddit_history.csv
python src/collect_news.py      # → data/raw/news_gdelt.csv
```

### Step 2 — Run notebooks in order

```bash
jupyter notebook
```

| Notebook | Input | Output |
|---|---|---|
| `01_eda.ipynb` | raw CSVs | plots, stationarity/normality/ARCH diagnostics |
| `02_features.ipynb` | raw CSVs | `data/processed/features.csv`, train/val/test splits |
| `03_arima_baseline.ipynb` | processed splits | `metrics_arima.csv` |
| `04_midas.ipynb` | processed splits + raw macro | `metrics_midas.csv` |
| `05_lstm.ipynb` | processed splits | `metrics_lstm.csv`, `lstm_best.pt` |
| `06_sentiment.ipynb` | raw Reddit + news | `daily_sentiment.csv` |
| `07_evaluation.ipynb` | all metrics CSVs | comparison charts, DM test, LaTeX table |

> **Note:** Run `06_sentiment.ipynb` before re-running `02_features.ipynb` if you want sentiment included in the LSTM and ARIMAX feature matrices.

---

## Data Sources

| Source | Data | Frequency | Access |
|---|---|---|---|
| [yfinance](https://pypi.org/project/yfinance/) | Silver (SI=F), Gold, Copper, USD index, S&P500 | Daily | Free, no key |
| [FRED](https://fred.stlouisfed.org) | CPI, Fed Funds Rate, M2, Industrial Production, Real Rates | Monthly | Free API key |
| [Arctic Shift](https://arctic-shift.photon-reddit.com) | Reddit post history (Pushshift mirror) | Daily | Free, no key |
| [PRAW](https://praw.readthedocs.io) | Recent Reddit posts | Daily | Free Reddit app |
| [GDELT](https://www.gdeltproject.org) | Global news headlines | Daily | Free, no key |
| [Alpha Vantage](https://www.alphavantage.co) | News with pre-scored sentiment | Daily | Free API key |

---

## Models

### ARIMA / ARIMAX (`03_arima_baseline.ipynb`)
Classical time series baseline. Order (p, d, q) selected by AIC grid search. ARIMAX extends ARIMA with exogenous covariates (gold returns, USD index, copper, S&P500). Evaluated using an expanding-window 1-step-ahead rolling forecast on the test set.

### MIDAS (`04_midas.ipynb`)
Mixed Data Sampling regression — the key methodological contribution for handling mixed frequencies. Directly regresses daily silver returns on monthly macro variables (CPI, Fed Funds) without lossy forward-filling. Two variants:
- **U-MIDAS**: unrestricted lag polynomial (OLS)
- **Beta MIDAS**: smooth polynomial weighting scheme (more parsimonious)

### LSTM (`05_lstm.ipynb`)
2-layer LSTM with a 20-day sliding window. Trained with early stopping and learning rate scheduling. Uses Apple Silicon MPS acceleration if available. The same feature set as ARIMAX is used for a fair comparison.

### Sentiment Index (`06_sentiment.ipynb`)
Daily sentiment scored by [FinBERT](https://huggingface.co/ProsusAI/finbert) (`ProsusAI/finbert`) — a BERT model fine-tuned on financial text. Outputs positive/negative/neutral probabilities per post or headline. Reddit posts are weighted by upvote score before daily aggregation. The composite index enters both ARIMAX and LSTM as an additional covariate.

---

## Evaluation (`07_evaluation.ipynb`)

| Metric | Description |
|---|---|
| RMSE | Root Mean Squared Error on log-returns |
| MAE | Mean Absolute Error |
| Directional Accuracy | % of days where sign of forecast matches actual |
| Diebold-Mariano test | Statistical test of equal predictive accuracy between two models |

The silver squeeze episode (Jan–Feb 2021) is analysed separately as a natural experiment for the retail sentiment hypothesis.

---

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.11. Tested on macOS (Apple Silicon, `tf` conda environment).

---

## Key Hypothesis

> Short-term silver price fluctuations are disproportionately influenced by retail trader sentiment. Sentiment-augmented models should outperform sentiment-free counterparts, particularly during episodes of coordinated retail activity such as the 2021 silver squeeze.
