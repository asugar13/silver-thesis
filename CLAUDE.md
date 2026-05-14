# CLAUDE.md — Project methodology reference

Master's thesis: **weekly silver price forecasting** with classical, tree-based, and
deep-learning models, with sentiment and technical-indicator ablations.

`README.md` predates the current notebook layout — **trust this file over the README**
for what's actually wired up.

---

## 1. Directory layout (what's actually used)

```
thesis/
├── data/
│   ├── raw/                          # never modified
│   │   ├── daily_prices.csv          # silver, gold, USD, copper, S&P500, VIX, TIPS, oil
│   │   ├── monthly_macro.csv         # CPI, Fed Funds, industrial production, real rates
│   │   ├── reddit_history.csv        # via Arctic Shift / PRAW
│   │   └── news_gdelt.csv            # GDELT news headlines
│   └── processed/
│       ├── train.csv val.csv test.csv  # daily, split 2015–2021 / 2022 / 2023–YTD
│       ├── daily_sentiment.csv       # FinBERT (news) + Twitter-RoBERTa (Reddit)
│       ├── metrics_<model>_weekly.csv
│       ├── period_<model>_weekly.csv
│       └── lstm_<variant>_weekly_*.pt  # PyTorch checkpoints
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_features.ipynb
│   ├── 02c_technical_features_weekly.ipynb   # selects 4 tech indicators by RF importance
│   ├── 03_sentiment.ipynb            # FinBERT + Twitter-RoBERTa scoring
│   ├── evaluation.ipynb              # cross-model comparison + DM tests
│   ├── daily/   01–06                # daily counterparts (less curated)
│   └── weekly/  01_arima 02_var 03_midas 04_random_forest 05_xgboost 06_lstm 06b_lstm_walkforward
└── src/
    ├── collect_*.py                  # data collection scripts
    └── eval_utils.py                 # shared evaluate / period_metrics / diebold_mariano
```

`notebooks/cqf/gold_classification.ipynb` is a **separate** CQF exam submission, not
part of the thesis. Methodologically independent — do not try to align it with the
silver weekly models.

---

## 2. Data conventions (weekly notebooks)

- **Aggregation**: `df.resample('W-FRI').sum().dropna()` for returns; `.mean()` for
  sentiment; `.last()` for price levels (when computing tech indicators).
- **Target**: `silver_return` = weekly log-return, Friday-to-Friday.
- **Rebalancing assumption**: observe features at Friday close $t-1$, take position
  at Friday close $t-1$, evaluate at Friday close $t$. All exogenous features are
  lagged 1 week before entering the feature matrix — no intra-week look-ahead.
- **EXOG set** (used by RF / XGBoost / LSTM): 6 lagged cross-asset returns
  `[gold, usd, copper, sp500, vix, oil]` + (for tree models) 3 silver autocorrelation
  lags `silver_lag1/2/3`.
- **TECH set**: `[macd_line, macd_hist, bb_bandwidth, silver_vol_5w]` — selected by RF
  importance in `02c_technical_features_weekly.ipynb`. Pre-lagged by 1 week before
  joining onto the weekly frame.
- **Sentiment**: weekly mean of daily FinBERT (news) and Twitter-RoBERTa (Reddit)
  scores; lagged 1 week.

---

## 3. Shared evaluation utilities (`src/eval_utils.py`)

```python
PERIODS = {                    # sub-period robustness — shared across all weekly notebooks
    "2023 (choppy)":     ("2023","2023"),
    "2024 (bull start)": ("2024","2024"),
    "2025 (bull run)":   ("2025","2025"),
    "2026 (YTD)":        ("2026","2026"),
    "── Full test ──":   ("2023","2026"),
}

evaluate(name, y_true, y_pred)          # prints + returns RMSE/MAE/DA/WDA dict
period_metrics(actual, pred, idx, PERIODS)
diebold_mariano(actual, pred1, pred2, name1, name2)  # squared-error loss, NW(1) variance
```

**Metric definitions**

| | Formula | Notes |
|---|---|---|
| **DA** | `mean(sign(y) == sign(ŷ))` | Directional accuracy — naïve hit rate |
| **WDA** | `Σ\|y_i\| · 1[sign(y_i)=sign(ŷ_i)] / Σ\|y_i\|` | Magnitude-weighted DA; primary metric for variant selection |
| **DM** | $\bar d / \sqrt{(\gamma_0 + 2\gamma_1)/n}$ | Newey-West lag-1; negative = `pred1` better |

`best_name = argmax_{name} WDA(name)` is the convention used in every notebook to
pick the best variant for sub-period breakdown + 2026 zoom.

---

## 4. Per-notebook methodology (weekly)

All weekly notebooks load `train.csv / val.csv / test.csv`, aggregate to W-FRI,
and produce `metrics_<model>_weekly.csv` + `period_<model>_weekly.csv`.

### `01_arima.ipynb` — ARIMA / ARIMAX baseline
- AIC grid search for $(p,d,q)$ on train+val.
- Walk-forward 1-step-ahead with **two windows**: expanding and rolling-100w.
- ARIMAX extends ARIMA with the EXOG set as `exog` regressors.
- Sentiment ablations: ARIMAX, ARIMAX+Reddit, ARIMAX+News, ARIMAX+Reddit+News.
- Outputs both `period_arima_weekly.csv` and `period_arimax_weekly.csv`.

### `02_var.ipynb` — VAR
- VAR(p) with p chosen by AIC. Granger causality + impulse-response plots.
- Walk-forward with expanding window.

### `03_midas.ipynb` — **CURRENTLY DAILY, NOT WEEKLY** (path B rewrite pending)
- See §6 below.

### `04_random_forest.ipynb` and `05_xgboost.ipynb` — fully aligned pair
- `build_features()` produces `silver_lag1/2/3` + 6 exog lags (= the EXOG base).
- `tune()` does grid-search hyperparameter tuning per variant via `TimeSeriesSplit(5)`.
- `walk_forward()` retrains every 4 weeks; supports expanding (default) and rolling-100w.
- Variant ladder (Section 8): `Tech`, `EXOG` (baseline), `EXOG+Tech`, `EXOG+Reddit`,
  `EXOG+News`, `EXOG+Reddit+News`, `EXOG+Tech+Sentiment` — each evaluated with both
  windows. `Y` variant intentionally omitted: tree-based AR(3) duplicates the ARIMA
  baseline without offering anything trees can exploit.
- DM tests baseline = `EXOG expanding`.

### `06_lstm.ipynb` — single train + batch test prediction
- §4 hyperparameter mini-grid: `SEQ_LEN × HIDDEN × DROPOUT = 2×2×3` tuned **once** on
  the EXOG variant via val loss with early stopping. Best config reused across all
  variants (per-variant tuning would 8× the runtime with little gain — only input
  dim changes).
- §5 trains each variant once on train+val, predicts the full test set in one pass.
- Variant ladder includes `LSTM-Y` (silver-only) — the recurrent architecture is the
  reason this exists here but not in RF/XGB (LSTM extracts AR signal from the SEQ_LEN
  window; trees can't).
- §6 includes the `Naive (t-1 week)` baseline.
- No rolling-vs-expanding split: a 100-week rolling LSTM has only ~80 sequences after
  SEQ_LEN warmup — borderline trainable. The walk-forward fine-tune (06b) is the
  LSTM-native equivalent of "refit on recent data".

### `06b_lstm_walkforward.ipynb` — periodic fine-tune
- Two-stage: Stage 1 = single train (same as 06); Stage 2 = walk forward, fine-tuning
  every `RETRAIN_EVERY=20` test steps for `FT_EPOCHS=2` at `LR × 0.3`.
- **Hard lesson**: fine-tuning every 4 steps catastrophically degraded results (WDA
  0.59 → 0.48) because each event ran 30 unguarded epochs on ~4 new sequences =
  catastrophic forgetting. 20-step retrains with `FT_EPOCHS=2` keep the total
  unguarded fine-tune budget small relative to the early-stopped Stage 1.
- Outputs use `_walkforward` suffix to coexist with 06's outputs.

---

## 5. Variant naming convention

| Family | Pattern | Example |
|---|---|---|
| LSTM | hyphen-separated, all-caps | `LSTM-EXOG-TECH-SENTIMENT` |
| RF / XGB | plus-separated | `EXOG+Tech+Sentiment` |
| Walk-forward windows | suffix on the variant | `EXOG+Tech expanding`, `EXOG+Tech rolling (100w)` |

The DM baseline is **always** the smallest variant containing the base regressors —
`LSTM-EXOG` for LSTM, `EXOG expanding` for RF/XGB, `ARIMAX expanding` for ARIMAX.

`Y` (silver-only) appears only in LSTM. `Tech` (silver lags + 4 tech indicators, no
cross-assets) appears in all three feature-based models.

---

## 6. Subtleties + cross-model differences

### What's aligned
- Same `train/val/test.csv`, same W-FRI aggregation, same `EXOG` and `TECH` definitions
- Same `period_metrics` / `PERIODS` / `diebold_mariano` from `eval_utils`
- Same headline metrics (RMSE, MAE, DA, **WDA**) — WDA is the primary thesis metric
- Same naïve $y_{t-1}$ baseline row in results (added to LSTM in this round)
- Same 2026 zoom plot at the end of each notebook
- Same DM test baseline pattern (smallest variant containing the base)

### Genuine, justified differences

| Aspect | RF / XGB | LSTM (06 / 06b) | Reason |
|---|---|---|---|
| Walk-forward retraining cadence | every 4 weeks | none / every 20 with fine-tune | LSTM training is expensive and the sample is tiny (~330 sequences) |
| Window scheme | expanding + rolling-100w | expanding only | 100w rolling = ~80 LSTM sequences = unstable training |
| Hyperparameter search | grid per variant | grid once on EXOG, reused | LSTM cost; per-variant tuning would 8× runtime |
| `Y` (silver-only) variant | omitted | included | tree AR(3) duplicates ARIMA; recurrent AR is novel |
| Feature importance plot | yes (MDI / gain) | no | LSTM has no clean per-feature importance |

### Known broken thing (path B target)
`03_midas.ipynb` predicts **daily** silver returns from monthly macro lags. It lives
in `notebooks/weekly/` and writes `period_midas_weekly.csv`, but the target frequency
is wrong and the feature set excludes EXOG entirely. That's why MIDAS underperforms in
the cross-model comparison — it's not actually competing on the same task. **Path B
rewrite plan**: aggregate target to weekly, add EXOG cross-asset lags as ADL terms,
keep Beta/Almon-weighted monthly macro as the MIDAS-native contribution, add the same
EXOG/Tech/Sentiment variant ladder, add naïve baseline + DM tests + 2026 zoom.

---

## 7. Output file naming

```
metrics_<model>_weekly.csv               # full metrics table (used by evaluation.ipynb)
metrics_lstm_weekly_walkforward.csv      # 06b parallel output
period_<model>_weekly.csv                # PERIODS breakdown of best variant by WDA
period_arima_weekly.csv vs period_arimax_weekly.csv   # ARIMA notebook saves both
lstm_<variant_lowercase>_weekly_best.pt  # single-train checkpoints
lstm_<variant_lowercase>_weekly_wf_best.pt   # walk-forward checkpoints
```

`evaluation.ipynb` consumes the `metrics_*_weekly.csv` and `period_*_weekly.csv`
files via a `period_map` dict. **If you rename or add a model's output CSV, update
`evaluation.ipynb` too.**

---

## 8. Working-with-this-codebase conventions

- Don't write into `data/raw/` — collection scripts are the only path to those files.
- Notebooks use absolute output paths via `../../data/processed/<name>.csv`.
- The `evaluation.ipynb` notebook expects model notebooks to have been re-run so
  their CSV outputs are fresh — note this when making changes.
- LSTM runs on Apple MPS by default; falls back to CUDA / CPU. Reproducibility seed
  is set globally (`SEED=42`) but MPS introduces some non-determinism vs CPU.
