# CLAUDE.md — Project methodology reference

Master's thesis: **weekly silver price forecasting** with classical, tree-based, and
deep-learning models, with sentiment and technical-indicator ablations. A parallel
**volatility-forecasting** chapter (`notebooks/volatility/`) targets weekly realised
volatility instead of returns — see §9.

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
│       ├── lstm_<variant>_weekly_*.pt  # PyTorch checkpoints
│       ├── volatility_weekly.csv     # shared volatility feature frame (see §9)
│       └── {metrics,period,pred}_<model>_volatility.csv  # volatility outputs (see §9)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_features.ipynb
│   ├── 02c_technical_features_weekly.ipynb   # selects 4 tech indicators by RF importance
│   ├── 03_sentiment.ipynb            # FinBERT + Twitter-RoBERTa scoring
│   ├── evaluation.ipynb              # cross-model comparison + DM tests
│   ├── daily/   01–06                # daily counterparts (less curated)
│   ├── weekly/  01_arima 02_var 03_midas 04_random_forest 05_xgboost 06_lstm 06b_lstm_walkforward
│   └── volatility/  00_features 01_garch 02_har 03_random_forest 04_xgboost evaluation
└── src/
    ├── collect_*.py                  # data collection scripts
    ├── eval_utils.py                 # shared evaluate / period_metrics / diebold_mariano
    └── vol_utils.py                  # volatility helpers — vol_evaluate / vol_period_metrics / dca
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

---

## 9. Volatility forecasting (`notebooks/volatility/`)

A parallel chapter that asks whether **volatility** is more forecastable than
**direction**. Same `train/val/test.csv` and W-FRI calendar as the return notebooks,
but the target is **weekly realised volatility**, not the log-return:

$$\text{RV}_t = \sqrt{\sum_{i \in \text{week } t} r_i^2}$$

— daily squared returns summed per W-FRI week, then square-rooted (realised *variance*
is additive across days, volatility is not, so we sum then sqrt).

### Layout — features notebook + one notebook per model

| Notebook | Contents |
|---|---|
| `00_features.ipynb` | Load daily data, weekly RV aggregation, EDA (ACF), build HAR + EXOG + Reddit-sentiment features, split → `volatility_weekly.csv` |
| `01_garch.ipynb` | GARCH(1,1), walk-forward refit |
| `02_har.ipynb` | Naïve floor + HAR-RV (Corsi 2009) + HAR-X sentiment / cross-asset ablation |
| `03_random_forest.ipynb` | RF on HAR + EXOG + MDI importance + sentiment ablation |
| `04_xgboost.ipynb` | XGBoost on HAR + EXOG + gain importance + sentiment ablation |
| `evaluation.ipynb` | Cross-model table, per-year breakdown, 2026 zoom, DM tests, sentiment-ablation summary |

Unlike the return notebooks (which each re-aggregate `train/val/test.csv`), every
volatility model notebook loads the single `volatility_weekly.csv` built by
`00_features.ipynb` — so the RV target, feature definitions and the train/val/test
`split` column are guaranteed identical across models. Run order: `00_features` →
`01`–`04` (any order) → `evaluation`.

### Feature sets

- **HAR** — three trailing averages of past RV (Corsi 2009): `rv_w_lag1` (1w),
  `rv_m_lag1` (4w mean), `rv_q_lag1` (12w mean). All `.shift(1)`-ed — no look-ahead.
- **EXOG** — 1-week lags of the six cross-asset RVs `[gold, copper, usd, sp500, vix,
  oil]`. Used by the tree models only; HAR-RV and GARCH stay univariate.
- **SENTIMENT** — three 1-week-lagged weekly Reddit features: `reddit_attention_lag1`
  (log post volume — an *attention* proxy), `reddit_sent_abs_lag1` (|weekly-mean tone|)
  and `reddit_sent_disp_lag1` (within-week tone dispersion — *intensity* proxies). Used
  only by the sentiment ablations in `01`/`03`/`04`. **Reddit only** — GDELT news
  coverage starts late 2017 and has zero-article weeks even inside the test window, too
  sparse for a clean RV regressor; recorded as a documented data limitation.
- `volatility_weekly.csv` also carries `silver_ret` (weekly log-return, used by GARCH)
  and a `split` column (`train` / `val` / `test`).

### Metrics — `src/vol_utils.py`

DA/WDA do not apply (RV ≥ 0), so volatility has its own helpers:

```python
vol_evaluate(name, actual, pred, prev_actual)                # RMSE / MAE / R² / DCA dict
vol_period_metrics(actual, pred, prev_actual, idx, PERIODS)  # per-year RMSE + DCA
dca(actual, pred, prev_actual)                               # direction-of-change accuracy
vol_diebold_mariano(actual, p1, p2, n1, n2, loss='qlike')    # DM test, loss-selectable
```

`PERIODS` is reused straight from `eval_utils`. **DCA** = Direction-of-Change Accuracy
on $\Delta\log\text{RV}$ — did the model call vol rising vs falling. The Naïve model
has DCA ≈ 0 by construction (predicting $\text{RV}_{t-1}$ implies no change).

`vol_diebold_mariano` **replaces** `eval_utils.diebold_mariano` for this chapter. RV is
heavy-tailed enough that squared-error DM is near-powerless — a handful of extreme
weeks dominate the loss differential and inflate its variance, so a real RMSE
improvement can still fail an MSE-DM test. The loss is therefore selectable and
defaults to **QLIKE**, the proxy-robust volatility loss (Patton 2011). `evaluation.ipynb`
reports QLIKE-DM as the primary test and squared-error DM only as a reference.

### HAR-X / sentiment ablation (`02_har`, `03` / `04`)

A focused study — separate from the headline cross-model comparison — of what an
extended HAR-RV gains from (a) **cross-asset volatility spillover** or (b) **public
Reddit sentiment**. Three mechanism groups, kept apart so any effect is attributable:
**Cross-asset** (the full EXOG set: 6 cross-asset RV lags), **Attention**
(`reddit_attention_lag1`) and **Sentiment intensity** (`reddit_sent_abs_lag1`,
`reddit_sent_disp_lag1`).

- `02_har`'s HAR-X ablation runs a 5-rung OLS ladder against bare HAR: `HAR+EXOG`
  (full linear spillover — the **linear sibling** of the RF/XGB models in `03`/`04`),
  `HAR+Attention`, `HAR+SentIntensity`, `HAR+Attention+SentIntensity`. Every rung is
  DM-tested against bare HAR. (An earlier version also tested the combined sentiment
  rung against `HAR+EXOG`, but `HAR+EXOG` turned out to be significantly *worse* than
  bare HAR on this sample, so that comparison is a lower bar than vs HAR and has been
  dropped.)
- `03` / `04` add one `HAR+EXOG+Sentiment` rung; their baselines already contain all
  six EXOG lags.
- Every rung is fitted and scored on the **same sample** (weeks where Reddit features
  exist — only the 2 boundary weeks drop, leaving 174 of 175 test weeks); the
  no-sentiment baseline is re-scored on that sample so the QLIKE-DM test is
  apples-to-apples.
- The headline models never see the sentiment columns, so the cross-model comparison
  (§1–§4 of `evaluation.ipynb`) is unaffected — the ablation is purely additive.
- Reading the `HAR+EXOG` row against `RF/XGB (HAR+EXOG)` in `03`/`04` cleanly isolates
  the nonlinear gain (or loss) of the trees on top of the linear cross-asset story —
  if both lose to HAR, the feature set is dry rather than the model class limiting.

### Output file naming

```
volatility_weekly.csv                    # shared feature frame (00_features)
metrics_<model>_volatility.csv           # har / garch / rf / xgb headline metrics
period_<model>_volatility.csv            # per-year RMSE + DCA breakdown
pred_<model>_volatility.csv              # test-set predictions, consumed by evaluation
metrics_<model>_sentiment_volatility.csv # har/rf/xgb sentiment-ablation rungs + QLIKE-DM
metrics_volatility_summary.csv           # evaluation.ipynb cross-model table
period_volatility_summary.csv            # evaluation.ipynb stacked per-year table
dm_volatility_summary.csv                # evaluation.ipynb QLIKE + MSE DM stats
metrics_sentiment_volatility_summary.csv # evaluation.ipynb stacked sentiment-ablation table
```

The top-level `notebooks/evaluation.ipynb` (returns) does **not** read these — the
volatility chapter has its own `evaluation.ipynb` inside `notebooks/volatility/`.

### Differences from the return notebooks (all justified)

| Aspect | Return notebooks | Volatility notebooks |
|---|---|---|
| Target | weekly log-return | weekly realised volatility (RV ≥ 0) |
| Primary metric | WDA | RMSE (DCA as the directional read) |
| Feature source | each notebook re-aggregates the splits | one shared `volatility_weekly.csv` |
| Variant ladder | EXOG/Tech/Sentiment rungs | none for the headline models; a Reddit-sentiment ablation on HAR/RF/XGB |
| Walk-forward windows | expanding + rolling-100w | GARCH refits walk-forward; HAR/RF/XGB single-fit |
| DM baseline | smallest variant w/ base regressors | Naïve ($\text{RV}_{t-1}$) |
| DM loss function | squared error | QLIKE primary (squared error kept as reference) |

---

## 10. Thesis framing — market efficiency

The two chapters test **two nested forms** of the Efficient Market Hypothesis
(Fama 1970, 1991), and the distinction should be made explicit in the writeup rather
than lumping everything under "weak form":

- **Weak form** — prices reflect all past *price/return* information. Tested by the
  own-history models: ARIMA, VAR own-lag terms, the `silver_lag1/2/3` terms in the tree
  models, `LSTM-Y`. Finding: weekly silver returns are statistically indistinguishable
  from white noise — no own-history model beats the naïve $y_{t-1}$ baseline on WDA.
- **Semi-strong form** — prices reflect all *public* information. Tested by the
  exogenous rungs: EXOG cross-asset lags, MIDAS monthly macro, and the Reddit / News
  sentiment ablations. Finding: no public-information variant delivers a significant
  DM improvement over its base. These null results are **semi-strong-form evidence** —
  a strictly stronger claim than weak-form efficiency alone.

**Predictable volatility does not contradict the EMH.** The hypothesis constrains the
conditional *mean* of returns (and risk-adjusted expected returns), not the conditional
*variance*. Volatility clustering is not a tradable arbitrage the way mean
predictability would be, so the volatility chapter's positive results (HAR / GARCH
beating the naïve RV floor) coexist with the efficiency findings without tension. The
framing also aligns with the adaptive-markets view of Lo (2004), which explicitly
accommodates time-varying second moments alongside (locally) efficient first moments.

The thesis is therefore **one coherent story**, not "returns failed, volatility is a
consolation prize": *weekly silver returns are unforecastable from past prices and
public information — weak- and semi-strong-form efficiency hold — yet the conditional
variance is strongly predictable.*
