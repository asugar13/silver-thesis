# Model methodology subtleties

Notes on the non-obvious design decisions across model notebooks — things that look
the same on the surface but aren't, or that differ between models in ways that affect
how results should be interpreted.

---

## Retraining during the test set

All models fix something upfront (order / hyperparameters) and vary something at each
test step. But *what* gets re-estimated and *how often* differs significantly.

| Model | What's fixed | What's re-estimated | How often |
|---|---|---|---|
| ARIMA / ARIMAX | Order (p,d,q) from AIC grid search | Coefficients (AR, MA, exog) | Every test step |
| VAR | Lag order from BIC selection | All coefficients (every equation) | Every test step |
| Random Forest | Hyperparameters from TimeSeriesSplit grid search | Tree ensemble (full refit) | Every **4 weeks** |
| XGBoost | Same as RF | Boosted tree ensemble | Every **4 weeks** |
| LSTM | Architecture + all hyperparameters | Weights (full retrain from scratch) | Every **20 test steps** |

**ARIMA/VAR**: 100 refits for 100 test weeks. Each refit re-estimates the model
coefficients from scratch on all available history. Expensive but fully adaptive.

**RF/XGBoost**: retrain every 4 weeks because "RF with 200 trees changes negligibly
from one extra observation" (notebook comment). Practical trade-off — 25 refits
instead of 100. The 4-week cadence is a design choice, not a statistical necessity.

**LSTM**: retrains every 20 test steps. Each retrain fits fresh weights from scratch
on all available history up to that point (expanding window), using a fixed 50-epoch
budget. This aligns the LSTM with the econometric walk-forward approach.

---

## RF and XGBoost re-tune hyperparameters for each sentiment variant (ablation only)

The grid search runs **once** on the training set before the test loop. During the
walk-forward evaluation, `best_params` are frozen — the model is refitted every 4
weeks with the same hyperparameters on new data. No re-tuning there.

In the **sentiment ablation section**, RF and XGBoost call `tune()` again for each
sentiment variant — a full TimeSeriesSplit grid search on the augmented feature set.
Each variant gets its own best hyperparameters. This is the correct approach: you want
to know what the best possible model *with* sentiment can do, not what happens when you
bolt sentiment onto params that were never optimised for it.

ARIMA/ARIMAX does not re-run AIC search when sentiment is added — it uses the same
order and just adds the sentiment column as an extra regressor. This is also fine: the
(p,d,q) order captures the autocorrelation structure of silver's residuals, which
doesn't depend on what exogenous variables are included. In practice it's moot anyway
because the selected order is (0,0,0) — pure OLS — so there are no AR/MA terms to
re-tune.

---

## LSTM sentiment enters as a 20-day sequence, not a scalar

All econometric models (ARIMAX, VAR, RF, XGBoost) receive sentiment as a single
lagged scalar — the previous week's (or day's) average sentiment score.

The LSTM receives the full 20-day history of daily sentiment in its input sequence.
It can in principle learn non-linear multi-lag patterns — e.g. "three weeks of rising
sentiment followed by a pullback precedes a price drop." Whether it actually learns
this given the training set size is an open empirical question.

---

## LSTM test window is shorter than other models

The LSTM loses the first `SEQ_LEN = 20` test days to sequence warmup. It produces
480 predictions over the 500-day test set. All other models produce predictions for
the full test set (100 weekly observations, or ~500 daily).

For the period breakdown and evaluation comparisons, this means LSTM's 2023 period
has slightly fewer observations than the other models.

---

## VAR includes all variables in the system, not just silver

VAR models the joint dynamics of all variables (silver, gold, USD, copper, etc.)
simultaneously. The silver forecast is extracted from the silver equation, but the
model estimates cross-variable dynamics — how gold's past affects USD's future, etc.

This is different from ARIMAX, which only models silver as the dependent variable and
treats the others as exogenous regressors. VAR is symmetric; ARIMAX is not.

The Granger causality tests in the VAR notebook test whether lags of other variables
have statistically significant coefficients in the silver equation — this is the
formal test of the cross-asset predictability hypothesis.

---

## Why only RF and XGBoost use `build_features()`

ARIMA, VAR, and LSTM each handle temporal structure internally:

- **ARIMA**: the (p,d,q) order IS the lag structure. AR terms are lagged values of the
  series; MA terms are lagged residuals. `statsmodels` handles all of this — you pass
  the raw series and the model does the rest.
- **VAR**: same idea, extended to multiple series. The lag order (selected by BIC)
  determines how many past values enter each equation. Lags are internal.
- **LSTM**: receives a raw sequence of the last `SEQ_LEN` observations. The temporal
  structure is the sequence itself — the network learns what to do with it through
  backpropagation. No manual lag construction needed.

**RF and XGBoost have no such mechanism.** They are stateless tabular models that treat
every row as an independent i.i.d. observation. If you feed them the raw series without
lag columns, they have no way of knowing what yesterday's return was. `build_features()`
manually constructs the temporal information they cannot infer themselves:

```python
def build_features(df):
    for col in EXOG:
        X[f'{col}_lag1'] = df[col].shift(1)   # yesterday's market return
    for lag in [1, 2, 3]:
        X[f'silver_lag{lag}'] = df[TARGET].shift(lag)  # silver autocorrelation
```

This also explains why the LASSO feature selection found `silver_lag1/2/5` and `mom_5d`
as the top linear signals: these are the same autocorrelation features that `build_features()`
constructs explicitly for RF/XGBoost. ARIMA's AR terms capture the same signal implicitly.

---

## Feature selection: RF importance vs LASSO, and daily vs weekly

Two feature selection methods are used, mapped to different model classes:

| Selection method | Output CSV | Used by |
|---|---|---|
| RF importance (MDI) | `selected_features_rf_daily.csv` | Daily RF, XGBoost, LSTM |
| LASSO (|coef| > 0.005) | `selected_features_lasso_daily.csv` | Daily ARIMAX, VAR, MIDAS |
| RF importance (MDI) | `selected_features_rf_weekly.csv` | Weekly RF, XGBoost, LSTM |
| LASSO (|coef| > 0.005) | `selected_features_lasso_weekly.csv` | Weekly ARIMAX, VAR, MIDAS |

**Why RF importance for non-linear models?**  
RF importance (Mean Decrease in Impurity) is model-agnostic and captures non-linear
interactions. It's the appropriate selector for RF, XGBoost, and LSTM because those
models can exploit non-linear signal that LASSO would miss. Features above the mean
importance are kept — roughly the top half.

**Why LASSO for linear models?**  
ARIMAX, VAR, and MIDAS are linear models where multicollinearity and too many regressors
genuinely hurt. Extra correlated indicators inflate standard errors in ARIMAX, add
redundant equations to VAR, and overfit MIDAS's lag polynomial. LASSO shrinks
uninformative coefficients to exactly zero and returns a compact, orthogonalised set.
A strict threshold (|coef| > 0.005 on standardised features) filters out features that
survived only because the regularisation was just short of zeroing them.

**Why separate notebooks for daily and weekly?**  
Feature importance is frequency-dependent. A 14-day RSI computed on daily bars is a
different signal from a 14-week RSI computed on weekly bars — both the window length
and the underlying noise structure differ. Running feature selection on daily data and
then applying the results to weekly models would conflate these two regimes.

- `02b_technical_features.ipynb`: operates on daily bars → produces `*_daily.csv`
- `02c_technical_features_weekly.ipynb`: aggregates to W-FRI first, computes indicators on
  weekly bars, then runs selection → produces `*_weekly.csv`

**Daily RF importance result** (from `02b`, training on ~1755 daily obs):
- Top features: `gold_return` (65%), `silver_vol_5d` (8%), `mom_5d` (5%), `copper_return` (3%)
- `gold_return` and `copper_return` are already in `build_features()` as `*_lag1` columns;
  `silver_vol_5d` and `mom_5d` are added explicitly in daily RF/XGBoost notebooks.

**Weekly RF importance result** (from `02c`, training on ~330 weekly obs):
- Importance is near-flat across 10 selected features (top: MACD_line 9.6%, vs gold_return 65% daily).
- Flat importance is a noise signature: RF spreads splits randomly when no predictor dominates.
- MACD_line / MACD_hist / MACD_signal together take 3 of the top 10 slots — they are the same two EMAs, so this is triple-counting one signal.
- LASSO weekly: **0 features selected** — everything shrunk to zero by cross-validated regularisation.

**Conclusion**: no technical indicator or lagged return survives regularisation at weekly frequency.
Weekly RF/XGBoost `build_features()` deliberately omits technical features; weekly ARIMAX/VAR/MIDAS use
no additional regressors beyond the base exogenous variables. The null LASSO result is the finding.
