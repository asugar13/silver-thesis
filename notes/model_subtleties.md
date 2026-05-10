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
| LSTM | Architecture + all hyperparameters | Nothing — weights frozen after training | Never |

**ARIMA/VAR**: 100 refits for 100 test weeks. Each refit re-estimates the model
coefficients from scratch on all available history. Expensive but fully adaptive.

**RF/XGBoost**: retrain every 4 weeks because "RF with 200 trees changes negligibly
from one extra observation" (notebook comment). Practical trade-off — 25 refits
instead of 100. The 4-week cadence is a design choice, not a statistical necessity.

**LSTM**: trains once on 2015–2021, then fires predictions across 2023–2024 with
frozen weights. Standard in deep learning (retraining is expensive), but inconsistent
with the econometric approach used everywhere else. Worth noting as a limitation.

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
