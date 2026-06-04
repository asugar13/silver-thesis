# Notes on the weekly log-return chapter

## Why we run ARIMA

ARIMA is the **weak-form-EMH test** (§10). The hypothesis "weekly silver returns
contain no exploitable own-history signal" is exactly the null an ARIMA(p,d,q)
fitted to the return series is built to reject. If past returns carried predictable
structure — momentum, mean reversion, seasonality — an AR(p) or MA(q) term would
pick it up, and the AIC grid in `01_arima.ipynb` would settle on a non-trivial order.

The model exists to be rejected: a clean win for ARIMA(0,0,0) (or anything
indistinguishable from it) is a documented weak-form-efficiency result, not a
failure of the chapter. It's also the cheapest model in the ladder and sets the
baseline that every later notebook (VAR, MIDAS, RF, XGB, LSTM) is implicitly
competing against.

## Why we run ARIMAX despite weak lagged cross-correlations

The weekly correlation matrix at lag 1 (silver(t) vs predictor(t−1w)) tops out at
|r| ≈ 0.11; most cross-asset and sentiment regressors sit inside [−0.10, +0.07].
With ~417 train+val weeks the 5%-significance threshold for a single marginal
correlation is ≈ 2/√417 ≈ 0.098, so almost nothing clears the bar individually.

Two reasons to run ARIMAX anyway:

1. **Marginal correlation ≠ partial regression coefficient.** A predictor with
   r ≈ 0.05 against silver can still carry a non-zero ARIMAX coefficient once the
   other regressors are conditioned on — suppressor / multicollinearity effects.
   gold, VIX, sp500 and usd are themselves strongly intercorrelated, so each one's
   *unique* contribution after partialling out the others can differ from its
   marginal r. The correlation heatmap is necessary but not sufficient evidence to
   skip the test.
2. **ARIMAX is the semi-strong-form-EMH test.** Where ARIMA tests "past prices
   don't help", ARIMAX tests "past public information (cross-asset returns,
   sentiment, macro proxies) doesn't help either". A clean ARIMAX null is a
   strictly stronger efficiency claim than the ARIMA null, and that's the
   contribution — not whether ARIMAX out-forecasts the constant mean.

Same logic as ARIMA: the model exists to be rejected. The marginal cost of running
it is near-zero (it's already wired up in `01_arima.ipynb` with expanding +
rolling-100w windows and the four sentiment rungs), and the marginal value is a
documented semi-strong-form null result.

## Starting from near-white-noise — why the methodology still holds

A standing methodological worry about this chapter is that we begin from an
essentially unpredictable target with covariates that show almost no linear
signal. The framing that makes the design defensible:

- **Silver price ≈ random walk.** Differencing to weekly log-returns is the only
  thing that gives us a stationary target. That's the entire reason we model
  returns, not prices.
- **Returns are white noise but not *strict* white noise.** The *level* is
  uncorrelated — ARIMA(0,0,0) wins, marginal correlations with covariates top
  out at |r| ≈ 0.11. But the *variance* is autocorrelated (visible in the ACF of
  squared returns — volatility clustering). So returns fail strict white noise
  on the second moment, not the first.
- **That gap is what justifies running the non-linear models.** ARIMA / OLS can
  only see linear structure in the conditional mean. RF, XGB and the LSTM can in
  principle pick up magnitude-dependent or interaction terms — second-moment
  structure leaking into the conditional mean. The prior is null, but a clean
  semi-strong-form-EMH claim requires actually running that test, not assuming
  it.
- **The improvement is marginal and DM-insignificant.** That isn't a failure —
  it *is* the result, and it's exactly what the EMH predicts: weekly returns
  unforecastable from past prices and public information, regardless of how
  flexible the model class is. The predictable second moment lives in the
  volatility chapter; the unpredictable first moment here is this chapter's
  contribution.

## Why direction (WDA), not magnitude, is the primary metric

Two things often get conflated. The models are **trained on magnitude** — ARIMA,
OLS, RF, XGB and the LSTM all minimise MSE/MLE, so the continuous value is what
they're trying to nail. DA and WDA are an **evaluation lens**, not a loss function.
Magnitude information enters the fit; direction is just how we score the result.

Why direction is the right lens at this horizon and on this asset:

- At the weekly horizon, RMSE is dominated by the variance of innovations (~5%/week),
  while the entire range of plausible conditional means is well inside ±0.5%. Every
  model's RMSE collapses onto roughly the unconditional std — **including the
  constant-mean baseline**. RMSE has essentially no power to discriminate models
  on this target. WDA does.
- WDA is not pure direction — it weights sign agreement by |y_t|:
  $$\text{WDA} = \frac{\sum_i |y_i| \cdot \mathbf{1}[\text{sign}(y_i)=\text{sign}(\hat y_i)]}{\sum_i |y_i|}$$
  so it credits "got the sign right on the weeks that mattered" and ignores chop.
  That's the closest thing to an economically meaningful score available on a
  near-EMH return series.

## The magnitude-of-returns question lives in the volatility chapter

Forecasting |y_t| is mathematically almost the same exercise as forecasting realised
volatility — both target the second moment of the innovation distribution, since
$E[y_t^2] \approx \text{RV}_t$ when the conditional mean is near zero. That's exactly
what `notebooks/volatility/` does, with RMSE as the primary metric (because
magnitude *is* forecastable for RV in a way it isn't for signed returns) and DCA
as the directional read.

So the two-chapter design isn't a workaround for the return chapter's null result —
it's the correct factorisation:

- **Returns chapter:** conditional *mean*, evaluated by direction-led metrics
  because the SNR for signed prediction is weak but non-zero and direction is what
  any economic interpretation hinges on.
- **Volatility chapter:** conditional *variance*, evaluated by magnitude-led
  metrics because that's where magnitude is actually forecastable.

Trying to force the return models to chase magnitude harder would just re-derive a
worse version of the volatility chapter on the wrong target.

## What role val plays in each notebook

The same `train/val/test.csv` split is used everywhere, but val isn't held out the same way in every model:

- **ARIMA / ARIMAX, RF, XGB** — val gets folded into the training sample. AIC (ARIMA) and `TimeSeriesSplit(5)` on train+val (RF/XGB) handle model selection in-sample, so val plays no distinct held-out role. Fine: AIC's complexity penalty does the job a held-out check would; using train+val just gives more stable estimates.
- **LSTM (`06` / `06b`)** — val is a genuine held-out set, used for early stopping (no clean likelihood → AIC unavailable; train loss is useless as a stopping signal) and the SEQ_LEN × HIDDEN × DROPOUT mini-grid.

So val exists in the project because LSTM needs it; the other notebooks inherit the split for cross-model test-set alignment.

## Why the MIDAS notebooks hand-roll the fit (and don't use `midasr`)

`03_midas` and `03b_midas_daily` are R notebooks that `library(midasr)` but **never call it** —
git confirms its API (`midas_r()`, `mls()`, `amweights`, …) appears in **zero commits, ever**; the
import has been vestigial since the first R MIDAS notebook. The estimation is hand-rolled base R:
`optim(L-BFGS-B)` over the Beta / exp-Almon shape params, with the linear coefficients profiled out
by closed-form OLS (`solve`).

Why hand-rolling is the right call here — i.e. why `midasr` would fight the design:

- **`mls(x, k, m)` assumes a fixed, regular high-freq-per-low-freq ratio `m`.** Neither block has
  that: monthly macro enters by real-time *publication availability* (each Friday takes the
  most-recent values observable by $F_{t-1}$, with per-series release delays), and the daily block
  (03b) uses the last K *trading* days, with holiday-variable week lengths. A custom lag-matrix
  builder is needed regardless, so `mls()` adds nothing.
- **The fit is "concentrated."** Given the shape params $\theta$, the linear coefficients are
  closed-form OLS, so only $\theta$ enters the optimizer — simpler and more stable than `midas_r()`'s
  full joint NLS, especially across the ~44 walk-forward refits (no formula to rebuild each window).
- **Multi-frequency with per-series weights** (03b: 6 daily + 4 monthly, each its own 2-param curve)
  is trivially "a list of lag matrices, weight each" in the custom engine; in `midasr` it's a
  multi-term `mls()` formula.

Upshot: the `library(midasr)` line is dead weight (it forces the package to be installed for
nothing), and "R/`midasr`" overstates the dependency — it's hand-rolled MIDAS in R. R buys nothing
here *unless* you adopt `midasr` properly (real `mls`/`midas_r` + analytical inference), which the
first two points make awkward for this real-time-availability, walk-forward design. Otherwise a
Python port (reusing `eval_utils`) consolidates the chapter to one language/env.
