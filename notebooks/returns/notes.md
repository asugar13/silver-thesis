# Notes — weekly return chapter

One reference file, three parts (the old `formulas.md` and `tests.md` are merged in here):

1. **Framing & notation** — the precise language for the thesis.
2. **Design decisions** — why the notebooks are built the way they are.
3. **Metrics & tests** — what every number in a §5 evaluation or §8 significance section means.

---

## Part 1 — Framing & notation

### The objects

- Log price $p_t = \log P_t$; weekly log return $r_{t+1} = p_{t+1} - p_t$.
- Two views of one process: the level accumulates shocks; returns *are* the shocks.

### "Random-walk-like", not "random walk"

A strict random walk $p_{t+1} = p_t + \mu + \varepsilon_{t+1}$ needs IID innovations — more
than the data supports, because volatility clusters. The safe wording:

> Silver log prices are random-walk-like: the level is non-stationary and shocks
> accumulate, but the return innovations are not IID because volatility clusters.

### The null: martingale difference (MDS)

Let $F_t$ = silver's own past returns, and $G_t$ = $F_t$ plus public information
(cross-asset returns, COT, sentiment, macro, technicals).

| EMH form | Null | Tested by |
|---|---|---|
| Weak | $E[r_{t+1} \mid F_t] = \mu$ | ARIMA, the GST (`01_eda` §4), silver lags, LSTM-Y |
| Semi-strong | $E[r_{t+1} \mid G_t] = \mu$ | ARIMAX, VAR, MIDAS, RF, XGB, LSTM |

Both nulls constrain the conditional **mean only**:

- MDS ≠ IID: $E[r_{t+1} \mid F_t] = \mu$ can hold while $\operatorname{Var}(r_{t+1} \mid F_t)$
  moves — that movement *is* volatility clustering.
- Strict (IID) white noise would also forbid dependence in $r_t^2$ and $|r_t|$; weekly
  silver fails that on the second moment, not the first.

> Weekly returns are close to white noise in the mean, but not strict white noise,
> because squared returns are autocorrelated. The predictable structure sits in the
> conditional variance.

### Why nonlinear models are part of the efficiency test

Linear models only test the restricted alternative $E[r_{t+1} \mid G_t] = \mu + \beta' G_t$.
Predictability could instead be nonlinear — thresholds, interactions, regimes, sequence
effects: each condition alone may carry nothing (high sentiment ≈ $\mu$, stretched COT
≈ $\mu$) while a combination does. RF / XGBoost / LSTM test that broader alternative
$E[r_{t+1} \mid G_t] = f(G_t)$; when even they fail to beat the drift out of sample, the
MDS reading is *strengthened*, not merely repeated.

### Directional accuracy — the caveat

Calling the sign right more often than chance is **not** the same as beating the
conditional-mean benchmark under squared error. RMSE / OOS R² / DM decide the efficiency
question; DA / WDA (with Pesaran–Timmermann for significance) are the secondary
directional lens. Details in Part 3.

### Markov — don't lean on it

The Markov property concerns the full conditional distribution and is not load-bearing
here; volatility clustering doesn't prove non-Markov behaviour (augment the state and
the process is Markov again). The relevant split is conditional **mean** vs conditional
**variance** predictability.

### One-paragraph thesis summary

> Silver log prices are random-walk-like: the level is non-stationary and shocks
> accumulate over time. Weekly log returns are therefore the appropriate modelling
> target. Those returns behave approximately as a martingale difference sequence in the
> conditional mean: own-history models select a constant-mean process, nonlinear MDS
> tests fail to reject, and public-information models do not significantly beat the
> drift benchmark out of sample. However, returns are not strict IID white noise,
> because squared returns display strong dependence. The predictable structure is
> therefore concentrated in the conditional variance rather than the conditional mean,
> motivating the separate volatility-forecasting chapter.

---

## Part 2 — Design decisions

### Why ARIMA

- ARIMA is the **weak-form test**: momentum, mean reversion or seasonality in past
  returns would push the AIC grid to a non-trivial $(p,d,q)$. It selects (0,0,0).
- The model exists to be rejected — a win for ARIMA(0,0,0) is a documented weak-form
  efficiency result, not a failure of the chapter.
- It is also the cheapest model and the implicit baseline every later notebook
  (VAR, MIDAS, RF, XGB, LSTM) competes against.

### Why ARIMAX despite |r| ≤ 0.11 lagged correlations

Weekly lag-1 correlations top out at |r| ≈ 0.11, mostly inside [−0.10, +0.07]; with
~417 train+val weeks the 5% cutoff is ≈ 2/√417 ≈ 0.098, so almost nothing clears it
individually. Two reasons to run ARIMAX anyway:

1. **Marginal correlation ≠ partial coefficient.** gold / VIX / sp500 / usd are strongly
   intercorrelated, so a regressor's *unique* contribution after partialling out the
   others can differ from its marginal r (suppressor effects). The heatmap is necessary
   but not sufficient evidence to skip the test.
2. **ARIMAX is the semi-strong-form test.** A clean ARIMAX null is a strictly stronger
   efficiency claim than the ARIMA null — that null *is* the contribution, not whether
   ARIMAX out-forecasts the constant mean.

### Starting from near-white-noise — why the methodology holds

- Differencing to weekly log-returns is what yields a stationary target — the entire
  reason we model returns, not prices.
- Returns are white noise in the mean but not strict white noise: the variance is
  autocorrelated (ACF of squared returns).
- That gap is what justifies the nonlinear models: RF / XGB / LSTM could in principle
  pick up interaction or magnitude-dependent structure invisible to ARIMA / OLS. The
  prior is null, but the semi-strong claim requires running the test, not assuming it.
- The marginal, DM-insignificant improvements *are* the result — exactly what the EMH
  predicts.

### Magnitude decides; direction is the secondary lens

- Every model is **trained on magnitude** (MSE / MLE); DA / WDA are an evaluation lens,
  not a loss function.
- The efficiency verdict lives on the **magnitude axis** (RMSE + OOS R² + DM-vs-Drift):
  the EMH is a conditional-mean claim.
- Raw RMSE alone can't separate models at this horizon, though: innovations run
  ~5%/week while plausible conditional means sit inside ±0.5%, so every model's RMSE —
  the drift's included — collapses onto the unconditional std. Significance therefore
  rides on OOS R² + DM, not the RMSE ranking itself.
- **WDA** is the economically-flavoured secondary read:
  $$\text{WDA} = \frac{\sum_i |y_i| \cdot \mathbf{1}[\text{sign}(y_i)=\text{sign}(\hat y_i)]}{\sum_i |y_i|}$$
  — credit for getting the sign right *on the weeks that mattered*, ignoring chop. It is
  noisy and bull-inflated, hence descriptive only.

(An earlier version of this file argued WDA should be primary; the chapter has since
standardised on RMSE / DM-vs-Drift as load-bearing — CLAUDE.md §3a.)

### Forecasting |r_t| belongs to the volatility chapter

$E[r_t^2] \approx \text{RV}_t$ when the conditional mean ≈ 0, so pushing the return
models to chase magnitude would just re-derive a worse volatility chapter on the wrong
target. The two-chapter split is the correct factorisation:

- **Returns chapter** — conditional *mean*; direction-flavoured secondary diagnostics
  because that's what an economic reading hinges on.
- **Volatility chapter** — conditional *variance*; magnitude-led metrics because there
  magnitude actually is forecastable.

### What role `val` plays

- **ARIMA / RF / XGB** — val folds into the training sample; AIC (ARIMA) and
  `TimeSeriesSplit(5)` on train+val (trees) handle selection in-sample.
- **MIDAS** — Stage 1 picks the weight family (Beta / exp-Almon / U-MIDAS) on val WDA.
- **LSTM (`06` / `lstm_daily`)** — genuinely held out: early stopping plus the
  SEQ_LEN × HIDDEN × DROPOUT mini-grid (no clean likelihood → no AIC; train loss is
  useless as a stopping signal).
- Net: val exists because the LSTM needs it; the other notebooks inherit the split for
  cross-model test-set alignment.

### Why the MIDAS fit is hand-rolled (no `midasr`)

`03_midas_py` / `midas_daily` are Python ports: custom Beta / exp-Almon weight curves +
an NLS fit where the linear coefficients are profiled out by closed-form OLS, so only
the shape parameters enter the optimiser. Package MIDAS (R `midasr`) would fight the
design:

- `mls(x, k, m)` assumes a fixed high-per-low frequency ratio; neither block has one —
  monthly macro enters by real-time *publication availability*, and the daily block uses
  the last K=20 *trading* days (holiday-variable weeks). A custom lag-matrix builder is
  needed regardless, so `mls()` adds nothing.
- The concentrated fit is simpler and more stable across the ~44 walk-forward refits
  than a full joint NLS.
- Multi-frequency with per-series weights (6 daily + 4 monthly curves) is just "a list
  of lag matrices, weight each" in the custom engine.
- One language/env for the whole chapter, and `eval_utils` is reused directly.

### Why only the tree models build explicit lag features

- **ARIMA / VAR** encode lags internally — the $(p,d,q)$ / VAR lag order *is* the lag
  structure; the package estimates it from the raw series.
- **LSTM** receives the last SEQ_LEN observations as a sequence — the temporal
  structure is the input itself.
- **RF / XGBoost** are stateless tabular models: every row is an independent
  observation, so without explicit columns (`silver_lag1/2/3` + the lagged EXOG set)
  they cannot see the past at all. `build_features()` constructs by hand the temporal
  information the other model classes get for free.

### Model objectives — not "all OLS"

The common thread is squared-error / conditional-mean fitting, not OLS:

- **ARIMA / ARIMAX** — maximum likelihood.
- **VAR** — OLS per equation (MLE-equivalent under Gaussian errors).
- **MIDAS** — least squares: linear coefficients by OLS, weight shapes by numerical
  optimisation.
- **RF / XGBoost** — squared-error split criterion / boosting objective.
- **LSTM** — `MSELoss` under Adam: a least-squares objective on a nonlinear recurrent
  function; no closed-form coefficients.

Thesis phrasing: *apart from ARIMA/ARIMAX's explicit likelihood framework, the models
are not uniformly OLS, but they are largely trained or selected to minimise squared
forecast error; the LSTM shares that objective through MSE loss while estimating a
nonlinear recurrent function by gradient descent.*

---

## Part 3 — Metrics & tests reference

Everything reported in a weekly notebook's §5 evaluation and §8 significance sections.
Implementations live in `src/eval_utils.py`; the chapter framing is CLAUDE.md §3 / §3a /
§10. `01_arima` carries the fullest version (the golden standard); `04` / `05` / `06` /
`lstm_daily` / the MIDAS notebooks run the same battery; `evaluation.ipynb` consolidates
it cross-model.

**The one-line mental model.** The chapter tests whether weekly silver returns are
forecastable. That breaks into three axes, and each test belongs to exactly one:

| Axis | What it's about | Describe it with | Test significance with |
|---|---|---|---|
| **Magnitude** | the conditional *mean* — the EMH claim | RMSE, MAE | **OOS R²** (effect size) + **DM** (significance) |
| **Direction** | the *sign* of the return — a softer "timing" read | DA, WDA | **Pesaran–Timmermann** |
| **Nonlinear own-history** | structure linear tests can't see | — | **Generalised Spectral Test** (`01_eda` §4) |

EMH constrains the conditional **mean**, so the **magnitude axis is load-bearing**;
direction is a secondary lens; the GST closes a gap the linear magnitude tests leave open.

### 0. The two benchmark rows (read these first)

Almost every test below is *"the model vs a benchmark"*, so the benchmark choice is the
whole game.

- **`Naive (t-1)`** — predict last week's return. A weak reference for a return target
  (high RMSE); shown for context, **not** the efficiency benchmark.
- **`Drift (prevailing mean)`** — the expanding historical mean of returns =
  random-walk-with-drift = **ARIMA(0,0,0) by construction**. The *correct* EMH benchmark
  for a return target (Welch–Goyal, Campbell–Thompson): "no predictability beyond the
  long-run average."
  - It doubles as the **always-up** directional line — its sign is constant-positive, so
    its WDA is just the magnitude-weighted up-share (≈0.59 full-sample, ≈0.49 ex-2025;
    the gap is the 2025 bull).

**Why beating Drift = rejecting efficiency.** If past prices / public info carried
exploitable signal, *some* model would forecast the mean better than its long-run
average. "Beats Drift" is the operational form of "market is inefficient." Every
load-bearing test asks exactly this.

### 1. Descriptive metrics (no p-value) — `evaluate`, `period_metrics`

These rank and describe; they don't establish significance on their own.

- **RMSE / MAE** — point-forecast error. **Primary descriptive metric**, because the
  efficiency claim *is* a conditional-mean claim. Caveat: at the weekly horizon RMSE is
  dominated by the ~5%/wk innovation variance, so every model (Drift included) collapses
  onto roughly the unconditional std — RMSE alone has little power to separate models,
  which is exactly why OOS R² + DM sit on top.
- **DA** = `mean(sign(y) == sign(ŷ))` — naïve directional hit rate. Easy to read, but
  blind to the base rate: in an up-market, "always up" scores high while saying nothing.
- **WDA** = `Σ|yᵢ|·1[sign match] / Σ|yᵢ|` — DA weighted by |return|, crediting the sign
  on the weeks that moved. The **secondary** (directional) lens. `best_name` in the
  notebooks is the WDA-argmax variant — **descriptive display only**; the verdict rests
  on DM-vs-Drift, not the (noisy, bull-inflated) WDA ranking.

### 2. OOS R² — Campbell–Thompson (2008) — `oos_r2`

- **Formula.** `R²_OS = 1 − SSE(model) / SSE(Drift)` — % reduction in out-of-sample
  squared error relative to the prevailing-mean benchmark.
- **What it answers.** "By how much does the model beat the random walk?" — the **effect
  size** on the magnitude axis, and the standard return-predictability number in the
  forecasting literature (it makes the comparison a recognised quantity rather than a
  raw RMSE gap).
- **How to read.** `R²_OS > 0` ⇒ model beats Drift OOS; `≤ 0` (the usual result for
  returns) ⇒ worse than predicting the long-run mean. The notebooks' loop annotates each
  row `-> winner: model` if R² > 0 else `Drift`.
- **Effect size ≠ significance** — a small positive R²_OS can still be noise; DM checks
  that next.

### 3. Diebold–Mariano (DM) — `diebold_mariano`

- **What it answers.** "Is the accuracy difference between two models **statistically
  significant**?" The significance partner to OOS R²'s effect size.
- **H0.** Equal predictive accuracy — the two error series have the same expected loss.
- **Statistic.** With per-period loss difference `dₜ = loss(model₁) − loss(model₂)`:
  $$\text{DM} = \frac{\bar d}{\sqrt{(\gamma_0 + 2\gamma_1)/n}}$$
  compared to a standard normal. **Sign convention:** `pred1` is always **Drift**, so
  negative DM = Drift wins; a positive, significant DM would be genuine predictability.
- **Why Newey–West lag-1** (the `γ₀ + 2γ₁`). Naïve DM assumes the `dₜ` are uncorrelated
  week-to-week; they usually aren't (volatility clustering), which understates the
  variance and overstates significance. The HAC correction at lag 1 handles first-order
  serial correlation — enough for a 1-week-ahead forecast.
- **Two losses, same test:** `loss='se'` (squared error) is the **headline** — it pairs
  with the conditional-mean benchmark. `loss='ae'` (absolute error) is the robustness
  re-run: returns are heavy-tailed, so a few extreme weeks dominate squared loss and
  sap its power; |error| guards against a tail-driven artefact.
- **The framing that matters: the floor test (model vs Drift).** The **load-bearing
  efficiency test** of the chapter — "does *any* model beat the random walk?" The older
  incremental DM (variant vs the EXOG base) was dropped as redundant: cross-asset
  returns are themselves public info, so the floor already puts the base on trial.
  - In **ARIMA**, Drift = ARIMA(0,0,0) — the floor cleanly isolates the **weak form**.
  - In **RF / XGB**, models bundle own-history + public info — their floor test is the
    **semi-strong form** (strictly stronger).
- **Finding.** R²_OS ≈ 0-or-negative and never DM-significant for any public-info
  model; feature-rich / rolling variants are significantly *worse*. No rejection of
  efficiency.

### 4. Pesaran–Timmermann (PT) — `pesaran_timmermann`

- **What it answers.** "Do the **sign** calls beat chance?" — the significance test the
  directional descriptives (DA / WDA) need.
- **H0.** Predicted and actual signs are **independent**, *accounting for the
  unconditional up-rate* — the key subtlety: the base rate is netted out, so "always up
  in an up-market" does **not** pass. Two-sided.
- **How to read.** Verdicts: `skill` (significant, PT > 0), `perverse` (significant,
  PT < 0 — systematically wrong, still information), `tie` (chance). A constant-sign
  forecast (the always-up Drift) is degenerate by design and returns `n/a`.
- **Secondary lens, heavily caveated.** (a) the best-by-WDA variant is a max over
  ~12–38 ablations, so its p-value is selection-biased; (b) ARIMA's COT-positioning PT
  signal does **not replicate** in RF (1/38 ≈ chance), XGB (0/38) or LSTM (0/15). The
  paid-news title shows the same pattern one notch stronger — PT-significant in both
  weekly *linear* models but not the nonlinear ones, and magnitude-worthless (CLAUDE.md
  §3a). Net: a non-replicating, adaptive-markets footnote (Lo 2004) — **not** a finding
  that touches the magnitude verdict.

### 5. ex-2025 robustness — the full battery, re-run

- **Not a new test** — the whole battery above (metrics + OOS R² + DM se/ae + PT) re-run
  on **2023 + 2024 + 2026 only**, after all full-window tests.
- **Why.** 2025 carries the entire always-up bull edge (up-share ≈0.49 → ≈0.59), so it
  flatters every long-biased model; dropping it checks against a single-regime artefact.
- **Mechanics.** Evaluation-only — forecasts unchanged; the remaining test years are
  pooled (per-year DM is underpowered).
- **Finding.** The efficiency conclusion *strengthens* without the bull. Lives in
  `01_arima` / `04` / `05` / `06` / `lstm_daily` / the MIDAS notebooks.

### 6. Generalised Spectral Test (GST) — Hong & Lee (2003) — `01_eda` §4

- **The gap it closes.** The drift floor + Ljung-Box only rule out **linear**
  own-history structure; a series can be serially uncorrelated yet nonlinearly
  predictable in the mean.
- **H0 (MDS).** `E[rₜ | rₜ₋₁, rₜ₋₂, …] = const` — against linear *and* nonlinear
  alternatives.
- **How.** Uses the empirical characteristic function `σⱼ(v) = Cov(rₜ, e^{iv·rₜ₋ⱼ})`,
  zero at every lag and every `v` **iff** returns are an MDS; sweeping `v` probes all
  nonlinear functions of the past. p-value from a wild (Rademacher) bootstrap, robust to
  volatility clustering.
- **Where it lives.** `01_eda` §4, as a property of the *series* (own-history, no
  covariates); `01_arima` §2 back-references it to motivate the (0,0,0) selection.
- **Finding.** Fail to reject MDS while squared returns are strongly dependent — one
  coherent statement: **the mean is a martingale, the variance is not.**

### 7. How they combine into the verdict

Reading order in a significance section, and what each step buys:

1. **RMSE / MAE / DA / WDA** — describe and rank, but don't decide.
2. **OOS R² vs Drift** — *how much* does the best model beat the random walk? (≈0 or negative.)
3. **DM vs Drift, squared error** — is the gap *significant*? (No — the load-bearing result.)
4. **DM vs Drift, absolute error** — does that survive heavy tails? (Yes.)
5. **PT** — do sign calls beat chance? (Secondary; mostly no, doesn't replicate.)
6. **ex-2025** — does all of the above hold without the bull? (Yes, strengthens.)
7. **GST** (`01_eda`) — any *nonlinear* own-history signal left? (No.)

Every printing test annotates its row `-> winner: …` so the verdict is legible at a
glance. **Bottom line:** the DM-vs-Drift floor (with OOS R² as its effect size) is the
arbiter; everything else is context, robustness, or the secondary directional lens.
