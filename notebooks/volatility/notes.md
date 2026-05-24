# Notes on the volatility chapter

## Why RF and XGBoost are included

**Short answer:** *they are negative-result robustness checks — without them, the
chapter's two key claims (linearity is sufficient; cross-asset spillover is null) are
vulnerable to the objection "maybe a more flexible model would extract the signal
you're missing."*

Using tree-based / boosted models for realised-volatility forecasting is normal
practice — Audrino, Sigrist & Ballinari (2020) (already cited for the sentiment
ablation), Christensen, Siggaard & Veliyev (2023), and Mittnik–Robinzonov–Spindler
(2015) all use RF and gradient boosting on RV targets. But the literature consensus
is that **linear HAR + extensions usually win on weekly RV** — nonlinear flexibility
doesn't pay back on a small, AR-dominated, heavy-tailed target.

So the trees aren't methodologically wrong, but their job in this chapter is **not
to win** — it's to test whether EXOG cross-asset RV spillovers add anything HAR can't
see, and whether **nonlinearity** on top of any linear spillover adds anything more.
With the `HAR+EXOG` (linear) rung now in `02_har` §5 alongside the trees in `03`/`04`,
the decomposition is clean:

| Step | Compare | What it isolates |
|---|---|---|
| 1 | `HAR` → `HAR+EXOG` (linear, OLS) | linear cross-asset spillover |
| 2 | `HAR+EXOG` (linear) → `RF/XGB (HAR+EXOG)` | nonlinearity on top of linear spillover |

### RF/XGB on HAR features alone

The literature (Bucci 2020; Christensen, Siggaard & Veliyev 2023) typically finds
that **trees on HAR features alone lose to OLS-HAR** — the three HAR lags are
overlapping moving averages, highly collinear and approximately linearly related to
next-week RV by construction (Corsi 2009), so trees overfit and lose the linear
structure HAR exploits.

For weekly silver RV under walk-forward, the data shows a different pattern. The
`RF (HAR)` rung in `03_random_forest` §6 is *competitive* with HAR-OLS on RMSE
(0.03177 vs 0.03205), and `XGB (HAR+Attention)` actually has the lowest RMSE of any
model in the chapter (0.03148). The gaps are small enough that no QLIKE-DM test
rejects equality, but the trees aren't strictly worse on HAR-only features the way
the cross-vol literature suggests. The ablations confirm this: each tree class's
best configuration is on bare HAR features (or HAR + a single sentiment regressor),
*not* HAR+EXOG.

ML-on-RV would likely earn more of its keep once the feature set contains something
OLS cannot exploit linearly — asymmetric / jump components (HAR-J / HARQ; Corsi &
Renò 2012, Patton & Sheppard 2015) — but those features require intraday data we
don't have. Without that richer feature set, the trees in this chapter function as
robustness checks rather than as superior forecasters.

## Why QLIKE, not MSE

Let $\sigma^2 = RV^2$ be the realised variance and $h = \hat{RV}^2$ the forecast
variance. The two losses used in this chapter are, as implemented in
[`src/vol_utils.py`](../../src/vol_utils.py):

**MSE** — squared error on volatility:

$$L_{\text{MSE}}(RV, \hat{RV}) = \big(RV - \hat{RV}\big)^2$$

Symmetric in the signed error $RV - \hat{RV}$: an over-forecast and an under-forecast
of equal magnitude carry the same penalty.

**QLIKE** — Patton (2011), on variance:

$$L_{\text{QLIKE}}(\sigma^2, h) = \frac{\sigma^2}{h} - \log\frac{\sigma^2}{h} - 1$$

A function of the **ratio** $r = \sigma^2 / h$ alone, minimised at $r = 1$. As
$r \to \infty$ (severe under-forecast, $h \ll \sigma^2$), $L$ grows **linearly** in
$r$; as $r \to 0$ (severe over-forecast, $h \gg \sigma^2$), $L$ grows only
**logarithmically** in $1/r$. So QLIKE penalises **underestimation of variance more
than overestimation** — the asymmetry risk management cares about.

Numerically, with $\sigma^2 = 1$:

| Forecast $h$ | Ratio $r = \sigma^2/h$ | $L_{\text{QLIKE}}$ | $L_{\text{MSE}}$ on $\sigma$ |
|---|---|---|---|
| $h = 0.5$ (under-forecast by ×2) | $2.0$ | **$0.307$** | $0.086$ |
| $h = 1.0$ (perfect)             | $1.0$ | $0$           | $0$ |
| $h = 2.0$ (over-forecast by ×2)  | $0.5$ | $0.193$       | $0.172$ |

Under-forecasting variance by a factor of 2 costs **≈ 60 % more QLIKE loss** than
over-forecasting by the same factor. MSE on volatility runs the *opposite* way in
this example (0.172 vs 0.086): because ×2 on the variance scale is a larger absolute
change in vol-scale than ÷2, MSE penalises the over-forecast more — exactly the
wrong asymmetry for risk management, where under-forecasting vol (under-sized VaR /
hedges) is the costlier real-world error.

Why this matters for the chapter:

- **Risk-management alignment.** Underestimating vol is the costlier real-world error
  (undersized VaR, insufficient hedging), so QLIKE's asymmetry matches the use-case
  loss better than MSE.
- **Statistical power.** The log-ratio form down-weights extreme upper-tail weeks that
  dominate squared error on a heavy-tailed target (excess kurtosis ≈ +58 for weekly
  silver RV — see `00_features` §2.2). This is why every "tie" under MSE-DM in
  `evaluation.ipynb` §4 is paired with a meaningful QLIKE-DM verdict.
- **Consistency.** Patton (2011) shows QLIKE is one of the small set of loss functions
  whose ranking of forecasts is *consistent* with the ranking under the true latent
  variance when the proxy is noisy — non-consistent losses (e.g. MAE on volatility,
  MSE on volatility) risk ranking forecasts contrary to the truth.

## Results

All four models now refit walk-forward (HAR refits weekly via OLS, RF and XGB refit
every 4 weeks with frozen hyperparameters on an expanding window, GARCH was already
walk-forward), so the cross-model DM tests below are no longer confounded by a
single-fit / walk-forward asymmetry. QLIKE-DM (Patton 2011), Newey-West (1987) lag-1
variance:

| Comparison | QLIKE-DM | Verdict |
|---|---|---|
| HAR-RV vs Naïve | **−2.902, p=0.004 \*\*** | HAR significantly beats the floor |
| GARCH(1,1) vs Naïve | **−2.594, p=0.009 \*\*** | GARCH significantly beats the floor |
| **RF (HAR+EXOG) vs Naïve** | **−2.370, p=0.018 \*** | **RF now significantly beats the floor** (was ns at p=0.19 under single-fit) |
| XGB (HAR+EXOG) vs Naïve | −1.466, p=0.143 (ns) | XGB ties the floor (improved from p=0.99 under single-fit) |
| `HAR+EXOG` (linear) vs HAR (`02_har` §5) | **+2.130, p=0.033 \*** | linear spillover **hurts** |

The story this tells:

1. **Bare HAR beats Naïve cleanly** (DM=−2.90, p=0.004 \*\*) — the chapter's
   headline claim holds and tightens slightly under walk-forward.
2. **Walk-forward rescues RF** from borderline to significant against Naïve
   (DM=−1.31, p=0.19 single-fit → DM=−2.37, p=0.018 \* walk-forward). The trees
   now see the 2026-spike regime in their training data rather than being frozen
   at end-2022; XGB benefits too in RMSE (−7.2%) but remains ns against Naïve.
3. **Linear cross-asset spillover still hurts.** `HAR+EXOG` (linear, OLS) is
   significantly worse than bare HAR even with weekly refit — six noisy regressors
   overfit OLS regardless of refit cadence (DM=+2.13, p=0.033 \*).
4. **Cross-asset spillover is null across all model classes.** RF and XGB ablations
   in `03_random_forest` §6 / `04_xgboost` §6 confirm bare-HAR-feature
   configurations are competitive or *better* than HAR+EXOG within each tree class
   (RF (HAR) RMSE 0.03177 vs RF (HAR+EXOG) 0.03370). So the EXOG null result is
   structural, not a linear-OLS artifact.
5. **The combined finding is a defensible negative result.** Cross-asset RV
   spillovers — linear or nonlinear — do not improve on bare HAR for weekly silver
   RV. That mirrors the returns chapter's null findings on cross-asset signals.

## Why HAR+Attention is the parsimonious sentiment rung

The three sentiment features behave as **three views of the same latent "Reddit
engagement level"**, not three independent signals:

| Pair | r |
|---|---|
| attention ↔ sent_abs (magnitude of weekly mean tone) | **−0.61** |
| attention ↔ sent_disp (within-week tone std) | **−0.51** |
| sent_abs ↔ sent_disp | +0.31 |

The negative attention↔intensity correlations are a **law-of-large-numbers effect**,
not a behavioural one: in high-attention weeks many posters with mixed views average to
a near-zero net tone (`sent_abs` falls), and each day's mean is itself a smoother
estimate so day-to-day swings shrink (`sent_disp` falls). Posts stay opinionated
individually; only the *aggregate* balances out. The Feb-2021 silver-squeeze week is
the textbook example — high attention coincided with mixed bullish-vs-skeptic coverage
and a near-zero weekly mean tone.

(Note: `sent_disp` is included in the EDA correlation analysis above but has been
*dropped* from the modelling ablation per its near-zero lead-lag correlation with the
target — see `00_features` §2.5. The model now uses `attention` and `sent_abs` only;
the three-feature correlation structure remains the right way to explain the
parsimony argument below.)

The modelling consequence (walk-forward HAR-X in `02_har` §5):

| Rung | RMSE | QLIKE-DM vs HAR | p |
|---|---|---|---|
| HAR | 0.03205 | — | — |
| HAR+Attention | 0.03191 | −2.61 | **0.009 \*\*** |
| HAR+SentIntensity (sent_abs) | 0.03194 | **−3.40** | **0.0007 \*\*\*** |
| HAR+Attention+SentIntensity | 0.03190 | −3.06 | **0.0022 \*\*** |

All three sentiment rungs significantly improve HAR under walk-forward. (Under
static single-fit, the combined rung was borderline at p=0.067 because the redundant
coefficients added per-week prediction noise; walk-forward refits adapt the
coefficients to current data and the combined rung crosses comfortably into
significance.) The RMSE differences across rungs are at the third decimal — all three
capture essentially the same latent "Reddit engagement level" signal via different
combinations of the correlated regressors.

`HAR+Attention` remains the cleanest *parsimonious* choice: one regressor, the most
interpretable channel ("how much is Reddit talking about silver this week?"), and
gives QLIKE-DM within ~0.5 of the more complex combined rung. The chapter recommends
`HAR+Attention` as the single best parsimonious extension of HAR-RV.

**Caveat — "strong sentiment lowers vol" is a confound, not a structural relationship.**
The marginal correlation between `sent_abs` and `silver_rv(t)` is negative (≈ −0.15
contemporaneous, ≈ −0.15 lagged) but it is **fully mediated by attention**: the OLS
coefficient on `sent_abs` collapses from −0.0082 (HAR+SentIntensity, *t* = −0.84) to
−0.0004 (HAR+Att+Int with attention also in the model, *t* = −0.03) — about 5% of its
standalone value, statistically indistinguishable from zero. Mechanism: high `sent_abs`
weeks are mechanically low-attention weeks (the LLN effect above), and low-attention
weeks are low-vol weeks (less retail trading). So strong sentiment intensity does not
"calm" the market — it just marks weeks where the market is less engaged.

## Where this sits in the thesis story

The volatility chapter has two distinct headline answers:

- **Yes, weekly silver RV is forecastable** — HAR and GARCH both significantly beat
  the Naïve $\text{RV}_{t-1}$ floor under QLIKE.
- **No, cross-asset RVs don't add anything** — not linearly, not nonlinearly. The
  only public-information addition that helps is Reddit sentiment (§5 of `02_har`),
  and the effect is small.

These line up with the framing in `CLAUDE.md` §10: predictable conditional variance
does not contradict the EMH, and the cross-asset / sentiment nulls are
semi-strong-form evidence on the public-information side.

## References

- Audrino, F., Sigrist, F., & Ballinari, D. (2020). The impact of sentiment and
  attention measures on stock market volatility. *International Journal of
  Forecasting*, 36(2), 334–357.
- Bucci, A. (2020). Realized volatility forecasting with neural networks. *Journal
  of Financial Econometrics*, 18(3), 502–531.
- Christensen, K., Siggaard, M., & Veliyev, B. (2023). A machine learning approach
  to volatility forecasting. *Journal of Financial Econometrics*, 21(5), 1680–1727.
- Mittnik, S., Robinzonov, N., & Spindler, M. (2015). Stock market volatility:
  Identifying major drivers and the nature of their impact. *Journal of Banking &
  Finance*, 58, 1–14.
- Corsi, F. (2009). A simple approximate long-memory model of realized volatility.
  *Journal of Financial Econometrics*, 7(2), 174–196.
- Corsi, F., & Renò, R. (2012). Discrete-time volatility forecasting with
  persistent leverage effect and the link with continuous-time volatility modeling.
  *Journal of Business & Economic Statistics*, 30(3), 368–380.
- Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility
  proxies. *Journal of Econometrics*, 160(1), 246–256.
- Patton, A. J., & Sheppard, K. (2015). Good volatility, bad volatility: Signed
  jumps and the persistence of volatility. *Review of Economics and Statistics*,
  97(3), 683–697.
