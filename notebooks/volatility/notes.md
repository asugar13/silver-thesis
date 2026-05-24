# Notes on the volatility chapter

## Why RF and XGBoost are included

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

### Why not RF/XGB on HAR features alone?

A natural variant would be **RF/XGB on the HAR features only** (silver lags, no EXOG)
— the direct nonlinear analogue of HAR-OLS. The ML-volatility literature
(Bucci 2020; Christensen, Siggaard & Veliyev 2023) sometimes includes it as a
baseline, but consistently finds it loses to OLS-HAR. The reason is structural: the
three HAR features are overlapping moving averages of the same series, highly
collinear and approximately linearly related to next-week RV by construction (Corsi
2009), so trees overfit and lose the linear structure HAR exploits.

ML-on-RV only earns its keep once the feature set contains something OLS cannot
exploit linearly — asymmetric / jump components (HAR-J / HARQ; Corsi & Renò 2012,
Patton & Sheppard 2015) or cross-asset RVs, as in `03`/`04`. So the `RF (HAR-only)`
rung is skipped; the decomposition above already isolates what the trees contribute.

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

QLIKE-DM (Patton 2011), Newey-West (1987) lag-1 variance:

| Comparison | QLIKE-DM | Verdict |
|---|---|---|
| HAR-RV vs Naïve | **−2.820, p=0.005 ** | HAR beats the floor |
| GARCH(1,1) vs Naïve | **−2.594, p=0.009 ** | GARCH beats the floor |
| `HAR+EXOG` (linear) vs HAR | **+2.333, p=0.020 *** | linear spillover **hurts** |
| RF (HAR+EXOG) vs Naïve | −1.309, p=0.190 (ns) | borderline |
| XGB (HAR+EXOG) vs Naïve | −0.006, p=0.995 (ns) | ties the floor |

The story this tells:

1. **Bare HAR beats Naïve cleanly.** The chapter's headline claim holds.
2. **Linear cross-asset spillover hurts.** Six noisy cross-asset RV lags overfit
   OLS on ~400 weeks: `HAR+EXOG` (linear) is significantly *worse* than bare HAR
   under QLIKE.
3. **Nonlinearity partly rescues the trees.** RF and XGBoost on the same HAR+EXOG
   feature set don't reach HAR's significance level against Naïve, but they don't
   fall as far as the linear OLS-HAR+EXOG did — depth/leaf regularisation mitigates
   the overfitting that wrecked the linear version.
4. **The combined finding is a defensible negative result.** Cross-asset RV
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

The modelling consequence: the sentiment block has roughly **one** degree of freedom of
real variation, not three. `HAR+Attention` alone matches `HAR+SentIntensity` alone on
QLIKE-DM (DM ≈ −2.83 / −2.86, both p ≈ 0.005 **); stacking them in
`HAR+Attention+SentIntensity` adds estimation noise without information, which is why
the combined rung loses significance (DM = −1.83, p = 0.067) even though each rung
alone clears p < 0.005. `HAR+Attention` is the parsimonious single-regressor winner.

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
