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
With the `HAR+EXOG` (linear) rung now in `01_har` §5 alongside the trees in `03`/`04`,
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

## Where this sits in the thesis story

The volatility chapter has two distinct headline answers:

- **Yes, weekly silver RV is forecastable** — HAR and GARCH both significantly beat
  the Naïve $\text{RV}_{t-1}$ floor under QLIKE.
- **No, cross-asset RVs don't add anything** — not linearly, not nonlinearly. The
  only public-information addition that helps is Reddit sentiment (§5 of `01_har`),
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
