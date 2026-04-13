import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from scipy.stats import shapiro, normaltest, anderson, kstest
import statsmodels.api as sm


def eda_transform(series, transform: str = None, lags: int = 40):
    """
    EDA on a pandas Series with optional transform.

    Parameters
    ----------
    series    : pd.Series
    transform : {'log', 'square', 'delta', None}
    lags      : int — ACF/PACF lags
    """
    # ── 1. Transform ──────────────────────────────────────────────────────────
    if transform == 'log':
        ts = np.log(series)
        print("Applied log transform.")
    elif transform == 'square':
        ts = series ** 2
        print("Applied square transform.")
    elif transform == 'delta':
        ts = series.diff().dropna()
        print("Applied one-lag difference (Δy).")
    else:
        ts = series.copy()
        print("No transform applied.")

    title_map = {'log': 'Log-transformed', 'square': 'Squared',
                 'delta': 'First-differenced'}
    label = title_map.get(transform, 'Original')

    # ── 2. Time series plot ───────────────────────────────────────────────────
    plt.figure(figsize=(10, 4))
    plt.plot(ts)
    plt.title(f"{label} Series")
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ── 3. ACF & PACF ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    plot_acf(ts, ax=axes[0], lags=lags)
    axes[0].set_title("ACF")
    plot_pacf(ts, ax=axes[1], lags=lags, method='ywm')
    axes[1].set_title("PACF")
    plt.tight_layout()
    plt.show()

    # ── 4. Histogram + KDE + Normal PDF ───────────────────────────────────────
    mu, sigma = np.mean(ts), np.std(ts)
    x = np.linspace(mu - 5*sigma, mu + 5*sigma, 1000)
    pdf = st.norm.pdf(x, mu, sigma)

    plt.figure(figsize=(8, 4))
    plt.plot(x, pdf, lw=2, color="g", label="Normal PDF")
    plt.hist(ts, density=True, range=(mu - 3*sigma, mu + 3*sigma),
             color="r", alpha=0.4, label="Histogram")
    sns.kdeplot(ts, linewidth=2, color="blue", label="KDE")
    plt.xlim(mu - 5*sigma, mu + 5*sigma)
    plt.title(f"Distribution: {label}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ── 5. QQ-plot ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 5))
    (osm, osr), (slope, intercept, r) = st.probplot(ts, dist="norm")
    ax.scatter(osm, osr, s=10, alpha=0.5, label="Data quantiles")
    ax.plot(osm, slope * np.array(osm) + intercept, color="r",
            lw=2, label="Normal reference line")
    ax.set_title(f"QQ-plot — {label}")
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Sample quantiles")
    ax.legend()
    plt.tight_layout()
    plt.show()
    print(f"  QQ R²: {r**2:.4f} (1.0 = perfect normal)")

    # ── 6. ADF test ───────────────────────────────────────────────────────────
    adf_stat, adf_p, _, _, adf_cv, _ = adfuller(ts)
    print("\nADF Test:")
    print(f"  Statistic: {adf_stat:.4f}")
    print(f"  p-value:   {adf_p:.4f}")
    for k, v in adf_cv.items():
        print(f"    {k}: {v:.4f}")
    print("  →", "Stationary (reject H₀)" if adf_p < 0.05 else "Non-stationary (fail to reject H₀)")

    # ── 7. Ljung-Box ──────────────────────────────────────────────────────────
    lb = acorr_ljungbox(ts, lags=[lags], return_df=True)
    pval = lb['lb_pvalue'].iloc[0]
    print(f"\nLjung-Box (lag={lags}): p-value = {pval:.4f}")
    print("  →", "Autocorrelation present" if pval < 0.05 else "No significant autocorrelation")

    # ── 8. Normality tests ────────────────────────────────────────────────────
    sw_stat, sw_p = shapiro(ts[:5000])   # Shapiro is unreliable on very large n
    print(f"\nShapiro-Wilk:         stat={sw_stat:.4f}, p={sw_p:.4f}"
          f"  → {'normal' if sw_p > 0.05 else 'NOT normal'}")

    k2_stat, k2_p = normaltest(ts)
    print(f"D'Agostino K²:        stat={k2_stat:.4f}, p={k2_p:.4f}"
          f"  → {'normal' if k2_p > 0.05 else 'NOT normal'}")

    ad_result = anderson(ts, dist='norm')
    print(f"Anderson-Darling:     stat={ad_result.statistic:.4f}")
    for sl, crit in zip(ad_result.significance_level, ad_result.critical_values):
        marker = " ← reject normality" if ad_result.statistic > crit else ""
        print(f"    {sl:>5}%: {crit:.4f}{marker}")

    ts_std = (ts - np.mean(ts)) / np.std(ts, ddof=1)
    ks_stat, ks_p = kstest(ts_std, 'norm')
    print(f"Kolmogorov-Smirnov:   stat={ks_stat:.4f}, p={ks_p:.4f}"
          f"  → {'normal' if ks_p > 0.05 else 'NOT normal'}")

    # ── 9. ARCH / volatility clustering ───────────────────────────────────────
    model = sm.tsa.ARIMA(ts, order=(0, 0, 0)).fit()
    resid = model.resid.dropna()

    lm_stat, lm_p, f_stat, f_p = het_arch(resid, nlags=12)
    print(f"\nARCH LM test (nlags=12):  stat={lm_stat:.2f}, p={lm_p:.4f}")
    print("  →", "ARCH effects present — consider GARCH" if lm_p < 0.05
          else "No significant ARCH effects")

    x = ts - ts.mean()
    lm_stat2, lm_p2, _, _ = het_arch(x.dropna(), nlags=lags)
    lb_sq = acorr_ljungbox(x.dropna() ** 2, lags=[lags], return_df=True)
    lb_sq_p = lb_sq['lb_pvalue'].iloc[0]
    print(f"Ljung-Box on squared (lag={lags}): p={lb_sq_p:.4f}")
    print("  →", "Autocorrelation in variance → ARCH likely"
          if lb_sq_p < 0.05 else "No autocorrelation in variance")

    return {
        'transformed_series': ts,
        'adf': {'stat': adf_stat, 'pvalue': adf_p, 'crit': adf_cv},
        'ljung_box': lb,
        'shapiro': {'stat': sw_stat, 'pvalue': sw_p},
    }
