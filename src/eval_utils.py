"""
eval_utils.py
-------------
Shared evaluation helpers used across all weekly model notebooks.
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.metrics import mean_squared_error, mean_absolute_error


PERIODS = {
    "2023  (choppy)":     ("2023", "2023"),
    "2024  (bull start)": ("2024", "2024"),
    "2025  (bull run)":   ("2025", "2025"),
    "2026  (YTD)":        ("2026", "2026"),
    "── Full test ──":    ("2023", "2026"),
}


def evaluate(name, y_true, y_pred):
    """Print and return RMSE / MAE / DA / WDA for one model."""
    mask = ~np.isnan(y_pred) & ~np.isnan(y_true)
    if mask.sum() == 0:
        print(f'{name:45s}  No valid predictions')
        return None
    y_t, y_p = y_true[mask], y_pred[mask]
    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    mae  = mean_absolute_error(y_t, y_p)
    da   = np.mean(np.sign(y_t) == np.sign(y_p))
    wda  = np.sum(np.abs(y_t) * (np.sign(y_t) == np.sign(y_p))) / np.sum(np.abs(y_t))
    print(f'{name:45s}  RMSE={rmse:.5f}  MAE={mae:.5f}  DA={da:.3f}  WDA={wda:.3f}')
    return {'model': name, 'rmse': rmse, 'mae': mae, 'dir_acc': da, 'wda': wda}


def period_metrics(actual_arr, pred_arr, index, periods):
    """RMSE / MAE / DA / WDA broken down by calendar sub-period."""
    df = pd.DataFrame({'actual': actual_arr, 'pred': pred_arr}, index=index).dropna()
    rows = []
    for label, (start, end) in periods.items():
        sub = df.loc[start:end]
        if len(sub) < 4:
            continue
        err = sub['actual'] - sub['pred']
        correct = np.sign(sub['actual']) == np.sign(sub['pred'])
        da  = correct.mean()
        wda = (np.abs(sub['actual']) * correct).sum() / np.abs(sub['actual']).sum()
        rows.append({'Period': label, 'n': len(sub),
                     'RMSE': np.sqrt((err ** 2).mean()), 'MAE': err.abs().mean(),
                     'DA': da, 'WDA': wda})
    return pd.DataFrame(rows).set_index('Period')


def diebold_mariano(actual, pred1, pred2, name1='Model 1', name2='Model 2', loss='se'):
    """DM test, Newey-West variance (lag=1). Negative stat = pred1 better.

    loss='se' (default) → squared-error loss — the headline test (pairs with the conditional
    mean / drift benchmark). loss='ae' → absolute-error loss — a robustness check for the
    heavy-tailed return series, where squared-error DM is low-powered (|error| is minimised by
    the conditional median, ≈0 here, so the mean-drift benchmark is a close proxy).

    Prints the formatted result + winner inline (matches `vol_utils.vol_diebold_mariano`)
    and returns a dict with keys (model, vs, dm, p, winner)."""
    mask = ~np.isnan(pred1) & ~np.isnan(pred2) & ~np.isnan(actual)
    actual, pred1, pred2 = actual[mask], pred1[mask], pred2[mask]
    if loss == 'ae':
        e1, e2 = np.abs(actual - pred1), np.abs(actual - pred2)
    elif loss == 'se':
        e1, e2 = (actual - pred1) ** 2, (actual - pred2) ** 2
    else:
        raise ValueError("loss must be 'se' (squared) or 'ae' (absolute)")
    d  = e1 - e2
    n  = len(d)
    d_bar  = np.mean(d)
    gamma0 = np.var(d, ddof=1)
    gamma1 = np.cov(d[:-1], d[1:])[0, 1] if n > 1 else 0
    var_d  = (gamma0 + 2 * gamma1) / n
    if var_d <= 0:
        print(f'{name1} vs {name2}: variance non-positive, skipping')
        return None
    dm_stat = d_bar / np.sqrt(var_d)
    p_val   = 2 * (1 - scipy_stats.norm.cdf(abs(dm_stat)))
    sig     = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else '(ns)'
    if p_val < 0.05:
        winner = name1 if dm_stat < 0 else name2
    else:
        winner = 'tie'
    print(f'{name1:<40} vs {name2:<40}  DM={dm_stat:+.3f}  '
          f'p={p_val:.3f}  {sig:4s}  -> winner: {winner}')
    return dict(model=name1, vs=name2, dm=dm_stat, p=p_val, winner=winner)


def pesaran_timmermann(actual, pred, name=None):
    """Pesaran-Timmermann (1992) test of directional / sign predictability.

    The DM test is a *magnitude* test (squared error); this is the matching *directional*
    test for the DA / WDA story. H0: predicted and actual signs are independent (no
    market-timing skill), accounting for the unconditional up-rate. A positive, significant
    PT stat = the sign calls beat chance.

    Returns a dict (n, DA, DA_indep, PT, p) — or PT/p = NaN when the test is **degenerate**:
    a constant-sign forecast (e.g. the always-up drift) carries no directional information,
    so PT cannot be computed. If `name` is given, prints a formatted line."""
    a = np.asarray(actual, dtype=float)
    p = np.asarray(pred,   dtype=float)
    m = ~(np.isnan(a) | np.isnan(p))
    a, p = a[m], p[m]
    n  = len(a)
    dy = (a > 0).astype(float)                      # 1 if actual up
    dx = (p > 0).astype(float)                      # 1 if forecast up
    hit = float(np.mean(dy == dx))                  # DA (sign hit rate)
    Py, Px = dy.mean(), dx.mean()
    Pstar  = Py * Px + (1 - Py) * (1 - Px)          # hit rate expected under independence
    var_hit   = Pstar * (1 - Pstar) / n
    var_pstar = (((2 * Py - 1) ** 2) * Px * (1 - Px) / n
                 + ((2 * Px - 1) ** 2) * Py * (1 - Py) / n
                 + 4 * Py * Px * (1 - Py) * (1 - Px) / n ** 2)
    denom = var_hit - var_pstar
    if Px in (0.0, 1.0) or Py in (0.0, 1.0) or denom <= 0:
        out = {'n': n, 'DA': hit, 'DA_indep': Pstar, 'PT': np.nan, 'p': np.nan, 'verdict': 'n/a'}
        if name is not None:
            print(f'{name:<40}  DA={hit:.3f}  DA|indep={Pstar:.3f}  '
                  f'PT degenerate (constant-sign forecast)  -> winner: n/a')
        return out
    pt   = (hit - Pstar) / np.sqrt(denom)
    pval = 2 * (1 - scipy_stats.norm.cdf(abs(pt)))  # two-sided: any sign dependence
    # verdict: significant + PT>0 = genuine timing skill; significant + PT<0 = perverse
    # (sign calls systematically wrong); otherwise no skill beyond chance.
    verdict = 'skill' if (pval < 0.05 and pt > 0) else 'perverse' if (pval < 0.05 and pt < 0) else 'tie'
    if name is not None:
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else '(ns)'
        winner = (name.strip() if verdict == 'skill'
                  else f'inverse({name.strip()})' if verdict == 'perverse' else 'tie (chance)')
        print(f'{name:<40}  DA={hit:.3f}  DA|indep={Pstar:.3f}  '
              f'PT={pt:+.3f}  p={pval:.3f}  {sig:4s}  -> winner: {winner}')
    return {'n': n, 'DA': hit, 'DA_indep': Pstar, 'PT': pt, 'p': pval, 'verdict': verdict}


def oos_r2(actual, pred, benchmark):
    """Campbell-Thompson (2008) out-of-sample R²:  R²_OS = 1 - SSE(pred) / SSE(benchmark).

    The standard return-predictability metric. `benchmark` is the prevailing-mean / drift
    forecast, so R²_OS is the % reduction in OOS squared error *relative to the random walk
    with drift* — the effect-size companion to the DM-vs-Drift significance test.
    Positive => the model beats the historical-mean benchmark OOS; negative (the usual result
    for returns) => worse than just predicting the prevailing mean."""
    a = np.asarray(actual, dtype=float)
    p = np.asarray(pred,   dtype=float)
    b = np.asarray(benchmark, dtype=float)
    m = ~(np.isnan(a) | np.isnan(p) | np.isnan(b))
    a, p, b = a[m], p[m], b[m]
    sse_model = np.sum((a - p) ** 2)
    sse_bench = np.sum((a - b) ** 2)
    return np.nan if sse_bench == 0 else 1 - sse_model / sse_bench
