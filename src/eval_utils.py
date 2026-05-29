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
    """DA and WDA broken down by calendar sub-period."""
    df = pd.DataFrame({'actual': actual_arr, 'pred': pred_arr}, index=index).dropna()
    rows = []
    for label, (start, end) in periods.items():
        sub = df.loc[start:end]
        if len(sub) < 4:
            continue
        correct = np.sign(sub['actual']) == np.sign(sub['pred'])
        da  = correct.mean()
        wda = (np.abs(sub['actual']) * correct).sum() / np.abs(sub['actual']).sum()
        rows.append({'Period': label, 'n': len(sub), 'DA': da, 'WDA': wda})
    return pd.DataFrame(rows).set_index('Period')


def diebold_mariano(actual, pred1, pred2, name1='Model 1', name2='Model 2'):
    """DM test, squared error loss, Newey-West variance (lag=1). Negative stat = pred1 better.

    Prints the formatted result + winner inline (matches `vol_utils.vol_diebold_mariano`)
    and returns a dict with keys (model, vs, dm, p, winner)."""
    mask = ~np.isnan(pred1) & ~np.isnan(pred2) & ~np.isnan(actual)
    actual, pred1, pred2 = actual[mask], pred1[mask], pred2[mask]
    e1 = (actual - pred1) ** 2
    e2 = (actual - pred2) ** 2
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
