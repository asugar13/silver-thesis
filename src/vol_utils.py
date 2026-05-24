"""
vol_utils.py
------------
Evaluation helpers for the weekly volatility notebooks (notebooks/volatility/).

The realised-volatility target is non-negative, so the directional metrics in
`eval_utils` (DA / WDA on signed returns) do not apply. These helpers swap in
regression-style metrics (R^2) plus DCA -- Direction-of-Change Accuracy on the
*change* in log RV, i.e. did the model correctly call volatility rising vs falling.

`PERIODS` and `diebold_mariano` are reused straight from `eval_utils` -- the DM test
is loss-based (squared error) and works unchanged for an RV target.
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def dca(actual, pred, prev_actual):
    """Direction-of-change accuracy on log-RV change vs the previously observed RV.

    a = sign(log RV_t      - log RV_{t-1})   -- did volatility actually rise or fall
    p = sign(log RV_hat_t  - log RV_{t-1})   -- did the model predict a rise or fall

    Returns the fraction of weeks where the two signs agree.
    """
    a = np.log(actual) - np.log(prev_actual)
    p = np.log(pred)   - np.log(prev_actual)
    ok = np.isfinite(a) & np.isfinite(p)
    if ok.sum() == 0:
        return np.nan
    return float(np.mean(np.sign(a[ok]) == np.sign(p[ok])))


def vol_evaluate(name, actual, pred, prev_actual):
    """Print and return RMSE / MAE / R^2 / DCA for one volatility model."""
    actual      = np.asarray(actual, dtype=float)
    pred        = np.asarray(pred, dtype=float)
    prev_actual = np.asarray(prev_actual, dtype=float)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae  = mean_absolute_error(actual, pred)
    r2   = r2_score(actual, pred)
    d    = dca(actual, pred, prev_actual)
    print(f'{name:30s}  RMSE={rmse:.5f}  MAE={mae:.5f}  R2={r2:+.3f}  DCA={d:.3f}')
    return dict(model=name, rmse=rmse, mae=mae, r2=r2, dca=d)


def vol_period_metrics(actual, pred, prev_actual, index, periods):
    """RMSE, MAE and DCA broken down by calendar sub-period.

    Volatility analogue of `eval_utils.period_metrics` -- same `PERIODS` dict, but
    RMSE/MAE/DCA in place of DA/WDA.
    """
    df = pd.DataFrame({'actual': np.asarray(actual, dtype=float),
                       'pred':   np.asarray(pred, dtype=float),
                       'prev':   np.asarray(prev_actual, dtype=float)},
                      index=index).dropna()
    rows = []
    for label, (start, end) in periods.items():
        sub = df.loc[start:end]
        if len(sub) < 4:
            continue
        rmse = np.sqrt(mean_squared_error(sub['actual'], sub['pred']))
        mae  = mean_absolute_error(sub['actual'], sub['pred'])
        d    = dca(sub['actual'].values, sub['pred'].values, sub['prev'].values)
        rows.append({'Period': label, 'n': len(sub), 'RMSE': rmse, 'MAE': mae, 'DCA': d})
    return pd.DataFrame(rows).set_index('Period')


def _dm_loss(actual, pred, loss):
    """Per-observation loss for the volatility DM test."""
    actual = np.asarray(actual, dtype=float)
    pred   = np.asarray(pred, dtype=float)
    if loss == 'mse':
        return (actual - pred) ** 2
    if loss == 'mae':
        return np.abs(actual - pred)
    if loss == 'qlike':
        # QLIKE on variance: realised-variance proxy = RV^2, forecast variance = pred^2.
        # L = sigma2/h - log(sigma2/h) - 1  (Patton 2011); lower is better.
        r = (actual ** 2) / (pred ** 2)
        return r - np.log(r) - 1.0
    raise ValueError(f"unknown loss '{loss}' -- use 'qlike', 'mse' or 'mae'")


def vol_diebold_mariano(actual, pred1, pred2, name1='Model 1', name2='Model 2', loss='qlike'):
    """Diebold-Mariano test for volatility forecasts. Newey-West lag-1 variance.

    Unlike `eval_utils.diebold_mariano` (hard-wired to squared error) the loss is
    selectable, because squared error has very low power on a heavy-tailed RV target
    -- a few extreme weeks dominate the loss differential and inflate its variance.

    loss='qlike' (default) -- QLIKE on variance, the literature-standard volatility
        loss (Patton 2011): proxy-robust and far less sensitive to extreme weeks.
    loss='mse'   -- squared error (matches `eval_utils.diebold_mariano`).
    loss='mae'   -- absolute error.

    Negative DM stat = pred1 has lower loss (is more accurate). The winner (name1 /
    name2 / 'tie' at the 0.05 level) is printed in the result line and also returned
    in the result dict under the `winner` key.
    """
    actual = np.asarray(actual, dtype=float)
    pred1  = np.asarray(pred1, dtype=float)
    pred2  = np.asarray(pred2, dtype=float)
    mask   = np.isfinite(actual) & np.isfinite(pred1) & np.isfinite(pred2)
    actual, pred1, pred2 = actual[mask], pred1[mask], pred2[mask]
    d = _dm_loss(actual, pred1, loss) - _dm_loss(actual, pred2, loss)
    n = len(d)
    gamma0 = np.var(d, ddof=1)
    gamma1 = np.cov(d[:-1], d[1:])[0, 1] if n > 1 else 0.0
    var_d  = (gamma0 + 2 * gamma1) / n
    if var_d <= 0:
        print(f'{name1} vs {name2}: variance non-positive, skipping')
        return None
    dm_stat = float(np.mean(d) / np.sqrt(var_d))
    p_val   = 2 * (1 - scipy_stats.norm.cdf(abs(dm_stat)))
    sig     = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else '(ns)'
    if p_val < 0.05:
        winner = name1 if dm_stat < 0 else name2
    else:
        winner = 'tie'
    print(f'{name1:<28} vs {name2:<12}  [{loss:5s}]  DM={dm_stat:+.3f}  '
          f'p={p_val:.3f}  {sig:4s}  -> winner: {winner}')
    return dict(model=name1, vs=name2, loss=loss, dm=dm_stat, p=p_val, winner=winner)
