# CLAUDE.md — Project methodology reference

Master's thesis — *Mean-Variance Dynamics and Market Efficiency in Silver: A Multi-Model
Forecasting Framework*. In practice: **weekly silver price forecasting** with classical,
tree-based, and deep-learning models, with sentiment and technical-indicator ablations. A
parallel **volatility-forecasting** chapter (`notebooks/volatility/`) targets weekly
realised volatility instead of returns — see §9.

---

## 1. Directory layout (what's actually used)

```
thesis/
├── data/
│   ├── raw/                          # never modified
│   │   ├── daily_prices.csv          # silver, gold, USD, copper, S&P500, VIX, TIPS, oil
│   │   ├── monthly_macro.csv         # CPI, Fed Funds, industrial production, M2
│   │   ├── monthly_china_pmi_proxy.csv # China PMI proxy (FRED CHNBSCICP02STSAM) → PMI group
│   │   ├── reddit_history.csv        # via Arctic Shift / PRAW
│   │   ├── news_gdelt.csv            # GDELT news headlines (title-only, multilingual, 2017+)
│   │   └── news_newsapi.csv          # NewsAPI.ai/Event Registry paid news (full EN bodies, 2015+) → NEWS_PAID
│   └── processed/
│       ├── train.csv val.csv test.csv  # daily, split 2015–2021 / 2022 / 2023–YTD
│       ├── features_weekly.csv       # shared weekly W-FRI feature frame (built by 02_features §8)
│       ├── feature_groups.json       # ablation group → column lists (TECH, MACRO, PMI, GKG, NEWS_PAID, BESTLAG_*)
│       ├── daily_sentiment.csv       # FinBERT news (GDELT/GKG/paid) + RoBERTa (Reddit); paid incl. Event-Registry VADER
│       ├── metrics_<model>_weekly.csv  # (LSTM: seed-mean + *_std cols; .pt checkpoints not persisted — §4/§7)
│       ├── period_<model>_weekly.csv
│       ├── volatility_weekly.csv     # shared volatility feature frame (see §9)
│       └── {metrics,period,pred}_<model>_volatility.csv  # volatility outputs (see §9)
├── notebooks/
│   ├── preparation/                  # shared upstream of BOTH chapters
│   │   ├── 01_eda.ipynb
│   │   ├── 02_features.ipynb        # daily splits + shared weekly frame (§8); macro folded in (was 02d)
│   │   └── 03_sentiment.ipynb       # FinBERT + Twitter-RoBERTa scoring
│   ├── exploratory/
│   │   └── technical_features_weekly.ipynb   # weekly feature selection (RF importance / LASSO; was 02c)
│   ├── returns/                      # returns chapter (was weekly/)
│   │   ├── models/  01_arima 02_var 03_midas 04_random_forest 05_xgboost 06_lstm   # the 6 headline weekly models
│   │   │   └── daily_inputs/  midas_daily lstm_daily   # daily-input siblings (weekly target)
│   │   ├── regime_efficiency.ipynb  # AMH/regime test of the directional signal (capstone, §4)
│   │   ├── evaluation.ipynb         # cross-model comparison + DM tests
│   │   └── notes.md
│   └── volatility/                   # volatility chapter (see §9)
│       ├── 00_features.ipynb        # chapter-specific feature frame
│       ├── models/  01_garch 02_har 03_random_forest 04_xgboost
│       ├── var_backtest.ipynb       # VaR capstone (was 05_)
│       ├── evaluation.ipynb
│       └── notes.md
├── docs/                             # proposal / supplementary PDFs
└── src/
    ├── collection/                   # collect_*.py data-collection scripts + config.py
    ├── eval_utils.py                 # shared evaluate / period_metrics / diebold_mariano
    ├── vol_utils.py                  # volatility helpers — vol_evaluate / vol_period_metrics / dca
    └── eda_utils.py
```

---

## 2. Data conventions (weekly notebooks)

- **Source of truth**: `02_features` §8 builds one shared weekly frame
  `features_weekly.csv` (W-FRI). Every weekly model notebook **reads it** instead of
  re-aggregating `train/val/test.csv`. Everything in the frame is **raw / un-lagged** —
  each model applies its own 1-week lag (trees `.shift(1)`, the LSTM window supplies it),
  `TECH` included (treated exactly like the EXOG returns). The **one exception is `MACRO`**,
  pre-lagged in the frame via publication-date availability (not a plain shift — see §9 /
  the macro cell). A `split` column carries train/val/test membership.
- **Aggregation** (applied once in `02_features` §8): `.sum()` for returns / FRED-Δ,
  `.mean()` for sentiment, `.last()` for levels (`gs_ratio_z`) / COT positioning.
- **Target**: `silver_return` = weekly log-return, Friday-to-Friday.
- **Rebalancing assumption**: observe features at Friday close $t-1$, take position
  at Friday close $t-1$, evaluate at Friday close $t$. All exogenous features are
  lagged 1 week before entering the feature matrix — no intra-week look-ahead.
- **EXOG set** (used by RF / XGBoost / LSTM): 6 lagged cross-asset returns
  `[gold, usd, copper, sp500, vix, oil]` + (for tree models) 3 silver autocorrelation
  lags `silver_lag1/2/3`.
- **TECH set**: momentum/trend indicators (MACD line/hist, RSI, 13/26/52w
  ROC, 5w momentum, price/MA, Donchian, Bollinger %B) — the earlier magnitude indicators
  (`bb_bandwidth`, `silver_vol_5w`) were dropped because they proxy the conditional
  *variance*, not the conditional *mean* a return forecast targets (they're not reused in the
  volatility chapter either, which builds its own RV/HAR features). Built **raw** in `02_features` §8, lagged 1 week by each model like EXOG (see intro). The
  live list is the `TECH` group in `feature_groups.json`; `technical_features_weekly` selects/justifies it; notebooks
  read `GROUPS['TECH']`.
- **Sentiment**: weekly mean of daily scores, lagged 1 week — **FinBERT** (financial) for news,
  **Twitter-RoBERTa** for Reddit. Three *separate* news feeds, kept as distinct columns/groups (not merged):
  - **GDELT** `news_sentiment` (group `SENT`, with Reddit) — title-only, ~86% non-EN, 2017+.
  - **GKG** `gkg_sentiment` (group `GKG`) — FinBERT on the GDELT-GKG URL slug.
  - **Paid news** (NewsAPI.ai **= Event Registry** — same service, `eventregistry.org` host; `news_newsapi.csv`,
    group `NEWS_PAID`) — reputable EN sources, full bodies, 2015+, ≥1 article/wk. Three encodings of the
    *same* articles: `news_paid_sentiment_body` (FinBERT on the **body**, chunked ≤512 tok → length-weighted
    mean; the primary), `news_paid_sentiment_title` (FinBERT on the **title**), and `news_paid_sentiment_vader`
    (Event Registry's *own* sentiment = **VADER** — a general lexicon scorer, *not* finance-tuned, run on
    the article's first ~5 sentences, −1…1). The body skews more negative than the title / VADER (FinBERT
    dilutes on long hedge-heavy text); choose the encoding in `technical_features_weekly`/EDA before featuring.
- **MACRO set**: publication-date-lagged monthly macro — `cpi`, `fed_funds`, `ind_prod`,
  `m2` (3 availability-lags each, `*_mlag1/2/3`). **m2, not real rates** — the real rate is
  covered daily by DFII10 (`real_rates_chg` in `FRED_DAILY`), so monthly real rates aren't
  collected. Pre-lagged in the frame (consumed without further shifting).
- **PMI set**: China PMI proxy (`china_pmi_proxy`, FRED CHNBSCICP02STSAM — monthly balance-of-opinion
  *level*, not differenced). **Not** the MACRO 3-lag machinery — a **single point-in-time level**:
  `_daily_macro_lags(..., n_lags=1)` keeps the most recent publication-available value (31d release lag),
  it's in `LAST_W` (weekly `.last()`, like `gs_ratio_z`/COT), **raw in the frame and lagged 1w by each
  model** (not pre-lagged). Its **own `PMI` group** = `['china_pmi_proxy']`, kept separate from MACRO.
- **BESTLAG sets** (filter feature-selection, `02_features` §8b): the *filter* sibling of
  `technical_features_weekly`'s embedded methods (RF importance / LASSO). For each exogenous predictor, the lag
  `k≥1` maximising `|corr(silver_return_t, x_{t-k})|` is chosen **on train+val only** (no
  test peek; monthly macro screened on publication-availability lags). Picks are
  pre-materialised as `*_blag{k}w` / `*_blag{k}m` columns (consumed without further
  shifting). Three nested groups: `BESTLAG_ALL` (every feature at its best lag),
  `BESTLAG_SIG90` (`p<0.10`), `BESTLAG_SIG` (`p<0.05`). p-values are uncorrected for the
  8×18 scan, so only `gs_ratio_z` is convincingly real — the full-sample correlations that
  motivated the screen turn out to be **test-period artefacts** (a semi-strong-efficiency
  robustness result), so these groups are for ablation, not a clean OOS feature set.

---

## 3. Shared evaluation utilities (`src/eval_utils.py`)

```python
PERIODS = {                    # sub-period robustness — shared across all weekly notebooks
    "2023 (choppy)":     ("2023","2023"),
    "2024 (bull start)": ("2024","2024"),
    "2025 (bull run)":   ("2025","2025"),
    "2026 (YTD)":        ("2026","2026"),
    "── Full test ──":   ("2023","2026"),
}

evaluate(name, y_true, y_pred)          # prints + returns RMSE/MAE/DA/WDA dict
period_metrics(actual, pred, idx, PERIODS)   # per-period RMSE/MAE/DA/WDA (was DA/WDA-only)
diebold_mariano(actual, pred1, pred2, name1, name2, loss='se')  # NW(1); loss='se' (squared) or 'ae' (absolute)
pesaran_timmermann(actual, pred, name=None)  # directional sign test; returns verdict skill/perverse/tie (see §3a)
oos_r2(actual, pred, benchmark)              # Campbell-Thompson OOS R² vs the drift benchmark
```

Every printing test annotates its row with `-> winner: …`: `diebold_mariano` (pairwise),
`pesaran_timmermann` (model vs chance — `tie`/`inverse(name)`/`n/a` for the degenerate drift), and
the notebook's OOS-R² loop (model if R²>0 else `Drift`).

**Metric definitions**

| | Formula | Notes |
|---|---|---|
| **RMSE / MAE** | — | **Primary** (magnitude) error metrics — the efficiency claim is a conditional-mean claim |
| **DA** | `mean(sign(y) == sign(ŷ))` | Directional accuracy — naïve hit rate |
| **WDA** | `Σ\|y_i\| · 1[sign(y_i)=sign(ŷ_i)] / Σ\|y_i\|` | Magnitude-weighted DA; **secondary** (directional) lens |
| **DM** | $\bar d / \sqrt{(\gamma_0 + 2\gamma_1)/n}$ | Newey-West lag-1; negative = `pred1` better. Loss-selectable (`se` headline, `ae` robustness) |
| **PT** | Pesaran-Timmermann (1992) | Directional significance test for DA — see §3a |

`best_name = argmax_{name} WDA(name)` still selects the variant shown in the sub-period
breakdown + 2026 zoom, but this is **descriptive** display only — the load-bearing efficiency
verdict is the **DM-vs-Drift** test (§3a), not the WDA ranking (which is noisy and bull-inflated).

### 3a. Baselines & significance testing

> **Status:** the full battery (**Drift + always-up line + OOS R² + DM floor se/ae + PT + ex-2025**)
> is in `01_arima`, `04`, `05`, `06`, `lstm_daily`, and the Python MIDAS notebooks
> (`03_midas`/`midas_daily`), which reuse `eval_utils` directly.

- **Benchmarks.** Two rows accompany every model: `Naive (t-1)` = persistence (a weak reference
  for a return target — high RMSE; *not* the efficiency benchmark) and **`Drift (prevailing mean)`**
  = expanding historical mean of returns = random-walk-with-drift = **ARIMA(0,0,0) by construction**.
  Drift is the **correct EMH benchmark** for a return target (Welch-Goyal / Campbell-Thompson), and
  it doubles as the **always-up** directional line (its sign is constant-positive, so its WDA = the
  magnitude-weighted up-share — ≈0.59 full-sample but ≈0.49 ex-2025; the rest is the 2025 bull).
- **DM framings** (the EMH arbiter is the *floor* test). The older **incremental** test (variant vs
  the EXOG/ARIMAX base) was **dropped** from `01_arima`/`04`/`05` as redundant — cross-asset returns
  are themselves public info, so the floor already puts the base on trial, and the incremental can
  mislead (a variant can beat EXOG while EXOG itself loses to drift).
  - **Weak + semi-strong floor** (vs **Drift**) — "does *any* model beat the random walk?" This is
    the **load-bearing** test, reported two ways: **OOS R²** (Campbell-Thompson 2008 — % MSE
    reduction vs the drift; the standard return-predictability metric) for *effect size*, and **DM**
    for *significance*. Finding: R²_OS is ≈0-or-negative for every public-info model and never
    DM-significant (the only marginally-positive value is own-history `ARIMA rolling`, a tie);
    feature-rich/rolling variants are significantly *worse*. (Weak form is cleanly isolated only in ARIMA, where Drift =
    ARIMA(0,0,0); trees bundle own-history + public info, so for `04`/`05` the floor test is semi-strong.)
  - **Robustness** — the floor test repeated under **absolute-error** loss (`loss='ae'`), since
    returns are heavy-tailed and squared-error DM is low-powered. Squared error stays the headline.
  - **ex-2025 robustness** — re-run the **full battery** (metrics + OOS R² + DM floor se/ae + PT) on
    2023+2024+2026 only, placed **after all full-window tests** (DM → PT → ex-2025). Evaluation-only
    (forecasts unchanged), pooled. The efficiency conclusion *strengthens* once the bull is removed.
    Lives per-notebook in `01_arima`/`04`/`05`/`06`/`lstm_daily`.
- **Pesaran-Timmermann (PT)** — DM is a *magnitude* test; WDA/DA are *directional*, so their
  significance needs PT (`pesaran_timmermann`, two-sided sign-independence test, base-rate aware;
  returns *degenerate* for a constant-sign forecast like the drift). It is the **secondary** lens.
  ARIMA finding: every COT-containing variant clears PT (`+COT` p=0.038, `+Macro+COT` p=0.002,
  `+ALL` p=0.007) while COT-free single groups don't — a coherent but *magnitude-worthless*
  directional signal from positioning. **But it does NOT replicate in any other model class** —
  RF 1/38 PT-significant (≈chance), XGB 0/38, LSTM 0/15. So it's a
  linear-ARIMAX-specific / selection artefact, not a robust signal. Net: do **not** treat
  directional predictability as a finding — at most a heavily-caveated, non-replicating footnote.
  - *Paid-news update (2026-06).* The reputable-source **paid news (title)** shows a similar PT
    signal that — unlike COT — *replicates across both weekly linear models* (VAR `+NewsPaid`
    p=0.002, 03_midas `+SentimentPaid` p=0.003), but **not** in the nonlinear models or the
    daily-MIDAS block, and stays **magnitude-worthless** (every paid rung DM-ties drift, OOS-R²≈0).
    Still a footnote — directional-only, selection-biased (title chosen a priori) — just a slightly
    stronger one than COT's.

---

## 4. Per-notebook methodology (weekly)

All weekly notebooks read the shared `features_weekly.csv` (built by `02_features` §8) +
`feature_groups.json`, and produce `metrics_<model>_weekly.csv` + `period_<model>_weekly.csv`.
**Exception:** the MIDAS notebooks (`03_midas`, `midas_daily`) are Python ports (custom
Beta / exp-Almon weight functions, not `midasr`) and re-aggregate `train/val/test.csv` themselves —
MIDAS needs raw daily/monthly lag *matrices* for the weight polynomials, which the pre-collapsed
frame doesn't expose.

**Paid-news + PMI ablation rungs (2026-06).** Every ablation notebook (ARIMA, VAR, RF, XGB, LSTM,
03_midas, lstm_daily, midas_daily) gained two rung families: (a) a **paid-news twin** of each
GDELT-news rung — the same rung with `news_paid_sentiment_title` swapped in for `news_sentiment`
(`+NewsPaid`, `+SentimentPaid`, `+GS+SentimentPaid`, `+ALLPaid`, …; paid = **title only**, never
body/VADER); and (b) **PMI** (`china_pmi_proxy`, a point-in-time level — see §2) added to each
FRED-daily ("Macro") rung (`+FRED_daily+PMI`, `+FRED_daily+COT+PMI`, `+ALL+PMI`). Exceptions:
**midas_daily** carries paid-news as its own daily-MIDAS HF block and **skips PMI** (a monthly
level doesn't fit the daily-HF structure); **VAR** additionally folds paid-news/PMI into its Granger
tests, and the RF/XGB feature-importance model adds both to its superset. **GKG is not used in any
ablation.** Headline result: **no paid/PMI rung DM-beats the drift floor in any model** — efficiency
holds; paid news is the RMSE-"best" variant for XGB/LSTM/LSTM-d but only at tie-level significance
(and the LSTM "best" is seed-noise-unstable across runs). Directional caveat: see the §3a PT
paid-news footnote.

### `01_arima.ipynb` — ARIMA / ARIMAX baseline  ⭐ **methodology golden standard** (see note)
- **§1** reads *only* the silver target (+ `split`) — pure ARIMA needs nothing else; exog columns
  are read in §4, where the ladder uses them.
- **§2** AIC grid search for $(p,d,q)$ on train+val → (0,0,0).
- **§3** walk-forward fns for **both** windows; run pure ARIMA expanding + rolling-100w. **§3.1**
  plots expanding (≡ the drift) vs rolling — frames expanding ARIMA(0,0,0) = the EMH floor.
- **§4 ablation ladder** — ARIMAX = ARIMA(0,0,0) + lagged exog, run in **both windows** for every
  rung: rung 0 = 6 cross-asset returns (incremental base), then GS / FRED_daily / COT / Reddit /
  News / Sentiment / Macro+COT / GS+Sentiment / ALL + the BestLag filter rungs. Groups from
  `feature_groups.json`, lagged 1w (BestLag pre-lagged, no extra shift).
- **§5** one evaluation table (Naïve, Drift, ARIMA×2, every ARIMAX rung).
- **§6** picks the best variant by **RMSE *and* WDA**; per-period table shows RMSE/MAE/DA/WDA. Saves
  WDA-best under the legacy names + RMSE-best alongside (`*_rmse_*`).
- **§7** predicted-vs-actual, **two panels** (WDA-best, RMSE-best).
- **§8** DM: **vs-Drift floor only** (OOS R² + DM se + ae) — the incremental test was dropped as
  redundant. **§8b** Pesaran-Timmermann. **§8c** ex-2025 (full battery, after all full-window tests).
  Every test prints `-> winner: …`.
- **§9** 2026 zoom, two panels (WDA-best, RMSE-best), drift overlaid. **§9b** ARIMAX coefficient
  inspection (interpretability) is the last section — the notebook ends there.
- The own-history **white-noise / MDS diagnostics** (Ljung-Box + the **Generalised Spectral Test**,
  Hong-Lee 2003) live in **`01_eda` §4** as a property of the *series* (not a model output); `01_arima`
  §2 just back-references them to motivate the AIC-selected ARIMA(0,0,0). The earlier residual
  white-noise battery was a duplicate of `01_eda` §4 and was dropped.
- Outputs `period_arima_weekly.csv` + `period_arimax_weekly.csv` (+ `*_rmse_*` siblings).

> **`01_arima` is the methodology golden standard for the weekly models.** Model idiosyncrasies
> aside (each keeps its justified differences — §6), it carries the **fullest** version of the
> shared methodology: both windows on every rung, dual RMSE+WDA selection, the complete §3a
> significance battery (Drift floor / OOS R² / DM se+ae / ex-2025 / PT) with winner annotations,
> two-panel plots, and the per-period RMSE/MAE/DA/WDA breakdown. **When extending or aligning
> another notebook, mirror `01_arima`'s structure and reporting** unless that model's nature forbids
> it (then document the divergence in §6).
>
> **Cell-structure pattern** (established in `01_arima`, replicated in `04`/`05`):
> - **§5 Evaluate** — a one-line metric summary (naming RMSE/MAE, OOS R², DA/WDA and linking to
>   `01_arima §5` or `notes.md`) sits between the intro and the code. **Full collapsible definitions
>   live in `01_arima` §5 only** — other notebooks point there. The evaluate table includes an
>   `r2_os` column (OOS R² vs Drift, ×100) next to the error metrics; the per-period table in §6/§7
>   does too. The Drift printout is labelled **"Drift WDA by period"** (not "Always-up WDA").
> - **§8/§9 Significance** — opens with an **overview table** (test / axis / role), then each test
>   gets a short `###` intro paragraph immediately before its run cell: DM-SE → DM-AE → PT → ex-2025.
>   **Full `<details>` collapsibles** (formulas, Newey–West, findings) **live in `01_arima` §8 and
>   `notes.md` (Part 3) only** — other notebooks use a brief paragraph and link back. No repeated definitions.

### `02_var.ipynb` — VAR
- VAR(p) with p chosen by AIC. Granger causality + impulse-response plots.
- Walk-forward with expanding window.

### `03_midas.ipynb` — monthly-macro MIDAS (Python port)
- Weekly W-FRI target. **Base** = silver AR lags + 6 cross-asset *weekly* lags (= EXOG).
- **MIDAS-native part**: monthly macro (`cpi, fed_funds, ind_prod, m2`), 3 monthly lags each,
  publication-availability-lagged, collapsed via Beta / exp-Almon / U-MIDAS weights. Stage 1 picks
  the weight family on val RMSE; Stage 2 walk-forwards (expanding + rolling-100w, refit every 4w).
- Ladder: `EXOG → EXOG+Macro → EXOG+Macro+{GS,FRED,COT,Sentiment} → EXOG+Macro+ALL` — every rung past the
  EXOG base carries the MIDAS-weighted macro; plain weekly-only rungs were dropped as ARIMAX duplicates
  (without macro they never invoke the weight functions). Full §3a battery (Drift floor,
  OOS R², DM-vs-Drift se+ae, PT, ex-2025). With only 3 monthly lags, restricted weights ≈ U-MIDAS here.

### `midas_daily.ipynb` — multi-frequency MIDAS (Python port)  *(in `returns/models/daily_inputs/`)*
- The **canonical** MIDAS: a weekly target from *two* higher frequencies at once.
  - **Daily block** — 6 cross-asset daily returns × **K=20 trading-day lags**, Beta/Almon-weighted,
    **replacing** the weekly cross-asset EXOG (a weekly lag is just the sum of its daily window).
  - **Monthly block** — the `03` macro MIDAS, riding along as the second frequency.
  - **Base** — silver's weekly AR lags only.
- **Restricted weights only** (Beta vs Almon by val RMSE): 6×20 = 120 daily lags would overfit as
  free U-MIDAS coefficients — this is the one place the weight polynomial is load-bearing.
  `fit_with_midas` is generic over a list of lag matrices, so daily + monthly compose for free.
- Ladder: `EXOG-d → EXOG-d+Macro` (the multi-frequency headline) `→ EXOG-d+GS → sentiment → EXOG-d+Macro+ALL`,
  each rung in both windows (expanding + rolling-100w, like `03`); same §3a battery + a daily look-ahead audit. **Not expected to beat drift** — the value is a
  *stronger* semi-strong null (public info at native daily resolution, optimally weighted, still
  doesn't forecast the weekly return).

### `04_random_forest.ipynb` and `05_xgboost.ipynb` — fully aligned pair
- **Unified structure (no EXOG special-casing).** §2 builds **all** feature groups up front (EXOG
  base + every ablation + the `EXOG+ALL` kitchen sink); §3 is just helpers; §4 is the **single**
  ladder where **EXOG is rung 0** — there is no standalone feature-matrix / EXOG-only grid-search /
  EXOG walk-forward anymore.
- `build_features()` produces `silver_lag1/2/3` + 6 exog lags (= the EXOG base).
- `tune()` grid-searches hyperparameters **per variant** via `TimeSeriesSplit(5)` on train+val
  (kept per-variant deliberately — strongest null defense: each rung gets its own best HP and still
  loses to drift). Each rung's tuned params are stashed in `variant_params`.
- `walk_forward()` retrains every 4 weeks; supports expanding (default) and rolling-100w.
- **Feature importance** is **permutation importance on the held-out test window** (was MDI / gain —
  switched for an unbiased out-of-sample read, free of MDI's cardinality bias), from a single
  **`EXOG+ALL`** fit with that rung's tuned params — every public-info feature ranked against the
  cross-asset returns, not just the EXOG base. It lives at the **trailing §10** (descriptive, the tree
  analogue of VAR's Granger/IRF — not load-bearing).
- Variant ladder (§4) — each rung evaluated with both windows:
  - `Tech` (silver lags + directional tech indicators, no cross-assets), `EXOG` (baseline)
  - **Feature groups** on EXOG: `EXOG+GS` (gs_ratio_z), `EXOG+NonLin` (squared lags),
    `EXOG+Tech`, `EXOG+Macro` (monthly macro — the frame's `MACRO` group)
  - **Public-info groups** (mirror `01_arima` §7, from `feature_groups.json`):
    `EXOG+FRED_daily`, `EXOG+COT`, `EXOG+FRED_daily+COT`
  - **Sentiment**: `EXOG+Reddit`, `EXOG+News`, `EXOG+Reddit+News`, `EXOG+GS+Sentiment`,
    `EXOG+NonLin+Sentiment`, `EXOG+Tech+Sentiment`
  - `EXOG+ALL` kitchen sink = GS + NonLin + Tech + FRED_daily + COT + Sentiment
- `Y` variant intentionally omitted: tree-based AR(3) duplicates the ARIMA baseline
  without offering anything trees can exploit.
- All ablation groups come from `features_weekly.csv` / `feature_groups.json` and are
  built unconditionally — a missing column errors loudly rather than silently dropping a
  rung (the frame always carries every group, so the old skip-guards are gone).
- **Aligned to the `01_arima` golden standard** — structure, reporting, dual RMSE+WDA selection,
  two-panel plots, the full §3a battery and the cell-structure pattern, all per the note above.
  Numbering matches ARIMA: §7 predicted-vs-actual, §8 significance (DM se+ae), **PT** §8b,
  **ex-2025** §8c, §9 2026 zoom — with **feature importance moved to a trailing §10** (the RF/XGB
  analogue of VAR's Granger/IRF: descriptive, kept after the OOS battery, not load-bearing).
  Justified extras vs ARIMA: feature-importance plot, per-variant tuning, the `NonLin` rung.

### `06_lstm.ipynb` — seed-averaged train + batch test prediction
- §4 hyperparameter mini-grid: `SEQ_LEN × HIDDEN × DROPOUT = 2×2×3` tuned **once** on
  the EXOG variant via val loss with early stopping. Best config reused across all
  variants (per-variant tuning would 8× the runtime with little gain — only input
  dim changes).
- §5 trains each variant over **5 seeds** (`SEEDS=[42,0,1,2,3]`) on train+val, predicting the full
  test set in one pass per seed — `run_variant(name, cols, seed)` seeds **torch + numpy at entry**
  (so MPS nondeterminism is the only residual variation) and checkpoints to a **per-call tempfile**
  (deleted after reload, no `.pt` persisted). The LSTM result is a single random draw (weight init + batch
  shuffle + MPS nondeterminism), so §6 reports **mean ± std** over seeds; §7 saves each variant's
  **median-RMSE seed** as the cross-model forecast (a representative draw — *not* the ensemble mean,
  which would understate error via variance reduction); §8 adds a per-seed DM-vs-Drift check
  (#seeds beating drift). This kills the single-draw artefact: the lone full-window DM-significant
  LSTM (EXOG-MACRO, +2.22 p=0.026) does **not** survive re-seeding (mean R²_OS ≈ −0.5%, 0/5 seeds
  beat drift), so the efficiency floor holds.
- Variant ladder includes `LSTM-Y` (silver-only) — the recurrent architecture is the
  reason this exists here but not in RF/XGB (LSTM extracts AR signal from the SEQ_LEN
  window; trees can't). Mirrors the tree group ladder otherwise: `GS`, `NONLIN`, `TECH`,
  `MACRO`, the public-info `FRED` / `COT` rungs, and the sentiment rungs.
- **Lagging:** the SEQ_LEN window supplies the 1-week lag (to predict `y_t` the input is weeks
  `t-SEQ_LEN..t-1`), so no feature carries an explicit `.shift(1)` — `TECH` included; only `MACRO`
  arrives pre-lagged (publication-dated), per §2.
- **Ladder & architecture:** the canonical 17-rung ablation, identical to `lstm_daily` and matching the
  RF/XGB groups + the `Y` rung (MACRO/FRED/COT inline; `LSTM-EXOG-ALL` = GS+NonLin+Tech+FRED+COT+
  Sentiment, no MACRO — the old RNG-append ordering was dropped). Single LSTM layer (low capacity for
  ~330 seqs) with an **explicit `nn.Dropout`** before the head (`nn.LSTM`'s own dropout is inert at
  1 layer), so the tuned `DROPOUT` is real; regularised by that + early stopping.
- §6 includes the `Naive (t-1 week)` baseline.
- No rolling-vs-expanding split (a 100-week rolling LSTM has only ~80 sequences after SEQ_LEN
  warmup — borderline trainable) and no walk-forward retraining variant. (Periodic-retraining
  robustness, if revisited, is planned as a cross-model study on each model's best variant, not a
  per-notebook pass.)

### `lstm_daily.ipynb` — daily-input LSTM (Friday-only weekly target)  *(in `returns/models/daily_inputs/`)*
- The **daily-input sibling** of `06_lstm` — the same relationship `midas_daily` has to the
  weekly notebooks. Reads the daily `train/val/test.csv` for the sequences (`LOOKBACK=60` trading
  days) and `features_weekly.csv` only for the canonical W-FRI target + `split` + Drift floor. Tests
  whether *daily-resolution* public info forecasts the weekly return.
- **Same template as `06` / the golden standard**: §1–§10, full §3a battery (DM se/ae, PT, ex-2025),
  dual RMSE+WDA selection, 2026 zoom. Variant naming uses the `LSTM-d-*` prefix (`-d` = daily, like
  MIDAS `EXOG-d`); the ladder is identical to `06`'s.
- **Daily features** are built in `02_features §2`: `TECH` is daily-native horizon-matched (the
  weekly windows ×5 trading days), `MACRO` is the monthly lags broadcast to daily point-in-time, and
  `NONLIN` (squared returns) is built inline. The 60-day window supplies the 1-week lag, like `06`.
- **Justified divergences from `06`**: only the input itself — daily-native TECH (not the weekly
  TECH) and MACRO as a monthly step-function at daily resolution. The variant ladder, architecture,
  §5b loss curves, the §3a battery and the **5-seed averaging** (mean ± std, median-seed forecast;
  see `06` §5) are otherwise identical to `06`.
- Outputs use the `lstm_daily` / `_daily` infix to coexist with `06`'s (§7); already wired into
  `evaluation.ipynb`.

### `regime_efficiency.ipynb` — regime/AMH test of the directional signal  *(capstone, not a headline model)*
- Tests whether the weak directional (sign) signal — COT (`cot_mm_net_pct`) + paid-news title
  (`news_paid_sentiment_title`), lagged 1w — is **time-varying** (the §3a / §10 adaptive-markets
  caveat), turning that footnote from interpretation into a tested result.
- **Part A:** expanding-window OOS directional forecast → rolling-52w PT. Pooled PT insignificant
  (p≈0.15) but significant in bursts (best window p<0.05 in 2021/2022/2024/2025) — episodic.
- **Part B:** 2-regime `MarkovRegression` (switching variance): long *calm* (~71wk) vs short
  *turbulent* (~12wk, 6× var) regimes; paid-news coef ~10× larger (β≈−2.2, p≈0.08) in the
  turbulent regime. Marginal, magnitude-worthless — does NOT overturn efficiency.
- Reads `features_weekly.csv`; outputs `metrics_regime_efficiency.csv` + `images/regime_efficiency.png`.
  Written into thesis.tex §6.3 (`fig:regime`). Build the notebook with `nbformat` (no jupytext); run in `tf`.

---

## 5. Variant naming convention

| Family | Pattern | Example |
|---|---|---|
| LSTM | hyphen-separated, all-caps | `LSTM-EXOG-TECH-SENTIMENT` (06); `LSTM-d-EXOG-TECH` (lstm_daily — `-d` = daily input) |
| RF / XGB | plus-separated | `EXOG+Tech+Sentiment` |
| MIDAS | plus-separated | `EXOG+Macro` (03); `EXOG-d+Macro` (midas_daily — `-d` = the daily block) |
| Walk-forward windows | suffix on the variant | `EXOG+Tech expanding`, `EXOG+Tech rolling (100w)` |
| Paid-news twin | `…Paid` on the sentiment slot (title only) | `+NewsPaid`, `+SentimentPaid`, `ARIMAX+ALLPaid`, `LSTM-EXOG-ALLPAID`, `EXOG-d+Sentiment_HF_Paid` |
| PMI rung | `+PMI` on a FRED-daily ("Macro") rung | `+FRED_daily+PMI`, `+ALL+PMI` (none in midas_daily) |

The **efficiency** DM baseline is the **Drift** floor (§3a) — used in all notebooks. (The older
incremental baseline — smallest variant with the base regressors — was dropped from `01_arima`/`04`/`05`
as redundant; it may still appear in not-yet-aligned notebooks like LSTM/MIDAS.)

`Y` (silver-only) appears only in LSTM. `Tech` (silver lags + the directional tech
indicators, no cross-assets) appears in all three feature-based models.

---

## 6. Subtleties + cross-model differences

### What's aligned
- Same shared `features_weekly.csv` (§2) → identical W-FRI aggregation, `EXOG`, `TECH` by construction
- Same `period_metrics` / `PERIODS` / `diebold_mariano` from `eval_utils`; same headline metrics
  (RMSE, MAE, DA, WDA — **primary = RMSE / DM-vs-Drift**, WDA the secondary directional lens; §3a)
- Same `Naive (t-1)` + **`Drift (prevailing mean)`** rows (Drift = the EMH floor), 2026 zoom plot,
  and **DM-vs-Drift** floor as the sole efficiency arbiter (incremental test dropped as redundant — §3a)

### Genuine, justified differences

| Aspect | RF / XGB | LSTM (06 / lstm_daily) | Reason |
|---|---|---|---|
| Walk-forward retraining | every 4 weeks | none; **seed-averaged** (5 seeds, mean ± std) | LSTM training is expensive and the sample is tiny (~330 sequences); seed-averaging defends the single draw |
| Window scheme | expanding + rolling-100w | expanding only | 100w rolling = ~80 LSTM sequences = unstable training |
| Hyperparameter search | grid per variant | grid once on EXOG, reused | LSTM cost; per-variant tuning would 8× runtime |
| `Y` (silver-only) variant | omitted | included | tree AR(3) duplicates ARIMA; recurrent AR is novel |
| Feature importance plot | yes (MDI / gain) | no | LSTM has no clean per-feature importance |

### MIDAS notebooks (`03_midas`, `midas_daily`) — justified divergences
**Python ports** — custom Beta / exp-Almon weight functions + an NLS `fit_with_midas` (not the R
`midasr` package); they import `eval_utils` directly. They **re-aggregate `train/val/test.csv`**
rather than reading `features_weekly.csv` — MIDAS needs raw
lag *matrices* for the weight polynomials (see §4 intro). `03` weights monthly macro only; `midas_daily`
adds the daily cross-asset block. Metrics keep the `metrics_midas*.csv` name (`03` has **no**
`_weekly` suffix); `evaluation.ipynb` maps both. Both forecast the weekly target — same task as the
other notebooks.

---

## 7. Output file naming

Inputs are the shared `features_weekly.csv` + `feature_groups.json` (§2). Outputs:

```
metrics_<model>_weekly.csv               # full metrics table (used by evaluation.ipynb)
metrics_lstm_daily_weekly.csv            # lstm_daily output (weekly Friday target)
period_<model>_weekly.csv                # PERIODS breakdown of best variant by WDA
period_arima_weekly.csv vs period_arimax_weekly.csv   # ARIMA notebook saves both (WDA-best, legacy names)
period_arimax_rmse_weekly.csv            # ARIMA §6 RMSE-best per-period sibling
preds_{arima,arimax}_best_weekly.csv     # ARIMA cross-model preds (WDA-best) — read by evaluation.ipynb
preds_{arima,arimax}_bestrmse_weekly.csv # RMSE-best preds siblings — read by evaluation.ipynb
period_{rf,xgboost}_rmse_weekly.csv      # RF/XGB §8 RMSE-best per-period siblings
preds_{rf,xgboost}_bestrmse_weekly.csv   # RF/XGB RMSE-best preds siblings — read by evaluation.ipynb
# MIDAS (`*_py` notebooks) — note `03` metrics file has NO _weekly suffix; evaluation.ipynb maps both:
metrics_midas.csv               period_midas_weekly.csv        preds_midas_best_weekly.csv         # 03
metrics_midas_daily_weekly.csv  period_midas_daily_weekly.csv  preds_midas_daily_best_weekly.csv   # midas_daily
midas_stage1_specs.csv / midas_daily_stage1_specs.csv   # Stage-1 weight-family pick
# 06 / lstm_daily no longer persist per-variant .pt checkpoints — each variant trains over 5 seeds
# with ephemeral per-seed checkpoints (tempfiles). Instead: metrics_lstm*.csv carries the seed-mean
# of each metric + a `*_std` column, and preds_lstm*_{best,bestrmse}_weekly.csv hold the
# median-RMSE-seed (representative) forecast.
# lstm_daily also writes period_lstm_daily_weekly.csv (+ _rmse_ sibling) and preds_lstm_daily_best_weekly.csv
# (+ _bestrmse_). The 06 public-info rungs (FRED / COT / FRED+COT) just add rows to
# metrics_lstm_weekly.csv — no new CSV.
```

`evaluation.ipynb` consumes the `metrics_*` files plus every family's
`preds_*_{best,bestrmse}_weekly.csv` (9 families × 2 selections; the Drift floor is
rebuilt from `features_weekly.csv`). It runs the consolidated DM-vs-Drift battery
(se + ae + OOS R²), PT (§5b), an ex-2025 re-run of all of it (§5c), and computes the
per-period breakdown (heatmap rows labelled `family — variant`) from the preds directly —
the `period_*.csv` files are **no longer read** by the top-level evaluation (they
remain per-notebook outputs). **If you rename or add a model's output CSV, update
`evaluation.ipynb` too.**

---

## 8. Working-with-this-codebase conventions

- Don't write into `data/raw/` — collection scripts are the only path to those files.
- Notebooks address data via relative paths matched to their depth (model notebooks: `../../../data/processed/<name>.csv`).
- The `evaluation.ipynb` notebook expects model notebooks to have been re-run so
  their CSV outputs are fresh — note this when making changes.
- LSTM runs on Apple MPS by default; falls back to CUDA / CPU. A global `SEED=42` seeds
  setup, but each LSTM variant is **re-seeded per run** inside `run_variant` (torch + numpy)
  across `SEEDS=[42,0,1,2,3]` (§4) — MPS still introduces some non-determinism vs CPU.
- **Documentation files must be easy to follow and digestible.** This applies to
  `CLAUDE.md`, every `notes.md`, and the markdown cells inside notebooks. Default to
  short: bullet lists over prose, one sentence per point, no "defensible alternative"
  / "exhaustive rationale" sections unless load-bearing. If a section can be 3 lines
  instead of 30, make it 3 lines — the user will ask for more if needed.
- **Don't action TODOs / FIXMEs / commented-out code unless the user asks.** Existing
  TODO comments, `# TODO:` markers, commented-out cells and similar leftovers are the
  user's pending work — they're notes-to-self, not work items for the assistant.
  Treat them as read-only context: useful for understanding intent, but do not
  resolve, refactor, delete, or "clean them up" while doing unrelated work. If a TODO
  is directly in the way of a task the user *did* ask for, surface it as a question
  before acting on it.
- **Comments prefixed with `Asier:` are off-limits.** Anything written as
  `# Asier: ...` (or `<!-- Asier: ... -->` in markdown) is a personal note the user
  has left for themselves — questions to research, decisions to make, reminders to
  follow up. Do not remove, rewrite, or answer them. Same rule as TODOs: read-only
  context, surface as a question if it blocks an asked-for task.

### Coding standards (notebooks) — read top-to-bottom, no surprises

- **Define a name right before its first use.** No define-early / use-later: a constant or
  variable belongs in (or just above) the cell/section that first consumes it, not parked in an
  earlier "setup" cell. E.g. `WINDOW` and the feature-column groups live in the build-systems
  section, not the load cell. Forward references make the notebook flow hard to follow.
- **Build and name parallel things uniformly.** Sibling objects should share one construction
  idiom and one naming convention — e.g. every feature group is `X_COLS = series_dict(...)` (a
  name→series dict), and bare names are taken with `list(X_COLS)`. Don't let one sibling be a raw
  list-comprehension named off-pattern (the old `EXOG_RETURNS` list next to the `*_COLS` dicts).
  Keep **one canonical form per concept** and derive the rest from it.

---

## 9. Volatility forecasting (`notebooks/volatility/`)

A parallel chapter that asks whether **volatility** is more forecastable than
**direction**. Same `train/val/test.csv` and W-FRI calendar as the return notebooks,
but the target is **weekly realised volatility**, not the log-return:

$$\text{RV}_t = \sqrt{\sum_{i \in \text{week } t} r_i^2}$$

— daily squared returns summed per W-FRI week, then square-rooted (realised *variance*
is additive across days, volatility is not, so we sum then sqrt).

### Layout — features notebook + one notebook per model

| Notebook | Contents |
|---|---|
| `00_features.ipynb` | Load daily data, weekly RV aggregation, EDA (ACF), build HAR + EXOG + Reddit-sentiment features, split → `volatility_weekly.csv` |
| `01_garch.ipynb` | GARCH(1,1), walk-forward refit |
| `02_har.ipynb` | Naïve floor + HAR-RV (Corsi 2009) + HAR-X sentiment / cross-asset ablation |
| `03_random_forest.ipynb` | RF on HAR + EXOG + MDI importance + sentiment ablation |
| `04_xgboost.ipynb` | XGBoost on HAR + EXOG + gain importance + sentiment ablation |
| `var_backtest.ipynb` | **Economic-significance capstone** — turns each RV forecast into a 1-week VaR, backtests Kupiec / Christoffersen / pinball / ES (see below) |
| `evaluation.ipynb` | Cross-model table, per-year breakdown, 2026 zoom, DM tests, sentiment-ablation summary |

Unlike the return notebooks (which each re-aggregate `train/val/test.csv`), every
volatility model notebook loads the single `volatility_weekly.csv` built by
`00_features.ipynb` — so the RV target, feature definitions and the train/val/test
`split` column are guaranteed identical across models. Run order: `00_features` →
`01`–`04` (any order) → `evaluation`.

### Feature sets

- **HAR** — three trailing averages of past RV (Corsi 2009): `rv_w_lag1` (1w),
  `rv_m_lag1` (4w mean), `rv_q_lag1` (12w mean). All `.shift(1)`-ed — no look-ahead.
- **EXOG** — 1-week lags of the six cross-asset RVs `[gold, copper, usd, sp500, vix,
  oil]`. Used by the tree models only; HAR-RV and GARCH stay univariate.
- **SENTIMENT** — two 1-week-lagged weekly Reddit features: `reddit_attention_lag1`
  (log post volume — an *attention* proxy) and `reddit_sent_abs_lag1` (|weekly-mean tone|
  — an *intensity* proxy). A third, `reddit_sent_disp_lag1` (within-week tone dispersion),
  stays in the frame for EDA but is dropped from every modelling rung — near-zero lead-lag
  correlation with RV (`00_features` §2.5; notes.md keeps the 3-view correlation story). Used
  only by the sentiment ablations in `02`/`03`/`04`. **Reddit only** for *sentiment* —
  GDELT news coverage starts late 2017 and has zero-article weeks even inside the test
  window, too sparse for a clean RV regressor; recorded as a documented data limitation.
- **TRENDS** — `trends_lag1`: log Google-Trends "silver" search interest (W-FRI mean,
  `.shift(1)`-ed) — a second, *broad-retail* attention proxy, only ~0.3 correlated with
  `reddit_attention_lag1`, with full 2015–2026 coverage (unlike GDELT news). Feeds the
  `HAR+Trends` ablation rung in `02`/`03`/`04`. EDA in `00_features` §5c.
- `volatility_weekly.csv` also carries `silver_ret` (weekly log-return, used by GARCH)
  and a `split` column (`train` / `val` / `test`).

### Metrics — `src/vol_utils.py`

DA/WDA do not apply (RV ≥ 0), so volatility has its own helpers:

```python
vol_evaluate(name, actual, pred, prev_actual)                # RMSE / MAE / R² / DCA dict
vol_period_metrics(actual, pred, prev_actual, idx, PERIODS)  # per-year RMSE + DCA
dca(actual, pred, prev_actual)                               # direction-of-change accuracy
vol_diebold_mariano(actual, p1, p2, n1, n2, loss='qlike')    # DM test, loss-selectable
```

`PERIODS` is reused straight from `eval_utils`. **DCA** = Direction-of-Change Accuracy
on $\Delta\log\text{RV}$ — did the model call vol rising vs falling. The Naïve model
has DCA ≈ 0 by construction (predicting $\text{RV}_{t-1}$ implies no change).

`vol_diebold_mariano` **replaces** `eval_utils.diebold_mariano` for this chapter. RV is
heavy-tailed enough that squared-error DM is near-powerless — a handful of extreme
weeks dominate the loss differential and inflate its variance, so a real RMSE
improvement can still fail an MSE-DM test. The loss is therefore selectable and
defaults to **QLIKE**, the proxy-robust volatility loss (Patton 2011). `evaluation.ipynb`
reports QLIKE-DM as the primary test and squared-error DM only as a reference.

### HAR-X / sentiment ablation (`02_har`, `03` / `04`)

A focused study — separate from the headline cross-model comparison — of what an
extended HAR-RV gains from (a) **cross-asset volatility spillover** or (b) **public
attention / sentiment**. Five mechanism groups, kept apart so any effect is attributable:
**Cross-asset** (the full EXOG set: 6 cross-asset RV lags), **Reddit attention**
(`reddit_attention_lag1`), **Reddit sentiment intensity** (`reddit_sent_abs_lag1`),
**Paid-news** (`paid_attention_lag1`, `paid_sent_abs_lag1`) and **Search attention**
(`trends_lag1`, Google Trends).

- `02_har`'s HAR-X ablation runs a **9-rung OLS ladder** against bare HAR: `HAR+EXOG`
  (full linear spillover — the **linear sibling** of the RF/XGB models in `03`/`04`),
  `HAR+RedditAttention`, `HAR+RedditSent`, `HAR+Reddit` (combined Reddit),
  `HAR+Trends`, `HAR+PaidAttention`, `HAR+PaidSent`, `HAR+Paid` (combined paid-news).
  Every rung is DM-tested against bare HAR. Empirically Trends gives the **lowest RMSE**
  of any rung yet is **not** QLIKE-significant (it tracks RV mainly through high-vol
  episodes), while the Reddit-sentiment rungs are strongly QLIKE-significant but barely
  move RMSE — a clean RMSE-vs-QLIKE divergence. (`HAR+EXOG` turned out to be
  significantly *worse* than bare HAR on this sample, so the secondary test of the
  combined Reddit rung vs `HAR+EXOG` is a lower bar than vs HAR and is reported
  alongside for completeness only.)
- `03` / `04` mirror the same 9-rung ladder with RF / XGB as the model; their
  baselines (`RF/XGB (HAR+EXOG)`) already contain the six EXOG lags.
- Every rung is fitted and scored on the **same sample** (weeks where all Reddit +
  Trends + paid-news features exist — Trends is fully covered, so the sample is 174 of
  175 test weeks); the no-sentiment baseline is re-scored on that sample so the
  QLIKE-DM test is apples-to-apples.
- The headline models never see the sentiment columns, so the cross-model comparison
  (§1–§4 of `evaluation.ipynb`) is unaffected — the ablation is purely additive.
- Reading the `HAR+EXOG` row against `RF/XGB (HAR+EXOG)` in `03`/`04` cleanly isolates
  the nonlinear gain (or loss) of the trees on top of the linear cross-asset story —
  if both lose to HAR, the feature set is dry rather than the model class limiting.

### VaR backtest (`var_backtest`) — economic significance

Turns each weekly RV forecast into a 1-week VaR for the silver return,
`VaR_t(α) = μ + σ̂_t·q_α` (μ = train drift; `q_α` = Normal **and** unit-variance Student-`t5`
quantile), breach if `r_t < VaR_t`. Backtests Kupiec (unconditional coverage), Christoffersen
(conditional coverage), the pinball/quantile loss, and ES, with a Newey-West DM on the pinball
loss. Compares **17 σ̂ rows** — naïve `RV_{t-1}`, GARCH, and five rungs (HAR, +RedditAttention,
+RedditSent, +PaidAttention, +Paid) each under OLS / RF / XGB (the chapter's same-sample ablation
fits) — on the 174-wk test window (only `σ̂` varies across rows). **Headline:** every family beats
the naïve floor at the 5% tail (pinball DM: HAR `p<0.001`, RF/XGB `p≤0.004`; naïve Kupiec-rejected
at 1%); `RF (HAR)` is best-calibrated (4.6% breach; nominally DM-beats HAR, `p=0.029`, 1 of 14
pairs); **HAR+RedditAttention** is the only HAR rung restoring the 1% Normal coverage HAR loses
(`p≈0.07`); tree sentiment rungs never significantly help (RF: significantly *worse* at t5-1%).
Thesis Table 6 = the 4 HAR-family rows + the two tree bases; tree sentiment rungs stay notebook-only.
Reads `pred_*_volatility.csv` + `volatility_weekly.csv` (`silver_ret`); built with `nbformat`, run
in `tf`. Written into thesis.tex **§7.4** (`tab:var-backtest`, `fig:var-backtest`).

### Output file naming

```
volatility_weekly.csv                    # shared feature frame (00_features)
metrics_<model>_volatility.csv           # har / garch / rf / xgb headline metrics
period_<model>_volatility.csv            # per-year RMSE + DCA breakdown
pred_<model>_volatility.csv              # test-set predictions, consumed by evaluation
metrics_<model>_sentiment_volatility.csv # har/rf/xgb attention/sentiment + Trends ablation rungs + QLIKE-DM
metrics_volatility_summary.csv           # evaluation.ipynb cross-model table
period_volatility_summary.csv            # evaluation.ipynb stacked per-year table
dm_volatility_summary.csv                # evaluation.ipynb QLIKE + MSE DM stats
metrics_sentiment_volatility_summary.csv # evaluation.ipynb stacked sentiment-ablation table
metrics_var_backtest_volatility.csv      # var_backtest: coverage/pinball/ES × Normal,t5 × 5%,1% (→ §7.4)
```

The top-level `notebooks/evaluation.ipynb` (returns) does **not** read these — the
volatility chapter has its own `evaluation.ipynb` inside `notebooks/volatility/`.

### Differences from the return notebooks (all justified)

| Aspect | Return notebooks | Volatility notebooks |
|---|---|---|
| Target | weekly log-return | weekly realised volatility (RV ≥ 0) |
| Primary metric | WDA | RMSE (DCA as the directional read) |
| Feature source | each notebook re-aggregates the splits | one shared `volatility_weekly.csv` |
| Variant ladder | EXOG/Tech/Sentiment rungs | none for the headline models; a Reddit-sentiment ablation on HAR/RF/XGB |
| Walk-forward windows | expanding + rolling-100w | GARCH refits walk-forward; HAR/RF/XGB single-fit |
| DM baseline | smallest variant w/ base regressors | Naïve ($\text{RV}_{t-1}$) |
| DM loss function | squared error | QLIKE primary (squared error kept as reference) |

---

## 10. Thesis framing — market efficiency

The two chapters test **two nested forms** of the Efficient Market Hypothesis
(Fama 1970, 1991), and the distinction should be made explicit in the writeup rather
than lumping everything under "weak form":

- **Weak form** — past *price/return* info. Tested by the own-history models: ARIMA, VAR own-lags,
  the `silver_lag1/2/3` terms in the trees, `LSTM-Y`. Finding (§3a): no own-history model beats the
  random-walk drift (= ARIMA(0,0,0)) on the DM floor — weekly silver returns are statistically white
  noise. (The naïve $y_{t-1}$ benchmark is shown for reference, not as the EMH floor.)
- **Semi-strong form** — all *public* info. Tested by the exogenous rungs: EXOG cross-asset lags,
  MIDAS macro, Reddit / News sentiment, COT. Finding (§3a): no model — including EXOG itself — beats
  the drift floor, and feature-rich / rolling variants are significantly *worse*. These nulls are
  strictly stronger than weak form.
  - *Directional caveat (secondary, §3a).* PT finds weak, magnitude-worthless sign predictability
    from **COT positioning** — pooled-window-significant but selection-biased. It doesn't overturn
    the magnitude verdict; an adaptive-markets footnote, consistent with Lo (2004).

**Predictable volatility does not contradict the EMH.** The hypothesis constrains the
conditional *mean* of returns (and risk-adjusted expected returns), not the conditional
*variance*. Volatility clustering is not a tradable arbitrage the way mean
predictability would be, so the volatility chapter's positive results (HAR / GARCH
beating the naïve RV floor) coexist with the efficiency findings without tension. The
framing also aligns with the adaptive-markets view of Lo (2004), which explicitly
accommodates time-varying second moments alongside (locally) efficient first moments.

The thesis is therefore **one coherent story**, not "returns failed, volatility is a
consolation prize": *weekly silver returns are unforecastable from past prices and
public information — weak- and semi-strong-form efficiency hold — yet the conditional
variance is strongly predictable.*
