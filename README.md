# FOMC Implied-vs-Realized Volatility Study

A modular Python pipeline for analysing the **implied / realized volatility dichotomy** around FOMC meetings, built for Bloomberg BQuant JupyterLab kernels using Polars DataFrames.

## What it does

For each FOMC meeting date × Treasury tenor, the pipeline:

1. Extracts **implied volatility** from front Treasury-future options (`HIST_CALL_IMP_VOL`)
2. Computes **realized volatility** from a configurable forward (or backward) window of daily log-returns
3. Derives the **Volatility Risk Premium**: `VRP = IV − RV`
   - `VRP > 0` → market over-priced the meeting (normal insurance premium)
   - `VRP < 0` → market under-priced; realized exceeded implied (thesis condition)
4. Joins with a labelled **FOMC regime set** and tests whether mean VRP differs from zero using **Newey-West HAC** standard errors (correcting for overlap in forward windows)
5. Produces publication-ready **visualizations** and saves results to Parquet

## Inputs

| Variable | Description |
|---|---|
| `impl_pl` | Polars DataFrame — first col = FOMC date; remaining cols = tenors `['2Y','5Y','10Y','30Y','20Y']` holding `HIST_CALL_IMP_VOL` (annualised %, front Treasury-future option) |
| `rv_pl` | Same structure, holding **Treasury future prices** (daily panel or one-price-per-meeting — see Phase 0) |
| `regime_pl` | Polars DataFrame — FOMC date + labelled regime per meeting |

Tenors are read dynamically from `impl_pl` column names — nothing is hard-coded.

## Usage

```python
from fomc_vol_study import run_fomc_vol_study
from pathlib import Path

results = run_fomc_vol_study(
    impl_pl    = impl_pl,
    rv_pl      = rv_pl,
    regime_pl  = regime_pl,
    # px_daily_pl = px_daily_pl,  # supply after BQL re-pull if rv_pl is sparse
    tau        = 21,              # forward window in trading days
    direction  = "forward",       # "forward" | "backward"
    hac_lags   = 21,              # Newey-West Bartlett truncation
    output_dir = Path("."),
)
```

If `rv_pl` contains only one price per FOMC meeting (detected automatically in Phase 0), the pipeline halts and prints a ready **BQL snippet** to re-pull daily `PX_LAST` for the correct date range. Pass the resulting frame as `px_daily_pl=` and re-run.

## Pipeline phases

| Phase | Function | Description |
|---|---|---|
| 0 | `inspect_and_validate` | Schema / shape / nulls; date casting; daily-vs-sparse detection; BQL snippet |
| 1 | `tidy_to_long` | Unpivot wide frames to `(fomc_date, tenor, iv)` and `(date, tenor, price)` |
| 2 | `compute_rv` | Forward/backward RV windows; annualised percentage points |
| 3 | `build_vrp` | Join IV & RV → VRP, ratio, under-pricing flag |
| 4 | `summarize_by_regime` | Group stats + Newey-West significance per (regime, tenor) |
| 5 | `plot_vrp_study` | Three-panel visualization |
| 6 | `save_and_summarize` | Parquet output + concise text summary |

## Public API

```python
compute_rv(px_long, fomc_dates, tau, direction)  -> pl.DataFrame  # (fomc_date, tenor, rv)
build_vrp(iv_long, rv_longf)                     -> pl.DataFrame  # (fomc_date, tenor, iv, rv, vrp, ratio, underpriced)
summarize_by_regime(vrp_pl, regime_pl, ...)      -> tuple[pl.DataFrame, pl.DataFrame]
```

## Outputs

| File | Contents |
|---|---|
| `vrp_pl.parquet` | Full per-(FOMC date, tenor) VRP table with regime labels |
| `regime_summary_pl.parquet` | Summary stats by (regime, tenor) and (regime, ALL) incl. NW t-stats |

### Visualizations

- **VRP time series** per tenor with regime-coloured background bands and over/under-pricing fill
- **Bar chart** of mean VRP by regime × tenor
- **IV vs RV scatter** coloured by regime with 45° identity line (points below = under-priced meeting)

## Statistical note

Forward RV windows overlap between adjacent FOMC meetings (~6–8 week spacing, 21-day window). Newey-West HAC errors with Bartlett kernel truncated at `tau` lags correct for this overlap-induced autocorrelation. Some regimes contain very few meetings (N=1–2); t-statistics are asymptotically valid but unreliable at small N — treat economic magnitude (`mean_vrp`) as the primary guide.

## Requirements

```
polars
numpy
matplotlib
scipy
statsmodels   # optional but recommended for HAC; manual fallback included
```

Bloomberg BQuant environment provides `bql` and `pandas` (used only at the BQL data-load boundary).

## License

MIT
