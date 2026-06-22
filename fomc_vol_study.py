# ============================================================
# FOMC Implied-vs-Realized Volatility Study
# Bloomberg BQuant / JupyterLab
# Paste into one cell or split at the ── CELL BREAK ── markers
# ============================================================

from __future__ import annotations

import warnings
from datetime import timedelta
from pathlib import Path
from typing import Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    warnings.warn("statsmodels not available; Newey-West variance computed manually.")

# ── Global params (override before calling run_fomc_vol_study) ─────────────
TAU         = 21          # forward window in trading days
DIRECTION   = "forward"   # "forward" | "backward"
HAC_LAGS    = TAU         # Newey-West Bartlett truncation = tau
OUTPUT_DIR  = Path(".")

# Stable colour palette — one colour per regime label (up to 10)
_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


# ──────────────────────────────────────────────────────────────────────────
# Internal utilities
# ──────────────────────────────────────────────────────────────────────────

def _unpivot(df: pl.DataFrame,
             on: list[str],
             index: str,
             variable_name: str,
             value_name: str) -> pl.DataFrame:
    """Polars unpivot / melt compatibility shim (handles <0.19 and ≥0.19)."""
    try:
        return df.unpivot(
            on=on, index=index,
            variable_name=variable_name, value_name=value_name,
        )
    except AttributeError:
        return df.melt(
            id_vars=index, value_vars=on,
            variable_name=variable_name, value_name=value_name,
        )


def _key(v) -> str:
    """Unwrap single-element tuple returned by Polars group_by iteration."""
    return v[0] if isinstance(v, tuple) else v


def _regime_colors(regime_series: pl.Series) -> dict:
    labels = sorted(regime_series.drop_nulls().unique().to_list())
    return {lab: _PALETTE[i % len(_PALETTE)] for i, lab in enumerate(labels)}


# ══════════════════════════════════════════════════════════════════════════
# PHASE 0 — INSPECT & VALIDATE
# ══════════════════════════════════════════════════════════════════════════

def inspect_and_validate(
    impl_pl:   pl.DataFrame,
    rv_pl:     pl.DataFrame,
    regime_pl: pl.DataFrame,
    tau:       int = TAU,
) -> tuple[
    pl.DataFrame,               # impl_pl   (cleaned)
    pl.DataFrame,               # rv_pl     (cleaned; "fomc_date" or "date" first col)
    pl.DataFrame,               # regime_pl (date cast)
    Optional[pl.DataFrame],     # px_daily_pl (None if BQL re-pull needed)
    list[str],                  # tenors
    str,                        # regime_date_col
    str,                        # regime_label_col
]:
    """
    Print schema/shape/head/nulls; rename first columns to fomc_date; cast
    dates; detect tenors from impl_pl; decide whether rv_pl is a usable daily
    price panel.

    Returns px_daily_pl=None and prints a BQL snippet if rv_pl has only one
    price per FOMC meeting.
    """

    def _report(name: str, df: pl.DataFrame) -> None:
        print(f"\n{'═'*60}\n  {name}   shape={df.shape}\n{'═'*60}")
        print("  Schema:")
        for col, dtype in df.schema.items():
            print(f"    {col:30s}  {dtype}")
        print(f"\n  Head (3 rows):\n{df.head(3)}")
        nulls = {c: df[c].null_count() for c in df.columns if df[c].null_count()}
        print(f"\n  Null counts: {nulls if nulls else 'none'}")

    _report("impl_pl",   impl_pl)
    _report("rv_pl",     rv_pl)
    _report("regime_pl", regime_pl)

    # ── tenors: all columns after the first in impl_pl ───────────────────
    tenors: list[str] = impl_pl.columns[1:]
    print(f"\n  Detected tenors: {tenors}")

    # ── rename & cast date columns ────────────────────────────────────────
    def _fix_date(df: pl.DataFrame, new_name: str) -> pl.DataFrame:
        old = df.columns[0]
        df = df.rename({old: new_name})
        if df[new_name].dtype not in (pl.Date,):
            try:
                df = df.with_columns(pl.col(new_name).cast(pl.Date))
            except Exception:
                df = df.with_columns(
                    pl.col(new_name).str.to_date(strict=False).alias(new_name)
                )
        return df

    impl_pl = _fix_date(impl_pl, "fomc_date")
    rv_pl   = _fix_date(rv_pl,   "fomc_date")   # may be renamed to "date" below

    # ── detect regime date / label columns ───────────────────────────────
    def _is_date_col(df: pl.DataFrame, col: str) -> bool:
        return df[col].dtype in (pl.Date, pl.Datetime) or "date" in col.lower()

    regime_date_col  = next(
        (c for c in regime_pl.columns if _is_date_col(regime_pl, c)),
        regime_pl.columns[0],
    )
    regime_label_col = next(
        (c for c in regime_pl.columns if c != regime_date_col),
        regime_pl.columns[1],
    )
    print(f"\n  regime_date_col='{regime_date_col}'  regime_label_col='{regime_label_col}'")

    if regime_pl[regime_date_col].dtype not in (pl.Date,):
        try:
            regime_pl = regime_pl.with_columns(
                pl.col(regime_date_col).cast(pl.Date)
            )
        except Exception:
            regime_pl = regime_pl.with_columns(
                pl.col(regime_date_col).str.to_date(strict=False)
            )

    # ── decide daily vs one-price-per-meeting ─────────────────────────────
    n_rv        = rv_pl.shape[0]
    n_fomc      = impl_pl.shape[0]
    dates_rv    = rv_pl["fomc_date"].sort()
    median_gap  = int(
        dates_rv.diff().drop_nulls().cast(pl.Int64).median() or 0
    )   # calendar days between consecutive dates

    is_daily = (n_rv > n_fomc * 3) or (0 < median_gap <= 7)

    if is_daily:
        print(f"\n  ✓ rv_pl: {n_rv} rows, median date gap {median_gap}d "
              "→ treating as daily price panel.")
        rv_pl       = rv_pl.rename({"fomc_date": "date"})
        px_daily_pl: Optional[pl.DataFrame] = rv_pl
    else:
        print(f"\n  ✗ rv_pl: {n_rv} rows ≈ {n_fomc} FOMC meetings "
              f"(median gap {median_gap}d).")
        print("  RV CANNOT be computed from one price per FOMC meeting.")

        min_d = (impl_pl["fomc_date"].min() - timedelta(days=10)).strftime("%Y-%m-%d")
        max_d = (impl_pl["fomc_date"].max() + timedelta(days=40)).strftime("%Y-%m-%d")

        _bbg_map = {
            "2Y": "TU1 Comdty", "5Y": "FV1 Comdty", "10Y": "TY1 Comdty",
            "20Y": "US1 Comdty", "30Y": "WN1 Comdty",
        }
        tickers_str = ", ".join(
            f'"{_bbg_map.get(t, t + " Comdty")}"' for t in tenors
        )
        tenor_map_str = "{" + ", ".join(
            f'"{_bbg_map.get(t, t)}": "{t}"' for t in tenors
        ) + "}"

        print("\n" + "=" * 60)
        print("  BQL SNIPPET — run in a new cell, then re-call run_fomc_vol_study")
        print("  with px_daily_pl=px_daily_pl")
        print("  NOTE: TU1/FV1/TY1/WN1 are generic front futures — flag roll dates")
        print("  inside any 21-day forward window (coincide with expiry ~Mar/Jun/Sep/Dec).")
        print("=" * 60)
        print(f"""
import bql, polars as pl
bq = bql.Service()

tickers   = [{tickers_str}]
tenor_map = {tenor_map_str}

date_rng = bq.func.range('{min_d}', '{max_d}')
fields   = {{'px': bq.data.px_last(dates=date_rng, per='D', fill='prev',
                                    currency='USD')}}
req  = bql.Request(tickers, fields)
resp = bq.execute(req)

raw = bql.combined_df(resp).reset_index()
# raw columns: DATE, ID, px

px_daily_pl = (
    pl.from_pandas(raw)
    .rename({{'DATE': 'date', 'ID': 'ticker', 'px': 'price'}})
    .with_columns([
        pl.col('date').cast(pl.Date),
        pl.col('ticker').replace(tenor_map).alias('tenor'),
    ])
    .filter(pl.col('tenor').is_in({tenors}))
    .select(['date', 'tenor', 'price'])
    .drop_nulls('price')
    .sort(['tenor', 'date'])
)
print(px_daily_pl.head())
""")
        print("=" * 60)
        px_daily_pl = None

    return (
        impl_pl, rv_pl, regime_pl,
        px_daily_pl,
        tenors, regime_date_col, regime_label_col,
    )


# ══════════════════════════════════════════════════════════════════════════
# PHASE 1 — TIDY TO LONG
# ══════════════════════════════════════════════════════════════════════════

def tidy_to_long(
    impl_pl:     pl.DataFrame,
    px_daily_pl: pl.DataFrame,
    tenors:      list[str],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Unpivot implied vol (wide→long) and standardise daily price panel.

    Parameters
    ----------
    impl_pl      : (fomc_date, *tenors) implied vols
    px_daily_pl  : daily prices — either wide (date, *tenors)
                   or already long (date, tenor, price)
    tenors       : tenor labels

    Returns
    -------
    iv_long : (fomc_date, tenor, iv)
    px_long : (date, tenor, price) sorted [tenor, date]
    """
    # ── implied vol ───────────────────────────────────────────────────────
    iv_long = (
        _unpivot(impl_pl, on=tenors, index="fomc_date",
                 variable_name="tenor", value_name="iv")
        .drop_nulls("iv")
        .sort(["tenor", "fomc_date"])
    )

    # ── daily prices ──────────────────────────────────────────────────────
    date_col = px_daily_pl.columns[0]
    if "tenor" in px_daily_pl.columns and "price" in px_daily_pl.columns:
        # Already long (from BQL re-pull)
        px_long = (
            px_daily_pl
            .rename({date_col: "date"})
            .filter(pl.col("tenor").is_in(tenors))
            .select(["date", "tenor", "price"])
            .drop_nulls("price")
            .sort(["tenor", "date"])
        )
    else:
        # Wide format
        wide_cols = [c for c in px_daily_pl.columns if c in tenors]
        px_long = (
            px_daily_pl
            .rename({date_col: "date"})
            .pipe(
                lambda df: _unpivot(df, on=wide_cols, index="date",
                                    variable_name="tenor", value_name="price")
            )
            .drop_nulls("price")
            .sort(["tenor", "date"])
        )

    print(f"\n  iv_long : {iv_long.shape}")
    print(iv_long.head(3))
    print(f"\n  px_long : {px_long.shape}")
    print(px_long.head(3))

    return iv_long, px_long


# ══════════════════════════════════════════════════════════════════════════
# PHASE 2 — REALIZED VOLATILITY
# ══════════════════════════════════════════════════════════════════════════

def compute_rv(
    px_long:    pl.DataFrame,
    fomc_dates: pl.Series,
    tau:        int = TAU,
    direction:  str = DIRECTION,
) -> pl.DataFrame:
    """
    Annualised realised volatility over a tau-day window per (FOMC date, tenor).

    Parameters
    ----------
    px_long    : (date, tenor, price) daily price panel
    fomc_dates : pl.Series[Date] — one entry per FOMC meeting
    tau        : window length in trading days (default 21 ≈ monthly option)
    direction  : "forward"  → tau days strictly after fomc_date
                 "backward" → tau days strictly before fomc_date

    Returns
    -------
    rv_longf : (fomc_date, tenor, rv, n_days)
        rv in annualised percentage points to match IV convention.
        Rows with < max(2, tau//2) return days are silently dropped.
    """
    if direction not in ("forward", "backward"):
        raise ValueError("direction must be 'forward' or 'backward'")

    # Daily log returns, computed once per tenor using .over() window
    lr = (
        px_long
        .sort(["tenor", "date"])
        .with_columns(
            (pl.col("price") / pl.col("price").shift(1).over("tenor"))
            .log()
            .alias("log_ret")
        )
        .drop_nulls("log_ret")
    )

    fomc_list  = fomc_dates.cast(pl.Date).to_list()
    min_obs    = max(2, tau // 2)
    records: list[dict] = []

    for tenor_key, grp in lr.group_by("tenor", maintain_order=True):
        tenor_str  = _key(tenor_key)
        grp_sorted = grp.sort("date")
        dates_arr  = grp_sorted["date"].to_list()
        rets_arr   = grp_sorted["log_ret"].to_numpy()

        for fd in fomc_list:
            if direction == "forward":
                idx = [i for i, d in enumerate(dates_arr) if d > fd]
                idx_window = idx[:tau]
            else:
                idx = [i for i, d in enumerate(dates_arr) if d < fd]
                idx_window = idx[-tau:]

            if len(idx_window) < min_obs:
                continue

            r       = rets_arr[idx_window]
            rv_val  = np.sqrt(252 / len(r) * np.sum(r ** 2)) * 100   # annualised pp

            records.append({
                "fomc_date": fd,
                "tenor":     tenor_str,
                "rv":        rv_val,
                "n_days":    len(idx_window),
            })

    rv_longf = (
        pl.DataFrame(records)
        .with_columns(pl.col("fomc_date").cast(pl.Date))
        .sort(["tenor", "fomc_date"])
    )

    print(f"\n  rv_longf : {rv_longf.shape}  ({direction}, tau={tau}d)")
    print(rv_longf.head(3))
    return rv_longf


# ══════════════════════════════════════════════════════════════════════════
# PHASE 3 — DICHOTOMY (VRP)
# ══════════════════════════════════════════════════════════════════════════

def build_vrp(
    iv_long:  pl.DataFrame,
    rv_longf: pl.DataFrame,
) -> pl.DataFrame:
    """
    Join IV and RV; derive VRP, ratio, and under-pricing flag.

    Convention
    ----------
    vrp  = iv - rv
    vrp > 0  →  market over-priced the meeting (normal vol-risk premium)
    vrp < 0  →  market under-priced; RV exceeded IV (thesis condition)

    Returns
    -------
    vrp_pl : (fomc_date, tenor, iv, rv, n_days, vrp, ratio, underpriced)
    """
    vrp_pl = (
        iv_long
        .join(rv_longf, on=["fomc_date", "tenor"], how="inner")
        .with_columns([
            (pl.col("iv") - pl.col("rv")).alias("vrp"),
            (pl.col("rv") / pl.col("iv")).alias("ratio"),
            (pl.col("rv") > pl.col("iv")).alias("underpriced"),
        ])
        .sort(["tenor", "fomc_date"])
    )

    print(f"\n  vrp_pl : {vrp_pl.shape}")
    print(vrp_pl.head(5))

    print("\n  Quick stats by tenor:")
    print(
        vrp_pl
        .group_by("tenor")
        .agg([
            pl.col("vrp").mean().round(3).alias("mean_vrp"),
            pl.col("vrp").median().round(3).alias("med_vrp"),
            pl.col("underpriced").mean().round(3).alias("pct_underpriced"),
            pl.len().alias("n"),
        ])
        .sort("tenor")
    )
    return vrp_pl


# ══════════════════════════════════════════════════════════════════════════
# PHASE 4 — REGIME JOIN & ANALYSIS
# ══════════════════════════════════════════════════════════════════════════

def _newey_west_manual(y: np.ndarray, lags: int) -> tuple[float, float]:
    """
    OLS of y on a constant; Bartlett (Newey-West) HAC variance.
    Returns (t_stat, p_value) via two-sided t distribution.
    """
    from scipy import stats as sp_stats

    n      = len(y)
    ybar   = y.mean()
    resid  = y - ybar
    gamma0 = np.sum(resid ** 2) / n
    nw_var = gamma0

    for h in range(1, lags + 1):
        w       = 1.0 - h / (lags + 1)       # Bartlett kernel
        gamma_h = np.sum(resid[h:] * resid[:-h]) / n
        nw_var += 2.0 * w * gamma_h

    nw_var = max(nw_var, 1e-20)               # prevent divide-by-zero
    se     = np.sqrt(nw_var / n)
    t_stat = ybar / se
    p_val  = 2.0 * float(sp_stats.t.sf(abs(t_stat), df=n - 1))
    return float(t_stat), float(p_val)


def _nw_test(y: np.ndarray, lags: int) -> tuple[float, float]:
    """Newey-West t-test H0: mean(y) == 0. Uses statsmodels when available."""
    if HAS_STATSMODELS and len(y) > lags + 2:
        try:
            res = sm.OLS(y, np.ones(len(y))).fit(
                cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": True}
            )
            return float(res.tvalues[0]), float(res.pvalues[0])
        except Exception:
            pass
    return _newey_west_manual(y, lags)


def summarize_by_regime(
    vrp_pl:          pl.DataFrame,
    regime_pl:       pl.DataFrame,
    regime_date_col: str,
    regime_label_col: str,
    hac_lags:        int = HAC_LAGS,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Join VRP with regime labels; compute per-(regime, tenor) summary stats
    with Newey-West HAC significance tests.

    Parameters
    ----------
    vrp_pl           : output of build_vrp
    regime_pl        : input regime frame
    regime_date_col  : name of the date column in regime_pl
    regime_label_col : name of the label column in regime_pl
    hac_lags         : Bartlett kernel truncation (default = tau)

    Returns
    -------
    vrp_regime_pl    : vrp_pl + regime column
    regime_summary_pl: summary by (regime, tenor) and (regime, 'ALL')
    """
    regime_slim = regime_pl.select([
        pl.col(regime_date_col).alias("fomc_date"),
        pl.col(regime_label_col).alias("regime"),
    ])

    vrp_regime_pl = vrp_pl.join(regime_slim, on="fomc_date", how="left")

    n_miss = vrp_regime_pl["regime"].null_count()
    if n_miss:
        print(f"\n  ⚠ {n_miss} VRP rows have no matching regime — check date alignment.")

    # Filter labelled rows before grouping
    vrp_labelled = vrp_regime_pl.filter(pl.col("regime").is_not_null())

    rows: list[dict] = []

    def _stats(y: np.ndarray, ratios: np.ndarray, regime: str, tenor: str) -> dict:
        t_stat, p_val = (np.nan, np.nan)
        if len(y) >= 2:
            lag = min(hac_lags, len(y) - 1)
            t_stat, p_val = _nw_test(y, lag)
        return {
            "regime":          regime,
            "tenor":           tenor,
            "n":               len(y),
            "mean_vrp":        float(np.mean(y))          if len(y) else np.nan,
            "median_vrp":      float(np.median(y))        if len(y) else np.nan,
            "std_vrp":         float(np.std(y, ddof=1))   if len(y) > 1 else np.nan,
            "pct_underpriced": float(np.mean(y < 0))      if len(y) else np.nan,
            "mean_ratio":      float(np.nanmean(ratios))  if len(ratios) else np.nan,
            "t_stat_nw":       t_stat,
            "p_value_nw":      p_val,
        }

    # (regime, tenor) breakdown
    for keys, grp in vrp_labelled.group_by(["regime", "tenor"], maintain_order=False):
        r, t = keys
        y    = grp["vrp"].drop_nulls().to_numpy()
        ra   = grp["ratio"].drop_nulls().to_numpy()
        rows.append(_stats(y, ra, str(r), str(t)))

    # (regime, ALL) pooled
    for r_key, grp in vrp_labelled.group_by("regime", maintain_order=False):
        r  = str(_key(r_key))
        y  = grp["vrp"].drop_nulls().to_numpy()
        ra = grp["ratio"].drop_nulls().to_numpy()
        rows.append(_stats(y, ra, r, "ALL"))

    regime_summary_pl = (
        pl.DataFrame(rows)
        .with_columns([
            pl.col("n").cast(pl.Int32),
            pl.col(c).cast(pl.Float64)
            for c in ["mean_vrp", "median_vrp", "std_vrp",
                       "pct_underpriced", "mean_ratio", "t_stat_nw", "p_value_nw"]
        ])
        .sort(["regime", "tenor"])
    )

    print("\n  ╔══ SMALL-SAMPLE CAVEAT ════════════════════════════════════╗")
    print("  ║ FOMC regimes typically hold few meetings (some may have  ║")
    print("  ║ N=1-2, e.g. Warsh, Waller). NW t-stats are asymptoti-  ║")
    print("  ║ cally valid but unreliable for N<10. Treat p-values as  ║")
    print("  ║ indicative; use economic magnitude (mean_vrp) as the    ║")
    print("  ║ primary guide. Forward windows also overlap, which NW   ║")
    print("  ║ only partially corrects via HAC.                        ║")
    print("  ╚══════════════════════════════════════════════════════════╝\n")

    print("  Regime Summary (tenor='ALL' = pooled across all tenors):")
    print(regime_summary_pl)

    latest_date  = vrp_regime_pl["fomc_date"].max()
    latest_regimes = (
        vrp_regime_pl
        .filter(pl.col("fomc_date") == latest_date)
        ["regime"]
        .unique()
        .to_list()
    )
    print(f"\n  ★ Most recent FOMC: {latest_date}  |  Regime: {latest_regimes}")

    return vrp_regime_pl, regime_summary_pl


# ══════════════════════════════════════════════════════════════════════════
# PHASE 5 — VISUALS
# ══════════════════════════════════════════════════════════════════════════

def plot_vrp_study(
    vrp_regime_pl:    pl.DataFrame,
    regime_summary_pl: pl.DataFrame,
) -> None:
    """
    Three figures:
      1. VRP time series per tenor with regime colour bands
      2. Mean VRP bar chart by regime × tenor
      3. IV vs RV scatter coloured by regime with 45° line
    """
    tenors  = sorted(t for t in vrp_regime_pl["tenor"].unique().to_list())
    regimes = sorted(vrp_regime_pl["regime"].drop_nulls().unique().to_list())
    colors  = _regime_colors(vrp_regime_pl["regime"])

    # ── Fig 1: VRP time series per tenor ─────────────────────────────────
    n = len(tenors)
    fig1, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, tenor in zip(axes, tenors):
        sub   = vrp_regime_pl.filter(pl.col("tenor") == tenor).sort("fomc_date")
        dates = sub["fomc_date"].to_list()
        vrp   = sub["vrp"].to_list()
        reg   = sub["regime"].to_list()

        ax.axhline(0, color="black", lw=0.9, ls="--", zorder=4)
        ax.plot(dates, vrp, color="#2166ac", lw=1.3, zorder=3)

        # Green above zero, red below
        ax.fill_between(dates, 0, vrp,
                         where=[v is not None and v >= 0 for v in vrp],
                         color="#4dac26", alpha=0.22, label="Over-priced (IV>RV)")
        ax.fill_between(dates, 0, vrp,
                         where=[v is not None and v <  0 for v in vrp],
                         color="#d01c8b", alpha=0.22, label="Under-priced (RV>IV)")

        # Regime background shading
        for i, (d, r) in enumerate(zip(dates, reg)):
            if r is None:
                continue
            x1 = dates[i + 1] if i + 1 < len(dates) else d + timedelta(days=46)
            ax.axvspan(d, x1, alpha=0.09, color=colors.get(r, "#aaaaaa"), zorder=1)

        ax.set_ylabel(f"{tenor}\nVRP (pp)", fontsize=8)
        ax.grid(axis="y", lw=0.4, alpha=0.5)

    # Regime legend on last panel
    regime_patches = [
        mpatches.Patch(color=colors[r], alpha=0.55, label=r) for r in regimes
    ]
    axes[-1].legend(
        handles=regime_patches, title="Regime", fontsize=7,
        title_fontsize=7, loc="lower right", ncol=2,
    )
    axes[-1].set_xlabel("FOMC Date")
    fig1.suptitle("VRP per Tenor Around FOMC Meetings  (VRP = IV − RV)", fontsize=11)
    fig1.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

    # ── Fig 2: Bar chart mean VRP by regime × tenor ───────────────────────
    sum_t = (
        regime_summary_pl
        .filter(pl.col("tenor") != "ALL")
        .sort(["regime", "tenor"])
    )
    regime_list = sorted(sum_t["regime"].unique().to_list())
    tenor_list  = sorted(sum_t["tenor"].unique().to_list())
    x           = np.arange(len(regime_list))
    width       = 0.8 / max(len(tenor_list), 1)
    cmap_t      = plt.cm.get_cmap("plasma", len(tenor_list) + 1)

    fig2, ax2 = plt.subplots(figsize=(max(8, len(regime_list) * 1.5), 5))
    for i, t in enumerate(tenor_list):
        heights = []
        for r in regime_list:
            v = sum_t.filter(
                (pl.col("regime") == r) & (pl.col("tenor") == t)
            )["mean_vrp"].to_list()
            heights.append(v[0] if v else 0.0)
        ax2.bar(x + i * width, heights, width, label=t,
                color=cmap_t(i), alpha=0.85, edgecolor="white", lw=0.5)

    ax2.axhline(0, color="black", lw=0.9, ls="--")
    ax2.set_xticks(x + width * (len(tenor_list) - 1) / 2)
    ax2.set_xticklabels(regime_list, rotation=30, ha="right", fontsize=8)
    ax2.set_ylabel("Mean VRP (pp)")
    ax2.set_title("Mean VRP by Regime × Tenor  (positive = IV over-priced)")
    ax2.legend(title="Tenor", fontsize=8)
    fig2.tight_layout()
    plt.show()

    # ── Fig 3: IV vs RV scatter coloured by regime ───────────────────────
    fig3, ax3 = plt.subplots(figsize=(8, 7))
    for r in regimes:
        sub = vrp_regime_pl.filter(pl.col("regime") == r)
        ax3.scatter(
            sub["iv"].to_list(), sub["rv"].to_list(),
            label=r, color=colors[r], alpha=0.65, s=28, edgecolors="none",
        )

    all_v = np.concatenate([
        vrp_regime_pl["iv"].drop_nulls().to_numpy(),
        vrp_regime_pl["rv"].drop_nulls().to_numpy(),
    ])
    mn, mx = float(np.nanmin(all_v)), float(np.nanmax(all_v))
    pad    = (mx - mn) * 0.05
    ax3.plot([mn - pad, mx + pad], [mn - pad, mx + pad],
             "k--", lw=1.2, label="IV = RV (45°)")
    ax3.set_xlim(mn - pad, mx + pad)
    ax3.set_ylim(mn - pad, mx + pad)
    ax3.set_xlabel("Implied Vol (pp)")
    ax3.set_ylabel("Realized Vol (pp)")
    ax3.set_title("IV vs RV by Regime  —  points below 45° line = under-priced meeting")
    ax3.legend(fontsize=8, title="Regime", title_fontsize=8)
    ax3.grid(lw=0.4, alpha=0.5)
    fig3.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════════
# PHASE 6 — SAVE & PRINT SUMMARY
# ══════════════════════════════════════════════════════════════════════════

def save_and_summarize(
    vrp_regime_pl:    pl.DataFrame,
    regime_summary_pl: pl.DataFrame,
    output_dir:       Path = OUTPUT_DIR,
) -> None:
    """Write parquet files and print a concise analytical summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    vrp_path = output_dir / "vrp_pl.parquet"
    sum_path = output_dir / "regime_summary_pl.parquet"
    vrp_regime_pl.write_parquet(vrp_path)
    regime_summary_pl.write_parquet(sum_path)
    print(f"\n  Saved → {vrp_path}")
    print(f"  Saved → {sum_path}")

    pooled = (
        regime_summary_pl
        .filter(pl.col("tenor") == "ALL")
        .sort("mean_vrp")
    )

    print("\n" + "═" * 62)
    print("  ANALYTICAL SUMMARY — FOMC Implied-vs-Realized VRP Study")
    print("═" * 62)

    under = pooled.filter(pl.col("mean_vrp") < 0)
    over  = pooled.filter(pl.col("mean_vrp") >= 0)

    def _sig(row: dict) -> str:
        p = row.get("p_value_nw", 1.0)
        if p is None or np.isnan(p):
            return ""
        if p < 0.01:
            return " ***"
        if p < 0.05:
            return " **"
        if p < 0.10:
            return " *"
        return ""

    if under.is_empty():
        print("\n  No regime shows negative mean VRP (all tenors pooled).")
    else:
        print("\n  Under-priced regimes  (RV > IV, mean_vrp < 0):")
        for row in under.iter_rows(named=True):
            print(
                f"    {str(row['regime']):22s}  "
                f"mean_vrp={row['mean_vrp']:+6.2f}pp  "
                f"pct_under={row['pct_underpriced']:.0%}  "
                f"n={row['n']}"
                + _sig(row)
            )

    if not over.is_empty():
        print("\n  Over-priced regimes  (IV > RV, normal insurance premium):")
        for row in over.iter_rows(named=True):
            print(
                f"    {str(row['regime']):22s}  "
                f"mean_vrp={row['mean_vrp']:+6.2f}pp  "
                f"pct_under={row['pct_underpriced']:.0%}  "
                f"n={row['n']}"
                + _sig(row)
            )

    print("\n  Significance (NW-HAC): *** p<0.01  ** p<0.05  * p<0.10")

    # Current regime read
    latest      = vrp_regime_pl["fomc_date"].max()
    curr_rows   = vrp_regime_pl.filter(pl.col("fomc_date") == latest)
    if not curr_rows.is_empty():
        curr_regime = curr_rows["regime"][0]
        curr_vrp    = curr_rows["vrp"].mean()
        print(f"\n  ★ Current FOMC ({latest})  regime='{curr_regime}'")
        print(f"    Latest-meeting VRP (avg across tenors): {curr_vrp:+.2f}pp")
        curr_sum = pooled.filter(pl.col("regime") == curr_regime)
        if not curr_sum.is_empty():
            r = curr_sum.row(0, named=True)
            print(
                f"    Full-regime avg VRP: {r['mean_vrp']:+.2f}pp  "
                f"under-priced {r['pct_underpriced']:.0%} of meetings"
            )

    print("\n  Note: 21-day forward windows overlap between adjacent meetings;")
    print("  NW-HAC corrects for overlap-induced autocorrelation but remains")
    print("  approximate — weight economic magnitude over p-values here.")
    print("═" * 62)


# ══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════

def run_fomc_vol_study(
    impl_pl:      pl.DataFrame,
    rv_pl:        pl.DataFrame,
    regime_pl:    pl.DataFrame,
    px_daily_pl:  Optional[pl.DataFrame] = None,
    tau:          int  = TAU,
    direction:    str  = DIRECTION,
    hac_lags:     int  = HAC_LAGS,
    output_dir:   Path = OUTPUT_DIR,
) -> dict:
    """
    End-to-end pipeline: Phases 0–6.

    Parameters
    ----------
    impl_pl, rv_pl, regime_pl : raw Bloomberg frames
    px_daily_pl : supply this (from the BQL snippet) if Phase 0 finds that
                  rv_pl has only one price per FOMC meeting
    tau         : RV window in trading days
    direction   : "forward" (post-meeting) | "backward" (pre-meeting)
    hac_lags    : Newey-West kernel truncation
    output_dir  : directory for parquet output

    Returns
    -------
    dict with keys: iv_long, px_long, rv_longf, vrp_pl,
                    vrp_regime_pl, regime_summary_pl
    """
    print("\n" + "█" * 62)
    print("  PHASE 0 — INSPECT & VALIDATE")
    print("█" * 62)
    impl_pl, rv_pl, regime_pl, _px, tenors, rdc, rlc = inspect_and_validate(
        impl_pl, rv_pl, regime_pl, tau=tau
    )

    daily_source = px_daily_pl if px_daily_pl is not None else _px
    if daily_source is None:
        print("\n  Pipeline halted — re-pull daily prices with the BQL snippet above,")
        print("  then call: run_fomc_vol_study(..., px_daily_pl=px_daily_pl)")
        return {}

    print("\n" + "█" * 62)
    print("  PHASE 1 — TIDY TO LONG")
    print("█" * 62)
    iv_long, px_long = tidy_to_long(impl_pl, daily_source, tenors)

    print("\n" + "█" * 62)
    print(f"  PHASE 2 — REALIZED VOLATILITY  (tau={tau}d, {direction})")
    print("█" * 62)
    rv_longf = compute_rv(px_long, impl_pl["fomc_date"], tau=tau, direction=direction)

    print("\n" + "█" * 62)
    print("  PHASE 3 — DICHOTOMY (VRP = IV − RV)")
    print("█" * 62)
    vrp_pl = build_vrp(iv_long, rv_longf)

    print("\n" + "█" * 62)
    print("  PHASE 4 — REGIME JOIN & ANALYSIS")
    print("█" * 62)
    vrp_regime_pl, regime_summary_pl = summarize_by_regime(
        vrp_pl, regime_pl,
        regime_date_col=rdc, regime_label_col=rlc,
        hac_lags=hac_lags,
    )

    print("\n" + "█" * 62)
    print("  PHASE 5 — VISUALS")
    print("█" * 62)
    plot_vrp_study(vrp_regime_pl, regime_summary_pl)

    print("\n" + "█" * 62)
    print("  PHASE 6 — SAVE & SUMMARY")
    print("█" * 62)
    save_and_summarize(vrp_regime_pl, regime_summary_pl, output_dir=output_dir)

    return {
        "iv_long":            iv_long,
        "px_long":            px_long,
        "rv_longf":           rv_longf,
        "vrp_pl":             vrp_pl,
        "vrp_regime_pl":      vrp_regime_pl,
        "regime_summary_pl":  regime_summary_pl,
    }


# ── ENTRY POINT ───────────────────────────────────────────────────────────
# Uncomment and run.  Supply px_daily_pl only after the BQL re-pull if needed.
#
# results = run_fomc_vol_study(
#     impl_pl     = impl_pl,
#     rv_pl       = rv_pl,
#     regime_pl   = regime_pl,
#     # px_daily_pl = px_daily_pl,   # ← from BQL re-pull snippet if rv_pl was sparse
#     tau         = 21,
#     direction   = "forward",
#     hac_lags    = 21,
#     output_dir  = Path("."),
# )
#
# # Access individual outputs:
# # results["vrp_regime_pl"]
# # results["regime_summary_pl"]
