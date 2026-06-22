# ============================================================
# FOMC Vol Study — Two-Part Pipeline
# Bloomberg BQuant / JupyterLab
# ── PART A: Unit Cleaning & Scale Normalisation  (run FIRST)
# ── PART B: Implied-vs-Realized VRP Study        (run after Part A)
# ============================================================
#
# Typical workflow
# ----------------
#   cleaning = run_cleaning_pipeline(impl_pl, rv_pl)
#
#   results  = run_fomc_vol_study(
#       impl_pl     = cleaning["impl_clean"],
#       rv_pl       = rv_pl,
#       regime_pl   = regime_pl,
#       px_daily_pl = cleaning.get("px_long"),
#   )
#
# Split at the ── CELL BREAK ── markers to paste into separate cells.
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

# ── Global params ──────────────────────────────────────────────────────────
TAU        = 21          # RV window in trading days
DIRECTION  = "forward"   # "forward" | "backward"
HAC_LAGS   = TAU         # Newey-West Bartlett truncation
IV_CAP     = 40.0        # cells above this (after unit fix) are bad ticks
OUTPUT_DIR = Path(".")

_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


# ──────────────────────────────────────────────────────────────────────────
# Shared helpers (used by both Part A and Part B)
# ──────────────────────────────────────────────────────────────────────────

def _unpivot(
    df: pl.DataFrame,
    on: list[str],
    index: str,
    variable_name: str,
    value_name: str,
) -> pl.DataFrame:
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


def _fix_date_col(df: pl.DataFrame, new_name: str) -> pl.DataFrame:
    """Rename first column to new_name and cast to pl.Date."""
    old = df.columns[0]
    df  = df.rename({old: new_name})
    if df[new_name].dtype is not pl.Date:
        try:
            df = df.with_columns(pl.col(new_name).cast(pl.Date))
        except Exception:
            df = df.with_columns(pl.col(new_name).str.to_date(strict=False))
    return df


# ── CELL BREAK ────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════
# PART A — UNIT CLEANING & SCALE NORMALISATION
# ══════════════════════════════════════════════════════════════════════════


# ── Step 0 ────────────────────────────────────────────────────────────────

def inspect_raw(
    impl_pl: pl.DataFrame,
    rv_pl:   pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, list[str]]:
    """
    Step 0: Print schema / shape / head / null counts for both frames.
    Rename first columns to fomc_date; cast to pl.Date.
    Cast all tenor columns to Float64 (strict=False so bad strings → null).

    Returns
    -------
    impl_pl : first col = fomc_date (Date); tenor cols = Float64
    rv_pl   : same
    tenors  : tenor labels read from impl_pl (everything after fomc_date)
    """
    def _report(name: str, df: pl.DataFrame) -> None:
        print(f"\n{'═'*60}\n  {name}   shape={df.shape}\n{'═'*60}")
        print("  Schema:")
        for col, dtype in df.schema.items():
            print(f"    {col:30s}  {dtype}")
        print(f"\n  Head (3 rows):\n{df.head(3)}")
        nulls = {c: df[c].null_count() for c in df.columns if df[c].null_count()}
        print(f"  Null counts: {nulls if nulls else 'none'}")

    impl_pl = _fix_date_col(impl_pl, "fomc_date")
    rv_pl   = _fix_date_col(rv_pl,   "fomc_date")

    tenors: list[str] = [c for c in impl_pl.columns if c != "fomc_date"]
    print(f"\n  Detected tenors: {tenors}")

    # Cast tenor cols to Float64 — bad strings become null (strict=False)
    impl_pl = impl_pl.with_columns(
        [pl.col(t).cast(pl.Float64, strict=False) for t in tenors]
    )
    rv_pl = rv_pl.with_columns(
        [pl.col(t).cast(pl.Float64, strict=False) for t in tenors
         if t in rv_pl.columns]
    )

    _report("impl_pl", impl_pl)
    _report("rv_pl",   rv_pl)

    return impl_pl, rv_pl, tenors


# ── Step 1 ────────────────────────────────────────────────────────────────

def clean_iv(
    impl_pl: pl.DataFrame,
    cap:     float = IV_CAP,
) -> pl.DataFrame:
    """
    Step 1: Detect the storage unit of HIST_CALL_IMP_VOL and normalise to
    annualised price-vol percentage points (~3–15 pp for Treasury options).

    Unit detection (applied once, globally, across all tenors):
      global_median > 100  →  divide by 100   (e.g. 750 bps → 7.50 pp)
      global_median <   1  →  multiply by 100  (e.g. 0.075 → 7.5 pp)
      else                 →  leave as-is      (already in pp)

    Any cell still above `cap` after scaling is treated as a data error
    (stale tick, roll artefact, etc.) and nulled.

    Parameters
    ----------
    impl_pl : output of inspect_raw — fomc_date + tenor cols as Float64
    cap     : ceiling for valid implied vol (pp); default 40

    Returns
    -------
    impl_clean : same shape, tenor values in annualised pp; bad ticks nulled
    """
    tenors = [c for c in impl_pl.columns if c != "fomc_date"]

    # Stack all tenor values to compute a single representative median
    stacked = (
        _unpivot(impl_pl, on=tenors, index="fomc_date",
                 variable_name="tenor", value_name="iv")
        ["iv"]
        .drop_nulls()
    )
    raw_median = float(stacked.median())

    if raw_median > 100:
        scale   = 1 / 100
        reason  = f"median {raw_median:.1f} > 100 → ÷100 (likely basis-points storage)"
    elif raw_median < 1:
        scale   = 100.0
        reason  = f"median {raw_median:.4f} < 1 → ×100 (likely decimal storage)"
    else:
        scale   = 1.0
        reason  = f"median {raw_median:.2f} already in pp range → no rescaling"

    print(f"\n  [clean_iv] raw global median = {raw_median:.4f}")
    print(f"  [clean_iv] scale factor      = {scale}  ({reason})")

    impl_scaled = impl_pl.with_columns(
        [pl.col(t) * scale for t in tenors]
    )

    # Cap: null anything above cap — count per tenor for auditability
    suspect_counts: dict[str, int] = {}
    for t in tenors:
        n = int(impl_scaled.filter(pl.col(t) > cap)[t].count())
        suspect_counts[t] = n

    total_suspects = sum(suspect_counts.values())
    if total_suspects:
        print(f"\n  [clean_iv] cells > {cap}pp (bad ticks) per tenor:")
        for t, n in suspect_counts.items():
            if n:
                print(f"    {t}: {n}")
        print(f"  Total nulled: {total_suspects}")
    else:
        print(f"  [clean_iv] no cells exceed cap={cap}pp after scaling — data looks clean")

    impl_clean = impl_scaled.with_columns(
        [
            pl.when(pl.col(t) > cap).then(None).otherwise(pl.col(t)).alias(t)
            for t in tenors
        ]
    )

    print(f"\n  impl_clean shape: {impl_clean.shape}")
    print(impl_clean.head(3))
    return impl_clean


# ── Step 2 ────────────────────────────────────────────────────────────────

def check_rv_source(
    rv_pl:   pl.DataFrame,
    impl_pl: pl.DataFrame,
    tenors:  list[str],
) -> tuple[bool, Optional[pl.DataFrame]]:
    """
    Step 2: Decide if rv_pl is a usable daily price panel (b) or a
    one-price-per-FOMC-meeting stub (a).

    Detection heuristic:
      - row count > 3× number of FOMC meetings, OR
      - median calendar gap between consecutive dates ≤ 7 days
      → treat as daily.

    Parameters
    ----------
    rv_pl   : from inspect_raw — fomc_date + tenor cols
    impl_pl : from inspect_raw — used for FOMC count and date range
    tenors  : from inspect_raw

    Returns
    -------
    is_daily : True if rv_pl is a daily panel
    px_long  : (date, tenor, price) sorted [tenor, date] if is_daily else None.
               Prints a BQL re-pull snippet and returns None if not daily.
    """
    n_rv    = rv_pl.shape[0]
    n_fomc  = impl_pl.shape[0]

    # Cast Date → Int32 (days since epoch) for arithmetic; diff → calendar days
    gaps     = rv_pl["fomc_date"].sort().cast(pl.Int32).diff().drop_nulls()
    med_gap  = float(gaps.median() or 0)

    is_daily = (n_rv > n_fomc * 3) or (0 < med_gap <= 7)

    if is_daily:
        print(f"\n  [check_rv_source] ✓ {n_rv} rows, median gap {med_gap:.0f}d "
              "→ daily price panel")
        rv_daily  = rv_pl.rename({"fomc_date": "date"})
        wide_cols = [c for c in rv_daily.columns if c in tenors]
        px_long   = (
            _unpivot(rv_daily, on=wide_cols, index="date",
                     variable_name="tenor", value_name="price")
            .drop_nulls("price")
            .sort(["tenor", "date"])
        )
        print(f"  px_long: {px_long.shape}")
        return True, px_long

    # ── sparse: one price per meeting → cannot compute RV ────────────────
    print(f"\n  [check_rv_source] ✗ {n_rv} rows ≈ {n_fomc} FOMC meetings "
          f"(median gap {med_gap:.0f}d).")
    print("  RV CANNOT be computed from one price per meeting.\n")

    min_d = (impl_pl["fomc_date"].min() - timedelta(days=10)).strftime("%Y-%m-%d")
    max_d = (impl_pl["fomc_date"].max() + timedelta(days=40)).strftime("%Y-%m-%d")

    _bbg = {
        "2Y": "TU1 Comdty", "5Y": "FV1 Comdty", "10Y": "TY1 Comdty",
        "20Y": "US1 Comdty", "30Y": "WN1 Comdty",
    }
    tickers_str  = ", ".join(f'"{_bbg.get(t, t + " Comdty")}"' for t in tenors)
    tenor_map_str = (
        "{" + ", ".join(f'"{_bbg.get(t, t)}": "{t}"' for t in tenors) + "}"
    )

    print("=" * 60)
    print("  BQL SNIPPET — run in a new cell, then re-call run_cleaning_pipeline")
    print("  with px_daily_pl=px_daily_pl")
    print("  NOTE: TU1/FV1/TY1/WN1 are generic front futures; flag any roll date")
    print("  that falls inside a 21-day forward window (Mar/Jun/Sep/Dec expiries).")
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

raw = bql.combined_df(resp).reset_index()  # columns: DATE, ID, px

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
    return False, None


# ── Step 3 ────────────────────────────────────────────────────────────────

def compute_rv(
    px_long:    pl.DataFrame,
    fomc_dates: pl.Series,
    tau:        int = TAU,
    direction:  str = DIRECTION,
) -> pl.DataFrame:
    """
    Step 3 / Phase 2: Annualised realised volatility over a tau-day window
    per (FOMC date, tenor).  Defined once here; called by both Part A and
    Part B.

    Parameters
    ----------
    px_long    : (date, tenor, price) — daily price panel
    fomc_dates : pl.Series[Date] — one entry per FOMC meeting
    tau        : window length in trading days (default 21 ≈ monthly option)
    direction  : "forward"  → tau days strictly after fomc_date
                 "backward" → tau days strictly before fomc_date

    Returns
    -------
    rv_longf : (fomc_date, tenor, rv, n_days)
        rv in annualised percentage points (matches IV convention).
        Rows with fewer than 60% of tau observations are returned as null
        and then dropped, rather than emitting a noisy estimate.
    """
    if direction not in ("forward", "backward"):
        raise ValueError("direction must be 'forward' or 'backward'")

    min_obs = max(2, int(np.ceil(tau * 0.6)))   # 60% of tau

    # Daily log returns, partitioned by tenor so no cross-tenor contamination
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

    fomc_list = fomc_dates.cast(pl.Date).to_list()
    records: list[dict] = []

    for tenor_key, grp in lr.group_by("tenor", maintain_order=True):
        tenor_str  = _key(tenor_key)
        grp_sorted = grp.sort("date")
        dates_arr  = grp_sorted["date"].to_list()
        rets_arr   = grp_sorted["log_ret"].to_numpy()

        for fd in fomc_list:
            if direction == "forward":
                idx        = [i for i, d in enumerate(dates_arr) if d > fd]
                idx_window = idx[:tau]
            else:
                idx        = [i for i, d in enumerate(dates_arr) if d < fd]
                idx_window = idx[-tau:]

            if len(idx_window) < min_obs:
                continue

            r      = rets_arr[idx_window]
            rv_val = np.sqrt(252 / len(r) * np.sum(r ** 2)) * 100   # annualised pp

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

    print(f"\n  rv_longf: {rv_longf.shape}  ({direction}, tau={tau}d, min_obs={min_obs})")
    print(rv_longf.head(3))
    return rv_longf


# ── Step 4 ────────────────────────────────────────────────────────────────

def sanity_check_scale(
    iv_long:    pl.DataFrame,
    rv_longf:   pl.DataFrame,
    warn_ratio: tuple[float, float] = (0.2, 5.0),
) -> None:
    """
    Step 4: Compare median(iv) and median(rv).  Both should be in the same
    order of magnitude (annualised pp) if the unit fix in Step 1 was correct.

    Warns loudly — and prints the raw IV median — if the ratio falls outside
    warn_ratio, so the user can recheck Step 1 without proceeding silently
    into a nonsense VRP study.

    Parameters
    ----------
    iv_long    : (fomc_date, tenor, iv)   — output of clean_iv unpivoted
    rv_longf   : (fomc_date, tenor, rv)   — output of compute_rv
    warn_ratio : acceptable (lo, hi) band for median_rv / median_iv
    """
    med_iv = float(iv_long["iv"].drop_nulls().median())
    med_rv = float(rv_longf["rv"].drop_nulls().median())

    print(f"\n  [sanity_check_scale]")
    print(f"    median IV : {med_iv:.3f} pp")
    print(f"    median RV : {med_rv:.3f} pp")

    if med_iv == 0:
        print("  ⚠ WARNING: median IV is zero — check impl_clean for data issues.")
        return

    ratio = med_rv / med_iv
    print(f"    ratio RV/IV : {ratio:.3f}")

    lo, hi = warn_ratio
    if lo <= ratio <= hi:
        print(f"  ✓ Ratio {ratio:.3f} is within [{lo}, {hi}] — scales match.")
    else:
        print(f"\n  ✗ WARNING: ratio {ratio:.3f} is OUTSIDE [{lo}, {hi}].")
        print("    This strongly suggests Step 1 unit detection picked the wrong branch.")
        print(f"    Raw IV global median before scaling: re-run inspect_raw and clean_iv")
        print(f"    and check the printed 'raw global median' line.")
        print("    Do NOT proceed to the VRP study until scales are reconciled.")


# ── Part A orchestrator ───────────────────────────────────────────────────

def run_cleaning_pipeline(
    impl_pl:     pl.DataFrame,
    rv_pl:       pl.DataFrame,
    px_daily_pl: Optional[pl.DataFrame] = None,
    tau:         int   = TAU,
    cap:         float = IV_CAP,
) -> dict:
    """
    Run Steps 0–4 and return cleaned outputs ready for the VRP study.

    Supply px_daily_pl if rv_pl was detected as sparse (one price per FOMC
    meeting) — paste it from the BQL snippet printed in Step 2 and re-run.

    Returns
    -------
    dict with keys:
        impl_clean  — scaled / capped implied vol frame (wide)
        iv_long     — (fomc_date, tenor, iv) long format
        rv_longf    — (fomc_date, tenor, rv, n_days)
        px_long     — (date, tenor, price) daily panel
        tenors      — list of tenor strings
    """
    print("\n" + "▓" * 62)
    print("  PART A — STEP 0: INSPECT RAW FRAMES")
    print("▓" * 62)
    impl_pl, rv_pl, tenors = inspect_raw(impl_pl, rv_pl)

    print("\n" + "▓" * 62)
    print("  PART A — STEP 1: CLEAN IMPLIED VOL")
    print("▓" * 62)
    impl_clean = clean_iv(impl_pl, cap=cap)

    print("\n" + "▓" * 62)
    print("  PART A — STEP 2: RV SOURCE CHECK")
    print("▓" * 62)
    if px_daily_pl is not None:
        print("  px_daily_pl supplied by caller — skipping sparse-detection.")
        # Normalise to long format if needed
        date_col = px_daily_pl.columns[0]
        if "tenor" in px_daily_pl.columns and "price" in px_daily_pl.columns:
            px_long = (
                px_daily_pl
                .rename({date_col: "date"})
                .filter(pl.col("tenor").is_in(tenors))
                .select(["date", "tenor", "price"])
                .drop_nulls("price")
                .sort(["tenor", "date"])
            )
        else:
            wide_cols = [c for c in px_daily_pl.columns if c in tenors]
            px_long = (
                _unpivot(
                    px_daily_pl.rename({date_col: "date"}),
                    on=wide_cols, index="date",
                    variable_name="tenor", value_name="price",
                )
                .drop_nulls("price")
                .sort(["tenor", "date"])
            )
        is_daily = True
    else:
        is_daily, px_long = check_rv_source(rv_pl, impl_pl, tenors)

    if not is_daily or px_long is None:
        print("\n  Part A halted — re-pull daily prices with the BQL snippet above,")
        print("  then re-call: run_cleaning_pipeline(..., px_daily_pl=px_daily_pl)")
        return {}

    print("\n" + "▓" * 62)
    print(f"  PART A — STEP 3: COMPUTE RV  (tau={tau}d, {DIRECTION})")
    print("▓" * 62)
    rv_longf = compute_rv(px_long, impl_clean["fomc_date"], tau=tau, direction=DIRECTION)

    # Long-format IV for sanity check and downstream use
    iv_long = (
        _unpivot(impl_clean, on=tenors, index="fomc_date",
                 variable_name="tenor", value_name="iv")
        .drop_nulls("iv")
        .sort(["tenor", "fomc_date"])
    )

    print("\n" + "▓" * 62)
    print("  PART A — STEP 4: SCALE SANITY CHECK")
    print("▓" * 62)
    sanity_check_scale(iv_long, rv_longf)

    # ── Change summary ────────────────────────────────────────────────────
    orig_nulls  = sum(impl_pl[t].null_count() for t in tenors)
    clean_nulls = sum(impl_clean[t].null_count() for t in tenors)
    new_nulls   = clean_nulls - orig_nulls

    print("\n" + "═" * 62)
    print("  PART A SUMMARY")
    print("═" * 62)
    print(f"  Tenors        : {tenors}")
    print(f"  IV scale      : see Step 1 output above")
    print(f"  Cells nulled  : {new_nulls} (bad ticks above cap={cap}pp)")
    print(f"  impl_clean    : {impl_clean.shape}")
    print(f"  iv_long       : {iv_long.shape}")
    print(f"  rv_longf      : {rv_longf.shape}")
    print(f"  px_long       : {px_long.shape}")
    print("═" * 62)

    return {
        "impl_clean": impl_clean,
        "iv_long":    iv_long,
        "rv_longf":   rv_longf,
        "px_long":    px_long,
        "tenors":     tenors,
    }


# ── CELL BREAK ────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════
# PART B — IMPLIED-vs-REALIZED VRP STUDY
# ══════════════════════════════════════════════════════════════════════════


# ── Phase 0 ───────────────────────────────────────────────────────────────

def inspect_and_validate(
    impl_pl:   pl.DataFrame,
    rv_pl:     pl.DataFrame,
    regime_pl: pl.DataFrame,
) -> tuple[
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    Optional[pl.DataFrame],
    list[str],
    str,
    str,
]:
    """
    Phase 0: Inspect all three frames, rename/cast date columns, detect
    regime column names, and check whether rv_pl is a usable daily panel.

    When called after run_cleaning_pipeline, impl_pl will already be clean
    and rv_pl sparse-check will be bypassed by passing px_daily_pl directly
    to run_fomc_vol_study.  This function is retained so Part B can also be
    run standalone on already-clean data.

    Returns
    -------
    impl_pl, rv_pl, regime_pl  — cast / renamed
    px_daily_pl                — daily price frame or None
    tenors                     — tenor label list
    regime_date_col            — name of the date col in regime_pl
    regime_label_col           — name of the label col in regime_pl
    """
    def _report(name: str, df: pl.DataFrame) -> None:
        print(f"\n{'═'*60}\n  {name}   shape={df.shape}\n{'═'*60}")
        print("  Schema:")
        for col, dtype in df.schema.items():
            print(f"    {col:30s}  {dtype}")
        print(f"\n  Head (3 rows):\n{df.head(3)}")
        nulls = {c: df[c].null_count() for c in df.columns if df[c].null_count()}
        print(f"  Null counts: {nulls if nulls else 'none'}")

    _report("impl_pl",   impl_pl)
    _report("rv_pl",     rv_pl)
    _report("regime_pl", regime_pl)

    tenors: list[str] = impl_pl.columns[1:]
    print(f"\n  Detected tenors: {tenors}")

    impl_pl = _fix_date_col(impl_pl, "fomc_date")
    rv_pl   = _fix_date_col(rv_pl,   "fomc_date")

    # ── regime column detection ───────────────────────────────────────────
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
    print(f"  regime_date_col='{regime_date_col}'  "
          f"regime_label_col='{regime_label_col}'")

    if regime_pl[regime_date_col].dtype is not pl.Date:
        try:
            regime_pl = regime_pl.with_columns(
                pl.col(regime_date_col).cast(pl.Date)
            )
        except Exception:
            regime_pl = regime_pl.with_columns(
                pl.col(regime_date_col).str.to_date(strict=False)
            )

    # ── rv_pl daily / sparse check ────────────────────────────────────────
    n_rv    = rv_pl.shape[0]
    n_fomc  = impl_pl.shape[0]
    gaps    = rv_pl["fomc_date"].sort().cast(pl.Int32).diff().drop_nulls()
    med_gap = float(gaps.median() or 0)

    is_daily = (n_rv > n_fomc * 3) or (0 < med_gap <= 7)

    if is_daily:
        print(f"\n  ✓ rv_pl: {n_rv} rows, median gap {med_gap:.0f}d → daily panel.")
        rv_pl       = rv_pl.rename({"fomc_date": "date"})
        px_daily_pl: Optional[pl.DataFrame] = rv_pl
    else:
        print(f"\n  ✗ rv_pl: {n_rv} rows ≈ {n_fomc} meetings (gap {med_gap:.0f}d).")
        print("  Pass px_daily_pl= to run_fomc_vol_study, or run Part A first.")
        px_daily_pl = None

    return (
        impl_pl, rv_pl, regime_pl,
        px_daily_pl, tenors, regime_date_col, regime_label_col,
    )


# ── Phase 1 ───────────────────────────────────────────────────────────────

def tidy_to_long(
    impl_pl:     pl.DataFrame,
    px_daily_pl: pl.DataFrame,
    tenors:      list[str],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Phase 1: Unpivot implied vol (wide→long) and standardise the daily
    price panel to long format.

    Parameters
    ----------
    impl_pl      : (fomc_date, *tenors) — cleaned implied vols
    px_daily_pl  : daily prices, wide (date, *tenors) or long (date, tenor, price)
    tenors       : tenor labels

    Returns
    -------
    iv_long : (fomc_date, tenor, iv)
    px_long : (date, tenor, price) sorted [tenor, date]
    """
    iv_long = (
        _unpivot(impl_pl, on=tenors, index="fomc_date",
                 variable_name="tenor", value_name="iv")
        .drop_nulls("iv")
        .sort(["tenor", "fomc_date"])
    )

    date_col = px_daily_pl.columns[0]
    if "tenor" in px_daily_pl.columns and "price" in px_daily_pl.columns:
        px_long = (
            px_daily_pl
            .rename({date_col: "date"})
            .filter(pl.col("tenor").is_in(tenors))
            .select(["date", "tenor", "price"])
            .drop_nulls("price")
            .sort(["tenor", "date"])
        )
    else:
        wide_cols = [c for c in px_daily_pl.columns if c in tenors]
        px_long = (
            _unpivot(
                px_daily_pl.rename({date_col: "date"}),
                on=wide_cols, index="date",
                variable_name="tenor", value_name="price",
            )
            .drop_nulls("price")
            .sort(["tenor", "date"])
        )

    print(f"\n  iv_long : {iv_long.shape}")
    print(iv_long.head(3))
    print(f"\n  px_long : {px_long.shape}")
    print(px_long.head(3))

    return iv_long, px_long


# ── Phase 2 ───────────────────────────────────────────────────────────────
# compute_rv is defined in Part A (Step 3) above — no redefinition needed.
# Call: rv_longf = compute_rv(px_long, fomc_dates, tau=tau, direction=direction)


# ── Phase 3 ───────────────────────────────────────────────────────────────

def build_vrp(
    iv_long:  pl.DataFrame,
    rv_longf: pl.DataFrame,
) -> pl.DataFrame:
    """
    Phase 3: Join IV and RV; derive VRP, ratio, and under-pricing flag.

    Convention
    ----------
    vrp  = iv − rv
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


# ── Phase 4 ───────────────────────────────────────────────────────────────

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
        w       = 1.0 - h / (lags + 1)
        gamma_h = np.sum(resid[h:] * resid[:-h]) / n
        nw_var += 2.0 * w * gamma_h

    nw_var = max(nw_var, 1e-20)
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
    vrp_pl:           pl.DataFrame,
    regime_pl:        pl.DataFrame,
    regime_date_col:  str,
    regime_label_col: str,
    hac_lags:         int = HAC_LAGS,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Phase 4: Join VRP with regime labels; compute per-(regime, tenor)
    summary stats with Newey-West HAC significance tests.

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

    vrp_labelled = vrp_regime_pl.filter(pl.col("regime").is_not_null())
    rows: list[dict] = []

    def _stats(y: np.ndarray, ratios: np.ndarray,
               regime: str, tenor: str) -> dict:
        t_stat, p_val = np.nan, np.nan
        if len(y) >= 2:
            lag = min(hac_lags, len(y) - 1)
            t_stat, p_val = _nw_test(y, lag)
        return {
            "regime":          regime,
            "tenor":           tenor,
            "n":               len(y),
            "mean_vrp":        float(np.mean(y))         if len(y) else np.nan,
            "median_vrp":      float(np.median(y))       if len(y) else np.nan,
            "std_vrp":         float(np.std(y, ddof=1))  if len(y) > 1 else np.nan,
            "pct_underpriced": float(np.mean(y < 0))     if len(y) else np.nan,
            "mean_ratio":      float(np.nanmean(ratios)) if len(ratios) else np.nan,
            "t_stat_nw":       t_stat,
            "p_value_nw":      p_val,
        }

    for keys, grp in vrp_labelled.group_by(["regime", "tenor"], maintain_order=False):
        r, t = keys
        rows.append(_stats(
            grp["vrp"].drop_nulls().to_numpy(),
            grp["ratio"].drop_nulls().to_numpy(),
            str(r), str(t),
        ))

    for r_key, grp in vrp_labelled.group_by("regime", maintain_order=False):
        r = str(_key(r_key))
        rows.append(_stats(
            grp["vrp"].drop_nulls().to_numpy(),
            grp["ratio"].drop_nulls().to_numpy(),
            r, "ALL",
        ))

    regime_summary_pl = (
        pl.DataFrame(rows)
        .with_columns([
            pl.col("n").cast(pl.Int32),
            *[pl.col(c).cast(pl.Float64)
              for c in ["mean_vrp", "median_vrp", "std_vrp",
                        "pct_underpriced", "mean_ratio", "t_stat_nw", "p_value_nw"]],
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

    latest_date    = vrp_regime_pl["fomc_date"].max()
    latest_regimes = (
        vrp_regime_pl
        .filter(pl.col("fomc_date") == latest_date)
        ["regime"].unique().to_list()
    )
    print(f"\n  ★ Most recent FOMC: {latest_date}  |  Regime: {latest_regimes}")

    return vrp_regime_pl, regime_summary_pl


# ── Phase 5 ───────────────────────────────────────────────────────────────

def plot_vrp_study(
    vrp_regime_pl:     pl.DataFrame,
    regime_summary_pl: pl.DataFrame,
) -> None:
    """
    Phase 5: Three figures:
      1. VRP time series per tenor with regime colour bands
      2. Mean VRP bar chart by regime × tenor
      3. IV vs RV scatter coloured by regime with 45° line
    """
    tenors  = sorted(vrp_regime_pl["tenor"].unique().to_list())
    regimes = sorted(vrp_regime_pl["regime"].drop_nulls().unique().to_list())
    colors  = _regime_colors(vrp_regime_pl["regime"])

    # Fig 1 — VRP time series
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
        ax.fill_between(dates, 0, vrp,
                         where=[v is not None and v >= 0 for v in vrp],
                         color="#4dac26", alpha=0.22, label="Over-priced (IV>RV)")
        ax.fill_between(dates, 0, vrp,
                         where=[v is not None and v <  0 for v in vrp],
                         color="#d01c8b", alpha=0.22, label="Under-priced (RV>IV)")

        for i, (d, r) in enumerate(zip(dates, reg)):
            if r is None:
                continue
            x1 = dates[i + 1] if i + 1 < len(dates) else d + timedelta(days=46)
            ax.axvspan(d, x1, alpha=0.09, color=colors.get(r, "#aaaaaa"), zorder=1)

        ax.set_ylabel(f"{tenor}\nVRP (pp)", fontsize=8)
        ax.grid(axis="y", lw=0.4, alpha=0.5)

    axes[-1].legend(
        handles=[mpatches.Patch(color=colors[r], alpha=0.55, label=r) for r in regimes],
        title="Regime", fontsize=7, title_fontsize=7, loc="lower right", ncol=2,
    )
    axes[-1].set_xlabel("FOMC Date")
    fig1.suptitle("VRP per Tenor Around FOMC Meetings  (VRP = IV − RV)", fontsize=11)
    fig1.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

    # Fig 2 — bar chart
    sum_t       = regime_summary_pl.filter(pl.col("tenor") != "ALL").sort(["regime", "tenor"])
    regime_list = sorted(sum_t["regime"].unique().to_list())
    tenor_list  = sorted(sum_t["tenor"].unique().to_list())
    x           = np.arange(len(regime_list))
    width       = 0.8 / max(len(tenor_list), 1)
    cmap_t      = plt.cm.get_cmap("plasma", len(tenor_list) + 1)

    fig2, ax2 = plt.subplots(figsize=(max(8, len(regime_list) * 1.5), 5))
    for i, t in enumerate(tenor_list):
        heights = [
            (sum_t.filter((pl.col("regime") == r) & (pl.col("tenor") == t))
             ["mean_vrp"].to_list() or [0.0])[0]
            for r in regime_list
        ]
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

    # Fig 3 — IV vs RV scatter
    fig3, ax3 = plt.subplots(figsize=(8, 7))
    for r in regimes:
        sub = vrp_regime_pl.filter(pl.col("regime") == r)
        ax3.scatter(sub["iv"].to_list(), sub["rv"].to_list(),
                    label=r, color=colors[r], alpha=0.65, s=28, edgecolors="none")

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
    ax3.set_title("IV vs RV by Regime  —  points below 45° = under-priced meeting")
    ax3.legend(fontsize=8, title="Regime", title_fontsize=8)
    ax3.grid(lw=0.4, alpha=0.5)
    fig3.tight_layout()
    plt.show()


# ── Phase 6 ───────────────────────────────────────────────────────────────

def save_and_summarize(
    vrp_regime_pl:     pl.DataFrame,
    regime_summary_pl: pl.DataFrame,
    output_dir:        Path = OUTPUT_DIR,
) -> None:
    """Phase 6: Write parquet files and print a concise analytical summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    vrp_regime_pl.write_parquet(output_dir / "vrp_pl.parquet")
    regime_summary_pl.write_parquet(output_dir / "regime_summary_pl.parquet")
    print(f"\n  Saved → {output_dir / 'vrp_pl.parquet'}")
    print(f"  Saved → {output_dir / 'regime_summary_pl.parquet'}")

    pooled = (
        regime_summary_pl
        .filter(pl.col("tenor") == "ALL")
        .sort("mean_vrp")
    )

    print("\n" + "═" * 62)
    print("  ANALYTICAL SUMMARY — FOMC Implied-vs-Realized VRP Study")
    print("═" * 62)

    def _sig(row: dict) -> str:
        p = row.get("p_value_nw", 1.0)
        if p is None or (isinstance(p, float) and np.isnan(p)):
            return ""
        return " ***" if p < 0.01 else " **" if p < 0.05 else " *" if p < 0.10 else ""

    under = pooled.filter(pl.col("mean_vrp") < 0)
    over  = pooled.filter(pl.col("mean_vrp") >= 0)

    if under.is_empty():
        print("\n  No regime shows negative mean VRP (all tenors pooled).")
    else:
        print("\n  Under-priced regimes  (RV > IV, mean_vrp < 0):")
        for row in under.iter_rows(named=True):
            print(f"    {str(row['regime']):22s}  "
                  f"mean_vrp={row['mean_vrp']:+6.2f}pp  "
                  f"pct_under={row['pct_underpriced']:.0%}  "
                  f"n={row['n']}" + _sig(row))

    if not over.is_empty():
        print("\n  Over-priced regimes  (IV > RV, normal insurance premium):")
        for row in over.iter_rows(named=True):
            print(f"    {str(row['regime']):22s}  "
                  f"mean_vrp={row['mean_vrp']:+6.2f}pp  "
                  f"pct_under={row['pct_underpriced']:.0%}  "
                  f"n={row['n']}" + _sig(row))

    print("\n  Significance (NW-HAC): *** p<0.01  ** p<0.05  * p<0.10")

    latest    = vrp_regime_pl["fomc_date"].max()
    curr_rows = vrp_regime_pl.filter(pl.col("fomc_date") == latest)
    if not curr_rows.is_empty():
        curr_regime = curr_rows["regime"][0]
        curr_vrp    = curr_rows["vrp"].mean()
        print(f"\n  ★ Current FOMC ({latest})  regime='{curr_regime}'")
        print(f"    Latest-meeting VRP (avg across tenors): {curr_vrp:+.2f}pp")
        curr_sum = pooled.filter(pl.col("regime") == curr_regime)
        if not curr_sum.is_empty():
            r = curr_sum.row(0, named=True)
            print(f"    Full-regime avg VRP: {r['mean_vrp']:+.2f}pp  "
                  f"under-priced {r['pct_underpriced']:.0%} of meetings")

    print("\n  Note: 21-day forward windows overlap between adjacent meetings;")
    print("  NW-HAC corrects for overlap-induced autocorrelation but remains")
    print("  approximate — weight economic magnitude over p-values here.")
    print("═" * 62)


# ── Part B orchestrator ───────────────────────────────────────────────────

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
    Part B end-to-end pipeline: Phases 0–6.

    Best called with impl_pl=cleaning['impl_clean'] and
    px_daily_pl=cleaning['px_long'] from run_cleaning_pipeline.

    Parameters
    ----------
    impl_pl, rv_pl, regime_pl : Bloomberg frames (impl_pl ideally pre-cleaned)
    px_daily_pl : daily price panel; overrides rv_pl for RV computation
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
        impl_pl, rv_pl, regime_pl
    )

    daily_source = px_daily_pl if px_daily_pl is not None else _px
    if daily_source is None:
        print("\n  Pipeline halted — supply px_daily_pl= (from Part A or BQL re-pull).")
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
        "iv_long":           iv_long,
        "px_long":           px_long,
        "rv_longf":          rv_longf,
        "vrp_pl":            vrp_pl,
        "vrp_regime_pl":     vrp_regime_pl,
        "regime_summary_pl": regime_summary_pl,
    }


# ── ENTRY POINTS ──────────────────────────────────────────────────────────
#
# ── Step 1: clean the raw frames ─────────────────────────────────────────
#
# cleaning = run_cleaning_pipeline(
#     impl_pl     = impl_pl,
#     rv_pl       = rv_pl,
#     # px_daily_pl = px_daily_pl,   # ← supply after BQL re-pull if rv_pl is sparse
#     tau         = 21,
#     cap         = 40.0,
# )
#
# ── Step 2: run the VRP study ─────────────────────────────────────────────
#
# results = run_fomc_vol_study(
#     impl_pl     = cleaning["impl_clean"],
#     rv_pl       = rv_pl,
#     regime_pl   = regime_pl,
#     px_daily_pl = cleaning["px_long"],
#     tau         = 21,
#     direction   = "forward",
#     hac_lags    = 21,
#     output_dir  = Path("."),
# )
#
# # Individual outputs:
# # cleaning["impl_clean"]     — scaled / capped implied vol (wide)
# # cleaning["iv_long"]        — (fomc_date, tenor, iv)
# # cleaning["rv_longf"]       — (fomc_date, tenor, rv, n_days)
# # results["vrp_regime_pl"]   — full VRP table with regime labels
# # results["regime_summary_pl"] — stats by (regime, tenor)
