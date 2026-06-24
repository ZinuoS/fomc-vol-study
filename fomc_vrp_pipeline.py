# %% [markdown]
# # FOMC VRP Arbitrage Pipeline
# **NLP-driven forecast of the implied-vs-realized vol gap across the Treasury curve**
#
# Produces `gap_forecasts.parquet` whose `signal_mult` column feeds directly into
# `fomc_straddle_mc.py`'s `MCConfig.signal_mult`.
#
# ## Environment split
# | Layer | Runs on | Data source |
# |-------|---------|-------------|
# | 1 — Realized vol | **Personal laptop** | FRED + Yahoo OHLC |
# | 2 — Implied vol | **Company laptop** | Bloomberg VCUB/OVDV → CSV stub |
# | 3 — Tokenisation | **Personal laptop** | spaCy NER + rules |
# | 4 — Claude scoring | **Personal laptop** | Anthropic API (cached) |
# | 5 — Gap forecast | **Personal laptop** | ElasticNet walk-forward |
# | 6 — Signal sizing | **Personal laptop** | → `gap_forecasts.parquet` |
#
# ## Files needed
# - `fomc_statements.parquet` — raw FOMC statement text (from `fomc_public_pipeline.py`)
# - `fomc_features.parquet`  — existing NLP features + 2/5/10/30Y forward RV
# - `implied_curve.csv`      — Bloomberg export (company laptop); optional, enables true VRP

# %% ── SECTION 0: CONFIG ───────────────────────────────────────────────────────

from __future__ import annotations
import os, re, json, hashlib, warnings, time
from math import sqrt, log
from pathlib import Path
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats as sp_stats
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNetCV, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
# In Jupyter: switch to inline backend so figures embed in cell output.
# In plain-Python script: use Agg (headless).
try:
    from IPython import get_ipython as _get_ipy
    _ipy = _get_ipy()
    if _ipy is not None:
        _ipy.run_line_magic("matplotlib", "inline")
    else:
        matplotlib.use("Agg")
except Exception:
    matplotlib.use("Agg")

# display() embeds a figure as a cell output in Jupyter; no-op elsewhere
try:
    from IPython.display import display as _ipy_display
except ImportError:
    def _ipy_display(*args, **kwargs): pass  # type: ignore[misc]

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
CACHE_DIR      = Path("fomc_cache")
VRP_CACHE_DIR  = Path("vrp_cache")
VIZ_OUT        = Path("fomc_viz")
NLP_PARQUET    = Path("fomc_features.parquet")
STMT_PARQUET   = Path("fomc_statements.parquet")
# Layer 2 seam: if this CSV exists, real Bloomberg IV is used; else proxy is used
IMPLIED_CSV    = Path("implied_curve.csv")
GAP_FORECASTS  = Path("gap_forecasts.parquet")

for d in (VRP_CACHE_DIR, VIZ_OUT, CACHE_DIR / "market"):
    d.mkdir(parents=True, exist_ok=True)

# ── Claude API ────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY   = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_FILTER_MODEL = "claude-haiku-4-5-20251001"   # word classification (cheap)
CLAUDE_SCORE_MODEL  = "claude-sonnet-4-6"            # passage scoring (nuanced)
FILTER_RUBRIC_VER   = "v1.1"
SCORE_RUBRIC_VER    = "v1.1"
MAX_TOKENS_FILTER   = 2048
MAX_TOKENS_SCORE    = 512
FILTER_BATCH_SIZE   = 40   # words per Claude call

# ── Realized vol config ───────────────────────────────────────────────────────
TRADING_DAYS = 252.0
START_DATE   = "2010-01-01"
END_DATE     = pd.Timestamp.today().strftime("%Y-%m-%d")
RV_WINDOWS   = [1, 5, 10]

TENORS = {
    "2Y":  {"fred": "DGS2",  "future": "ZT=F", "estimator": "gk"},
    "5Y":  {"fred": "DGS5",  "future": "ZF=F", "estimator": "gk"},
    "7Y":  {"fred": "DGS7",  "future": None,    "estimator": "yc", "cash_only": True},
    "10Y": {"fred": "DGS10", "future": "ZN=F",  "estimator": "gk"},
    "20Y": {"fred": "DGS20", "future": None,    "estimator": "yc", "cash_only": True},
    "30Y": {"fred": "DGS30", "future": "ZB=F",  "estimator": "gk"},
}

# ── Forecasting config ────────────────────────────────────────────────────────
MIN_TRAIN_MEETINGS = 30    # walk-forward starts predicting here
KAPPA              = 0.30  # signal_mult = 1 + kappa * max(0, z)
GAP_THRESHOLD_Z    = 0.50  # z above this => trade signal

# ── spaCy boilerplate stoplist ────────────────────────────────────────────────
FOMC_BOILERPLATE: set[str] = {
    "committee","federal","open","market","reserve","board","governor",
    "participant","vote","chair","meeting","member","session","policymaker",
    "president","staff","secretary","system","fiscal","january","february",
    "march","april","june","july","august","september","october","november",
    "december","noted","said","stated","suggest","discuss","consider",
    "view","indicate","report","observe","add","reflect","cite","refer",
    "current","recent","overall","further","additional","level","term",
    "basis","point","range","target","short","long","number","period","time",
    "percent","fund","rate","total","end","average","given","various",
    "include","although","even","also","however","while","would","could",
    "might","may","will","shall","have","been","were","that","this","with",
    "their","some","more","than","into","about","from","continue","remain",
    "generally","broad","particular","several","likely","remain","maintain",
}

NLP_VOL_FEATURES = [
    "word_count_zscore","novelty_prev","novelty_window","guidance_change",
    "uncertainty_density","disagree_density","polarity_hd",
]

CHAIR_PERIODS = {
    "Bernanke": (date(2010,1,1),  date(2014,1,31)),
    "Yellen":   (date(2014,2,1),  date(2018,2,2)),
    "Powell":   (date(2018,2,3),  date(2026,6,16)),
    "Warsh":    (date(2026,6,17), date(2030,1,1)),
}

print("Section 0 config loaded.")

# %% [markdown]
# ---
# # LAYER 1 — FULL-CURVE REALIZED VOL (personal laptop, public data)
# Parkinson and Garman-Klass for tenors with futures OHLC;
# yield-change vol for cash-only tenors (7Y, 20Y). Both flagged clearly.

# %% ── 1a. FRED yield fetch ────────────────────────────────────────────────────

import requests

def fetch_fred_series(series_id: str, start: str = START_DATE) -> pd.Series:
    """Pull daily yield from FRED. Returns Series indexed by date."""
    cache = CACHE_DIR / "market" / f"fred_{series_id}.csv"
    if cache.exists():
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        s = df.iloc[:, 0].dropna()
    else:
        url = (f"https://fred.stlouisfed.org/graph/fredgraph.csv"
               f"?id={series_id}&vintage_date={END_DATE}")
        try:
            s = pd.read_csv(url, index_col=0, parse_dates=True).iloc[:, 0]
            s.to_csv(cache)
        except Exception as e:
            print(f"  FRED fetch failed for {series_id}: {e}")
            return pd.Series(dtype=float, name=series_id)
    s.name = series_id
    return s[s.index >= start].replace(".", np.nan).astype(float).dropna()


fred_yields: dict[str, pd.Series] = {}
for tenor, cfg in TENORS.items():
    fred_id = cfg["fred"]
    s = fetch_fred_series(fred_id)
    fred_yields[tenor] = s
    print(f"  FRED {fred_id:7s} → {len(s):4d} obs  "
          f"({s.index[0].date()} – {s.index[-1].date()})")

# %% ── 1b. Futures OHLC fetch ─────────────────────────────────────────────────

import yfinance as yf

def fetch_ohlc(ticker: str, start: str = START_DATE) -> pd.DataFrame:
    """Pull daily OHLC for a futures ticker from Yahoo/Stooq. Returns DataFrame."""
    cache = VRP_CACHE_DIR / f"ohlc_{ticker.replace('=','').replace('^','')}.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
    else:
        try:
            df = yf.download(ticker, start=start, end=END_DATE,
                             progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df.to_parquet(cache)
        except Exception as e:
            print(f"  OHLC fetch failed for {ticker}: {e}")
            return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df.index = pd.to_datetime(df.index)
    return df[["Open","High","Low","Close"]].dropna()


ohlc_data: dict[str, pd.DataFrame] = {}
for tenor, cfg in TENORS.items():
    if cfg.get("future"):
        df = fetch_ohlc(cfg["future"])
        ohlc_data[tenor] = df
        print(f"  OHLC {cfg['future']:8s} ({tenor}) → {len(df):4d} obs  "
              f"estimator={cfg['estimator']}")
    else:
        print(f"  Cash-only {tenor}: yield-change vol (no OHLC), basis vs futures FLAGGED")

# %% ── 1c. Realized vol estimators ────────────────────────────────────────────

_LN2 = log(2.0)

def parkinson_var(H: float, L: float) -> float:
    """Parkinson (1980): Var_P = (ln(H/L))^2 / (4 ln 2)."""
    if H <= 0 or L <= 0 or H < L:
        return np.nan
    return (log(H / L)) ** 2 / (4 * _LN2)


def garman_klass_var(O: float, H: float, L: float, C: float) -> float:
    """Garman-Klass: 0.5*(ln H/L)^2 - (2ln2-1)*(ln C/O)^2. ~7x more efficient."""
    if any(x <= 0 for x in (O, H, L, C)) or H < L:
        return np.nan
    return 0.5 * (log(H / L)) ** 2 - (2 * _LN2 - 1) * (log(C / O)) ** 2


def rv_annualised(var_daily: float) -> float:
    """Annualise a daily variance estimate → % points."""
    if np.isnan(var_daily) or var_daily <= 0:
        return np.nan
    return sqrt(TRADING_DAYS * var_daily) * 100


def yield_change_var(y_series: pd.Series, date: pd.Timestamp,
                     window: int = 1) -> float:
    """
    Yield-change volatility for cash-only tenors.
    Returns annualised σ in bps (note: DIFFERENT unit from GK — FLAGGED downstream).
    Uses `window` trading days of changes centred on event date.
    """
    idx = y_series.index.searchsorted(date)
    start = max(0, idx - window + 1)
    end   = min(len(y_series), idx + window + 1)
    vals  = y_series.iloc[start:end].dropna().values
    if len(vals) < 2:
        return np.nan
    diffs = np.diff(vals) * 100    # yield in %, changes in bps
    return float(np.std(diffs, ddof=1) * sqrt(TRADING_DAYS))   # annualised bps σ


def compute_forward_rv_yield(y_series: pd.Series, event_date: pd.Timestamp,
                              k: int) -> float:
    """k-day forward close-to-close yield-change RV from event_date+1."""
    idx = y_series.index.searchsorted(event_date)
    fwd = y_series.iloc[idx + 1: idx + k + 2].dropna()
    if len(fwd) < 2:
        return np.nan
    diffs = np.diff(fwd.values) * 100  # bps
    return float(np.std(diffs, ddof=1) * sqrt(TRADING_DAYS))


# %% ── 1d. Build realized_curve ───────────────────────────────────────────────

def build_realized_curve(fomc_dates: list, fred_yields: dict,
                          ohlc_data: dict) -> pd.DataFrame:
    """
    For every (meeting_date, tenor): compute event-day RV + k-day forward RV.

    Columns:
        rv_event_gk, rv_event_park  — futures-based (pp ann.); NaN for cash-only
        rv_event_yc                 — yield-change σ (bps ann.); non-null for all
        rv_fwd_{k}d                 — k-day forward yield-change RV (bps ann.)
        estimator_type              — 'gk_futures' | 'yc_cash' (cash-only tenors)
        cash_only_flag              — True when OHLC unavailable (7Y, 20Y)
    """
    rows = []
    for fd in fomc_dates:
        ts = pd.Timestamp(fd)
        for tenor, cfg in TENORS.items():
            row = {"meeting_date": ts, "tenor": tenor}
            is_cash = cfg.get("cash_only", False)
            row["cash_only_flag"] = is_cash
            row["estimator_type"] = "yc_cash" if is_cash else "gk_futures"

            # ── Futures GK + Parkinson (where available) ──────────────────────
            if not is_cash and tenor in ohlc_data:
                df_ohlc = ohlc_data[tenor]
                if ts in df_ohlc.index:
                    r  = df_ohlc.loc[ts]
                    O, H, L, C = float(r["Open"]), float(r["High"]), float(r["Low"]), float(r["Close"])
                    row["rv_event_gk"]   = rv_annualised(garman_klass_var(O, H, L, C))
                    row["rv_event_park"] = rv_annualised(parkinson_var(H, L))
                else:
                    row["rv_event_gk"] = row["rv_event_park"] = np.nan
            else:
                row["rv_event_gk"] = row["rv_event_park"] = np.nan

            # ── Yield-change vol (all tenors, different unit — FLAGGED) ───────
            if tenor in fred_yields:
                y  = fred_yields[tenor]
                row["rv_event_yc"] = yield_change_var(y, ts, window=1)
                for k in RV_WINDOWS:
                    row[f"rv_fwd_{k}d"] = compute_forward_rv_yield(y, ts, k)
            else:
                row["rv_event_yc"] = np.nan
                for k in RV_WINDOWS:
                    row[f"rv_fwd_{k}d"] = np.nan

            rows.append(row)

    df = pd.DataFrame(rows)
    df["meeting_date"] = pd.to_datetime(df["meeting_date"])
    return df.sort_values(["meeting_date", "tenor"]).reset_index(drop=True)


# Load FOMC meeting dates + NLP features (used throughout all layers)
_feats = pd.read_parquet(NLP_PARQUET)
_feats["meeting_date"] = pd.to_datetime(_feats["meeting_date"])
fomc_dates_list = sorted(_feats["meeting_date"].dt.date.tolist())
feats_df = _feats.copy()   # named alias used in Layer 4d + 5a

realized_curve = build_realized_curve(fomc_dates_list, fred_yields, ohlc_data)

print(f"\nrealized_curve: {realized_curve.shape}")
print(realized_curve.groupby("tenor")[["rv_event_gk","rv_event_yc"]].agg(
    lambda x: f"{x.notna().sum()} obs / mean={x.mean():.1f}"
).to_string())

# %% [markdown]
# ---
# # LAYER 2 — IMPLIED VOL SEAM (company laptop only)
# Clean import seam: tries Bloomberg (bql → xbbg), else reads `implied_curve.csv`.
# On personal laptop this returns a PROXY using rolling realized vol percentiles.
#
# **Bloomberg export snippet** (add at end of your BQuant notebook):
# ```python
# # For each FOMC meeting, per tenor, query VCUB swaption vols + OVDV futures option vols
# # Then isolate event-day implied variance from the vol-kink:
# #   iv_event_var = sigma_span^2 * T_span - sigma_pre^2 * T_pre
# # Export:
# impl_df.to_csv("implied_curve.csv", index=False)
# # Required columns: meeting_date, tenor, iv_event_vol, iv_percentile
# # Optional: iv_event_var, sigma_span, sigma_pre, T_span, T_pre
# ```

# %% ── 2a. Implied vol loader ─────────────────────────────────────────────────

def load_implied_curve(csv_path: Path = IMPLIED_CSV) -> tuple[pd.DataFrame, str]:
    """
    Returns (implied_df, source_tag).

    Priority:
      1. bql (BQuant native) — VCUB + OVDV per tenor
      2. xbbg               — same fields via blpapi
      3. implied_curve.csv  — manual Bloomberg export
      4. PROXY              — rolling GK-vol percentile (personal laptop fallback)

    implied_df columns: meeting_date, tenor, iv_event_vol, iv_percentile [, iv_event_var]
    """
    # ── 1. bql ────────────────────────────────────────────────────────────────
    try:
        import bql                                  # company laptop only
        svc = bql.Service()
        # NOTE: replace PLACEHOLDER_FIELD with actual Bloomberg fields
        # e.g. VCUB = "NORMAL_VOL" on swaption tickers, OVDV = "IVOL_MID" on futures
        # Returning early here; full implementation left to user with live terminal
        raise NotImplementedError("Fill in your VCUB/OVDV BQL queries here.")
    except Exception:
        pass

    # ── 2. xbbg ───────────────────────────────────────────────────────────────
    try:
        from xbbg import blp                        # company laptop only
        raise NotImplementedError("Fill in your xbbg VCUB/OVDV fields here.")
    except Exception:
        pass

    # ── 3. CSV export ─────────────────────────────────────────────────────────
    if csv_path.exists():
        df = pd.read_csv(csv_path, parse_dates=["meeting_date"])
        print(f"[implied] loaded from CSV: {df.shape}  tenors={df['tenor'].unique().tolist()}")
        return df, "csv"

    # ── 4. PROXY: rolling historical GK percentile ────────────────────────────
    # Uses event-day GK vol as a proxy implied vol; rolling 2Y pctile = iv_percentile.
    # IMPORTANT: this is NOT a true IV. Gap computed here ≈ 0 by construction.
    # Replace with Bloomberg data on company laptop.
    rows = []
    for tenor, cfg in TENORS.items():
        sub = realized_curve[realized_curve["tenor"] == tenor].copy()
        vol_col = "rv_event_gk" if not cfg.get("cash_only") else "rv_event_yc"
        sub = sub.dropna(subset=[vol_col]).sort_values("meeting_date")
        sub["iv_event_vol"] = (sub[vol_col]
                                .rolling(8, min_periods=4)
                                .mean()
                                .shift(1))
        sub["iv_percentile"] = (sub[vol_col]
                                 .expanding(min_periods=4)
                                 .rank(pct=True)
                                 .shift(1) * 100)
        rows.append(sub[["meeting_date","tenor","iv_event_vol","iv_percentile"]])

    df = pd.concat(rows, ignore_index=True)
    print("[implied] PROXY mode: using rolling-mean GK as IV proxy. "
          "Replace with Bloomberg export (implied_curve.csv) for true VRP.")
    return df, "proxy"


implied_curve, _iv_source = load_implied_curve()

# ── 2b. VRP gap in variance space ─────────────────────────────────────────────

def compute_vrp_gap(realized: pd.DataFrame, implied: pd.DataFrame) -> pd.DataFrame:
    """
    Merge realized + implied curves; compute VRP gap in variance space.

    gap_var  = rv_event_var - iv_event_var  (pp^2; > 0 => straddle underpriced)
    rv_event_var = (rv_event_gk / 100)^2  (annualised variance, unitless)
    iv_event_var = (iv_event_vol / 100)^2

    For cash-only tenors: rv is in bps; noted as different estimator.
    """
    rv_sel = realized[["meeting_date","tenor","rv_event_gk","rv_event_yc",
                         "rv_event_park","cash_only_flag","estimator_type"]].copy()
    impl   = implied[["meeting_date","tenor","iv_event_vol","iv_percentile"]].copy()

    panel = rv_sel.merge(impl, on=["meeting_date","tenor"], how="left")

    # Use GK for futures tenors, yc for cash-only tenors
    panel["rv_primary"] = np.where(
        panel["cash_only_flag"],
        panel["rv_event_yc"],
        panel["rv_event_gk"],
    )
    panel["rv_event_var"] = (panel["rv_primary"] / 100) ** 2
    panel["iv_event_var"] = (panel["iv_event_vol"] / 100) ** 2
    panel["gap_var"]      = panel["rv_event_var"] - panel["iv_event_var"]

    print(f"\nVRP panel: {panel.shape}")
    print(f"  IV source: {_iv_source}")
    gp = panel.groupby("tenor")["gap_var"]
    print(gp.agg(lambda x: f"mean={x.mean():.4f}  n={x.notna().sum()}").to_string())
    if _iv_source == "proxy":
        print("  ⚠  gap_var ≈ 0 in proxy mode (IV = smoothed RV). "
              "Gap analysis requires Bloomberg IV.")
    return panel


vrp_panel = compute_vrp_gap(realized_curve, implied_curve)

# %% [markdown]
# ---
# # LAYER 3 — TOKENISATION / FILTERING (spaCy NER + rules)
# Drops PERSON, GPE, ORG, DATE, NORP entities; FOMC boilerplate; stopwords.
# Lemmatises; keeps NOUN/VERB/ADJ/ADV + guidance bigrams.

# %% ── 3a. spaCy prefilter ────────────────────────────────────────────────────

import spacy
_nlp_spacy = spacy.load("en_core_web_sm", disable=["parser"])  # NER + tagger

ENTITY_DROPS = {"PERSON","GPE","ORG","DATE","NORP","FAC","LOC","PRODUCT","EVENT"}
POS_KEEP     = {"NOUN","VERB","ADJ","ADV"}


def spacy_prefilter(text: str, add_bigrams: bool = True) -> tuple[list[str], dict]:
    """
    Strip non-characteristic tokens with spaCy NER + rules.
    Returns (token_list, audit_dict).

    audit_dict has: n_raw, n_entity_dropped, n_boilerplate_dropped, n_pos_dropped, n_kept
    """
    if not text or not text.strip():
        return [], {}

    doc = _nlp_spacy(text[:50_000])   # cap length for efficiency

    # Mark entity spans
    ent_token_ids: set[int] = {
        tok.i for ent in doc.ents if ent.label_ in ENTITY_DROPS
        for tok in ent
    }

    raw_n = ent_drop = boilerplate_drop = pos_drop = 0
    kept_tokens: list[str] = []

    for tok in doc:
        if tok.is_space or tok.is_punct or len(tok.text.strip()) < 2:
            continue
        raw_n += 1
        lemma = tok.lemma_.lower().strip()

        if tok.i in ent_token_ids:
            ent_drop += 1
            continue
        if tok.is_stop or lemma in FOMC_BOILERPLATE:
            boilerplate_drop += 1
            continue
        if tok.pos_ not in POS_KEEP:
            pos_drop += 1
            continue

        kept_tokens.append(lemma)

    # Bigrams for guidance phrases
    if add_bigrams and len(kept_tokens) >= 2:
        bigrams = [f"{a} {b}" for a, b in zip(kept_tokens, kept_tokens[1:])]
        kept_tokens = kept_tokens + bigrams

    audit = {
        "n_raw": raw_n, "n_entity_dropped": ent_drop,
        "n_boilerplate_dropped": boilerplate_drop,
        "n_pos_dropped": pos_drop,
        "n_kept": len([t for t in kept_tokens if " " not in t]),
    }
    return kept_tokens, audit


# %% ── 3b. Filter audit ───────────────────────────────────────────────────────

def run_filter_audit(stmt_df: pd.DataFrame) -> pd.DataFrame:
    """Apply spaCy prefilter to all statements; print before/after summary."""
    results = []
    for _, row in stmt_df.iterrows():
        tokens, audit = spacy_prefilter(str(row.get("text", "")), add_bigrams=False)
        results.append({
            "meeting_date": row["meeting_date"],
            "chair": row.get("chair", "Unknown"),
            "filtered_tokens": tokens,
            **audit,
        })

    audit_df = pd.DataFrame(results)
    totals = audit_df[["n_raw","n_entity_dropped","n_boilerplate_dropped",
                        "n_pos_dropped","n_kept"]].sum()
    print("\n" + "═" * 60)
    print("  LAYER 3 — FILTER AUDIT")
    print("═" * 60)
    print(f"  Total raw tokens      : {int(totals['n_raw']):>8,}")
    print(f"  Entity drops (NER)    : {int(totals['n_entity_dropped']):>8,}  "
          f"({totals['n_entity_dropped']/totals['n_raw']:.1%})")
    print(f"  Boilerplate drops     : {int(totals['n_boilerplate_dropped']):>8,}  "
          f"({totals['n_boilerplate_dropped']/totals['n_raw']:.1%})")
    print(f"  POS-filter drops      : {int(totals['n_pos_dropped']):>8,}  "
          f"({totals['n_pos_dropped']/totals['n_raw']:.1%})")
    print(f"  Kept (unigrams)       : {int(totals['n_kept']):>8,}  "
          f"({totals['n_kept']/totals['n_raw']:.1%})")

    # Sample comparison
    sample = audit_df.iloc[0]
    raw_sample = stmt_df.iloc[0]["text"][:300]
    kept_sample = " ".join(sample["filtered_tokens"][:40])
    print(f"\n  Sample (first statement, first 300 chars):")
    print(f"  RAW:     {raw_sample!r}")
    print(f"  FILTERED: {kept_sample!r}")
    print("═" * 60)
    return audit_df


stmt_df = pd.read_parquet(STMT_PARQUET) if STMT_PARQUET.exists() else pd.DataFrame()
if stmt_df.empty:
    print("WARNING: fomc_statements.parquet not found. Run fomc_public_pipeline.py first.")
    stmt_df = pd.DataFrame(columns=["meeting_date","chair","text"])

stmt_df["meeting_date"] = pd.to_datetime(stmt_df["meeting_date"])
filter_audit_df = run_filter_audit(stmt_df)

# %% [markdown]
# ---
# # LAYER 4 — CLAUDE-API VOL SCORING
# Rubric discipline: temp=0, JSON-only, versioned rubric, sha256 cache (no re-billing).
# Prompt forbids world/market knowledge — scores the words, not the era.

# %% ── 4a. Rubric strings + cache ─────────────────────────────────────────────

FILTER_RUBRIC = f"""RUBRIC_VERSION={FILTER_RUBRIC_VER}
You are a monetary policy linguist. Classify each word/phrase as KEEP or DROP.

KEEP: words that carry monetary communication or volatility-relevant meaning
  — e.g. "gradual", "uncertain", "conditional", "asymmetric", "elevated",
    "contingent", "transitory", "symmetric", "attentive", "cautious"
DROP: residual filler, procedural, or generic words with no vol signal
  — e.g. "noted", "cited", "various", "given", "broad", "consistent", "form"

CRITICAL RULE: assess the word's LINGUISTIC character only.
DO NOT use any knowledge of economic outcomes, market moves, or historical events.
Score the word itself — not what era it appeared in.

Respond with JSON array only:
[{{"word": "...", "decision": "KEEP"|"DROP", "reason": "one phrase"}}]
"""

SCORE_RUBRIC = f"""RUBRIC_VERSION={SCORE_RUBRIC_VER}
You are a monetary policy linguist. Score this FOMC passage on 5 dimensions (each 0.0–1.0).

1 ambiguity          — vague qualifiers, hedged phrasing, multiple possible readings
2 uncertainty        — explicit hedging / modality about the economy or rate path
3 conditionality     — state-contingent constructions: "if/should X ... we would Y"
4 guidance_specificity — concrete rate-path language, explicit forward commitment (INVERSE: high=suppresses vol)
5 dissent            — divergence of views, explicit range of opinions, noted disagreements

CRITICAL RULES:
  • Score ONLY the LINGUISTIC properties of the passage.
  • DO NOT use knowledge of economic outcomes, market reactions, or historical context.
  • temperature is 0; this must be reproducible — score the words, not the era.

Respond with JSON only (no explanation):
{{"ambiguity": 0.0, "uncertainty": 0.0, "conditionality": 0.0, "guidance_specificity": 0.0, "dissent": 0.0}}
"""

# ── Cache ─────────────────────────────────────────────────────────────────────
_FILTER_CACHE_PATH = VRP_CACHE_DIR / "claude_filter_cache.json"
_SCORE_CACHE_PATH  = VRP_CACHE_DIR / "claude_score_cache.json"

def _load_cache(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}

def _save_cache(cache: dict, path: Path) -> None:
    path.write_text(json.dumps(cache, indent=None, separators=(",", ":")))

def _sha(text: str, version: str) -> str:
    return hashlib.sha256((text + version).encode()).hexdigest()

_filter_cache = _load_cache(_FILTER_CACHE_PATH)
_score_cache  = _load_cache(_SCORE_CACHE_PATH)

print(f"Filter cache: {len(_filter_cache)} entries  |  Score cache: {len(_score_cache)} entries")

# ── Anthropic client ──────────────────────────────────────────────────────────
_claude_client = None
if ANTHROPIC_API_KEY:
    try:
        import anthropic
        _claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        print("Claude client: connected.")
    except Exception as e:
        print(f"Claude client error: {e}")
else:
    print("ANTHROPIC_API_KEY not set — Claude calls skipped, lexicon fallback active.")

# %% ── 4b. Claude word filter ─────────────────────────────────────────────────

def _call_claude_filter(words: list[str]) -> list[dict]:
    """Send a batch of words to Claude for KEEP/DROP classification. Returns list of dicts."""
    if not _claude_client:
        return [{"word": w, "decision": "KEEP", "reason": "no_api"} for w in words]

    prompt = FILTER_RUBRIC + f"\n\nWords to classify:\n{json.dumps(words)}"
    try:
        resp = _claude_client.messages.create(
            model=CLAUDE_FILTER_MODEL, max_tokens=MAX_TOKENS_FILTER,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        # Extract JSON array robustly
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if m:
            return json.loads(m.group())
        return json.loads(raw)
    except Exception as e:
        print(f"  Claude filter error: {e}")
        return [{"word": w, "decision": "KEEP", "reason": f"error:{e}"} for w in words]


def claude_classify_words(candidate_words: list[str]) -> dict[str, dict]:
    """
    Classify candidate words as KEEP or DROP.
    Caches by sha256(word + rubric_version). Returns {word: {decision, reason}}.
    """
    results: dict[str, dict] = {}
    to_call: list[str] = []

    for w in candidate_words:
        key = _sha(w, FILTER_RUBRIC_VER)
        if key in _filter_cache:
            results[w] = _filter_cache[key]
        else:
            to_call.append(w)

    # Batch uncached words
    for i in range(0, len(to_call), FILTER_BATCH_SIZE):
        batch = to_call[i: i + FILTER_BATCH_SIZE]
        items = _call_claude_filter(batch)
        for item in items:
            w   = item.get("word", "")
            key = _sha(w, FILTER_RUBRIC_VER)
            rec = {"decision": item.get("decision","KEEP"), "reason": item.get("reason","")}
            _filter_cache[key] = rec
            if w:
                results[w] = rec
        _save_cache(_filter_cache, _FILTER_CACHE_PATH)
        if to_call:
            time.sleep(0.3)  # gentle rate limit

    return results


def build_kept_lexicon(filter_audit_df: pd.DataFrame,
                        min_doc_freq: int = 3) -> pd.DataFrame:
    """
    1. Collect all filtered token sets.
    2. Run Claude classification on the union vocabulary.
    3. Return kept_lexicon DataFrame: (word, doc_freq, decision, reason).
    """
    from collections import Counter
    all_tokens: list[str] = []
    for tokens in filter_audit_df["filtered_tokens"]:
        all_tokens.extend([t for t in tokens if " " not in t])  # unigrams only

    freq = Counter(all_tokens)
    candidates = [w for w, c in freq.items() if c >= min_doc_freq]
    print(f"\nLayer 4 — word filter: {len(freq)} unique tokens → {len(candidates)} candidates "
          f"(min_df={min_doc_freq})")

    decisions = claude_classify_words(candidates)

    rows = []
    for w in candidates:
        d = decisions.get(w, {"decision": "KEEP", "reason": "fallback"})
        rows.append({"word": w, "doc_freq": freq[w],
                     "decision": d["decision"], "reason": d["reason"]})

    df = pd.DataFrame(rows).sort_values("doc_freq", ascending=False)
    kept = df[df["decision"] == "KEEP"]
    dropped = df[df["decision"] == "DROP"]
    print(f"  KEEP: {len(kept):4d}   DROP: {len(dropped):4d}   "
          f"(API {'active' if _claude_client else 'offline — all kept'})")
    print(f"\n  Sample KEPT  : {kept['word'].head(10).tolist()}")
    print(f"  Sample DROPPED: {dropped['word'].head(10).tolist()}")
    return df


kept_lexicon = build_kept_lexicon(filter_audit_df, min_doc_freq=3)
_kept_words: set[str] = set(kept_lexicon[kept_lexicon["decision"] == "KEEP"]["word"])

# %% ── 4c. Claude passage scoring ─────────────────────────────────────────────

SCORE_DIMS = ["ambiguity","uncertainty","conditionality","guidance_specificity","dissent"]
_SCORE_DEFAULTS = {d: 0.3 for d in SCORE_DIMS}


def _call_claude_score(passage: str) -> dict:
    """Score one passage on 5 dimensions via Claude. Returns {dim: float}."""
    if not _claude_client:
        return {**_SCORE_DEFAULTS, "_source": "no_api"}

    prompt = SCORE_RUBRIC + f"\n\nPASSAGE:\n{passage[:3000]}"
    try:
        resp = _claude_client.messages.create(
            model=CLAUDE_SCORE_MODEL, max_tokens=MAX_TOKENS_SCORE,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        m   = re.search(r"\{.*\}", raw, re.DOTALL)
        scores = json.loads(m.group() if m else raw)
        # Clamp to [0,1]
        return {d: float(np.clip(scores.get(d, 0.3), 0, 1)) for d in SCORE_DIMS}
    except Exception as e:
        return {**_SCORE_DEFAULTS, "_error": str(e)}


def score_statements_claude(stmt_df: pd.DataFrame,
                             kept_words: set[str]) -> pd.DataFrame:
    """
    Score each statement's FILTERED text on 5 vol-relevant Claude dimensions.
    Passage sent = only kept-lexicon tokens rejoined as prose (reduces noise).
    """
    rows = []
    n_cached = 0

    for _, row in stmt_df.iterrows():
        # Filter passage to kept lexicon only
        tokens, _ = spacy_prefilter(str(row.get("text","")), add_bigrams=False)
        filtered_passage = " ".join(t for t in tokens if t in kept_words)
        if len(filtered_passage) < 20:
            filtered_passage = str(row.get("text",""))[:2000]

        key = _sha(filtered_passage, SCORE_RUBRIC_VER)
        if key in _score_cache:
            scores = _score_cache[key]
            n_cached += 1
        else:
            scores = _call_claude_score(filtered_passage)
            _score_cache[key] = scores
            _save_cache(_score_cache, _SCORE_CACHE_PATH)
            time.sleep(0.25)

        rows.append({
            "meeting_date": row["meeting_date"],
            "chair": row.get("chair","Unknown"),
            "filtered_passage": filtered_passage,
            **{d: scores.get(d, 0.3) for d in SCORE_DIMS},
        })

    df = pd.DataFrame(rows)
    df["meeting_date"] = pd.to_datetime(df["meeting_date"])
    print(f"\nClaude scores: {len(df)} meetings  ({n_cached} from cache)")
    return df


claude_scores_raw = score_statements_claude(stmt_df, _kept_words)

# %% ── 4d. Aggregate to meeting level + PCA ───────────────────────────────────

def aggregate_claude_scores(scores_df: pd.DataFrame,
                             features_df: Optional[pd.DataFrame] = None) -> tuple:
    """
    Aggregate per-passage scores to meeting level (already 1 row/meeting here).
    PCA the 5 raw scores → 2 factors.

    OFFLINE FALLBACK: If Claude API was offline all scores are constant (0.3).
    PCA of a constant matrix yields NaN variance. In that case, fall back to
    offline NLP features (novelty_window, uncertainty_density) as factor proxies.
    """
    df = scores_df[["meeting_date","chair"] + SCORE_DIMS].copy()
    df = df.dropna(subset=SCORE_DIMS)

    # Check if scores have meaningful variance (Claude was online)
    score_var = df[SCORE_DIMS].var().sum()
    _api_active = score_var > 1e-6

    if not _api_active:
        print("\n  PCA: Claude API was offline — scores are constant."
              " Using offline NLP proxies for factor_1/factor_2.")
        # Merge offline NLP features as factor substitutes
        if features_df is not None:
            fdf = features_df[["meeting_date","uncertainty_density","novelty_window"]].copy()
            fdf["meeting_date"] = pd.to_datetime(fdf["meeting_date"])
            df["meeting_date"] = pd.to_datetime(df["meeting_date"])
            df = df.merge(fdf, on="meeting_date", how="left")
            # Standardise to zero-mean unit-variance
            for proxy, factor in [("uncertainty_density","factor_1"),
                                   ("novelty_window","factor_2")]:
                col = df[proxy].fillna(df[proxy].mean())
                std = col.std() or 1.0
                df[factor] = (col - col.mean()) / std
        else:
            df["factor_1"] = 0.0
            df["factor_2"] = 0.0
        print("  factor_1 = uncertainty_density (standardised)")
        print("  factor_2 = novelty_window (standardised)")
        return df, None, None

    # PCA on raw scores (standardised)
    X = df[SCORE_DIMS].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    n_comp = min(2, len(SCORE_DIMS), X.shape[0])
    pca = PCA(n_components=n_comp, random_state=42)
    factors = pca.fit_transform(Xs)
    df["factor_1"] = factors[:, 0]   # typically ambiguity/uncertainty
    df["factor_2"] = factors[:, 1] if factors.shape[1] > 1 else 0.0

    ev = pca.explained_variance_ratio_
    print(f"\nPCA explained variance: PC1={ev[0]:.1%}"
          + (f"  PC2={ev[1]:.1%}" if len(ev) > 1 else ""))

    print(f"\n  PCA loadings (PC1 = 'ambiguity/uncertainty', PC2 = 'specificity'):")
    print(f"  {'Feature':22s}  {'PC1':>8s}  {'PC2':>8s}")
    pc2_loads = pca.components_[1] if n_comp > 1 else [0] * 5
    for feat, l1, l2 in zip(SCORE_DIMS, pca.components_[0], pc2_loads):
        print(f"  {feat:22s}  {l1:>8.3f}  {l2:>8.3f}")

    return df, pca, scaler


claude_scores, _pca_model, _pca_scaler = aggregate_claude_scores(claude_scores_raw, feats_df)

# %% ── 4e. Validate vs offline lexicon ────────────────────────────────────────

def validate_claude_vs_lexicon(claude_df: pd.DataFrame,
                                features_df: pd.DataFrame) -> None:
    """
    Correlate Claude factor scores vs offline lexicon features.
    Low correlation (|r| < 0.3) → WARN: LLM may be using hindsight.
    High correlation → validates that linguistic-only rubric picks up the same signal.
    """
    merged = claude_df.merge(features_df[["meeting_date"] + NLP_VOL_FEATURES],
                             on="meeting_date", how="inner")
    print("\n" + "═" * 60)
    print("  LAYER 4 — CLAUDE vs LEXICON VALIDATION")
    print("  Pearson r between Claude factors and offline NLP features")
    print("  (|r| < 0.3 → WARN: possible hindsight leakage)")
    print("═" * 60)
    print(f"  {'Feature':25s}  {'vs factor_1':>12s}  {'vs factor_2':>12s}")
    print(f"  {'─'*52}")
    warned = False
    for feat in NLP_VOL_FEATURES:
        # In offline mode, uncertainty_density / novelty_window were used AS factors;
        # they will be renamed to _x/_y after merge — use the _x suffix if present.
        col = feat if feat in merged.columns else (f"{feat}_x" if f"{feat}_x" in merged.columns else None)
        if col is None:
            continue
        valid = merged[[col,"factor_1","factor_2"]].dropna()
        valid = valid.rename(columns={col: feat})
        if len(valid) < 10:
            continue
        # Skip if feature has zero variance (e.g. disagree_density all-zeros)
        feat_std = valid[feat].std()
        if feat_std < 1e-8:
            print(f"  {feat:25s}  {'const (zero var)':>12s}  {'':>12s}")
            continue
        # Skip if factor has zero variance (offline mode: constant scores)
        v1_std = valid["factor_1"].std()
        v2_std = valid["factor_2"].std()
        if v1_std < 1e-8 and v2_std < 1e-8:
            print(f"  {feat:25s}  {'N/A (offline)':>12s}  {'N/A (offline)':>12s}")
            continue
        try:
            r1, p1 = (sp_stats.pearsonr(valid[feat], valid["factor_1"])
                      if v1_std > 1e-8 else (0.0, 1.0))
            r2, p2 = (sp_stats.pearsonr(valid[feat], valid["factor_2"])
                      if v2_std > 1e-8 else (0.0, 1.0))
        except Exception:
            print(f"  {feat:25s}  {'error':>12s}  {'error':>12s}")
            continue
        flag = " ← WARN" if max(abs(r1), abs(r2)) < 0.15 else ""
        if flag:
            warned = True
        print(f"  {feat:25s}  {r1:>+10.3f}{'*' if p1<0.05 else ' '}  "
              f"{r2:>+10.3f}{'*' if p2<0.05 else ' '}{flag}")
    if warned:
        print("\n  ⚠  Low correlations detected. The Claude rubric may be picking up")
        print("     content unrelated to the offline lexicon signal. Inspect filtered")
        print("     passages. Offline lexicon scores are retained as fallback model.")
    print("═" * 60)


validate_claude_vs_lexicon(claude_scores, feats_df)

# %% [markdown]
# ---
# # LAYER 5 — FORECASTING THE GAP (elastic-net, walk-forward)
# Panel: (meeting_date, tenor). Target = gap_var (variance space).
# Walk-forward expanding window; cluster SE by meeting.

# %% ── 5a. Build forecasting panel ────────────────────────────────────────────

def build_panel(vrp_panel: pd.DataFrame, claude_scores: pd.DataFrame,
                features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join realized+implied gap with Claude PCA factors + controls.

    Feature set:
      - factor_1, factor_2            (Claude PCA)
      - NLP_VOL_FEATURES              (offline lexicon)
      - iv_percentile                 (mean-reversion control)
      - regime_id                     (market regime)
      - policy_surprise_2y_chg        (GSS/Bauer-Swanson proxy)
      - lagged rv_event_var           (AR component)
      - tenor_code                    (ordinal tenor fixed effect)
      - factor_1 × tenor_code, factor_2 × tenor_code  (key interactions)
    """
    TENOR_CODE = {"2Y": 0, "5Y": 1, "7Y": 2, "10Y": 3, "20Y": 4, "30Y": 5}

    # Core controls from existing features (meeting-level)
    ctrl_cols = ["meeting_date","iv_percentile","regime_id",
                 "policy_surprise_2y_chg"] + NLP_VOL_FEATURES
    ctrl_df = features_df[[c for c in ctrl_cols if c in features_df.columns]].copy()
    ctrl_df["meeting_date"] = pd.to_datetime(ctrl_df["meeting_date"])

    # Claude factors (meeting-level)
    cf = claude_scores[["meeting_date","factor_1","factor_2"]].copy()
    cf["meeting_date"] = pd.to_datetime(cf["meeting_date"])

    # VRP panel (meeting × tenor)
    vp = vrp_panel[["meeting_date","tenor","gap_var","rv_event_var",
                    "iv_percentile","cash_only_flag","estimator_type"]].copy()
    vp["meeting_date"] = pd.to_datetime(vp["meeting_date"])
    vp["tenor_code"] = vp["tenor"].map(TENOR_CODE).fillna(2).astype(int)

    # Merge
    panel = (vp
             .merge(cf,    on="meeting_date", how="left")
             .merge(ctrl_df, on="meeting_date", how="left"))

    # Lagged rv_event_var per tenor (shift by 1 meeting)
    panel = panel.sort_values(["tenor","meeting_date"])
    panel["rv_lag1"] = panel.groupby("tenor")["rv_event_var"].shift(1)

    # Tenor × factor interactions
    panel["f1_x_tenor"] = panel["factor_1"] * panel["tenor_code"]
    panel["f2_x_tenor"] = panel["factor_2"] * panel["tenor_code"]

    # Fill iv_percentile from VRP panel if not in ctrl_df
    if "iv_percentile" not in ctrl_df.columns or panel["iv_percentile"].isna().all():
        pass  # already have it from vrp_panel merge

    # One-hot encode regime
    if "regime_id" in panel.columns:
        panel["regime_id"] = panel["regime_id"].fillna(0).astype(int)
        for r in panel["regime_id"].unique():
            panel[f"regime_{r}"] = (panel["regime_id"] == r).astype(float)

    print(f"\nForecasting panel: {panel.shape}  tenors={panel['tenor'].unique().tolist()}")
    print(f"  Target (gap_var) non-null: {panel['gap_var'].notna().sum()}")
    return panel.reset_index(drop=True)


forecast_panel = build_panel(vrp_panel, claude_scores, feats_df)

# %% ── 5b. Walk-forward ElasticNet ────────────────────────────────────────────

def _get_feature_cols(panel: pd.DataFrame) -> list[str]:
    """Return all usable feature columns (no leakage, no target)."""
    drop_cols = {"meeting_date","tenor","gap_var","rv_event_var","iv_event_var",
                 "rv_event_yc","rv_event_gk","rv_event_park","rv_primary",
                 "iv_event_vol","cash_only_flag","estimator_type","regime_id",
                 "_source","_error"}
    return [c for c in panel.columns
            if c not in drop_cols and panel[c].dtype in (float, int, np.float64, np.int64)
            and not c.startswith("_")]


def fit_walkforward(panel: pd.DataFrame,
                    min_train: int = MIN_TRAIN_MEETINGS) -> pd.DataFrame:
    """
    Walk-forward expanding-window ElasticNetCV.
    For each meeting strictly after the first `min_train` unique meeting dates,
    fit on all prior observations (all tenors), predict on current meeting.

    Returns prediction DataFrame with (meeting_date, tenor, gap_hat, sigma_f, model_coef).
    """
    meetings     = sorted(panel["meeting_date"].unique())
    feat_cols    = _get_feature_cols(panel)
    target_col   = "gap_var"

    all_preds = []

    for i, pred_date in enumerate(meetings):
        if i < min_train:
            continue

        train_mask = panel["meeting_date"] < pred_date
        test_mask  = panel["meeting_date"] == pred_date
        train      = panel[train_mask].dropna(subset=[target_col] + feat_cols[:5])
        test       = panel[test_mask]

        if len(train) < 20 or test.empty:
            continue

        # Impute missing features with training mean
        Xt = train[feat_cols].fillna(train[feat_cols].mean())
        yt = train[target_col].values
        Xv = test[feat_cols].fillna(train[feat_cols].mean())

        scaler = StandardScaler()
        Xts    = scaler.fit_transform(Xt)
        Xvs    = scaler.transform(Xv)

        model  = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 0.99],
                               cv=min(5, len(train) // 10 + 1),
                               random_state=42, max_iter=2000)
        model.fit(Xts, yt)
        preds = model.predict(Xvs)

        # Forecast uncertainty: rolling RMSE of past residuals
        train_preds = model.predict(Xts)
        sigma_f     = float(np.std(yt - train_preds)) or 0.01

        for j, (_, trow) in enumerate(test.iterrows()):
            all_preds.append({
                "meeting_date":  pred_date,
                "tenor":         trow["tenor"],
                "gap_hat":       float(preds[j]),
                "sigma_f":       sigma_f,
                "gap_actual":    trow[target_col],
                "cash_only":     trow.get("cash_only_flag", False),
                "alpha_used":    float(model.alpha_),
                "l1_used":       float(model.l1_ratio_),
                # Key coefficients
                "coef_factor_1": float(model.coef_[feat_cols.index("factor_1")]
                                       if "factor_1" in feat_cols else 0),
                "coef_factor_2": float(model.coef_[feat_cols.index("factor_2")]
                                       if "factor_2" in feat_cols else 0),
                "coef_f1_x_tenor": float(model.coef_[feat_cols.index("f1_x_tenor")]
                                          if "f1_x_tenor" in feat_cols else 0),
            })

    df = pd.DataFrame(all_preds)
    df["meeting_date"] = pd.to_datetime(df["meeting_date"])

    # OOS R² per tenor
    print(f"\n{'─'*60}")
    print(f"  Walk-forward ElasticNet — OOS R² per tenor")
    print(f"  (n_train starts at {min_train} meetings)")
    print(f"{'─'*60}")
    pooled_hat, pooled_act = [], []
    for t in sorted(df["tenor"].unique()):
        sub = df[df["tenor"] == t].dropna(subset=["gap_hat","gap_actual"])
        if len(sub) < 5:
            continue
        r2 = r2_score(sub["gap_actual"], sub["gap_hat"])
        print(f"  {t:4s}  n={len(sub):3d}  OOS R²={r2:>+6.3f}"
              + ("  (yield-change estimator — basis vs futures)" if sub["cash_only"].any() else ""))
        pooled_hat.extend(sub["gap_hat"].tolist())
        pooled_act.extend(sub["gap_actual"].tolist())
    if pooled_act:
        print(f"  POOLED  n={len(pooled_act):3d}  OOS R²={r2_score(pooled_act, pooled_hat):>+6.3f}")

    return df


gap_preds = fit_walkforward(forecast_panel)

# %% ── 5c. Tenor × factor loading matrix ──────────────────────────────────────

def print_tenor_factor_matrix(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Fit one ElasticNet on full historical sample per tenor.
    Print the factor_1, factor_2, f1×tenor loading matrix.
    Hypothesis: front-end loads on specificity/conditionality; long-end on uncertainty.
    """
    feat_cols  = _get_feature_cols(panel)
    target_col = "gap_var"

    tenor_coefs = []
    for t in sorted(panel["tenor"].unique()):
        sub = panel[panel["tenor"] == t].dropna(subset=[target_col] + ["factor_1","factor_2"])
        if len(sub) < 15:
            continue
        Xt = sub[feat_cols].fillna(0)
        yt = sub[target_col].values
        scaler = StandardScaler()
        Xts    = scaler.fit_transform(Xt)
        model  = Ridge(alpha=1.0)
        model.fit(Xts, yt)
        coef_dict = dict(zip(feat_cols, model.coef_))
        tenor_coefs.append({
            "tenor":           t,
            "factor_1":        coef_dict.get("factor_1", 0),
            "factor_2":        coef_dict.get("factor_2", 0),
            "f1_x_tenor":      coef_dict.get("f1_x_tenor", 0),
            "uncertainty_d":   coef_dict.get("uncertainty_density", 0),
            "guidance_change": coef_dict.get("guidance_change", 0),
            "novelty_prev":    coef_dict.get("novelty_prev", 0),
            "iv_percentile":   coef_dict.get("iv_percentile", 0),
        })

    mat = pd.DataFrame(tenor_coefs).set_index("tenor")

    print("\n" + "═" * 70)
    print("  TENOR × FACTOR LOADING MATRIX (in-sample Ridge, for interpretation)")
    print("  Hypothesis: 2Y/5Y loads on specificity; 10Y/30Y loads on uncertainty")
    print("═" * 70)
    print(mat.round(4).to_string())
    print("\n  PC1 (factor_1) = ambiguity/uncertainty composite")
    print("  PC2 (factor_2) = specificity/conditionality composite")
    print("═" * 70)
    return mat


loading_matrix = print_tenor_factor_matrix(forecast_panel)

# %% [markdown]
# ---
# # LAYER 6 — SIGNAL → SIZING (feeds fomc_straddle_mc.py)
# Per meeting: pick the tenor with largest gap_hat > 0; compute z-score → signal_mult.
# Export `gap_forecasts.parquet` — plugs directly into MCConfig.signal_mult.

# %% ── 6a. Compute signal table ────────────────────────────────────────────────

def build_signal_table(gap_preds: pd.DataFrame,
                        kappa: float = KAPPA,
                        z_threshold: float = GAP_THRESHOLD_Z) -> pd.DataFrame:
    """
    Per meeting, select the trade tenor (max gap_hat > 0) and compute:
      z           = gap_hat / sigma_f
      signal_mult = 1 + kappa * max(0, z)

    The signal table is the handoff to fomc_straddle_mc.py:
      MCConfig.signal_mult = row["signal_mult"]  (where row = the next forecast)
    """
    rows = []
    for date, grp in gap_preds.groupby("meeting_date"):
        # Filter to positive gap_hat (straddle underpriced)
        pos = grp[grp["gap_hat"] > 0].copy()
        if pos.empty:
            # No positive signal — choose minimum negative gap (least overpriced)
            chosen = grp.loc[grp["gap_hat"].idxmax()]
        else:
            chosen = pos.loc[pos["gap_hat"].idxmax()]

        z    = float(chosen["gap_hat"]) / max(float(chosen["sigma_f"]), 1e-8)
        mult = 1.0 + kappa * max(0.0, z)

        rows.append({
            "meeting_date":  date,
            "trade_tenor":   chosen["tenor"],
            "gap_hat":       float(chosen["gap_hat"]),
            "gap_actual":    float(chosen.get("gap_actual", np.nan)),
            "sigma_f":       float(chosen["sigma_f"]),
            "z":             z,
            "signal_mult":   mult,
            "trade_signal":  "long" if z > z_threshold else "flat",
        })

    df = pd.DataFrame(rows).sort_values("meeting_date").reset_index(drop=True)
    df.to_parquet(GAP_FORECASTS, index=False)

    print("\n" + "═" * 60)
    print(f"  LAYER 6 — SIGNAL TABLE  ({len(df)} meetings OOS)")
    print(f"  signal_mult: mean={df['signal_mult'].mean():.3f}  "
          f"max={df['signal_mult'].max():.3f}  "
          f"min={df['signal_mult'].min():.3f}")
    print(f"  Long signals: {(df['trade_signal']=='long').sum()} / {len(df)}")
    print(f"  Top predicted tenors: {dict(df['trade_tenor'].value_counts().head())}")
    print(f"\n  → Handoff: load gap_forecasts.parquet, join on meeting_date")
    print(f"    mc_cfg.signal_mult = gap_forecasts.loc[meeting, 'signal_mult']")
    print(f"\n  Columns: {list(df.columns)}")
    print("═" * 60)
    return df


signal_table = build_signal_table(gap_preds)
print(signal_table.tail(5).to_string(index=False))

# %% [markdown]
# ---
# # EVALUATION — OOS Gap-Sign Hit Rate, P&L Simulation, Bootstrap CIs

# %% ── Eval A: Gap-sign hit rate ───────────────────────────────────────────────

def eval_sign_hit_rate(gap_preds: pd.DataFrame) -> pd.DataFrame:
    """
    Did we correctly predict whether the gap was positive (straddle underpriced)?
    Report hit rate per tenor and pooled, with Wilson CI.
    """
    valid = gap_preds.dropna(subset=["gap_hat","gap_actual"])

    def wilson_ci(n: int, k: int, z: float = 1.645) -> tuple:
        if n == 0:
            return 0.0, 0.0
        p = k / n
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2*n)) / denom
        margin = z * sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
        return max(0, centre - margin), min(1, centre + margin)

    rows = []
    for tenor in sorted(valid["tenor"].unique()) + ["POOLED"]:
        if tenor == "POOLED":
            sub = valid
        else:
            sub = valid[valid["tenor"] == tenor]
        if len(sub) < 5:
            continue
        correct = ((sub["gap_hat"] > 0) == (sub["gap_actual"] > 0)).sum()
        n = len(sub)
        hit = correct / n
        lo, hi = wilson_ci(n, int(correct))
        rows.append({"tenor": tenor, "n": n, "hit_rate": hit,
                     "ci_lo_90": lo, "ci_hi_90": hi, "correct": int(correct)})

    df = pd.DataFrame(rows)
    print("\n" + "═" * 60)
    print("  EVALUATION A — Gap-Sign Hit Rate (90% Wilson CI)")
    print("  (Null: 50%. Small n is real — state that explicitly)")
    print("═" * 60)
    print(f"  {'Tenor':8s}  {'n':>5s}  {'Hit rate':>10s}  {'90% CI':>18s}")
    print(f"  {'─'*48}")
    for _, r in df.iterrows():
        beat = " ◄" if r["hit_rate"] > 0.55 else ""
        print(f"  {r['tenor']:8s}  {int(r['n']):>5d}  {r['hit_rate']:>9.1%}  "
              f"[{r['ci_lo_90']:.2f}, {r['ci_hi_90']:.2f}]{beat}")
    print("\n  ⚠  n≈100 OOS meetings. Bootstrap CIs are wide. Treat as exploratory.")
    return df


hit_rate_df = eval_sign_hit_rate(gap_preds)

# %% ── Eval B: P&L simulation ──────────────────────────────────────────────────

def eval_pnl_simulation(signal_table: pd.DataFrame,
                         gap_preds: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate P&L of four strategies (all on the SAME trade-tenor selection):
      1. NLP model   — long vol when signal z > GAP_THRESHOLD_Z
      2. Always long — buy straddle every meeting
      3. Never trade — flat (benchmark)
      4. Naive IV    — long vol when iv_percentile < 40th pctile (mean reversion)

    P&L proxy: +1 when gap_actual > 0 (straddle underpriced → we win),
               −1 when gap_actual < 0 (straddle overpriced → we lose).
    NLP model must beat naive IV model or text adds nothing.
    """
    sig = signal_table[["meeting_date","trade_tenor","z","signal_mult","trade_signal"]].copy()
    pred_merged = gap_preds.merge(
        sig, on=["meeting_date"], how="inner", suffixes=("","_sig")
    )
    valid = pred_merged[pred_merged["tenor"] == pred_merged["trade_tenor"]].dropna(
        subset=["gap_actual","z"])

    # Merge IV percentile for naive model
    if "iv_percentile" in forecast_panel.columns:
        iv_pct = (forecast_panel.groupby("meeting_date")["iv_percentile"]
                   .mean().reset_index())
        valid = valid.merge(iv_pct, on="meeting_date", how="left")

    def pnl(trade_flag: pd.Series, payoff: pd.Series) -> dict:
        net = (trade_flag * payoff).dropna()
        if net.empty:
            return {"mean": 0, "sr": 0, "n_trades": 0}
        return {"mean": float(net.mean()), "sr": float(net.mean() / (net.std() or 1)),
                "n_trades": int(trade_flag.sum())}

    payoff = np.sign(valid["gap_actual"])   # +1 / -1 per meeting
    results = {
        "NLP model":   pnl((valid["z"] > GAP_THRESHOLD_Z).astype(float), payoff),
        "Always long": pnl(pd.Series(1.0, index=valid.index), payoff),
        "Never trade": pnl(pd.Series(0.0, index=valid.index), payoff),
        "Naive IV pct": pnl((valid.get("iv_percentile", pd.Series(50, index=valid.index)) < 40).astype(float), payoff),
    }

    print("\n" + "═" * 60)
    print("  EVALUATION B — P&L Simulation (gap-sign payoff proxy)")
    print("  NLP model must beat Naive IV model to justify text signal")
    print("═" * 60)
    print(f"  {'Strategy':16s}  {'Trades':>7s}  {'Mean P&L':>10s}  {'Sharpe':>8s}")
    print(f"  {'─'*48}")
    for strat, r in results.items():
        flag = " ◄ NLP" if strat == "NLP model" else ""
        must_beat = " ← beat this" if strat == "Naive IV pct" else ""
        print(f"  {strat:16s}  {r['n_trades']:>7d}  {r['mean']:>+9.4f}  "
              f"{r['sr']:>8.3f}{flag}{must_beat}")
    print("═" * 60)
    return pd.DataFrame(results).T


pnl_df = eval_pnl_simulation(signal_table, gap_preds)

# %% ── Eval C: Bootstrap CIs ───────────────────────────────────────────────────

def bootstrap_hit_sharpe(signal_table: pd.DataFrame,
                          gap_preds: pd.DataFrame,
                          n_boot: int = 1000,
                          seed: int = 42) -> None:
    """
    Bootstrap CIs on hit rate and Sharpe for the NLP model.
    Samples meetings with replacement (not observations — meetings are the unit).
    """
    rng = np.random.default_rng(seed)
    sig = signal_table[["meeting_date","trade_tenor","z"]].copy()
    pred_m = gap_preds.merge(sig, on=["meeting_date"], how="inner")
    valid = pred_m[pred_m["tenor"] == pred_m["trade_tenor"]].dropna(subset=["gap_actual","z"])
    valid = valid[valid["z"] > GAP_THRESHOLD_Z]

    if len(valid) < 5:
        print("\n  Bootstrap: insufficient OOS trades for CI (need > 5).")
        return

    payoffs = np.sign(valid["gap_actual"].values)
    hits    = (payoffs > 0).astype(float)
    n       = len(payoffs)

    boot_hr, boot_sr = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        s   = payoffs[idx]
        boot_hr.append(s.mean() > 0)
        boot_sr.append(s.mean() / (s.std() or 1))

    hr_ci  = np.percentile([np.mean(payoffs[rng.integers(0,n,n)] > 0) for _ in range(n_boot)], [5, 95])
    sr_ci  = np.percentile(boot_sr, [5, 95])

    print(f"\n  Bootstrap CIs (n_boot={n_boot}, n_trades={n}):")
    print(f"  Hit rate:  {np.mean(hits):.1%}  90% CI [{hr_ci[0]:.1%}, {hr_ci[1]:.1%}]")
    print(f"  Sharpe:    {payoffs.mean() / (payoffs.std() or 1):.3f}  "
          f"90% CI [{sr_ci[0]:.3f}, {sr_ci[1]:.3f}]")
    print(f"  ⚠  n={n} is small. CIs are wide. State this in any research note.")


bootstrap_hit_sharpe(signal_table, gap_preds)

# %% ── Eval D: Warsh out-of-sample readout ────────────────────────────────────

def warsh_vrp_readout(signal_table: pd.DataFrame,
                       gap_preds: pd.DataFrame,
                       realized_curve: pd.DataFrame) -> None:
    """Print Warsh June 17 2026 VRP readout: gap_hat, chosen tenor, realized gap."""
    warsh_date = pd.Timestamp("2026-06-17")
    sig_row  = signal_table[signal_table["meeting_date"] == warsh_date]
    pred_row = gap_preds[gap_preds["meeting_date"] == warsh_date]
    rv_row   = realized_curve[realized_curve["meeting_date"] == warsh_date]

    print("\n" + "═" * 62)
    print("  WARSH VRP OUT-OF-SAMPLE READOUT  (June 17, 2026)")
    print("═" * 62)

    if not sig_row.empty:
        sr = sig_row.iloc[0]
        print(f"  Trade tenor   : {sr['trade_tenor']}")
        print(f"  gap_hat       : {sr['gap_hat']:+.6f}  (variance space)")
        print(f"  sigma_f       : {sr['sigma_f']:.6f}")
        print(f"  z-score       : {sr['z']:+.3f}")
        print(f"  signal_mult   : {sr['signal_mult']:.3f}  "
              f"(→ MCConfig.signal_mult)")
        print(f"  Trade signal  : {sr['trade_signal'].upper()}")
    else:
        print("  No signal table entry for Warsh (too recent / OOS)")

    if not pred_row.empty:
        print(f"\n  Per-tenor gap_hat:")
        print(f"  {'Tenor':6s}  {'gap_hat':>10s}  {'gap_actual':>12s}  {'z':>6s}")
        print(f"  {'─'*42}")
        for _, r in pred_row.iterrows():
            act_str = f"{r['gap_actual']:>+10.6f}" if pd.notna(r.get("gap_actual")) else "        N/A"
            print(f"  {r['tenor']:6s}  {r['gap_hat']:>+10.6f}  {act_str}  "
                  f"{r['gap_hat']/max(r['sigma_f'],1e-8):>+6.2f}")

    if not rv_row.empty:
        print(f"\n  Realized vol on FOMC day:")
        for _, r in rv_row.iterrows():
            gk_str = f"{r['rv_event_gk']:.2f}pp GK" if pd.notna(r['rv_event_gk']) else "(no OHLC)"
            print(f"  {r['tenor']:4s}  {gk_str}  |  "
                  f"{r.get('rv_event_yc', np.nan):.1f}bps yc-σ  "
                  f"[{r['estimator_type']}]")

    print(f"\n  ⚠  Warsh is 1 OOS meeting. VRP signal is illustrative, not validated.")
    print("═" * 62)


warsh_vrp_readout(signal_table, gap_preds, realized_curve)

# %% [markdown]
# ---
# # LAYER 7 — CLAUDE-FILTERED WORD CLOUD SUITE
# Builds the per-chair word cloud on the FILTERED lexicon only.
# Colour = vol-direction: warm (amber) → widens vol; teal → suppresses vol.
# Framework preserved from existing suite; lexicon is now Claude-cleaned.

# %% ── 7.0 Per-word vol-direction scoring ─────────────────────────────────────

VOL_DIR_SCORE_RUBRIC = f"""RUBRIC_VERSION={SCORE_RUBRIC_VER}
You are a monetary policy linguist. Score this single WORD or PHRASE on:
  warm_score: 0.0–1.0. High = widens realized vol (ambiguity, uncertainty, conditionality, dissent)
  teal_score: 0.0–1.0. High = suppresses realized vol (guidance specificity, concrete commitment)

RULES:
  • Score ONLY the word's LINGUISTIC meaning. No market or historical knowledge.
  • warm + teal do NOT need to sum to 1; a neutral word scores ~0.3 on both.
  • Examples of warm words: "uncertain", "asymmetric", "downside", "conditional", "unclear"
  • Examples of teal words: "commitment", "gradual", "anchored", "symmetric", "path"

Respond with JSON only: {{"warm_score": 0.0, "teal_score": 0.0}}
"""
_VOL_DIR_CACHE_PATH = VRP_CACHE_DIR / "claude_voldir_cache.json"
_voldir_cache = _load_cache(_VOL_DIR_CACHE_PATH)


def _score_word_vol_direction(word: str) -> dict:
    key = _sha(word, SCORE_RUBRIC_VER + "_voldir")
    if key in _voldir_cache:
        return _voldir_cache[key]
    if not _claude_client:
        return {"warm_score": 0.3, "teal_score": 0.3, "_source": "no_api"}

    prompt = VOL_DIR_SCORE_RUBRIC + f"\n\nWord: {word}"
    try:
        resp = _claude_client.messages.create(
            model=CLAUDE_FILTER_MODEL, max_tokens=64, temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        raw  = resp.content[0].text.strip()
        m    = re.search(r"\{.*\}", raw, re.DOTALL)
        res  = json.loads(m.group() if m else raw)
        res  = {"warm_score": float(np.clip(res.get("warm_score",0.3),0,1)),
                "teal_score": float(np.clip(res.get("teal_score",0.3),0,1))}
    except Exception as e:
        res = {"warm_score": 0.3, "teal_score": 0.3, "_error": str(e)}

    _voldir_cache[key] = res
    _save_cache(_voldir_cache, _VOL_DIR_CACHE_PATH)
    time.sleep(0.1)
    return res


def score_kept_lexicon_vol_direction(kept_lexicon: pd.DataFrame) -> pd.DataFrame:
    """Score each KEPT word for vol-direction (warm vs teal)."""
    kept = kept_lexicon[kept_lexicon["decision"] == "KEEP"].copy()
    warm, teal = [], []
    for word in kept["word"]:
        d = _score_word_vol_direction(word)
        warm.append(d.get("warm_score", 0.3))
        teal.append(d.get("teal_score", 0.3))

    kept["warm_score"] = warm
    kept["teal_score"] = teal
    kept["vol_direction"] = np.where(
        np.array(warm) > np.array(teal) + 0.1, "warm",
        np.where(np.array(teal) > np.array(warm) + 0.1, "teal", "neutral")
    )
    print(f"\nVol-direction: {dict(kept['vol_direction'].value_counts())}")
    print(f"  Top warm:  {kept[kept['vol_direction']=='warm']['word'].head(8).tolist()}")
    print(f"  Top teal:  {kept[kept['vol_direction']=='teal']['word'].head(8).tolist()}")
    return kept


scored_lexicon = score_kept_lexicon_vol_direction(kept_lexicon)
scored_lexicon.to_parquet(VRP_CACHE_DIR / "kept_lexicon.parquet", index=False)

# %% ── 7.1 TF-IDF on filtered lexicon ─────────────────────────────────────────

from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

_CHAIR_COLORS = {
    "Bernanke": "#1f6091", "Yellen": "#c97a22",
    "Powell":   "#2c7a45", "Warsh":  "#762a83",
}
CHAIRS = ["Bernanke", "Yellen", "Powell", "Warsh"]

_kept_set = set(scored_lexicon[scored_lexicon["decision"] == "KEEP"]["word"])
_word_to_dir = dict(zip(scored_lexicon["word"], scored_lexicon["vol_direction"]))

# Warm = amber (#d97706), Teal = teal (#0d9488), Neutral = grey
_VOL_DIR_COLOR = {"warm": "#d97706", "teal": "#0d9488", "neutral": "#94a3b8"}


def build_chair_filtered_corpus(stmt_df: pd.DataFrame,
                                 kept_set: set) -> dict[str, list[str]]:
    """Per-chair: list of kept-lexicon tokens across all statements."""
    corpora: dict[str, list[str]] = {c: [] for c in CHAIRS}
    for _, row in stmt_df.iterrows():
        chair = row.get("chair","Unknown")
        if chair not in CHAIRS:
            continue
        tokens, _ = spacy_prefilter(str(row.get("text","")), add_bigrams=False)
        kept = [t for t in tokens if t in kept_set]
        corpora[chair].extend(kept)
    return corpora


def build_tfidf_scores(corpora: dict[str, list[str]]) -> dict[str, dict[str, float]]:
    """
    TF-IDF across 4 chair documents (each chair corpus = one document).
    Corpus is a list of pre-tokenised lemmas; we re-join with spaces so that
    TfidfVectorizer re-splits on whitespace boundaries. The default token
    pattern r"(?u)\b\w\w+\b" matches only single words — correct since our
    tokens are already individual lemmas and we want unigram frequencies.
    """
    # Join tokens with space; TfidfVectorizer re-tokenises on word boundaries
    docs  = [" ".join(corpora[c]) for c in CHAIRS]
    vec   = TfidfVectorizer(sublinear_tf=True, min_df=1,
                            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
    mat   = vec.fit_transform(docs)
    vocab = vec.get_feature_names_out()
    return {CHAIRS[i]: dict(zip(vocab, mat.toarray()[i])) for i in range(len(CHAIRS))}


chair_corpora = build_chair_filtered_corpus(stmt_df, _kept_set)
tfidf_scores  = build_tfidf_scores(chair_corpora)

for c in CHAIRS:
    freq = Counter(chair_corpora[c])
    top  = [w for w, _ in freq.most_common(5)]
    print(f"  {c:10s}: {len(chair_corpora[c]):5d} tokens  top5={top}")

# %% ── 7.2 Word cloud colour function ─────────────────────────────────────────

from wordcloud import WordCloud

def make_color_func(word_to_dir: dict) -> callable:
    """WordCloud color_func: maps word → vol-direction colour (warm/teal/neutral)."""
    def _color_func(word, **kw):
        return _VOL_DIR_COLOR.get(word_to_dir.get(word.lower(), "neutral"), "#94a3b8")
    return _color_func


_WC_PARAMS = dict(
    max_words=120, max_font_size=90, min_font_size=9,
    width=1300, height=680, background_color="white",
    collocations=False, prefer_horizontal=0.85,
)

def make_wordcloud(freq_dict: dict, color_func: callable) -> WordCloud:
    """Shared factory — same params across all chairs for comparability."""
    if not freq_dict:
        return None
    wc = WordCloud(**_WC_PARAMS, color_func=color_func)
    wc.generate_from_frequencies(freq_dict)
    return wc

_color_func = make_color_func(_word_to_dir)

# %% ── 7.3 Fig A: Small multiples (2×2 grid) ──────────────────────────────────

def plot_small_multiples(tfidf_scores: dict, chair_corpora: dict) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(20, 11))
    fig.suptitle("FOMC Statements — Filtered Lexicon Word Clouds\n"
                 "Size = TF-IDF (distinctive words)  ·  "
                 "Amber = widens vol  ·  Teal = suppresses vol  ·  "
                 "Grey = neutral",
                 fontsize=14, fontweight="bold", y=1.01)

    for ax, chair in zip(axes.flat, CHAIRS):
        scores = tfidf_scores.get(chair, {})
        wc = make_wordcloud(scores, _color_func)
        n  = len(chair_corpora.get(chair, []))
        start, end = CHAIR_PERIODS[chair]

        if wc:
            ax.imshow(wc, interpolation="bilinear")
        ax.set_title(f"{chair}  ·  {n:,} kept tokens  ·  {start.year}–{end.year}",
                     fontsize=12, fontweight="bold", color=_CHAIR_COLORS.get(chair, "#333"))
        ax.axis("off")

    # Legend
    handles = [mpatches.Patch(color=c, label=l) for l, c in
               [("Widens vol (warm)", "#d97706"),
                ("Suppresses vol (teal)", "#0d9488"),
                ("Neutral", "#94a3b8")]]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=11,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    fig.savefig(VIZ_OUT / "fig_wc_a_small_multiples.png", dpi=150, bbox_inches="tight")
    print(f"  Saved → {VIZ_OUT}/fig_wc_a_small_multiples.png")
    _ipy_display(fig)
    plt.close(fig)
    return fig


plot_small_multiples(tfidf_scores, chair_corpora)

# %% ── 7.4 Fig B: Powell vs Warsh comparison cloud ────────────────────────────

def plot_comparison_cloud(tfidf_scores: dict) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
    fig.suptitle("Filtered Lexicon — Powell vs Warsh\n"
                 "Size = TF-IDF (distinctive to that chair)  ·  "
                 "Colour = vol-direction",
                 fontsize=13, fontweight="bold")
    for ax, chair, col in [(ax1, "Powell", "#2c7a45"), (ax2, "Warsh", "#762a83")]:
        wc = make_wordcloud(tfidf_scores.get(chair, {}), _color_func)
        if wc:
            ax.imshow(wc, interpolation="bilinear")
        ax.set_title(f"{chair}", fontsize=13, fontweight="bold", color=col)
        ax.axis("off")
    fig.add_artist(plt.Line2D([0.5, 0.5], [0.05, 0.95],
                               transform=fig.transFigure, color="#ccc", lw=1.5))
    handles = [mpatches.Patch(color=c, label=l) for l, c in
               [("Widens vol", "#d97706"), ("Suppresses vol", "#0d9488"), ("Neutral", "#94a3b8")]]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10)
    fig.tight_layout()
    fig.savefig(VIZ_OUT / "fig_wc_b_powell_warsh.png", dpi=150, bbox_inches="tight")
    print(f"  Saved → {VIZ_OUT}/fig_wc_b_powell_warsh.png")
    _ipy_display(fig)
    plt.close(fig)
    return fig


plot_comparison_cloud(tfidf_scores)

# %% ── 7.5 Fig C: Bucket-mix bars (QUANTITATIVE ANCHOR) ──────────────────────

def plot_bucket_mix_bars(chair_corpora: dict, scored_lexicon: pd.DataFrame) -> plt.Figure:
    """
    Horizontal stacked bar per chair: % warm / % teal / % neutral tokens.
    This is the QUANTITATIVE anchor — every cloud gets a bar; clouds alone aren't rigorous.
    """
    word_dir = dict(zip(scored_lexicon["word"], scored_lexicon["vol_direction"]))
    mixes = {}
    for chair in CHAIRS:
        tokens = chair_corpora.get(chair, [])
        if not tokens:
            mixes[chair] = {"warm": 0, "teal": 0, "neutral": 1}
            continue
        dirs = [word_dir.get(t, "neutral") for t in tokens]
        n    = len(dirs)
        mixes[chair] = {
            "warm":    dirs.count("warm") / n,
            "teal":    dirs.count("teal") / n,
            "neutral": dirs.count("neutral") / n,
        }

    fig, ax = plt.subplots(figsize=(11, 5))
    y_pos  = np.arange(len(CHAIRS))
    bar_h  = 0.45

    for i, chair in enumerate(CHAIRS):
        m = mixes[chair]
        left = 0
        for cat, color, label in [
            ("warm",    "#d97706", "Widens vol (warm)"),
            ("teal",    "#0d9488", "Suppresses vol (teal)"),
            ("neutral", "#94a3b8", "Neutral"),
        ]:
            w = m[cat]
            ax.barh(i, w, left=left, height=bar_h,
                    color=color, label=label if i == 0 else "")
            if w > 0.05:
                ax.text(left + w/2, i, f"{w:.0%}", ha="center", va="center",
                        fontsize=9, color="white", fontweight="bold")
            left += w
        ax.text(-0.01, i, chair, ha="right", va="center",
                fontsize=11, fontweight="bold", color=_CHAIR_COLORS.get(chair,"#333"))

    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Token share (filtered lexicon)", fontsize=11)
    ax.set_title("Bucket Mix — Vol-Direction of Filtered Tokens per Chair\n"
                 "(Filtered lexicon only: entity/boilerplate stripped; KEEP tokens only)",
                 fontsize=12, fontweight="bold")
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    ax.legend(seen.values(), seen.keys(), loc="lower right", fontsize=10)
    ax.grid(axis="x", lw=0.4, alpha=0.4)
    fig.tight_layout()
    fig.savefig(VIZ_OUT / "fig_wc_c_bucket_mix.png", dpi=150, bbox_inches="tight")
    print(f"  Saved → {VIZ_OUT}/fig_wc_c_bucket_mix.png")
    _ipy_display(fig)
    plt.close(fig)
    return fig, mixes


fig_c, chair_mixes = plot_bucket_mix_bars(chair_corpora, scored_lexicon)

# %% ── 7.6 Section 4: Warsh word cloud readout ────────────────────────────────

def warsh_wordcloud_readout(chair_mixes: dict, tfidf_scores: dict,
                             chair_corpora: dict) -> None:
    """
    Print Warsh's word_count, warm/teal mix vs other chairs' average,
    top distinctive kept words (TF-IDF). State whether cloud is sparser
    and tilted warm (the vol-widening thesis).
    """
    print("\n" + "═" * 64)
    print("  SECTION 4 — WARSH WORD CLOUD READOUT (vol-widening thesis check)")
    print("═" * 64)
    w = chair_mixes.get("Warsh", {})
    others = {c: chair_mixes[c] for c in CHAIRS if c != "Warsh" and c in chair_mixes}
    avg_warm    = np.mean([v["warm"]    for v in others.values()])
    avg_teal    = np.mean([v["teal"]    for v in others.values()])
    avg_neutral = np.mean([v["neutral"] for v in others.values()])

    n_warsh = len(chair_corpora.get("Warsh", []))
    n_avg   = np.mean([len(chair_corpora.get(c,[])) for c in CHAIRS if c != "Warsh"]) / 3

    print(f"\n  Token count:  Warsh={n_warsh:,}  (vs avg other chairs {n_avg:.0f})")
    print(f"  Cloud sparse? {'YES — far fewer tokens' if n_warsh < n_avg * 0.3 else 'No'}\n")
    print(f"  {'':12s}  {'Warm':>8s}  {'Teal':>8s}  {'Neutral':>8s}")
    print(f"  {'─'*42}")
    print(f"  {'Warsh':12s}  {w.get('warm',0):>8.1%}  {w.get('teal',0):>8.1%}  {w.get('neutral',0):>8.1%}")
    print(f"  {'Other avg':12s}  {avg_warm:>8.1%}  {avg_teal:>8.1%}  {avg_neutral:>8.1%}")
    print(f"  {'Δ (Warsh)':12s}  {w.get('warm',0)-avg_warm:>+8.1%}  "
          f"{w.get('teal',0)-avg_teal:>+8.1%}  "
          f"{w.get('neutral',0)-avg_neutral:>+8.1%}")

    # Thesis check
    warm_tilted = w.get("warm", 0) > avg_warm + 0.05
    print(f"\n  Thesis (Warsh cloud sparse + tilted warm = vol-widening):")
    if n_warsh < n_avg * 0.3 and warm_tilted:
        print("  ✓  CONFIRMED — sparse cloud, above-average warm share.")
    elif n_warsh < n_avg * 0.3:
        print("  △  PARTIAL — cloud is sparse but warm tilt not pronounced.")
    elif warm_tilted:
        print("  △  PARTIAL — warm-tilted but not particularly sparse.")
    else:
        print("  ✗  NOT CONFIRMED — no unusual sparsity or warm tilt.")

    # Top TF-IDF words
    warsh_tfidf = tfidf_scores.get("Warsh", {})
    top_words = sorted(warsh_tfidf, key=warsh_tfidf.get, reverse=True)[:15]
    print(f"\n  Top 15 distinctive kept words (TF-IDF):")
    print(f"  {'Word':25s}  {'TF-IDF':>8s}  {'Vol direction':>14s}")
    print(f"  {'─'*52}")
    for w_str in top_words:
        print(f"  {w_str:25s}  {warsh_tfidf[w_str]:>8.4f}  "
              f"{_word_to_dir.get(w_str, 'neutral'):>14s}")

    print(f"\n  CAVEAT: clouds are illustrative; the bucket-mix bars + scored")
    print(f"  lexicon table are the rigorous backing. The Claude filtering/scoring")
    print(f"  is cached and hindsight-controlled (linguistic rubric only), so the")
    print(f"  lexicon is reproducible. n=1 Warsh statement — treat as exploratory.")
    print("═" * 64)


warsh_wordcloud_readout(chair_mixes, tfidf_scores, chair_corpora)

# %% [markdown]
# ---
# # FINAL SUMMARY & HANDOFF

# %% ── Final summary ───────────────────────────────────────────────────────────

print(f"\n{'═'*66}")
print("  FOMC VRP PIPELINE — OUTPUTS")
print("═" * 66)
print(f"  gap_forecasts.parquet    → {GAP_FORECASTS.resolve()}")
print(f"    Cols: meeting_date, trade_tenor, gap_hat, sigma_f, z, signal_mult")
print(f"    Handoff: MCConfig.signal_mult = gap_forecasts.loc[date, 'signal_mult']")
print(f"\n  kept_lexicon.parquet     → {(VRP_CACHE_DIR/'kept_lexicon.parquet').resolve()}")
print(f"    Cols: word, doc_freq, decision, reason, warm_score, teal_score, vol_direction")
print(f"\n  fig_wc_a_small_multiples.png   (headline 2×2 grid)")
print(f"  fig_wc_b_powell_warsh.png      (Powell vs Warsh comparison)")
print(f"  fig_wc_c_bucket_mix.png        (quantitative bucket-mix anchor)")
print(f"\n  IV source: {_iv_source}  (replace with Bloomberg for true VRP)")
print(f"\n  CAVEATS:")
print(f"  1. Statistical, not riskless arbitrage. Small n (~100 OOS meetings).")
print(f"  2. LLM hindsight risk mitigated: linguistic-only rubric, lexicon validation.")
print(f"  3. Cash-tenor (7Y, 20Y) RV in bps — different estimator, note the basis.")
print(f"  4. Range estimators (GK/Parkinson) understate jump vol — conservative bias.")
print(f"  5. Proxy IV (no Bloomberg): gap ≈ 0 by construction. Bloomberg IV needed.")
print("═" * 66)
