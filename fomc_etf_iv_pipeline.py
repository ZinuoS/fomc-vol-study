"""
fomc_etf_iv_pipeline.py  —  Public-data ETF IV/RV gap pipeline for FOMC vol forecasting.
No Bloomberg.  Every series is public and scrapable.

UNIT CONVENTION (consistent across all layers):
  rv_event_var = (GK_annual_pct / 100)^2   [lognormal, fraction^2, annualised]
  iv_event_var = (IV_annual_pct / 100)^2   [lognormal, same space]
  gap = rv_event_var - iv_event_var         [positive → RV > IV → buy-vol edge]
  GapSpread_proxy = gap(SHY) - gap(TLT)    [positive → buy 2Y-proxy, sell 30Y-proxy]

  RATIONALE: both GK estimator (log(H/L)) and option IV (AlphaVantage) are naturally
  lognormal-convention quantities on the same underlying (ETF price).  No convention
  mismatch.  The Bachelier pricer (fomc_straddle_sim.bachelier) is available for
  downstream conversion to normal vol if needed — see lognormal_to_normal() helper.

PROXY-TO-TRADE BASIS: signal is trained on ETF price-vol.  The live trade is expressed
on ZT (2Y) / ZB (30Y) futures.  Basis sources: ETF tracking error, duration ≠ exact
tenor, ETF-specific flows.  Flag is propagated in the source column.

ENV VARS:
  FRED_API_KEY        — FRED REST API key (free, https://fred.stlouisfed.org/docs/api/)
  ALPHAVANTAGE_API_KEY — AlphaVantage key (free tier: 25 req/day; premium 75/min)
  WRDS_USERNAME        — optional; enables WRDS/OptionMetrics full history
"""
from __future__ import annotations

import json
import os
import time
import warnings
from math import log, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_DIR   = Path(__file__).parent
CACHE_DIR  = REPO_DIR / "etf_cache"
VIZ_DIR    = REPO_DIR / "fomc_viz"
OUT_PARQ   = REPO_DIR / "etf_gap_curve.parquet"

for _d in (CACHE_DIR, CACHE_DIR / "ohlc", CACHE_DIR / "options", CACHE_DIR / "fred",
           CACHE_DIR / "validation", VIZ_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Secrets from env ──────────────────────────────────────────────────────────
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
AV_API_KEY   = os.environ.get("ALPHAVANTAGE_API_KEY", "")
WRDS_USER    = os.environ.get("WRDS_USERNAME", "")

# ── ETF proxy map ─────────────────────────────────────────────────────────────
# Maps from ETF ticker to approximate Treasury tenor proxy.
# SHY ≈ 2Y (front), TLT ≈ 30Y (long-end).  These drive GapSpread_proxy.
ETF_MAP: dict[str, str] = {
    "SHY": "2Y",    # iShares 1-3Y Treasury Bond  → front proxy
    "IEI": "5Y",    # iShares 3-7Y Treasury Bond
    "IEF": "10Y",   # iShares 7-10Y Treasury Bond → cross-check vs VXTYN
    "TLH": "20Y",   # iShares 10-20Y Treasury Bond
    "TLT": "30Y",   # iShares 20+Y Treasury Bond  → long proxy + cross-check vs VXTLT
}

TRADING_DAYS = 252


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 0 — FOMC MEETING CALENDAR
# ─────────────────────────────────────────────────────────────────────────────

def load_fomc_dates(features_parquet: Path = REPO_DIR / "fomc_features.parquet") -> list[pd.Timestamp]:
    """Load meeting dates from existing features parquet, or return hardcoded fallback."""
    if features_parquet.exists():
        df = pd.read_parquet(features_parquet, columns=["meeting_date"])
        dates = sorted(pd.to_datetime(df["meeting_date"].unique()))
        print(f"[calendar] {len(dates)} FOMC meetings from {features_parquet.name}")
        return dates
    # Minimal hardcoded fallback (extend as needed)
    _FALLBACK = [
        "2010-01-27","2010-03-16","2010-04-28","2010-06-23","2010-08-10",
        "2010-09-21","2010-11-03","2010-12-14","2011-01-26","2011-03-15",
        "2011-04-27","2011-06-22","2011-08-09","2011-09-21","2011-11-02",
    ]
    dates = [pd.Timestamp(d) for d in _FALLBACK]
    print(f"[calendar] {len(dates)} FOMC meetings from hardcoded fallback")
    return dates


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1 — FRED (yields + curve features + VXTYN pre-2020 bridge)
# ─────────────────────────────────────────────────────────────────────────────

_FRED_REST_BASE  = "https://api.stlouisfed.org/fred/series/observations"
_FRED_CSV_URL    = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"

FRED_SERIES = {
    # Treasury yields
    "DGS2":   "yield_2y",
    "DGS5":   "yield_5y",
    "DGS7":   "yield_7y",
    "DGS10":  "yield_10y",
    "DGS20":  "yield_20y",
    "DGS30":  "yield_30y",
    # Curve-shape regime features
    "T10Y2Y": "curve_10y2y",
    "T10Y3M": "curve_10y3m",
    # VXTYN: DISCONTINUED 2020-05-15; used only as pre-2020 IV cross-check
    "VXTYN":  "vxtyn",
}


def fetch_fred_series(series_id: str, start: str = "2004-01-01",
                      end: str | None = None) -> pd.Series:
    """
    Pull a FRED series via REST API with disk cache (CSV).
    Returns a pd.Series with DatetimeIndex.  Missing "." values → NaN.
    """
    cache_path = CACHE_DIR / "fred" / f"{series_id}.csv"
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    end   = end or today

    if cache_path.exists():
        df_c = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        s    = df_c.iloc[:, 0].replace(".", np.nan).astype(float)
        last_cached = s.index.max()
        # Re-fetch only if the cache is stale by more than 5 business days
        if pd.Timestamp(end) <= last_cached + pd.offsets.BusinessDay(5):
            return s.rename(series_id)
        print(f"  [FRED] {series_id}: refreshing cache (last={last_cached.date()})")

    def _parse(raw: pd.DataFrame) -> pd.Series:
        s = raw.iloc[:, 0].replace(".", np.nan).astype(float)
        s.index = pd.to_datetime(s.index)
        return s.rename(series_id)

    try:
        time.sleep(0.3)
        if FRED_API_KEY:
            # REST API path (requires key)
            params = dict(series_id=series_id, observation_start=start,
                          observation_end=end, file_type="json",
                          api_key=FRED_API_KEY)
            r   = requests.get(_FRED_REST_BASE, params=params, timeout=20)
            r.raise_for_status()
            obs = r.json().get("observations", [])
            if not obs:
                return pd.Series(dtype=float, name=series_id)
            s = (pd.DataFrame(obs).set_index("date")["value"]
                   .replace(".", np.nan).astype(float))
            s.index = pd.to_datetime(s.index)
            s.name  = series_id
        else:
            # Public CSV endpoint — no key required
            url = _FRED_CSV_URL.format(series=series_id)
            raw = pd.read_csv(url, index_col=0, parse_dates=True)
            s   = _parse(raw)

        s.to_frame().to_csv(cache_path)
        return s
    except Exception as exc:
        print(f"  [FRED] {series_id} fetch failed: {exc}")
        if cache_path.exists():
            df_c = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            return _parse(df_c)
        return pd.Series(dtype=float, name=series_id)


def build_fred_panel(start: str = "2004-01-01") -> pd.DataFrame:
    """
    Fetch all FRED series; return daily DataFrame with forward-fill on non-trading days.
    Columns match FRED_SERIES.values().
    """
    print("\n[FRED] Fetching yield / curve-feature / VXTYN series ...")
    parts = {}
    for sid, col in FRED_SERIES.items():
        s = fetch_fred_series(sid, start=start)
        if not s.empty:
            parts[col] = s
            n_valid = s.dropna().shape[0]
            print(f"  {sid:<10} → {col:<14} {n_valid} obs  "
                  f"[{s.dropna().index.min().date()} – {s.dropna().index.max().date()}]")

    panel = pd.DataFrame(parts)
    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index()

    # ffill yields/curve features but NOT VXTYN past its discontinuation date
    _VXTYN_END = pd.Timestamp("2020-05-15")
    vxtyn_col  = parts.get("vxtyn")
    panel = panel.ffill()
    if "vxtyn" in panel.columns and vxtyn_col is not None:
        panel.loc[panel.index > _VXTYN_END, "vxtyn"] = np.nan

    vxtyn_last = panel["vxtyn"].dropna().index.max().date() if "vxtyn" in panel.columns else "N/A"
    print(f"[FRED] Panel: {panel.shape}, VXTYN available through {vxtyn_last}"
          " (discontinued; pre-2020 cross-check only)")
    return panel


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2 — ETF REALIZED VOL  (Yahoo Finance OHLC + GK/Parkinson)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_etf_ohlc(ticker: str, start: str = "2004-01-01",
                   end: str | None = None) -> pd.DataFrame:
    """
    Pull daily OHLC for an ETF from Yahoo Finance via yfinance.  Cache to parquet.
    Returns DataFrame with columns [Open, High, Low, Close, Volume].
    """
    import yfinance as yf

    cache_path = CACHE_DIR / "ohlc" / f"{ticker}.parquet"
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    end   = end or today

    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        last = df.index.max()
        # Re-fetch only if stale
        if pd.Timestamp(end) <= last + pd.offsets.BusinessDay(3):
            return df
        print(f"  [OHLC] {ticker}: refreshing from {last.date()}")
        start = (last - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        new   = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if not new.empty:
            # Flatten MultiIndex from yfinance ≥0.2.18
            if isinstance(new.columns, pd.MultiIndex):
                new.columns = new.columns.get_level_values(0)
            new.index = pd.to_datetime(new.index)
            combined = pd.concat([df[df.index < new.index.min()], new])
            combined.to_parquet(cache_path)
            return combined
        return df

    print(f"  [OHLC] {ticker}: full download {start}→{end}")
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        print(f"  [OHLC] {ticker}: no data returned")
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    df.to_parquet(cache_path)
    return df


def gk_daily_var(O: float, H: float, L: float, C: float) -> float | None:
    """
    Garman-Klass daily lognormal variance (dimensionless, per-day).
    Returns None if any price is ≤ 0 or H < L.
      σ²_GK = 0.5 × ln(H/L)² - (2ln2 - 1) × ln(C/O)²
    Annualised: multiply by TRADING_DAYS.
    """
    if any(p <= 0 for p in (O, H, L, C)) or H < L:
        return None
    hl = log(H / L)
    co = log(C / O)
    return max(0.0, 0.5 * hl * hl - (2 * log(2) - 1) * co * co)


def parkinson_daily_var(H: float, L: float) -> float | None:
    """Parkinson daily lognormal variance.  σ²_PK = ln(H/L)² / (4 × ln2)"""
    if H <= 0 or L <= 0 or H < L:
        return None
    hl = log(H / L)
    return hl * hl / (4 * log(2))


def rv_for_window(ohlc: pd.DataFrame, start_date: pd.Timestamp,
                  n_days: int) -> float | None:
    """
    Realized variance over n_days starting at start_date (close-to-close).
    Returns annualized variance fraction^2.
    """
    sub = ohlc.loc[start_date:]
    if sub.empty:
        return None
    sub = sub.head(n_days + 1)
    if len(sub) < 2:
        return None
    rets = np.log(sub["Close"] / sub["Close"].shift(1)).dropna().values
    if len(rets) == 0:
        return None
    return float(np.mean(rets ** 2) * TRADING_DAYS)


def build_realized_curve(fomc_dates: list[pd.Timestamp],
                         etf_map: dict[str, str] | None = None,
                         fwd_windows: list[int] | None = None,
                         ohlc_cache: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """
    For each FOMC meeting × ETF:
      • rv_event_gk_pct  : annualized GK vol on the event day (%)
      • rv_event_park_pct: annualized Parkinson vol on the event day (%)
      • rv_event_var     : (rv_event_gk_pct/100)^2  — primary RV in fraction^2
      • rv_fwd_{k}d_var  : forward close-to-close RV over k days, fraction^2
    Pass ohlc_cache to reuse already-downloaded OHLC DataFrames.
    """
    etf_map = etf_map or ETF_MAP
    fwd_windows = fwd_windows or [1, 5, 10]
    if ohlc_cache is None:
        ohlc_cache = {}

    print(f"\n[Layer 2] ETF realized vol: {list(etf_map.keys())} × {len(fomc_dates)} meetings")
    rows = []

    for ticker in etf_map:
        if ticker not in ohlc_cache:
            ohlc_cache[ticker] = fetch_etf_ohlc(ticker)
        n_rows = len(ohlc_cache[ticker])
        print(f"  {ticker}: {n_rows} OHLC rows")

    for fomc_dt in fomc_dates:
        for ticker, tenor in etf_map.items():
            ohlc = ohlc_cache.get(ticker, pd.DataFrame())
            if ohlc.empty:
                continue

            # FOMC event day row
            day_row = ohlc.loc[ohlc.index == fomc_dt]
            if day_row.empty:
                # Try nearest trading day (T+1) in case FOMC falls on a non-trading day
                later = ohlc.loc[ohlc.index > fomc_dt].head(1)
                day_row = later if not later.empty else day_row

            rec: dict = {"meeting_date": fomc_dt, "etf": ticker, "tenor_proxy": tenor}

            if not day_row.empty:
                r   = day_row.iloc[0]
                var_gk  = gk_daily_var(r["Open"], r["High"], r["Low"], r["Close"])
                var_pk  = parkinson_daily_var(r["High"], r["Low"])
                pct_gk  = sqrt(var_gk  * TRADING_DAYS) * 100 if var_gk  is not None else np.nan
                pct_pk  = sqrt(var_pk  * TRADING_DAYS) * 100 if var_pk  is not None else np.nan
                rec["rv_event_gk_pct"]  = pct_gk
                rec["rv_event_park_pct"]= pct_pk
                rec["rv_event_var"]     = (pct_gk / 100) ** 2 if not np.isnan(pct_gk) else np.nan
                rec["rv_event_park_var"]= (pct_pk / 100) ** 2 if not np.isnan(pct_pk) else np.nan
            else:
                for c in ("rv_event_gk_pct","rv_event_park_pct","rv_event_var","rv_event_park_var"):
                    rec[c] = np.nan

            for k in fwd_windows:
                fwd_var = rv_for_window(ohlc, fomc_dt, k)
                rec[f"rv_fwd_{k}d_var"] = fwd_var

            rows.append(rec)

    df = pd.DataFrame(rows)
    n_with_rv = df["rv_event_var"].notna().sum()
    print(f"[Layer 2] Realized curve: {len(df)} rows, {n_with_rv} with event-day GK RV")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 3 — ETF IMPLIED VOL
# ─────────────────────────────────────────────────────────────────────────────

# ── 3a. AlphaVantage historical option chains ─────────────────────────────────

_AV_BASE = "https://www.alphavantage.co/query"
_AV_RATE_SLEEP = 13.0    # 60/5 = 12 s between calls; add 1 s safety margin
_AV_CACHE_DIR  = CACHE_DIR / "options" / "alphavantage"
_AV_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _av_chain_path(ticker: str, obs_date: str) -> Path:
    return _AV_CACHE_DIR / f"{ticker}_{obs_date}.json"


def fetch_av_options(ticker: str, obs_date: str,
                     api_key: str = AV_API_KEY) -> list[dict] | None:
    """
    Fetch AlphaVantage HISTORICAL_OPTIONS for (ticker, obs_date='YYYY-MM-DD').
    Caches every response.  Returns None on rate-limit or missing key.
    Returns [] if the API returns an empty chain (e.g. pre-history date).
    """
    cache_path = _av_chain_path(ticker, obs_date)
    if cache_path.exists():
        with open(cache_path) as f:
            payload = json.load(f)
        return payload.get("data", [])

    if not api_key:
        return None  # no key → skip

    params = dict(function="HISTORICAL_OPTIONS", symbol=ticker,
                  date=obs_date, apikey=api_key)
    try:
        time.sleep(_AV_RATE_SLEEP)
        r = requests.get(_AV_BASE, params=params, timeout=30)
        r.raise_for_status()
        payload = r.json()

        # Rate-limit signal: response contains "Note" or "Information"
        if "Note" in payload or "Information" in payload:
            msg = payload.get("Note") or payload.get("Information", "")
            print(f"  [AV] rate-limit/key error for {ticker}/{obs_date}: {msg[:80]}")
            return None  # don't cache; will retry next run

        with open(cache_path, "w") as f:
            json.dump(payload, f)
        return payload.get("data", [])

    except Exception as exc:
        print(f"  [AV] {ticker}/{obs_date}: {exc}")
        return None


def _parse_av_chain(chain: list[dict]) -> pd.DataFrame:
    """Parse raw AlphaVantage option chain list into a clean DataFrame."""
    if not chain:
        return pd.DataFrame()
    df = pd.DataFrame(chain)
    numeric_cols = ["strike","last","mark","bid","ask","implied_volatility",
                    "delta","gamma","theta","vega","open_interest","volume"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "expiration" in df.columns:
        df["expiration"] = pd.to_datetime(df["expiration"])
    return df


def extract_atm_iv_bracket(chain_df: pd.DataFrame,
                            fomc_date: pd.Timestamp,
                            etf_close: float,
                            min_open_interest: int = 5) -> dict | None:
    """
    From an option chain observed on obs_date, extract ATM IV for the expiry
    that BRACKETS the FOMC event:

    Returns dict with:
      iv_pre_pct    : ATM lognormal IV (%) for pre-FOMC expiry
      iv_post_pct   : ATM lognormal IV (%) for post-FOMC expiry
      T_pre_years   : time-to-expiry in years for pre-FOMC expiry
      T_post_years  : time-to-expiry in years for post-FOMC expiry
      expiry_pre    : date of pre-FOMC expiry
      expiry_post   : date of post-FOMC expiry
      window_flag   : "weekly" | "monthly" depending on gap between expiries
      iv_event_var  : iv_post^2 * T_post - iv_pre^2 * T_pre  (fraction^2)

    Returns None if both expiries cannot be found with usable IV.
    """
    if chain_df.empty or "implied_volatility" not in chain_df.columns:
        return None

    if "expiration" not in chain_df.columns:
        return None

    # Keep only options with reasonable IV and some open interest / volume
    valid = chain_df[
        (chain_df["implied_volatility"].notna()) &
        (chain_df["implied_volatility"] > 0.0005) &
        (chain_df["implied_volatility"] < 5.0) &
        ((chain_df.get("open_interest", pd.Series(0, index=chain_df.index)) >= min_open_interest) |
         (chain_df.get("volume", pd.Series(0, index=chain_df.index)) >= 1))
    ].copy()

    if valid.empty:
        return None

    all_expiries = sorted(valid["expiration"].dropna().unique())
    pre_expiries  = [e for e in all_expiries if e <  fomc_date]
    post_expiries = [e for e in all_expiries if e >= fomc_date]

    if not pre_expiries or not post_expiries:
        return None

    expiry_pre  = pre_expiries[-1]   # last expiry before FOMC
    expiry_post = post_expiries[0]   # first expiry on or after FOMC

    # Determine obs_date from context — not passed here, infer from earliest expiry relative
    # (T_years will be computed in the caller using obs_date)
    results = {}
    for label, expiry in [("pre", expiry_pre), ("post", expiry_post)]:
        leg = valid[valid["expiration"] == expiry]
        if leg.empty:
            return None

        # ATM: closest strike to current price, average call + put IV
        leg_call = leg[leg.get("type", pd.Series(dtype=str)).str.lower().isin(["call","c"])]
        leg_put  = leg[leg.get("type", pd.Series(dtype=str)).str.lower().isin(["put","p"])]

        def _best_atm_iv(grp: pd.DataFrame) -> float | None:
            if grp.empty:
                return None
            grp = grp.copy()
            grp["dist"] = (grp["strike"] - etf_close).abs()
            best = grp.nsmallest(1, "dist")
            iv = float(best["implied_volatility"].iloc[0])
            return iv if 0 < iv < 5 else None

        iv_c = _best_atm_iv(leg_call)
        iv_p = _best_atm_iv(leg_put)

        if iv_c is not None and iv_p is not None:
            iv_pct = ((iv_c + iv_p) / 2) * 100  # fraction → percent
        elif iv_c is not None:
            iv_pct = iv_c * 100
        elif iv_p is not None:
            iv_pct = iv_p * 100
        else:
            return None

        results[f"iv_{label}_pct"]   = iv_pct
        results[f"expiry_{label}"]   = expiry

    if "iv_pre_pct" not in results or "iv_post_pct" not in results:
        return None

    # Window flag
    gap_days = (results["expiry_post"] - results["expiry_pre"]).days
    results["window_flag"] = "weekly" if gap_days <= 10 else "monthly"
    return results


def _obs_date(fomc_date: pd.Timestamp, ohlc: pd.DataFrame) -> pd.Timestamp | None:
    """Return the observation date = 2 business days before FOMC (last trading day ≤ fomc-2bd)."""
    target = fomc_date - pd.offsets.BusinessDay(2)
    available = ohlc.index[ohlc.index <= target]
    return available[-1] if len(available) > 0 else None


# ── 3b. WRDS / OptionMetrics (optional full-history seam) ────────────────────

def try_wrds_optionmetrics(ticker: str, fomc_date: pd.Timestamp) -> dict | None:
    """
    Attempt to pull ATM IV from WRDS OptionMetrics IvyDB.
    Returns same dict structure as extract_atm_iv_bracket(), or None.
    Requires wrds package and WRDS_USERNAME env var.
    """
    if not WRDS_USER:
        return None
    try:
        import wrds
        db = wrds.Connection(wrds_username=WRDS_USER)
        obs_date_str = (fomc_date - pd.offsets.BusinessDay(2)).strftime("%Y-%m-%d")
        query = f"""
            SELECT h.date, h.expiration, h.strike_price/1000.0 AS strike,
                   h.impl_volatility, h.best_bid, h.best_offer, h.open_interest,
                   h.cp_flag
            FROM optionm.opprcd{fomc_date.year} h
            JOIN optionm.securd s ON s.secid = h.secid
            WHERE s.ticker = '{ticker}'
              AND h.date    = '{obs_date_str}'
              AND h.impl_volatility IS NOT NULL
              AND h.volume > 0
            ORDER BY h.expiration, ABS(h.strike_price/1000.0 - (
                SELECT close FROM optionm.opprcd{fomc_date.year}
                WHERE secid = h.secid AND date = '{obs_date_str}' LIMIT 1
            ))
        """
        df = db.raw_sql(query)
        db.close()
        if df.empty:
            return None
        df["expiration"] = pd.to_datetime(df["expiration"])
        df["impl_volatility"] = pd.to_numeric(df["impl_volatility"], errors="coerce")
        df = df.rename(columns={"impl_volatility": "implied_volatility",
                                  "cp_flag": "type", "strike": "strike"})
        df["type"] = df["type"].map({"C": "call", "P": "put"})
        # Use the same extraction function on the normalized chain
        spot = df.get("close", pd.Series([None])).iloc[0]
        if spot is None or pd.isna(spot):
            return None
        return extract_atm_iv_bracket(df, fomc_date, float(spot))
    except Exception as exc:
        print(f"  [WRDS] {ticker}/{fomc_date.date()}: {exc}")
        return None


# ── 3c. Manual CSV seam ───────────────────────────────────────────────────────

_MANUAL_CSV = REPO_DIR / "etf_implied_manual.csv"
_MANUAL_SCHEMA = ["date","etf","expiry_pre","expiry_post",
                   "iv_pre_pct","iv_post_pct","source_note"]


def load_manual_iv() -> pd.DataFrame:
    """
    Load hand-filled implied vol from etf_implied_manual.csv.
    Column schema: date, etf, expiry_pre, expiry_post, iv_pre_pct, iv_post_pct, source_note.
    """
    if not _MANUAL_CSV.exists():
        # Create an empty stub so the user knows what to fill
        pd.DataFrame(columns=_MANUAL_SCHEMA).to_csv(_MANUAL_CSV, index=False)
        return pd.DataFrame(columns=_MANUAL_SCHEMA)
    df = pd.read_csv(_MANUAL_CSV, parse_dates=["date","expiry_pre","expiry_post"])
    print(f"[manual CSV] {len(df)} rows from {_MANUAL_CSV.name}")
    return df


# ── 3d. Main implied vol builder ──────────────────────────────────────────────

def build_implied_curve(fomc_dates: list[pd.Timestamp],
                        etf_map: dict[str, str] | None = None,
                        ohlc_cache: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """
    For each FOMC meeting × ETF, attempt to build iv_event_var using:
      1. WRDS/OptionMetrics (if WRDS_USERNAME set)
      2. AlphaVantage HISTORICAL_OPTIONS (if ALPHAVANTAGE_API_KEY set)
      3. Manual CSV fallback
    Output: (meeting_date, etf, tenor_proxy, iv_event_var, iv_event_vol_pct,
             iv_percentile, source, expiry_pre, expiry_post, window_flag)
    """
    etf_map = etf_map or ETF_MAP
    manual_df = load_manual_iv()

    print(f"\n[Layer 3] ETF implied vol: {list(etf_map.keys())} × {len(fomc_dates)} meetings")
    if not AV_API_KEY:
        print("  WARNING: ALPHAVANTAGE_API_KEY not set — AlphaVantage source disabled.")
    if not WRDS_USER:
        print("  INFO: WRDS_USERNAME not set — OptionMetrics source disabled.")

    # Pre-load OHLC for spot prices
    if ohlc_cache is None:
        ohlc_cache = {}
    for ticker in etf_map:
        if ticker not in ohlc_cache:
            ohlc_cache[ticker] = fetch_etf_ohlc(ticker)

    rows = []
    n_av_fetched = 0
    n_cached = 0

    for fomc_dt in fomc_dates:
        for ticker, tenor in etf_map.items():
            ohlc = ohlc_cache.get(ticker, pd.DataFrame())
            obs_dt = _obs_date(fomc_dt, ohlc) if not ohlc.empty else None
            spot   = None
            if obs_dt is not None and obs_dt in ohlc.index:
                spot = float(ohlc.loc[obs_dt, "Close"])

            rec: dict = {
                "meeting_date": fomc_dt,
                "etf": ticker,
                "tenor_proxy": tenor,
                "obs_date": obs_dt,
                "spot": spot,
            }

            bracket: dict | None = None
            source = "missing"

            # ── Source 1: WRDS / OptionMetrics ────────────────────────────────
            if bracket is None and WRDS_USER:
                bracket = try_wrds_optionmetrics(ticker, fomc_dt)
                if bracket:
                    source = "wrds_optionmetrics"

            # ── Source 2: AlphaVantage HISTORICAL_OPTIONS ──────────────────────
            if bracket is None and AV_API_KEY and obs_dt is not None:
                obs_str = obs_dt.strftime("%Y-%m-%d")
                # Check cache first (no API call if cached)
                cached_path = _av_chain_path(ticker, obs_str)
                if cached_path.exists():
                    n_cached += 1
                else:
                    n_av_fetched += 1
                    if n_av_fetched % 10 == 0:
                        print(f"    [AV] {n_av_fetched} new API calls so far ...")

                chain_raw = fetch_av_options(ticker, obs_str)
                if chain_raw is not None:
                    chain_df = _parse_av_chain(chain_raw)
                    if spot is not None and not chain_df.empty:
                        bracket = extract_atm_iv_bracket(chain_df, fomc_dt, spot)
                        if bracket:
                            source = "alphavantage"

            # ── Source 3: Manual CSV ───────────────────────────────────────────
            if bracket is None and not manual_df.empty:
                man = manual_df[
                    (manual_df["date"] == fomc_dt) &
                    (manual_df["etf"] == ticker)
                ]
                if not man.empty:
                    m = man.iloc[0]
                    bracket = {
                        "iv_pre_pct":    float(m["iv_pre_pct"]),
                        "iv_post_pct":   float(m["iv_post_pct"]),
                        "expiry_pre":    pd.Timestamp(m["expiry_pre"]),
                        "expiry_post":   pd.Timestamp(m["expiry_post"]),
                        "window_flag":   "manual",
                    }
                    source = f"manual_csv:{m.get('source_note','')}"

            # ── Compute iv_event_var from bracket ─────────────────────────────
            if bracket is not None and obs_dt is not None:
                iv_pre_frac  = bracket["iv_pre_pct"]  / 100.0
                iv_post_frac = bracket["iv_post_pct"] / 100.0
                expiry_pre   = pd.Timestamp(bracket["expiry_pre"])
                expiry_post  = pd.Timestamp(bracket["expiry_post"])
                T_pre  = max(0.0, (expiry_pre  - obs_dt).days / 365.25)
                T_post = max(0.0, (expiry_post - obs_dt).days / 365.25)

                # Event variance via expiry kink:
                #   iv_event_var = σ_post² × T_post - σ_pre² × T_pre   [fraction²]
                # Both in lognormal-variance space, same convention as GK realized vol.
                iv_event_var = iv_post_frac**2 * T_post - iv_pre_frac**2 * T_pre

                # Back out an effective ATM annualised vol for the event window
                # (used for cross-checks and percentile ranking)
                T_event = max(T_post - T_pre, 1.0 / 365.25)
                iv_event_vol_pct = sqrt(max(0.0, iv_event_var) / T_event) * 100

                rec.update({
                    "iv_pre_pct":     bracket["iv_pre_pct"],
                    "iv_post_pct":    bracket["iv_post_pct"],
                    "expiry_pre":     expiry_pre,
                    "expiry_post":    expiry_post,
                    "T_pre_years":    T_pre,
                    "T_post_years":   T_post,
                    "iv_event_var":   iv_event_var,
                    "iv_event_vol_pct": iv_event_vol_pct,
                    "window_flag":    bracket.get("window_flag", "unknown"),
                    "source":         source,
                })
            else:
                for c in ("iv_pre_pct","iv_post_pct","expiry_pre","expiry_post",
                          "T_pre_years","T_post_years","iv_event_var",
                          "iv_event_vol_pct","window_flag"):
                    rec[c] = np.nan
                rec["source"] = source

            rows.append(rec)

    df = pd.DataFrame(rows)
    print(f"[Layer 3] Implied curve: {len(df)} rows  "
          f"{df['iv_event_var'].notna().sum()} with IV  "
          f"(cached={n_cached}, new API calls={n_av_fetched})")
    print("  Source breakdown:")
    for src, cnt in df["source"].value_counts().items():
        print(f"    {src:<30} {cnt:>4}")
    return df


# ── Percentile rank per ETF per year ──────────────────────────────────────────

def add_iv_percentile(implied_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add iv_percentile column: within each ETF, rank iv_event_vol_pct relative
    to a trailing 4-year expanding window of meetings.  Range [0, 100].
    """
    implied_df = implied_df.sort_values("meeting_date").copy()
    implied_df["iv_percentile"] = np.nan

    for ticker in implied_df["etf"].unique():
        mask = implied_df["etf"] == ticker
        sub  = implied_df.loc[mask, ["meeting_date","iv_event_vol_pct"]].dropna()
        pcts = {}
        for i, (idx, row) in enumerate(sub.iterrows()):
            hist = sub.iloc[:i]["iv_event_vol_pct"].dropna()
            if hist.empty:
                pcts[idx] = 50.0
            else:
                pcts[idx] = float((hist < row["iv_event_vol_pct"]).mean() * 100)
        for idx, pct in pcts.items():
            implied_df.at[idx, "iv_percentile"] = pct

    return implied_df


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 4 — GAP (RV − IV) + GAPSPREAD PROXY
# ─────────────────────────────────────────────────────────────────────────────

# Unit guard: assert both legs are in the same convention before merging.
_UNIT_CONVENTION = "lognormal_fraction_sq"


def merge_gap_curve(realized: pd.DataFrame, implied: pd.DataFrame) -> pd.DataFrame:
    """
    Merge realized and implied curves; compute gap = rv_event_var - iv_event_var.
    Both must be in lognormal fraction^2 units (asserted by the _UNIT_CONVENTION tag).
    """
    assert "rv_event_var"  in realized.columns, "realized missing rv_event_var"
    assert "iv_event_var"  in implied.columns,  "implied missing iv_event_var"

    _iv_cols = ["meeting_date","etf","iv_event_var","iv_event_vol_pct",
                "iv_percentile","source","expiry_pre","expiry_post","window_flag"]
    _iv_cols_present = [c for c in _iv_cols if c in implied.columns]
    merged = realized.merge(implied[_iv_cols_present], on=["meeting_date","etf"], how="left")

    # UNIT GUARD: both sides are lognormal fraction^2.  GK uses log(H/L) → lognormal.
    # AlphaVantage IV is Black-Scholes lognormal.  No mismatch.
    merged.attrs["iv_rv_unit_convention"] = _UNIT_CONVENTION

    merged["gap"]      = merged["rv_event_var"] - merged["iv_event_var"]
    merged["has_iv"]   = merged["iv_event_var"].notna()
    merged["has_rv"]   = merged["rv_event_var"].notna()
    merged["has_both"] = merged["has_iv"] & merged["has_rv"]

    return merged


def add_gap_spread(df: pd.DataFrame,
                   front_etf: str = "SHY", long_etf: str = "TLT") -> pd.DataFrame:
    """
    Compute GapSpread_proxy = gap(front_etf) - gap(long_etf) per meeting.
    Positive → front RV > front IV more than long RV > long IV → BUY_FRONT_SELL_LONG.
    """
    front = (df[df["etf"] == front_etf][["meeting_date","gap"]]
               .rename(columns={"gap": f"gap_{front_etf}"}))
    long_ = (df[df["etf"] == long_etf][["meeting_date","gap"]]
               .rename(columns={"gap": f"gap_{long_etf}"}))

    spread = front.merge(long_, on="meeting_date", how="inner")
    spread["GapSpread_proxy"] = spread[f"gap_{front_etf}"] - spread[f"gap_{long_etf}"]
    spread["spread_direction"] = np.where(
        spread["GapSpread_proxy"] > 0, "buy_front_sell_long", "buy_long_sell_front"
    )
    spread = spread.sort_values("meeting_date")

    # Merge back into the full panel
    df = df.merge(spread[["meeting_date","GapSpread_proxy","spread_direction"]],
                  on="meeting_date", how="left")
    n_spread = spread["GapSpread_proxy"].notna().sum()
    print(f"\n[Layer 4] GapSpread_proxy: {n_spread}/{len(spread)} meetings have both legs")
    return df


def add_proxy_basis_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basis_flag column explaining the ETF → futures proxy basis.
    Propagates into sample weighting: 'etf_proxy' records are usable but have tracking basis.
    """
    df["basis_flag"] = (
        "signal=ETF_proxy; trade=ZT/ZB_futures; basis=duration_mismatch+tracking+flows"
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 5 — VALIDATION & CROSS-CHECKS (public)
# ─────────────────────────────────────────────────────────────────────────────

_VXTLT_URLS = [
    "https://cdn.cboe.com/api/global/us_indices/daily_prices/VXTLT_History.csv",
    "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIXTLT_History.csv",
]


def _fetch_cboe_csv(urls: list[str], cache_name: str) -> pd.DataFrame:
    """Download a Cboe index CSV, cache it, return a DataFrame with [Date, Close]."""
    cache_path = CACHE_DIR / "validation" / cache_name
    if cache_path.exists():
        try:
            df = pd.read_csv(cache_path, parse_dates=["Date"], index_col="Date")
            print(f"  [{cache_name}] {len(df)} obs from cache")
            return df
        except Exception:
            pass

    for url in urls:
        try:
            time.sleep(0.5)
            df = pd.read_csv(url, parse_dates=[0])
            # Cboe CSV column 0 may be "Date" or an unnamed date column
            if df.columns[0].lower() in ("date",""):
                df.columns = ["Date"] + list(df.columns[1:])
                df = df.set_index("Date")
            else:
                df.index.name = "Date"
            df.to_csv(cache_path)
            print(f"  [{cache_name}] {len(df)} obs from {url}")
            return df
        except Exception as exc:
            print(f"  [{cache_name}] {url}: {exc}")
    print(f"  [{cache_name}]: all sources failed")
    return pd.DataFrame()


def cross_check_vxtyn(gap_df: pd.DataFrame, fred_panel: pd.DataFrame) -> dict:
    """
    Cross-check IEF implied vol vs FRED VXTYN on 2010-2020 overlap.
    Correlation and level scale diagnostics.  A LARGE discrepancy flags a Layer 3 bug.
    """
    print("\n[Layer 5] Cross-check: IEF iv_event_vol_pct vs VXTYN ...")
    result = {"corr": np.nan, "level_scale": np.nan, "n": 0}

    ief_df = gap_df[
        (gap_df["etf"] == "IEF") & gap_df["iv_event_vol_pct"].notna()
    ][["meeting_date","iv_event_vol_pct"]].copy()

    if "vxtyn" not in fred_panel.columns or ief_df.empty:
        print("  Skipped: no IEF IV or VXTYN data")
        return result

    vxtyn = fred_panel["vxtyn"].dropna()
    rows  = []
    for _, r in ief_df.iterrows():
        fomc_dt = pd.Timestamp(r["meeting_date"])
        # VXTYN ~ 30-day constant horizon; match to 5 days before FOMC
        obs_win = vxtyn.loc[fomc_dt - pd.Timedelta(days=7):fomc_dt - pd.Timedelta(days=1)]
        if obs_win.empty:
            continue
        rows.append({"meeting_date": fomc_dt,
                     "ief_iv_pct":  r["iv_event_vol_pct"],
                     "vxtyn":       float(obs_win.mean())})

    if not rows:
        print("  No overlap found")
        return result

    cdf = pd.DataFrame(rows)
    cdf = cdf[(cdf["meeting_date"] >= "2010-01-01") &
              (cdf["meeting_date"] <= "2020-05-15")]
    if len(cdf) < 5:
        print(f"  Only {len(cdf)} overlapping points — insufficient")
        return result

    corr  = float(cdf["ief_iv_pct"].corr(cdf["vxtyn"]))
    scale = float(cdf["ief_iv_pct"].mean() / cdf["vxtyn"].mean()) if cdf["vxtyn"].mean() else np.nan

    result.update({"corr": corr, "level_scale": scale, "n": len(cdf)})
    status = "OK" if corr > 0.7 and 0.5 < scale < 2.0 else "WARN — possible IV extraction bug"
    print(f"  IEF vs VXTYN: corr={corr:.2f}  level_scale={scale:.2f}  n={len(cdf)}  [{status}]")
    if corr < 0.7:
        print("  *** LOW CORRELATION: check Layer 3 IV extraction for IEF ***")
    if not (0.5 < scale < 2.0):
        print(f"  *** LEVEL MISMATCH (scale={scale:.2f}): check ETF vs futures unit basis ***")
    return result


def cross_check_vxtlt(gap_df: pd.DataFrame) -> dict:
    """
    Cross-check TLT implied vol vs Cboe VXTLT (2024+, long-end).
    Correlation only (level basis expected due to ETF vs index construction).
    """
    print("\n[Layer 5] Cross-check: TLT iv_event_vol_pct vs CBOE VXTLT ...")
    result = {"corr": np.nan, "n": 0}

    tlt_df = gap_df[
        (gap_df["etf"] == "TLT") & gap_df["iv_event_vol_pct"].notna()
    ][["meeting_date","iv_event_vol_pct"]].copy()

    vxtlt = _fetch_cboe_csv(_VXTLT_URLS, "vxtlt.csv")
    if vxtlt.empty or tlt_df.empty:
        print("  Skipped: missing TLT IV or VXTLT data")
        return result

    close_col = [c for c in vxtlt.columns if "close" in c.lower() or "CLOSE" in c]
    if not close_col:
        close_col = [vxtlt.columns[0]]
    vxtlt_series = vxtlt[close_col[0]].dropna().astype(float)

    rows = []
    for _, r in tlt_df.iterrows():
        fomc_dt = pd.Timestamp(r["meeting_date"])
        obs_win = vxtlt_series.loc[fomc_dt - pd.Timedelta(days=7):fomc_dt - pd.Timedelta(days=1)]
        if obs_win.empty:
            continue
        rows.append({"meeting_date": fomc_dt,
                     "tlt_iv_pct": r["iv_event_vol_pct"],
                     "vxtlt": float(obs_win.mean())})

    if len(rows) < 3:
        print(f"  Only {len(rows)} overlapping points (VXTLT is very recent)")
        return result

    cdf  = pd.DataFrame(rows)
    corr = float(cdf["tlt_iv_pct"].corr(cdf["vxtlt"]))
    result.update({"corr": corr, "n": len(cdf)})
    print(f"  TLT vs VXTLT: corr={corr:.2f}  n={len(cdf)}  "
          f"[{'OK' if corr > 0.6 else 'LOW — verify TLT IV extraction'}]")
    return result


def print_coverage_table(gap_df: pd.DataFrame) -> None:
    """
    Print coverage table: by YEAR × ETF, how many meetings have RV / IV / both.
    This is the headline deliverable for assessing post-2020 gap coverage.
    """
    df = gap_df.copy()
    df["year"] = pd.to_datetime(df["meeting_date"]).dt.year

    print("\n" + "═" * 72)
    print("  COVERAGE TABLE — meetings with usable RV, IV, and gap (per ETF per year)")
    print("═" * 72)

    etfs = sorted(df["etf"].unique())
    years = sorted(df["year"].unique())

    # Header
    header = f"  {'Year':>4} | " + " | ".join(f"{e:^7}" for e in etfs) + " | Total"
    print(header)
    print("  " + "-" * (len(header) - 2))

    total_both = 0
    for yr in years:
        yr_df = df[df["year"] == yr]
        meetings_this_year = yr_df["meeting_date"].nunique()
        parts  = []
        yr_any = 0
        for etf in etfs:
            sub = yr_df[yr_df["etf"] == etf]
            n_both = sub["has_both"].sum() if "has_both" in sub.columns else 0
            n_rv   = sub["has_rv"].sum()   if "has_rv"   in sub.columns else 0
            parts.append(f"{n_both}/{n_rv:>1}")
            if n_both > 0:
                yr_any += 1
        yr_total = yr_df["has_both"].sum() if "has_both" in yr_df.columns else 0
        total_both += yr_total
        print(f"  {yr:>4} | " + " | ".join(f"{p:^7}" for p in parts) +
              f" | {yr_total:>4}/{meetings_this_year}")

    print("  " + "-" * (len(header) - 2))
    print(f"  {'TOTAL':>4} | {'':^{len(header) - 18}} | {total_both:>4}")
    print("  Format: BOTH/RV  (BOTH = has RV and IV; RV = has realized vol only)")
    print("═" * 72)

    # Highlight post-2020 (no VXTYN)
    post2020 = gap_df[pd.to_datetime(gap_df["meeting_date"]).dt.year >= 2020]
    n_post_both = post2020["has_both"].sum() if "has_both" in post2020.columns else 0
    n_post_total = post2020["meeting_date"].nunique()
    print(f"\n  Post-2020 (no TYVIX): {n_post_both} rows with RV+IV across "
          f"{n_post_total} meetings × ETFs")
    if n_post_both == 0:
        print("  *** ZERO post-2020 IV: pipeline needs AlphaVantage key or manual CSV fill ***")


# ─────────────────────────────────────────────────────────────────────────────
# OPTIONAL: Bachelier lognormal → normal vol converter  (reuses fomc_straddle_sim)
# ─────────────────────────────────────────────────────────────────────────────

def lognormal_to_normal(F: float, K: float, T: float,
                         sigma_ln: float, is_call: bool = True) -> float:
    """
    Convert lognormal (Black-Scholes) implied vol to Bachelier normal vol.
    Uses the in-house bachelier pricer from fomc_straddle_sim by solving:
      bachelier_price(F, K, T, σ_N) = black76_price(F, K, T, σ_LN)
    via a simple bisection over σ_N.
    """
    from fomc_straddle_sim import bachelier as _bach

    # Black-76 ATM price (for ATM: d1 = σ_LN×sqrt(T)/2)
    from scipy.stats import norm
    d1   = 0.5 * sigma_ln * sqrt(T)
    d2   = -d1
    call_px = F * norm.cdf(d1) - K * norm.cdf(d2)   # assume df=1 (FOMC short tenor)
    target  = call_px if is_call else call_px - (F - K)

    # Bisect σ_N in [0, F*2]
    lo, hi = 1e-6, F * 2.0
    for _ in range(60):
        mid  = (lo + hi) / 2
        p    = _bach(F, K, T, mid, call=is_call)["price"]
        if p < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — full pipeline run
# ─────────────────────────────────────────────────────────────────────────────

CAVEATS = """
CAVEATS (printed once at runtime):
  1. PROXY BASIS: ETF options proxy Treasury vol with a basis.  Duration of SHY/IEF/TLT
     differs from the exact 2Y/10Y/30Y benchmark because the ETF holds a rolling basket.
     Also: ETF-specific flows and management fees create idiosyncratic vol not present
     in futures (ZT/ZB/ZN).  Signal is trained on ETFs; live trade expressed on futures.
     Flag: basis_flag column = "signal=ETF_proxy; trade=ZT/ZB_futures; ..."

  2. ALPHAVANTAGE RATE LIMITS: free tier = 25 requests/day.  Pipeline is restartable
     (every response cached by ticker+date).  With 118 meetings × 5 ETFs = 590+ calls,
     full coverage takes ~24 days on free tier.  Premium (~$50/mo) allows 75 req/min
     and covers the full history in ~2 hours.

  3. EVENT ISOLATION APPROXIMATION: iv_event_var = σ_post² × T_post - σ_pre² × T_pre.
     Assumes diffusive vol is constant between observation date and expiry.  In practice
     the strip can be negative (rare: usually when IV term structure is inverted ahead of
     FOMC).  Negative values are set to NaN in the output.

  4. WINDOW FLAG: where SHY/IEI have only monthly expiries (not weekly), the event
     isolation window spans ~30 days.  This dilutes the FOMC-specific var extraction.
     window_flag='weekly' (TLT) is more precise than window_flag='monthly' (SHY old).

  5. LOGNORMAL CONVENTION: both GK estimator and AlphaVantage IV are in lognormal
     (Black-Scholes) space.  The Bachelier converter is available via lognormal_to_normal()
     for downstream use in the futures straddle simulator (fomc_straddle_sim.py uses
     Bachelier normal vol).

  6. VXTYN (FRED): discontinued 2020-05-15.  Used ONLY as a pre-2020 cross-check for
     IEF IV.  NOT used as a gap source post-2020.  All post-2020 gaps come from ETF
     options (Layer 3).
"""


def run(fomc_dates: list[pd.Timestamp] | None = None,
        skip_fred: bool = False) -> pd.DataFrame:
    """
    End-to-end pipeline run.  Returns the merged gap DataFrame and writes
    etf_gap_curve.parquet.
    """
    print(CAVEATS)

    # ── Calendar ──────────────────────────────────────────────────────────────
    if fomc_dates is None:
        fomc_dates = load_fomc_dates()

    # ── Layer 1: FRED ──────────────────────────────────────────────────────────
    fred_panel = pd.DataFrame()
    if not skip_fred:
        fred_panel = build_fred_panel()

    # ── Layer 2: Realized vol ──────────────────────────────────────────────────
    ohlc_cache: dict[str, pd.DataFrame] = {}
    for ticker in ETF_MAP:
        ohlc_cache[ticker] = fetch_etf_ohlc(ticker)

    realized = build_realized_curve(fomc_dates, ohlc_cache=ohlc_cache)

    # ── Layer 3: Implied vol ───────────────────────────────────────────────────
    implied = build_implied_curve(fomc_dates, ohlc_cache=ohlc_cache)
    implied = add_iv_percentile(implied)

    # ── Layer 4: Gap ────────────────────────────────────────────────────────────
    gap_df = merge_gap_curve(realized, implied)
    gap_df = add_gap_spread(gap_df)
    gap_df = add_proxy_basis_flag(gap_df)

    # Clip negative iv_event_var to NaN (event isolation approximation artifact)
    neg_iv = (gap_df["iv_event_var"] < 0).sum()
    if neg_iv > 0:
        print(f"\n[Layer 4] {neg_iv} rows with negative iv_event_var (inverted term structure)"
              f" → set to NaN")
        gap_df.loc[gap_df["iv_event_var"] < 0, ["iv_event_var","gap"]] = np.nan

    # ── Layer 5: Validation ────────────────────────────────────────────────────
    if not fred_panel.empty:
        cross_check_vxtyn(gap_df, fred_panel)
    cross_check_vxtlt(gap_df)

    # ── Coverage table ─────────────────────────────────────────────────────────
    print_coverage_table(gap_df)

    # ── Export ────────────────────────────────────────────────────────────────
    export_cols = [
        "meeting_date","etf","tenor_proxy",
        "rv_event_var","rv_event_park_var",
        "rv_fwd_1d_var","rv_fwd_5d_var","rv_fwd_10d_var",
        "iv_event_var","iv_event_vol_pct","iv_percentile",
        "gap","has_both","source","window_flag",
        "GapSpread_proxy","spread_direction","basis_flag",
    ]
    export_cols = [c for c in export_cols if c in gap_df.columns]
    out = gap_df[export_cols].copy()
    out.to_parquet(OUT_PARQ, index=False)
    n_both = out["has_both"].sum() if "has_both" in out.columns else 0
    print(f"\n[done] Written {len(out)} rows ({n_both} with both RV+IV) → {OUT_PARQ}")
    return gap_df


if __name__ == "__main__":
    run()
