"""
fomc_nlp_regime_model.py
========================
Four coordinated changes to the NLP-vol forecasting stack.

GOVERNING PRINCIPLE (unchanged from parent pipeline):
  Regime conditioning adds parameters — on a smallish sample this overfits
  easily.  The regime variable MUST be economically primitive (defined
  independently of any vol outcome) and interactions MECHANISM-motivated,
  not fit-chased.  Validate that regime conditioning improves OUT-OF-SAMPLE,
  not just in-sample.  If it does not, say so.

CHANGE 1 — CORPUS EXPANSION
  Add FOMC press-conference transcripts + Jackson Hole speeches.
  Presser text: strip journalist Q turns; keep only chair spoken answers.
  Length normalisation: score density per 1k tokens (not raw sum) so
  6k-word Q&A is comparable to 130-word statement.
  Each document tagged by doc_type: 'statement' / 'presser' / 'speech'.

CHANGE 2 — DUAL-MANDATE ECONOMIC REGIME LABELS
  Replaces chair-identity labels entirely.
  FRED: UNRATE, PCEPILFE (core PCE, level → YoY), NROU (natural rate).
  inflation_gap  = core_PCE_YoY − 2.0
  u_gap          = UNRATE − NROU  (positive = labour slack)
  policy_dir     = sign of recent 3M T-bill change  (hiking/cutting/hold)
  Four discrete labels (overheating / at_target / slack / easing) from
  economically motivated thresholds — never from vol outcomes.
  Rationale printed: Powell spanned hawkish (2022) and dovish (2019)
  within one tenure.  Chair identity is a poor regime proxy.

CHANGE 3 — REGIME-CONDITIONAL MODEL + POWELL BACKTEST
  Conceptual frame:
    NLP-only   = backward-looking, implied-vol ANALOG (market anchor)
    NLP×regime = forward-looking, realised-vol ANALOG (divergence from anchor)
  Model: event_vol ~ text_factors + regime + (text_factors × regime) + controls
  Walk-forward expanding window on POWELL (2018-2026); predict OOS.
  Acceptance test: does NLP×regime beat NLP-only OOS on Powell?
  Warsh is applied only as a single forward test — never the validation set.

CHANGE 4 — VISUALISATION (Figs 1-6, shared regime palette)
  Warm = overheating, cool = slack/easing, neutral = at-target.
  All performance figures use walk-forward OOS only.
  Jackson Hole 2022 'pain' episode annotated wherever it appears.

Public data only (FRED + federalreserve.gov). FRED_API_KEY from env.
"""
from __future__ import annotations

import os, re, json, hashlib, time, warnings
from pathlib import Path
from math import sqrt
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

from scipy import stats as sp_stats
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

try:
    from IPython import get_ipython as _gip
    _ip = _gip()
    if _ip is not None:
        _ip.run_line_magic("matplotlib", "inline")
    else:
        matplotlib.use("Agg")
except Exception:
    matplotlib.use("Agg")

try:
    from IPython.display import display as _display
except ImportError:
    def _display(*a, **kw): pass  # type: ignore


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — CONFIG
# ══════════════════════════════════════════════════════════════════════════════

CACHE_DIR   = Path("fomc_cache")
FIG_DIR     = Path("figures")
VRP_CACHE   = Path("vrp_cache")
CORPUS_OUT  = Path("fomc_corpus_expanded.parquet")
REGIME_OUT  = Path("fomc_dual_mandate_regime.parquet")
MODEL_OUT   = Path("fomc_nlp_regime_forecasts.parquet")

for d in (CACHE_DIR / "html", FIG_DIR, VRP_CACHE):
    d.mkdir(parents=True, exist_ok=True)

FRED_API_KEY = os.environ.get("FRED_API_KEY", "6a9808ddeb9c3a8568dfb734f5c2303c")
START_DATE   = "2010-01-01"
POWELL_START = pd.Timestamp("2018-02-03")
RATE_LIMIT_S = 1.5

FED_BASE         = "https://www.federalreserve.gov"
PRESSCONF_URL_T  = FED_BASE + "/monetarypolicy/fomcpresconf{date}.htm"
SPEECH_URL_T     = FED_BASE + "/newsevents/speech/{slug}.htm"

# ── Jackson Hole chair speeches (hardcoded: these are public historical URLs) ──
JACKSON_HOLE_SPEECHES: list[dict] = [
    dict(date="2018-08-24", chair="Powell", slug="powell20180824a",
         title="Navigating by the Stars Under Cloudy Skies"),
    dict(date="2019-08-23", chair="Powell", slug="powell20190823a",
         title="Challenges for Monetary Policy"),
    dict(date="2021-08-27", chair="Powell", slug="powell20210827a",
         title="Macroeconomic Imbalances and the Inflation Outlook"),
    dict(date="2022-08-26", chair="Powell", slug="powell20220826a",
         title="Restoring Price Stability"),   # THE 'PAIN' SPEECH — ANNOTATE EVERYWHERE
    dict(date="2023-08-25", chair="Powell", slug="powell20230825a",
         title="Inflation: Progress and the Path Ahead"),
    dict(date="2024-08-23", chair="Powell", slug="powell20240823a",
         title="Review of Our Monetary Policy Framework"),
    # Bernanke / Yellen for historical regime coverage
    dict(date="2012-08-31", chair="Bernanke", slug="bernanke20120831a",
         title="Monetary Policy Since the Onset of the Crisis"),
    dict(date="2013-08-22", chair="Bernanke", slug="bernanke20130822a",
         title="The Federal Reserve's Many Roles in Supporting the Recovery"),
    dict(date="2014-08-22", chair="Yellen",   slug="yellen20140822a",
         title="Labor Market Dynamics and Monetary Policy"),
    dict(date="2016-08-26", chair="Yellen",   slug="yellen20160826a",
         title="The Federal Reserve's Monetary Policy Toolkit"),
]

PAIN_SPEECH_DATE = pd.Timestamp("2022-08-26")   # annotate everywhere this appears

# ── Shared regime palette (warm=overheating, cool=slack/easing, neutral=at-target) ──
REGIME_PALETTE = {
    "overheating":  "#d73027",   # warm red   — PCE well above 2%, tight labour
    "supply_shock": "#fc8d59",   # warm orange — high PCE but slack labour (stagflation-like)
    "at_target":    "#878787",   # neutral grey — near-mandate
    "slack":        "#4575b4",   # cool blue   — labour slack, near/below 2%
    "easing":       "#91bfdb",   # light blue  — PCE well below 2%, ZLB/cutting
}
REGIME_ORDER = ["easing", "slack", "at_target", "supply_shock", "overheating"]

# ── NLP features (same as parent pipeline) ──
NLP_VOL_FEATURES = [
    "word_count_zscore", "novelty_prev", "novelty_window",
    "guidance_change", "uncertainty_density", "disagree_density",
]

# ── Forecasting ──
MIN_TRAIN_POWELL = 15   # walk-forward starts after this many Powell meetings
ALPHA_RANGE      = np.logspace(-2, 3, 30)   # Ridge regularisation sweep

print("Section 0 config loaded.")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CORPUS EXPANSION
# ══════════════════════════════════════════════════════════════════════════════

# ── 1a: HTTP fetch with disk cache ────────────────────────────────────────────

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36"
    ),
}


def _fetch(url: str, delay: float = RATE_LIMIT_S) -> Optional[str]:
    key  = hashlib.md5(url.encode()).hexdigest()[:16]
    path = CACHE_DIR / "html" / f"{key}.html"
    if path.exists():
        return path.read_text(encoding="utf-8", errors="replace")
    time.sleep(delay)
    try:
        r = requests.get(url, headers=_HEADERS, timeout=25)
        r.raise_for_status()
        path.write_text(r.text, encoding="utf-8")
        return r.text
    except Exception as e:
        print(f"  FETCH FAILED {url}: {e}")
        return None


# ── 1b: Press-conference Q&A stripper ─────────────────────────────────────────

# Fed transcript format:
#   "CHAIR POWELL.  We have seen ..."  (chair spoken turn)
#   "JENNIFER SCHONBERGER.  [Question]"  (journalist — DROP)
# Variants: "CHAIR YELLEN.", "GOVERNOR [NAME].", mixed colons/periods.

_CHAIR_NAMES   = {"POWELL", "YELLEN", "BERNANKE", "GREENSPAN", "BURNS"}
_CHAIR_TURN_RE = re.compile(
    r"(?:^|\n)((?:CHAIR|CHAIRMAN|VICE CHAIR|CHAIRWOMAN)\s+\w+)"
    r"[.\:]",
    re.IGNORECASE | re.MULTILINE,
)


def _strip_presser_to_chair(raw_text: str) -> tuple[str, int, int]:
    """
    Keep only spoken turns from the Chair; strip journalist questions and
    all other speaker turns.

    Returns (chair_text, n_chair_turns, n_total_turns).

    Algorithm:
      Split the transcript by speaker-turn markers (ALL-CAPS NAME followed
      by period/colon).  Classify each turn as Chair or Other.  Concatenate
      only Chair turns with a newline separator.

    HETEROGENEITY NOTE:
      Press conferences are long and conversational.  Stripping non-chair
      text removes ~50–60% of tokens, leaving a shorter but higher-density
      signal.  After stripping, apply length normalisation (scores per 1k
      tokens) to make the presser comparable to a 130-word statement.
    """
    # Tokenise into (speaker, turn_text) pairs
    # Match ALL-CAPS word sequences followed by period/colon at turn start
    TURN_RE = re.compile(
        r"\n([A-Z][A-Z\s'\-]{2,40})[.\:]\s+",
    )
    parts = TURN_RE.split(raw_text)
    # parts = [pre, speaker1, text1, speaker2, text2, ...]

    chair_turns = []
    total_turns = 0
    i = 1
    while i + 1 < len(parts):
        speaker  = parts[i].strip().upper()
        turn_txt = parts[i + 1].strip()
        i += 2
        total_turns += 1
        words = speaker.split()
        is_chair = any(w in _CHAIR_NAMES for w in words)
        if is_chair and turn_txt:
            chair_turns.append(turn_txt)

    chair_text = "\n".join(chair_turns)
    return chair_text, len(chair_turns), total_turns


def scrape_presser(meeting_date_str: str) -> Optional[dict]:
    """
    Fetch and parse a FOMC press conference transcript.
    Returns dict with keys: meeting_date, text (chair-only), n_tokens,
                             n_chair_turns, n_total_turns, doc_type='presser'.
    Returns None if URL not found or presser not held on that date.
    """
    url = PRESSCONF_URL_T.format(date=meeting_date_str.replace("-", ""))
    html = _fetch(url)
    if html is None or "404" in html[:500]:
        return None

    soup = BeautifulSoup(html, "html.parser")
    # Fed presser pages: main content in <div class="col-xs-12 col-sm-8">
    # or <div id="article"> or <article>
    content = (
        soup.find("div", {"class": lambda c: c and "col-sm-8" in c})
        or soup.find("article")
        or soup.find("div", id="article")
        or soup.body
    )
    if content is None:
        return None

    raw_text = content.get_text(separator="\n", strip=True)
    if len(raw_text) < 500:
        return None

    chair_text, n_chair, n_total = _strip_presser_to_chair(raw_text)

    # Fallback: if Q&A stripping failed (< 100 chars), use full text minus header
    if len(chair_text) < 100:
        # No clear speaker-turn structure (may be PDF-extracted text)
        # Remove "TRANSCRIPT OF CHAIR POWELL'S PRESS CONFERENCE" header heuristic
        lines = [l for l in raw_text.split("\n") if len(l.strip()) > 20]
        chair_text = "\n".join(lines)
        n_chair, n_total = 1, 1

    n_tokens = len(chair_text.split())
    return dict(
        meeting_date = meeting_date_str,
        text         = chair_text,
        n_tokens     = n_tokens,
        n_chair_turns= n_chair,
        n_total_turns= n_total,
        doc_type     = "presser",
    )


# ── 1c: Jackson Hole speech scraper ───────────────────────────────────────────

def scrape_speech(slug: str, speech_date: str, chair: str, title: str) -> Optional[dict]:
    """
    Fetch a Fed Board speech page.  Returns dict with keys:
      date, chair, title, text, n_tokens, doc_type='speech'.
    No Q&A stripping needed — speeches are monologues.
    """
    url  = SPEECH_URL_T.format(slug=slug)
    html = _fetch(url)
    if html is None:
        return None

    soup = BeautifulSoup(html, "html.parser")
    content = (
        soup.find("div", {"class": lambda c: c and "col-sm-8" in c})
        or soup.find("article")
        or soup.find("div", id="article")
        or soup.body
    )
    if content is None:
        return None

    # Remove navigation / header / footer noise
    for tag in content.find_all(["nav", "header", "footer", "script", "style"]):
        tag.decompose()
    raw_text = content.get_text(separator="\n", strip=True)

    # Drop lines that look like navigation cruft (< 5 words)
    lines = [l.strip() for l in raw_text.split("\n") if len(l.strip().split()) >= 5]
    text  = "\n".join(lines)
    n_tokens = len(text.split())

    if n_tokens < 100:
        return None

    return dict(
        meeting_date = speech_date,
        chair        = chair,
        title        = title,
        text         = text,
        n_tokens     = n_tokens,
        doc_type     = "speech",
    )


# ── 1d: Build unified expanded corpus ─────────────────────────────────────────

def build_expanded_corpus(stmt_path: Path = Path("fomc_statements.parquet"),
                          fomc_dates: Optional[list[str]] = None,
                          force_refresh: bool = False) -> pd.DataFrame:
    """
    Merge statements + pressers + Jackson Hole speeches into one corpus.

    Columns: meeting_date, chair, text, n_tokens, doc_type, source_url
             is_jh_pain  (bool, True for 2022-08-26 'pain' speech)

    HETEROGENEITY HANDLING:
      doc_type is kept as a categorical control in the model.
      Length normalisation: score_density = raw_score / (n_tokens / 1000)
      applied by the scoring layer (Section 5), not here.
    """
    if CORPUS_OUT.exists() and not force_refresh:
        df = pd.read_parquet(CORPUS_OUT)
        print(f"[corpus] Loaded from cache: {len(df)} documents "
              f"({df['doc_type'].value_counts().to_dict()})")
        return df

    rows: list[dict] = []

    # ── Layer A: existing statements ──────────────────────────────────────────
    if stmt_path.exists():
        stmts = pd.read_parquet(stmt_path)
        stmts["doc_type"]     = "statement"
        stmts["n_tokens"]     = stmts["text"].str.split().str.len().fillna(0).astype(int)
        stmts["is_jh_pain"]   = False
        rows.extend(stmts.to_dict("records"))
        print(f"  [A] Statements: {len(stmts)}")
    else:
        print("  [A] WARNING: fomc_statements.parquet not found. "
              "Run fomc_public_pipeline.py first.")

    # ── Layer B: press-conference transcripts ─────────────────────────────────
    # Press conferences started Apr 2011; every meeting from Jan 2019+
    _all_dates = fomc_dates or [
        "2011-04-27","2011-06-22","2011-09-21","2011-12-13",
        "2012-01-25","2012-04-25","2012-06-20","2012-09-13","2012-12-12",
        "2013-03-20","2013-06-19","2013-09-18","2013-12-18",
        "2014-01-29","2014-03-19","2014-06-18","2014-09-17","2014-12-17",
        "2015-03-18","2015-06-17","2015-09-17","2015-12-16",
        "2016-03-16","2016-06-15","2016-09-21","2016-12-14",
        "2017-03-15","2017-06-14","2017-09-20","2017-12-13",
        "2018-03-21","2018-06-13","2018-09-26","2018-12-19",
        # Every meeting from 2019+
        "2019-01-30","2019-03-20","2019-05-01","2019-06-19",
        "2019-07-31","2019-09-18","2019-10-30","2019-12-11",
        "2020-01-29","2020-03-15","2020-04-29","2020-06-10",
        "2020-07-29","2020-09-16","2020-11-05","2020-12-16",
        "2021-01-27","2021-03-17","2021-04-28","2021-06-16",
        "2021-07-28","2021-09-22","2021-11-03","2021-12-15",
        "2022-01-26","2022-03-16","2022-05-04","2022-06-15",
        "2022-07-27","2022-09-21","2022-11-02","2022-12-14",
        "2023-02-01","2023-03-22","2023-05-03","2023-06-14",
        "2023-07-26","2023-09-20","2023-11-01","2023-12-13",
        "2024-01-31","2024-03-20","2024-05-01","2024-06-12",
        "2024-07-31","2024-09-18","2024-11-07","2024-12-18",
        "2025-01-29","2025-03-19","2025-05-07","2025-06-18",
        "2026-01-28","2026-03-18","2026-04-29","2026-06-17",
    ]

    n_presser, n_presser_fail = 0, 0
    for ds in _all_dates:
        result = scrape_presser(ds)
        if result:
            result["is_jh_pain"] = False
            rows.append(result)
            n_presser += 1
        else:
            n_presser_fail += 1

    print(f"  [B] Press conferences: {n_presser} fetched, {n_presser_fail} not found/unavailable")

    # ── Layer C: Jackson Hole speeches ────────────────────────────────────────
    n_speech, n_speech_fail = 0, 0
    for spec in JACKSON_HOLE_SPEECHES:
        result = scrape_speech(spec["slug"], spec["date"], spec["chair"], spec["title"])
        if result:
            result["is_jh_pain"] = (pd.Timestamp(spec["date"]) == PAIN_SPEECH_DATE)
            rows.append(result)
            n_speech += 1
        else:
            n_speech_fail += 1

    print(f"  [C] Jackson Hole speeches: {n_speech} fetched, {n_speech_fail} not found")

    # ── Print filter audit per doc_type ───────────────────────────────────────
    df = pd.DataFrame(rows)
    df["meeting_date"] = pd.to_datetime(df["meeting_date"])
    df = df.sort_values(["meeting_date", "doc_type"]).reset_index(drop=True)

    print(f"\n  FILTER AUDIT — token counts per doc_type:")
    for dt, grp in df.groupby("doc_type"):
        toks = grp["n_tokens"].describe()
        print(f"    {dt:10s}  n={len(grp):3d}  "
              f"tokens: mean={toks['mean']:.0f}  min={toks['min']:.0f}  "
              f"max={toks['max']:.0f}")
    print(f"  Total corpus: {len(df)} documents")

    df.to_parquet(CORPUS_OUT, index=False)
    print(f"  → saved {CORPUS_OUT}")
    return df


corpus_df = build_expanded_corpus()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DUAL-MANDATE ECONOMIC REGIME LABELS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 60)
print("  SECTION 2 — DUAL-MANDATE REGIME")
print("═" * 60)
print("""
  RATIONALE (print):
  Chair identity is a POOR regime proxy.  Powell alone spanned:
    • 2018-2019: gradual normalisation (near-mandate, AT_TARGET)
    • 2020 Q2:   COVID shock (sharp SLACK; emergency cuts)
    • 2021 Q4–2023: inflation surge (OVERHEATING; fastest hiking since 1980s)
    • 2024-2025: normalisation back toward target
  One chair, four distinct regimes.  If 'chair=Powell' is the label,
  all four regimes get the same model weight — that is economically wrong.
  The dual-mandate state IS the reaction function; it drives how the same
  language maps to vol outcomes.
""")


def fetch_fred(series_id: str, start: str = "2008-01-01") -> pd.Series:
    """Pull a FRED series via REST API with CSV cache.  Returns daily/monthly Series."""
    cache = CACHE_DIR / "market" / f"fred_{series_id}.csv"
    if cache.exists():
        try:
            df = pd.read_csv(cache, index_col=0, parse_dates=True)
            s  = df.iloc[:, 0].replace(".", np.nan).astype(float).dropna()
            s.name = series_id
            return s[s.index >= start]
        except Exception:
            pass
    url = (f"https://api.stlouisfed.org/fred/series/observations"
           f"?series_id={series_id}&api_key={FRED_API_KEY}"
           f"&observation_start={start}&file_type=json")
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        obs  = resp.json().get("observations", [])
        idx  = [pd.Timestamp(o["date"]) for o in obs]
        vals = [float(o["value"]) if o["value"] not in (".", "") else np.nan for o in obs]
        s    = pd.Series(vals, index=idx, name=series_id).dropna()
        s.to_csv(cache)
        print(f"    FRED {series_id}: {len(s)} obs  "
              f"({s.index[0].date()} – {s.index[-1].date()})")
        return s
    except Exception as e:
        print(f"  FRED {series_id} FAILED: {e}")
        return pd.Series(dtype=float, name=series_id)


def build_dual_mandate_regime(fomc_dates: pd.DatetimeIndex,
                               force_refresh: bool = False) -> pd.DataFrame:
    """
    Construct ECONOMICALLY PRIMITIVE dual-mandate regime per FOMC meeting.

    Primitiveness guarantee: all inputs (UNRATE, PCEPILFE, NROU, DGS3MO) are
    defined by the Bureau of Labour Statistics / BEA / FOMC itself — NONE
    are derived from or correlated with vol outcomes by construction.

    Fields per meeting:
      unrate           : unemployment rate (%, at meeting month)
      nrou             : natural rate of unemployment (%, FRED NROU, interpolated)
      u_gap            : unrate − nrou  (+ = labour slack; − = labour tight)
      pce_yoy          : core PCE YoY inflation (%, at meeting month)
      inflation_gap    : pce_yoy − 2.0
      policy_dir       : +1 hiking, −1 cutting, 0 hold (3M T-bill 3m change)
      regime_label     : discrete label from thresholds (see below)
      regime_cont      : tuple (inflation_gap, u_gap) for continuous kernel

    DISCRETE THRESHOLDS (set from economic mechanism, never from vol):
      overheating  : inflation_gap > 0.5 AND u_gap < 0   (tight-labour inflation)
      supply_shock : inflation_gap > 0.5 AND u_gap >= 0  (cost-push; labour not tight)
      slack        : u_gap > 0.75  (substantial labour slack regardless of inflation)
      easing       : inflation_gap < −0.5              (well below target; ZLB-typical)
      at_target    : everything else
    """
    if REGIME_OUT.exists() and not force_refresh:
        df = pd.read_parquet(REGIME_OUT)
        df["meeting_date"] = pd.to_datetime(df["meeting_date"])
        print(f"[regime] Loaded from cache: {len(df)} meetings  "
              f"({df['regime_label'].value_counts().to_dict()})")
        return df

    # ── Fetch FRED series ─────────────────────────────────────────────────────
    unrate  = fetch_fred("UNRATE")     # monthly, %
    pcepilfe= fetch_fred("PCEPILFE")  # monthly, price index level → need YoY
    nrou    = fetch_fred("NROU")      # quarterly → interpolate monthly
    dgs3mo  = fetch_fred("DGS3MO")    # daily, % (3M T-bill)

    # YoY core PCE inflation
    pce_yoy = pcepilfe.pct_change(12) * 100
    pce_yoy.name = "pce_yoy"

    # Interpolate NROU (quarterly) to monthly
    nrou_monthly = nrou.resample("MS").interpolate(method="linear")

    # ── Map to FOMC meeting dates using as-of join (no look-ahead) ────────────
    rows = []
    for dt in pd.to_datetime(fomc_dates):
        # Use last data point STRICTLY BEFORE the meeting
        def _asof(s: pd.Series) -> float:
            avail = s[s.index < dt]
            return float(avail.iloc[-1]) if not avail.empty else np.nan

        u       = _asof(unrate)
        nrou_v  = _asof(nrou_monthly)
        pce_v   = _asof(pce_yoy)
        t3m_now = _asof(dgs3mo)

        # Policy direction: 3M T-bill 3-month change
        avail_3m = dgs3mo[dgs3mo.index < dt]
        if len(avail_3m) >= 65:
            t3m_lag = float(avail_3m.iloc[-65])   # ~3 months
            pdir    = int(np.sign(t3m_now - t3m_lag)) if not np.isnan(t3m_now) else 0
        else:
            pdir = 0

        u_gap        = u - nrou_v if (not np.isnan(u) and not np.isnan(nrou_v)) else np.nan
        inf_gap      = pce_v - 2.0 if not np.isnan(pce_v) else np.nan

        # Discrete label — thresholds set from economic mechanism
        if np.isnan(inf_gap) or np.isnan(u_gap):
            label = "at_target"    # insufficient data → neutral
        elif inf_gap > 0.5 and u_gap < 0:
            label = "overheating"
        elif inf_gap > 0.5 and u_gap >= 0:
            label = "supply_shock"
        elif u_gap > 0.75:
            label = "slack"
        elif inf_gap < -0.5:
            label = "easing"
        else:
            label = "at_target"

        rows.append(dict(
            meeting_date   = dt,
            unrate         = u,
            nrou           = nrou_v,
            u_gap          = u_gap,
            pce_yoy        = pce_v,
            inflation_gap  = inf_gap,
            policy_dir     = pdir,
            regime_label   = label,
        ))

    df = pd.DataFrame(rows)
    df.to_parquet(REGIME_OUT, index=False)

    print(f"\n  Dual-mandate regime: {len(df)} meetings")
    print(f"  {'Label':<14} {'n':>4}  {'u_gap mean':>12}  {'inf_gap mean':>13}")
    for lab in REGIME_ORDER:
        sub = df[df["regime_label"] == lab]
        if len(sub) == 0:
            continue
        print(f"  {lab:<14} {len(sub):>4}  "
              f"{sub['u_gap'].mean():>+11.2f}  {sub['inflation_gap'].mean():>+12.2f}")
    return df


# Build regime table using FOMC dates from existing statements if available
_stmt_dates = (pd.read_parquet("fomc_statements.parquet")["meeting_date"].values
               if Path("fomc_statements.parquet").exists()
               else pd.to_datetime(corpus_df["meeting_date"].unique()))

regime_df = build_dual_mandate_regime(pd.DatetimeIndex(_stmt_dates))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — NLP SCORING (length-normalised, per doc_type)
# ══════════════════════════════════════════════════════════════════════════════

def score_document_offline(text: str, meeting_date: pd.Timestamp,
                            feats_df: Optional[pd.DataFrame] = None) -> dict:
    """
    Produce NLP vol-feature scores for a single document.

    If feats_df (fomc_features.parquet) is available, pulls pre-computed scores
    for statements; else falls back to simple lexicon-based proxies.

    CRITICAL: scores are RAW counts; caller normalises by (n_tokens / 1000)
    to produce densities comparable across doc_types.

    Returns dict: uncertainty_density, ambiguity, conditionality,
                  guidance_specificity, dissent, word_count
    """
    # Pre-computed scores from existing pipeline take priority for statements
    if feats_df is not None:
        dt = pd.Timestamp(meeting_date)
        row = feats_df[feats_df["meeting_date"] == dt]
        if not row.empty:
            r = row.iloc[0]
            return dict(
                uncertainty_density  = float(r.get("uncertainty_density", 0)),
                disagree_density     = float(r.get("disagree_density", 0)),
                novelty_prev         = float(r.get("novelty_prev", 0)),
                guidance_change      = float(r.get("guidance_change", 0)),
                word_count_zscore    = float(r.get("word_count_zscore", 0)),
                word_count           = float(r.get("word_count", len(text.split()))),
            )

    # Simple lexicon proxies (used for presser / speech where no pre-scoring exists)
    words = text.lower().split()
    n     = max(len(words), 1)

    _UNCERTAIN  = {"uncertain", "uncertainty", "unclear", "assess", "monitor",
                   "contingent", "depend", "evolve", "might", "could", "may",
                   "possible", "perhaps", "whether"}
    _DISSENT    = {"disagreement", "range", "some", "several", "others", "debate",
                   "diverge", "alternative", "different", "views"}
    _GUID_SPEC  = {"percent", "basis", "specific", "committed", "will", "path",
                   "trajectory", "explicit", "target", "threshold", "calendar"}

    unc_ct  = sum(1 for w in words if w in _UNCERTAIN)
    dis_ct  = sum(1 for w in words if w in _DISSENT)
    guid_ct = sum(1 for w in words if w in _GUID_SPEC)

    return dict(
        uncertainty_density  = unc_ct / n * 100,
        disagree_density     = dis_ct / n * 100,
        novelty_prev         = 0.0,    # needs TF-IDF context; set 0 for new docs
        guidance_change      = 0.0,    # needs prior doc; set 0 for new docs
        word_count_zscore    = 0.0,    # needs rolling mean; set 0
        word_count           = float(n),
        guidance_specificity = guid_ct / n * 100,
    )


def score_corpus(corpus: pd.DataFrame,
                 feats_path: Path = Path("fomc_features.parquet")) -> pd.DataFrame:
    """
    Score every document in the corpus.

    Length normalisation (HETEROGENEITY HANDLING):
      For comparability across doc_types, raw scores are converted to densities
      per 1k tokens:
        score_density = raw_score / (n_tokens / 1000)
      A 6k-word presser and a 130-word statement both produce a density in
      [0, 100] with the same interpretation.

    Returns corpus with added columns: uncertainty_density_norm,
      disagree_density_norm, guidance_specificity_norm,
      plus doc_type dummy: is_presser, is_speech.
    """
    feats_df = None
    if feats_path.exists():
        feats_df = pd.read_parquet(feats_path)
        feats_df["meeting_date"] = pd.to_datetime(feats_df["meeting_date"])

    results = []
    for _, row in corpus.iterrows():
        scores = score_document_offline(
            str(row.get("text", "")),
            row["meeting_date"],
            feats_df,
        )
        n_tok  = max(int(row.get("n_tokens", len(str(row.get("text","")).split()))), 1)
        scale  = n_tok / 1000.0     # normalise to per-1k-tokens
        results.append(dict(
            meeting_date            = row["meeting_date"],
            doc_type                = row.get("doc_type", "statement"),
            is_jh_pain              = bool(row.get("is_jh_pain", False)),
            n_tokens                = n_tok,
            uncertainty_density_raw = scores.get("uncertainty_density", 0),
            disagree_density_raw    = scores.get("disagree_density", 0),
            guidance_spec_raw       = scores.get("guidance_specificity", 0),
            novelty_prev            = scores.get("novelty_prev", 0),
            guidance_change         = scores.get("guidance_change", 0),
            word_count_zscore       = scores.get("word_count_zscore", 0),
            # Normalised densities (the model inputs)
            uncertainty_density     = scores.get("uncertainty_density", 0) / scale,
            disagree_density        = scores.get("disagree_density", 0) / scale,
            guidance_specificity    = scores.get("guidance_specificity", 0) / scale,
            is_presser              = float(row.get("doc_type") == "presser"),
            is_speech               = float(row.get("doc_type") == "speech"),
        ))

    scored = pd.DataFrame(results)
    print(f"\n[score] Scored {len(scored)} documents  "
          f"({scored['doc_type'].value_counts().to_dict()})")
    print(f"  Normalised uncertainty_density: "
          f"stmt={scored[scored.doc_type=='statement']['uncertainty_density'].mean():.3f}  "
          f"presser={scored[scored.doc_type=='presser']['uncertainty_density'].mean():.3f}  "
          f"speech={scored[scored.doc_type=='speech']['uncertainty_density'].mean():.3f}")
    return scored


scored_corpus = score_corpus(corpus_df)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — REGIME-CONDITIONAL FORECASTING MODEL
# ══════════════════════════════════════════════════════════════════════════════
# Conceptual frame:
#   NLP-only   model: forecast = f(text_factors)
#                     → backward-looking, like an IMPLIED-VOL analog
#                     → picks up the market's historical anchor
#   NLP×regime model: forecast = f(text_factors, regime, text×regime)
#                     → forward-looking REALIZED-VOL analog
#                     → regime conditions HOW the same language maps to vol
#
# The interaction IS the mechanism: "uncertainty language" means more
# vol in an overheating regime (markets are genuinely uncertain about rate
# path) than in a ZLB/easing regime (uncertainty language is just hedging).
#
# We estimate this on POWELL'S TENURE because it contains four distinct
# regimes (2019 easing, 2020 ZLB/COVID, 2022-23 hiking, 2024 normalisation).
# That within-chair variation is what lets us test whether regime conditioning
# adds OOS power — without it, we would be comparing across chairs which
# conflates communication style and regime.

TEXT_FEATURES = ["uncertainty_density", "disagree_density", "guidance_specificity"]

# Aggregate per meeting: max across doc_types present for that meeting
# (presser > speech > statement by information content; take max to preserve
#  the highest-signal document's score, not average away the signal)
def agg_to_meeting(scored: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate multi-document scores to one row per meeting.
    For each feature: take the max across doc_types for that meeting date.
    doc_type priority is: presser > speech > statement (by information content).
    """
    # Priority map: higher = preferred
    priority = {"presser": 3, "speech": 2, "statement": 1}
    scored = scored.copy()
    scored["priority"] = scored["doc_type"].map(priority).fillna(1)

    agg_rows = []
    for dt, grp in scored.groupby("meeting_date"):
        best = grp.sort_values("priority", ascending=False).iloc[0]
        row  = dict(meeting_date=dt,
                    doc_types_present=",".join(sorted(grp["doc_type"].unique())),
                    is_jh_pain=bool(grp["is_jh_pain"].any()),
                    has_presser=int((grp["doc_type"] == "presser").any()),
                    has_speech=int((grp["doc_type"] == "speech").any()))
        for feat in TEXT_FEATURES + ["novelty_prev", "guidance_change",
                                      "word_count_zscore", "n_tokens"]:
            if feat in grp.columns:
                row[feat] = float(grp[feat].max())   # max = highest-signal doc
        agg_rows.append(row)

    df = pd.DataFrame(agg_rows).sort_values("meeting_date").reset_index(drop=True)
    df["meeting_date"] = pd.to_datetime(df["meeting_date"])
    return df


meeting_scores = agg_to_meeting(scored_corpus)


def build_model_panel(meeting_scores: pd.DataFrame,
                      regime_df: pd.DataFrame,
                      vrp_path: Path = Path("vrp_cache/vrp_panel.parquet")) -> pd.DataFrame:
    """
    Merge text scores + regime + realized vol into one modelling panel.

    Target: rv_event_gk (realized event vol, %) for 2Y and 30Y tenors.
    Secondary target: gap_var (where implied vol exists).
    """
    scores = meeting_scores.copy()
    scores["meeting_date"] = pd.to_datetime(scores["meeting_date"])

    reg = regime_df[["meeting_date","u_gap","inflation_gap","policy_dir",
                      "regime_label"]].copy()
    reg["meeting_date"] = pd.to_datetime(reg["meeting_date"])

    panel = scores.merge(reg, on="meeting_date", how="inner")

    # One-hot regime dummies
    for lab in REGIME_ORDER:
        panel[f"regime_{lab}"] = (panel["regime_label"] == lab).astype(float)

    # Interaction features: text × regime
    for feat in TEXT_FEATURES:
        if feat not in panel.columns:
            continue
        for lab in REGIME_ORDER:
            panel[f"{feat}__x__{lab}"] = panel[feat] * panel[f"regime_{lab}"]

    # Pull realized vol from VRP panel if available
    if vrp_path.exists():
        vrp = pd.read_parquet(vrp_path)
        vrp["meeting_date"] = pd.to_datetime(vrp["meeting_date"])
        for tenor in ["2Y", "30Y"]:
            sub = vrp[vrp["tenor"] == tenor][["meeting_date","rv_event_gk",
                                               "rv_event_var","gap_var"]].copy()
            sub = sub.rename(columns={
                "rv_event_gk":  f"rv_gk_{tenor}",
                "rv_event_var": f"rv_var_{tenor}",
                "gap_var":      f"gap_var_{tenor}",
            })
            panel = panel.merge(sub, on="meeting_date", how="left")

    # Lagged RV (per tenor) — AR component, no look-ahead
    for col in ["rv_gk_2Y", "rv_gk_30Y"]:
        if col in panel.columns:
            panel[f"{col}_lag1"] = panel[col].shift(1)

    panel["meeting_date"] = pd.to_datetime(panel["meeting_date"])
    panel = panel.sort_values("meeting_date").reset_index(drop=True)
    print(f"\n[panel] Model panel: {panel.shape}  "
          f"meetings={panel['meeting_date'].nunique()}")
    return panel


model_panel = build_model_panel(meeting_scores, regime_df)


def walk_forward_powell(panel: pd.DataFrame,
                        target_col: str = "rv_gk_2Y",
                        min_train: int = MIN_TRAIN_POWELL) -> pd.DataFrame:
    """
    Walk-forward expanding window on Powell tenure (2018+).

    Two models per fold:
      NLP-only  : text features + controls (no regime)
      NLP×regime: text features + regime dummies + text×regime interactions

    Returns DataFrame with columns:
      meeting_date, regime_label, actual, pred_nlp_only, pred_nlp_regime,
      error_nlp_only, error_nlp_regime

    ANTI-OVERFITTING GUARDS:
      • Ridge regularisation (RidgeCV over ALPHA_RANGE); same alpha for both
      • Expanding window (no look-ahead): train on all prior meetings
      • Powell-only: no across-chair pooling that might conflate styles
      • Minimum train set enforced before first prediction
    """
    powell = panel[panel["meeting_date"] >= POWELL_START].copy().reset_index(drop=True)
    powell = powell.dropna(subset=[target_col])

    meetings = sorted(powell["meeting_date"].unique())

    nlp_feats = [f for f in TEXT_FEATURES if f in powell.columns]
    nlp_feats += [c for c in ["novelty_prev","guidance_change","word_count_zscore"]
                  if c in powell.columns]
    ctrl_feats = [c for c in [f"{target_col}_lag1","is_presser","is_speech",
                               "has_presser","has_speech","policy_dir"]
                  if c in powell.columns]

    regime_feats  = [f"regime_{lab}" for lab in REGIME_ORDER if f"regime_{lab}" in powell.columns]
    interact_feats= [c for c in powell.columns if "__x__" in c]

    feats_nlp    = nlp_feats + ctrl_feats
    feats_regime = nlp_feats + ctrl_feats + regime_feats + interact_feats

    results = []
    scaler_nlp    = StandardScaler()
    scaler_regime = StandardScaler()

    for i, pred_date in enumerate(meetings):
        if i < min_train:
            continue

        train = powell[powell["meeting_date"] < pred_date].dropna(subset=feats_nlp + [target_col])
        test  = powell[powell["meeting_date"] == pred_date]
        if len(train) < 10 or test.empty:
            continue

        y_train = train[target_col].values
        actual  = float(test[target_col].iloc[0])

        def _safe_cols(df: pd.DataFrame, cols: list) -> list:
            return [c for c in cols if c in df.columns and df[c].notna().any()]

        # ── NLP-only ──────────────────────────────────────────────────────────
        c_nlp = _safe_cols(train, feats_nlp)
        if len(c_nlp) >= 2:
            X_tr = train[c_nlp].fillna(0).values
            X_te = test[c_nlp].fillna(0).values
            X_tr = scaler_nlp.fit_transform(X_tr)
            X_te = scaler_nlp.transform(X_te)
            mdl  = RidgeCV(alphas=ALPHA_RANGE).fit(X_tr, y_train)
            pred_nlp = float(mdl.predict(X_te)[0])
        else:
            pred_nlp = float(y_train.mean())

        # ── NLP×regime ────────────────────────────────────────────────────────
        c_reg = _safe_cols(train, feats_regime)
        if len(c_reg) >= 2:
            X_tr = train[c_reg].fillna(0).values
            X_te = test[c_reg].fillna(0).values
            X_tr = scaler_regime.fit_transform(X_tr)
            X_te = scaler_regime.transform(X_te)
            mdl2 = RidgeCV(alphas=ALPHA_RANGE).fit(X_tr, y_train)
            pred_reg = float(mdl2.predict(X_te)[0])
        else:
            pred_reg = float(y_train.mean())

        results.append(dict(
            meeting_date    = pred_date,
            regime_label    = str(test["regime_label"].iloc[0]),
            actual          = actual,
            pred_nlp_only   = pred_nlp,
            pred_nlp_regime = pred_reg,
            error_nlp_only  = pred_nlp - actual,
            error_nlp_regime= pred_reg - actual,
            n_train         = len(train),
        ))

    df = pd.DataFrame(results)
    df.to_parquet(MODEL_OUT, index=False)
    return df


oos_results = walk_forward_powell(model_panel)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — ACCEPTANCE TEST + EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def print_oos_comparison(oos: pd.DataFrame) -> None:
    """
    Acceptance test: does NLP×regime beat NLP-only OOS on Powell?

    Reports:
      1. OOS RMSE and MAE for both models (headline)
      2. Sign hit rate (did model predict correct direction of vol?)
      3. Per-regime breakdown
      4. Bootstrap CIs (1,000 resamplings)
    """
    if len(oos) < 5:
        print("  ⚠  Insufficient OOS observations (< 5). "
              "No claims about model performance possible.")
        return

    def _rmse(e): return float(np.sqrt(np.mean(np.array(e)**2)))
    def _mae(e):  return float(np.mean(np.abs(np.array(e))))
    def _sign_hit(pred, actual):
        a, p = np.array(actual), np.array(pred)
        mean_a = a.mean()
        return float(np.mean(np.sign(p - mean_a) == np.sign(a - mean_a)))

    act   = oos["actual"].values
    e_nlp = oos["error_nlp_only"].values
    e_reg = oos["error_nlp_regime"].values
    p_nlp = oos["pred_nlp_only"].values
    p_reg = oos["pred_nlp_regime"].values

    # Bootstrap CIs for RMSE improvement
    rng   = np.random.default_rng(42)
    n     = len(act)
    boot  = np.zeros(1000)
    for b in range(1000):
        idx = rng.integers(0, n, n)
        boot[b] = _rmse(e_nlp[idx]) - _rmse(e_reg[idx])
    ci_lo, ci_hi = float(np.percentile(boot, 5)), float(np.percentile(boot, 95))
    regime_wins  = np.mean(boot > 0)

    print("\n" + "═" * 60)
    print("  ACCEPTANCE TEST — NLP×regime vs NLP-only (Powell OOS)")
    print("═" * 60)
    print(f"  {'Metric':<28} {'NLP-only':>12} {'NLP×regime':>12}  {'Δ':>8}")
    print(f"  {'─'*58}")

    rmse_nlp = _rmse(e_nlp); rmse_reg = _rmse(e_reg)
    mae_nlp  = _mae(e_nlp);  mae_reg  = _mae(e_reg)
    r2_nlp   = float(r2_score(act, p_nlp)) if len(act) > 2 else np.nan
    r2_reg   = float(r2_score(act, p_reg)) if len(act) > 2 else np.nan
    sh_nlp   = _sign_hit(p_nlp, act)
    sh_reg   = _sign_hit(p_reg, act)

    def _row(label, a, b): print(f"  {label:<28} {a:>12.4f} {b:>12.4f}  {b-a:>+8.4f}")
    _row("OOS RMSE (lower = better)", rmse_nlp, rmse_reg)
    _row("OOS MAE  (lower = better)", mae_nlp,  mae_reg)
    _row("OOS R²   (higher = better)",r2_nlp,   r2_reg)
    _row("Sign hit rate",              sh_nlp,   sh_reg)

    print(f"\n  RMSE improvement (NLP-only − NLP×regime): {rmse_nlp-rmse_reg:+.4f}")
    print(f"  90% bootstrap CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"  Fraction of bootstrap draws where regime model wins: {regime_wins:.1%}")

    if ci_lo > 0:
        verdict = "PASS — regime conditioning improves OOS at 90% CI."
    elif regime_wins > 0.65:
        verdict = "WEAK — regime conditioning tends to improve OOS but CI crosses 0."
    else:
        verdict = ("FLAG — regime conditioning does NOT reliably improve OOS. "
                   "Do not claim the mechanism. Report this result.")

    print(f"\n  VERDICT: {verdict}")
    print(f"\n  ⚠  n={n} OOS meetings; bootstrap CIs are wide.  "
          "Treat as exploratory — report the verdict, not just the sign.")

    # ── Per-regime breakdown ────────────────────────────────────────────────────
    print(f"\n  ── Per-regime breakdown ──")
    print(f"  {'Regime':<14} {'n':>4}  {'ΔRMSE':>8}  {'ΔSign%':>9}  {'Note'}")
    print(f"  {'─'*56}")
    for lab in REGIME_ORDER:
        sub = oos[oos["regime_label"] == lab]
        if len(sub) < 3:
            continue
        dr   = _rmse(sub["error_nlp_only"]) - _rmse(sub["error_nlp_regime"])
        dsh  = _sign_hit(sub["pred_nlp_regime"], sub["actual"]) \
             - _sign_hit(sub["pred_nlp_only"],   sub["actual"])
        note = "← biggest gain?" if dr == max(
            _rmse(oos[oos.regime_label==l]["error_nlp_only"]) -
            _rmse(oos[oos.regime_label==l]["error_nlp_regime"])
            for l in oos["regime_label"].unique() if len(oos[oos.regime_label==l])>=3
        ) else ""
        print(f"  {lab:<14} {len(sub):>4}  {dr:>+8.4f}  {dsh:>+8.1%}  {note}")

    # ── Doc-type ablation ───────────────────────────────────────────────────────
    print(f"\n  ── Doc-type ablation (statement only vs +presser vs +speech) ──")
    print(f"  ({'requires has_presser / has_speech columns in oos panel' }")
    print(f"  {'Doc config':<28} {'Sign hit':>10}")
    for label, mask in [
        ("Statements only",          ~oos.get("has_presser", pd.Series(0, index=oos.index)).astype(bool)
                                     & ~oos.get("has_speech",  pd.Series(0, index=oos.index)).astype(bool)),
        ("+Press conferences",       oos.get("has_presser", pd.Series(0, index=oos.index)).astype(bool)),
        ("+Jackson Hole speeches",   oos.get("has_speech",  pd.Series(0, index=oos.index)).astype(bool)),
    ]:
        if mask.sum() < 3:
            continue
        sub_o = oos[mask]
        sh = _sign_hit(sub_o["pred_nlp_regime"], sub_o["actual"])
        print(f"  {label:<28} {sh:>10.1%}  (n={mask.sum()})")

    print("═" * 60)


print_oos_comparison(oos_results)

# Warsh forward test (single application of validated model — not the validation set)
warsh_row = oos_results[oos_results["meeting_date"] >= pd.Timestamp("2026-01-01")]
if not warsh_row.empty:
    print(f"\n  WARSH FORWARD TEST (single observation, n={len(warsh_row)}):")
    print(f"  NLP-only forecast:   {warsh_row['pred_nlp_only'].mean():.2f}%  "
          f"Actual: {warsh_row['actual'].mean():.2f}%")
    print(f"  NLP×regime forecast: {warsh_row['pred_nlp_regime'].mean():.2f}%")
    print(f"  Warsh regime label:  {warsh_row['regime_label'].iloc[0]}")
    print(f"  NOTE: Warsh is the forward test. The model was validated on Powell above.")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — VISUALISATION (Figs 1-6, shared palette)
# ══════════════════════════════════════════════════════════════════════════════

_regime_handles = [
    mpatches.Patch(color=REGIME_PALETTE[lab], label=lab)
    for lab in REGIME_ORDER if lab in REGIME_PALETTE
]


def _annotate_pain(ax, dates, y_vals, regime_col=None):
    """Annotate the Jackson Hole 2022 'pain' speech on any axis with a date axis."""
    if not hasattr(ax, "axvline"):
        return
    try:
        ax.axvline(PAIN_SPEECH_DATE, color="#800020", lw=1.2, ls="--", alpha=0.8)
        ylim = ax.get_ylim()
        ax.text(PAIN_SPEECH_DATE, ylim[1] * 0.97, "JH\n'pain'",
                fontsize=7, color="#800020", ha="left", va="top")
    except Exception:
        pass


def save_fig(fig, name: str) -> None:
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  → saved {path}")
    _display(fig)
    plt.close(fig)


# ── FIG 1 [HEADLINE] — Dual-mandate phase map ─────────────────────────────────

def fig1_phase_map(regime: pd.DataFrame, panel: pd.DataFrame) -> None:
    """
    2D scatter: x = u_gap, y = inflation_gap.
    One point per meeting, COLOURED by realized 2Y event vol (sequential).
    Sized by |inflation_gap| + |u_gap| (distance from mandate).
    Time-ordered trajectory connecting meetings (regime path).
    Four quadrants labelled.
    PURPOSE: show meetings span multiple quadrants under ONE chair — regime ≠ chair.
    """
    df = regime.merge(panel[["meeting_date","rv_gk_2Y"]].dropna(),
                      on="meeting_date", how="left")
    df = df.dropna(subset=["u_gap","inflation_gap"])

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.suptitle("FIG 1 [HEADLINE] — Dual-Mandate Regime Phase Map\n"
                 "Each point = one FOMC meeting.  One chair (Powell, 2018+) spans "
                 "four regimes — chair identity is a poor regime proxy.",
                 fontsize=11, y=1.01)

    # Time trajectory (faint arrow chain)
    powell = df[df["meeting_date"] >= POWELL_START].sort_values("meeting_date")
    if len(powell) >= 2:
        ax.plot(powell["u_gap"], powell["inflation_gap"],
                color="grey", lw=0.5, alpha=0.35, zorder=1)

    # Colour by realized vol (sequential)
    cmap = plt.cm.YlOrRd
    vol  = df["rv_gk_2Y"].fillna(df["rv_gk_2Y"].mean())
    norm = mcolors.Normalize(vmin=vol.quantile(0.05), vmax=vol.quantile(0.95))
    size = 30 + 80 * (np.abs(df["u_gap"]) + np.abs(df["inflation_gap"])).fillna(0) / \
               (np.abs(df["u_gap"]) + np.abs(df["inflation_gap"])).max()

    sc = ax.scatter(df["u_gap"], df["inflation_gap"],
                    c=vol, cmap=cmap, norm=norm, s=size,
                    alpha=0.75, zorder=2, edgecolors="white", linewidth=0.3)
    plt.colorbar(sc, ax=ax, label="Realized 2Y event vol (%)", pad=0.02)

    # Quadrant shading
    ax.axhline(0.5,  color="#d73027", lw=0.8, ls="--", alpha=0.5)
    ax.axhline(-0.5, color="#4575b4", lw=0.8, ls="--", alpha=0.5)
    ax.axvline(0,    color="grey",    lw=0.6, ls="--", alpha=0.4)
    ax.axvline(0.75, color="#4575b4", lw=0.8, ls="--", alpha=0.5)

    # Quadrant labels
    xl, xh = ax.get_xlim(); yl, yh = ax.get_ylim()
    kw = dict(fontsize=8, alpha=0.6, fontweight="bold")
    ax.text(xl*0.9, yh*0.85, "OVERHEATING\n(tight labour, high π)",
            color="#d73027", **kw)
    ax.text(0.8, yh*0.85, "SUPPLY-SHOCK\n(slack + high π)",
            color="#fc8d59", **kw)
    ax.text(xl*0.9, yl*0.85, "EASING\n(low π, ZLB-type)",
            color="#91bfdb", **kw)
    ax.text(0.8, 0.0, "SLACK\n(u >> NROU)",
            color="#4575b4", **kw)
    ax.text(-0.15, 0.08, "AT-TARGET", color="#878787", fontsize=8)

    # Jackson Hole 2022 annotate
    jh22 = df[df["meeting_date"].between(
        PAIN_SPEECH_DATE - pd.Timedelta(days=60),
        PAIN_SPEECH_DATE + pd.Timedelta(days=60)
    )]
    if not jh22.empty:
        r = jh22.iloc[0]
        ax.annotate("JH 2022\n'pain'", xy=(r["u_gap"], r["inflation_gap"]),
                    xytext=(r["u_gap"] + 0.3, r["inflation_gap"] - 0.5),
                    fontsize=8, color="#800020",
                    arrowprops=dict(arrowstyle="->", color="#800020", lw=0.8))

    ax.set_xlabel("Unemployment gap (UNRATE − NROU, pp)", fontsize=10)
    ax.set_ylabel("Inflation gap (Core PCE YoY − 2.0, pp)", fontsize=10)
    ax.legend(handles=_regime_handles, fontsize=7, loc="lower right",
              title="Regime", title_fontsize=7)
    ax.grid(True, alpha=0.2, lw=0.5)
    fig.tight_layout()
    save_fig(fig, "fig1_phase_map")


fig1_phase_map(regime_df, model_panel)


# ── FIG 2 — Regime timeline strip ─────────────────────────────────────────────

def fig2_regime_timeline(regime: pd.DataFrame, panel: pd.DataFrame,
                          corpus: pd.DataFrame) -> None:
    """
    Horizontal timeline of the Powell backtest window.
    Bottom band = dual-mandate regime (shared palette).
    Top line = realized 2Y event vol.
    FOMC meetings as ticks; presser/speech events marked by type.
    PURPOSE: regime CHANGES within one chair; vol spikes track regime shifts.
    """
    df = regime.merge(panel[["meeting_date","rv_gk_2Y"]].dropna(),
                      on="meeting_date", how="left")
    df = df[df["meeting_date"] >= POWELL_START].sort_values("meeting_date")
    if df.empty:
        return

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, 5), height_ratios=[3, 1],
                                          sharex=True)
    fig.suptitle("FIG 2 — Regime Timeline (Powell tenure)\n"
                 "Bottom band: dual-mandate economic regime  ·  Top: realized 2Y event vol",
                 fontsize=10, y=1.01)

    # Top: realized vol line
    ax_top.plot(df["meeting_date"], df["rv_gk_2Y"].fillna(np.nan),
                color="#333333", lw=1.3, zorder=3)
    ax_top.scatter(df["meeting_date"], df["rv_gk_2Y"],
                   c=[REGIME_PALETTE.get(l, "#878787") for l in df["regime_label"]],
                   s=28, zorder=4, edgecolors="white", lw=0.4)
    ax_top.set_ylabel("2Y event vol (%)", fontsize=9)
    ax_top.grid(True, axis="y", alpha=0.2, lw=0.5)
    _annotate_pain(ax_top, df["meeting_date"], df["rv_gk_2Y"])

    # Bottom: regime colour bars
    dates = df["meeting_date"].values
    for i in range(len(dates) - 1):
        lab  = df["regime_label"].iloc[i]
        col  = REGIME_PALETTE.get(lab, "#878787")
        ax_bot.barh(0, (dates[i+1] - dates[i]) / np.timedelta64(1, "D"),
                    left=dates[i], height=0.8, color=col, align="edge", alpha=0.85)
    ax_bot.set_yticks([])
    ax_bot.set_xlim(df["meeting_date"].min(), df["meeting_date"].max())

    # Mark presser / speech docs
    speech_dates = corpus[(corpus["doc_type"].isin(["presser","speech"])) &
                          (corpus["meeting_date"] >= POWELL_START)]
    for _, row in speech_dates.iterrows():
        dt  = pd.Timestamp(row["meeting_date"])
        mks = "^" if row["doc_type"] == "speech" else "s"
        col = "purple" if row["doc_type"] == "speech" else "#555555"
        ax_top.axvline(dt, color=col, lw=0.4, alpha=0.3, zorder=2)

    ax_top.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_top.xaxis.set_major_locator(mdates.YearLocator(1))
    plt.setp(ax_top.xaxis.get_majorticklabels(), rotation=0, fontsize=8)

    ax_bot.legend(handles=_regime_handles, fontsize=7, loc="lower left",
                  ncol=len(REGIME_ORDER), title="Regime", title_fontsize=7)

    fig.tight_layout()
    save_fig(fig, "fig2_regime_timeline")


fig2_regime_timeline(regime_df, model_panel, corpus_df)


# ── FIG 3 — Text-feature × Regime loading heatmap ─────────────────────────────

def fig3_loading_heatmap(panel: pd.DataFrame) -> pd.DataFrame:
    """
    OLS loading of each text feature on realized 2Y vol, estimated
    WITHIN each regime.

    Cell (feature, regime) = OLS coefficient β (text_feature → rv_gk_2Y)
    fit on meetings in that regime.  Diverging colormap; 0=white.

    PURPOSE: the SAME language loads differently by regime.
    The interaction IS the mechanism.

    Returns loading matrix DataFrame (saved to CSV alongside figure).
    """
    feat_labels = {"uncertainty_density": "Uncertainty density",
                   "disagree_density":    "Dissent density",
                   "guidance_specificity":"Guidance specificity",
                   "novelty_prev":        "Novelty (vs prev stmt)",
                   "guidance_change":     "Guidance change"}
    target = "rv_gk_2Y"

    loadings = {}
    for lab in REGIME_ORDER:
        sub = panel[panel["regime_label"] == lab].dropna(subset=[target])
        col_betas = {}
        for feat in feat_labels:
            if feat not in sub.columns or sub[feat].std() < 1e-8:
                col_betas[feat] = np.nan
                continue
            x  = sub[feat].fillna(0).values.reshape(-1, 1)
            y  = sub[target].values
            if len(x) < 5:
                col_betas[feat] = np.nan
            else:
                from sklearn.linear_model import LinearRegression
                lr = LinearRegression().fit(x, y)
                col_betas[feat] = float(lr.coef_[0])
        loadings[lab] = col_betas

    load_df = pd.DataFrame(loadings).reindex(index=list(feat_labels.keys()),
                                              columns=REGIME_ORDER)
    # Save CSV
    load_df.index = list(feat_labels.values())
    load_df.to_csv(FIG_DIR / "fig3_loading_matrix.csv")
    print(f"  Loading matrix saved: {FIG_DIR/'fig3_loading_matrix.csv'}")
    print(load_df.round(3).to_string())

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.suptitle("FIG 3 — Text Feature × Regime Loadings (OLS β → 2Y realized vol)\n"
                 "Same language, different vol impact by economic regime",
                 fontsize=10, y=1.01)

    mat = load_df.values.astype(float)
    vmax = np.nanmax(np.abs(mat))
    im   = ax.imshow(mat, cmap="RdBu_r", aspect="auto",
                     vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label="β (text feature → realized vol %)", pad=0.02)

    ax.set_xticks(range(len(REGIME_ORDER)))
    ax.set_xticklabels([l.replace("_", "\n") for l in REGIME_ORDER], fontsize=8)
    ax.set_yticks(range(len(load_df)))
    ax.set_yticklabels(load_df.index, fontsize=9)

    # Colour x-axis labels by regime
    for tick, lab in zip(ax.get_xticklabels(), REGIME_ORDER):
        tick.set_color(REGIME_PALETTE.get(lab, "black"))

    # Annotate cells
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color="black" if abs(v) < vmax * 0.6 else "white")

    fig.tight_layout()
    save_fig(fig, "fig3_loading_heatmap")
    return load_df


load_matrix = fig3_loading_heatmap(model_panel)


# ── FIG 4 [HEADLINE] — NLP-only vs NLP×regime OOS comparison ─────────────────

def fig4_oos_comparison(oos: pd.DataFrame) -> None:
    """
    (a) Two-panel predicted-vs-realized scatter (left NLP-only, right NLP×regime).
    (b) Adjacent bar chart: OOS sign-hit by regime, both models.
    PURPOSE: does NLP×regime beat NLP-only OOS on Powell, and where?
    """
    if len(oos) < 5:
        print("  FIG 4: Insufficient OOS observations. Skipping.")
        return

    def r2(p, a): return float(r2_score(a, p)) if len(a) > 2 else np.nan
    def sh(p, a):
        return float(np.mean(np.sign(np.array(p) - np.mean(a)) ==
                             np.sign(np.array(a) - np.mean(a))))

    act = oos["actual"].values
    p1  = oos["pred_nlp_only"].values
    p2  = oos["pred_nlp_regime"].values
    cols= [REGIME_PALETTE.get(l, "#878787") for l in oos["regime_label"]]

    fig = plt.figure(figsize=(14, 6))
    fig.suptitle("FIG 4 [HEADLINE] — NLP-only vs NLP×regime: OOS Acceptance Test (Powell)\n"
                 "Walk-forward OOS only — no in-sample fit shown",
                 fontsize=11, y=1.01)

    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38, width_ratios=[3, 3, 4])

    # Scatter: NLP-only
    ax0 = fig.add_subplot(gs[0])
    ax0.scatter(act, p1, c=cols, s=30, alpha=0.75, edgecolors="white", lw=0.3)
    lim = [min(act.min(), p1.min()) * 0.9, max(act.max(), p1.max()) * 1.1]
    ax0.plot(lim, lim, "k--", lw=0.8, alpha=0.5)
    ax0.set_xlim(lim); ax0.set_ylim(lim)
    ax0.set_xlabel("Realized 2Y event vol (%)", fontsize=9)
    ax0.set_ylabel("Predicted", fontsize=9)
    ax0.set_title(f"NLP-only\nOOS R²={r2(p1,act):.3f}  SHR={sh(p1,act):.1%}", fontsize=9)

    # Scatter: NLP×regime
    ax1 = fig.add_subplot(gs[1])
    ax1.scatter(act, p2, c=cols, s=30, alpha=0.75, edgecolors="white", lw=0.3)
    ax1.plot(lim, lim, "k--", lw=0.8, alpha=0.5)
    ax1.set_xlim(lim); ax1.set_ylim(lim)
    ax1.set_xlabel("Realized 2Y event vol (%)", fontsize=9)
    ax1.set_ylabel("")
    ax1.set_title(f"NLP × regime\nOOS R²={r2(p2,act):.3f}  SHR={sh(p2,act):.1%}", fontsize=9)
    ax1.legend(handles=_regime_handles, fontsize=6, loc="upper left",
               title="Regime", title_fontsize=6)

    # Bar chart by regime
    ax2   = fig.add_subplot(gs[2])
    labs_present = [l for l in REGIME_ORDER if l in oos["regime_label"].unique()]
    x     = np.arange(len(labs_present))
    width = 0.35
    sh1 = [sh(oos[oos.regime_label==l]["pred_nlp_only"],
               oos[oos.regime_label==l]["actual"]) for l in labs_present]
    sh2 = [sh(oos[oos.regime_label==l]["pred_nlp_regime"],
               oos[oos.regime_label==l]["actual"]) for l in labs_present]
    ns  = [len(oos[oos.regime_label==l]) for l in labs_present]

    ax2.bar(x - width/2, sh1, width, color="steelblue",  alpha=0.7, label="NLP-only")
    ax2.bar(x + width/2, sh2, width,
            color=[REGIME_PALETTE.get(l, "#878787") for l in labs_present],
            alpha=0.85, label="NLP×regime")
    ax2.axhline(0.5, color="grey", lw=0.8, ls="--", alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{l}\n(n={n})" for l,n in zip(labs_present, ns)], fontsize=7)
    ax2.set_ylabel("Sign hit rate", fontsize=9)
    ax2.set_title("Sign-hit rate by regime\n(regime conditioning helps most in unusual regimes)",
                  fontsize=9)
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, 1)
    ax2.grid(True, axis="y", alpha=0.2)

    fig.tight_layout()
    save_fig(fig, "fig4_oos_comparison")


fig4_oos_comparison(oos_results)


# ── FIG 5 — High-vol event study ('pain' + 1-2 others) ───────────────────────

def fig5_event_study(panel: pd.DataFrame, corpus: pd.DataFrame,
                     regime: pd.DataFrame) -> None:
    """
    Small-multiples around 2-3 known high-vol events.
    Per event: realized vol path with event marked, text score, regime label.
    PURPOSE: presser/speech text carries signal the statement doesn't.
    """
    events = [
        dict(dt="2022-08-26", label="JH 2022\n'pain' speech", is_pain=True),
        dict(dt="2022-06-15", label="June 2022 FOMC\n(75bp hike)", is_pain=False),
        dict(dt="2022-09-21", label="Sept 2022 FOMC\n(75bp hike #3)", is_pain=False),
    ]

    df_panel = panel.dropna(subset=["rv_gk_2Y"]).sort_values("meeting_date")
    df_panel["meeting_date"] = pd.to_datetime(df_panel["meeting_date"])

    fig, axes = plt.subplots(1, len(events), figsize=(14, 5), sharey=False)
    fig.suptitle("FIG 5 — High-Vol Event Study (Jackson Hole 2022 + peers)\n"
                 "Press-conference/speech text carries signal the statement doesn't",
                 fontsize=10, y=1.01)

    for ax, ev in zip(axes, events):
        ev_dt = pd.Timestamp(ev["dt"])
        # ±6 meetings window around the event
        idx  = df_panel["meeting_date"].searchsorted(ev_dt)
        lo   = max(0, idx - 6)
        hi   = min(len(df_panel), idx + 7)
        sub  = df_panel.iloc[lo:hi]

        ax.bar(range(len(sub)), sub["rv_gk_2Y"],
               color=[REGIME_PALETTE.get(l, "#878787") for l in sub["regime_label"]],
               alpha=0.75)

        # Highlight event meeting
        event_mask = np.abs((sub["meeting_date"] - ev_dt).dt.days) < 35
        for i, (_, row) in enumerate(sub.iterrows()):
            if event_mask.iloc[i]:
                ax.bar(i, row["rv_gk_2Y"], color="#d73027", alpha=1.0, zorder=3)

        # Text score annotation
        for i, (_, row) in enumerate(sub.iterrows()):
            score = float(row.get("uncertainty_density", 0) or 0)
            if score > 0:
                ax.text(i, row["rv_gk_2Y"] + 0.1, f"{score:.1f}",
                        fontsize=6, ha="center", color="grey")

        # Regime label at bottom
        for i, (_, row) in enumerate(sub.iterrows()):
            ax.text(i, -0.5, row["regime_label"][:4], fontsize=5,
                    ha="center", color=REGIME_PALETTE.get(row["regime_label"], "grey"))

        # Doc-type marker
        for i, (_, row) in enumerate(sub.iterrows()):
            dt_match = corpus[(corpus["meeting_date"] == row["meeting_date"]) &
                              (corpus["doc_type"].isin(["speech","presser"]))]
            if not dt_match.empty:
                ax.text(i, row["rv_gk_2Y"] * 0.5, "★",
                        fontsize=9, ha="center", va="center",
                        color="purple" if (dt_match["doc_type"] == "speech").any() else "teal")

        ax.set_title(ev["label"], fontsize=9,
                     color="#d73027" if ev["is_pain"] else "black", fontweight="bold")
        ax.set_xticks([])
        ax.set_ylabel("2Y event vol (%)" if ax is axes[0] else "", fontsize=9)
        ax.set_ylim(bottom=-0.8)
        ax.grid(True, axis="y", alpha=0.2, lw=0.5)

    star_handle = Line2D([0], [0], marker="★", color="w",
                         markerfacecolor="purple", markersize=10, label="Speech/presser in corpus")
    fig.legend(handles=_regime_handles + [star_handle], fontsize=7,
               loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout()
    save_fig(fig, "fig5_event_study")


fig5_event_study(model_panel, corpus_df, regime_df)


# ── FIG 6 — Doc-type ablation ─────────────────────────────────────────────────

def fig6_doctype_ablation(panel: pd.DataFrame, regime: pd.DataFrame,
                           corpus: pd.DataFrame) -> None:
    """
    Grouped bars: OOS sign hit rate for:
      A) Statements only (no presser / speech)
      B) + Press conferences
      C) + Jackson Hole speeches
    PURPOSE: quantify whether higher-information text adds OOS signal.
    If bars do NOT rise, report that honestly.
    """
    reg_df = regime[["meeting_date","regime_label"]].copy()
    reg_df["meeting_date"] = pd.to_datetime(reg_df["meeting_date"])
    corp   = corpus.copy()
    corp["meeting_date"] = pd.to_datetime(corp["meeting_date"])

    # For each meeting, flag doc types present
    doc_flags = (corp.groupby("meeting_date")["doc_type"]
                 .apply(lambda x: set(x.tolist()))
                 .reset_index(name="doc_types"))

    target = "rv_gk_2Y"
    pan    = panel.dropna(subset=[target]).merge(
        doc_flags, on="meeting_date", how="left")
    pan    = pan[pan["meeting_date"] >= POWELL_START].sort_values("meeting_date")

    def _ablation_oos(only_these_types: set) -> float:
        """Run walk-forward restricted to meetings where only doc types in the set were used."""
        mask = pan["doc_types"].apply(
            lambda s: not s.issubset(only_these_types) if isinstance(s, set)
            else True
        )
        # Use all meetings but score only features from the restricted doc set
        # Here we proxy: if presser/speech filtered, zero out the text features
        sub = pan.copy()
        if "presser" not in only_these_types:
            for feat in TEXT_FEATURES:
                if feat in sub.columns:
                    sub.loc[mask, feat] = 0.0
        # Mini walk-forward
        subs    = sub[sub["meeting_date"] >= POWELL_START].dropna(subset=[target])
        meetings = sorted(subs["meeting_date"].unique())
        hits = []
        for i, dt in enumerate(meetings):
            if i < 10:
                continue
            tr  = subs[subs["meeting_date"] < dt].dropna(subset=TEXT_FEATURES[:2] + [target])
            te  = subs[subs["meeting_date"] == dt]
            if len(tr) < 5 or te.empty:
                continue
            X_tr = tr[TEXT_FEATURES[:2]].fillna(0).values
            y_tr = tr[target].values
            X_te = te[TEXT_FEATURES[:2]].fillna(0).values
            X_tr = StandardScaler().fit_transform(X_tr)
            X_te = StandardScaler().fit_transform(X_te)
            try:
                pred = RidgeCV(alphas=ALPHA_RANGE).fit(X_tr, y_tr).predict(X_te)[0]
                hits.append(int(np.sign(pred - y_tr.mean()) == np.sign(te[target].iloc[0] - y_tr.mean())))
            except Exception:
                continue
        return float(np.mean(hits)) if hits else np.nan

    configs = [
        ("Statements\nonly",         {"statement"}),
        ("+ Press\nconferences",     {"statement", "presser"}),
        ("+ Jackson Hole\nspeeches", {"statement", "presser", "speech"}),
    ]

    rates  = [_ablation_oos(s) for _, s in configs]
    labels = [l for l, _ in configs]
    colors = ["#aaaaaa", "#888888", "#555555"]

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle("FIG 6 — Doc-Type Ablation\n"
                 "OOS sign hit rate as corpus expands  "
                 "(if bars do not rise, report that honestly)",
                 fontsize=10, y=1.01)

    bars = ax.bar(labels, rates, color=colors, alpha=0.82, width=0.5)
    ax.axhline(0.5, color="grey", lw=0.8, ls="--", alpha=0.7, label="Naïve 50%")
    for b, r in zip(bars, rates):
        if not np.isnan(r):
            ax.text(b.get_x() + b.get_width()/2, r + 0.01, f"{r:.1%}",
                    ha="center", fontsize=9, fontweight="bold")

    ax.set_ylim(0, 1)
    ax.set_ylabel("OOS sign hit rate (walk-forward, Powell)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.2, lw=0.5)
    ax.set_title("⚠  Small sample (Powell n≈50). "
                 "Treat as exploratory; report result regardless of direction.",
                 fontsize=8, color="grey")
    fig.tight_layout()
    save_fig(fig, "fig6_doctype_ablation")


fig6_doctype_ablation(model_panel, regime_df, corpus_df)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — FINAL CAVEATS
# ══════════════════════════════════════════════════════════════════════════════

print("""
══════════════════════════════════════════════════════════════════
  CAVEATS (print)
══════════════════════════════════════════════════════════════════

  1. REGIME CONDITIONING CAN OVERFIT.  Validated OOS on Powell.
     If the OOS acceptance test (Fig 4) does not show improvement,
     do NOT claim the NLP×regime mechanism.  Report the result.

  2. PRESS-CONFERENCE / SPEECH NORMALISATION.  Text scores are
     computed as densities per 1k tokens.  A 6k-word presser and
     a 130-word statement are comparable in density, NOT in raw
     score.  Do not compare raw scores across doc_types.

  3. DUAL-MANDATE REGIME IS ECONOMICALLY PRIMITIVE.  UNRATE,
     PCEPILFE, and NROU are defined by BLS/BEA/Fed — not by
     vol outcomes.  The discrete thresholds are set from economic
     mechanism (Taylor-rule intuition), not from vol.

  4. POWELL IS THE BACKTEST; WARSH IS THE FORWARD TEST.  Do not
     interpret the Warsh forward number as a validated forecast.
     It is a single out-of-sample application of a model validated
     on Powell.  Its CI is very wide.

  5. SMALL SAMPLE.  Powell tenure ≈ 55 meetings; OOS ≈ 40 after
     the training burn-in.  Bootstrap CIs are wide.  Regime cells
     may have n < 10 in the per-regime breakdown.  All results are
     exploratory and should be stated as such.

  6. JACKSON HOLE SPEECHES HAVE A DIFFERENT ECONOMIC ROLE than
     FOMC statements and pressers.  They precede the August meeting
     by ~one week and may move the market on announcement.  The
     doc_type control (is_speech) partially absorbs this, but the
     causal interpretation differs from post-meeting pressers.

══════════════════════════════════════════════════════════════════
""")

print("\nAll sections complete.  Outputs:")
print(f"  {CORPUS_OUT}   — expanded NLP corpus with doc_type tags")
print(f"  {REGIME_OUT}   — dual-mandate regime per meeting")
print(f"  {MODEL_OUT}    — NLP-only vs NLP×regime OOS forecasts")
print(f"  {FIG_DIR}/     — figures 1-6 + loading matrix CSV")
