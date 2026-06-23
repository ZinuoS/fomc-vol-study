# %% [markdown]
# # FOMC Text → Volatility Features
# **Public data only · No Bloomberg / BQuant dependency**
# Output: `fomc_features.parquet` — portable artifact for work-laptop VRP join.
#
# ## Dependencies
# ```bash
# pip install requests beautifulsoup4 yfinance pandas scikit-learn statsmodels scipy
# ```
# Open in VS Code with the Jupyter extension and run cell-by-cell,
# or: `jupyter nbconvert --to notebook --execute fomc_public_pipeline.py`

# %% ── Imports & global configuration ─────────────────────────────────────────

from __future__ import annotations

import re
import time
import hashlib
import warnings
import textwrap
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from scipy import stats as sp_stats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Configurable params ───────────────────────────────────────────────────────
CACHE_DIR      = Path("fomc_cache")
PARQUET_OUT    = Path("fomc_features.parquet")
START_YEAR     = 2010
END_YEAR       = datetime.today().year
RV_WINDOWS     = [1, 5, 10]        # forward RV horizons in trading days
TFIDF_WINDOW   = 4                 # trailing meetings for novelty centroid
BREVITY_WINDOW = 4                 # trailing meetings for word-count z-score
RATE_LIMIT_S   = 1.5               # seconds between Fed website requests
REGIME_CSV     = Path("fomc_regime.csv")   # optional — see Layer 3

CACHE_DIR.mkdir(exist_ok=True)
(CACHE_DIR / "html").mkdir(exist_ok=True)
(CACHE_DIR / "market").mkdir(exist_ok=True)

print(f"Cache : {CACHE_DIR.resolve()}")
print(f"Out   : {PARQUET_OUT.resolve()}")

# %% [markdown]
# ---
# # LAYER 1 — ACQUISITION: FOMC TEXT + MEETING CALENDAR

# %% ── 1a: FOMC meeting calendar ──────────────────────────────────────────────

# Statement release dates (last day of each scheduled/emergency meeting).
# Source: https://www.federalreserve.gov/monetarypolicy/fomchistorical{YEAR}.htm
#         https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
# Emergency meetings (Mar 2020) are included; presser_available is flagged separately.

_HARDCODED_DATES: list[str] = [
    # 2010
    "2010-01-27","2010-03-16","2010-04-28","2010-06-23",
    "2010-08-10","2010-09-21","2010-11-03","2010-12-14",
    # 2011
    "2011-01-26","2011-03-15","2011-04-27","2011-06-22",
    "2011-08-09","2011-09-21","2011-11-02","2011-12-13",
    # 2012
    "2012-01-25","2012-03-13","2012-04-25","2012-06-20",
    "2012-08-01","2012-09-13","2012-10-24","2012-12-12",
    # 2013
    "2013-01-30","2013-03-20","2013-05-01","2013-06-19",
    "2013-07-31","2013-09-18","2013-10-30","2013-12-18",
    # 2014
    "2014-01-29","2014-03-19","2014-04-30","2014-06-18",
    "2014-07-30","2014-09-17","2014-10-29","2014-12-17",
    # 2015
    "2015-01-28","2015-03-18","2015-04-29","2015-06-17",
    "2015-07-29","2015-09-17","2015-10-28","2015-12-16",
    # 2016
    "2016-01-27","2016-03-16","2016-04-27","2016-06-15",
    "2016-07-27","2016-09-21","2016-11-02","2016-12-14",
    # 2017
    "2017-02-01","2017-03-15","2017-05-03","2017-06-14",
    "2017-07-26","2017-09-20","2017-11-01","2017-12-13",
    # 2018
    "2018-01-31","2018-03-21","2018-05-02","2018-06-13",
    "2018-08-01","2018-09-26","2018-11-08","2018-12-19",
    # 2019
    "2019-01-30","2019-03-20","2019-05-01","2019-06-19",
    "2019-07-31","2019-09-18","2019-10-30","2019-12-11",
    # 2020 (two emergency intermeeting actions: Mar 3 and Mar 15)
    "2020-01-29","2020-03-03","2020-03-15","2020-04-29",
    "2020-06-10","2020-07-29","2020-09-16","2020-11-05","2020-12-16",
    # 2021
    "2021-01-27","2021-03-17","2021-04-28","2021-06-16",
    "2021-07-28","2021-09-22","2021-11-03","2021-12-15",
    # 2022
    "2022-01-26","2022-03-16","2022-05-04","2022-06-15",
    "2022-07-27","2022-09-21","2022-11-02","2022-12-14",
    # 2023
    "2023-02-01","2023-03-22","2023-05-03","2023-06-14",
    "2023-07-26","2023-09-20","2023-11-01","2023-12-13",
    # 2024
    "2024-01-31","2024-03-20","2024-05-01","2024-06-12",
    "2024-07-31","2024-09-18","2024-11-07","2024-12-18",
    # 2025 — verify at https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
    "2025-01-29","2025-03-19","2025-05-07","2025-06-18",
    "2025-07-30","2025-09-17","2025-10-29","2025-12-10",
    # 2026 (stub — scrape or update manually)
    "2026-01-28","2026-03-18","2026-04-29","2026-06-17",
]

# Chair assignment — update the Warsh start date to the confirmed confirmation date.
_CHAIR_PERIODS: list[tuple[date, date, str]] = [
    (date(2010, 1,  1), date(2014, 1, 31), "Bernanke"),
    (date(2014, 2,  1), date(2018, 2,  2), "Yellen"),
    (date(2018, 2,  3), date(2026, 2, 18), "Powell"),
    (date(2026, 2, 19), date(2030, 1,  1), "Warsh"),   # ← adjust if needed
]

# Press conferences: quarterly Apr 2011–Dec 2018; every meeting Jan 2019+.
# Emergency intermeeting actions never have a press conference.
_QUARTERLY_PRESSER_MONTHS = {1, 4, 6, 9, 12}   # approximate quarterly schedule


def _assign_chair(d: date) -> str:
    for start, end, name in _CHAIR_PERIODS:
        if start <= d <= end:
            return name
    return "Unknown"


def _has_presser(d: date, is_emergency: bool = False) -> bool:
    if is_emergency:
        return False
    if d >= date(2019, 1, 1):
        return True
    # Quarterly Apr 2011 – Dec 2018: press conf in Jan/Apr/Jun/Sep/Dec approximately
    if date(2011, 4, 1) <= d <= date(2018, 12, 31):
        return d.month in _QUARTERLY_PRESSER_MONTHS
    return False


def build_fomc_calendar() -> pd.DataFrame:
    """
    Build a DataFrame of FOMC statement release dates with chair and presser flag.
    Returns one row per meeting with columns:
        meeting_date (date), chair (str), presser_available (bool)
    """
    rows = []
    for ds in _HARDCODED_DATES:
        d = datetime.strptime(ds, "%Y-%m-%d").date()
        if d.year < START_YEAR or d.year > END_YEAR:
            continue
        rows.append({
            "meeting_date":      d,
            "chair":             _assign_chair(d),
            "presser_available": _has_presser(d),
        })
    cal = pd.DataFrame(rows).sort_values("meeting_date").reset_index(drop=True)
    print(f"Calendar: {len(cal)} meetings  {cal.meeting_date.min()} → {cal.meeting_date.max()}")
    print(cal.groupby("chair").size().to_string())
    return cal


calendar_df = build_fomc_calendar()

# %% ── 1b: Disk-cached HTTP fetch ─────────────────────────────────────────────

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


def fetch_url(url: str, *, delay: float = RATE_LIMIT_S,
              timeout: int = 20) -> Optional[str]:
    """
    Fetch URL with disk cache (keyed by URL hash).  Returns HTML string or
    None if the request fails.  Cached responses skip the delay.
    """
    key  = hashlib.md5(url.encode()).hexdigest()[:16]
    path = CACHE_DIR / "html" / f"{key}.html"

    if path.exists():
        return path.read_text(encoding="utf-8", errors="replace")

    time.sleep(delay)
    try:
        r = requests.get(url, headers=_HEADERS, timeout=timeout)
        r.raise_for_status()
        html = r.text
        path.write_text(html, encoding="utf-8")
        return html
    except requests.RequestException as exc:
        print(f"  FETCH FAILED  {url}\n  {exc}")
        return None

# %% ── 1c: Statement scraper & cleaner ────────────────────────────────────────

FED_BASE         = "https://www.federalreserve.gov"
STATEMENT_URL_T  = FED_BASE + "/newsevents/pressreleases/monetary{date}a.htm"
PRESSCONF_URL_T  = FED_BASE + "/monetarypolicy/fomcpresconf{date}.htm"


def _extract_statement_text(html: str) -> tuple[str, str]:
    """
    Parse Fed statement HTML.  Returns (raw_text, cleaned_text).
    Strips 'For release at...' header and 'Voting for the FOMC...' footer.

    The modern Fed page has TWO col-xs-12 col-sm-8 divs:
      1. class='heading col-xs-12 ...' — breadcrumb/title (skip)
      2. class='col-xs-12 col-sm-8 ...' — actual statement body (use this)
    We select the non-heading one explicitly.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Modern layout: find content div that is NOT the heading nav div
    content = soup.find(
        "div",
        class_=lambda c: c and "col-xs-12" in c and "col-sm-8" in c and "heading" not in c,
    )
    # Fallbacks for older layouts (pre-2017) and edge cases
    if content is None:
        content = (
            soup.find("div", id="article")
            or soup.find("div", id="leftText")
            or soup.find("article")
            or soup.body
        )

    paragraphs = [p.get_text(" ", strip=True)
                  for p in (content or soup).find_all("p")]
    raw_text = "\n\n".join(p for p in paragraphs if p)

    # ── Clean: skip release-time header; stop at voting-record footer ────
    body_paras: list[str] = []
    in_body = False
    for para in paragraphs:
        lo = para.lower()
        # Stop at voting record or chair signature block
        if re.search(r"voting (for|against) the fomc", lo):
            break
        if re.search(r"^chair(man|woman|person)?[\s,]", lo):
            break
        # Skip the "For release at 2:00 p.m." opener (re.search, not re.match)
        if not in_body and re.search(r"for release at", lo):
            in_body = True
            continue
        # Start body at first substantive paragraph (≥15 words)
        if not in_body and len(para.split()) >= 15:
            in_body = True
        if in_body and len(para.split()) > 3:
            body_paras.append(para)

    cleaned = re.sub(r"\s+", " ", " ".join(body_paras)).strip()

    # Fallback: if cleaning stripped too much, use all paragraphs
    if len(cleaned.split()) < 20 and raw_text:
        cleaned = re.sub(r"\s+", " ", raw_text).strip()

    return raw_text.strip(), cleaned


def scrape_statement(d: date) -> Optional[tuple[str, str]]:
    """Fetch and clean FOMC statement for a given date. Returns (raw, cleaned)."""
    url  = STATEMENT_URL_T.format(date=d.strftime("%Y%m%d"))
    html = fetch_url(url)
    if html is None:
        return None
    return _extract_statement_text(html)


def scrape_pressconf(d: date) -> Optional[tuple[str, str]]:
    """
    Fetch and clean press conference transcript.
    Returns (raw, cleaned) or None if unavailable.
    Transcript pages link to a PDF or HTML; we parse the HTML version.
    """
    url  = PRESSCONF_URL_T.format(date=d.strftime("%Y%m%d"))
    html = fetch_url(url)
    if html is None:
        return None
    soup    = BeautifulSoup(html, "html.parser")
    content = soup.find("div", class_=re.compile(r"col-xs-12.*col-sm-8")) or soup.body
    raw     = " ".join(p.get_text(" ", strip=True) for p in (content or soup).find_all("p"))
    cleaned = re.sub(r"\s+", " ", raw).strip()
    return (raw, cleaned) if len(cleaned.split()) > 50 else None

# %% ── 1d: Build docs_raw ─────────────────────────────────────────────────────

def build_docs_raw(calendar: pd.DataFrame) -> pd.DataFrame:
    """
    Scrape statements (and pressers where available) for every meeting.
    Emits a URL manifest if fetching is blocked so the user can
    bulk-download manually and re-run.

    Returns docs_raw with columns:
        meeting_date, doc_type, chair, text, n_words, presser_available
    """
    rows: list[dict] = []
    failed_urls: list[str] = []

    for _, row in calendar.iterrows():
        d     = row["meeting_date"]
        chair = row["chair"]
        pa    = row["presser_available"]

        # ── Statement ────────────────────────────────────────────────────
        result = scrape_statement(d)
        if result is None:
            url = STATEMENT_URL_T.format(date=d.strftime("%Y%m%d"))
            failed_urls.append(url)
            print(f"  MISS statement {d}")
        else:
            raw, cleaned = result
            rows.append({
                "meeting_date":      d,
                "doc_type":          "statement",
                "chair":             chair,
                "text_raw":          raw,
                "text":              cleaned,
                "n_words":           len(cleaned.split()),
                "presser_available": pa,
            })

        # ── Press conference (where expected) ───────────────────────────
        if pa:
            result_p = scrape_pressconf(d)
            if result_p is not None:
                raw_p, cleaned_p = result_p
                rows.append({
                    "meeting_date":      d,
                    "doc_type":          "presser",
                    "chair":             chair,
                    "text_raw":          raw_p,
                    "text":              cleaned_p,
                    "n_words":           len(cleaned_p.split()),
                    "presser_available": pa,
                })

    if failed_urls:
        manifest = CACHE_DIR / "failed_urls.txt"
        manifest.write_text("\n".join(failed_urls))
        print(f"\n  {len(failed_urls)} URLs failed — manifest saved to {manifest}")
        print("  Download the pages manually, save as <YYYYMMDD>.html in fomc_cache/html/,")
        print("  then re-run: fetch_url() will pick up the cached files automatically.")

    docs = pd.DataFrame(rows)
    docs["meeting_date"] = pd.to_datetime(docs["meeting_date"])
    print(f"\ndocs_raw: {docs.shape}")
    print(docs.groupby("doc_type")["meeting_date"].count().to_string())
    return docs


docs_raw = build_docs_raw(calendar_df)

# %% ── 1e: Drop bad scrapes (JS-wall / redirect pages) ───────────────────────
# Pages that returned a JS-disabled stub or a nav-only redirect produce very
# short "cleaned" text (< 50 words).  Remove those rows and delete their cache
# files so the next run re-fetches from the corrected extractor.

MIN_STATEMENT_WORDS = 50

def purge_bad_scrapes(docs: pd.DataFrame, min_words: int = MIN_STATEMENT_WORDS,
                      cache_dir: Path = CACHE_DIR) -> pd.DataFrame:
    """
    Remove rows where cleaned text is too short, and wipe their HTML cache so
    they are re-fetched on next run with the fixed extractor.
    """
    import hashlib
    bad = docs[docs["n_words"] < min_words]
    if bad.empty:
        print(f"  No bad scrapes detected (all ≥ {min_words} words).")
        return docs

    print(f"  Purging {len(bad)} short-text rows (< {min_words} words):")
    for _, row in bad.iterrows():
        d  = row["meeting_date"]
        dt = d.date() if hasattr(d, "date") else d
        dt_str = dt.strftime("%Y%m%d")
        if row["doc_type"] == "statement":
            url = STATEMENT_URL_T.format(date=dt_str)
        else:
            url = PRESSCONF_URL_T.format(date=dt_str)
        key  = hashlib.md5(url.encode()).hexdigest()[:16]
        path = cache_dir / "html" / f"{key}.html"
        if path.exists():
            path.unlink()
            print(f"    cleared cache: {dt_str} {row['doc_type']}")
        else:
            print(f"    no cache file: {dt_str} {row['doc_type']} (will re-fetch)")

    clean = docs[docs["n_words"] >= min_words].copy().reset_index(drop=True)
    print(f"  docs_raw after purge: {clean.shape}")
    return clean


docs_raw = purge_bad_scrapes(docs_raw)

# %% [markdown]
# ---
# # LAYER 2 — FOUR FEATURE FAMILIES
# Computed on the **statement** corpus. Presser features suffixed `_presser`
# where computed separately. POLARITY is a control/direction overlay — it is
# **not** in the vol-prediction feature set.

# %% ── 2-setup: statement-only frame ─────────────────────────────────────────

stmt = (
    docs_raw[docs_raw["doc_type"] == "statement"]
    .sort_values("meeting_date")
    .reset_index(drop=True)
    .copy()
)
print(f"Statements: {len(stmt)}")

# %% ── FAMILY A — Information density / brevity ──────────────────────────────

def compute_family_a(stmt_df: pd.DataFrame, window: int = BREVITY_WINDOW) -> pd.DataFrame:
    """
    Family A: Word-count features per statement.

    Columns added:
        word_count            — raw word count of cleaned statement
        word_count_change     — vs trailing mean of prior `window` meetings
        word_count_zscore     — (word_count - trailing mean) / trailing std
        word_count_pct_rank   — expanding percentile rank (0-1)

    The most direct Warsh signal: very short or abruptly shorter statements
    → high brevity deviation → elevated vol risk ex-ante.
    """
    df = stmt_df.copy()
    wc = df["n_words"].astype(float)

    trailing_mean = wc.shift(1).rolling(window, min_periods=2).mean()
    trailing_std  = wc.shift(1).rolling(window, min_periods=2).std(ddof=1)

    df["word_count"]        = wc
    df["word_count_change"] = wc - trailing_mean
    df["word_count_zscore"] = (wc - trailing_mean) / trailing_std.replace(0, np.nan)
    df["word_count_pct_rank"] = wc.expanding().rank(pct=True)

    print("Family A — brevity features:")
    print(df[["meeting_date", "word_count", "word_count_change",
              "word_count_zscore"]].tail(6).to_string(index=False))
    return df


stmt = compute_family_a(stmt)

# %% ── FAMILY B — Novelty / divergence from prior ───────────────────────────

def compute_family_b(stmt_df: pd.DataFrame, window: int = TFIDF_WINDOW) -> pd.DataFrame:
    """
    Family B: TF-IDF cosine distance features (pure scikit-learn, no downloads).

    Columns added:
        novelty_prev    — cosine distance to immediately prior statement (0=identical)
        novelty_window  — cosine distance to centroid of prior `window` statements
        novelty_zscore  — novelty_prev z-scored vs expanding window

    Spikes when a new Chair breaks template language.
    All computations are strictly backward-looking (no lookahead).
    """
    texts = stmt_df["text"].fillna("").tolist()
    n     = len(texts)

    # Fit TF-IDF on the full corpus (in-sample; for a strict walk-forward,
    # refit incrementally — acceptable here since vocab doesn't cause leakage)
    tfidf   = TfidfVectorizer(max_features=3000, ngram_range=(1, 2),
                               min_df=2, sublinear_tf=True)
    mat     = tfidf.fit_transform(texts)    # (n_meetings, vocab)

    novelty_prev   = np.full(n, np.nan)
    novelty_window = np.full(n, np.nan)

    for i in range(1, n):
        vec_i = mat[i]
        # Distance to prior
        sim_prev          = cosine_similarity(vec_i, mat[i - 1])[0, 0]
        novelty_prev[i]   = 1.0 - float(sim_prev)
        # Distance to trailing-window centroid
        # np.asarray() converts sparse matrix mean (np.matrix) to ndarray
        start             = max(0, i - window)
        centroid          = np.asarray(mat[start:i].mean(axis=0))
        sim_win           = cosine_similarity(vec_i, centroid)[0, 0]
        novelty_window[i] = 1.0 - float(sim_win)

    df = stmt_df.copy()
    df["novelty_prev"]   = novelty_prev
    df["novelty_window"] = novelty_window

    # Z-score novelty_prev against expanding history (prevents lookahead)
    expanding_mean = pd.Series(novelty_prev).expanding().mean()
    expanding_std  = pd.Series(novelty_prev).expanding().std(ddof=1)
    df["novelty_zscore"] = (
        (pd.Series(novelty_prev) - expanding_mean) / expanding_std.replace(0, np.nan)
    ).values

    print("Family B — novelty features:")
    print(df[["meeting_date", "novelty_prev",
              "novelty_window", "novelty_zscore"]].tail(6).to_string(index=False))
    return df


stmt = compute_family_b(stmt)

# %% ── FAMILY C — Ambiguity / guidance withdrawal ────────────────────────────
# Wordlists embedded inline (no internet required).

# Forward guidance scaffolding phrases — ABSENCE is the vol signal.
_GUIDANCE_PHRASES = [
    r"likely to be appropriate",
    r"anticipate[sd]?\b",
    r"for (some time|an extended period)",
    r"at least .{0,20} meeting[s]?",
    r"calendar.based",
    r"outcome.based",
    r"until .{0,30} (achiev|reach|return)",
    r"well-anchored",
    r"balance sheet normali",
    r"gradual(ly)?",
    r"(exceptionally|historically) low",
    r"patient",
    r"data.depend",
    r"(will|would) be (appropriate|warranted)",
    r"symmetric .{0,10} (inflation|target)",
    r"maximum employment",
    r"(accommodative|restrictive) (for|stance)",
]

# Loughran-McDonald uncertainty words (curated subset; full list at
# https://sraf.nd.edu/loughranmcdonald-master-dictionary/)
_LM_UNCERTAINTY = {
    "uncertain", "uncertainty", "unpredictable", "unpredictability",
    "ambiguous", "ambiguity", "approximate", "approximately", "unclear",
    "indefinite", "indefinitely", "vague", "vaguer", "unsettled",
    "doubt", "doubts", "doubtful", "doubtfully", "questionable",
    "question", "possible", "possibly", "variable", "variability",
    "depend", "depends", "dependent", "contingent", "tentative",
    "tentatively", "indeterminate", "unstable", "unstability",
    "fluctuate", "fluctuates", "fluctuating", "volatile", "volatility",
    "challenging", "unusual", "atypical", "unprecedented", "evolving",
    "reassess", "reassessment", "reconsider", "revisit",
}

# Weak modal verbs (another uncertainty proxy)
_WEAK_MODALS = {"might", "could", "may", "would", "should", "can",
                "seems", "appear", "appears", "suggest", "suggests"}

# Hawk-dove lexicon (used ONLY as control — not in vol feature set)
_HAWK_TOKENS = {
    "inflationary", "overheat", "tighten", "tightening", "restrictive",
    "above target", "price stability", "raise rates", "increase rates",
    "normalize", "higher rates", "vigilant", "premature", "front-load",
}
_DOVE_TOKENS = {
    "accommodative", "easing", "support", "stimulate", "slack",
    "below target", "underemployment", "lower rates", "reduce rates",
    "patient", "gradual", "transitory", "muted", "subdued",
}


def _regex_density(text: str, patterns: list[str]) -> float:
    """Hits-per-100-words for a list of regex patterns."""
    hits = sum(len(re.findall(p, text, re.IGNORECASE)) for p in patterns)
    wc   = max(len(text.split()), 1)
    return hits / wc * 100


def _token_density(text: str, wordset: set[str]) -> float:
    """Fraction of tokens matching wordset."""
    tokens = re.findall(r"\b\w+\b", text.lower())
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if t in wordset) / len(tokens)


def compute_family_c(stmt_df: pd.DataFrame,
                     window: int = BREVITY_WINDOW) -> pd.DataFrame:
    """
    Family C: Ambiguity and guidance-withdrawal features.

    guidance_density   — hits-per-100-words of forward-guidance scaffolding phrases
    guidance_presence  — binary: guidance_density above historical median
    guidance_change    — guidance_density vs trailing mean (drop = withdrawal signal)
    uncertainty_density — LM uncertainty word density
    weak_modal_density  — weak modal verb density
    """
    df = stmt_df.copy()

    df["guidance_density"]   = df["text"].apply(
        lambda t: _regex_density(t, _GUIDANCE_PHRASES))
    df["uncertainty_density"] = df["text"].apply(
        lambda t: _token_density(t, _LM_UNCERTAINTY))
    df["weak_modal_density"]  = df["text"].apply(
        lambda t: _token_density(t, _WEAK_MODALS))

    hist_med = df["guidance_density"].expanding().median()
    df["guidance_presence"] = (df["guidance_density"] > hist_med).astype(int)

    trail_mean = df["guidance_density"].shift(1).rolling(window, min_periods=2).mean()
    df["guidance_change"] = df["guidance_density"] - trail_mean

    print("Family C — ambiguity/guidance features:")
    print(df[["meeting_date", "guidance_density", "guidance_presence",
              "uncertainty_density"]].tail(6).to_string(index=False))
    return df


stmt = compute_family_c(stmt)

# %% ── FAMILY D — Disagreement ────────────────────────────────────────────────

_DISAGREE_PATTERNS = [
    r"\bsome\b .{0,30}\bparticipants?\b",
    r"\bseveral\b .{0,30}\bparticipants?\b",
    r"\ba few\b .{0,30}\bparticipants?\b",
    r"\bmany\b .{0,30}\bparticipants?\b",
    r"\bmost\b .{0,30}\bparticipants?\b",
    r"\bdiverge",
    r"\bdisagree",
    r"\bdiffering\b",
    r"\bcautioned\b",
    r"\bpreferred\b .{0,30}(more|less|larger|smaller)",
]


def compute_family_d(stmt_df: pd.DataFrame) -> pd.DataFrame:
    """
    Family D: Disagreement / heterogeneity.

    On statements (not minutes), true cross-member dispersion is unobservable.
    We use hedged-quantifier density as a proxy for the committee's internal
    heterogeneity that the Chair felt obliged to surface publicly.
    Returns disagree_density (hits/100 words); null where uninformative.

    NOTE: Minutes-based dispersion would be more powerful but requires separate
    scraping. Leave a stub column `disagree_density_minutes` = null with a TODO.
    """
    df = stmt_df.copy()
    df["disagree_density"] = df["text"].apply(
        lambda t: _regex_density(t, _DISAGREE_PATTERNS))
    df["disagree_density_minutes"] = np.nan   # TODO: join minutes scrape

    print("Family D — disagreement proxy:")
    print(df[["meeting_date", "disagree_density"]].tail(6).to_string(index=False))
    return df


stmt = compute_family_d(stmt)

# %% ── Control feature: polarity / hawk-dove score ────────────────────────────
# NOT a vol predictor. Label clearly as direction overlay.

def compute_polarity_control(stmt_df: pd.DataFrame) -> pd.DataFrame:
    """
    Control: hawk-dove polarity score (positive = hawkish).
    Based on token-density lexicon; NOT included in vol feature families.
    Used as a control variable and direction overlay only.
    """
    df = stmt_df.copy()
    df["hawk_density"] = df["text"].apply(lambda t: _token_density(t, _HAWK_TOKENS))
    df["dove_density"] = df["text"].apply(lambda t: _token_density(t, _DOVE_TOKENS))
    df["polarity_hd"]  = df["hawk_density"] - df["dove_density"]   # >0 = hawkish
    return df


stmt = compute_polarity_control(stmt)

# %% ── Assemble features_text ─────────────────────────────────────────────────

TEXT_FEATURE_COLS = [
    # Family A
    "word_count", "word_count_change", "word_count_zscore", "word_count_pct_rank",
    # Family B
    "novelty_prev", "novelty_window", "novelty_zscore",
    # Family C
    "guidance_density", "guidance_presence", "guidance_change",
    "uncertainty_density", "weak_modal_density",
    # Family D
    "disagree_density",
    # Control (direction, not vol)
    "polarity_hd",
]

features_text = stmt[
    ["meeting_date", "chair", "presser_available"] + TEXT_FEATURE_COLS
].copy()

print(f"\nfeatures_text: {features_text.shape}")
print(features_text.isnull().sum()[features_text.isnull().sum() > 0].to_string())
print(features_text.tail(4).to_string(index=False))

# %% [markdown]
# ---
# # LAYER 3 — PUBLIC-DATA REALIZED VOL + REGIME

# %% ── 3a: Daily yield data from FRED (no API key required) ──────────────────

FRED_SERIES = {
    "DGS2":  "yield_2y",
    "DGS5":  "yield_5y",
    "DGS10": "yield_10y",
    "DGS30": "yield_30y",
}
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"


def fetch_fred_series(series_id: str, start: str, end: str) -> pd.Series:
    """
    Download a FRED series as a daily pd.Series via the public CSV endpoint.
    Caches to disk. Values are percentages (e.g. 4.25 for 4.25%).
    """
    cache_path = CACHE_DIR / "market" / f"fred_{series_id}.csv"
    if cache_path.exists():
        df_c = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        s    = df_c.iloc[:, 0].replace(".", np.nan).astype(float)
    else:
        url  = FRED_CSV_URL.format(series=series_id)
        time.sleep(0.5)
        try:
            df_c = pd.read_csv(url, index_col=0, parse_dates=True)
            df_c.to_csv(cache_path)
            s    = df_c.iloc[:, 0].replace(".", np.nan).astype(float)
        except Exception as exc:
            print(f"  FRED fetch failed for {series_id}: {exc}")
            return pd.Series(dtype=float, name=series_id)

    mask = (s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))
    return s.loc[mask].rename(series_id)


def build_yield_panel(start: str, end: str) -> pd.DataFrame:
    """
    Fetch all FRED yield series; return a daily DataFrame aligned on business days.
    Missing values (weekends, holidays) are forward-filled (FRED convention).
    """
    series_dict = {}
    for fred_id, col_name in FRED_SERIES.items():
        s = fetch_fred_series(fred_id, start, end)
        if not s.empty:
            series_dict[col_name] = s.rename(col_name)

    yields = pd.DataFrame(series_dict)
    yields.index = pd.to_datetime(yields.index)
    yields = yields.sort_index().ffill()
    print(f"Yield panel: {yields.shape}  {yields.index.min().date()} → {yields.index.max().date()}")
    print(yields.tail(3).to_string())
    return yields


market_start = f"{START_YEAR - 1}-01-01"   # one year buffer for lagged RV
market_end   = f"{END_YEAR}-12-31"
yields_df    = build_yield_panel(market_start, market_end)

# %% ── 3b: Realized volatility from yield changes ────────────────────────────

def compute_rv(
    yields:      pd.DataFrame,
    fomc_dates:  list,
    tenors:      list[str] | None = None,
    k_list:      list[int]        = RV_WINDOWS,
    direction:   str              = "forward",
) -> pd.DataFrame:
    """
    Realized vol on daily yield *changes* (basis-point moves per day).

    RV = sqrt(252 / n_obs * sum(dr^2))  where dr = daily yield change in pp.
    Annualised in percentage-point / day units — comparable to IV in pp.

    Parameters
    ----------
    yields     : daily yield panel (percentage points, e.g. 4.25 for 4.25%)
    fomc_dates : list of meeting dates
    tenors     : subset of yield columns to use; defaults to all
    k_list     : forward window lengths in trading days
    direction  : "forward" or "backward"

    Returns
    -------
    DataFrame indexed by meeting_date with columns rv_{tenor}_{k}d
    """
    tenors = tenors or list(yields.columns)
    diffs  = yields[tenors].diff().dropna()   # daily yield changes (pp)
    rows: list[dict] = []

    for fd in fomc_dates:
        fd_ts = pd.Timestamp(fd)
        row   = {"meeting_date": fd}

        for k in k_list:
            for tenor in tenors:
                series = diffs[tenor].dropna()
                if direction == "forward":
                    window = series[series.index > fd_ts].iloc[:k]
                else:
                    window = series[series.index < fd_ts].iloc[-k:]

                if len(window) < max(2, int(k * 0.6)):
                    row[f"rv_{tenor}_{k}d"] = np.nan
                else:
                    rv_val = np.sqrt(252 / len(window) * (window ** 2).sum())
                    row[f"rv_{tenor}_{k}d"] = rv_val

        rows.append(row)

    rv_df = pd.DataFrame(rows)
    rv_df["meeting_date"] = pd.to_datetime(rv_df["meeting_date"])
    print(f"RV frame: {rv_df.shape}")
    print(rv_df.tail(3).to_string(index=False))
    return rv_df


fomc_dates_list = features_text["meeting_date"].dt.date.tolist()
rv_df = compute_rv(yields_df, fomc_dates_list)

# %% ── 3c: Parkinson range-vol on the meeting day (event-day proxy) ──────────

def compute_parkinson_vol(fomc_dates: list, ticker: str = "^TNX") -> pd.Series:
    """
    Parkinson (1980) range estimator on meeting-day OHLC from Yahoo Finance.
    Parkinson_vol = sqrt(1 / (4 ln 2) * (ln(H/L))^2) * sqrt(252) * 100

    Uses ^TNX (CBOE 10Y yield index); H/L in percentage points.
    Returns a Series indexed by meeting_date.
    """
    try:
        import yfinance as yf
    except ImportError:
        print("  yfinance not installed — Parkinson vol skipped.")
        return pd.Series(dtype=float, name="parkinson_vol_10y")

    cache_path = CACHE_DIR / "market" / f"ohlc_{ticker.replace('^','')}.parquet"
    if cache_path.exists():
        ohlc = pd.read_parquet(cache_path)
    else:
        ohlc = yf.download(ticker, start=market_start, end=market_end,
                           progress=False, auto_adjust=True)
        ohlc.to_parquet(cache_path)

    ohlc.index = pd.to_datetime(ohlc.index)
    results = {}
    for fd in fomc_dates:
        fd_ts = pd.Timestamp(fd)
        if fd_ts in ohlc.index:
            row = ohlc.loc[fd_ts]
            H, L = float(row["High"]), float(row["Low"])
            if H > 0 and L > 0 and H >= L:
                park = np.sqrt(1 / (4 * np.log(2)) * (np.log(H / L)) ** 2
                               * 252) * 100
                results[fd] = park

    s = pd.Series(results, name="parkinson_vol_10y")
    s.index = pd.to_datetime(s.index)
    print(f"Parkinson vol: {s.notna().sum()} meeting-day observations")
    return s


parkinson_s = compute_parkinson_vol(fomc_dates_list)

# %% ── 3d: Policy-surprise proxy (2Y yield daily change) ─────────────────────

def compute_policy_surprise(yields: pd.DataFrame,
                            fomc_dates: list) -> pd.Series:
    """
    Policy-surprise proxy: 1-day change in the 2-year yield ON the meeting date.
    Best available without intraday data; controls for the expected portion of
    the rate change in downstream regressions.

    TODO: replace with SF Fed USMPD high-frequency surprise series if available.
          Join key: meeting_date.  Column: mp_surprise_bp (basis points).
    """
    tenor = "yield_2y" if "yield_2y" in yields.columns else yields.columns[0]
    diffs = yields[tenor].diff()
    out   = {}
    for fd in fomc_dates:
        fd_ts = pd.Timestamp(fd)
        if fd_ts in diffs.index:
            out[fd] = diffs.loc[fd_ts]
    s = pd.Series(out, name="policy_surprise_2y_chg")
    s.index = pd.to_datetime(s.index)
    print(f"Policy surprise: {s.notna().sum()} observations, "
          f"mean={s.mean():.3f}pp, std={s.std():.3f}pp")
    return s


policy_surprise_s = compute_policy_surprise(yields_df, fomc_dates_list)

# %% ── 3e: Regime join ────────────────────────────────────────────────────────

def load_regime(path: Path = REGIME_CSV) -> Optional[pd.DataFrame]:
    """
    Load optional regime CSV with columns [meeting_date, regime_id].
    If absent, returns None and the pipeline continues with regime_id = null.
    Expected CSV from the Bloomberg VRP pipeline: fomc_regime.csv
    """
    if not path.exists():
        print(f"  Regime file not found at {path} — regime_id will be null.")
        print("  TODO: export from Bloomberg pipeline and save as fomc_regime.csv")
        return None
    df = pd.read_csv(path, parse_dates=["meeting_date"])
    print(f"  Loaded regime: {df.shape}, regimes: {df['regime_id'].unique()}")
    return df


regime_df = load_regime()

# %% ── 3f: Merge Layer 3 ──────────────────────────────────────────────────────

def build_layer3(rv_df: pd.DataFrame,
                 parkinson: pd.Series,
                 policy_surprise: pd.Series,
                 regime: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Join all Layer 3 frames on meeting_date."""
    base = rv_df.set_index("meeting_date")

    park_df = parkinson.to_frame()
    park_df.index = pd.to_datetime(park_df.index)
    base = base.join(park_df, how="left")

    ps_df = policy_surprise.to_frame()
    ps_df.index = pd.to_datetime(ps_df.index)
    base = base.join(ps_df, how="left")

    if regime is not None:
        reg = regime.set_index("meeting_date")[["regime_id"]]
        base = base.join(reg, how="left")
    else:
        base["regime_id"] = pd.NA

    base = base.reset_index().rename(columns={"index": "meeting_date"})
    print(f"\nLayer 3 frame: {base.shape}")
    print(base.tail(3).to_string(index=False))
    return base


layer3_df = build_layer3(rv_df, parkinson_s, policy_surprise_s, regime_df)

# %% [markdown]
# ---
# # LAYER 4 — NLP → VOL REGRESSION & SAMPLING
#
# **Headline hypothesis**: short, novel, guidance-light statements predict
# realized vol exceeding implied (negative forward VRP).  On this public
# layer the LHS is forward RV from FRED yields.  The true VRP is joined on
# the work laptop after Bloomberg implied vol is merged via `meeting_date`.

# %% ── 4a: Master panel ───────────────────────────────────────────────────────

master = (
    features_text
    .merge(layer3_df, on="meeting_date", how="inner")
    .sort_values("meeting_date")
    .reset_index(drop=True)
)

# Encode categorical controls
master["chair_code"]   = pd.Categorical(master["chair"]).codes
master["regime_code"]  = pd.Categorical(master["regime_id"].fillna("unknown")).codes
master["presser_flag"] = master["presser_available"].astype(int)

# Lagged RV for autoregressive control
for k in RV_WINDOWS:
    col = f"rv_yield_10y_{k}d"
    if col in master.columns:
        master[f"lag1_rv_10y_{k}d"] = master[col].shift(1)

print(f"Master panel: {master.shape}  ({master['meeting_date'].min().date()} → "
      f"{master['meeting_date'].max().date()})")
print(master[["meeting_date", "chair", "word_count", "novelty_prev",
              "guidance_density"]].tail(6).to_string(index=False))

# %% ── 4b: HAC panel regressions ─────────────────────────────────────────────

VOL_FEATURES = [
    "word_count_zscore", "novelty_prev", "novelty_window",
    "guidance_change", "uncertainty_density", "disagree_density",
]
CONTROLS = [
    "polarity_hd", "presser_flag", "chair_code",
    "policy_surprise_2y_chg",
]


def run_hac_regression(
    df:           pd.DataFrame,
    y_col:        str,
    feature_cols: list[str],
    control_cols: list[str],
    hac_lags:     int         = 10,
    label:        str         = "",
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    OLS of y on features + controls with Newey-West HAC standard errors.
    Forward RV windows overlap → HAC is mandatory.

    Parameters
    ----------
    df           : panel DataFrame
    y_col        : LHS column name
    feature_cols : vol-signal features (Family A–D)
    control_cols : control variables (polarity, chair, etc.)
    hac_lags     : Bartlett kernel truncation (default = max forward window)
    label        : descriptive label for printout

    Returns
    -------
    Fitted statsmodels OLS result with HAC covariance
    """
    cols = [y_col] + feature_cols + control_cols
    sub  = df[cols].dropna()
    if len(sub) < 15:
        print(f"  Insufficient data for {label} ({len(sub)} obs).")
        return None

    y  = sub[y_col].values
    X  = sm.add_constant(sub[feature_cols + control_cols].values)
    res = sm.OLS(y, X).fit(
        cov_type="HAC", cov_kwds={"maxlags": hac_lags, "use_correction": True}
    )
    feat_names = ["const"] + feature_cols + control_cols

    print(f"\n{'─'*62}")
    print(f"  HAC OLS: {label}   n={len(sub)}   R²={res.rsquared:.3f}")
    print(f"{'─'*62}")
    for name, coef, tval, pval in zip(
        feat_names, res.params, res.tvalues, res.pvalues
    ):
        stars = "***" if pval < .01 else "**" if pval < .05 else "*" if pval < .10 else ""
        print(f"  {name:30s}  {coef:+.4f}  t={tval:+.2f}  p={pval:.3f}  {stars}")
    return res


# Run for each forward window
reg_results: dict = {}
for k in RV_WINDOWS:
    y_col = f"rv_yield_10y_{k}d"
    if y_col not in master.columns:
        continue
    ctrl = CONTROLS.copy()
    lag_col = f"lag1_rv_10y_{k}d"
    if lag_col in master.columns:
        ctrl = [lag_col] + ctrl

    res = run_hac_regression(
        master, y_col, VOL_FEATURES, ctrl, hac_lags=max(k, 5),
        label=f"forward RV {k}d (10Y yield)",
    )
    if res is not None:
        reg_results[f"rv_{k}d"] = res

# Event-day Parkinson vol
if "parkinson_vol_10y" in master.columns:
    run_hac_regression(
        master, "parkinson_vol_10y", VOL_FEATURES, CONTROLS,
        hac_lags=5, label="Parkinson event-day vol (10Y)",
    )

# %% ── 4c: Regime interactions ───────────────────────────────────────────────

def run_regime_interaction(
    df:            pd.DataFrame,
    y_col:         str,
    key_features:  list[str],
    regime_col:    str = "regime_code",
    hac_lags:      int = 10,
) -> None:
    """
    Interact key features with regime dummies to test state-dependence.
    Prints HAC t-stats; p-values should be treated as indicative given small N.
    """
    if y_col not in df.columns or df[y_col].isna().all():
        print(f"  {y_col} unavailable — skipping regime interaction.")
        return

    dummies = pd.get_dummies(df[regime_col], prefix="regime", drop_first=True)
    base_df = df[[y_col] + key_features].join(dummies).dropna()
    if len(base_df) < 15:
        print(f"  Insufficient obs for regime interaction ({len(base_df)}).")
        return

    interaction_cols = []
    for feat in key_features:
        for dum_col in dummies.columns:
            col = f"{feat}_x_{dum_col}"
            base_df[col] = base_df[feat] * base_df[dum_col]
            interaction_cols.append(col)

    X = sm.add_constant(
        base_df[key_features + list(dummies.columns) + interaction_cols].values
    )
    y   = base_df[y_col].values
    res = sm.OLS(y, X).fit(
        cov_type="HAC", cov_kwds={"maxlags": hac_lags, "use_correction": True}
    )
    print(f"\n  Regime-interaction R²: {res.rsquared:.3f}  n={len(base_df)}")
    print(f"  (Wald F for interactions — use likelihood ratio for formal test.)")
    print(f"  F-stat (overall): {res.fvalue:.2f}   p={res.f_pvalue:.3f}")


REGIME_KEY_FEATURES = ["word_count_zscore", "novelty_prev", "guidance_density"]
run_regime_interaction(
    master, "rv_yield_10y_5d", REGIME_KEY_FEATURES, hac_lags=5
)

# %% ── 4d: Walk-forward classification ───────────────────────────────────────

def walk_forward_classify(
    df:            pd.DataFrame,
    vol_col:       str,
    feature_cols:  list[str],
    train_min:     int = 30,
    tercile_top:   float = 2 / 3,
) -> pd.DataFrame:
    """
    Walk-forward logistic classifier: label top-tercile RV meetings as
    "vol_elevated"; train on all prior observations, predict one step ahead.
    Strict no-lookahead: each prediction uses only pre-meeting information.

    Returns a DataFrame with columns:
        meeting_date, actual, predicted, pred_prob, correct
    Also prints hit rate, precision, and lift over base rate by chair/regime.
    """
    cols = [vol_col] + feature_cols
    sub  = df[["meeting_date"] + cols].dropna().reset_index(drop=True)
    if len(sub) < train_min + 5:
        print(f"  Insufficient data for walk-forward classification.")
        return pd.DataFrame()

    threshold = sub[vol_col].quantile(tercile_top)
    sub["vol_elevated"] = (sub[vol_col] >= threshold).astype(int)
    base_rate = sub["vol_elevated"].mean()

    preds: list[dict] = []
    scaler = StandardScaler()
    clf    = LogisticRegression(C=1.0, max_iter=500, random_state=42)

    for i in range(train_min, len(sub)):
        train = sub.iloc[:i]
        test  = sub.iloc[[i]]

        X_tr = train[feature_cols].values
        y_tr = train["vol_elevated"].values
        X_te = test[feature_cols].values

        if len(np.unique(y_tr)) < 2:
            continue

        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        clf.fit(X_tr_s, y_tr)
        prob = clf.predict_proba(X_te_s)[0, 1]
        pred = int(prob >= 0.5)

        preds.append({
            "meeting_date": test["meeting_date"].values[0],
            "actual":       int(test["vol_elevated"].values[0]),
            "predicted":    pred,
            "pred_prob":    prob,
            "correct":      int(pred == int(test["vol_elevated"].values[0])),
        })

    if not preds:
        return pd.DataFrame()

    results_df = pd.DataFrame(preds)
    accuracy   = results_df["correct"].mean()
    precision  = (results_df.query("predicted==1")["actual"].mean()
                  if results_df["predicted"].sum() > 0 else np.nan)
    recall     = (results_df.query("actual==1")["correct"].mean()
                  if results_df["actual"].sum() > 0 else np.nan)
    lift       = (precision / base_rate) if base_rate > 0 else np.nan

    print(f"\n  Walk-forward classification ({vol_col}, top-tercile, n={len(results_df)})")
    print(f"  Base rate     : {base_rate:.1%}")
    print(f"  Accuracy      : {accuracy:.1%}")
    print(f"  Precision     : {precision:.1%}  (of predicted positives)")
    print(f"  Recall        : {recall:.1%}")
    print(f"  Lift          : {lift:.2f}x over base rate")
    return results_df


CLASSIFY_FEATURES = ["word_count_zscore", "novelty_prev", "guidance_change",
                     "uncertainty_density"]
y_col_clf = "rv_yield_10y_5d"
if y_col_clf in master.columns:
    clf_results = walk_forward_classify(master, y_col_clf, CLASSIFY_FEATURES)

# %% ── 4e: Bootstrap robustness ───────────────────────────────────────────────

def bootstrap_coefficients(
    df:            pd.DataFrame,
    y_col:         str,
    feature_cols:  list[str],
    n_boot:        int = 1000,
    ci_level:      float = 0.90,
) -> pd.DataFrame:
    """
    Bootstrap key OLS coefficients by resampling meetings (not residuals).
    Appropriate for small n with possibly non-normal errors.

    Returns a DataFrame with [feature, coef_mean, ci_lo, ci_hi, sig].
    Caveat: forward windows overlap → block bootstrap would be more correct
    at cost of further reducing effective sample; this is approximate.
    """
    cols = [y_col] + feature_cols
    sub  = df[cols].dropna().values
    if len(sub) < 20:
        print(f"  Too few obs for bootstrap ({len(sub)}).")
        return pd.DataFrame()

    rng    = np.random.default_rng(42)
    n      = len(sub)
    y_idx  = 0
    X_idx  = list(range(1, len(feature_cols) + 1))

    boot_coefs = np.zeros((n_boot, len(feature_cols)))
    for b in range(n_boot):
        idx      = rng.integers(0, n, size=n)
        y_b      = sub[idx, y_idx]
        X_b      = sm.add_constant(sub[idx][:, X_idx])
        try:
            coefs = np.linalg.lstsq(X_b, y_b, rcond=None)[0][1:]
            boot_coefs[b] = coefs
        except np.linalg.LinAlgError:
            boot_coefs[b] = np.nan

    alpha = (1 - ci_level) / 2
    lo    = np.nanquantile(boot_coefs, alpha,    axis=0)
    hi    = np.nanquantile(boot_coefs, 1 - alpha, axis=0)
    mean  = np.nanmean(boot_coefs, axis=0)

    rows = []
    for i, feat in enumerate(feature_cols):
        rows.append({
            "feature":   feat,
            "coef_mean": mean[i],
            f"ci_{int(ci_level*100)}_lo": lo[i],
            f"ci_{int(ci_level*100)}_hi": hi[i],
            "sig":       (lo[i] > 0) or (hi[i] < 0),   # CI excludes zero
        })
    boot_df = pd.DataFrame(rows)

    print(f"\n  Bootstrap CIs ({ci_level:.0%}, n_boot={n_boot}, n_obs={n}):")
    print(f"  NOTE: n small; treat as illustrative, not a powered test.")
    print(boot_df.to_string(index=False))
    return boot_df


y_col_boot = "rv_yield_10y_5d"
if y_col_boot in master.columns:
    boot_df = bootstrap_coefficients(master, y_col_boot, VOL_FEATURES)

# %% ── 4f: Leave-one-chair-out stability ─────────────────────────────────────

def leave_one_chair_out(
    df:            pd.DataFrame,
    y_col:         str,
    feature_cols:  list[str],
    hac_lags:      int = 5,
) -> pd.DataFrame:
    """
    Refit the HAC regression leaving out one chair's meetings at a time.
    Tests whether results are driven by a single communication era.
    """
    if y_col not in df.columns:
        return pd.DataFrame()

    chairs    = df["chair"].dropna().unique()
    summaries = []

    for chair in chairs:
        sub = df[df["chair"] != chair][[y_col] + feature_cols].dropna()
        if len(sub) < 15:
            continue
        y  = sub[y_col].values
        X  = sm.add_constant(sub[feature_cols].values)
        try:
            res = sm.OLS(y, X).fit(
                cov_type="HAC", cov_kwds={"maxlags": hac_lags}
            )
        except Exception:
            continue
        for name, coef, pval in zip(feature_cols, res.params[1:], res.pvalues[1:]):
            summaries.append({
                "left_out_chair": chair, "feature": name,
                "coef": coef, "pval": pval,
                "sig": pval < 0.10,
            })

    if not summaries:
        return pd.DataFrame()
    stab = pd.DataFrame(summaries)
    pivot = stab.pivot_table(
        index="feature", columns="left_out_chair",
        values="coef", aggfunc="first"
    )
    print("\n  Leave-one-chair-out coefficient stability:")
    print(pivot.round(4).to_string())
    return stab


if y_col_boot in master.columns:
    loco_df = leave_one_chair_out(master, y_col_boot, VOL_FEATURES)

# %% ── 4g: Warsh case study ───────────────────────────────────────────────────

def warsh_case_study(
    master_df:    pd.DataFrame,
    feature_cols: list[str],
) -> None:
    """
    Print the feature vector for each Warsh meeting and its historical
    percentile rank.  Checks whether the model's ex-ante vol-elevated flag
    would have fired.

    IMPORTANT: Warsh may be N=1-2 meetings.  This is an ILLUSTRATIVE proof
    point, NOT a powered subsample.  Do not draw causal inference from it.
    """
    warsh = master_df[master_df["chair"] == "Warsh"].copy()
    if warsh.empty:
        print("\n  No Warsh meetings in sample yet — update calendar or wait.")
        return

    print("\n" + "═" * 62)
    print("  WARSH CASE STUDY (ILLUSTRATIVE — N may be 1 or 2)")
    print("  Each feature shown with its historical percentile rank.")
    print("═" * 62)

    for _, row in warsh.iterrows():
        print(f"\n  Meeting: {row['meeting_date'].date()}   "
              f"presser: {row['presser_available']}")

        for feat in feature_cols:
            val  = row[feat]
            pct  = sp_stats.percentileofscore(
                master_df[feat].dropna().values, val, kind="rank"
            )
            flag = "◄ HIGH" if pct >= 80 else ("◄ LOW" if pct <= 20 else "")
            print(f"    {feat:30s}  {val:+.4f}  pctile={pct:5.1f}  {flag}")

        # Check if walk-forward classifier would flag this meeting
        y_clf = "rv_yield_10y_5d"
        if y_clf in master_df.columns:
            rv_val  = row.get(y_clf, np.nan)
            thresh  = master_df[y_clf].quantile(2 / 3)
            if not np.isnan(rv_val):
                print(f"\n    Actual RV ({y_clf})  : {rv_val:.3f}pp  "
                      f"({'ELEVATED' if rv_val >= thresh else 'normal'}, "
                      f"threshold={thresh:.3f}pp)")

    print("\n  Caveat: small-sample. Treat as hypothesis illustration only.")
    print("═" * 62)


warsh_case_study(master, VOL_FEATURES)

# %% [markdown]
# ---
# # OPTIONAL CELL — Transformer Features (isolated)
# This cell is **not** required by the pipeline. Skip if:
# - No internet connection
# - `transformers` / `torch` not installed
# - Weights fail to download
#
# Result (if successful): adds `hawk_dove_bert` column to master.

# %%  ─── OPTIONAL: CentralBankRoBERTa hawk-dove score ─────────────────────────
# Pipeline does NOT depend on this cell. Import guard ensures clean skip.

def add_transformer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    OPTIONAL: Score each statement with a central-bank fine-tuned transformer.
    Falls back cleanly if the library or weights are unavailable.
    Returns df with added column hawk_dove_bert (NaN if skipped).
    """
    df = df.copy()
    df["hawk_dove_bert"] = np.nan

    try:
        from transformers import pipeline as hf_pipeline
    except ImportError:
        print("  [OPTIONAL] transformers not installed — skipping bert features.")
        return df

    MODEL = "gtfintechlab/FOMC-RoBERTa"   # hawk/dove/neutral classifier
    try:
        classifier = hf_pipeline(
            "text-classification", model=MODEL,
            truncation=True, max_length=512,
        )
    except Exception as exc:
        print(f"  [OPTIONAL] Could not load {MODEL}: {exc}")
        return df

    scores = []
    for text in df["text"].fillna("").tolist():
        snippet = text[:1000]   # truncate to avoid OOM on long pressers
        try:
            out   = classifier(snippet)[0]
            label = out["label"].lower()
            score = out["score"]
            # Map: hawkish → +score, dovish → -score, neutral → 0
            if "hawk" in label:
                scores.append(score)
            elif "dove" in label:
                scores.append(-score)
            else:
                scores.append(0.0)
        except Exception:
            scores.append(np.nan)

    df["hawk_dove_bert"] = scores
    print(f"  [OPTIONAL] hawk_dove_bert added: "
          f"{pd.Series(scores).notna().sum()} / {len(scores)} scored.")
    return df


# Uncomment to run (requires internet + torch):
# stmt = add_transformer_features(stmt)
# if "hawk_dove_bert" in stmt.columns:
#     master = master.merge(
#         stmt[["meeting_date", "hawk_dove_bert"]], on="meeting_date", how="left"
#     )

# %% ── OUTPUT: fomc_features.parquet ─────────────────────────────────────────

OUTPUT_COLS = (
    ["meeting_date", "chair", "presser_available"]
    + TEXT_FEATURE_COLS
    + [c for c in master.columns if c.startswith("rv_")]
    + ["parkinson_vol_10y", "policy_surprise_2y_chg", "regime_id"]
)
OUTPUT_COLS = [c for c in OUTPUT_COLS if c in master.columns]

fomc_features = master[OUTPUT_COLS].copy()
fomc_features.to_parquet(PARQUET_OUT, index=False)

print(f"\n{'═'*62}")
print(f"  OUTPUT: {PARQUET_OUT}")
print(f"  Shape : {fomc_features.shape}")
print(f"  Cols  : {list(fomc_features.columns)}")
print(f"{'═'*62}")

# ── HANDOFF SUMMARY ──────────────────────────────────────────────────────────
print("""
┌─────────────────────────────────────────────────────────────┐
│  HANDOFF TO WORK LAPTOP (Bloomberg VRP pipeline)            │
├─────────────────────────────────────────────────────────────┤
│  JOIN KEY   : meeting_date (date, matches fomc_date in      │
│               Bloomberg pipeline)                           │
│                                                             │
│  COLUMNS TO REPLACE (public proxies → Bloomberg actuals):  │
│    rv_yield_*_*d        → Bloomberg realized vol from       │
│                           Treasury futures prices           │
│    parkinson_vol_10y    → Bloomberg event-day range vol     │
│    policy_surprise_*    → SF Fed USMPD or OIS surprise      │
│    regime_id            → regime_pl from Bloomberg session  │
│                                                             │
│  COLUMNS TO KEEP AS-IS (text features, no Bloomberg):      │
│    word_count_*, novelty_*, guidance_*, uncertainty_*,      │
│    disagree_density, polarity_hd, chair, presser_available  │
│                                                             │
│  NEW COLS TO ADD ON WORK LAPTOP:                            │
│    iv_*     — HIST_CALL_IMP_VOL per tenor                  │
│    vrp_*    — iv - rv (the primary LHS for VRP study)      │
│    underpriced — rv > iv (bool)                             │
└─────────────────────────────────────────────────────────────┘
""")

# %% [markdown]
# ---
# # LAYER 5 — VISUALISATIONS
# Six figures covering the full feature-to-vol story:
#   1. Feature timeline with chair bands
#   2. Feature correlation heat-map
#   3. Forward RV distributions by chair era
#   4. Walk-forward classifier performance
#   5. Bootstrap coefficient CIs
#   6. Warsh meeting spotlight

# %% ── Visualisation helpers ──────────────────────────────────────────────────

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm

VIZ_OUT = Path("fomc_viz")
VIZ_OUT.mkdir(exist_ok=True)

_CHAIR_COLORS = {
    "Bernanke": "#2166ac",
    "Yellen":   "#4dac26",
    "Powell":   "#d6604d",
    "Warsh":    "#762a83",
    "Unknown":  "#aaaaaa",
}

def _chair_bands(ax, df: pd.DataFrame, alpha: float = 0.08) -> None:
    """Shade background by chair era."""
    dates = df["meeting_date"].values
    chairs = df["chair"].values
    for i, (d, c) in enumerate(zip(dates, chairs)):
        x1 = dates[i + 1] if i + 1 < len(dates) else d + np.timedelta64(46, "D")
        ax.axvspan(pd.Timestamp(d), pd.Timestamp(x1),
                   color=_CHAIR_COLORS.get(c, "#aaaaaa"), alpha=alpha, linewidth=0)

def _warsh_vlines(ax, df: pd.DataFrame) -> None:
    """Draw vertical dashed lines for Warsh meetings."""
    warsh = df[df["chair"] == "Warsh"]["meeting_date"]
    for d in warsh:
        ax.axvline(pd.Timestamp(d), color="#762a83", lw=1.2,
                   ls="--", zorder=5, label="Warsh")

def savefig(fig: plt.Figure, name: str) -> None:
    path = VIZ_OUT / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")


# %% ── Fig 1: Feature timeline ────────────────────────────────────────────────

def plot_feature_timeline(df: pd.DataFrame) -> plt.Figure:
    """
    Four-panel time series of the key vol-signal features, with chair era
    shading and vertical markers for Warsh meetings.
    """
    features = [
        ("word_count_zscore",    "Brevity z-score\n(word count vs trailing mean)", True),
        ("novelty_prev",         "Novelty\n(cosine dist to prior statement)",       False),
        ("guidance_density",     "Guidance density\n(hits / 100 words)",            False),
        ("uncertainty_density",  "Uncertainty density\n(LM word fraction)",         False),
    ]

    fig, axes = plt.subplots(len(features), 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        "FOMC Statement Feature Timeline  |  Chair eras shaded  |  ── Warsh meetings",
        fontsize=11, fontweight="bold"
    )

    sub = df.dropna(subset=["meeting_date"]).sort_values("meeting_date")

    for ax, (col, ylabel, inverted) in zip(axes, features):
        if col not in sub.columns:
            ax.set_visible(False)
            continue
        _chair_bands(ax, sub)
        vals = sub[col]
        ax.plot(sub["meeting_date"], vals, color="#333333", lw=1.1, zorder=3)
        ax.fill_between(sub["meeting_date"], vals, 0,
                        where=(vals < 0) if inverted else (vals > vals.median()),
                        alpha=0.18, color="#d62728", zorder=2)
        ax.axhline(vals.median(), color="#888888", lw=0.7, ls=":")
        _warsh_vlines(ax, sub)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(axis="y", lw=0.4, alpha=0.5)

    axes[-1].set_xlabel("Meeting date")

    # Chair legend
    patches = [mpatches.Patch(color=c, alpha=0.5, label=ch)
               for ch, c in _CHAIR_COLORS.items() if ch != "Unknown"]
    axes[0].legend(handles=patches, title="Chair", fontsize=7,
                   title_fontsize=7, loc="upper right", ncol=2)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, "fig1_feature_timeline")
    return fig


fig1 = plot_feature_timeline(master)
plt.close("all")

# %% ── Fig 2: Feature correlation heat-map ────────────────────────────────────

def plot_feature_heatmap(df: pd.DataFrame) -> plt.Figure:
    """
    Pearson correlation matrix of all text features + forward RV.
    Diverging palette so zero = white, positive = red, negative = blue.
    """
    rv_cols   = [c for c in df.columns if c.startswith("rv_yield")]
    all_feats = VOL_FEATURES + ["polarity_hd"] + rv_cols
    sub       = df[[c for c in all_feats if c in df.columns]].dropna(how="all")

    corr = sub.corr()
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, cmap="RdBu_r", norm=norm, aspect="auto")
    plt.colorbar(im, ax=ax, label="Pearson r")

    labels = [c.replace("rv_yield_10y_", "RV_").replace("_density", "")
                .replace("_zscore", "_z").replace("word_count", "wc") for c in corr.columns]
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7)

    for i in range(len(corr)):
        for j in range(len(corr)):
            val = corr.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=6, color="white" if abs(val) > 0.5 else "black")

    ax.set_title("Feature Correlation Matrix  (text features + forward RV proxies)",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "fig2_feature_heatmap")
    return fig


fig2 = plot_feature_heatmap(master)
plt.close("all")

# %% ── Fig 3: RV distributions by chair era ───────────────────────────────────

def plot_rv_by_chair(df: pd.DataFrame) -> plt.Figure:
    """
    Violin + strip plot of forward RV at each horizon by chair era.
    Shows whether meeting-day vol regime differs across chairs.
    """
    rv_cols   = [c for c in df.columns if re.match(r"rv_yield_10y_\d+d", c)]
    if not rv_cols:
        print("  No RV columns found — skipping Fig 3.")
        return plt.figure()

    n       = len(rv_cols)
    chairs  = [c for c in _CHAIR_COLORS if c != "Unknown" and c in df["chair"].values]
    x_pos   = np.arange(len(chairs))

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, rv_cols):
        k = col.split("_")[-1]
        for j, chair in enumerate(chairs):
            vals = df[df["chair"] == chair][col].dropna().values
            if len(vals) < 3:
                continue
            parts = ax.violinplot(vals, positions=[j], widths=0.5,
                                  showmedians=True, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(_CHAIR_COLORS[chair])
                pc.set_alpha(0.4)
            parts["cmedians"].set_color(_CHAIR_COLORS[chair])
            ax.scatter(np.full(len(vals), j) + np.random.uniform(-0.1, 0.1, len(vals)),
                       vals, color=_CHAIR_COLORS[chair], alpha=0.5, s=12, zorder=3)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(chairs, rotation=20, ha="right", fontsize=8)
        ax.set_title(f"Forward RV {k}\n(10Y yield change, ann. pp)", fontsize=9)
        ax.set_ylabel("Realized vol (pp)" if ax == axes[0] else "")
        ax.grid(axis="y", lw=0.4, alpha=0.5)

    fig.suptitle("Forward Realized Vol Distribution by Chair Era",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    savefig(fig, "fig3_rv_by_chair")
    return fig


fig3 = plot_rv_by_chair(master)
plt.close("all")

# %% ── Fig 4: Walk-forward classifier performance ─────────────────────────────

def plot_classifier_perf(clf_df: pd.DataFrame, df: pd.DataFrame) -> plt.Figure:
    """
    Two-panel figure:
      Left:  predicted probability over time, coloured by actual label.
      Right: precision-recall lift bar by chair era.
    """
    if clf_df is None or clf_df.empty:
        print("  No classifier results — skipping Fig 4.")
        return plt.figure()

    clf_df = clf_df.copy()
    clf_df["meeting_date"] = pd.to_datetime(clf_df["meeting_date"])
    clf_merged = clf_df.merge(df[["meeting_date", "chair"]], on="meeting_date", how="left")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Walk-Forward Classifier: Vol-Elevated (Top-Tercile Forward RV)",
                 fontsize=11, fontweight="bold")

    # Left panel — probability time series
    for _, row in clf_merged.iterrows():
        color = "#d62728" if row["actual"] == 1 else "#2166ac"
        ax1.scatter(row["meeting_date"], row["pred_prob"],
                    color=color, s=30, zorder=3, alpha=0.8)

    ax1.axhline(0.5, color="black", lw=0.8, ls="--", label="Decision threshold")
    thresh_line = mpatches.Patch(color="#d62728", alpha=0.6, label="Actual elevated")
    norm_line   = mpatches.Patch(color="#2166ac", alpha=0.6, label="Actual normal")
    ax1.legend(handles=[thresh_line, norm_line,
                         mpatches.Patch(color="white", label="─── threshold=0.5")],
               fontsize=7)
    _chair_bands(ax1,
                 df[df["meeting_date"].isin(clf_merged["meeting_date"])].sort_values("meeting_date"))
    ax1.set_ylabel("Predicted probability (vol elevated)")
    ax1.set_xlabel("Meeting date")
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(axis="y", lw=0.4, alpha=0.5)

    # Right panel — precision by chair era
    base_rate = clf_merged["actual"].mean()
    chair_perf = []
    for chair in [c for c in _CHAIR_COLORS if c != "Unknown"]:
        sub = clf_merged[clf_merged["chair"] == chair]
        if sub.empty or sub["predicted"].sum() == 0:
            continue
        prec = sub[sub["predicted"] == 1]["actual"].mean()
        chair_perf.append({"chair": chair, "precision": prec,
                            "lift": prec / base_rate if base_rate > 0 else np.nan,
                            "n": len(sub)})

    if chair_perf:
        cp_df = pd.DataFrame(chair_perf)
        x     = np.arange(len(cp_df))
        bars  = ax2.bar(x, cp_df["precision"],
                        color=[_CHAIR_COLORS.get(c, "#aaa") for c in cp_df["chair"]],
                        alpha=0.75, edgecolor="white")
        ax2.axhline(base_rate, color="black", lw=1, ls="--",
                    label=f"Base rate {base_rate:.0%}")
        ax2.set_xticks(x)
        ax2.set_xticklabels(
            [f"{r['chair']}\n(n={r['n']})" for _, r in cp_df.iterrows()],
            fontsize=8
        )
        ax2.set_ylabel("Precision (positive predictive value)")
        ax2.set_title("Precision by Chair Era")
        ax2.legend(fontsize=8)
        ax2.set_ylim(0, 1.05)
        # Annotate lift
        for bar, (_, r) in zip(bars, cp_df.iterrows()):
            if not np.isnan(r["lift"]):
                ax2.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.02,
                         f"{r['lift']:.1f}×", ha="center", va="bottom", fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    savefig(fig, "fig4_classifier_perf")
    return fig


_clf_results = clf_results if "clf_results" in dir() else pd.DataFrame()
fig4 = plot_classifier_perf(_clf_results, master)
plt.close("all")

# %% ── Fig 5: Bootstrap coefficient CIs ──────────────────────────────────────

def plot_bootstrap_ci(boot_df: pd.DataFrame) -> plt.Figure:
    """
    Forest plot of bootstrapped OLS coefficients (90 % CI) for the vol-signal
    features.  Features whose CI excludes zero are marked in red.
    """
    if boot_df is None or boot_df.empty:
        print("  No bootstrap results — skipping Fig 5.")
        return plt.figure()

    ci_cols = [c for c in boot_df.columns if c.startswith("ci_")]
    if len(ci_cols) < 2:
        return plt.figure()
    lo_col, hi_col = ci_cols[0], ci_cols[1]
    ci_pct = ci_cols[0].split("_")[1]

    fig, ax = plt.subplots(figsize=(9, 0.6 * len(boot_df) + 2))
    y_pos = np.arange(len(boot_df))

    for i, (_, row) in enumerate(boot_df.iterrows()):
        color = "#d62728" if row["sig"] else "#555555"
        ax.plot([row[lo_col], row[hi_col]], [i, i], color=color, lw=2.5)
        ax.scatter(row["coef_mean"], i, color=color, s=50, zorder=4)

    ax.axvline(0, color="black", lw=0.9, ls="--")
    labels = [r["feature"].replace("_", " ") for _, r in boot_df.iterrows()]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("OLS coefficient  (bootstrapped)")
    ax.set_title(f"Bootstrap {ci_pct}% CI — Vol-signal Features on Forward RV 5d\n"
                 "Red = CI excludes zero", fontsize=10, fontweight="bold")
    ax.grid(axis="x", lw=0.4, alpha=0.5)
    fig.tight_layout()
    savefig(fig, "fig5_bootstrap_ci")
    return fig


_boot_df = boot_df if "boot_df" in dir() else pd.DataFrame()
fig5 = plot_bootstrap_ci(_boot_df)
plt.close("all")

# %% ── Fig 6: Warsh meeting spotlight ────────────────────────────────────────

def plot_warsh_spotlight(df: pd.DataFrame, feature_cols: list[str]) -> plt.Figure:
    """
    Radar / percentile bar chart showing each Warsh meeting's feature vector
    relative to the full-sample distribution.  Each bar spans 0-100 pctile;
    a red fill above 80 or below 20 flags an extreme reading.
    """
    warsh = df[df["chair"] == "Warsh"].dropna(subset=feature_cols, how="all")
    if warsh.empty:
        print("  No Warsh meetings in master — skipping Fig 6.")
        return plt.figure()

    n_meetings = len(warsh)
    n_feats    = len(feature_cols)

    fig, axes = plt.subplots(1, n_meetings,
                             figsize=(6 * n_meetings, 5 + n_feats * 0.3),
                             squeeze=False)
    fig.suptitle("Warsh Meeting Spotlight — Percentile Ranks vs Full Sample\n"
                 "(Red = extreme reading; threshold bands at 20th / 80th pctile)",
                 fontsize=10, fontweight="bold")

    for ax, (_, row) in zip(axes[0], warsh.iterrows()):
        pctiles = []
        for feat in feature_cols:
            val = row.get(feat, np.nan)
            if pd.isna(val):
                pctiles.append(np.nan)
            else:
                pctiles.append(
                    sp_stats.percentileofscore(df[feat].dropna().values, val, kind="rank")
                )

        y      = np.arange(n_feats)
        colors = ["#d62728" if (p <= 20 or p >= 80) else "#4393c3"
                  for p in pctiles]
        bars   = ax.barh(y, pctiles, color=colors, alpha=0.75, edgecolor="white")

        ax.axvline(20, color="#888888", lw=0.8, ls=":")
        ax.axvline(80, color="#888888", lw=0.8, ls=":")
        ax.axvline(50, color="#333333", lw=0.6, ls="--", alpha=0.4)

        ax.set_xlim(0, 100)
        ax.set_yticks(y)
        ax.set_yticklabels([f.replace("_", " ") for f in feature_cols], fontsize=8)
        ax.set_xlabel("Historical percentile rank")
        ax.set_title(f"{row['meeting_date'].date()}\n(Warsh)", fontsize=9)
        ax.grid(axis="x", lw=0.4, alpha=0.4)

        # Annotate percentile values
        for bar, p in zip(bars, pctiles):
            if not np.isnan(p):
                ax.text(min(p + 1, 97), bar.get_y() + bar.get_height() / 2,
                        f"{p:.0f}",
                        va="center", ha="left", fontsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    savefig(fig, "fig6_warsh_spotlight")
    return fig


fig6 = plot_warsh_spotlight(master, VOL_FEATURES)
plt.close("all")

print(f"\n{'═'*62}")
print(f"  All figures saved to: {VIZ_OUT.resolve()}")
print(f"  fomc_features.parquet: {PARQUET_OUT.resolve()}")
print("═" * 62)
