"""
rebuild_presser_corpus.py
Re-download FOMC press-conference PDFs, extract real transcripts, recompute NLP
features, and update fomc_corpus_expanded.parquet + fomc_features.parquet.
"""

from __future__ import annotations
import os, re, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import pdfplumber

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO = Path("/Users/zinuoshi/fomc-vol-study-repo")
CORPUS_PATH  = REPO / "fomc_corpus_expanded.parquet"
FEATURES_PATH = REPO / "fomc_features.parquet"
PDF_CACHE    = REPO / "fomc_cache" / "pressers_pdf"
PDF_CACHE.mkdir(parents=True, exist_ok=True)

PDF_URL_T = "https://www.federalreserve.gov/mediacenter/files/FOMCpresconf{date}.pdf"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Referer": "https://www.federalreserve.gov/",
}

RATE_LIMIT = 1.5  # seconds between requests

# ── Lexicons ──────────────────────────────────────────────────────────────────
UNCERTAINTY_LEXICON = {
    "uncertain", "uncertainty", "unclear", "risk", "risks", "unstable", "volatile",
    "volatility", "challenge", "challenging", "unpredictable", "ambiguous",
    "moderate", "modestly", "roughly", "broadly", "approximately", "flexible",
}
GUIDANCE_LEXICON = {
    "expect", "expected", "expects", "anticipate", "anticipated", "anticipates",
    "project", "projected", "projects", "forecast", "likely", "likely to", "forward",
    "guidance", "future", "longer-run", "longer run", "long-run", "medium-term",
    "coming months", "in coming", "appropriate", "remain appropriate",
    "will be", "will remain", "intend", "intended", "plan to",
}
DISSENT_LEXICON = {
    "dissent", "dissented", "voted against", "disagreed", "objected", "opposed"
}

CHAIR_RE = re.compile(
    r'\n(CHAIR(?:MAN)?\s+(?:POWELL|YELLEN|BERNANKE|GREENSPAN|WARSH|BURNS))[.\s]+',
    re.IGNORECASE,
)

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Download PDFs
# ─────────────────────────────────────────────────────────────────────────────

def download_pdfs(dates: list[str]) -> dict[str, Path]:
    """Download PDFs for given YYYYMMDD date strings. Returns {date: path} for successes."""
    downloaded: dict[str, Path] = {}
    n_skipped = 0
    n_downloaded = 0
    n_failed = 0

    for i, d in enumerate(dates):
        path = PDF_CACHE / f"{d}.pdf"
        if path.exists() and path.stat().st_size > 5_000:
            downloaded[d] = path
            n_skipped += 1
            continue

        url = PDF_URL_T.format(date=d)
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            if resp.status_code == 200 and len(resp.content) > 5_000:
                path.write_bytes(resp.content)
                downloaded[d] = path
                n_downloaded += 1
                print(f"  [OK]  {d}  {len(resp.content):,} bytes")
            else:
                n_failed += 1
                print(f"  [SKIP] {d}  HTTP {resp.status_code} / {len(resp.content)} bytes")
        except Exception as e:
            n_failed += 1
            print(f"  [ERR] {d}  {e}")

        if i < len(dates) - 1:
            time.sleep(RATE_LIMIT)

    print(f"\nDownload summary: {n_downloaded} new, {n_skipped} cached, {n_failed} failed")
    return downloaded

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Extract text from PDFs
# ─────────────────────────────────────────────────────────────────────────────

def extract_pdf(path: Path) -> tuple[str, str]:
    """Return (text_full, text_chair) from a PDF."""
    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                pages.append(t)
    text_full = "\n".join(pages)
    text_chair = extract_chair_turns(text_full)
    return text_full, text_chair


def extract_chair_turns(text: str) -> str:
    """Extract chair-only turns from transcript text."""
    # Split by chair speaker marker
    parts = CHAIR_RE.split(text)
    # parts: [pre_text, speaker1, turn1, speaker2, turn2, ...]
    chair_texts = []
    i = 1
    while i + 1 < len(parts):
        speaker = parts[i]
        turn_text = parts[i + 1]
        # Keep if CHAIR or CHAIRMAN in the speaker tag
        if re.search(r'CHAIR(?:MAN)?', speaker, re.IGNORECASE):
            chair_texts.append(turn_text.strip())
        i += 2
    return "\n\n".join(chair_texts)


def count_turns(text: str) -> tuple[int, int]:
    """Return (n_chair_turns, n_total_turns) from transcript text."""
    # Find all speaker markers — the chair pattern + general SPEAKER: or NAME. pattern
    all_speakers = re.findall(
        r'\n([A-Z][A-Z\s]+(?:CHAIR(?:MAN)?|MR\.|MS\.|VICE|PRESIDENT|REPORTER|QUESTIONER)[A-Z\s]*)[.\s]+',
        text,
    )
    chair_speakers = re.findall(CHAIR_RE, text)
    # Fallback: count by how many times the chair regex fires
    n_chair = len(re.findall(CHAIR_RE, text))
    n_total = max(n_chair, len(all_speakers))
    # Simple heuristic: each newline-uppercase-name segment is a turn
    # Use a broader pattern to count total speaker turns
    all_turns = re.findall(r'\n[A-Z][A-Z\s]{2,}[.:]\s', text)
    if all_turns:
        n_total = max(n_total, len(all_turns))
    return n_chair, n_total

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: NLP features
# ─────────────────────────────────────────────────────────────────────────────

def count_lex(text: str, lexicon: set[str]) -> int:
    words = re.findall(r'\b\w+\b', text.lower())
    return sum(1 for w in words if w in lexicon)


def compute_nlp(text_full: str, n_tokens_full: int) -> dict:
    if n_tokens_full == 0:
        return dict(uncertainty_density=0.0, guidance_density=0.0, disagree_density=0.0)
    scale = n_tokens_full / 1000
    return dict(
        uncertainty_density=count_lex(text_full, UNCERTAINTY_LEXICON) / scale,
        guidance_density=count_lex(text_full, GUIDANCE_LEXICON) / scale,
        disagree_density=count_lex(text_full, DISSENT_LEXICON) / scale,
    )


def jaccard(a: str, b: str) -> float:
    s1 = set(re.findall(r'\b\w+\b', a.lower()))
    s2 = set(re.findall(r'\b\w+\b', b.lower()))
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("FOMC PRESSER CORPUS REBUILD")
    print("=" * 60)

    # Load existing corpus
    corpus = pd.read_parquet(CORPUS_PATH)
    features = pd.read_parquet(FEATURES_PATH)

    print(f"Corpus: {corpus.shape}  |  Features: {features.shape}")
    print(f"Presser rows in corpus: {(corpus.doc_type=='presser').sum()}")

    # --- Step 1: Identify presser dates and download ---
    presser_rows = corpus[corpus.doc_type == 'presser'].copy()
    dates_dt = sorted(presser_rows['meeting_date'].dt.strftime('%Y%m%d').tolist())
    print(f"\n--- Step 1: Downloading {len(dates_dt)} PDFs ---")
    pdf_paths = download_pdfs(dates_dt)
    print(f"Available PDFs: {len(pdf_paths)}")

    # --- Step 2: Extract text ---
    print("\n--- Step 2: Extracting text from PDFs ---")

    records = []
    prev_chair_text = ""

    for date_str in dates_dt:
        if date_str not in pdf_paths:
            records.append({
                'date_str': date_str,
                'text_full': '',
                'text_chair': '',
                'n_tokens_full': 0,
                'n_tokens_chair': 0,
                'n_chair_turns': 0,
                'n_total_turns': 0,
                'novelty_prev': np.nan,
                'uncertainty_density': 0.0,
                'guidance_density': 0.0,
                'disagree_density': 0.0,
                'success': False,
            })
            continue

        path = pdf_paths[date_str]
        try:
            text_full, text_chair = extract_pdf(path)
            n_tokens_full = len(text_full.split())
            n_tokens_chair = len(text_chair.split())
            n_chair_turns, n_total_turns = count_turns(text_full)

            nlp = compute_nlp(text_full, n_tokens_full)

            nov = 1 - jaccard(text_chair, prev_chair_text) if prev_chair_text else np.nan
            if text_chair:
                prev_chair_text = text_chair

            print(f"  {date_str}  full={n_tokens_full:,}  chair={n_tokens_chair:,}  "
                  f"turns={n_chair_turns}/{n_total_turns}")

            records.append({
                'date_str': date_str,
                'text_full': text_full,
                'text_chair': text_chair,
                'n_tokens_full': n_tokens_full,
                'n_tokens_chair': n_tokens_chair,
                'n_chair_turns': n_chair_turns,
                'n_total_turns': n_total_turns,
                'novelty_prev': nov,
                'uncertainty_density': nlp['uncertainty_density'],
                'guidance_density': nlp['guidance_density'],
                'disagree_density': nlp['disagree_density'],
                'success': True,
            })

        except Exception as e:
            print(f"  [ERR] {date_str}  {e}")
            records.append({
                'date_str': date_str,
                'text_full': '',
                'text_chair': '',
                'n_tokens_full': 0,
                'n_tokens_chair': 0,
                'n_chair_turns': 0,
                'n_total_turns': 0,
                'novelty_prev': np.nan,
                'uncertainty_density': 0.0,
                'guidance_density': 0.0,
                'disagree_density': 0.0,
                'success': False,
            })

    rec_df = pd.DataFrame(records)
    rec_df['meeting_date'] = pd.to_datetime(rec_df['date_str'], format='%Y%m%d')

    n_success = rec_df['success'].sum()
    n_fail = len(rec_df) - n_success
    print(f"\nExtraction: {n_success} succeeded, {n_fail} failed")

    # --- Step 4a: Update fomc_corpus_expanded.parquet ---
    print("\n--- Step 4a: Updating fomc_corpus_expanded.parquet ---")

    # Save old presser token stats for comparison
    old_presser_tokens_mean = presser_rows['n_tokens'].mean()

    # Merge new data into presser rows
    new_presser = presser_rows.copy().reset_index(drop=True)
    # Build a lookup by meeting_date
    rec_lookup = rec_df.set_index('meeting_date')

    for col in ['text_full', 'text_chair', 'n_tokens_full', 'n_tokens_chair',
                'n_chair_turns', 'n_total_turns', 'uncertainty_density',
                'guidance_density', 'disagree_density']:
        new_presser[col] = new_presser['meeting_date'].map(
            rec_lookup[col] if col in rec_lookup.columns else {}
        )

    # Replace text with text_full for the 'text' column (keep backward compat)
    new_presser['text'] = new_presser['text_full'].fillna('')

    # Update n_tokens
    new_presser['n_tokens'] = new_presser['n_tokens_full'].fillna(0).astype(int)

    # Recalculate n_chair_turns and n_total_turns from new data
    if 'n_chair_turns' in rec_lookup.columns:
        new_presser['n_chair_turns'] = new_presser['meeting_date'].map(
            rec_lookup['n_chair_turns']
        ).fillna(0).astype(int)
    if 'n_total_turns' in rec_lookup.columns:
        new_presser['n_total_turns'] = new_presser['meeting_date'].map(
            rec_lookup['n_total_turns']
        ).fillna(0).astype(int)

    # Statements remain untouched
    stmt_rows = corpus[corpus.doc_type != 'presser'].copy()

    # Add missing columns to stmt_rows so concat works
    for col in ['text_full', 'text_chair', 'n_tokens_full', 'n_tokens_chair',
                'uncertainty_density', 'guidance_density', 'disagree_density']:
        if col not in stmt_rows.columns:
            stmt_rows[col] = np.nan

    new_corpus = pd.concat([stmt_rows, new_presser], ignore_index=True)
    new_corpus = new_corpus.sort_values(['meeting_date', 'doc_type']).reset_index(drop=True)
    new_corpus.to_parquet(CORPUS_PATH, index=False)
    print(f"  Saved corpus: {new_corpus.shape}")
    print(f"  Old presser mean tokens: {old_presser_tokens_mean:.0f}")
    new_presser_tokens_mean = new_presser[new_presser['n_tokens_full'] > 0]['n_tokens_full'].mean()
    print(f"  New presser mean tokens: {new_presser_tokens_mean:.0f}")

    # --- Step 4b: Update fomc_features.parquet ---
    print("\n--- Step 4b: Updating fomc_features.parquet ---")

    feat = features.copy()

    # Create presser-specific columns
    presser_feat_cols = [
        'uncertainty_density_presser', 'guidance_density_presser',
        'disagree_density_presser', 'n_tokens_presser',
    ]
    for col in presser_feat_cols:
        if col not in feat.columns:
            feat[col] = np.nan

    # Fill presser feature columns for meetings that have a presser
    feat['meeting_date'] = pd.to_datetime(feat['meeting_date'])
    rec_lookup2 = rec_df.set_index('meeting_date')

    for idx, row in feat.iterrows():
        md = row['meeting_date']
        if md in rec_lookup2.index:
            r = rec_lookup2.loc[md]
            if r['success']:
                feat.at[idx, 'uncertainty_density_presser'] = r['uncertainty_density']
                feat.at[idx, 'guidance_density_presser']   = r['guidance_density']
                feat.at[idx, 'disagree_density_presser']   = r['disagree_density']
                feat.at[idx, 'n_tokens_presser']           = r['n_tokens_full']

    # Recompute composite features using weighted average
    # uncertainty_density: 0.6 presser + 0.4 statement (when both exist)
    stmt_feat = new_corpus[new_corpus.doc_type == 'statement'].set_index('meeting_date')

    for idx, row in feat.iterrows():
        md = row['meeting_date']
        has_presser = (
            pd.notna(row['uncertainty_density_presser']) and
            row.get('n_tokens_presser', 0) > 0
        )
        has_stmt = md in stmt_feat.index

        for base_col, p_col in [
            ('uncertainty_density', 'uncertainty_density_presser'),
            ('guidance_density', 'guidance_density_presser'),
            ('disagree_density', 'disagree_density_presser'),
        ]:
            p_val = feat.at[idx, p_col]
            s_val = stmt_feat.at[md, 'uncertainty_density'] if (
                has_stmt and base_col == 'uncertainty_density' and
                'uncertainty_density' in stmt_feat.columns
            ) else None

            if has_presser and has_stmt and s_val is not None and pd.notna(s_val):
                feat.at[idx, base_col] = 0.6 * p_val + 0.4 * float(s_val)
            elif has_presser:
                feat.at[idx, base_col] = p_val
            # else keep existing value from statement

        # Handle guidance and disagree specifically
        if has_presser:
            p_g = feat.at[idx, 'guidance_density_presser']
            p_d = feat.at[idx, 'disagree_density_presser']

            # guidance
            if has_stmt and 'guidance_density' in stmt_feat.columns and md in stmt_feat.index:
                s_g = stmt_feat.at[md, 'guidance_density'] if pd.notna(
                    stmt_feat.at[md, 'guidance_density'] if 'guidance_density' in stmt_feat.columns else np.nan
                ) else None
                if s_g is not None and pd.notna(s_g):
                    feat.at[idx, 'guidance_density'] = 0.6 * p_g + 0.4 * float(s_g)
                else:
                    feat.at[idx, 'guidance_density'] = p_g
            else:
                feat.at[idx, 'guidance_density'] = p_g

            # disagree
            if has_stmt and 'disagree_density' in stmt_feat.columns and md in stmt_feat.index:
                s_d_val = stmt_feat.at[md, 'disagree_density'] if 'disagree_density' in stmt_feat.columns else np.nan
                if pd.notna(s_d_val):
                    feat.at[idx, 'disagree_density'] = 0.6 * p_d + 0.4 * float(s_d_val)
                else:
                    feat.at[idx, 'disagree_density'] = p_d
            else:
                feat.at[idx, 'disagree_density'] = p_d

    feat.to_parquet(FEATURES_PATH, index=False)
    print(f"  Saved features: {feat.shape}")
    print(f"  Meetings with presser data: {feat['n_tokens_presser'].notna().sum()}")

    # --- Step 5: Print detailed statistics ---
    print("\n\n" + "=" * 50)
    print("=== CORPUS REBUILD SUMMARY ===")
    print("=" * 50)

    succ_df = rec_df[rec_df['success']]
    print(f"PDFs downloaded: {len(pdf_paths)} / {len(dates_dt)}")
    print(f"PDF extraction successes: {n_success}")
    print(f"Mean tokens (was {old_presser_tokens_mean:.0f}, now): {succ_df['n_tokens_full'].mean():.0f}")
    print(f"Token range: [{succ_df['n_tokens_full'].min()}, {succ_df['n_tokens_full'].max()}]")
    if n_success > 0:
        print(f"Chair turns per meeting: [{succ_df['n_chair_turns'].mean():.1f}, "
              f"{succ_df['n_chair_turns'].std():.1f}] (mean, std)")

    print("\n=== NLP FEATURES DELTA ===")
    feat_reload = pd.read_parquet(FEATURES_PATH)
    old_feat = features

    print(f"{'Feature':<30} {'Old mean':>12} {'New mean':>12} {'Delta':>12}")
    print("-" * 68)
    for col in ['uncertainty_density', 'guidance_density', 'disagree_density']:
        if col in old_feat.columns and col in feat_reload.columns:
            old_m = old_feat[col].mean()
            new_m = feat_reload[col].mean()
            delta = new_m - old_m
            print(f"{col:<30} {old_m:>12.4f} {new_m:>12.4f} {delta:>+12.4f}")

    # Warsh 2026-06-17 details
    print("\n=== WARSH (2026-06-17) ===")
    warsh_date = '20260617'
    warsh_rec = rec_df[rec_df['date_str'] == warsh_date]
    if not warsh_rec.empty and warsh_rec.iloc[0]['success']:
        wr = warsh_rec.iloc[0]
        print(f"Full text tokens: {wr['n_tokens_full']:,}  Chair tokens: {wr['n_tokens_chair']:,}")
        print(f"uncertainty_density: {wr['uncertainty_density']:.4f}  "
              f"guidance_density: {wr['guidance_density']:.4f}  "
              f"disagree_density: {wr['disagree_density']:.4f}")
        print(f"Chair turns: {wr['n_chair_turns']}  Total turns: {wr['n_total_turns']}")
        chair_preview = wr['text_chair'][:500]
        print(f"\n[First 500 chars of Warsh chair text]\n{chair_preview}")
    else:
        print("  Warsh record not found or failed extraction")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
