"""
Generator: fomc_spread_model.py  →  fomc_spread_model.ipynb
Run: python3 _make_spread_model_nb.py
Splits on # ═══ section banners; each major section becomes its own cell.
"""
import json, re
from pathlib import Path
from uuid import uuid4

SRC  = Path("fomc_spread_model.py")
DEST = Path("fomc_spread_model.ipynb")

raw   = SRC.read_text(encoding="utf-8")
lines = raw.splitlines()

# ── cell factories ────────────────────────────────────────────────────────────
cells = []

def md(src: str) -> None:
    src = src.strip()
    if src:
        cells.append({"cell_type": "markdown", "id": uuid4().hex[:8],
                      "metadata": {}, "source": src})

def code(src: str) -> None:
    src = src.strip()
    if src:
        cells.append({"cell_type": "code", "id": uuid4().hex[:8],
                      "metadata": {}, "source": src,
                      "outputs": [], "execution_count": None})

# ── split source into sections on # ══ banners ────────────────────────────────
BANNER_RE = re.compile(r"^# [═]{20,}")

sections: list[list[str]] = []
current: list[str] = []
for ln in lines:
    if BANNER_RE.match(ln):
        if current:
            sections.append(current)
        current = [ln]
    else:
        current.append(ln)
if current:
    sections.append(current)

# ── helper: extract banner label from a section block ────────────────────────
LABEL_RE = re.compile(r"^#\s+([A-Z][^\n]+)")

def section_label(sec: list[str]) -> str:
    for ln in sec[:5]:
        m = LABEL_RE.match(ln)
        if m:
            return m.group(1).strip()
    return ""

# ── TITLE cell ────────────────────────────────────────────────────────────────
file_doc = ""
in_doc = False
for ln in lines:
    stripped = ln.strip()
    if stripped.startswith('"""') and not in_doc:
        in_doc = True
        rest = stripped[3:]
        if rest.endswith('"""'):
            file_doc = rest[:-3].strip()
            break
        file_doc += rest + "\n"
        continue
    if in_doc:
        if '"""' in stripped:
            file_doc += stripped[:stripped.index('"""')]
            break
        file_doc += ln.rstrip() + "\n"

md(f"""# FOMC Vol-Spread Model (`fomc_spread_model.py`)

{file_doc.strip()}

---
**Five structural fixes (C1–C5)** for the Warsh failure:

| Fix | Description |
|---|---|
| **C1** | Target = GapSpread = Gap(2Y) − Gap(30Y) — fixes wrong-signed signal |
| **C2** | Regime-similarity sample weighting |
| **C3** | Mechanism-prior shrinkage (Bayesian ridge) |
| **C4** | Two separate models: GAP model (IV-gated) + FEATURE model (full corpus) |
| **C5** | Communication-architecture regime layer (REMOVE / ADD) |

> All outputs land in `spread_model_figs/`.  Run all cells top-to-bottom (clean kernel) to reproduce.
""")

# ── SETUP cell: matplotlib backend ───────────────────────────────────────────
code("""\
# Notebook display config — run before importing the module
import matplotlib
matplotlib.use("Agg")   # headless; figures are saved to spread_model_figs/
import warnings
warnings.filterwarnings("ignore")
print("Matplotlib backend: Agg (headless; figures saved to disk)")
""")

# ── emit each section as a cell pair (markdown header + code block) ───────────
IMPORT_SECTION = True   # first real code section becomes the imports cell

# Sections we turn into MARKDOWN-only cells (pure comment/docstring blocks)
COMMENT_ONLY_RE = re.compile(r"^\s*(#.*)?$")

for sec in sections:
    label = section_label(sec)
    src   = "\n".join(sec).strip()

    # Pure banner / comment-only sections → markdown
    non_comment = [ln for ln in sec if ln.strip() and not ln.strip().startswith("#")]
    if not non_comment:
        txt = "\n".join(
            ln[2:] if ln.startswith("# ") else ("" if ln.strip() == "#" else ln.strip("#").strip())
            for ln in sec
            if not BANNER_RE.match(ln)
        ).strip()
        if txt:
            md(f"---\n\n### {label or txt[:60]}")
        continue

    # All other sections: emit a markdown header + a code cell
    if label:
        md(f"---\n\n### {label}")

    # Suppress the __main__ guard — we call run() explicitly in the run cell below
    src = re.sub(r'if __name__\s*==\s*["\']__main__["\'].*', "# run() called in the cell below", src, flags=re.DOTALL)
    code(src)

# ── RUN cell ──────────────────────────────────────────────────────────────────
# The module already has   if __name__ == "__main__": preds = run()
# In the notebook we call run() explicitly so outputs/figures render.
md("""---

## Run the full model

`run()` executes the walk-forward in one shot:
- Loads VRP panel + NLP features
- Assigns C5 communication-architecture regime
- Builds GapSpread target (C1)
- Fits Bayesian ridge with mechanism prior (C3)
- Validates sign hit rates (C2) and g-posterior evolution
- Runs Warsh acceptance test
- Saves four visualisation figures to `spread_model_figs/`
- Exports `gap_forecasts_spread.parquet`
""")

code("""\
# Execute — all figures are saved to spread_model_figs/ and shown inline below
preds = run()
""")

# ── FIGURE display cells ──────────────────────────────────────────────────────
md("---\n\n### Fig 1 — GapSpread Walk-Forward OOS Predictions")
code("""\
from IPython.display import Image, display
import pathlib as _pl

_fdir = _pl.Path("spread_model_figs")
for _f in sorted(_fdir.glob("fig*.png")):
    print(f"\\n{'─'*60}")
    print(f"  {_f.name}")
    print('─'*60)
    display(Image(str(_f), width=1050))
""")

# ── INSPECT cell ──────────────────────────────────────────────────────────────
md("---\n\n### Inspect OOS predictions table")
code("""\
import pandas as pd
if 'preds' in dir() and not preds.empty:
    _disp_cols = ["meeting_date","predicted_gap_spread","std_gap_spread",
                  "z_spread","steepener_signal","gap_actual_spread",
                  "has_implied","regime_label","ess",
                  "g_posterior_mean","g_posterior_ci_lo","g_posterior_ci_hi"]
    _cols = [c for c in _disp_cols if c in preds.columns]

    def _style(df):
        def _cr(row):
            if row.get("steepener_signal") == "buy_front_sell_long":
                return ["background-color:#e8f0fb"] * len(row)
            if row.get("steepener_signal") == "sell_front_buy_long":
                return ["background-color:#fef0ee"] * len(row)
            return [""] * len(row)
        return (df.style
                  .apply(_cr, axis=1)
                  .format({
                      "predicted_gap_spread": "{:+.4f}",
                      "std_gap_spread":       "{:.4f}",
                      "z_spread":             "{:+.2f}",
                      "gap_actual_spread":    lambda v: f"{v:+.4f}" if pd.notna(v) else "—",
                      "ess":                  "{:.1f}",
                      "g_posterior_mean":     "{:+.3f}",
                      "g_posterior_ci_lo":    "{:+.3f}",
                      "g_posterior_ci_hi":    "{:+.3f}",
                  }, na_rep="—")
                  .set_caption("OOS walk-forward predictions (BLUE=steepener, RED=flattener)")
                  .set_properties(**{"font-size": "11px", "text-align": "left"})
                  .hide(axis="index"))

    from IPython.display import display as _d
    _d(_style(preds[_cols]))
    print(f"\\n{len(preds)} OOS rows  |  "
          f"IV-matched: {preds['has_implied'].sum()}  |  "
          f"Steepener signals: {(preds['steepener_signal']=='buy_front_sell_long').sum()}")
else:
    print("preds not yet computed — run the cell above first.")
""")

# ── CAVEATS cell ──────────────────────────────────────────────────────────────
md("""---

## Caveats

| # | Caveat |
|---|---|
| 1 | Statistical, not riskless. GapSpread requires matched IV; post-2020 meetings lack VXTYN → n_IV ≈ 84, n_REMOVE-regime ≈ 1-2. |
| 2 | n ≈ 1-2 for Warsh regime. Result leans on mechanism + prior; significance tests are not interpretable. Report CIs honestly. |
| 3 | GK/Parkinson understate jump vol → conservative RV bias. |
| 4 | Greenspan text informs features only, never the gap target. |
| 5 | Short-30Y-vol leg is short gamma: negative convexity, real tail risk. Size the short leg ≤ premium collected from the long-2Y leg. |
| 6 | To use live: replace VRP panel with post-2020 IV data (Bloomberg MOVE/TYVIX); retrain walk-forward. |
""")

# ── write notebook ────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.9.6"},
    },
    "cells": cells,
}

DEST.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
n_code = sum(1 for c in cells if c["cell_type"] == "code")
n_md   = sum(1 for c in cells if c["cell_type"] == "markdown")
print(f"Generated {DEST}  ({len(cells)} cells: {n_code} code, {n_md} markdown)")
