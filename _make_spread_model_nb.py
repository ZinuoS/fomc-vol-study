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

# ── TRADE SIGNAL SUMMARY section ─────────────────────────────────────────────
md("""\
---

## Section 5 — Trade Signal Summary

This section synthesises the GapSpread model output into an actionable trade
signal and maps it directly to the inputs consumed by `trade_ticket_pricer.ipynb`
and the hedge simulation.

| Panel | Content |
|---|---|
| **5A — Signal dashboard** | Latest meeting signal, z-score, regime, g-posterior |
| **5B — Signal history chart** | OOS GapSpread predictions + ±1σ bands + z-score strip |
| **5C — Model → pricer handoff** | Explicit mapping of every model output to its pricer role |
""")

code('''\
# 5A — Trade signal dashboard (latest meeting)

import pandas as pd
import numpy as np
from IPython.display import HTML, display as _d

latest = preds.iloc[-1]
mtg    = str(latest['meeting_date'])[:10]
signal = latest['steepener_signal']
z      = float(latest['z_spread'])
gsp    = float(latest['predicted_gap_spread'])
gsp_s  = float(latest['std_gap_spread'])
g_post = float(latest['g_posterior_mean'])
g_lo   = float(latest['g_posterior_ci_lo'])
g_hi   = float(latest['g_posterior_ci_hi'])
ess    = float(latest['ess'])
regime = latest['regime_label']
has_iv = bool(latest['has_implied'])

if signal == 'buy_front_sell_long':
    hdr_bg = '#1a3a6a'; sig_label = 'STEEPENER  ▲'
    trade_txt = 'LONG 2Y vol (ZT straddle / 2Y swaption)  |  SHORT 30Y vol (WN straddle / 30Y swaption)'
elif signal == 'sell_front_buy_long':
    hdr_bg = '#6a1a1a'; sig_label = 'FLATTENER  ▼'
    trade_txt = 'SHORT 2Y vol (ZT straddle / 2Y swaption)  |  LONG 30Y vol (WN straddle / 30Y swaption)'
else:
    hdr_bg = '#3a3a3a'; sig_label = 'NEUTRAL  —'
    trade_txt = 'No position (|z| < threshold)'

n_stars  = min(5, max(1, round(abs(z))))
stars    = '★' * n_stars + '☆' * (5 - n_stars)
pct_rank = (preds['predicted_gap_spread'] < gsp).mean() * 100
pct_pos  = (preds['predicted_gap_spread'] > 0).mean() * 100
iv_flag  = 'with matched IV' if has_iv else 'IV not matched (no VXTYN)'

def _row(label, val, note='', shade=False):
    bg = '#f0f4fb' if shade else 'white'
    return (f'<tr>'
            f'<td style="padding:7px 14px;font-size:11px;color:#555;border-right:1px solid #dde;background:{bg};width:36%">{label}</td>'
            f'<td style="padding:7px 14px;font-size:12px;font-weight:bold;background:{bg}">{val}</td>'
            f'<td style="padding:7px 14px;font-size:10.5px;color:#888;background:{bg}">{note}</td>'
            f'</tr>')

body = (
    _row('Predicted GapSpread',  f'{gsp:+.4f} ± {gsp_s:.4f} pp²', f'{pct_rank:.0f}th pctile of OOS history', shade=True)
  + _row('z-score',              f'{z:+.2f}σ  ({stars})',          'signal threshold: |z|≥1 → directional')
  + _row('g-posterior (Warsh)',  f'{g_post:+.4f}  [{g_lo:+.4f}, {g_hi:+.4f}]', f'ESS {ess:.0f}; >0 = REMOVE regime supports steepener', shade=True)
  + _row('Regime (C5)',          regime,                            'REMOVE direction: RegimeTransition=1')
  + _row('IV availability',      iv_flag,                           f'GapSpread>0 in {pct_pos:.0f}% of all OOS meetings', shade=True)
)

dash = f"""
<div style="font-family:'Helvetica Neue',Arial,sans-serif;max-width:760px;
     border:1px solid #aab;border-radius:4px;overflow:hidden;margin:12px 0;">
  <div style="background:{hdr_bg};color:#f0f4ff;padding:11px 18px;display:flex;
       justify-content:space-between;align-items:center;">
    <div>
      <span style="font-size:14px;font-weight:bold;letter-spacing:.4px;">
        FOMC VOL-SPREAD SIGNAL — Meeting {mtg}
      </span><br>
      <span style="font-size:10.5px;color:#aac;">{trade_txt}</span>
    </div>
    <div style="text-align:right;">
      <span style="font-size:21px;font-weight:bold;color:#fff;">{sig_label}</span><br>
      <span style="font-size:11px;color:#ccd;">Conviction: {stars} ({abs(z):.2f}σ)</span>
    </div>
  </div>
  <table style="width:100%;border-collapse:collapse;">
    <thead>
      <tr>
        <th style="background:#2c3e50;color:#cdd;padding:5px 14px;font-size:10px;text-align:left;width:36%;border-right:1px solid #445;">Metric</th>
        <th style="background:#2c3e50;color:#cdd;padding:5px 14px;font-size:10px;text-align:left;">Value</th>
        <th style="background:#2c3e50;color:#cdd;padding:5px 14px;font-size:10px;text-align:left;">Interpretation</th>
      </tr>
    </thead>
    <tbody>{body}</tbody>
  </table>
  <div style="background:#f7f9fc;border-top:1px solid #ccd;padding:6px 18px;font-size:9.5px;color:#888;">
    GapSpread > 0 → front vol underpriced vs long-end → STEEPENER (long ZT/2Y, short WN/30Y).
    This signal is ONE input; validate against live market vol (Group A), MC model (Group B), and risk limits.
  </div>
</div>
"""
_d(HTML(dash))
''')

code("""\
# 5B — Signal history chart: predicted GapSpread + z-score strip

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np, pandas as pd

_C_STEEP = '#2c5fa8'
_C_FLAT  = '#a82c2c'
_C_NEUT  = '#888888'

cols = ['meeting_date','predicted_gap_spread','std_gap_spread',
        'z_spread','steepener_signal','gap_actual_spread','has_implied']
df = preds[cols].copy()
df['dt'] = pd.to_datetime(df['meeting_date'])
df = df.sort_values('dt').reset_index(drop=True)

x = np.arange(len(df))
clr = [_C_STEEP if s=='buy_front_sell_long' else (_C_FLAT if s=='sell_front_buy_long' else _C_NEUT)
       for s in df['steepener_signal']]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7),
                                gridspec_kw={'height_ratios': [3, 1.2]},
                                sharex=True)
fig.patch.set_facecolor('#fafbfc')

# ── TOP: GapSpread bars + ±1σ + actual ────────────────────────────────────────
ax1.bar(x, df['predicted_gap_spread'], color=clr, alpha=0.72, width=0.72,
        zorder=3, label='Predicted GapSpread')
ax1.fill_between(x,
                 df['predicted_gap_spread'] - df['std_gap_spread'],
                 df['predicted_gap_spread'] + df['std_gap_spread'],
                 alpha=0.18, color='#444', zorder=2, label='±1σ band')
mask_iv = df['has_implied'].astype(bool)
ax1.scatter(x[mask_iv], df.loc[mask_iv, 'gap_actual_spread'],
            color='#1a1a1a', s=22, zorder=5, label='Actual (IV-matched)', marker='o')
ax1.axhline(0, color='#222', lw=0.9, ls='--', zorder=4)

# highlight latest
last_i = len(df) - 1
ax1.axvline(last_i, color='#e07b00', lw=1.8, ls=':', zorder=6, label='Latest signal')
ax1.annotate(
    f"{df.iloc[-1]['steepener_signal'].replace('buy_front_sell_long','STEEPENER').replace('sell_front_buy_long','FLATTENER').replace('flat','NEUTRAL')}\\n{str(df.iloc[-1]['dt'])[:7]}",
    xy=(last_i, df.iloc[-1]['predicted_gap_spread']),
    xytext=(last_i - 12, df.iloc[-1]['predicted_gap_spread'] + 0.012),
    fontsize=8.5, color='#e07b00', fontweight='bold',
    arrowprops=dict(arrowstyle='->', color='#e07b00', lw=1.1))

ax1.set_ylabel('GapSpread (pp²)', fontsize=10)
ax1.set_title('OOS GapSpread Predictions — Signal History\\n'
              '(BLUE = steepener, RED = flattener, GREY = neutral)', fontsize=11, fontweight='bold', pad=8)
ax1.set_facecolor('#fafbfc')

legends = [mpatches.Patch(color=_C_STEEP, alpha=0.75, label='Steepener (→ long ZT/2Y, short WN/30Y)'),
           mpatches.Patch(color=_C_FLAT,  alpha=0.75, label='Flattener (→ reverse)'),
           mpatches.Patch(color=_C_NEUT,  alpha=0.75, label='Neutral')]
from matplotlib.lines import Line2D
legends += [Line2D([0],[0], marker='o', color='w', markerfacecolor='#1a1a1a', markersize=7, label='Actual (IV-matched)'),
            Line2D([0],[0], color='#e07b00', ls=':', lw=1.8, label='Latest')]
ax1.legend(handles=legends, fontsize=8.5, ncol=3, loc='upper left')

# ── BOTTOM: z-score strip ──────────────────────────────────────────────────────
ax2.bar(x, df['z_spread'], color=clr, alpha=0.65, width=0.72, zorder=3)
ax2.axhline(0,    color='#333', lw=0.6, zorder=4)
ax2.axhline( 1.0, color=_C_STEEP, lw=0.8, ls='--', alpha=0.6, zorder=4)
ax2.axhline(-1.0, color=_C_FLAT,  lw=0.8, ls='--', alpha=0.6, zorder=4)
ax2.axvline(last_i, color='#e07b00', lw=1.8, ls=':', zorder=6)
ax2.set_ylabel('z-score', fontsize=9.5)
ax2.set_title('Signal Conviction (z-score  |  dashed = ±1σ threshold)', fontsize=9.5)
ax2.set_facecolor('#fafbfc')

# x-axis: year labels
year_ticks = [i for i, d in enumerate(df['dt']) if d.month == 1 or i == 0]
ax2.set_xticks(year_ticks)
ax2.set_xticklabels([str(df['dt'].iloc[i].year) for i in year_ticks], fontsize=9)
ax2.set_xlim(-1, len(df))

plt.tight_layout(rect=[0, 0, 1, 1])
_out = 'spread_model_figs/fig5_signal_summary.png'
plt.savefig(_out, dpi=150, bbox_inches='tight', facecolor='#fafbfc')
plt.close()
print(f'[VIS] Fig 5 saved → {_out}')

from IPython.display import Image, display as _dimg
_dimg(Image(_out, width=1050))
""")

code("""\
# 5C — Model → Pricer handoff table

import pandas as pd
from IPython.display import display as _d

latest = preds.iloc[-1]

_rows = [
    # (Model output, Value, Pricer destination, Role / how to use)
    ('steepener_signal',
     latest['steepener_signal'].replace('buy_front_sell_long','buy_front_sell_long\\n→ STEEPENER'),
     'trade_ticket_pricer.ipynb\\nLEG direction',
     'Sets LEG 1 = BUY 2Y vol, LEG 2 = SELL 30Y vol when buy_front_sell_long'),

    ('predicted_gap_spread',
     f'{float(latest["predicted_gap_spread"]):+.4f} pp²',
     'No direct pricer input',
     'Qualitative conviction; the trade is sized by vega-neutral rule, not GapSpread level'),

    ('z_spread',
     f'{float(latest["z_spread"]):+.2f}σ',
     'Optional: size scalar',
     'Can scale N_30Y anchor up/down around the desk minimum; NOT a required input'),

    ('signal_mult_2y  /  signal_mult_30y',
     f'{float(latest["signal_mult_2y"]):.2f}x  /  {float(latest["signal_mult_30y"]):.2f}x',
     'Optional: anchor multipliers',
     'Suggested notional multiplier relative to desk-minimum if using signal-scaled sizing'),

    ('g_posterior_mean',
     f'{float(latest["g_posterior_mean"]):+.4f}  [{float(latest["g_posterior_ci_lo"]):+.4f}, {float(latest["g_posterior_ci_hi"]):+.4f}]',
     'Context for regime narrative',
     '>0 means Warsh REMOVE mechanism active; CI width = model uncertainty (wide → lean on prior)'),

    ('ess',
     f'{float(latest["ess"]):.1f}',
     'Confidence weight on g_posterior',
     'ESS ≈ 30-37 in warsh_era; low ESS → g estimate is prior-dominated, not data-driven'),

    ('regime_label',
     latest['regime_label'],
     'C5 regime for narrative',
     'warsh_era → REMOVE direction; RegimeTransition=1 activates g = f1 × RegimeTransition term'),

    ('— (separate pipeline) —',
     'fomc_nlp_regime_model.py\\nwalk_forward_full()',
     'MODEL dict (Group B)\\nEVENT_SD_2Y / 30Y_BPS_DAY',
     'MC jump SD comes from the LEVEL model, NOT from GapSpread model. Never conflate.'),

    ('— MARKET dict (Group A) —',
     'F, K, implied vol, DV01',
     'MARKET dict\\n(pricer CELL 3)',
     'Premium set by live market vol (OMON / BGN); independent of GapSpread prediction'),
]

df_ho = pd.DataFrame(_rows, columns=['Model Output', 'Latest Value', 'Pricer Destination', 'Role / How to Use'])

def _style_ho(df):
    def _cr(row):
        if '(separate pipeline)' in str(row['Model Output']) or 'MARKET dict' in str(row['Model Output']):
            return ['background:#fff8e8'] * len(row)
        return ['background:#f0f4fb' if row.name % 2 == 0 else 'background:white'] * len(row)
    return (df.style
              .apply(_cr, axis=1)
              .set_caption('Model → Pricer Handoff  |  YELLOW rows = separate pipelines, do not conflate')
              .set_properties(**{'font-size': '11px', 'text-align': 'left', 'white-space': 'pre-wrap'})
              .hide(axis='index'))

_d(_style_ho(df_ho))
print('\\nKey invariant: GapSpread model sets DIRECTION only.')
print('Premium is always from MARKET implied vol (Group A).')
print('MC jump is always from LEVEL model EVENT_SD (Group B).')
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
