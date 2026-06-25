# FOMC Volatility Study

End-to-end research pipeline that forecasts short-term realized volatility
in Treasury futures around FOMC meetings using NLP-derived features,
dual-mandate economic regime labels, and VRP calibration.

---

## Repository Layers

### Layer 0 — Raw Data Collection
*Fetches public data; writes `.parquet` files to disk.*

| File | Purpose | Outputs |
|------|---------|---------|
| `fomc_public_pipeline.py` | FOMC statements, FRED macro series, GK realized vol (ZT/ZB futures) | `fomc_statements.parquet`, `vrp_cache/vrp_panel.parquet` |
| `fomc_vrp_pipeline.py` | Variance risk premium (IV − RV) and gap-var calculations | `vrp_cache/vrp_panel.parquet` (extended) |
| `fomc_spread_model.py` | Yield-spread and Eurodollar curve data | `gap_forecasts_spread.parquet` |
| `fomc_etf_iv_pipeline.py` | ETF-implied-vol estimates (TLT straddles) | `etf_gap_curve.parquet` |
| `rebuild_presser_corpus.py` | Downloads all 86 FOMC press-conference PDFs (full Q&A, ~8,500 tokens/meeting) and rebuilds corpus | `fomc_cache/pressers_pdf/*.pdf`, `fomc_corpus_expanded.parquet` |

---

### Layer 1 — NLP Processing
*Tokenises FOMC documents, builds lexicon scores, joins with regime labels.*

| Notebook | Source | What it does |
|----------|--------|-------------|
| **`fomc_nlp_regime_model.ipynb`** | `fomc_nlp_regime_model_nb.py` | Builds 219-document corpus (133 statements + 86 full Q&A pressers); computes `uncertainty_density`, `guidance_density`, `disagree_density` per 1k tokens; dual-mandate regime labels; NLP×regime RidgeCV walk-forward; 6 publication figures |

Key outputs: `fomc_corpus_expanded.parquet`, `fomc_features.parquet`,
`fomc_dual_mandate_regime.parquet`, `fomc_nlp_regime_forecasts.parquet`

NLP features (per 1,000 tokens, length-normalised):
- `uncertainty_density` — hedging & risk language
- `guidance_density` — forward-guidance language
- `disagree_density` — dissent signals
- `novelty_prev` — Jaccard distance from prior meeting text

---

### Layer 2 — Regime Diagnosis & VRP Calibration
*OOS walk-forward tests; always produces BOTH forecasting modes.*

| Notebook | Source | What it does |
|----------|--------|-------------|
| **`regime_diagnosis.ipynb`** | `regime_diagnosis_nb.py` | 5-stage diagnosis: parameter-count audit, 4 pre-registered re-specs, model lock, full-history 2010→present, Warsh n=1 forward test |
| **`regime_vrp_calibration.ipynb`** | `regime_vrp_calibration_nb.py` | VRP distribution by regime; dual-mode forecasts; IV/RV gap analysis (Figs V1–V7) |

**Two invariant forecasting modes (both always produced):**

| Mode | Analogy | Mechanism |
|------|---------|-----------|
| NLP-only | Backward-looking · IV analog | Text-vol mapping pooled across regimes; anchors to QE-era positive VRP |
| NLP×regime | Forward-looking · regime-conditional | PC1 × inflation-gap; adjusts upward in overheating, downward in slack |

**Key empirical findings:**
- VRP significantly positive in all IV-observable regimes (slack +1.57pp p<0.001, easing +1.11pp p=0.033, at-target +0.47pp p=0.017)
- No IV data for overheating/supply-shock (TYVIX discontinued May 2020)
- NLP×regime parsimony spec (p/n=0.16) is WEAK on Powell OOS — CI crosses zero for both 2Y and 30Y
- 2Y shows directional improvement (R² 0.127→0.247, SHR 69%→78.6%); 30Y degrades
- Warsh 2026-06-17: 2Y forecast error −1.1%; 30Y over-predicted +200% (QE-era extrapolation)

---

### Layer 3 — Strategy Simulation
*Back-tests straddle and delta-hedge strategies using the forecasts above.*

| Notebook | Source | What it does |
|----------|--------|-------------|
| **`fomc_straddle_notebook.ipynb`** | `fomc_straddle_notebook.py` | Monte-Carlo straddle P&L simulation; vol-regime conditional entry |
| **`fomc_backtest.ipynb`** | `fomc_backtest.py` | Historical walk-forward back-test of FOMC vol strategies |
| **`fomc_spread_model.ipynb`** | `fomc_spread_nb.py` | 2Y/30Y spread forecasting; GapSpread target model |
| **`fomc_etf_iv.ipynb`** | `fomc_etf_iv_nb.py` | ETF implied-vol based signals |

---

### Layer 4 — Execution
*Trade sizing, delta-hedging, and ticket generation.*

| Notebook | Source | What it does |
|----------|--------|-------------|
| **`fomc_delta_hedge.ipynb`** | `fomc_delta_hedge_nb.py` | ZT/ZB straddle delta-hedge mechanics; Greeks and hedge ratios |
| **`trade_ticket.ipynb`** | `trade_ticket_nb.py` | Auto-generated trade ticket with entry/exit levels and risk sizing |

---

## Build System

Notebooks are generated from `.py` source files with `# %% [markdown]` / `# %%` cell markers:

```bash
python3 build_{name}_notebook.py          # generate .ipynb
python3 -m nbconvert --to notebook --execute --inplace {name}.ipynb  # run it
```

---

## Data Files

| File | Rows | Description |
|------|------|-------------|
| `fomc_statements.parquet` | 133 | FOMC statement text + metadata |
| `fomc_corpus_expanded.parquet` | 219 | All NLP documents (statements + full Q&A pressers) |
| `fomc_features.parquet` | 133 | Per-meeting NLP features (composite presser + statement) |
| `fomc_dual_mandate_regime.parquet` | 133 | FRED dual-mandate regime labels per meeting |
| `fomc_nlp_regime_forecasts.parquet` | 51 | Walk-forward OOS forecasts (Powell era) |
| `vrp_cache/vrp_panel.parquet` | 798 | GK realized vol, IV, VRP by tenor and meeting |
| `fomc_cache/pressers_pdf/` | 86 PDFs | Full press-conference transcripts (~8,500 tokens each) |

---

## Corpus Update — Full Q&A Pressers (2025-06)

Press conferences rebuilt from official Fed PDFs (full Q&A transcripts):

| Metric | Before | After |
|--------|--------|-------|
| Mean tokens / presser | 79 (HTML boilerplate) | **8,568** (full transcript) |
| Token range | 54–105 | 6,581–11,611 |
| Mean chair turns | ~1 (fake) | **26.8 ± 5.9** |

Run `python3 rebuild_presser_corpus.py` to refresh if Fed PDFs are updated.
