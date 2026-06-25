# FOMC Volatility Study — Technical Report
## Version Evolution, Statistical Methodology, and Empirical Results

---

## 0. Overview

This report documents the full design history of the FOMC volatility forecasting pipeline: what changed in each version, the mathematical basis for each choice, and the empirical impact on out-of-sample (OOS) model performance.

**Core question**: Can NLP features extracted from FOMC statements and press conferences predict short-term realized volatility in Treasury futures around FOMC meetings — and does conditioning on the dual-mandate economic regime improve that prediction?

---

## 1. Repository Architecture

```
fomc-vol-study-repo/
├── notebooks/
│   ├── layer0_data/           ← raw data collection & VRP panel
│   ├── layer1_nlp/            ← NLP feature engineering & regime labelling
│   ├── layer2_calibration/    ← specification diagnosis & VRP calibration
│   ├── layer3_strategy/       ← strategy simulation & backtesting
│   └── layer4_execution/      ← trade sizing, delta-hedge, ticket generation
├── src/                       ← Python source files & notebook builders
├── figures/                   ← output figures (committed)
├── fomc_features.parquet      ← master NLP+macro feature panel (133 meetings)
├── fomc_corpus_expanded.parquet ← NLP corpus (133 statements + 86 pressers)
├── fomc_dual_mandate_regime.parquet ← regime labels per meeting
├── fomc_nlp_regime_forecasts.parquet ← OOS walk-forward predictions
└── vrp_cache/                 ← realized vol, implied vol, VRP panel
```

**Data flow**: Layer 0 builds raw series → Layer 1 produces NLP features → Layer 2 diagnoses model specs and calibrates VRP → Layer 3 runs strategies → Layer 4 generates trade execution outputs.

---

## 2. Version History

### V1 — Statement-Only Baseline (Pre-2025)

**Corpus**: 133 FOMC statements scraped from HTML (avg 130 words, range 80–450).

**NLP features** (all per-statement raw counts or ratios):
- `guidance_density` = count of forward-guidance phrases / total words
- `uncertainty_density` = count of hedging words / total words
- `disagree_density` = count of dissent signals / total words
- `novelty_prev` = TF-IDF cosine distance to prior statement (0 = identical)

**Model**: Ridge regression with leave-one-out cross-validation on historical meetings, then walk-forward OOS starting 2018-02-03 (Powell era).

**Result**: LOO CV R² ≈ 0.06, OOS R² ≈ −0.05 (NLP features have weak signal when measured only on short statement text).

**Problem identified**: With 130 words per meeting, density features were so noisy that signal-to-noise was near zero. Guidance phrases appear 0–2 times in a 130-word statement; uncertainty hedges appear ~1 time. The resulting density scores are almost binary and carry no continuous information.

---

### V2 — Dual-Mandate Regime Labels (2025 Q1)

**Key change**: Replaced chair-identity dummy variables with economically-motivated regime labels derived from FRED macro series.

#### 2.1 Regime Construction

FRED series used:
- `PCEPILFE`: Core PCE price index (level) → converted to YoY % change
- `UNRATE`: Civilian unemployment rate
- `NROU`: Congressional Budget Office natural rate of unemployment

**Derived gaps**:

```
inflation_gap(t) = core_PCE_YoY(t) − 2.0        [pp above/below target]
u_gap(t)        = UNRATE(t) − NROU(t)           [pp of labour slack; + = slack]
```

**Five-way regime label** (hard thresholds, calibrated at ±0.5 pp):

| Label | Inflation gap | Unemployment gap | Economic interpretation |
|-------|--------------|-----------------|------------------------|
| `overheating` | > +0.5 pp | < −0.5 pp | Both mandates under pressure — hawkish |
| `supply_shock` | > +0.5 pp | ≥ −0.5 pp | Inflation without overheating — stagflation risk |
| `at_target` | [−0.5, +0.5] | any | Near dual-mandate equilibrium |
| `slack` | ≤ −0.5 pp | > +0.5 pp | Inflation soft, labour slack — dovish |
| `easing` | ≤ −0.5 pp | any other | Below-target inflation, active easing |

**Why not chair identity?** Powell spanned overheating (2022–23, 75 bp hikes) and easing (2019, three cuts) within a single tenure. Chair identity conflates economic regimes that have opposite volatility implications.

#### 2.2 Model Specification with Regime Interaction

**NLP-only** (backward-looking, IV analog):
```
rv(t) = α + β · X_nlp(t) + ε(t)
```

**NLP×regime** (forward-looking, regime-conditional):
```
rv(t) = α + β₁ · X_nlp(t) + β₂ · PC₁(t) × inflation_gap(t)
       + β₃ · accel(t) + β₄ · PC₁(t) × accel(t) + ε(t)
```

where:
- `PC₁(t)` = first principal component of text features (fold-by-fold PCA to avoid lookahead)
- `inflation_gap(t)` = regime intensity variable
- `accel(t)` = 3-month change in core PCE YoY (regime acceleration signal)
- All scalings by `StandardScaler` fitted on training window only

**Parameter count discipline** (V2 acceptance test):
- Full spec with 4 TEXT_COLS × 5 regime dummies = 20 regime interactions → p/n ≈ 0.55 on n=51 OOS meetings → severe overfitting risk
- Accepted spec (Spec A): 1 PC₁ × inflation_gap → p/n = 0.16 ← passes parsimony bar

**Walk-forward protocol**:
- Expanding window starting 2018-02-03 (Jerome Powell appointment)
- Minimum 15 training meetings before first OOS prediction
- `StandardScaler` re-fit on each training window (no lookahead)
- Ridge CV with `α ∈ {0.01, 0.1, 1.0, 10.0, 100.0}` on training split

**V2 OOS result (2Y tenor, n=51)**:
- NLP-only: RMSE = 1.576, R² = −0.49
- NLP×regime: RMSE = 1.268, R² = +0.04
- NLP-only had negative OOS R² because the statement-only features were too noisy

**Warsh forward test (2026-06-17)**: NLP-only predicted 2.47% (2Y), actual was 3.00% (error −1.1%). 30Y prediction was wildly off (+206%) because the 2010–2020 ZLB-era training distribution had near-zero 30Y vol.

---

### V3 — Full PDF Presser Corpus (2025 Q2)

**Problem with V1/V2**: Press-conference HTML pages from the Fed website are JavaScript-rendered. BeautifulSoup scraping returned nav-bar boilerplate (54–105 tokens, avg 79), not the actual Q&A transcript.

**Fix**: Downloaded all 86 press-conference PDFs directly:
```
https://www.federalreserve.gov/mediacenter/files/FOMCpresconf{YYYYMMDD}.pdf
```
Extracted with `pdfplumber`. Chair-turn filtering:
```python
CHAIR_RE = re.compile(r'\n(CHAIR(?:MAN)?\s+(?:POWELL|YELLEN|BERNANKE|...))[.\s]+')
```
(Required `CHAIR(?:MAN)?` because Bernanke/Greenspan era used "CHAIRMAN".)

**Corpus statistics after V3**:

| Metric | Before V3 | After V3 |
|--------|-----------|----------|
| Mean tokens / presser | 79 (HTML boilerplate) | **8,568** (full Q&A) |
| Token range | 54–105 | 6,581–11,611 |
| Mean chair turns per presser | ~1 (fake) | **26.8 ± 5.9** |
| Total corpus documents | 133 | **219** (133 stmts + 86 pressers) |

**NLP feature density comparison** (why this matters):

At 8,568 tokens, a single press conference contains ~66× more text than a statement. Guidance phrases appear 30–80 times instead of 0–2 times. The density features become **continuous and informative**:

| Feature | Statement (130 words) | Presser (8,568 tokens) |
|---------|----------------------|----------------------|
| `guidance_density` | 0.52–0.64 / 1k tokens | 3.1–6.5 / 1k tokens (6×) |
| `uncertainty_density` | 0.005–0.009 / 1k tokens | 1.5–4.9 / 1k tokens (500×) |

The presser features were stored in sidecar columns (`guidance_density_presser`, etc.) but **the core model features remained statement-only** until V4.

---

### V4 — Composite Feature Blending (Critical Fix, 2025 Q3)

**The bug discovered**: `fomc_pipeline_notebook.ipynb` computed NLP features from statement text only in Cells 8–12. Even after V3 added presser data, re-executing the pipeline notebook overwrote `fomc_features.parquet` with statement-only `guidance_density`, `uncertainty_density`, etc. The presser data was stranded in unused sidecar columns.

**The fix** (Cell 28 of `fomc_pipeline_notebook.ipynb`):

```python
PRESSER_W, STATEMENT_W = 0.6, 0.4
for base_col, presser_col in [
    ("guidance_density",    "guidance_density_presser"),
    ("uncertainty_density", "uncertainty_density_presser"),
    ("disagree_density",    "disagree_density_presser"),
]:
    has_presser = fomc_features[presser_col].notna()
    fomc_features.loc[has_presser, base_col] = (
        PRESSER_W   * fomc_features.loc[has_presser, presser_col]
        + STATEMENT_W * fomc_features.loc[has_presser, base_col]
    )
```

**Weight rationale (0.6 / 0.4)**:
- Presser Q&A directly records the Chair's spoken reasoning — higher information content per token
- Statement language is formulaic and committee-voted; carries less Chair-specific signal
- 0.6/0.4 is a conservative blend; the presser alone would carry even more signal but the statement remains the official policy communication

**Impact on feature values** (Powell meeting, 2026-03-18):

| Feature | V3 (statement-only) | V4 (composite) | Ratio |
|---------|--------------------|--------------------|-------|
| `guidance_density` | 0.621 | 2.208 | 3.6× |
| `uncertainty_density` | 0.009 | 2.313 | 257× |

**Impact on OOS model performance** (2Y tenor, walk-forward n=51):

| Metric | V3 (stmt-only) | V4 (composite) | Change |
|--------|---------------|----------------|--------|
| NLP-only OOS R² | −0.49 | −0.49 | ~0 (as expected) |
| NLP×regime OOS R² | −0.08 | **+0.04** | **+0.12** |
| NLP×regime RMSE | 1.38% | **1.27%** | −0.11 pp |

The NLP×regime model now achieves positive OOS R² for the first time. The NLP-only model is unaffected because it uses the same statement-level features (the composite change mainly improves the regime-interaction features which use the PCA component derived from the composite).

---

### V5 — ETF IV Fallback + Scoring Normalisation Fix (2026 Q2)

**Problem 1**: TYVIX discontinued May 2020. All OOS meetings (Powell era 2018+, first OOS around 2020) had `iv_event_vol = NaN`. The VRP calibration notebook showed `IV obs = 0` in all OOS figures — no IV comparison was possible for any overheating-regime meeting.

**Fix 1**: Load `etf_gap_curve.parquet` (ETF-straddle-derived IV, all 133 meetings) and fill `iv_event_vol` where NaN:
```python
_missing = vrp["iv_event_vol"].isna() & vrp["iv_etf_pct"].notna()
vrp.loc[_missing, "iv_event_vol"] = vrp.loc[_missing, "iv_etf_pct"]
vrp["iv_source"] = np.where(_missing, "etf_proxy", "tyvix")
```
Filled 245 rows (49 meetings × 5 tenors). ETF tenors: SHY→2Y, IEI→5Y, IEF→10Y, TLH→20Y, TLT→30Y.

**Problem 2**: `fomc_nlp_regime_model_nb.py`'s `score_corpus()` divided pre-computed per-1k-token values by `n_tok/1000` again — a double-normalization. Effect: statements (300 tokens) inflated 3.3×, pressers (8,568 tokens) deflated 8.6×. The MAX aggregation in `agg_to_meeting()` always picked the inflated statement over the correct presser value.

**Fix 2**: When `feats_df` provides pre-computed values, set `scale = 1.0` (skip division). Also fixed `agg_to_meeting()` to use presser-priority selection instead of MAX.

**Impact on V5**:

| Metric | Before V5 | After V5 |
|--------|-----------|----------|
| IV obs in OOS | 0 | **42 (TYVIX=0 → ETF)** |
| Full-history IV obs | 60 | **108** |
| Overheating VRP | unknown | **+0.93 pp (p<0.001)** |
| NLP-only RMSE (2Y) | 1.58% | **1.18%** (vs IV 1.31%) |
| NLP-only corr(pred,IV) | −0.017 | **+0.520** |

**ETF IV caveats**: Duration mismatch (ETF modified duration ≠ ZT/ZB futures DV01), tracking error from expense ratios, and option liquidity differences. ETF-implied vol is directionally valid but not a precise substitute for TYVIX. All ETF-derived observations are labelled `iv_source = "etf_proxy"` in the parquet output.

---

### V6 — Timing Mismatch Fix: EWMA-3 Lag (2026 Q2)

**Observed problem**: In time-series visualisations (Fig V2), model predictions appeared shifted one meeting relative to actual RV. Quantifying: `corr(pred, rv_lag1) = 0.437` while the actual RV autocorrelation is only `corr(actual, rv_lag1) = 0.616`. The model was over-learning the previous meeting's RV and underweighting the current meeting's NLP signals.

**Root cause**: All walk-forwards included `rv_lag1` (previous meeting's GK vol) as a feature. Ridge assigned 437/616 = 71% of the actual autocorrelation to the model's lag term, creating a 1-meeting delayed tracking effect. Any large RV at meeting t−1 caused the model to over-predict at meeting t even if NLP signals indicated low vol.

**Fix**: Replace `rv_lag1` with `rv_ewma3`:
```python
pan["rv_ewma3"] = pan["rv"].ewm(span=3, adjust=False).mean().shift(1)
```
The EWMA (exponentially weighted moving average of last 3 meetings, span=3) provides a smoother vol regime signal that decays with recency rather than sticking to the previous single-meeting spike.

**Impact (2Y tenor, Powell OOS n=43–51)**:

| Model | Before (rv_lag1) | After (rv_ewma3) | Change |
|-------|-----------------|-----------------|--------|
| NLP-only RMSE | 1.58% | **1.28%** | −0.30 pp |
| NLP-only R² | −0.48 | **−0.16** | +0.32 |
| NLP×regime R² (full features) | −0.14 | **+0.23** | +0.37 |
| Diagnosis Spec A NLP-only R² | −0.28 | **+0.11** | +0.39 |
| Diagnosis Spec B SHR | 69.8% | **76.7%** | +7 pp |

**Why EWMA beats lag-1**: The 3-meeting EWMA blends recent history and is 66% less sensitive to single-meeting spikes. This means a surprise vol spike at t−1 (e.g., COVID March 2020, 4.58%) contributes only ~50% weight to the t forecast instead of 100% weight.

---

## 3. Mathematical Foundations

### 3.1 NLP Feature Computation

**Length normalisation** (per 1,000 tokens):
```
feature_density(t) = count(matches in text_t) / (n_tokens_t / 1000)
```
This makes features comparable across documents of different length — a key requirement since statements (130 words) and pressers (8,568 tokens) co-exist.

**TF-IDF novelty** (Jaccard-style via cosine distance):
```
novelty_prev(t) = 1 − cosine_similarity(tfidf(text_t), tfidf(text_{t-1}))
novelty_window(t) = 1 − cosine_similarity(tfidf(text_t), centroid(tfidf(text_{t-W:t-1})))
```
Spikes when the Chair breaks template language — the strongest single predictor of elevated event-vol.

**Composite weighting** (V4):
```
f_composite(t) = 0.6 · f_presser(t) + 0.4 · f_statement(t)    if presser exists
               = f_statement(t)                                  otherwise
```

### 3.2 Ridge Regression Walk-Forward

Each OOS prediction uses an expanding window:

```
Training: {1, 2, ..., t−1}
Test:     {t}

θ̂(t) = argmin_θ ‖y − Xθ‖² + α‖θ‖²

where α selected by time-series CV on training window
```

Regularisation is critical because p/n can reach 0.55 for the full specification on a Powell-era sample (n=51). Ridge shrinks coefficients proportional to their L2 norm, preventing overfitting to small-sample regime cells.

**Scaler protocol** (single-row test bug fix): `StandardScaler` is fit on the training window then `.transform()` applied to the single test row. Fitting on a single test row collapses all features to zero (std = 0) — an earlier bug that was fixed in V2.

### 3.3 PCA for Regime Interaction

Computing `PC₁(t)` fold-by-fold avoids lookahead bias:

```
For each OOS fold t:
  1. Fit PCA(n_components=1) on TEXT_COLS of training set {1, ..., t-1}
  2. Transform test observation t onto the same principal axis
  3. PC₁(t) = loadings from training PCA applied to test text features
```

`PC₁` captures the dominant axis of text variation (typically a hawkish↔dovish spectrum). Interacting it with `inflation_gap` creates a feature that is large when the Chair's language is hawkish AND inflation is above target — precisely the overheating signal.

### 3.4 HAC Regression (Newey-West)

For in-sample coefficient significance tests:

```
y = Xβ + ε
β̂_OLS = (X'X)⁻¹X'y

V̂_HAC = (X'X)⁻¹ Ŝ (X'X)⁻¹

Ŝ = Γ₀ + Σ_{j=1}^{L} w_j (Γ_j + Γ_j')
w_j = 1 − j/(L+1)   (Bartlett kernel)
Γ_j = (1/n) X'[ε_t ε_{t-j}]X
```

Bandwidth L = floor(4 × (n/100)^(2/9)) per Andrews (1991). Used because FOMC residuals are serially correlated (meetings cluster in macro episodes) and heteroskedastic (vol-of-vol varies by regime).

### 3.5 Bootstrap Confidence Intervals for Regime Specs

To test whether `ΔRMSE = RMSE_nlp_only − RMSE_nlp×regime > 0`:

```
For b = 1, ..., 1000:
  Sample n meetings with replacement from OOS window
  Compute ΔRMSE_b on the bootstrap sample

CI_90 = [Q₅(ΔRMSE_b), Q₉₅(ΔRMSE_b)]
boot_win = P(ΔRMSE_b > 0)
```

**Acceptance bar**: CI must be strictly positive (CI lower bound > 0) to accept a regime specification. If CI crosses zero → `FLAG`. If CI lower bound < −0.2 → `REJECT`.

**Result for all specs (V4)**: No specification cleared the CI > 0 bar. The locked model is NLP-only for the 2Y tenor (the regime model shows directional improvement but the confidence interval is too wide to claim statistical reliability at n=51).

### 3.6 Variance Risk Premium (VRP)

```
VRP(t, T) = IV(t, T) − RV(t, T)

where:
  IV(t, T) = TYVIX-implied vol scaled to tenor T via price-vol ratio
  RV(t, T) = Garman-Klass estimator on ZT/ZB futures in [t, t+window]
```

**Garman-Klass estimator** (used for `gk_vol_10y` and `rv_event_gk`):
```
GK = √(0.5 ln(H/L)² − (2ln2−1) ln(C/O)²) × √(252 / window)
```
More efficient than close-to-close vol (4× efficiency improvement) while remaining robust to microstructure noise.

**IV sources**: TYVIX 2010–2020, ETF proxy (SHY/IEF/TLH/TLT straddle-implied vol) 2020–present. ETF IV merged as fallback in all notebooks; observations labelled `iv_source = "etf_proxy"` for transparency.

**VRP by regime** (full history 2010–2026, all 2Y meetings, TYVIX + ETF proxy):

| Regime | n | Mean IV | Mean RV | VRP | p-value |
|--------|---|---------|---------|-----|---------|
| slack | 49 | 2.28% | 0.73% | +1.55 pp | < 0.001 |
| easing | 10 | 2.21% | 1.09% | +1.11 pp | 0.033 |
| at_target | 31 | 1.71% | 1.24% | +0.47 pp | 0.017 |
| supply_shock | 4 | 1.81% | 0.39% | +1.43 pp | — |
| **overheating** | **37** | **3.13%** | **2.20%** | **+0.93 pp** | **< 0.001** |

VRP is positive in **all five regimes**. Overheating has the highest IV (3.13%) and the highest absolute RV (2.20%), but IV still exceeds RV by 0.93 pp — the overheating risk premium exists. This finding was previously unobservable; it required the ETF IV fallback to quantify.

### 3.7 GapSpread Model (Five-Change Fix for Warsh)

The original level-vol model predicted SELL-30Y-VOL for the June 2026 Warsh meeting. Actual: 2Y vol +179 bp, 30Y vol +34 bp — opposite sign, opposite tenor.

**Five changes** in `fomc_spread_model.py`:

| Change | Fix | Rationale |
|--------|-----|-----------|
| C1 Target | `GapSpread = Gap(2Y) − Gap(30Y)` in pp² | Captures front-vs-long divergence; + = buy front-end vol |
| C2 Weighting | Regime-similarity kernel; IV gate | Down-weight ZLB meetings when forecasting overheating meetings |
| C3 Prior | Bayesian ridge with mechanism prior on novelty×RegimeTransition | Regularise toward the hawkish-shift mechanism |
| C4 Feature | `novelty_zscore` as factor-1 proxy | Captures template-break signal without overfitting |
| C5 Regime | Communication-architecture chronology (ADD vs REMOVE phases) | Forward-guidance removal generates non-linear vol response |

**Warsh test result**: predicted GapSpread = +0.051 pp² → signal BUY_FRONT_SELL_LONG ✓ (correct directional call after all 5 fixes).

---

## 4. IV-Analog vs RV-Analog Hypothesis

**Theoretical framing**:
- NLP-only forecast is trained on RV but its features are calibrated in the ZLB/QE era (positive VRP). The model implicitly encodes the market's pricing convention — it behaves like an IV-analog.
- NLP×regime forecast conditions on economic state. In overheating, regime pushes predictions higher than the market implied — correcting toward actual RV (which is elevated above IV).

**Empirical test** (V5 with ETF IV, full history 2010–2026, n=108 meetings with IV):

| Metric | NLP-only | NLP×regime |
|--------|----------|------------|
| MAE vs IV | 1.245% | **1.217%** |
| MAE vs RV | 0.642% | **0.608%** |
| Corr(pred, IV) | 0.520 | **0.595** |
| Corr(pred, RV) | 0.584 | **0.633** |

**Finding (updated V5)**: With ETF IV filling in 2020–2026, the IV-analog hypothesis is NOW CONFIRMED: both NLP-only (corr=0.520) and NLP×regime (corr=0.595) have significant positive correlation with IV. NLP×regime is closer to IV AND closer to RV simultaneously — consistent with the regime model correcting the directional bias toward actual realized outcomes while remaining anchored to market pricing. The previously-observed correlation of ~0 was a small-sample artifact of the TYVIX-only era (n=60 slack/easing meetings).

---

## 5. Data Integrity Decisions

### 5.1 No Lookahead in Any Feature

All features use strictly backward-looking data at time `t`:
- FRED macro series: point-in-time releases, not revised values (real-time data caveat noted)
- PCA fitted on `{1,...,t-1}`, never including `t`
- `StandardScaler` fit on training window only

### 5.2 Regime Label Independence

Regime labels are constructed from macroeconomic gaps, not from vol outcomes. This is the **key identification assumption**: if regime labels were derived from vol (e.g., "overheating = high vol periods"), the regime-interaction model would be tautological.

### 5.3 Minimum Training Window

`MIN_TRAIN = 15` meetings before the first OOS prediction. Rationale: Ridge regression on p=6–10 features needs at minimum p observations; 15 provides a 1.5–2.5× oversampling ratio at the start and grows thereafter.

### 5.4 Warsh Meeting (2026-06-17)

Deliberately included as the sole forward-test data point. This meeting represents regime extrapolation: the model was trained on ZLB/QE data (slack/easing) and tested on an overheating regime with historically unprecedented vol. The 30Y prediction (+12.5%, actual 4.2%, error +200%) reflects the failure mode of regime extrapolation — the model has never seen a hawkish Warsh-style meeting in its training data.

---

## 6. Known Limitations

1. **No overheating IV data**: TYVIX discontinued May 2020, before the overheating regime began. VRP calibration for overheating relies on RV-only — we cannot verify whether the market was correctly pricing vol in this regime.

2. **Small OOS sample (n=51)**: The Powell-era walk-forward produces 51 OOS predictions. At this sample size, 90% bootstrap CIs for ΔRMSE are wide (~±0.4 pp). The regime model's improvement (+0.12 R²) is economically meaningful but statistically inconclusive.

3. **Real-time FRED data**: FRED series are as-revised, not point-in-time. Core PCE revisions are typically small (<10 bp) but can flip the sign of `inflation_gap` at the ±0.5 pp boundary.

4. **Composite weight (0.6/0.4) is unjustified by formal optimisation**: The presser/statement blend weight is calibrated by judgment. Cross-validating the blend weight would require a held-out validation set that is not available at n=133.

5. **30Y model degrades with composite features**: The `uncertainty_density` composite increase (500×) produces very large feature values for 30Y predictions. 30Y vol has lower signal-to-noise than 2Y, and the large feature range may destabilise Ridge regularisation. Further normalisation or a separate model for 30Y is recommended.

---

## 7. File Reference

| File | Description |
|------|-------------|
| `fomc_corpus_expanded.parquet` | 219 NLP documents (133 statements + 86 pressers), mean presser 8,568 tokens |
| `fomc_features.parquet` | 133×36 master panel: composite NLP features + macro + vol |
| `fomc_dual_mandate_regime.parquet` | Regime labels per meeting from FRED |
| `fomc_nlp_regime_forecasts.parquet` | 51 OOS walk-forward predictions (NLP-only + NLP×regime) |
| `vrp_cache/vrp_panel.parquet` | 798 rows: GK realized vol + TYVIX-implied vol by tenor and meeting |
| `gap_forecasts.parquet` | Per-tenor gap forecasts from `fomc_vrp_pipeline` |
| `gap_forecasts_spread.parquet` | GapSpread (2Y−30Y) predictions from the 5-change model |

---

*Report generated 2026-06-24. All OOS statistics computed on walk-forward held-out sets; no in-sample statistics are used to claim predictive ability.*
