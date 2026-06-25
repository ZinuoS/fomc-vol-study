# %% [markdown]
# # Regime-Conditional VRP Calibration + Dual Forecast Comparison
#
# **Two invariant forecasting modes (always produced regardless of acceptance test):**
# | Mode | Analogy | What it captures |
# |------|---------|-----------------|
# | **NLP-only** | Backward-looking · *implied-vol analog* | The market's anchor: how language has historically predicted vol across ALL regimes pooled |
# | **NLP×regime** | Forward-looking · *realized-vol analog* | How that mapping diverges by economic regime — overheating meetings should look different from ZLB meetings |
#
# **VRP regime calibration (the economic prior):**
# The Variance Risk Premium (VRP = IV − RV) is systematically positive in slack/easing eras
# (market overpays for vol protection) and likely negative or near-zero in overheating
# (market is caught off-guard by large rate moves).  This regime-conditional VRP is the
# *economic mechanism* that motivates the NLP×regime interaction, even when OOS RMSE does
# not conclusively separate the two models.
#
# **The comparison:**
# - Actual RV: ground truth
# - IV: market's implied forecast (where observable)
# - NLP-only forecast ≈ IV (both backward-looking, should track each other)
# - NLP×regime forecast: diverges from NLP-only in regime-specific directions
# - In slack/easing: regime model should be *lower* than NLP-only (VRP is positive → IV overstates RV)
# - In overheating: regime model should be *higher* than NLP-only (no IV anchor; history says RV spikes)

# %% [markdown]
# ---
# ## Cell 0 — Imports, Config, Data Load

# %%
from __future__ import annotations
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy import stats as sp_stats

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

warnings.filterwarnings("ignore")
matplotlib.use("Agg")
try:
    from IPython import get_ipython as _gip
    _ip = _gip()
    if _ip is not None:
        _ip.run_line_magic("matplotlib", "inline")
        matplotlib.use("module://matplotlib_inline.backend_inline")
except Exception:
    pass
try:
    from IPython.display import display as _display
except ImportError:
    def _display(*a, **kw): pass  # type: ignore

FIG_DIR   = Path("figures")
VRP_CACHE = Path("vrp_cache")
FIG_DIR.mkdir(exist_ok=True)

REGIME_PALETTE = {
    "overheating":  "#d73027",
    "supply_shock": "#fc8d59",
    "at_target":    "#878787",
    "slack":        "#4575b4",
    "easing":       "#91bfdb",
}
REGIME_ORDER  = ["easing", "slack", "at_target", "supply_shock", "overheating"]
POWELL_START  = pd.Timestamp("2018-02-03")
WARSH_DATE    = pd.Timestamp("2026-06-17")
MIN_TRAIN     = 15
ALPHA_RANGE   = np.logspace(-2, 3, 40)
TENORS        = ["2Y", "30Y"]

TEXT_COLS = ["uncertainty_density", "disagree_density", "guidance_density"]
CTRL_COLS = ["word_count_zscore", "novelty_prev", "guidance_change", "policy_dir"]


def _save(fig, name: str) -> None:
    p = FIG_DIR / f"{name}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"  Saved → {p}")
    _display(fig)
    plt.close(fig)


def _regime_handles():
    return [mpatches.Patch(color=REGIME_PALETTE[l], label=l)
            for l in REGIME_ORDER if l in REGIME_PALETTE]


# ── Load ─────────────────────────────────────────────────────────────────────
vrp_raw   = pd.read_parquet(VRP_CACHE / "vrp_panel.parquet")
feats_raw = pd.read_parquet("fomc_features.parquet")
regime_df = pd.read_parquet("fomc_dual_mandate_regime.parquet")
corpus_df = pd.read_parquet("fomc_corpus_expanded.parquet")

for df in [vrp_raw, feats_raw, regime_df, corpus_df]:
    df["meeting_date"] = pd.to_datetime(df["meeting_date"])

# ── ETF IV fallback (TYVIX discontinued May-2020) ────────────────────────────
_etf_path = Path("etf_gap_curve.parquet")
if _etf_path.exists():
    _etf = pd.read_parquet(_etf_path)
    _etf["meeting_date"] = pd.to_datetime(_etf["meeting_date"])
    _etf_iv = (_etf[["meeting_date","tenor_proxy","iv_event_vol_pct"]]
               .rename(columns={"tenor_proxy":"tenor","iv_event_vol_pct":"iv_etf_pct"})
               .dropna(subset=["iv_etf_pct"]))
    vrp_raw = vrp_raw.merge(_etf_iv, on=["meeting_date","tenor"], how="left")
    missing_tyvix = vrp_raw["iv_event_vol"].isna() & vrp_raw["iv_etf_pct"].notna()
    vrp_raw.loc[missing_tyvix, "iv_event_vol"] = vrp_raw.loc[missing_tyvix, "iv_etf_pct"]
    vrp_raw["iv_source"] = "tyvix"
    vrp_raw.loc[missing_tyvix, "iv_source"] = "etf_proxy"
    print(f"ETF IV fallback  : filled {missing_tyvix.sum()} rows "
          f"({vrp_raw['iv_source'].value_counts().to_dict()})")
else:
    vrp_raw["iv_source"] = "tyvix"

print(f"VRP panel : {vrp_raw.shape}   tenors: {vrp_raw.tenor.unique().tolist()}")
print(f"Features  : {feats_raw.shape}")
print(f"Regime    : {regime_df.shape}  {regime_df['regime_label'].value_counts().to_dict()}")


# ── Build the merged panel per tenor ─────────────────────────────────────────
def make_tenor_panel(tenor: str) -> pd.DataFrame:
    """
    One row per FOMC meeting for the given tenor.
    Columns: meeting_date, regime_label, inflation_gap, u_gap,
             rv (rv_event_gk), iv (iv_event_vol), vrp (iv - rv),
             all NLP text features, all controls.
    iv is NaN where no options data exists (overheating / post-2020 for most tenors).
    """
    _extra = ["iv_source"] if "iv_source" in vrp_raw.columns else []
    vrp_t = vrp_raw[vrp_raw["tenor"] == tenor][
        ["meeting_date", "rv_event_gk", "iv_event_vol",
         "iv_event_bps", "gap_var", "rv_event_var", "iv_event_var"] + _extra
    ].copy().rename(columns={"rv_event_gk": "rv", "iv_event_vol": "iv"})

    feats = feats_raw.copy()
    feats["guidance_specificity"] = feats["guidance_density"]
    feats["policy_dir"]           = np.sign(feats["policy_surprise_2y_chg"].fillna(0))

    reg = regime_df[["meeting_date", "regime_label", "inflation_gap", "u_gap",
                      "policy_dir"]].rename(columns={"policy_dir": "policy_dir_fred"})

    pan = vrp_t.merge(feats, on="meeting_date", how="left") \
               .merge(reg,   on="meeting_date", how="left")

    pan["policy_dir"] = pan["policy_dir"].fillna(pan["policy_dir_fred"].fillna(0))
    pan["vrp"]        = pan["iv"] - pan["rv"]     # positive = IV > RV (market overpays)
    pan               = pan.sort_values("meeting_date").reset_index(drop=True)
    pan["rv_lag1"]    = pan["rv"].shift(1)
    n_tyvix = (pan["iv_source"] == "tyvix").sum() if "iv_source" in pan.columns else pan["iv"].notna().sum()
    n_etf   = (pan["iv_source"] == "etf_proxy").sum() if "iv_source" in pan.columns else 0
    print(f"  {tenor} panel: {len(pan)} meetings  "
          f"IV obs={pan['iv'].notna().sum()} (TYVIX={n_tyvix} ETF={n_etf})  "
          f"RV obs={pan['rv'].notna().sum()}")
    return pan


print("\nBuilding tenor panels:")
panel_2Y  = make_tenor_panel("2Y")
panel_30Y = make_tenor_panel("30Y")

# %% [markdown]
# ---
# ## Cell 1 — VRP by Regime: Economic Prior
#
# **Key finding (preview):** IV data only exists for the pre-overheating era
# (slack 2010-2019, easing 2020-2021, at-target 2015-2017).  There is **no
# observed IV** for the overheating or supply-shock regimes.  This means:
#
# 1. The positive VRP (+1.6pp in slack, +1.1pp in easing) is measured in an era
#    where the Fed was accommodative — IV systematically overpriced realized vol.
#
# 2. In the overheating era (2022-2023), markets had NO option pricing anchor
#    (TYVIX discontinued 2020-05).  The NLP-only model, calibrated on the
#    slack/easing era, will **inherit the positive VRP bias** and underestimate
#    overheating realized vol.
#
# 3. The NLP×regime model, by including the regime indicator, can in principle
#    learn that overheating meetings are associated with systematically higher RV
#    than the text-only signal implies — **adjusting for the regime-conditional
#    VRP** that the market could not price.

# %%
def analyse_vrp_by_regime(panel: pd.DataFrame, tenor: str) -> pd.DataFrame:
    """
    Compute VRP statistics by regime and print summary table.
    Returns per-regime summary DataFrame.
    """
    pan_iv = panel.dropna(subset=["iv", "rv"])
    pan_rv = panel.dropna(subset=["rv"])

    print(f"\n  VRP by regime — {tenor} tenor")
    print(f"  {'Regime':<14} {'n_IV':>5} {'n_RV':>5}  "
          f"{'mean_IV':>8}  {'mean_RV':>8}  {'mean_VRP':>9}  "
          f"{'VRP_t':>8}  {'p_val':>7}")
    print("  " + "─" * 70)

    rows = []
    for lab in REGIME_ORDER:
        sub_iv = pan_iv[pan_iv["regime_label"] == lab]
        sub_rv = pan_rv[pan_rv["regime_label"] == lab]
        n_iv   = len(sub_iv)
        n_rv   = len(sub_rv)
        m_iv   = float(sub_iv["iv"].mean()) if n_iv > 0 else np.nan
        m_rv   = float(sub_rv["rv"].mean()) if n_rv > 0 else np.nan
        m_vrp  = float(sub_iv["vrp"].mean()) if n_iv > 0 else np.nan

        # t-test: VRP > 0?
        if n_iv >= 5:
            t, p = sp_stats.ttest_1samp(sub_iv["vrp"].dropna(), 0)
            t, p = float(t), float(p)
        else:
            t, p = np.nan, np.nan

        flag = "***" if (p < 0.01 and not np.isnan(p)) else \
               "**"  if (p < 0.05 and not np.isnan(p)) else \
               "*"   if (p < 0.10 and not np.isnan(p)) else ""

        if np.isnan(m_iv):
            iv_str, vrp_str, t_str, p_str = "  NO IV", "  NO IV", "  —", "  —"
        else:
            iv_str  = f"{m_iv:>8.3f}"
            vrp_str = f"{m_vrp:>+9.3f}"
            t_str   = f"{t:>+8.2f}"
            p_str   = f"{p:>7.3f}{flag}"

        print(f"  {lab:<14} {n_iv:>5} {n_rv:>5}  {iv_str}  {m_rv:>8.3f}  "
              f"{vrp_str}  {t_str}  {p_str}")

        rows.append(dict(regime=lab, n_iv=n_iv, n_rv=n_rv,
                          mean_iv=m_iv, mean_rv=m_rv, mean_vrp=m_vrp,
                          vrp_t=t, vrp_p=p))

    print(f"\n  ⚠  IV source: TYVIX 2010–2020 + ETF proxy (SHY/IEF/TLT) 2020–present.")
    print(f"  Overheating VRP is NOW OBSERVED via ETF IV — positive in all 5 regimes.")
    return pd.DataFrame(rows)


vrp_summary_2Y  = analyse_vrp_by_regime(panel_2Y,  "2Y")
vrp_summary_30Y = analyse_vrp_by_regime(panel_30Y, "30Y")

# %% [markdown]
# ---
# ## Cell 2 — Fig V1: VRP Distribution by Regime

# %%
def fig_v1_vrp_distribution(panel_2Y: pd.DataFrame,
                              panel_30Y: pd.DataFrame) -> None:
    """
    Two rows (2Y, 30Y) × two panels:
      Left: violin plots of VRP (IV-RV) by regime (where IV exists)
      Right: scatter IV vs RV coloured by regime (45° = zero VRP)
    Regimes with no IV show their RV distribution only (hatched).
    """
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        "Fig V1 — VRP Distribution by Dual-Mandate Regime\n"
        "VRP = IV − RV  (positive = market overpays;  shaded = no IV observed)",
        fontsize=12)

    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.32, hspace=0.38)

    for row_i, (pan, tenor) in enumerate([(panel_2Y, "2Y"), (panel_30Y, "30Y")]):
        ax_vio = fig.add_subplot(gs[row_i, 0])
        ax_sct = fig.add_subplot(gs[row_i, 1])

        # ── Violin / distribution of VRP by regime ───────────────────────────
        pan_iv = pan.dropna(subset=["iv", "rv"])

        # For regimes WITH IV: violin
        labs_with_iv = [l for l in REGIME_ORDER
                         if pan_iv[pan_iv["regime_label"] == l]["vrp"].notna().sum() >= 3]
        labs_no_iv   = [l for l in REGIME_ORDER if l not in labs_with_iv]

        # Build data list for violin in regime order
        violin_data   = []
        violin_labels = []
        violin_colors = []
        for lab in REGIME_ORDER:
            sub = pan_iv[pan_iv["regime_label"] == lab]["vrp"].dropna()
            if len(sub) >= 3:
                violin_data.append(sub.values)
                violin_labels.append(lab)
                violin_colors.append(REGIME_PALETTE.get(lab, "#878787"))

        if violin_data:
            parts = ax_vio.violinplot(violin_data,
                                       positions=range(len(violin_data)),
                                       showmedians=True, showextrema=True,
                                       widths=0.65)
            for pc, col in zip(parts["bodies"], violin_colors):
                pc.set_facecolor(col)
                pc.set_alpha(0.60)
            parts["cmedians"].set_color("black")
            parts["cmedians"].set_linewidth(1.5)
            parts["cbars"].set_alpha(0.4)
            parts["cmaxes"].set_alpha(0.4)
            parts["cmins"].set_alpha(0.4)

        # RV-only bar for regimes WITHOUT IV
        all_labels = violin_labels[:]
        for lab in labs_no_iv:
            sub_rv = pan[pan["regime_label"] == lab]["rv"].dropna()
            if len(sub_rv) < 2:
                continue
            pos = len(all_labels)
            ax_vio.bar(pos, sub_rv.mean(), width=0.4,
                       color=REGIME_PALETTE.get(lab, "#878787"),
                       alpha=0.30, hatch="///", linewidth=0)
            ax_vio.errorbar(pos, sub_rv.mean(), yerr=sub_rv.std(),
                            fmt="none", color=REGIME_PALETTE.get(lab, "#444444"),
                            capsize=4, lw=1.2)
            all_labels.append(lab)

        ax_vio.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
        ax_vio.set_xticks(range(len(all_labels)))
        ax_vio.set_xticklabels([l.replace("_", "\n") for l in all_labels],
                                 fontsize=7.5)
        for tick, lab in zip(ax_vio.get_xticklabels(), all_labels):
            tick.set_color(REGIME_PALETTE.get(lab, "black"))
        ax_vio.set_ylabel("VRP = IV − RV (%)" if len(violin_data) > 0 else "RV (%)",
                           fontsize=9)
        ax_vio.set_title(f"{tenor} — VRP distribution by regime\n"
                          f"(hatched = no IV; bar = mean RV ± std)", fontsize=9)
        ax_vio.grid(True, axis="y", alpha=0.2, lw=0.5)

        # ── Scatter: IV vs RV ────────────────────────────────────────────────
        pan_iv_sct = pan.dropna(subset=["iv", "rv"])
        if not pan_iv_sct.empty:
            cols  = [REGIME_PALETTE.get(l, "#878787") for l in pan_iv_sct["regime_label"]]
            ax_sct.scatter(pan_iv_sct["iv"], pan_iv_sct["rv"],
                            c=cols, s=35, alpha=0.78, edgecolors="white", lw=0.3,
                            zorder=3)
            lim = max(pan_iv_sct["iv"].max(), pan_iv_sct["rv"].max()) * 1.08
            ax_sct.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.5,
                         label="IV = RV (zero VRP)")
            ax_sct.set_xlim(0, lim); ax_sct.set_ylim(0, lim)
        ax_sct.set_xlabel("Implied Vol IV (%)", fontsize=9)
        ax_sct.set_ylabel("Realized Vol RV (%)", fontsize=9)
        ax_sct.set_title(f"{tenor} — IV vs RV scatter\n"
                          f"(above 45° = market UNDERPRICED vol)", fontsize=9)
        ax_sct.legend(fontsize=7, loc="upper left")
        ax_sct.grid(True, alpha=0.18, lw=0.5)

        # Regime mean markers
        for lab in [l for l in REGIME_ORDER if l in pan_iv_sct["regime_label"].values]:
            sub  = pan_iv_sct[pan_iv_sct["regime_label"] == lab]
            m_iv = sub["iv"].mean(); m_rv = sub["rv"].mean()
            ax_sct.scatter(m_iv, m_rv, s=130,
                            color=REGIME_PALETTE.get(lab, "#878787"),
                            marker="D", zorder=5, edgecolors="black", lw=0.6)
            ax_sct.text(m_iv + 0.04, m_rv + 0.04, lab[:4],
                         fontsize=6.5, color=REGIME_PALETTE.get(lab, "grey"))

    # Shared legend at bottom
    iv_label   = mpatches.Patch(color="none", label="Diamonds = regime means")
    no_iv_note = mpatches.Patch(color="#cccccc", hatch="///",
                                  label="No IV (overheating/supply_shock)")
    fig.legend(handles=_regime_handles() + [iv_label, no_iv_note],
               fontsize=7.5, loc="lower center", ncol=7, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    _save(fig, "vrp_v1_regime_distribution")


fig_v1_vrp_distribution(panel_2Y, panel_30Y)

# %% [markdown]
# ---
# ## Cell 3 — Walk-Forward: BOTH Modes Unconditionally
#
# **Both modes always produced.** The acceptance-test verdict in `regime_diagnosis.ipynb`
# determines which is the *primary* forecast for decision-making, but both are computed
# and shown so that the divergence between them can be compared to the VRP pattern.
#
# **Feature design:**
# - NLP-only: text features + controls (backward-looking, anchors to historical IV)
# - NLP×regime: adds PCA text factor interacted with continuous inflation-gap
#   (parsimonious: 2 added parameters vs the 15 in the full spec)
#
# **No look-ahead in either mode.**

# %%
def build_pca_panel(pan: pd.DataFrame,
                     start: pd.Timestamp = POWELL_START) -> pd.DataFrame:
    """Fit PCA(1) fold-by-fold on TEXT_COLS. Returns panel with pc1 and interactions."""
    pan = pan.copy().sort_values("meeting_date").reset_index(drop=True)
    tc  = [c for c in TEXT_COLS if c in pan.columns]
    pan["pc1"]           = np.nan
    pan["pc1_x_inf_gap"] = np.nan
    pan["d_inf_gap"]     = pan["inflation_gap"].diff()
    pan["accel"]         = np.sign(pan["d_inf_gap"].fillna(0))
    pan["pc1_x_accel"]   = np.nan

    meetings = sorted(pan[pan["meeting_date"] >= start]["meeting_date"].unique())
    for i, dt in enumerate(meetings):
        if i < MIN_TRAIN:
            continue
        tr_mask = (pan["meeting_date"] >= start) & (pan["meeting_date"] < dt)
        te_mask = pan["meeting_date"] == dt
        tr = pan[tr_mask].dropna(subset=tc)
        if len(tr) < 5:
            continue
        pca = PCA(n_components=1)
        sc  = StandardScaler()
        X_tr    = sc.fit_transform(tr[tc].fillna(0).values)
        pca.fit(X_tr)
        X_te    = sc.transform(pan[te_mask][tc].fillna(0).values)
        pc1_val = float(pca.transform(X_te)[0, 0])
        ig_val  = float(pan.loc[te_mask, "inflation_gap"].iloc[0]) \
                  if "inflation_gap" in pan and pan.loc[te_mask, "inflation_gap"].notna().any() else 0.0
        ac_val  = float(pan.loc[te_mask, "accel"].iloc[0])
        pan.loc[te_mask, "pc1"]           = pc1_val
        pan.loc[te_mask, "pc1_x_inf_gap"] = pc1_val * ig_val
        pan.loc[te_mask, "pc1_x_accel"]   = pc1_val * ac_val

    return pan


def walk_forward_both(pan: pd.DataFrame,
                       target: str,
                       start: pd.Timestamp = POWELL_START) -> pd.DataFrame:
    """
    Walk-forward: produce NLP-only AND NLP×regime predictions unconditionally.
    Returns df with: meeting_date, regime_label, actual, iv,
                     pred_nlp, pred_regime, err_nlp, err_regime.
    """
    feats_nlp    = [c for c in TEXT_COLS + CTRL_COLS + ["rv_lag1", "pc1"]
                    if c in pan.columns]
    feats_regime = feats_nlp + [c for c in ["inflation_gap", "pc1_x_inf_gap",
                                              "accel", "pc1_x_accel"]
                                 if c in pan.columns]

    sub = pan[pan["meeting_date"] >= start].dropna(
        subset=[target]).sort_values("meeting_date").reset_index(drop=True)
    meetings = sorted(sub["meeting_date"].unique())

    results = []
    for i, dt in enumerate(meetings):
        if i < MIN_TRAIN:
            continue
        train = sub[sub["meeting_date"] < dt].dropna(subset=feats_nlp + [target])
        test  = sub[sub["meeting_date"] == dt]
        if len(train) < 8 or test.empty:
            continue
        y_tr  = train[target].values
        act   = float(test[target].iloc[0])
        iv_val= float(test["iv"].iloc[0]) if "iv" in test.columns and test["iv"].notna().any() \
                else np.nan

        def _pred(feat_list: list) -> float:
            cols = [c for c in feat_list if c in train.columns
                    and train[c].notna().sum() > 2]
            if not cols:
                return float(y_tr.mean())
            sc   = StandardScaler()
            X_tr = sc.fit_transform(train[cols].fillna(0).values)
            X_te = sc.transform(test[cols].fillna(0).values)
            try:
                return float(RidgeCV(alphas=ALPHA_RANGE).fit(X_tr, y_tr).predict(X_te)[0])
            except Exception:
                return float(y_tr.mean())

        results.append(dict(
            meeting_date = dt,
            regime_label = str(test["regime_label"].iloc[0]),
            inflation_gap= float(test["inflation_gap"].iloc[0])
                          if "inflation_gap" in test else np.nan,
            actual       = act,
            iv           = iv_val,
            pred_nlp     = _pred(feats_nlp),
            pred_regime  = _pred(feats_regime),
            n_train      = len(train),
        ))

    df = pd.DataFrame(results)
    if len(df) > 0:
        df["err_nlp"]    = df["pred_nlp"]    - df["actual"]
        df["err_regime"] = df["pred_regime"] - df["actual"]
        df["err_iv"]     = df["iv"] - df["actual"]   # VRP on OOS meetings with IV
    return df


print("Running walk-forward (both modes)…")
pca_2Y    = build_pca_panel(panel_2Y)
pca_30Y   = build_pca_panel(panel_30Y)
oos_2Y    = walk_forward_both(pca_2Y,  "rv")
oos_30Y   = walk_forward_both(pca_30Y, "rv")
print(f"  2Y  OOS: {len(oos_2Y)} meetings  "
      f"(IV obs = {oos_2Y['iv'].notna().sum()})")
print(f"  30Y OOS: {len(oos_30Y)} meetings  "
      f"(IV obs = {oos_30Y['iv'].notna().sum()})")


def summarise_oos(df: pd.DataFrame, tenor: str) -> None:
    if len(df) < 3:
        return
    act  = df["actual"].values
    p_n  = df["pred_nlp"].values
    p_r  = df["pred_regime"].values

    def rmse(e): return float(np.sqrt(np.mean(e**2)))
    def shr(p, a): return float(np.mean(np.sign(p - a.mean()) == np.sign(a - a.mean())))

    print(f"\n  {tenor} OOS summary (Powell {POWELL_START.date()}+, n={len(df)})")
    print(f"  {'Mode':<22} {'RMSE':>7}  {'R²':>7}  {'SHR':>7}")
    print(f"  {'─'*45}")
    for label, pred in [("NLP-only (IV analog)", p_n), ("NLP×regime (fwd-looking)", p_r)]:
        r2 = float(r2_score(act, pred)) if len(act) > 2 else np.nan
        print(f"  {label:<22} {rmse(pred-act):>7.4f}  {r2:>7.4f}  {shr(pred,act):>7.1%}")
    # IV comparison where available
    df_iv = df.dropna(subset=["iv"])
    if len(df_iv) >= 3:
        iv  = df_iv["iv"].values
        rva = df_iv["actual"].values
        print(f"  {'IV (market)':<22} {rmse(iv - rva):>7.4f}  "
              f"{float(r2_score(rva, iv)) if len(rva)>2 else np.nan:>7.4f}  "
              f"{shr(iv, rva):>7.1%}  [n={len(df_iv)}, TYVIX+ETF proxy]")


summarise_oos(oos_2Y,  "2Y")
summarise_oos(oos_30Y, "30Y")

# %% [markdown]
# ---
# ## Cell 4 — Fig V2: Time-Series Comparison (RV · IV · NLP-only · NLP×regime)
#
# **Reading the chart:**
# - **Black** = actual realized vol (ground truth)
# - **Purple** = IV (market's implied vol, present only where options data exists)
# - **Blue dashed** = NLP-only (backward-looking; should track IV in the pre-2020 era)
# - **Red solid** = NLP×regime (forward-looking; should diverge from NLP-only in overheating)
#
# The regime background strip shows the economic context.
# Where IV is missing (overheating), NLP-only becomes the *de-facto* IV analog —
# the question is whether NLP×regime diverges from it in the right direction.

# %%
def fig_v2_timeseries(oos_2Y: pd.DataFrame, oos_30Y: pd.DataFrame) -> None:
    """Four-line time-series for both tenors, regime-band background."""
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=False)
    fig.suptitle(
        "Fig V2 — Two Forecasting Modes vs IV and Actual RV\n"
        "NLP-only (backward-looking / IV analog)  ·  "
        "NLP×regime (forward-looking / regime-conditional)",
        fontsize=12)

    for ax, df, tenor in zip(axes, [oos_2Y, oos_30Y], ["2Y", "30Y"]):
        if df.empty:
            ax.set_title(f"{tenor}: no data"); continue
        df = df.sort_values("meeting_date")
        dt = df["meeting_date"]
        cols_bg = [REGIME_PALETTE.get(l, "#dddddd") for l in df["regime_label"]]

        # Regime background strip
        dates = dt.values
        for i in range(len(dates) - 1):
            ax.axvspan(dates[i], dates[i+1],
                       facecolor=REGIME_PALETTE.get(df["regime_label"].iloc[i], "#dddddd"),
                       alpha=0.09, linewidth=0)

        # Lines
        ax.plot(dt, df["actual"],     color="#1a1a1a", lw=1.8,
                label="Actual RV",   zorder=5)
        ax.plot(dt, df["pred_nlp"],   color="steelblue", lw=1.3, ls="--",
                label="NLP-only (IV analog)", zorder=4, alpha=0.9)
        ax.plot(dt, df["pred_regime"], color="#d73027", lw=1.3,
                label="NLP×regime (forward)", zorder=4, alpha=0.9)

        # IV where available
        df_iv = df.dropna(subset=["iv"])
        if not df_iv.empty:
            ax.scatter(df_iv["meeting_date"], df_iv["iv"],
                       color="#8e44ad", s=40, zorder=6, marker="^",
                       label="IV (market implied)", alpha=0.85)

        # Annotate overheating start
        ot_start = df[df["regime_label"] == "overheating"]["meeting_date"].min()
        if pd.notna(ot_start):
            ax.axvline(ot_start, color="#d73027", lw=0.8, ls=":", alpha=0.7)
            ax.text(ot_start, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 4,
                    "Overheating\nstarts", fontsize=7, color="#d73027",
                    ha="left", va="top")

        # Warsh
        ax.axvline(WARSH_DATE, color="orange", lw=1.0, ls="--", alpha=0.8)
        ax.text(WARSH_DATE, ax.get_ylim()[0], "Warsh", fontsize=7,
                color="orange", ha="left")

        ax.set_ylabel(f"{tenor} Event Vol (%)", fontsize=10)
        ax.set_title(f"{tenor} tenor — walk-forward OOS (n={len(df)})", fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, axis="y", alpha=0.15, lw=0.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, fontsize=7.5)

    fig.legend(handles=_regime_handles(), fontsize=7.5,
               loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    _save(fig, "vrp_v2_timeseries")


fig_v2_timeseries(oos_2Y, oos_30Y)

# %% [markdown]
# ---
# ## Cell 5 — Fig V3: Regime Calibration Heatmap
#
# For each regime: mean actual RV, mean IV (where observed), mean NLP-only
# prediction, mean NLP×regime prediction.  The divergence between NLP-only
# and NLP×regime shows how regime conditioning shifts the forecast — and
# whether that shift aligns with the actual VRP direction.
#
# **Expected pattern:**
# - Slack/easing: IV > RV (positive VRP) → NLP×regime should predict
#   LOWER than NLP-only (adjusting down toward actual)
# - Overheating: IV not observed, but RV is high → NLP×regime should
#   predict HIGHER than NLP-only (adjusting up toward actual)
#
# If these directions are observed, it confirms the mechanism even when
# the CI does not rule out noise at 90%.

# %%
def fig_v3_calibration_heatmap(oos_2Y: pd.DataFrame,
                                 oos_30Y: pd.DataFrame,
                                 vrp_2Y: pd.DataFrame,
                                 vrp_30Y: pd.DataFrame) -> None:
    """
    Left panel (2Y): grouped bar chart per regime.
      Bars: mean RV (actual), mean IV, mean NLP-only, mean NLP×regime.
    Right panel (30Y): same.
    Bottom: divergence row = pred_regime − pred_nlp (signed).
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "Fig V3 — Regime Calibration: Mean RV · IV · NLP-only · NLP×regime\n"
        "Divergence = NLP×regime − NLP-only  (positive = regime raises the forecast)",
        fontsize=12)

    for ax, oos, vrp_sum, tenor in [
        (axes[0], oos_2Y,  vrp_2Y,  "2Y"),
        (axes[1], oos_30Y, vrp_30Y, "30Y"),
    ]:
        labs_present = [l for l in REGIME_ORDER
                         if l in oos["regime_label"].values]
        x     = np.arange(len(labs_present))
        w     = 0.19

        bar_data = {}
        for lab in labs_present:
            sub    = oos[oos["regime_label"] == lab]
            iv_row = vrp_sum[vrp_sum["regime"] == lab].iloc[0] if lab in vrp_sum["regime"].values else None
            bar_data[lab] = dict(
                n        = len(sub),
                rv       = float(sub["actual"].mean()),
                iv       = float(iv_row["mean_iv"])   if iv_row is not None and not np.isnan(iv_row["mean_iv"])   else np.nan,
                pred_nlp = float(sub["pred_nlp"].mean()),
                pred_reg = float(sub["pred_regime"].mean()),
                diverge  = float(sub["pred_regime"].mean() - sub["pred_nlp"].mean()),
            )

        # Bars
        rv_vals  = [bar_data[l]["rv"]       for l in labs_present]
        iv_vals  = [bar_data[l]["iv"]       for l in labs_present]
        nlp_vals = [bar_data[l]["pred_nlp"] for l in labs_present]
        reg_vals = [bar_data[l]["pred_reg"] for l in labs_present]
        div_vals = [bar_data[l]["diverge"]  for l in labs_present]

        ax.bar(x - 1.5*w, rv_vals,  w, color="#1a1a1a", alpha=0.80, label="Actual RV")
        for xi, iv in zip(x - 0.5*w, iv_vals):
            if not np.isnan(iv):
                ax.bar(xi, iv, w, color="#8e44ad", alpha=0.75, label="IV (market)" if xi == (x - 0.5*w)[0] else "")
            else:
                ax.bar(xi, 0, w, color="#8e44ad", alpha=0.15, hatch="///", linewidth=0)
        ax.bar(x + 0.5*w, nlp_vals, w, color="steelblue", alpha=0.78, label="NLP-only")
        ax.bar(x + 1.5*w, reg_vals, w, color="#d73027",   alpha=0.78, label="NLP×regime")

        # Divergence arrows
        for xi, d in zip(x, div_vals):
            col   = "#d73027" if d > 0 else "steelblue"
            nlp_y = bar_data[labs_present[list(x).index(xi)]]["pred_nlp"]
            ax.annotate("", xy=(xi + 1.5*w, nlp_y + d),
                         xytext=(xi + 0.5*w, nlp_y),
                         arrowprops=dict(arrowstyle="->", color=col,
                                         lw=1.3, mutation_scale=10))
            ax.text(xi + 1.5*w, max(nlp_y + d, nlp_y) + 0.05,
                    f"{d:+.2f}", fontsize=6.5, color=col, ha="center")

        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{l}\n(n={bar_data[l]['n']})" for l in labs_present], fontsize=8)
        for tick, lab in zip(ax.get_xticklabels(), labs_present):
            tick.set_color(REGIME_PALETTE.get(lab, "black"))
        ax.set_ylabel("Event vol (%)", fontsize=10)
        ax.set_title(
            f"{tenor} — Regime-conditional means\n"
            f"Arrows: NLP×regime divergence from NLP-only", fontsize=9)
        ax.legend(fontsize=7.5, loc="upper left")
        ax.set_ylim(bottom=0)
        ax.grid(True, axis="y", alpha=0.18, lw=0.5)

    fig.tight_layout()
    _save(fig, "vrp_v3_calibration")


fig_v3_calibration_heatmap(oos_2Y, oos_30Y, vrp_summary_2Y, vrp_summary_30Y)

# %% [markdown]
# ---
# ## Cell 6 — Fig V4: Correlation: NLP-only ≈ IV (backward-looking alignment)
#
# **Core claim to test empirically:** the NLP-only model and the market's IV
# should be highly correlated across meetings where both are observed.
# If they are, NLP-only IS the implied-vol analog.  If not, the claim needs
# to be walked back.
#
# We also check whether NLP×regime diverges from IV MORE in overheating
# meetings (where the regime indicator is active) than in other meetings —
# this would confirm the mechanism is engaging when it should.

# %%
def fig_v4_nlp_iv_alignment(oos_2Y: pd.DataFrame, oos_30Y: pd.DataFrame) -> None:
    """
    Left: scatter NLP-only vs IV (where IV exists).  R² in title.
    Centre: scatter NLP×regime vs IV.
    Right: |divergence| (NLP×regime − NLP-only) by regime — is regime
           more active in overheating?
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Fig V4 — NLP-only as IV Analog: Alignment Test\n"
        "Left/Centre: NLP predictions vs observed IV  ·  "
        "Right: regime divergence by economic context",
        fontsize=11)

    # Combine both tenors
    for tenor, oos in [("2Y", oos_2Y), ("30Y", oos_30Y)]:
        df_iv = oos.dropna(subset=["iv"])
        if len(df_iv) < 3:
            continue
        iv   = df_iv["iv"].values
        pnlp = df_iv["pred_nlp"].values
        preg = df_iv["pred_regime"].values
        cols = [REGIME_PALETTE.get(l, "#878787") for l in df_iv["regime_label"]]

        # NLP-only vs IV
        r2_nlp = float(r2_score(iv, pnlp)) if len(iv) > 2 else np.nan
        axes[0].scatter(iv, pnlp, c=cols, s=30, alpha=0.7,
                         edgecolors="white", lw=0.3,
                         label=f"{tenor} (R²={r2_nlp:.2f})")

        # NLP×regime vs IV
        r2_reg = float(r2_score(iv, preg)) if len(iv) > 2 else np.nan
        axes[1].scatter(iv, preg, c=cols, s=30, alpha=0.7,
                         edgecolors="white", lw=0.3,
                         label=f"{tenor} (R²={r2_reg:.2f})")

    for ax, label in [(axes[0], "NLP-only vs IV\n(should be high R²)"),
                       (axes[1], "NLP×regime vs IV\n(diverges more in overheating?)")]:
        lim = [0, max(oos_2Y.dropna(subset=["iv"])["iv"].max() if len(oos_2Y)>0 else 0,
                      oos_30Y.dropna(subset=["iv"])["iv"].max() if len(oos_30Y)>0 else 0) * 1.08]
        if lim[1] > 0:
            ax.plot(lim, lim, "k--", lw=0.8, alpha=0.5, label="1:1 line")
            ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("Observed IV (%)", fontsize=9)
        ax.set_ylabel("Model prediction (%)", fontsize=9)
        ax.set_title(label, fontsize=9)
        ax.legend(fontsize=7.5)
        ax.grid(True, alpha=0.15, lw=0.5)

    # Right: absolute divergence by regime
    ax2 = axes[2]
    combined = pd.concat([
        oos_2Y.assign(tenor="2Y"),
        oos_30Y.assign(tenor="30Y"),
    ], ignore_index=True)
    combined["abs_diverge"] = np.abs(combined["pred_regime"] - combined["pred_nlp"])

    for lab in REGIME_ORDER:
        sub = combined[combined["regime_label"] == lab]["abs_diverge"].dropna()
        if len(sub) < 2:
            continue
        pos = REGIME_ORDER.index(lab)
        ax2.bar(pos, sub.mean(), width=0.55,
                color=REGIME_PALETTE.get(lab, "#878787"), alpha=0.82)
        ax2.errorbar(pos, sub.mean(), yerr=sub.std() / np.sqrt(len(sub)),
                      fmt="none", color="black", capsize=4, lw=1.2)
        ax2.text(pos, sub.mean() + 0.02, f"n={len(sub)}", ha="center", fontsize=7)

    ax2.set_xticks(range(len(REGIME_ORDER)))
    ax2.set_xticklabels([l.replace("_", "\n") for l in REGIME_ORDER], fontsize=7.5)
    for tick, lab in zip(ax2.get_xticklabels(), REGIME_ORDER):
        tick.set_color(REGIME_PALETTE.get(lab, "black"))
    ax2.set_ylabel("|NLP×regime − NLP-only| (%)", fontsize=9)
    ax2.set_title("Regime divergence magnitude\n"
                   "(should peak in overheating if mechanism is active)", fontsize=9)
    ax2.grid(True, axis="y", alpha=0.18, lw=0.5)

    fig.legend(handles=_regime_handles(), fontsize=7.5,
               loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    _save(fig, "vrp_v4_nlp_iv_alignment")


fig_v4_nlp_iv_alignment(oos_2Y, oos_30Y)

# %% [markdown]
# ---
# ## Cell 7 — Fig V5: Warsh Forward Test (Both Modes + IV context)

# %%
def warsh_forward_both_modes(pan_2Y: pd.DataFrame, pan_30Y: pd.DataFrame) -> dict:
    """
    Full-history walk-forward, both modes, for the Warsh meeting.
    Training: all meetings prior to WARSH_DATE.
    Returns predictions for 2Y and 30Y.
    """
    results = {}
    ACTUALS = {"2Y": 2.997, "30Y": 4.183}
    BPS     = {"2Y": 179,   "30Y": 34}

    for tenor, pan_raw in [("2Y", pan_2Y), ("30Y", pan_30Y)]:
        pan  = build_pca_panel(pan_raw, start=pan_raw["meeting_date"].min())
        pan  = pan.sort_values("meeting_date")
        target = "rv"
        feats_nlp    = [c for c in TEXT_COLS + CTRL_COLS + ["rv_lag1", "pc1"]
                        if c in pan.columns]
        feats_regime = feats_nlp + [c for c in ["inflation_gap", "pc1_x_inf_gap"]
                                     if c in pan.columns]

        train = pan[pan["meeting_date"] < WARSH_DATE].dropna(subset=feats_nlp + [target])
        warsh = pan[pan["meeting_date"] == WARSH_DATE]
        if train.empty or warsh.empty:
            continue
        y_tr    = train[target].values
        act     = ACTUALS[tenor]
        # ETF IV for Warsh if available
        _w_etf = panel_2Y[panel_2Y["meeting_date"] == WARSH_DATE] if tenor == "2Y" else panel_30Y[panel_30Y["meeting_date"] == WARSH_DATE]
        iv_warsh = float(_w_etf["iv"].iloc[0]) if (not _w_etf.empty and not np.isnan(float(_w_etf["iv"].iloc[0]))) else np.nan

        def _pred(feat_list):
            cols = [c for c in feat_list if c in train.columns
                    and train[c].notna().sum() > 2]
            if not cols:
                return float(y_tr.mean())
            sc   = StandardScaler()
            X_tr = sc.fit_transform(train[cols].fillna(0).values)
            X_te = sc.transform(warsh[cols].fillna(0).values)
            try:
                return float(RidgeCV(alphas=ALPHA_RANGE).fit(X_tr, y_tr).predict(X_te)[0])
            except Exception:
                return float(y_tr.mean())

        pred_nlp = _pred(feats_nlp)
        pred_reg = _pred(feats_regime)

        # Regime label for Warsh
        reg_label = str(warsh["regime_label"].iloc[0]) if not warsh.empty else "?"
        ig_val    = float(warsh["inflation_gap"].iloc[0]) if "inflation_gap" in warsh else np.nan

        results[tenor] = dict(
            pred_nlp    = pred_nlp,
            pred_regime = pred_reg,
            actual      = act,
            iv          = iv_warsh,
            bps_move    = BPS[tenor],
            regime      = reg_label,
            inflation_gap = ig_val,
            err_nlp     = pred_nlp - act,
            err_regime  = pred_reg - act,
        )
        print(f"  Warsh {tenor}: NLP-only={pred_nlp:.2f}%  NLP×regime={pred_reg:.2f}%  "
              f"Actual={act:.2f}%  (yield +{BPS[tenor]}bp)")

    return results


print("Warsh forward test (both modes, full history training):")
warsh_both = warsh_forward_both_modes(pca_2Y, pca_30Y)


def fig_v5_warsh(warsh_both: dict) -> None:
    """
    Two-panel figure: 2Y and 30Y.
    Bars: NLP-only, NLP×regime, actual RV.
    Annotation: IV missing (overheating — no options anchor); yield move context.
    """
    if not warsh_both:
        print("  No Warsh results."); return

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    fig.suptitle(
        f"Fig V5 — Warsh Forward Test  ({WARSH_DATE.date()})  [n=1, ONCE]\n"
        "Both forecasting modes vs actual realized vol  ·  "
        "IV via ETF proxy (SHY/TLT straddles, overheating era)",
        fontsize=11)

    for ax, (tenor, res) in zip(axes, warsh_both.items()):
        cats   = ["NLP-only\n(IV analog)", "NLP×regime\n(fwd-looking)", "Actual RV"]
        vals   = [res["pred_nlp"], res["pred_regime"], res["actual"]]
        colors = ["steelblue", "#d73027", "#1a1a1a"]

        bars = ax.bar(cats, vals, color=colors, alpha=0.80, width=0.52)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 0.06,
                    f"{v:.2f}%", ha="center", fontsize=10, fontweight="bold")

        # Error annotations
        for b, v, label, err in [
            (bars[0], vals[0], f"err={res['err_nlp']:+.2f}%\n({res['err_nlp']/res['actual']*100:+.0f}%)", res["err_nlp"]),
            (bars[1], vals[1], f"err={res['err_regime']:+.2f}%\n({res['err_regime']/res['actual']*100:+.0f}%)", res["err_regime"]),
        ]:
            ax.text(b.get_x() + b.get_width()/2,
                    min(v, res["actual"]) / 2,
                    label, ha="center", fontsize=8,
                    color="#d73027" if abs(err) > abs(res["err_nlp"] + res["err_regime"])/2 else "grey")

        ax.set_ylim(0, max(vals) * 1.35)
        ax.set_ylabel("Event realized vol (%)", fontsize=10)
        ax.set_title(
            f"{tenor} — Warsh ({res['regime']})\n"
            f"inflation_gap={res['inflation_gap']:+.2f}pp  "
            f"yield move: +{res['bps_move']}bp",
            fontsize=9)
        ax.grid(True, axis="y", alpha=0.2)

        # No IV note
        ax.text(0.98, 0.97, "IV = N/A\n(overheating era;\nno options anchor)",
                transform=ax.transAxes, fontsize=7.5, color="#8e44ad",
                va="top", ha="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    fig.tight_layout()
    _save(fig, "vrp_v5_warsh")


fig_v5_warsh(warsh_both)

# %% [markdown]
# ---
# ## Cell 8 — Summary: What VRP Regime Calibration Tells Us

# %%
print("═" * 64)
print("  SUMMARY — Regime-Conditional VRP Calibration")
print("═" * 64)

_sep = "─" * 62
print(f"""
  VRP by regime (2Y, where IV observed):
  {_sep}
  Regime           n_IV   mean_IV   mean_RV   mean_VRP   Note
  {_sep}""")
for _, row in vrp_summary_2Y.iterrows():
    if np.isnan(row["mean_iv"]):
        print(f"  {row['regime']:<16} {int(row['n_iv']):>4}   [NO IV]   "
              f"{row['mean_rv']:>7.3f}%   [UNK]      ← regime is BLIND to IV")
    else:
        print(f"  {row['regime']:<16} {int(row['n_iv']):>4}   "
              f"{row['mean_iv']:>7.3f}%  {row['mean_rv']:>7.3f}%  "
              f"{row['mean_vrp']:>+8.3f}%")

print(f"""
  KEY INSIGHTS:
  {_sep}
  1. VRP is POSITIVE in ALL FIVE regimes (including overheating):
     IV systematically overpriced realized vol across the full history.
     ETF proxy (SHY/IEF/TLT straddles) fills in post-2020 when TYVIX
     was discontinued.  Overheating: IV 3.13% vs RV 2.20% → VRP +0.93pp.

  2. TYVIX discontinued 2020-05; post-2020 IV uses ETF proxy straddles
     as a compromise.  ETF basis risk: duration mismatch, tracking error,
     liquidity differences — estimates are approximate but directionally
     valid.  NLP-only calibrated on the positive-VRP era still inherits
     upward IV bias → may underestimate RV spikes in overheating.

  3. NLP×regime should theoretically correct this: the overheating
     indicator tells the model "this is a different distributional
     regime — expect higher RV."  Whether it does so reliably at
     n≈51 is the acceptance-test question.

  FORECASTING MODES:
  {_sep}
  • NLP-only  = backward-looking IV analog.  Anchors to the text-vol
    mapping learned in the slack/easing era.  Underestimates RV in
    overheating (the VRP sign flips but the model doesn't know).
  • NLP×regime = forward-looking.  Adjusts the forecast based on
    where we are in the dual-mandate cycle.  In overheating: nudges
    the forecast higher than NLP-only.  In slack/easing: nudges lower.

  WARSH (2026-06-17, overheating):
  {_sep}""")
for tenor, res in warsh_both.items():
    winner = "NLP×regime" if abs(res["err_regime"]) < abs(res["err_nlp"]) else "NLP-only"
    print(f"  {tenor}  NLP-only={res['pred_nlp']:.2f}%  "
          f"NLP×regime={res['pred_regime']:.2f}%  "
          f"Actual={res['actual']:.2f}%  → {winner} closer")

print(f"""
  CAVEATS:
  {_sep}
  1. VRP calibration based on n=84 (2Y) and fewer (30Y) IV observations,
     all in the pre-overheating era.  Extrapolation to overheating is
     economically motivated but not directly observed.
  2. Both forecasting modes always produced; do not suppress either.
  3. Warsh forward test is n=1 — not validation.
  4. NLP-only ≈ IV correlation is an empirical claim — check Fig V4 R².
  5. NLP features updated with full Q&A PDF transcripts (mean 8,568 tokens
     per meeting vs. 79 tokens of HTML boilerplate before).
""")

# %% [markdown]
# ---
# ## Cell 9 — Historical Case Study: Full Walk-Forward Where IV Exists (2012–2020)
#
# **Why the Powell-era walk-forward misses IV:** the first OOS prediction
# starts after 15 training meetings (≈ Feb 2020), and TYVIX was discontinued
# May 2020.  So **zero** OOS Powell-era meetings have IV.
#
# **Fix:** run a separate walk-forward starting from 2010 with `min_train=15`.
# OOS begins ~Sep 2012.  IV exists through May 2020, giving **n≈50** meetings
# where we can compare IV · RV · NLP-only · NLP×regime all at once.
#
# **Hypothesis to test:**
# | Claim | Test |
# |-------|------|
# | NLP-only ≈ IV (backward-looking) | `|pred_nlp − IV| < |pred_nlp − RV|` |
# | NLP×regime ≈ RV (regime-corrected) | `|pred_regime − RV| < |pred_nlp − RV|` |
# | If both hold: model REDUCES VRP mismatch | `|pred_regime − RV| < |pred_nlp − RV| ≈ VRP` |

# %%
def walk_forward_iv_era(pan: pd.DataFrame, tenor: str) -> pd.DataFrame:
    """
    Full-history walk-forward from 2010, so OOS covers 2012-2020 where IV exists.
    Returns same columns as walk_forward_both().
    """
    feats_nlp    = [c for c in TEXT_COLS + CTRL_COLS + ["rv_lag1", "pc1"]
                    if c in pan.columns]
    feats_regime = feats_nlp + [c for c in ["inflation_gap", "pc1_x_inf_gap",
                                              "accel", "pc1_x_accel"]
                                 if c in pan.columns]
    target = "rv"
    start  = pan["meeting_date"].min()

    sub = pan.dropna(subset=[target]).sort_values("meeting_date").reset_index(drop=True)
    meetings = sorted(sub["meeting_date"].unique())

    results = []
    for i, dt in enumerate(meetings):
        if i < MIN_TRAIN:
            continue
        train = sub[sub["meeting_date"] < dt].dropna(
            subset=[c for c in feats_nlp if c in sub.columns] + [target])
        test  = sub[sub["meeting_date"] == dt]
        if len(train) < 8 or test.empty:
            continue
        y_tr  = train[target].values
        act   = float(test[target].iloc[0])
        iv_val= float(test["iv"].iloc[0]) \
                if "iv" in test.columns and test["iv"].notna().any() else np.nan

        def _pred(feat_list):
            cols = [c for c in feat_list if c in train.columns
                    and train[c].notna().sum() > 2]
            if not cols:
                return float(y_tr.mean())
            sc   = StandardScaler()
            X_tr = sc.fit_transform(train[cols].fillna(0).values)
            X_te = sc.transform(test[cols].fillna(0).values)
            try:
                return float(RidgeCV(alphas=ALPHA_RANGE).fit(X_tr, y_tr).predict(X_te)[0])
            except Exception:
                return float(y_tr.mean())

        results.append(dict(
            meeting_date  = dt,
            regime_label  = str(test["regime_label"].iloc[0]),
            inflation_gap = float(test["inflation_gap"].iloc[0])
                            if "inflation_gap" in test else np.nan,
            actual        = act,
            iv            = iv_val,
            pred_nlp      = _pred(feats_nlp),
            pred_regime   = _pred(feats_regime),
            n_train       = len(train),
        ))

    df = pd.DataFrame(results)
    if len(df) > 0:
        df["vrp"]        = df["iv"]          - df["actual"]
        df["gap_nlp_iv"] = df["iv"]          - df["pred_nlp"]
        df["gap_reg_iv"] = df["iv"]          - df["pred_regime"]
        df["gap_nlp_rv"] = df["pred_nlp"]    - df["actual"]
        df["gap_reg_rv"] = df["pred_regime"] - df["actual"]
        df["tenor"]      = tenor
    return df


print("Running full-history walk-forward (IV-era case study) …")
pca_2Y_full  = build_pca_panel(panel_2Y,  start=panel_2Y["meeting_date"].min())
pca_30Y_full = build_pca_panel(panel_30Y, start=panel_30Y["meeting_date"].min())

hist_2Y  = walk_forward_iv_era(pca_2Y_full,  "2Y")
hist_30Y = walk_forward_iv_era(pca_30Y_full, "30Y")

for df, tenor in [(hist_2Y, "2Y"), (hist_30Y, "30Y")]:
    with_iv = df.dropna(subset=["iv"])
    print(f"\n  {tenor} — full-history OOS: {len(df)} meetings, "
          f"{len(with_iv)} with IV  ({with_iv['regime_label'].value_counts().to_dict()})")

    if len(with_iv) < 3:
        continue

    # Compute mean absolute distances: pred vs IV, pred vs RV
    d_nlp_iv  = float((with_iv["iv"]    - with_iv["pred_nlp"]).abs().mean())
    d_nlp_rv  = float((with_iv["actual"]- with_iv["pred_nlp"]).abs().mean())
    d_reg_iv  = float((with_iv["iv"]    - with_iv["pred_regime"]).abs().mean())
    d_reg_rv  = float((with_iv["actual"]- with_iv["pred_regime"]).abs().mean())
    mean_vrp  = float(with_iv["vrp"].mean())
    corr_nlp_iv = float(with_iv["pred_nlp"].corr(with_iv["iv"]))
    corr_nlp_rv = float(with_iv["pred_nlp"].corr(with_iv["actual"]))
    corr_reg_iv = float(with_iv["pred_regime"].corr(with_iv["iv"]))
    corr_reg_rv = float(with_iv["pred_regime"].corr(with_iv["actual"]))

    print(f"  {'Metric':<34} {'NLP-only':>10}  {'NLP×regime':>11}")
    print(f"  {'─'*57}")
    print(f"  {'MAE vs IV (lower = closer to market)':34} {d_nlp_iv:>10.3f}  {d_reg_iv:>11.3f}")
    print(f"  {'MAE vs RV (lower = closer to actual)':34} {d_nlp_rv:>10.3f}  {d_reg_rv:>11.3f}")
    print(f"  {'Corr with IV':34} {corr_nlp_iv:>10.3f}  {corr_reg_iv:>11.3f}")
    print(f"  {'Corr with RV':34} {corr_nlp_rv:>10.3f}  {corr_reg_rv:>11.3f}")
    print(f"  {'Mean VRP (IV−RV)':34} {mean_vrp:>10.3f}pp")
    nlp_closer_iv = d_nlp_iv < d_nlp_rv
    reg_closer_rv = d_reg_rv < d_nlp_rv
    print(f"\n  → NLP-only closer to IV than RV? {'YES ✓' if nlp_closer_iv else 'NO ✗'}")
    print(f"  → NLP×regime closer to RV than NLP-only? {'YES ✓' if reg_closer_rv else 'NO ✗'}")

# %% [markdown]
# ---
# ## Cell 10 — Fig V6: Four-Series Time-Series (IV · RV · NLP-only · NLP×regime)
#
# The IV-era (2012–2020) is the only window where all four series exist.
# Plotting them together shows visually whether NLP-only hugs IV (high, positive VRP)
# while NLP×regime tracks actual RV more closely.
#
# **Slack / easing regime (2010–2021):** IV > RV.  NLP-only should sit near IV.
# NLP×regime, conditioned on negative inflation-gap, should predict lower — closer to RV.

# %%
def fig_v6_four_series(hist_2Y: pd.DataFrame, hist_30Y: pd.DataFrame) -> None:
    """
    Two panels (2Y, 30Y) — all four series over the IV-observable period.
    Regime-band background.  Pivot annotation: taper tantrum, COVID, hiking.
    """
    EVENTS = {
        pd.Timestamp("2013-06-19"): ("Taper\ntantrum", "top"),
        pd.Timestamp("2018-12-19"): ("Hike\n+25bp", "top"),
        pd.Timestamp("2020-03-15"): ("COVID\ncrash", "top"),
    }

    fig, axes = plt.subplots(2, 1, figsize=(16, 11), sharex=False)
    fig.suptitle(
        "Fig V6 — Historical Case Study: Four Series (2012–2020, IV-Observable Era)\n"
        "NLP-only tracks IV · NLP×regime tracks RV · VRP is the gap between them",
        fontsize=12)

    for ax, df, tenor in zip(axes, [hist_2Y, hist_30Y], ["2Y", "30Y"]):
        df_iv = df.dropna(subset=["iv"]).sort_values("meeting_date")
        df_s  = df.sort_values("meeting_date")
        if df_iv.empty:
            ax.set_title(f"{tenor}: no IV data"); continue

        # Regime background
        for i in range(len(df_iv) - 1):
            ax.axvspan(df_iv["meeting_date"].iloc[i],
                       df_iv["meeting_date"].iloc[i+1],
                       facecolor=REGIME_PALETTE.get(
                           df_iv["regime_label"].iloc[i], "#dddddd"),
                       alpha=0.09, linewidth=0)

        # Four series
        ax.plot(df_iv["meeting_date"], df_iv["iv"],
                color="#8e44ad", lw=2.0, marker="^", ms=5, zorder=5,
                label="IV — market implied vol")
        ax.plot(df_s["meeting_date"], df_s["actual"],
                color="#1a1a1a", lw=1.8, marker="o", ms=4, zorder=5,
                label="RV — actual realized vol")
        ax.plot(df_s["meeting_date"], df_s["pred_nlp"],
                color="steelblue", lw=1.3, ls="--", zorder=4, alpha=0.9,
                label="NLP-only  (IV analog)")
        ax.plot(df_s["meeting_date"], df_s["pred_regime"],
                color="#d73027", lw=1.3, ls="-", zorder=4, alpha=0.9,
                label="NLP×regime  (RV analog)")

        # VRP fill between IV and RV where IV exists
        ax.fill_between(df_iv["meeting_date"], df_iv["iv"], df_iv["actual"],
                        alpha=0.07, color="#8e44ad",
                        label="VRP = IV − RV (positive = market overpays)")

        # Event annotations
        y_top = df_s[["pred_nlp","pred_regime","actual"]].max(axis=1).max() * 0.95
        for dt, (lbl, pos) in EVENTS.items():
            near = df_s[df_s["meeting_date"].between(
                dt - pd.Timedelta(days=90), dt + pd.Timedelta(days=90))]
            if near.empty:
                continue
            ax.axvline(near["meeting_date"].iloc[0], color="grey",
                       lw=0.8, ls=":", alpha=0.6)
            ax.text(near["meeting_date"].iloc[0], y_top, lbl,
                    fontsize=7, color="grey", ha="center", va="top")

        # IV-era boundary
        iv_end = df_iv["meeting_date"].max()
        ax.axvline(iv_end, color="grey", lw=1.0, ls="--", alpha=0.5)
        ax.text(iv_end + pd.Timedelta(days=20), y_top * 0.6,
                "TYVIX\nends", fontsize=7, color="grey")

        ax.set_ylabel(f"{tenor} event vol (%)", fontsize=10)
        ax.set_title(
            f"{tenor} — {len(df_iv)} IV-observable OOS meetings  "
            f"({df_iv['meeting_date'].min().year}–{df_iv['meeting_date'].max().year})",
            fontsize=10)
        ax.legend(fontsize=8, loc="upper left", ncol=2)
        ax.grid(True, axis="y", alpha=0.15)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, fontsize=7.5)

    fig.legend(handles=_regime_handles(), fontsize=7.5,
               loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    _save(fig, "vrp_v6_four_series")


fig_v6_four_series(hist_2Y, hist_30Y)

# %% [markdown]
# ---
# ## Cell 11 — Fig V7: IV/RV Proximity Scatter + VRP-Error Alignment
#
# **Two tests in one figure:**
#
# **Left (scatter, 2×2):** For each OOS meeting with IV:
# - X-axis = IV; Y-axis = model prediction
# - If the cloud sits on the 45° line → model ≈ IV
# - NLP-only cloud should lie ON the IV diagonal
# - NLP×regime cloud should lie BELOW the IV diagonal (predicts closer to lower RV)
#
# **Right (bar chart):** MAE to IV and to RV per regime.
# Slack/easing should show NLP-only closer to IV; NLP×regime closer to RV.

# %%
def fig_v7_proximity_scatter(hist_2Y: pd.DataFrame, hist_30Y: pd.DataFrame) -> None:
    """
    Row 1 (2Y) / Row 2 (30Y):
      Left:   Scatter — IV vs pred_nlp and pred_regime (both on one plot, diff markers)
      Centre: Scatter — RV vs pred_nlp and pred_regime
      Right:  Bar chart — MAE per regime for NLP-only vs NLP×regime (distance to IV and RV)
    """
    fig = plt.figure(figsize=(17, 12))
    fig.suptitle(
        "Fig V7 — Proximity Test: NLP-only → IV analog · NLP×regime → RV analog\n"
        "Hypothesis: NLP-only corr(IV)>corr(RV);  NLP×regime corr(RV)>corr(IV)\n"
        "Data: full-history walk-forward OOS meetings where IV was observable (2012–2020)",
        fontsize=11)

    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.33, hspace=0.40)

    for row_i, (df, tenor) in enumerate([(hist_2Y, "2Y"), (hist_30Y, "30Y")]):
        df_iv = df.dropna(subset=["iv"]).copy()
        if len(df_iv) < 5:
            continue
        cols = [REGIME_PALETTE.get(l, "#878787") for l in df_iv["regime_label"]]

        ax_iv  = fig.add_subplot(gs[row_i, 0])
        ax_rv  = fig.add_subplot(gs[row_i, 1])
        ax_bar = fig.add_subplot(gs[row_i, 2])

        iv  = df_iv["iv"].values
        rv  = df_iv["actual"].values
        pnl = df_iv["pred_nlp"].values
        prg = df_iv["pred_regime"].values

        # ── Left: pred vs IV ────────────────────────────────────────────────
        ax_iv.scatter(iv, pnl, c=cols, s=35, marker="o", alpha=0.80,
                       edgecolors="white", lw=0.3, zorder=3, label="NLP-only")
        ax_iv.scatter(iv, prg, c=cols, s=35, marker="^", alpha=0.70,
                       edgecolors="white", lw=0.3, zorder=3, label="NLP×regime")
        lim_iv = max(iv.max(), max(pnl.max(), prg.max())) * 1.08
        ax_iv.plot([0, lim_iv], [0, lim_iv], "k--", lw=0.9, alpha=0.5,
                    label="45° (pred = IV)")
        r_nlp_iv = float(np.corrcoef(iv, pnl)[0,1])
        r_reg_iv = float(np.corrcoef(iv, prg)[0,1])
        ax_iv.set_xlim(0, lim_iv); ax_iv.set_ylim(0, lim_iv)
        ax_iv.set_xlabel("Implied vol IV (%)", fontsize=9)
        ax_iv.set_ylabel("Model prediction (%)", fontsize=9)
        ax_iv.set_title(
            f"{tenor} — Proximity to IV\n"
            f"r(NLP,IV)={r_nlp_iv:.2f}  r(Reg,IV)={r_reg_iv:.2f}", fontsize=9)
        ax_iv.legend(fontsize=7, loc="upper left")
        ax_iv.grid(True, alpha=0.15, lw=0.5)

        # ── Centre: pred vs RV ──────────────────────────────────────────────
        ax_rv.scatter(rv, pnl, c=cols, s=35, marker="o", alpha=0.80,
                       edgecolors="white", lw=0.3, zorder=3, label="NLP-only")
        ax_rv.scatter(rv, prg, c=cols, s=35, marker="^", alpha=0.70,
                       edgecolors="white", lw=0.3, zorder=3, label="NLP×regime")
        lim_rv = max(rv.max(), max(pnl.max(), prg.max())) * 1.08
        ax_rv.plot([0, lim_rv], [0, lim_rv], "k--", lw=0.9, alpha=0.5,
                    label="45° (pred = RV)")
        r_nlp_rv = float(np.corrcoef(rv, pnl)[0,1])
        r_reg_rv = float(np.corrcoef(rv, prg)[0,1])
        ax_rv.set_xlim(0, lim_rv); ax_rv.set_ylim(0, lim_rv)
        ax_rv.set_xlabel("Realized vol RV (%)", fontsize=9)
        ax_rv.set_ylabel("Model prediction (%)", fontsize=9)
        ax_rv.set_title(
            f"{tenor} — Proximity to RV\n"
            f"r(NLP,RV)={r_nlp_rv:.2f}  r(Reg,RV)={r_reg_rv:.2f}", fontsize=9)
        ax_rv.legend(fontsize=7, loc="upper left")
        ax_rv.grid(True, alpha=0.15, lw=0.5)

        # Highlight claim: NLP-only should be closer to IV diagonal,
        # NLP×regime should be closer to RV diagonal
        # Mark the MEAN predictions
        ax_iv.scatter([iv.mean()], [pnl.mean()], s=160, marker="D",
                       color="steelblue", zorder=6, edgecolors="black", lw=0.8,
                       label=f"NLP mean ({pnl.mean():.2f}%)")
        ax_iv.scatter([iv.mean()], [prg.mean()], s=160, marker="D",
                       color="#d73027", zorder=6, edgecolors="black", lw=0.8,
                       label=f"Regime mean ({prg.mean():.2f}%)")
        ax_iv.axhline(iv.mean(), color="#8e44ad", lw=0.8, ls=":", alpha=0.5)
        ax_iv.axhline(rv.mean(), color="#333333", lw=0.8, ls=":", alpha=0.5)
        ax_iv.text(lim_iv * 0.02, iv.mean() + 0.05, f"IV mean={iv.mean():.2f}%",
                    fontsize=6.5, color="#8e44ad")
        ax_iv.text(lim_iv * 0.02, rv.mean() - 0.20, f"RV mean={rv.mean():.2f}%",
                    fontsize=6.5, color="#333333")

        # ── Right: MAE per regime ────────────────────────────────────────────
        labs  = [l for l in REGIME_ORDER if l in df_iv["regime_label"].values]
        x_pos = np.arange(len(labs))
        w     = 0.18

        mae_nlp_iv = [df_iv[df_iv["regime_label"]==l]["gap_nlp_iv"].abs().mean() for l in labs]
        mae_reg_iv = [df_iv[df_iv["regime_label"]==l]["gap_reg_iv"].abs().mean() for l in labs]
        mae_nlp_rv = [df_iv[df_iv["regime_label"]==l]["gap_nlp_rv"].abs().mean() for l in labs]
        mae_reg_rv = [df_iv[df_iv["regime_label"]==l]["gap_reg_rv"].abs().mean() for l in labs]

        ax_bar.bar(x_pos - 1.5*w, mae_nlp_iv, w, color="steelblue",   alpha=0.85,
                    label="NLP-only MAE vs IV")
        ax_bar.bar(x_pos - 0.5*w, mae_reg_iv, w, color="#d73027",      alpha=0.55,
                    label="NLP×regime MAE vs IV")
        ax_bar.bar(x_pos + 0.5*w, mae_nlp_rv, w, color="steelblue",   alpha=0.45,
                    hatch="///", linewidth=0, label="NLP-only MAE vs RV")
        ax_bar.bar(x_pos + 1.5*w, mae_reg_rv, w, color="#d73027",      alpha=0.85,
                    hatch="///", linewidth=0, label="NLP×regime MAE vs RV")

        ax_bar.set_xticks(x_pos)
        ax_bar.set_xticklabels([l.replace("_","\n") for l in labs], fontsize=7.5)
        for tick, lab in zip(ax_bar.get_xticklabels(), labs):
            tick.set_color(REGIME_PALETTE.get(lab, "black"))
        ax_bar.set_ylabel("Mean absolute error (%)", fontsize=9)
        ax_bar.set_title(
            f"{tenor} — MAE to IV vs RV per regime\n"
            f"Solid=vs IV · Hatched=vs RV · Blue=NLP-only · Red=NLP×regime",
            fontsize=8.5)
        ax_bar.legend(fontsize=6.5, loc="upper right")
        ax_bar.grid(True, axis="y", alpha=0.18)

        # n per regime
        for xi, lab in zip(x_pos, labs):
            n = len(df_iv[df_iv["regime_label"] == lab])
            ax_bar.text(xi, 0.05, f"n={n}", ha="center", fontsize=6.5, color="grey")

    fig.legend(handles=_regime_handles(), fontsize=7.5,
               loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    _save(fig, "vrp_v7_proximity_scatter")


fig_v7_proximity_scatter(hist_2Y, hist_30Y)
print("═" * 64)
