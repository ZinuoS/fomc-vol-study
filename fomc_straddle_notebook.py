# %% [markdown]
# # FOMC Event-Straddle — Interactive Simulation Notebook
#
# Two input modes (switchable in Section 1):
# - **Backtest / Public** — loads from `fomc_features.parquet` + `fomc_backtest_results.parquet`
# - **Bloomberg** — pulls live F0, σ_N via bql or xbbg on company device
#
# Three vol signals compared side-by-side in the MC:
# | Signal | Source | signal_mult |
# |--------|--------|-------------|
# | Bloomberg implied | Market | 1.0 (baseline) |
# | NLP feature Ridge | `fomc_features.parquet` | nlp_pred / implied_event_sd |
# | Word-bag Ridge | `fomc_statements.parquet` | word_pred / implied_event_sd |
#
# **Files to copy to company device:**
# `fomc_straddle_sim.py`, `fomc_straddle_mc.py`,
# `fomc_features.parquet`, `fomc_backtest_results.parquet`
#
# ---

# %% ── 0. CONFIG ──────────────────────────────────────────────────────────────

from pathlib import Path

# ── Parquet paths ─────────────────────────────────────────────────────────────
NLP_PARQUET       = Path("fomc_features.parquet")
BACKTEST_PARQUET  = Path("fomc_backtest_output/fomc_backtest_results.parquet")

# ── Bloomberg tickers (company device only) ───────────────────────────────────
BBG_FUTURE_TICKER  = "TYU6 Comdty"    # update to front-month e.g. TYZ6, TYH7
BBG_NORMAL_VOL_TKR = None             # e.g. "TYU6C 112 Comdty" normal vol field
BBG_STRADDLE_TKR   = None             # or the ATM straddle ticker to invert

# ── Default market assumptions (manual fallback) ──────────────────────────────
DEFAULT_F0              = 112.00   # TY future price — pull from live screen
DEFAULT_SIGMA_N         = 0.95     # ATM normal vol (pts/yr) — from OMON
DEFAULT_IMPLIED_EVENT_SD = 0.45   # FOMC-day σ extracted from vol-kink (pts)
DEFAULT_DTE             = 9        # trading days to expiry at entry
DEFAULT_YIELD_PCT       = 4.5      # current 10Y yield (%) — for unit conversion
DEFAULT_DV01_PER_BP     = 0.065   # TY pts per 1bp yield move

# ── MC defaults ───────────────────────────────────────────────────────────────
DEFAULT_CONTRACTS   = 1000
DEFAULT_N_PATHS     = 5000
DEFAULT_VOL_CRUSH   = 0.28
DEFAULT_AUCTION_MULT = 1.6
DEFAULT_BAND_PTS    = 0.05
DEFAULT_COST_TICKS  = 0.5

print("Config loaded.")

# %% ── 1. IMPORTS ─────────────────────────────────────────────────────────────

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML

from fomc_straddle_sim import (TradeConfig, DayEvent, FOMCDataLayer,
                                straddle, run_path, base_schedule, event_study)
from fomc_straddle_mc  import (MarketInputs, MCConfig, BacktestInputs,
                                fetch_bloomberg_inputs, load_backtest_inputs,
                                signal_mult_from_backtest, compare_bloomberg_vs_nlp,
                                print_comparison, simulate, summarize, signal_sweep)

print("Imports OK.")

# %% ── 2. LOAD BACKTEST DATA ──────────────────────────────────────────────────

_nlp_df = None
_bt_df  = None

if NLP_PARQUET.exists():
    _nlp_df = pd.read_parquet(NLP_PARQUET)
    _nlp_df["meeting_date"] = pd.to_datetime(_nlp_df["meeting_date"])
    print(f"NLP features loaded: {_nlp_df.shape}  "
          f"({_nlp_df['meeting_date'].min().date()} → {_nlp_df['meeting_date'].max().date()})")
else:
    print(f"WARNING: {NLP_PARQUET} not found. Backtest mode will be limited.")

if BACKTEST_PARQUET.exists():
    _bt_df = pd.read_parquet(BACKTEST_PARQUET)
    _bt_df["meeting_date"] = pd.to_datetime(_bt_df["meeting_date"])
    print(f"Backtest results loaded: {_bt_df.shape}")
else:
    print(f"NOTE: {BACKTEST_PARQUET} not found — nlp_pred / word_pred unavailable. "
          "Run fomc_backtest.ipynb first.")

_meeting_dates = (FOMCDataLayer.available_meetings(str(NLP_PARQUET))
                  if NLP_PARQUET.exists() else [])

# %% ── 3. PANEL 1 — TRADE INPUTS WIDGET ──────────────────────────────────────
# Select data source, pick meeting date (backtest mode), or enter Bloomberg tickers.

_STYLE    = {"description_width": "160px"}
_LAYOUT_W = widgets.Layout(width="380px")
_LAYOUT_N = widgets.Layout(width="200px")

# ── Source selector ───────────────────────────────────────────────────────────
w_source = widgets.ToggleButtons(
    options=["Backtest / Public", "Bloomberg", "Manual"],
    value="Backtest / Public",
    description="Vol source:",
    style={"description_width": "80px"},
    layout=widgets.Layout(width="520px"),
)

# ── Backtest: meeting date dropdown ───────────────────────────────────────────
w_meeting = widgets.Dropdown(
    options=_meeting_dates or ["(no parquet)"],
    value=_meeting_dates[-1] if _meeting_dates else "(no parquet)",
    description="Meeting date:",
    style=_STYLE, layout=_LAYOUT_W,
)
w_signal_choice = widgets.ToggleButtons(
    options=["word-bag", "nlp-features", "actual (hindsight)"],
    value="word-bag",
    description="NLP signal:",
    style={"description_width": "80px"},
    layout=widgets.Layout(width="520px"),
)

# ── Market inputs (auto-filled or manual) ─────────────────────────────────────
w_F0      = widgets.FloatText(value=DEFAULT_F0,   description="F0 (TY price pts):",
                               style=_STYLE, layout=_LAYOUT_W)
w_sigma   = widgets.FloatText(value=DEFAULT_SIGMA_N, description="σ_N (pts/yr):",
                               style=_STYLE, layout=_LAYOUT_W)
w_ev_sd   = widgets.FloatText(value=DEFAULT_IMPLIED_EVENT_SD,
                               description="Implied event σ (pts):",
                               style=_STYLE, layout=_LAYOUT_W)
w_dte     = widgets.IntSlider(value=DEFAULT_DTE, min=1, max=30, step=1,
                               description="DTE (trading days):",
                               style=_STYLE, layout=_LAYOUT_W)
w_yield   = widgets.FloatSlider(value=DEFAULT_YIELD_PCT, min=1.0, max=8.0, step=0.05,
                                 description="10Y yield (%):",
                                 readout_format=".2f",
                                 style=_STYLE, layout=_LAYOUT_W)
w_dv01    = widgets.FloatText(value=DEFAULT_DV01_PER_BP,
                               description="DV01/bp (TY pts):",
                               style=_STYLE, layout=_LAYOUT_W)

# ── Trade size ────────────────────────────────────────────────────────────────
w_contracts  = widgets.IntText(value=DEFAULT_CONTRACTS, description="Contracts:",
                                style=_STYLE, layout=_LAYOUT_N)
w_band       = widgets.FloatText(value=DEFAULT_BAND_PTS, description="Band (Δ units):",
                                  style=_STYLE, layout=_LAYOUT_N)
w_cost_ticks = widgets.FloatText(value=DEFAULT_COST_TICKS, description="Cost (ticks RT):",
                                  style=_STYLE, layout=_LAYOUT_N)

# ── Derived signal display (read-only) ────────────────────────────────────────
w_gk_actual  = widgets.Text(value="N/A", description="GK actual vol (pp):",
                              disabled=True, style=_STYLE, layout=_LAYOUT_W)
w_nlp_pred   = widgets.Text(value="N/A", description="NLP predicted (pp):",
                              disabled=True, style=_STYLE, layout=_LAYOUT_W)
w_word_pred  = widgets.Text(value="N/A", description="Word-bag pred (pp):",
                              disabled=True, style=_STYLE, layout=_LAYOUT_W)
w_sig_mult   = widgets.Text(value="N/A", description="→ signal_mult:",
                              disabled=True, style=_STYLE, layout=_LAYOUT_W)

# ── Auto-fill from parquet ────────────────────────────────────────────────────
def _autofill(_=None):
    """Populate derived fields from the selected meeting's parquet data."""
    if not NLP_PARQUET.exists():
        return
    try:
        data = FOMCDataLayer.load_meeting(
            str(NLP_PARQUET), w_meeting.value,
            str(BACKTEST_PARQUET) if BACKTEST_PARQUET.exists() else None,
        )
    except Exception as e:
        w_gk_actual.value = str(e)
        return

    gk  = data.get("gk_vol_10y", None)
    nlp = data.get("nlp_pred",   None)
    wrd = data.get("word_pred",  None)

    w_gk_actual.value = f"{gk:.2f}" if gk and not pd.isna(gk) else "N/A"
    w_nlp_pred.value  = f"{nlp:.2f}" if nlp and not pd.isna(nlp) else "N/A"
    w_word_pred.value = f"{wrd:.2f}" if wrd and not pd.isna(wrd) else "N/A"

    # Pick the active signal and compute signal_mult
    sig_map = {"word-bag": ("word_pred", wrd),
               "nlp-features": ("nlp_pred", nlp),
               "actual (hindsight)": ("gk_vol_10y", gk)}
    _, pp_val = sig_map.get(w_signal_choice.value, ("word_pred", wrd))
    if pp_val and not pd.isna(pp_val):
        pts  = FOMCDataLayer.gk_to_ty_pts(float(pp_val), w_yield.value, w_dv01.value)
        mult = pts / w_ev_sd.value if w_ev_sd.value > 0 else 0.0
        w_sig_mult.value = f"{mult:.3f}  ({pts:.3f} pts / {w_ev_sd.value:.3f} pts implied)"
    else:
        w_sig_mult.value = "N/A"


w_meeting.observe(_autofill, names="value")
w_signal_choice.observe(_autofill, names="value")
w_yield.observe(_autofill, names="value")
w_ev_sd.observe(_autofill, names="value")
_autofill()  # initial fill

# ── Layout ────────────────────────────────────────────────────────────────────
_panel1_backtest = widgets.VBox([
    widgets.HTML("<b>Backtest / Public data mode</b>"),
    w_meeting, w_signal_choice,
    widgets.HTML("<hr><b>Derived from parquet (read-only)</b>"),
    w_gk_actual, w_nlp_pred, w_word_pred, w_sig_mult,
])
_panel1_bbg = widgets.VBox([
    widgets.HTML("<b>Bloomberg mode — tickers set in CONFIG cell</b>"),
    widgets.HTML(f"<code>Future: {BBG_FUTURE_TICKER}</code><br>"
                 f"<code>Normal vol / straddle ticker: {BBG_NORMAL_VOL_TKR or BBG_STRADDLE_TKR or 'not set'}</code>"),
    widgets.HTML("Bloomberg data will be fetched when you click Run."),
])
_panel1_manual = widgets.VBox([
    widgets.HTML("<b>Manual override — edit the fields below directly</b>"),
])
_panel1_switcher = widgets.Output()

def _switch_source(_=None):
    with _panel1_switcher:
        clear_output()
        if w_source.value == "Backtest / Public":
            display(_panel1_backtest)
        elif w_source.value == "Bloomberg":
            display(_panel1_bbg)
        else:
            display(_panel1_manual)

w_source.observe(_switch_source, names="value")
_switch_source()

panel1 = widgets.VBox([
    widgets.HTML("<h3>Panel 1 — Trade Inputs</h3>"),
    w_source,
    _panel1_switcher,
    widgets.HTML("<hr><b>Market inputs (override as needed)</b>"),
    widgets.HBox([
        widgets.VBox([w_F0, w_sigma, w_ev_sd, w_dte]),
        widgets.VBox([w_yield, w_dv01, w_contracts]),
        widgets.VBox([w_band, w_cost_ticks]),
    ]),
], layout=widgets.Layout(border="1px solid #ddd", padding="12px", margin="6px"))

display(panel1)

# %% ── 4. PANEL 2 — SCHEDULE BUILDER ─────────────────────────────────────────
# Build the pre-FOMC path parametrically (no hardcoded Jul dates).

w_n_diffuse     = widgets.IntSlider(value=4, min=0, max=10, description="Quiet days:",
                                     style=_STYLE, layout=_LAYOUT_W)
w_n_auction     = widgets.IntSlider(value=2, min=0, max=5,  description="Auction days:",
                                     style=_STYLE, layout=_LAYOUT_W)
w_diffuse_move  = widgets.FloatSlider(value=0.04, min=0.0, max=0.5, step=0.01,
                                       description="Quiet move (pts):",
                                       readout_format=".3f",
                                       style=_STYLE, layout=_LAYOUT_W)
w_auction_move  = widgets.FloatSlider(value=0.07, min=0.0, max=0.5, step=0.01,
                                       description="Auction move (pts):",
                                       readout_format=".3f",
                                       style=_STYLE, layout=_LAYOUT_W)
w_fomc_move     = widgets.FloatSlider(value=0.55, min=-2.0, max=2.0, step=0.05,
                                       description="FOMC base move (pts):",
                                       readout_format=".3f",
                                       style=_STYLE, layout=_LAYOUT_W)
w_vol_crush     = widgets.FloatSlider(value=DEFAULT_VOL_CRUSH, min=0.0, max=1.0, step=0.01,
                                       description="Vol crush at FOMC:",
                                       readout_format=".3f",
                                       style=_STYLE, layout=_LAYOUT_W)

_sched_out = widgets.Output()

def _preview_schedule(_=None):
    sched = FOMCDataLayer.build_schedule(
        n_diffuse=w_n_diffuse.value, n_auction=w_n_auction.value,
        diffuse_move=w_diffuse_move.value, auction_move=w_auction_move.value,
        fomc_move=w_fomc_move.value, vol_crush=w_vol_crush.value,
    )
    with _sched_out:
        clear_output()
        rows = [f"<tr><td>{e.label}</td><td>{e.move_pts:+.3f}</td>"
                f"<td>{e.implied_change:+.3f}</td><td>{e.hedge}</td>"
                f"<td>{'✓' if e.is_expiry else ''}</td></tr>"
                for e in sched]
        display(HTML(
            "<table border='1' cellpadding='4' style='border-collapse:collapse'>"
            "<tr><th>Day</th><th>Move (pts)</th><th>Δσ</th><th>Hedge</th><th>Expiry</th></tr>"
            + "".join(rows) + "</table>"
        ))

for w in (w_n_diffuse, w_n_auction, w_diffuse_move, w_auction_move, w_fomc_move, w_vol_crush):
    w.observe(_preview_schedule, names="value")
_preview_schedule()

panel2 = widgets.VBox([
    widgets.HTML("<h3>Panel 2 — Schedule Builder</h3>"),
    widgets.HBox([
        widgets.VBox([w_n_diffuse, w_n_auction, w_diffuse_move]),
        widgets.VBox([w_auction_move, w_fomc_move, w_vol_crush]),
    ]),
    widgets.HTML("<b>Schedule preview:</b>"),
    _sched_out,
], layout=widgets.Layout(border="1px solid #ddd", padding="12px", margin="6px"))

display(panel2)

# %% ── 5. PANEL 3 — MC PARAMETERS ────────────────────────────────────────────

w_n_paths      = widgets.IntSlider(value=DEFAULT_N_PATHS, min=1000, max=50000,
                                    step=1000, description="N paths:",
                                    style=_STYLE, layout=_LAYOUT_W)
w_auction_mult = widgets.FloatSlider(value=DEFAULT_AUCTION_MULT, min=1.0, max=4.0,
                                      step=0.1, description="Auction vol mult:",
                                      readout_format=".2f",
                                      style=_STYLE, layout=_LAYOUT_W)
w_seed         = widgets.IntText(value=42, description="RNG seed:",
                                  style=_STYLE, layout=_LAYOUT_N)
w_sweep_lo     = widgets.FloatSlider(value=0.5, min=0.1, max=1.0, step=0.05,
                                      description="Sweep min mult:",
                                      readout_format=".2f",
                                      style=_STYLE, layout=_LAYOUT_W)
w_sweep_hi     = widgets.FloatSlider(value=3.0, min=1.0, max=5.0, step=0.1,
                                      description="Sweep max mult:",
                                      readout_format=".2f",
                                      style=_STYLE, layout=_LAYOUT_W)
w_sweep_steps  = widgets.IntSlider(value=10, min=3, max=30, step=1,
                                    description="Sweep steps:",
                                    style=_STYLE, layout=_LAYOUT_W)

panel3 = widgets.VBox([
    widgets.HTML("<h3>Panel 3 — Monte Carlo Parameters</h3>"),
    widgets.HBox([
        widgets.VBox([w_n_paths, w_auction_mult, w_seed]),
        widgets.VBox([w_sweep_lo, w_sweep_hi, w_sweep_steps]),
    ]),
], layout=widgets.Layout(border="1px solid #ddd", padding="12px", margin="6px"))

display(panel3)

# %% ── 6. RUN — SIMULATION ENGINE ────────────────────────────────────────────

_run_out    = widgets.Output()
_chart_out  = widgets.Output()

btn_run_both   = widgets.Button(description="Run Bloomberg vs NLP",
                                 button_style="primary",
                                 layout=widgets.Layout(width="220px", height="36px"))
btn_run_det    = widgets.Button(description="Run Deterministic Path",
                                 button_style="info",
                                 layout=widgets.Layout(width="220px", height="36px"))
btn_sweep      = widgets.Button(description="Signal Sweep Chart",
                                 button_style="warning",
                                 layout=widgets.Layout(width="220px", height="36px"))
_status        = widgets.HTML("")


def _get_market_inputs() -> MarketInputs:
    """Assemble MarketInputs from current widget values + chosen source."""
    src = w_source.value
    if src == "Bloomberg":
        return fetch_bloomberg_inputs(
            future_ticker     = BBG_FUTURE_TICKER,
            normal_vol_ticker = BBG_NORMAL_VOL_TKR,
            atm_straddle_ticker = BBG_STRADDLE_TKR,
            days_to_expiry    = w_dte.value,
            implied_event_sd  = w_ev_sd.value,
            manual            = MarketInputs(F0=w_F0.value, sigma_N0=w_sigma.value,
                                             implied_event_sd=w_ev_sd.value,
                                             days_to_expiry=w_dte.value, source="manual"),
        )
    return MarketInputs(F0=w_F0.value, sigma_N0=w_sigma.value,
                        implied_event_sd=w_ev_sd.value,
                        days_to_expiry=w_dte.value,
                        source=src.lower())


def _get_mc_config(signal_mult: float = 1.0) -> MCConfig:
    sched = FOMCDataLayer.build_schedule(
        n_diffuse=w_n_diffuse.value, n_auction=w_n_auction.value,
        diffuse_move=w_diffuse_move.value, auction_move=w_auction_move.value,
        fomc_move=w_fomc_move.value, vol_crush=w_vol_crush.value,
    )
    calendar = [(e.label, ("fomc" if e.is_expiry else
                            "auction" if e.hedge == "bracket" else "diffuse"))
                for e in sched]
    return MCConfig(contracts=w_contracts.value, band_pts=w_band.value,
                    cost_ticks=w_cost_ticks.value, auction_mult=w_auction_mult.value,
                    vol_crush=w_vol_crush.value, signal_mult=signal_mult,
                    n_paths=w_n_paths.value, seed=w_seed.value, calendar=calendar)


def _plot_comparison(comp: dict) -> None:
    """Two-panel comparison chart: P&L distributions + signal sweep."""
    fig = plt.figure(figsize=(16, 6))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.3)

    # ── Left: P&L distribution overlay ───────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    colors = {"bbg": "#2471a3", "nlp": "#c0392b", "actual": "#27ae60"}
    labels = {"bbg": "Bloomberg (signal=1.0)", "nlp": "NLP signal", "actual": "Actual realized"}
    for key in ("bbg", "nlp", "actual"):
        res = comp.get(f"res_{key}")
        if res is None:
            continue
        s   = comp[key]
        t   = res["totals"] / 1e6   # in $M
        ax1.hist(t, bins=60, alpha=0.5, color=colors[key],
                 label=f"{labels[key]}\nE[P&L]=${s['mean']/1e6:+.2f}M  P(profit)={s['prob_profit']:.0%}",
                 density=True)
        ax1.axvline(np.mean(t), color=colors[key], lw=1.5, linestyle="--")

    ax1.axvline(0, color="#333", lw=0.8)
    ax1.set_xlabel("Total P&L ($M)", fontsize=10)
    ax1.set_ylabel("Density", fontsize=10)
    ax1.set_title("P&L Distribution — Bloomberg vs NLP signal\n"
                  f"Entry premium: ${comp['res_bbg']['entry_premium']/1e6:.2f}M  "
                  f"Contracts: {w_contracts.value:,}", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(lw=0.4, alpha=0.4)

    # ── Right: signal sweep (P(profit) + E[P&L] vs signal_mult) ─────────────
    ax2r = fig.add_subplot(gs[1])
    ax2l = ax2r.twinx()

    mkt  = comp["mkt"]
    cfg0 = _get_mc_config(1.0)
    mults = np.linspace(w_sweep_lo.value, w_sweep_hi.value, w_sweep_steps.value)
    sweep = signal_sweep(mkt, cfg0, mults)
    sm_vals   = [m for m, _ in sweep]
    pp_vals   = [s["prob_profit"] for _, s in sweep]
    mean_vals = [s["mean"] / 1e6  for _, s in sweep]

    ax2l.plot(sm_vals, mean_vals, color="#c0392b", lw=2, label="E[P&L] ($M)")
    ax2l.axhline(0, color="#c0392b", lw=0.6, linestyle=":")
    ax2r.plot(sm_vals, pp_vals, color="#2471a3", lw=2, linestyle="--", label="P(profit)")
    ax2r.axhline(0.5, color="#2471a3", lw=0.6, linestyle=":")

    # Mark the NLP signal_mult
    nlp_m = comp["edge"]["signal_mult"]
    ax2r.axvline(nlp_m, color="#c0392b", lw=1.2, linestyle=":", alpha=0.8)
    ax2r.text(nlp_m + 0.03, 0.52, f"NLP\n{nlp_m:.2f}x", fontsize=8, color="#c0392b")

    ax2l.set_xlabel("signal_mult  (realized FOMC σ / implied)", fontsize=10)
    ax2l.set_ylabel("E[P&L] ($M)", color="#c0392b", fontsize=10)
    ax2r.set_ylabel("P(profit)", color="#2471a3", fontsize=10)
    ax2r.set_ylim(0, 1)
    ax2r.set_title("Signal Sweep — how edge scales with NLP signal strength", fontsize=10)
    lines1, lbl1 = ax2l.get_legend_handles_labels()
    lines2, lbl2 = ax2r.get_legend_handles_labels()
    ax2r.legend(lines1 + lines2, lbl1 + lbl2, fontsize=8, loc="upper left")
    ax2r.grid(lw=0.4, alpha=0.4)

    plt.suptitle(f"FOMC Straddle Sim  ·  {comp['bt'].meeting_date if comp.get('bt') else ''}  "
                 f"·  {w_n_paths.value:,} paths",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.savefig("fomc_straddle_output.png", dpi=150, bbox_inches="tight")
    plt.show()


def on_run_both(b):
    _status.value = "<i>Running Bloomberg vs NLP comparison…</i>"
    with _run_out:
        clear_output()
        try:
            mkt = _get_market_inputs()
            _status.value = f"<i>Market inputs [{mkt.source}]: F0={mkt.F0}  σ_N={mkt.sigma_N0:.3f}</i>"

            # Load backtest inputs (NLP signal)
            bt = None
            if NLP_PARQUET.exists():
                sig_map = {"word-bag": "word", "nlp-features": "nlp",
                           "actual (hindsight)": "actual"}
                bt = load_backtest_inputs(
                    str(NLP_PARQUET), w_meeting.value,
                    str(BACKTEST_PARQUET) if BACKTEST_PARQUET.exists() else None,
                    yield_pct=w_yield.value, dv01_per_bp=w_dv01.value,
                )
                use_sig = sig_map.get(w_signal_choice.value, "word")
            else:
                # No parquet: dummy BacktestInputs (signal_mult=1.0 only)
                bt = BacktestInputs(
                    meeting_date="N/A", gk_vol_actual_pp=0, nlp_predicted_pp=0,
                    word_predicted_pp=0, gk_actual_pts=0, nlp_predicted_pts=0,
                    word_predicted_pts=0, yield_pct=w_yield.value,
                    dv01_per_bp=w_dv01.value,
                )
                use_sig = "actual"

            cfg  = _get_mc_config()
            comp = compare_bloomberg_vs_nlp(mkt, bt, cfg, use_signal=use_sig)
            print_comparison(comp)

        except Exception as e:
            import traceback
            print(f"ERROR: {e}")
            traceback.print_exc()
            _status.value = f"<b style='color:red'>Error: {e}</b>"
            return

    with _chart_out:
        clear_output()
        _plot_comparison(comp)

    _status.value = "<b style='color:green'>Done. Chart saved → fomc_straddle_output.png</b>"


def on_run_det(b):
    """Run the deterministic path simulator (fomc_straddle_sim.run_path)."""
    _status.value = "<i>Running deterministic path…</i>"
    with _run_out:
        clear_output()
        try:
            cfg   = FOMCDataLayer.build_trade_config(
                F0=w_F0.value, sigma_N0=w_sigma.value,
                days_to_expiry=w_dte.value, contracts=w_contracts.value,
                band_pts=w_band.value, cost_ticks=w_cost_ticks.value,
            )
            sched = FOMCDataLayer.build_schedule(
                n_diffuse=w_n_diffuse.value, n_auction=w_n_auction.value,
                diffuse_move=w_diffuse_move.value, auction_move=w_auction_move.value,
                fomc_move=w_fomc_move.value, vol_crush=w_vol_crush.value,
            )
            print("=" * 70)
            print("DETERMINISTIC PATH")
            print("=" * 70)
            run_path(cfg, sched, verbose=True)
            event_study(cfg)
        except Exception as e:
            import traceback
            print(f"ERROR: {e}")
            traceback.print_exc()
    _status.value = "<b style='color:green'>Deterministic path done.</b>"


def on_sweep(b):
    """Standalone signal sweep chart."""
    _status.value = "<i>Running signal sweep…</i>"
    with _chart_out:
        clear_output()
        try:
            mkt   = _get_market_inputs()
            cfg0  = _get_mc_config(1.0)
            mults = np.linspace(w_sweep_lo.value, w_sweep_hi.value, w_sweep_steps.value)
            sweep = signal_sweep(mkt, cfg0, mults)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            sm_vals   = [m for m, _ in sweep]
            pp_vals   = [s["prob_profit"] for _, s in sweep]
            mean_vals = [s["mean"] / 1e6   for _, s in sweep]
            p5_vals   = [s["p5"] / 1e6     for _, s in sweep]
            p95_vals  = [s["p95"] / 1e6    for _, s in sweep]

            ax1.plot(sm_vals, mean_vals, "r-", lw=2, label="E[P&L]")
            ax1.fill_between(sm_vals, p5_vals, p95_vals, alpha=0.2, color="red", label="p5–p95")
            ax1.axhline(0, color="#333", lw=0.8)
            ax1.set_xlabel("signal_mult"); ax1.set_ylabel("P&L ($M)")
            ax1.set_title("E[P&L] and 90% interval vs signal_mult"); ax1.legend()
            ax1.grid(lw=0.4, alpha=0.4)

            ax2.plot(sm_vals, pp_vals, "b-", lw=2)
            ax2.axhline(0.5, color="#333", lw=0.8, linestyle="--")
            ax2.set_xlabel("signal_mult"); ax2.set_ylabel("P(profit)")
            ax2.set_title("Probability of profit vs signal_mult")
            ax2.set_ylim(0, 1); ax2.grid(lw=0.4, alpha=0.4)

            plt.suptitle(f"Signal Sweep  ·  {w_n_paths.value:,} paths  "
                         f"·  F0={mkt.F0}  σ_N={mkt.sigma_N0:.3f}  "
                         f"implied event σ={mkt.implied_event_sd:.3f}",
                         fontsize=11, fontweight="bold")
            plt.tight_layout()
            plt.savefig("fomc_sweep_chart.png", dpi=150, bbox_inches="tight")
            plt.show()
            _status.value = "<b style='color:green'>Sweep done → fomc_sweep_chart.png</b>"
        except Exception as e:
            import traceback
            print(f"ERROR: {e}")
            traceback.print_exc()
            _status.value = f"<b style='color:red'>Error: {e}</b>"


btn_run_both.on_click(on_run_both)
btn_run_det.on_click(on_run_det)
btn_sweep.on_click(on_sweep)

panel4 = widgets.VBox([
    widgets.HTML("<h3>Panel 4 — Run</h3>"),
    widgets.HBox([btn_run_both, btn_run_det, btn_sweep]),
    _status,
    widgets.HTML("<hr><b>Text output:</b>"),
    _run_out,
    widgets.HTML("<b>Charts:</b>"),
    _chart_out,
], layout=widgets.Layout(border="1px solid #ddd", padding="12px", margin="6px"))

display(panel4)
