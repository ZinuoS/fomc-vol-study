# %% [markdown]
# # FOMC Vol-Spread Steepener — P&L Simulation & Position Sizer
#
# **Trade thesis (updated v2)**: Long ATM straddle on ZT (2Y futures) +
# Short ATM straddle on ZB (30Y futures).  The GapSpread model predicts
# the 2Y event vol is underpriced relative to 30Y — buy front, sell long.
#
# At `z_spread = 0` (no edge) the simulation breaks even after costs.
# At `z_spread > 0` (buy-front signal) the 2Y FOMC jump is amplified and
# both the long-leg sizing and the simulated jump point in the same direction.
#
# ---
# ## Catalyst calendar (Warsh FOMC — June 2026)
#
# | Day | Event | Phase | Hedge action |
# |-----|-------|-------|--------------|
# | −10 bd | CPI print → **entry** | — | Buy ZT straddle, sell ZB straddle |
# | −2 bd | **5Y note auction** | 2 | Bracket both legs |
# | −1 bd | **7Y note auction** | 2 | Same bracket |
# | 0 (2pm) | **FOMC decision** | 3 | Drop hedges; keep convexity |
# | +1 bd | **Exit** | — | Unwind both straddles + residual hedges |
#
# ---
# ## Three-phase hedge logic
#
# **Phase 1** (run-up): daily delta-band hedge on both legs.
#
# **Phase 2** (auctions): event-bracketed hedge — flatten Δ before,
# hold through the directional reaction, flatten after.  Auction directional
# P&L nets to zero; only gamma/convexity survives.
#
# **Phase 3** (event → exit): no hedging.  Full convexity preserved through
# the FOMC jump; short 30Y leg has a stop-loss cap at `stop_loss_30y × premium`.

# %% ── Imports & signal ────────────────────────────────────────────────────────

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import copy

from fomc_delta_hedge_sim import (
    StraddleConfig, run_mc, size_position,
    print_report, band_sensitivity, plot_results,
    b76_straddle,
    load_spread_signal, acceptance_tests, ANNUAL,
    sigma_fomc_leg,
)

# Load the GapSpread signal from the spread model output
sig = load_spread_signal(Path("gap_forecasts_spread.parquet"))

print(f"\n── Warsh FOMC Signal (gap_forecasts_spread.parquet) ─────────────────")
print(f"  z_spread        = {sig['z_spread']:+.3f}  (GapSpread / sigma_f)")
print(f"  steepener_signal= {sig['steepener_signal']}")
print(f"  signal_mult_2y  = {sig['signal_mult_2y']:.3f}×")
print(f"  signal_mult_30y = {sig['signal_mult_30y']:.3f}×")

# %% ── Trade configuration ─────────────────────────────────────────────────────
#
# ZT Sep26 (2Y) ≈ 108.50, ZB Sep26 (30Y) ≈ 118.00.
# Event vols extracted from FOMC-expiry option vol kink:
#   sigma_iv_event     = 20.0%  (2Y: front is cheap = BUY)
#   sigma_iv_event_30y = 16.0%  (30Y: long end less reactive per Warsh RV data)
# sigma_quiet is AUTO-CALIBRATED so E[P&L | z=0] = −costs.

cfg = StraddleConfig(
    # ── 2Y front leg (LONG)
    F0=108.50,          K=108.50,
    iv_entry=0.082,     iv_exit=0.062,
    sigma_iv_event=0.200,
    # ── Calendar
    days_entry=-10,     days_5y_auction=-2,
    days_7y_auction=-1, days_fomc=0,
    days_exit=1,        T_entry=14.0 / ANNUAL,
    # ── Vol params (sigma_quiet auto-calibrated)
    sigma_auction=0.110,
    auto_calibrate=True,
    band_phase1=0.10,
    half_spread=1.0 / 64.0,
    tick_value=1_000.0,         # ZN/ZT tick value
    n_lots=1, n_paths=5_000, seed=42,
    # ── Risk limits
    portfolio_nav=1_000_000.0,
    max_loss_budget=50_000.0,
    kelly_fraction=0.25,
    # ── TWO-LEG STEEPENER
    is_steepener=True,
    F0_30y=118.00,      K_30y=118.00,
    iv_entry_30y=0.085, iv_exit_30y=0.075,
    sigma_iv_event_30y=0.160,
    sigma_auction_30y=0.090,
    tick_value_30y=1_000.0,
    stop_loss_30y=3.0,           # cap short-leg loss at 3× premium received
    rho_curve=0.60,
    # ── GapSpread signal (FIX 1 coherence)
    z_spread        = sig["z_spread"],
    signal_mult_2y  = sig["signal_mult_2y"],
    signal_mult_30y = sig["signal_mult_30y"],
)

# ── Display calibrated vols
print(f"\n── Calibrated parameters (FIX 2: auto-calibrate=True) ──────────────")
print(f"  sigma_quiet (2Y):  {cfg.sigma_quiet*100:.2f}%  "
      f"(was 6.00%; calibrated from iv_entry={cfg.iv_entry*100:.1f}%, "
      f"sigma_iv_event={cfg.sigma_iv_event*100:.0f}%)")
print(f"  sigma_quiet (30Y): {cfg.sigma_quiet_30y*100:.2f}%  "
      f"(calibrated from iv_entry_30y={cfg.iv_entry_30y*100:.1f}%)")
sig_f_2y  = sigma_fomc_leg(cfg.sigma_iv_event,     cfg.z_spread, cfg.kappa, True)
sig_f_30y = sigma_fomc_leg(cfg.sigma_iv_event_30y, cfg.z_spread, cfg.kappa, False)
print(f"  sigma_fomc_2y:     {sig_f_2y*100:.1f}%  "
      f"(= iv_event × (1 + {cfg.kappa}×{cfg.z_spread:.2f}))")
print(f"  sigma_fomc_30y:    {sig_f_30y*100:.1f}%  "
      f"(= iv_event_30y × max(0.01, 1 − {cfg.kappa}×{cfg.z_spread:.2f}))")

V0_2y  = b76_straddle(cfg.F0, cfg.K, cfg.T_entry, cfg.iv_entry)
V0_30y = b76_straddle(cfg.F0_30y, cfg.K_30y, cfg.T_entry, cfg.iv_entry_30y)
print(f"\n  2Y  premium: {V0_2y:.4f} pts = ${V0_2y*cfg.tick_value:,.0f}/lot")
print(f"  30Y premium: {V0_30y:.4f} pts = ${V0_30y*cfg.tick_value_30y:,.0f}/lot")
print(f"  Net premium paid (long − received): "
      f"${(V0_2y*cfg.tick_value - V0_30y*cfg.tick_value_30y):+,.0f}/lot-pair")

# %% ── Monte Carlo simulation ──────────────────────────────────────────────────

print("\nRunning 5,000 MC paths (two-leg steepener) ...")
df_sim = run_mc(cfg)
sizing = size_position(df_sim, cfg)
print_report(df_sim, cfg, sizing)

# %% [markdown]
# ## Reading the P&L output
#
# ### Two-leg net payoff
#
# The long 2Y straddle has a convex, right-skewed payoff: limited loss (premium),
# unlimited gain on large FOMC moves.
#
# The short 30Y straddle has a CONCAVE payoff: receive premium upfront,
# unlimited loss if 30Y rates move sharply.  The stop-loss caps this at
# `stop_loss_30y × premium_received`.
#
# ### Why the trade works at positive z_spread
#
# At `z_spread > 0` (GapSpread says 2Y RV > 2Y IV):
# - The 2Y leg jumps MORE than the IV implies → straddle value explodes → **profit**
# - The 30Y leg jumps LESS than the IV implies → short straddle keeps the premium → **profit**
# - Both legs benefit: the model's edge manifests in both the jump size and the sizing
#
# ### Warsh specifics (z = +2.67)
#
# | Parameter | Value |
# |-----------|-------|
# | sigma_fomc_2y (realized) | ~46.7% (vs 20% IV → front is underpriced) |
# | sigma_fomc_30y (realized) | ~1–2% (vs 16% IV → 30Y doesn't move) |
# | signal_mult_2y | 2.335× (upsize long leg) |
# | signal_mult_30y | 2.335× (upsize short leg — 30Y quiescent) |

# %% ── Acceptance tests (FIX 1 + 2 + 3) ──────────────────────────────────────

print("\nRunning acceptance tests ...")
results = acceptance_tests(cfg, n_paths=8_000, seed=99)

# %% [markdown]
# ## Acceptance test summary
#
# | Test | What it checks | Pass criterion |
# |------|---------------|----------------|
# | **T1 Break-even** | E[P&L \| z=0] ≈ −costs | ratio ≤ 5% |
# | **T2 Coherence** | sigma_fomc_2y > iv_event AND signal_mult > 1 at z>0 | both True |
# | **T3 Monotone** | Mean P&L rises monotonically with z | all Δ ≥ 0 |
# | **T4 Two-leg** | Short 30Y stop-loss cap works; worst < 50% stop rate | both True |
#
# If T1 fails, the calibration is off — only DIFFERENCES across z are meaningful.
# T2 is the coherence invariant: the simulator can no longer be bullish on jumps
# but bearish on sizing (the old contradiction).

# %% ── Band sensitivity ────────────────────────────────────────────────────────

print("\nPhase-1 hedge band sensitivity (2,000 paths each)...")
sens = band_sensitivity(cfg)
print(sens.to_string(index=False,
      float_format=lambda x: f"{x:,.2f}" if abs(x) >= 1 else f"{x:.4f}"))

# %% [markdown]
# ## Band sensitivity
#
# As in the single-leg case, Sharpe is remarkably flat across Phase-1 bands.
# The FOMC jump (Phase 3, no hedge) dominates the joint P&L distribution.
# Default band 0.10–0.20 Δ is adequate for both legs.
#
# The short 30Y leg's Phase-1 band controls how much delta drift accumulates
# before the FOMC — a wider band means slightly more directional risk in the
# short leg entering Phase 3, but minimal Sharpe impact.

# %% ── Figures ─────────────────────────────────────────────────────────────────

fig = plot_results(df_sim, cfg, sens)
plt.savefig("fomc_viz/fig_delta_hedge_sim.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Figure commentary
#
# **Top-left — Net P&L distribution**: Two-humped shape reflects the two competing
# legs.  Left cluster = small FOMC moves (neither leg pays off well; long 2Y
# bleeds theta, short 30Y retains premium but barely).  Right tail = large 2Y
# moves where the long straddle explodes.
#
# **Top-middle — P&L components**: The short 30Y hedge P&L appears as a positive
# contribution when 30Y is quiet (premium retained) and negative when 30Y moves.
#
# **Top-right — Per-leg P&L**: Long 2Y (green) has classic long-straddle shape;
# short 30Y (red) is the mirror — premium collected but tail losses are visible
# until the stop fires.  The stop-loss truncates the red tail.
#
# **Bottom-left — Convexity scatter**: Joint net P&L vs 2Y FOMC move.
# Quadratic curvature from the long leg; short leg dampens the flat section.
# At z=+2.67 the long leg dominates for large moves.
#
# **Bottom-middle — Rebalances**: ~8–12 per path across both legs.
#
# **Bottom-right — Joint jump distribution**: 2Y vs 30Y FOMC moves, colored by
# net P&L.  High rho_curve=0.60 is visible.  Top-right paths (both legs move up)
# have mixed P&L; the best paths are those where 2Y moves a lot but 30Y doesn't.

# %% ── z-sweep: edge as a function of signal ──────────────────────────────────

print("\n── z-sweep: E[P&L] vs GapSpread z-score ──────────────────────────────")
z_sweep_vals = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 2.67, 3.0]
sweep_rows = []
for z in z_sweep_vals:
    c_z = copy.deepcopy(cfg)
    c_z.z_spread = z
    c_z.auto_calibrate = True
    c_z.__post_init__()
    df_z = run_mc(c_z)
    tv   = c_z.tick_value
    sweep_rows.append({
        "z_spread":     z,
        "mean_pnl":     df_z["total_pnl"].mean() * tv,
        "win_rate_%":   (df_z["total_pnl"] > 0).mean() * 100,
        "mean_pnl_2y":  df_z["total_pnl_2y"].mean() * tv,
        "mean_pnl_30y": df_z["total_pnl_30y"].mean() * tv,
        "stop_rate_%":  df_z["stop_fired"].mean() * 100,
    })

df_sweep = pd.DataFrame(sweep_rows)
print(df_sweep.to_string(index=False, float_format=lambda x: f"{x:+.1f}" if abs(x) > 1 else f"{x:.3f}"))

# %% [markdown]
# ## z-sweep interpretation
#
# The sweep shows the P&L monotonically rising with `z_spread` — this is Test 3
# in the acceptance suite.  At `z=0` the mean P&L ≈ −costs (break-even).
# At `z=+2.67` (Warsh) the mean is strongly positive, driven by the amplified
# 2Y jump.  Negative z (buy-long, sell-front) loses money on average as the
# short 2Y leg pays out and the long 30Y leg underperforms.
#
# The short-leg stop rate rises with |z|, reflecting that extreme forecasts imply
# larger moves on all tenors.  The stop-loss at 3× premium keeps the tail bounded.

# %% ── Position sizing summary ─────────────────────────────────────────────────

print("\n── Final Position Sizing ─────────────────────────────────────────────")
print(f"  Portfolio NAV:         ${cfg.portfolio_nav:>12,.0f}")
print(f"  Max loss budget (VaR): ${cfg.max_loss_budget:>12,.0f}")
print(f"  VaR99 ($/lot):         ${sizing['q01_per_lot']:>+12,.0f}")
print(f"  VaR-constrained lots:  {sizing['var_lots']:>12.1f}")
print(f"  Kelly lots:            {sizing['kelly_lots']:>12.1f}")
print(f"  signal_mult_2y:        {sizing['signal_mult']:>12.3f}×")
print(f"  FINAL 2Y lots:         {sizing['lots_2y']:>12d}")
print(f"  FINAL 30Y lots:        {sizing['lots_30y']:>12d}")
premium_2y  = V0_2y  * sizing["lots_2y"]  * cfg.tick_value
premium_30y = V0_30y * sizing["lots_30y"] * cfg.tick_value_30y
print(f"\n  2Y  premium deployed:  ${premium_2y:>12,.0f}  ({sizing['lots_2y']} × ${V0_2y*cfg.tick_value:,.0f})")
print(f"  30Y premium received:  ${premium_30y:>12,.0f}  ({sizing['lots_30y']} × ${V0_30y*cfg.tick_value_30y:,.0f})")
print(f"  Net premium at risk:   ${premium_2y - premium_30y:>+12,.0f}")
exp_net = sizing["lots_2y"] * df_sim["total_pnl_2y"].mean() * cfg.tick_value + \
          sizing["lots_30y"] * df_sim["total_pnl_30y"].mean() * cfg.tick_value
print(f"\n  Expected net P&L at scale: ${exp_net:>+,.0f}")
print(f"  Short 30Y tail (worst case per lot @ stop):  "
      f"${df_sim['total_pnl_30y'].min()*cfg.tick_value:+,.0f}")
print(f"  Short 30Y tail at scale ({sizing['lots_30y']} lots):  "
      f"${df_sim['total_pnl_30y'].min()*cfg.tick_value*sizing['lots_30y']:+,.0f}")

# %% [markdown]
# ---
# ## ETF Pipeline Signal Integration
#
# The ETF IV pipeline (`etf_gap_curve.parquet`) provides an independent
# cross-check on the GapSpread signal using ETF realized vol (GK/Parkinson
# estimator on SHY/TLT OHLC) vs ETF implied vol (VXTYN bridge 2010-2020,
# MOVE bridge 2020-2026).
#
# This section:
# 1. Loads the ETF gap curve for the latest FOMC meeting
# 2. Cross-checks vs `gap_forecasts_spread.parquet`
# 3. Derives yield-space event SDs suitable for `trade_ticket.py`
# 4. Runs an ETF-pipeline-calibrated MC variant as a sensitivity check
#
# ### Key conversion: ETF price vol → yield bps/day
#
# ETF pipeline stores IV and RV as **annualized price-return vol** (%).
# MOVE delivers an annualized yield vol (bps/year), so:
#
# ```
# σ_yield_bps_yr = σ_price_vol_pct / duration_etf
# σ_yield_bps_day = σ_yield_bps_yr / √252
# ```
#
# where `duration_etf ∈ {1.9 (SHY), 4.5 (IEI), 7.5 (IEF), 11 (TLH), 16 (TLT)}`.
#
# For FOMC event sizing, `σ_yield_bps_day` is the RV/IV input to `trade_ticket.py`.

# %% ── ETF pipeline: load & extract latest FOMC signal ────────────────────────

import numpy as np
from pathlib import Path

ETF_GAP_PATH = Path("etf_gap_curve.parquet")
ETF_DURATION = {"SHY": 1.9, "IEI": 4.5, "IEF": 7.5, "TLH": 11.0, "TLT": 16.0}

etf_gap = pd.read_parquet(ETF_GAP_PATH)
etf_gap["meeting_date"] = pd.to_datetime(etf_gap["meeting_date"])

# Latest FOMC meeting with full SHY+TLT coverage
latest_dt = etf_gap.loc[etf_gap["has_both"], "meeting_date"].max()
latest = etf_gap[etf_gap["meeting_date"] == latest_dt].set_index("etf")

shy  = latest.loc["SHY"] if "SHY" in latest.index else None
tlt  = latest.loc["TLT"] if "TLT" in latest.index else None

print(f"\n── ETF Pipeline: {latest_dt.date()} FOMC ──────────────────────────────────")
print(f"  Source (SHY IV): {shy['source']}")
print(f"  Source (TLT IV): {tlt['source']}")
print(f"\n  {'Metric':<28} {'SHY (2Y)':>14} {'TLT (30Y)':>14}")
print(f"  {'─'*28} {'─'*14} {'─'*14}")
for label, col in [
    ("RV event vol %",      "rv_vol_pct"),
    ("IV event vol %",      "iv_event_vol_pct"),
    ("IV percentile",       "iv_percentile"),
    ("Gap (RV−IV var)",     "gap"),
]:
    sv = shy[col] if shy is not None else float("nan")
    tv = tlt[col] if tlt is not None else float("nan")
    fmt = f"{sv:>14.4f}" if not (isinstance(sv, str)) else f"{sv:>14}"
    fmtb = f"{tv:>14.4f}" if not (isinstance(tv, str)) else f"{tv:>14}"
    print(f"  {label:<28} {fmt} {fmtb}")

gap_spread_etf  = latest["GapSpread_proxy"].iloc[0]
spread_dir_etf  = latest["spread_direction"].iloc[0]
print(f"\n  GapSpread_proxy (ETF): {gap_spread_etf:+.6f}  →  {spread_dir_etf}")
print(f"  GapSpread (existing ):  {sig['z_spread']:+.3f} z-score  →  {sig['steepener_signal']}")

# %% ── ETF-derived yield SDs for trade_ticket.py ──────────────────────────────

def etf_vol_to_yield_bps_day(price_vol_pct: float, duration: float) -> float:
    """
    Convert ETF annualized price-return vol (%) → yield bps per event day.

    σ_yield_pct_yr  = σ_price_pct / duration        (dur in years)
    σ_yield_bps_yr  = σ_yield_pct_yr × 100          (% → bps)
    σ_yield_bps_day = σ_yield_bps_yr / √252
    """
    return (price_vol_pct / duration) * 100 / np.sqrt(252)

if shy is not None and tlt is not None:
    # Use IV (forward-looking from MOVE bridge) as the pricing input
    iv_2y_bps_day  = etf_vol_to_yield_bps_day(shy["iv_event_vol_pct"],  ETF_DURATION["SHY"])
    iv_30y_bps_day = etf_vol_to_yield_bps_day(tlt["iv_event_vol_pct"],  ETF_DURATION["TLT"])
    # Use RV (backward-looking) as the MC event-jump input
    rv_2y_bps_day  = etf_vol_to_yield_bps_day(shy["rv_vol_pct"],  ETF_DURATION["SHY"])
    rv_30y_bps_day = etf_vol_to_yield_bps_day(tlt["rv_vol_pct"],  ETF_DURATION["TLT"])

    print(f"\n── ETF → Yield bps/day conversion ({latest_dt.date()}) ────────────────")
    print(f"  {'Input':<32} {'SHY / 2Y':>12} {'TLT / 30Y':>12}")
    print(f"  {'─'*32} {'─'*12} {'─'*12}")
    print(f"  {'IV price vol % (annualised)':<32} {shy['iv_event_vol_pct']:>12.3f} {tlt['iv_event_vol_pct']:>12.3f}")
    print(f"  {'RV price vol % (annualised)':<32} {shy['rv_vol_pct']:>12.3f} {tlt['rv_vol_pct']:>12.3f}")
    print(f"  {'Duration (years)':<32} {ETF_DURATION['SHY']:>12.1f} {ETF_DURATION['TLT']:>12.1f}")
    print(f"  {'IV → yield bps/day':<32} {iv_2y_bps_day:>12.3f} {iv_30y_bps_day:>12.3f}")
    print(f"  {'RV → yield bps/day':<32} {rv_2y_bps_day:>12.3f} {rv_30y_bps_day:>12.3f}")
    print()
    print(f"  ── trade_ticket.py CFG overrides for {latest_dt.date()} ──")
    print(f"  # Paste these into trade_ticket.CFG to use ETF pipeline signal:")
    print(f"  SIGMA_2Y_YIELD_BPS_DAY  = {rv_2y_bps_day:.4f}  # RV from SHY GK estimator")
    print(f"  SIGMA_30Y_YIELD_BPS_DAY = {rv_30y_bps_day:.4f}  # RV from TLT GK estimator")
    print(f"  # For premium pricing (pass to SIGMA_N_*_OVERRIDE via sigma_price_from_event_sd):")
    print(f"  # SIGMA_N_2Y_OVERRIDE  = sigma_price_from_event_sd({iv_2y_bps_day:.4f},  DV01_2Y_PER_1M)")
    print(f"  # SIGMA_N_30Y_OVERRIDE = sigma_price_from_event_sd({iv_30y_bps_day:.4f}, DV01_30Y_PER_1M)")

# %% ── ETF-calibrated MC sensitivity variant ───────────────────────────────────

if shy is not None and tlt is not None:
    # Convert ETF price vols → delta_hedge_sim event vol scale (fraction of forward)
    # fomc_delta_hedge_sim uses sigma_iv_event in price-vol fraction (0.20 = 20%)
    etf_sigma_iv_2y  = shy["iv_event_vol_pct"] / 100   # ETF price-return vol
    etf_sigma_iv_30y = tlt["iv_event_vol_pct"] / 100
    etf_sigma_rv_2y  = shy["rv_vol_pct"] / 100
    etf_sigma_rv_30y = tlt["rv_vol_pct"] / 100

    cfg_etf = copy.deepcopy(cfg)
    cfg_etf.sigma_iv_event     = etf_sigma_iv_2y
    cfg_etf.sigma_iv_event_30y = etf_sigma_iv_30y
    cfg_etf.auto_calibrate = True
    cfg_etf.__post_init__()

    print(f"\n── ETF-calibrated MC variant ────────────────────────────────────────")
    print(f"  sigma_iv_event (ETF / 2Y):  {etf_sigma_iv_2y*100:.2f}%  "
          f"vs baseline {cfg.sigma_iv_event*100:.1f}%")
    print(f"  sigma_iv_event (ETF / 30Y): {etf_sigma_iv_30y*100:.2f}%  "
          f"vs baseline {cfg.sigma_iv_event_30y*100:.1f}%")
    print(f"  RV / IV ratio (2Y):  {etf_sigma_rv_2y / etf_sigma_iv_2y:.3f}×  "
          f"(>1 = 2Y realised > implied)")
    print(f"  RV / IV ratio (30Y): {etf_sigma_rv_30y / etf_sigma_iv_30y:.3f}×  "
          f"(<1 = 30Y realised < implied)")
    print(f"\n  Running ETF-calibrated MC ({cfg.n_paths:,} paths) ...")

    df_etf = run_mc(cfg_etf)
    tv = cfg_etf.tick_value

    print(f"\n  {'Scenario':<28} {'Mean P&L':>12} {'Win%':>8} {'VaR99':>12} {'Stop%':>8}")
    print(f"  {'─'*28} {'─'*12} {'─'*8} {'─'*12} {'─'*8}")
    for label, df_, tv_ in [
        ("Baseline (gap_forecasts)",  df_sim,  cfg.tick_value),
        (f"ETF pipeline ({latest_dt.date()})", df_etf, cfg_etf.tick_value),
    ]:
        mean_  = df_["total_pnl"].mean() * tv_
        winr_  = (df_["total_pnl"] > 0).mean() * 100
        var99_ = df_["total_pnl"].quantile(0.01) * tv_
        stop_  = df_["stop_fired"].mean() * 100
        print(f"  {label:<28} {mean_:>+12,.0f} {winr_:>8.1f}% {var99_:>+12,.0f} {stop_:>8.1f}%")

    print(f"\n  GapSpread signal: {gap_spread_etf:+.6f} variance-units  "
          f"→  {spread_dir_etf}")
    print(f"  ETF-calibrated variant uses pipeline IV as event-vol input.")
    print(f"  Use RV/IV ratios above to verify the front-end vol is mispriced.")

# %% [markdown]
# ## ETF pipeline integration notes
#
# ### Signal consistency check
#
# The `gap_forecasts_spread.parquet` z-score and the ETF `GapSpread_proxy` measure
# the same signal through different data paths (direct Treasury yield vol for
# gap_forecasts; ETF price vol via MOVE bridge for etf_gap_curve).  They should
# agree on direction; magnitude will differ because:
# - gap_forecasts uses bond yield vol directly; ETF uses price returns
# - MOVE bridge covers 2020–2026 (ETF options chains not available post-TYVIX)
#
# ### How to feed ETF output into trade_ticket.py
#
# The printed `trade_ticket.CFG overrides` block above gives exact values
# to paste into `CFG` in `trade_ticket.py`:
#
# ```python
# # Realized vol from ETF pipeline → MC event-jump input
# SIGMA_2Y_YIELD_BPS_DAY  = <rv_2y_bps_day>
# SIGMA_30Y_YIELD_BPS_DAY = <rv_30y_bps_day>
#
# # OMON-implied vol → premium pricing (override from screen for best accuracy)
# # If OMON is unavailable, use ETF-IV-derived sigma_n as proxy:
# # SIGMA_N_2Y_OVERRIDE  = sigma_price_from_event_sd(iv_2y_bps_day,  DV01_2Y_PER_1M)
# # SIGMA_N_30Y_OVERRIDE = sigma_price_from_event_sd(iv_30y_bps_day, DV01_30Y_PER_1M)
# ```
#
# ### RV/IV interpretation
#
# | RV/IV (2Y) | RV/IV (30Y) | Interpretation |
# |---|---|---|
# | > 1 | < 1 | **Steepener signal**: front realized > implied, long end quiet |
# | > 1 | > 1 | Parallel move — both legs expensive; reconsider sizing |
# | < 1 | < 1 | Vol is richly priced everywhere; avoid straddles |
# | < 1 | > 1 | Flattener signal (opposite trade) |
#
# ### ETF-calibrated vs baseline MC
#
# The ETF-calibrated variant uses pipeline IV (MOVE-derived) as the event vol
# rather than the hand-tuned `sigma_iv_event` in the baseline config.  Use it to:
# - Stress-test sensitivity to vol-level assumptions
# - Verify that the steepener P&L is positive for a range of MOVE-implied vol levels
