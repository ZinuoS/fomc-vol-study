"""
fomc_delta_hedge_sim.py
=======================
Delta-hedged FOMC straddle P&L simulation and position sizer.

Trade structure
---------------
Long straddle (post-FOMC expiry) entered post-CPI, exited pre month-end/PCE.

Three-phase hedge calendar
--------------------------
Phase 1 — Run-up (entry → day before first auction)
    Daily delta-band hedge.  Rebalance whenever |Δ_net| > ε₁.
    Bleeds some realised gamma but controls overnight directional drift.

Phase 2 — Auctions (5Y auction day + 7Y auction day)
    Event-bracketed:
        (a) Flatten delta → 0 just before each auction opens.
        (b) Hold that flat hedge through the auction's price reaction.
        (c) Flatten back to straddle-delta after the dust settles.
    Result: auction directional move nets to zero in the hedge book;
    only the gamma convexity of the straddle benefits (or suffers) from
    the auction's vol spike.

Phase 3 — Event → exit (post-auction → FOMC + 1 day)
    Stop all hedging ~5 min before 2 pm FOMC.
    Full convexity preserved through the decision jump.
    Unwind the straddle (and close residual hedge) the next morning,
    before month-end / PCE.

P&L identity (per path)
-----------------------
  Total = Straddle_terminal − Straddle_entry
        + Hedge_accumulated          (futures P&L from delta management)
        − Costs                      (½-spread × |ΔH| per rebalance)

Positioning
-----------
  Recommended lots = min(Kelly_lots, VaR_lots)
  Kelly  : f* = μ/σ²;  kelly_lots = f* × NAV / (F₀ × tick_value)
  VaR    : lots such that 99th-percentile loss ≤ max_loss_budget

  NLP signal_mult from gap_forecasts.parquet scales the base lots:
      final_lots = round(recommended_lots × signal_mult)
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from math import exp, log, sqrt
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore", category=RuntimeWarning)

ANNUAL = 252  # trading days per year


# ── Black-76 helpers ──────────────────────────────────────────────────────────

def _d1d2(F: float, K: float, T: float, sigma: float) -> tuple[float, float]:
    if T <= 0 or sigma <= 0 or F <= 0 or K <= 0:
        return 0.0, 0.0
    d1 = (log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrt(T))
    return d1, d1 - sigma * sqrt(T)


def b76_straddle(F: float, K: float, T: float, sigma: float) -> float:
    """ATM straddle price via Black-76 (call + put on futures)."""
    if T <= 0:
        return abs(F - K)
    d1, d2 = _d1d2(F, K, T, sigma)
    call = F * norm.cdf(d1) - K * norm.cdf(d2)
    put  = K * norm.cdf(-d2) - F * norm.cdf(-d1)
    return call + put


def b76_delta(F: float, K: float, T: float, sigma: float) -> float:
    """Straddle delta = Δ_call + Δ_put = N(d1) - N(-d1) = 2N(d1) - 1."""
    if T <= 0:
        return float(np.sign(F - K))
    d1, _ = _d1d2(F, K, T, sigma)
    return 2.0 * norm.cdf(d1) - 1.0


def b76_gamma(F: float, K: float, T: float, sigma: float) -> float:
    """Gamma of the straddle (same for call and put, so 2× call gamma)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1, _ = _d1d2(F, K, T, sigma)
    return 2.0 * norm.pdf(d1) / (F * sigma * sqrt(T))


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class StraddleConfig:
    # ── Option ────────────────────────────────────────────────────────────────
    F0: float = 108.50          # Futures price at entry (e.g. ZN Sep26)
    K:  float = 108.50          # Strike (ATM at entry)
    iv_entry: float = 0.082     # Implied vol at entry (Black-76, annualised)
    iv_exit: float  = 0.062     # IV at exit: post-FOMC vol usually compresses

    # ── Calendar (trading days relative to FOMC decision = day 0) ─────────────
    days_entry:      int = -10  # post-CPI entry (FOMC – 10 bd)
    days_5y_auction: int = -2   # 5-year note auction
    days_7y_auction: int = -1   # 7-year note auction
    days_fomc:       int =  0   # FOMC decision (2 pm ET)
    days_exit:       int =  1   # exit next morning, pre month-end

    # time-to-expiry at entry: option spans FOMC → expires ~2 weeks post-entry
    T_entry: float = 14.0 / ANNUAL   # in years

    # ── Vol regimes (annualised, Black-76 / GK frame) ────────────────────────
    sigma_quiet:   float = 0.060   # Phase 1 "quiet" daily vol
    sigma_auction: float = 0.110   # Vol on 5Y / 7Y auction days
    sigma_fomc:    float = 0.200   # FOMC event vol (concentrated in ~90 min)

    # ── Delta-hedge parameters ────────────────────────────────────────────────
    band_phase1: float = 0.10   # Phase 1: rebalance if |Δ_net| > band (fraction of 1 lot)

    # ── Transaction costs ─────────────────────────────────────────────────────
    # Treasury futures: 1 tick = 1/32 of a price point.
    # half-spread in price-point units: typically 0.5 ticks for ZN = 1/64
    half_spread: float = 1.0 / 64.0

    # ── Contract sizing ───────────────────────────────────────────────────────
    tick_value: float = 1_000.0   # $ per price point per lot (ZN: 1pt = $1,000)
    n_lots: int = 1               # straddle lots to simulate (scale up for sizing)

    # ── Simulation ────────────────────────────────────────────────────────────
    n_paths: int  = 5_000
    seed:    int  = 42

    # ── Sizing constraints ────────────────────────────────────────────────────
    portfolio_nav:    float = 1_000_000.0   # portfolio NAV ($)
    max_loss_budget:  float =    50_000.0   # max acceptable 99th-pctile loss ($)
    kelly_fraction:   float = 0.25          # fractional Kelly cap (safety)

    # ── NLP signal integration ────────────────────────────────────────────────
    signal_mult: float = 1.0               # loaded from gap_forecasts.parquet


# ── Single-path simulation ────────────────────────────────────────────────────

def _simulate_path(cfg: StraddleConfig, rng: np.random.Generator) -> dict:
    """
    Simulate one path.  Returns dict of P&L components (in price points per lot).

    State variables
    ---------------
    F   : futures price (starts at F0)
    H   : cumulative futures hedge in lots (+ve = long futures)
          convention: H + straddle_delta = net_delta
                      rebalance sets H = −straddle_delta  (net_delta = 0)
    """
    dt   = 1.0 / ANNUAL         # one trading day in years
    F    = cfg.F0
    K    = cfg.K
    H    = 0.0                  # no initial hedge
    hedge_pnl = 0.0
    cost_pnl  = 0.0
    n_rebal   = 0
    fomc_move = 0.0

    # Entry straddle mark
    T_at_entry = cfg.T_entry
    V_entry    = b76_straddle(F, K, T_at_entry, cfg.iv_entry)

    def T_rem(day: int) -> float:
        elapsed = (day - cfg.days_entry) * dt
        return max(0.0, T_at_entry - elapsed)

    def phase(day: int) -> str:
        if day < cfg.days_5y_auction:
            return "P1"
        if day in (cfg.days_5y_auction, cfg.days_7y_auction):
            return "P2"
        return "P3"

    for day in range(cfg.days_entry, cfg.days_exit + 1):
        ph   = phase(day)
        T_t  = T_rem(day)
        T_tp = T_rem(day + 1)

        # Regime vol
        if   day == cfg.days_fomc:                   sigma = cfg.sigma_fomc
        elif day in (cfg.days_5y_auction, cfg.days_7y_auction): sigma = cfg.sigma_auction
        else:                                          sigma = cfg.sigma_quiet

        dF = F * sigma * sqrt(dt) * rng.standard_normal()

        if ph == "P2":
            # ── Auction bracket ───────────────────────────────────────────────
            # (a) Flatten before: set H so net delta = 0
            delta_pre = b76_delta(F, K, T_t, cfg.iv_entry) * cfg.n_lots
            H_pre     = -delta_pre
            dH_pre    = H_pre - H
            cost_pnl -= abs(dH_pre) * cfg.half_spread
            hedge_pnl += H * dF   # old H earns on the move…
            H = H_pre             # …then we flatten
            n_rebal += 1

            # (b) Price moves; hold H flat through the auction reaction
            F += dF

            # (c) Flatten after: re-hedge to new straddle delta
            delta_post = b76_delta(F, K, T_tp, cfg.iv_entry) * cfg.n_lots
            H_post     = -delta_post
            dH_post    = H_post - H
            cost_pnl  -= abs(dH_post) * cfg.half_spread
            H = H_post
            n_rebal += 1

        elif ph == "P3":
            # ── No hedge — preserve full convexity ────────────────────────────
            hedge_pnl += H * dF
            F += dF
            if day == cfg.days_fomc:
                fomc_move = dF

        else:
            # ── Phase 1: daily delta-band hedge ───────────────────────────────
            hedge_pnl += H * dF
            F += dF

            delta_pos = b76_delta(F, K, T_tp, cfg.iv_entry) * cfg.n_lots
            net_delta = delta_pos + H

            if abs(net_delta) > cfg.band_phase1:
                dH = -net_delta          # re-centre to zero
                cost_pnl -= abs(dH) * cfg.half_spread
                H += dH
                n_rebal += 1

    # Close residual hedge at exit
    cost_pnl -= abs(H) * cfg.half_spread

    # Terminal straddle mark (use exit IV — vol compresses post-FOMC)
    T_exit = T_rem(cfg.days_exit + 1)
    if T_exit > 0:
        V_exit = b76_straddle(F, K, T_exit, cfg.iv_exit)
    else:
        V_exit = abs(F - K)

    straddle_pnl = (V_exit - V_entry) * cfg.n_lots
    total_pnl    = straddle_pnl + hedge_pnl + cost_pnl

    return {
        "total_pnl":    total_pnl,
        "straddle_pnl": straddle_pnl,
        "hedge_pnl":    hedge_pnl,
        "cost_pnl":     cost_pnl,
        "n_rebal":      n_rebal,
        "F_final":      F,
        "fomc_move":    fomc_move,
    }


# ── Monte Carlo runner ────────────────────────────────────────────────────────

def run_mc(cfg: StraddleConfig) -> pd.DataFrame:
    """Run cfg.n_paths independent paths. Returns DataFrame of results."""
    rng = np.random.default_rng(cfg.seed)
    return pd.DataFrame([_simulate_path(cfg, rng) for _ in range(cfg.n_paths)])


# ── Position sizing ───────────────────────────────────────────────────────────

def size_position(df: pd.DataFrame, cfg: StraddleConfig) -> dict:
    """
    Compute optimal lot count from simulated P&L distribution.

    Kelly is computed on *returns* (P&L / premium invested), which is the correct
    base for capital allocation.  Raw dollar Kelly (μ/σ² in $ units) gives a tiny
    fraction because it ignores that each lot costs premium_per_lot, not $1.

    Signal_mult convention for LONG straddle:
        signal_mult > 1  →  VRP model says IV > RV  →  sell-vol wins
                         →  Phase 1-2 gamma LOSES (realised < implied)
                         →  REDUCE long straddle  →  use inverse mult
        signal_mult < 1  →  unusual: buy-vol edge  →  INCREASE long straddle
    Final size = round(VaR_lots / signal_mult) clipped to Kelly upper bound.
    """
    pnl_d = df["total_pnl"] * cfg.tick_value    # $ P&L per lot

    # Premium invested per lot (cost of the straddle at entry)
    prem = b76_straddle(cfg.F0, cfg.K, cfg.T_entry, cfg.iv_entry) * cfg.tick_value

    mu     = float(pnl_d.mean())
    sigma  = float(pnl_d.std())
    q01    = float(pnl_d.quantile(0.01))
    q05    = float(pnl_d.quantile(0.05))
    sharpe = mu / sigma if sigma > 0 else 0.0

    # Kelly fraction on return: f* = μ_ret / σ_ret²
    #   μ_ret   = E[P&L per lot] / premium per lot
    #   σ_ret   = std[P&L per lot] / premium per lot
    # This gives the fraction of NAV to allocate to premium.
    mu_ret    = mu / prem if prem > 0 else 0.0
    sigma_ret = sigma / prem if prem > 0 else 1.0
    kelly_f   = min(max(mu_ret / sigma_ret**2, 0.0), cfg.kelly_fraction)
    kelly_budget = kelly_f * cfg.portfolio_nav        # total premium to deploy ($)
    kelly_lots   = kelly_budget / prem if prem > 0 else 0.0

    # VaR-constrained lots: 99th-pctile dollar loss per lot ≤ max_loss_budget
    loss99 = -q01    # positive = loss
    var_lots = cfg.max_loss_budget / loss99 if loss99 > 0 else 999.0

    # Base size: min of Kelly and VaR
    base_lots = min(kelly_lots, var_lots)

    # NLP signal_mult is a sell-vol scaler → invert for long straddle.
    # A strong sell-vol signal means we are fighting the VRP edge in Phase 1-2,
    # so we size DOWN.  Phase 3 (FOMC convexity) is independent.
    inv_mult   = 1.0 / max(cfg.signal_mult, 0.1)
    final_lots = max(1, round(base_lots * inv_mult))

    return {
        "premium_per_lot": prem,
        "mu_per_lot":      mu,
        "sigma_per_lot":   sigma,
        "sharpe":          sharpe,
        "q01_per_lot":     q01,
        "q05_per_lot":     q05,
        "kelly_f":         kelly_f,
        "kelly_lots":      kelly_lots,
        "var_lots":        var_lots,
        "base_lots":       base_lots,
        "signal_mult":     cfg.signal_mult,
        "inv_mult":        inv_mult,
        "final_lots":      final_lots,
    }


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(df: pd.DataFrame, cfg: StraddleConfig, sizing: dict) -> None:
    mu    = sizing["mu_per_lot"]
    sigma = sizing["sigma_per_lot"]
    q01   = sizing["q01_per_lot"]
    q05   = sizing["q05_per_lot"]
    sr    = sizing["sharpe"]

    sep = "─" * 64
    print(f"\n{'═'*64}")
    print(f"  DELTA-HEDGED FOMC STRADDLE  ({cfg.n_paths:,} MC paths)")
    print(f"{'═'*64}")
    print(f"\n  Trade parameters")
    print(sep)
    print(f"  Futures:          F₀ = {cfg.F0:.4f}   K = {cfg.K:.4f}   (ATM)")
    V0 = b76_straddle(cfg.F0, cfg.K, cfg.T_entry, cfg.iv_entry)
    print(f"  Entry straddle:   IV = {cfg.iv_entry*100:.1f}%   "
          f"T = {cfg.T_entry*ANNUAL:.0f} bd   "
          f"Premium = {V0:.4f} pts  (${V0*cfg.tick_value:,.0f} per lot)")
    print(f"  Exit IV:          {cfg.iv_exit*100:.1f}%  (post-FOMC compression)")
    print(f"\n  Vol regimes       Phase1 = {cfg.sigma_quiet*100:.1f}%  "
          f"Auction = {cfg.sigma_auction*100:.1f}%  "
          f"FOMC = {cfg.sigma_fomc*100:.1f}%")
    print(f"  Phase 1 band:     ε = {cfg.band_phase1:.2f}Δ")
    print(f"  Half-spread:      {cfg.half_spread:.4f} pts  (= {cfg.half_spread*32:.1f} ticks)")

    print(f"\n  P&L distribution ($ per lot, {cfg.n_paths:,} paths)")
    print(sep)
    print(f"  Mean:          {mu:>+10,.0f}")
    print(f"  Std:           {sigma:>10,.0f}")
    print(f"  Sharpe:        {sr:>10.3f}")
    print(f"  5th pctile:    {q05:>+10,.0f}  (loss scenario)")
    print(f"  1st pctile:    {q01:>+10,.0f}  (tail / VaR99)")
    print(f"  Win rate:      {(df['total_pnl']>0).mean()*100:>9.1f}%")

    print(f"\n  P&L components (mean $ per lot)")
    print(sep)
    print(f"  Straddle:      {df['straddle_pnl'].mean()*cfg.tick_value:>+10,.0f}")
    print(f"  Hedge:         {df['hedge_pnl'].mean()*cfg.tick_value:>+10,.0f}")
    print(f"  Costs:         {df['cost_pnl'].mean()*cfg.tick_value:>+10,.0f}")
    print(f"  Avg rebalances:{df['n_rebal'].mean():>10.1f}")

    print(f"\n  Position sizing")
    print(sep)
    print(f"  Premium per lot:              {sizing['premium_per_lot']:>8,.0f} USD")
    print(f"  Portfolio NAV:                {cfg.portfolio_nav:>8,.0f} USD")
    print(f"  Max loss budget (99%):        {cfg.max_loss_budget:>8,.0f} USD")
    print(f"  Kelly f (capped {cfg.kelly_fraction*100:.0f}%):       {sizing['kelly_f']:>8.3f}")
    print(f"  Kelly lots  (Kelly × NAV / prem):  {sizing['kelly_lots']:>6.1f}")
    print(f"  VaR lots    (budget / VaR99):      {sizing['var_lots']:>6.1f}")
    print(f"  Base lots   (min of above):        {sizing['base_lots']:>6.1f}")
    print(f"  NLP signal_mult (sell-vol):        {cfg.signal_mult:>6.3f}×")
    print(f"  Inverse mult (long straddle):      {sizing['inv_mult']:>6.3f}×")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  FINAL SIZE:  {sizing['final_lots']:>4d} lots  "
          f"(~{sizing['final_lots'] * sizing['premium_per_lot']:,.0f} USD premium)")
    print(f"{'═'*64}")


# ── Band sensitivity ──────────────────────────────────────────────────────────

def band_sensitivity(cfg: StraddleConfig,
                     bands: list[float] | None = None,
                     n_paths: int = 2_000) -> pd.DataFrame:
    """Sweep Phase-1 hedge band; return Sharpe, mean P&L, cost per band."""
    if bands is None:
        bands = [0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50]

    rows = []
    for b in bands:
        c = StraddleConfig(**{**cfg.__dict__, "band_phase1": b, "n_paths": n_paths})
        df = run_mc(c)
        tv = c.tick_value
        rows.append({
            "band (Δ)":     b,
            "mean P&L ($)": df["total_pnl"].mean() * tv,
            "std P&L ($)":  df["total_pnl"].std()  * tv,
            "Sharpe":       df["total_pnl"].mean() / df["total_pnl"].std(),
            "win rate %":   (df["total_pnl"] > 0).mean() * 100,
            "cost ($)":     df["cost_pnl"].mean() * tv,
            "avg rebal":    df["n_rebal"].mean(),
        })
    return pd.DataFrame(rows)


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_results(df: pd.DataFrame, cfg: StraddleConfig,
                 sens: pd.DataFrame | None = None) -> plt.Figure:
    ncols = 3 if sens is not None else 2
    fig, axes = plt.subplots(2, ncols, figsize=(6*ncols, 10))
    fig.suptitle(
        f"Delta-Hedged FOMC Straddle — {cfg.n_paths:,} MC Paths\n"
        f"F₀={cfg.F0:.2f}, IV={cfg.iv_entry*100:.1f}%, "
        f"Band={cfg.band_phase1:.2f}Δ, Phase3=free",
        fontsize=13, fontweight="bold",
    )
    tv = cfg.tick_value

    # 1. Total P&L histogram
    ax = axes[0, 0]
    pnl_d = df["total_pnl"] * tv
    ax.hist(pnl_d, bins=80, color="#2c7a45", alpha=0.75, edgecolor="none")
    ax.axvline(0, color="k", lw=1.5, ls="--")
    ax.axvline(float(pnl_d.mean()), color="#e05c2d", lw=2,
               label=f"Mean = {pnl_d.mean():+,.0f}")
    ax.axvline(float(pnl_d.quantile(0.05)), color="#762a83", lw=1.5, ls=":",
               label=f"5th pctile = {pnl_d.quantile(0.05):+,.0f}")
    ax.set_title("Total P&L (USD per lot)")
    ax.set_xlabel("USD")
    ax.legend(fontsize=9)

    # 2. P&L components bar
    ax = axes[0, 1]
    comps = {
        "Straddle": df["straddle_pnl"].mean() * tv,
        "Hedge":    df["hedge_pnl"].mean()    * tv,
        "Costs":    df["cost_pnl"].mean()     * tv,
        "Total":    df["total_pnl"].mean()    * tv,
    }
    colors = ["#2c7a45", "#1f77b4", "#d62728", "#ff7f0e"]
    bars = ax.bar(comps.keys(), comps.values(), color=colors, alpha=0.85)
    ax.axhline(0, color="k", lw=1)
    ax.set_title("Mean P&L by Component ($ per lot)")
    ax.set_ylabel("$")
    for bar, v in zip(bars, comps.values()):
        ax.text(bar.get_x() + bar.get_width()/2,
                v + (50 if v >= 0 else -50),
                f"${v:,.0f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=9)

    # 3. FOMC jump vs total P&L scatter
    ax = axes[1, 0]
    jumps = df["fomc_move"] * 100    # in ticks / 100ths
    color_vals = df["total_pnl"] * tv
    sc = ax.scatter(jumps, color_vals, c=color_vals, cmap="RdYlGn",
                    alpha=0.25, s=10, vmin=color_vals.quantile(0.05),
                    vmax=color_vals.quantile(0.95))
    plt.colorbar(sc, ax=ax, label="Total P&L ($)")
    ax.axhline(0, color="k", lw=1, ls="--")
    ax.axvline(0, color="k", lw=1, ls="--")
    ax.set_xlabel("FOMC day futures move (pts)")
    ax.set_ylabel("Total P&L ($)")
    ax.set_title("Convexity: P&L vs FOMC Jump")

    # 4. Rebalance count
    ax = axes[1, 1]
    max_r = int(df["n_rebal"].max())
    ax.hist(df["n_rebal"], bins=range(max_r + 2), align="left",
            color="#ff7f0e", alpha=0.8, edgecolor="white")
    ax.axvline(df["n_rebal"].mean(), color="red", lw=2,
               label=f"Mean = {df['n_rebal'].mean():.1f}")
    ax.set_title("Rebalances per Path")
    ax.set_xlabel("Count")
    ax.legend()

    # 5. Band sensitivity (if provided)
    if sens is not None and ncols == 3:
        ax = axes[0, 2]
        ax.plot(sens["band (Δ)"], sens["Sharpe"], "o-", color="#2c7a45", lw=2)
        ax.set_xlabel("Phase-1 Hedge Band (Δ)")
        ax.set_ylabel("Sharpe Ratio")
        ax.set_title("Sharpe vs Hedge Band Width")
        ax.axhline(0, color="k", lw=1, ls="--")
        ax2 = ax.twinx()
        ax2.plot(sens["band (Δ)"], sens["cost ($)"], "s--", color="#d62728", lw=1.5)
        ax2.set_ylabel("Mean Cost ($ per lot)", color="#d62728")
        ax2.tick_params(axis="y", labelcolor="#d62728")

        ax = axes[1, 2]
        ax.plot(sens["band (Δ)"], sens["avg rebal"], "o-", color="#762a83", lw=2)
        ax.set_xlabel("Phase-1 Hedge Band (Δ)")
        ax.set_ylabel("Avg Rebalances per Path")
        ax.set_title("Rebalances vs Band Width")

    plt.tight_layout()
    return fig


# ── NLP signal loading ────────────────────────────────────────────────────────

def load_signal_mult(parquet_path: Path = Path("gap_forecasts.parquet"),
                     meeting_date: str | None = None) -> float:
    """
    Load signal_mult from the VRP pipeline output.
    If meeting_date is None, returns the most recent available value.
    """
    if not parquet_path.exists():
        print(f"  [signal] {parquet_path} not found — using signal_mult=1.0")
        return 1.0
    df = pd.read_parquet(parquet_path)
    if meeting_date:
        row = df[df["meeting_date"] == pd.Timestamp(meeting_date)]
        if not row.empty:
            return float(row.iloc[0]["signal_mult"])
    # Fallback: most recent
    val = float(df.sort_values("meeting_date").iloc[-1]["signal_mult"])
    print(f"  [signal] signal_mult = {val:.3f}  (most recent meeting)")
    return val


# ── Entry point ───────────────────────────────────────────────────────────────

def run(cfg: StraddleConfig | None = None,
        run_sensitivity: bool = True,
        save_fig: Path | None = Path("fomc_viz/fig_delta_hedge_sim.png"),
        quiet: bool = False) -> tuple[pd.DataFrame, dict, pd.DataFrame | None]:
    """
    Run MC simulation, print report, optionally run band sensitivity.
    Returns (sim_df, sizing_dict, sensitivity_df).
    """
    if cfg is None:
        # Default: July 2026 FOMC trade
        signal = load_signal_mult()
        cfg = StraddleConfig(
            F0=108.50, K=108.50,
            iv_entry=0.082, iv_exit=0.062,
            days_entry=-10, days_5y_auction=-2, days_7y_auction=-1,
            days_fomc=0, days_exit=1,
            T_entry=14.0 / ANNUAL,
            sigma_quiet=0.060, sigma_auction=0.110, sigma_fomc=0.200,
            band_phase1=0.10,
            half_spread=1.0 / 64.0,
            tick_value=1_000.0,
            n_lots=1,
            n_paths=5_000,
            portfolio_nav=1_000_000.0,
            max_loss_budget=50_000.0,
            kelly_fraction=0.25,
            signal_mult=signal,
        )

    if not quiet:
        print(f"Running {cfg.n_paths:,} MC paths ...")
    df = run_mc(cfg)

    sizing = size_position(df, cfg)
    if not quiet:
        print_report(df, cfg, sizing)

    sens = None
    if run_sensitivity:
        if not quiet:
            print("\nPhase-1 band sensitivity (2,000 paths each) ...")
        sens = band_sensitivity(cfg)
        if not quiet:
            print(sens.to_string(index=False,
                  float_format=lambda x: f"{x:,.2f}" if abs(x) > 1 else f"{x:.4f}"))

    if save_fig:
        save_fig.parent.mkdir(exist_ok=True)
        fig = plot_results(df, cfg, sens)
        fig.savefig(save_fig, dpi=150, bbox_inches="tight")
        if not quiet:
            print(f"\nFigure → {save_fig}")
        plt.close(fig)

    return df, sizing, sens


if __name__ == "__main__":
    run()
