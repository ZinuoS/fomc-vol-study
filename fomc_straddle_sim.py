"""
================================================================================
FOMC EVENT-STRADDLE — P&L / DELTA SIMULATOR + EVENT-STUDY FRAMEWORK
================================================================================
Instrument : ATM straddle on the CME 10Y T-Note future (TYU6), listed weekly options.
Pricing    : BACHELIER (normal / bp vol) — the correct convention for rate futures
             options. American-future early exercise on TY is negligible for short-
             dated ATM (forward = future, no carry on the option), so European
             Bachelier is an accurate proxy; a tiny early-exercise premium can be
             added if needed. (SABR would only be needed for a full-smile/skew trade.)
Counterparty: listed -> CME Clearing via FCM (SPAN margin). No ISDA. Simulator is
             agnostic to this, but P&L is gross of clearing fees/margin cost.

Three parts:
  1. Bachelier pricer (price, delta, gamma, vega, theta) in price space.
  2. Path simulator: walks a daily TY path through entry -> auctions -> FOMC -> exit,
     applies the EVENT-BRACKETED delta-hedge schedule, and records P&L + delta.
  3. Event study: re-runs the SAME path with one scenario knob changed
     ("what if the FOMC jump is 2x / the 7y auction tails / implied crushes harder")
     and reports the P&L delta attributable to that event.

All sizes/vols are PLACEHOLDERS — overwrite TICK_VALUE, vols, and DV01 with live
screen values. The architecture is the point.
================================================================================
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from math import sqrt, pi, exp

# ----------------------------------------------------------------------------- 
# CONTRACT SPECS (TY = 10Y T-Note future)  — verify against CME before trusting
# -----------------------------------------------------------------------------
TICK_VALUE   = 15.625      # $ per 1/64 tick of TY (1 point = $1,000; 1/64 = $15.625)
PRICE_PER_PT = 1000.0      # $ per 1.0 price point of TY
CONTRACTS    = 1000        # straddles (1000 calls + 1000 puts); set from premium budget
TRADING_DAYS = 252.0

# -----------------------------------------------------------------------------
# 1. BACHELIER (NORMAL-VOL) PRICER  — vol is in PRICE points/yr (sigma_N)
#    For an option on a future, forward F = current future price; r-discount ~1 for
#    short tenor (set df=1). Greeks returned per ONE option, in price-point units.
# -----------------------------------------------------------------------------
def _phi(x):  return exp(-0.5 * x * x) / sqrt(2 * pi)
def _Phi(x):  return 0.5 * (1.0 + _erf(x / sqrt(2)))
def _erf(x):
    # Abramowitz-Stegun erf, vectorizable via numpy if needed
    t = 1.0 / (1.0 + 0.3275911 * abs(x))
    y = 1.0 - (((((1.061405429*t - 1.453152027)*t) + 1.421413741)*t - 0.284496736)*t + 0.254829592)*t*exp(-x*x)
    return y if x >= 0 else -y

def bachelier(F, K, T, sigma_N, call=True, df=1.0):
    """Price + Greeks (per one option) under normal vol. sigma_N in price points/yr."""
    if T <= 0:
        intrinsic = max(F - K, 0.0) if call else max(K - F, 0.0)
        return dict(price=df*intrinsic, delta=(1.0 if (call and F>K) else (-1.0 if (not call and F<K) else 0.0)),
                    gamma=0.0, vega=0.0, theta=0.0)
    s = sigma_N * sqrt(T)
    d = (F - K) / s
    price = df * (((F - K) * _Phi(d) if call else (K - F) * _Phi(-d)) + s * _phi(d))
    delta = df * (_Phi(d) if call else _Phi(d) - 1.0)
    gamma = df * _phi(d) / s
    vega  = df * sqrt(T) * _phi(d)                 # per 1.0 of sigma_N (price-pt vol)
    theta = -df * sigma_N * _phi(d) / (2 * sqrt(T))  # per year
    return dict(price=price, delta=delta, gamma=gamma, vega=vega, theta=theta)

def straddle(F, K, T, sigma_N, df=1.0):
    c = bachelier(F, K, T, sigma_N, True, df)
    p = bachelier(F, K, T, sigma_N, False, df)
    return {g: c[g] + p[g] for g in c}   # straddle greeks = call + put

# -----------------------------------------------------------------------------
# 2. PATH + HEDGE SCHEDULE
# -----------------------------------------------------------------------------
@dataclass
class DayEvent:
    label: str
    move_pts: float = 0.0          # realized price move THAT day (points), signed
    implied_change: float = 0.0    # change in sigma_N applied at end of day
    hedge: str = "band"            # "band" (loose), "bracket" (flatten pre+post), "none" (let run)
    is_expiry: bool = False

@dataclass
class TradeConfig:
    F0: float = 112.00             # entry TY price (placeholder)
    sigma_N0: float = 0.95         # entry normal vol in PRICE points/yr (~ maps to ~7bp/day; placeholder)
    days_to_expiry: int = 9        # at entry
    contracts: int = CONTRACTS
    band_pts: float = 0.05         # delta rebalance band (in straddle-delta units)
    cost_ticks: float = 0.5        # round-trip hedge cost per future, in ticks

def run_path(cfg: TradeConfig, schedule: list[DayEvent], verbose=True):
    """
    Walk the straddle through the schedule. Each day: apply the day's price move,
    decay T by 1 day, apply implied change, then hedge per the day's policy.
    Tracks option MTM P&L, hedge (future) P&L, cumulative net delta, and costs.
    """
    F = cfg.F0
    sig = cfg.sigma_N0
    T = cfg.days_to_expiry / TRADING_DAYS
    n = cfg.contracts
    K = round(cfg.F0 * 4) / 4    # ATM strike snapped to 1/4 pt (TY strikes ~ in 1/4)

    # initial straddle value + delta hedge to flat
    g = straddle(F, K, T, sig)
    opt_val0 = g["price"] * n * PRICE_PER_PT
    hedge_pos = -g["delta"] * n          # futures held (delta-hedge to zero); + = long futures
    cash_costs = abs(hedge_pos) * cfg.cost_ticks * TICK_VALUE  # entry hedge cost
    rows = []
    prev_F = F

    for ev in schedule:
        # 1) market move for the day
        F = F + ev.move_pts
        # 2) time decay
        T = max(T - 1.0 / TRADING_DAYS, 0.0)
        # 3) implied vol change
        sig = max(sig + ev.implied_change, 1e-6)

        # hedge P&L from the future position held INTO this move
        hedge_pnl_today = hedge_pos * (F - prev_F) * PRICE_PER_PT

        # reprice straddle
        g = straddle(F, K, T, sig) if not ev.is_expiry else straddle(F, K, 0.0, sig)
        opt_val = g["price"] * n * PRICE_PER_PT
        net_delta = g["delta"] * n + hedge_pos   # straddle delta + futures hedge

        # 4) hedge policy
        traded = 0.0
        if ev.hedge == "bracket":
            # flatten fully (pre + post handled as one end-of-event reset here)
            traded = -net_delta
        elif ev.hedge == "band":
            if abs(net_delta) > cfg.band_pts * n:
                traded = -net_delta
        elif ev.hedge == "none":
            traded = 0.0
        hedge_pos += traded
        cash_costs += abs(traded) * cfg.cost_ticks * TICK_VALUE

        rows.append(dict(day=ev.label, F=round(F,4), sigma_N=round(sig,4),
                         opt_val=round(opt_val), hedge_pnl_today=round(hedge_pnl_today),
                         net_delta_pre_hedge=round(g["delta"]*n + (hedge_pos-traded)),
                         futures_after=round(hedge_pos), gamma=round(g["gamma"]*n,1),
                         vega=round(g["vega"]*n*PRICE_PER_PT)))
        prev_F = F

    # total P&L = option MTM change + cumulative hedge P&L - costs
    opt_pnl = rows[-1]["opt_val"] - opt_val0
    hedge_pnl = sum(r["hedge_pnl_today"] for r in rows)
    total = opt_pnl + hedge_pnl - cash_costs
    summary = dict(opt_pnl=round(opt_pnl), hedge_pnl=round(hedge_pnl),
                   costs=round(cash_costs), total_pnl=round(total),
                   premium_at_risk=round(opt_val0))
    if verbose:
        print(f"  entry straddle premium (max loss): ${opt_val0:,.0f}")
        for r in rows:
            print(f"   {r['day']:<14} F={r['F']:.3f}  sig={r['sigma_N']:.3f}  "
                  f"optMTM=${r['opt_val']:>10,}  hedgePnL=${r['hedge_pnl_today']:>9,}  "
                  f"netDelta={r['net_delta_pre_hedge']:>6}  fut={r['futures_after']:>6}")
        print(f"  --> option P&L ${summary['opt_pnl']:,} | hedge P&L ${summary['hedge_pnl']:,} | "
              f"costs ${summary['costs']:,} | TOTAL ${summary['total_pnl']:,}")
    return rows, summary

# =============================================================================
# 3b. DATA LAYER — connect fomc_features.parquet / fomc_backtest_results.parquet
# =============================================================================
# Unit chain (GK yield-vol → TY price pts):
#   gk_vol_pp ÷ 100 → σ_annual  ÷ √252 → σ_daily  × yield × 100 → bps  × dv01 → pts
#
# Sanity check: hist-mean gk=32.3pp → 9.1bps @ 4.5% → 0.59pts ≈ Bloomberg 0.45pts (signal_mult≈1.3 ✓)

class FOMCDataLayer:
    """Converts fomc_features.parquet / fomc_backtest_results.parquet to sim inputs."""

    GK_YIELD_PCT  = 4.5    # default current 10Y yield (%) — update from live screen
    DV01_PER_BP   = 0.065  # TY pts per 1bp yield move (≈ 7.5yr duration)

    @staticmethod
    def gk_to_ty_pts(gk_vol_pp: float,
                     yield_pct: float   = None,
                     dv01_per_bp: float = None) -> float:
        """GK annualised yield-vol (pp) → 1-sigma FOMC-day TY price move (pts)."""
        y  = yield_pct   or FOMCDataLayer.GK_YIELD_PCT
        dv = dv01_per_bp or FOMCDataLayer.DV01_PER_BP
        sigma_daily   = (gk_vol_pp / 100.0) / sqrt(252)
        yield_move_bp = sigma_daily * y * 100
        return yield_move_bp * dv

    @staticmethod
    def load_meeting(nlp_parquet: str, meeting_date: str,
                     backtest_parquet: str = None) -> dict:
        """
        Load one FOMC meeting from parquets. Returns all available columns.
        backtest_parquet (fomc_backtest_results.parquet) adds nlp_pred / word_pred.
        """
        import pandas as pd
        df = pd.read_parquet(nlp_parquet)
        df["meeting_date"] = pd.to_datetime(df["meeting_date"])
        row = df[df["meeting_date"] == pd.Timestamp(meeting_date)]
        if row.empty:
            dates = df["meeting_date"].dt.strftime("%Y-%m-%d").tolist()
            raise ValueError(f"No entry for {meeting_date}. Available: {dates}")
        data = row.iloc[0].to_dict()
        if backtest_parquet:
            import os
            if os.path.exists(backtest_parquet):
                bt = pd.read_parquet(backtest_parquet)
                bt["meeting_date"] = pd.to_datetime(bt["meeting_date"])
                bt_row = bt[bt["meeting_date"] == pd.Timestamp(meeting_date)]
                if not bt_row.empty:
                    for k, v in bt_row.iloc[0].to_dict().items():
                        if k in ("nlp_pred", "word_pred") or k not in data:
                            data[k] = v
        return data

    @staticmethod
    def available_meetings(nlp_parquet: str) -> list:
        import pandas as pd
        df = pd.read_parquet(nlp_parquet)
        return sorted(pd.to_datetime(df["meeting_date"]).dt.strftime("%Y-%m-%d").tolist())

    @staticmethod
    def build_trade_config(F0: float = 112.0, sigma_N0: float = 0.95,
                           days_to_expiry: int = 9, contracts: int = 1000,
                           band_pts: float = 0.05, cost_ticks: float = 0.5) -> "TradeConfig":
        return TradeConfig(F0=F0, sigma_N0=sigma_N0, days_to_expiry=days_to_expiry,
                           contracts=contracts, band_pts=band_pts, cost_ticks=cost_ticks)

    @staticmethod
    def build_schedule(n_diffuse: int = 4, n_auction: int = 2,
                       diffuse_move: float = 0.04, auction_move: float = 0.07,
                       fomc_move: float = 0.55, vol_crush: float = 0.25,
                       fomc_label: str = "FOMC") -> list:
        """Build a generic pre-FOMC path from parameters (no hardcoded dates)."""
        sched = []
        for i in range(n_diffuse):
            sign = 1 if i % 2 == 0 else -1
            sched.append(DayEvent(
                label=f"Diffuse-{i+1}",
                move_pts=sign * diffuse_move * (0.8 + 0.4 * (i % 3) / max(2, n_diffuse - 1)),
                implied_change=0.01 if i == n_diffuse - 1 else 0.0,
                hedge="band",
            ))
        for j in range(n_auction):
            sign = 1 if j % 2 == 0 else -1
            sched.append(DayEvent(label=f"Auction-{j+1}", move_pts=sign * auction_move,
                                  hedge="bracket"))
        sched.append(DayEvent(label=fomc_label, move_pts=fomc_move,
                              implied_change=-vol_crush, hedge="none", is_expiry=True))
        return sched


# -----------------------------------------------------------------------------
# 3. EVENT-STUDY FRAMEWORK  — "what if event A happens"
#    Define a BASE schedule, then a set of SCENARIOS each of which mutates one
#    event's move/implied. Re-run, diff vs base, attribute P&L to that event.
# -----------------------------------------------------------------------------
def base_schedule():
    """Earlier-entry version: hedge bands in run-up, BRACKET the auctions
    (flatten pre+post so their directional move nets out), let FOMC run."""
    return [
        DayEvent("Jul17 quiet",  move_pts=+0.03, implied_change=-0.01, hedge="band"),
        DayEvent("Jul20 quiet",  move_pts=-0.05, implied_change=0.0,   hedge="band"),
        DayEvent("Jul22 quiet",  move_pts=+0.04, implied_change=+0.01, hedge="band"),
        DayEvent("Jul24 quiet",  move_pts=-0.02, implied_change=0.0,   hedge="band"),
        DayEvent("Jul27 5Y auc", move_pts=+0.06, implied_change=0.0,   hedge="bracket"),
        DayEvent("Jul28 7Y auc", move_pts=-0.08, implied_change=+0.03, hedge="bracket"),
        DayEvent("Jul29 FOMC",   move_pts=+0.55, implied_change=-0.25, hedge="none", is_expiry=True),
    ]

def scenarios():
    """Each scenario = (name, mutation function on a fresh base schedule)."""
    def mut(fn):
        s = base_schedule(); fn(s); return s
    return {
        "BASE (median FOMC move)":        base_schedule(),
        "A: hawkish surprise, 2x jump":   mut(lambda s: setattr(s[-1], "move_pts", +1.10)),
        "B: dovish surprise, big rally":  mut(lambda s: setattr(s[-1], "move_pts", -1.00)),
        "C: damp squib (no move)":        mut(lambda s: setattr(s[-1], "move_pts", +0.08)),
        "D: 7Y auction tails hard":       mut(lambda s: setattr(s[4 if False else 5], "move_pts", -0.45)),
        "E: implied crushes harder":      mut(lambda s: setattr(s[-1], "implied_change", -0.55)),
        "F: unscheduled shock Jul22":     mut(lambda s: setattr(s[2], "move_pts", -0.70)),
    }

def event_study(cfg: TradeConfig):
    print("\n" + "="*70)
    print("EVENT STUDY — P&L under alternative scenarios (one knob each)")
    print("="*70)
    base_total = None
    results = {}
    for name, sched in scenarios().items():
        print(f"\n[{name}]")
        _, summ = run_path(cfg, sched, verbose=False)
        results[name] = summ["total_pnl"]
        if base_total is None:
            base_total = summ["total_pnl"]
        attribution = summ["total_pnl"] - base_total
        print(f"   total P&L ${summ['total_pnl']:>10,}   "
              f"(vs base: {'+' if attribution>=0 else ''}{attribution:,})   "
              f"opt ${summ['opt_pnl']:,} / hedge ${summ['hedge_pnl']:,} / cost ${summ['costs']:,}")
    print("\n--- scenario P&L table ---")
    for k, v in results.items():
        print(f"   {k:<32} ${v:>12,}")
    return results

# -----------------------------------------------------------------------------
def main():
    cfg = TradeConfig()
    print("="*70)
    print("BASE PATH (earlier-entry, auction-bracketed hedging)")
    print("="*70)
    run_path(cfg, base_schedule(), verbose=True)
    event_study(cfg)
    print("\nNOTE: vols/moves/sizes are PLACEHOLDERS. Replace F0, sigma_N0, "
          "TICK_VALUE, CONTRACTS, and the per-day move/implied assumptions with "
          "live OMON / screen values. Bachelier normal-vol is the right pricer for "
          "TY options; swap to SABR only if you extend to a full-smile/skew trade.")

if __name__ == "__main__":
    main()
