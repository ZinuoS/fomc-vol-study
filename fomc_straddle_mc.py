"""
================================================================================
FOMC EVENT-STRADDLE — MONTE CARLO WRAPPER + BLOOMBERG/BQUANT DATA LAYER
================================================================================
Runs in BQuant on the company laptop. Pulls F0 + implied vol from Bloomberg,
then simulates thousands of price paths through the auctions and the FOMC,
pricing every mark with our OWN Bachelier (normal-vol) calculator, applying the
event-bracketed delta hedge, and drawing the FOMC-day jump from a distribution
whose width is set by the NLP predicted-realized signal.

The thesis, made quantitative:
  * the straddle is PRICED at the implied vol we pay (from Bloomberg);
  * the realized FOMC jump is drawn at signal_mult x the implied event move;
  * signal_mult = 1.0 means the market is right (expect ~ -costs);
  * signal_mult > 1.0 is the NLP edge -> probability of profit rises.

Requires fomc_straddle_sim.py (the Bachelier pricer + hedge primitives) in the
same folder. Data layer degrades gracefully: bql -> xbbg -> manual override.
================================================================================
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field

from fomc_straddle_sim import (bachelier, straddle, PRICE_PER_PT, TICK_VALUE,
                               TRADING_DAYS, FOMCDataLayer)


# ============================================================================
# LAYER A — BLOOMBERG / BQUANT DATA  (runs on company laptop; isolated)
# ----------------------------------------------------------------------------
# Pulls the live inputs. Tries BQuant-native bql, then xbbg, then falls back to
# a ManualInputs object so the MC still runs offline. CONFIRM every ticker/field
# on the terminal (FLDS) before trusting — option ticker strings especially.
# ============================================================================
@dataclass
class MarketInputs:
    F0: float                 # TY future price
    sigma_N0: float           # ATM NORMAL vol, price points / yr (blended, incl. event)
    implied_event_sd: float   # market-priced FOMC-day move, price points (from the vol kink)
    days_to_expiry: int       # trading days entry -> option expiry
    source: str = "manual"


def _normal_iv_from_price(straddle_px_points, F, K, T, lo=0.05, hi=8.0, tol=1e-6):
    """Invert an ATM straddle price to NORMAL vol using our own Bachelier pricer
    (bisection; straddle price is monotincreasing in sigma_N). straddle_px in points."""
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        px = straddle(F, K, T, mid)["price"]
        if abs(px - straddle_px_points) < tol:
            return mid
        if px < straddle_px_points:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def fetch_bloomberg_inputs(future_ticker="TYU6 Comdty",
                           atm_straddle_px_field="PX_LAST",
                           atm_straddle_ticker=None,
                           normal_vol_ticker=None,
                           days_to_expiry=11,
                           implied_event_sd=0.45,
                           manual: MarketInputs | None = None) -> MarketInputs:
    """
    Returns MarketInputs. Order of attempts:
      1. bql (BQuant native)  -- future PX_LAST, and either a listed normal-vol
         field OR an ATM straddle price we invert with our own pricer.
      2. xbbg (blp.bdp)       -- same fields if blpapi is available.
      3. manual override      -- whatever you pass, so the MC always runs.
    CONFIRM field/ticker names on the terminal. implied_event_sd should come from
    your event-isolation vol-kink calc (long-minus-short expiry), not a guess.
    """
    # --- attempt 1: bql ---
    try:
        import bql                                   # noqa
        svc = bql.Service()
        # NOTE: confirm these BQL field names on the terminal; placeholders.
        px = svc.execute(f"get(PX_LAST) for('{future_ticker}')")
        F0 = float(px[0].df()["PX_LAST"].iloc[0])
        if normal_vol_ticker:
            v = svc.execute(f"get(PX_LAST) for('{normal_vol_ticker}')")
            sigma_N0 = float(v[0].df()["PX_LAST"].iloc[0])
        elif atm_straddle_ticker:
            sp = svc.execute(f"get({atm_straddle_px_field}) for('{atm_straddle_ticker}')")
            straddle_px = float(sp[0].df()[atm_straddle_px_field].iloc[0])
            K = round(F0 * 4) / 4
            sigma_N0 = _normal_iv_from_price(straddle_px, F0, K, days_to_expiry / TRADING_DAYS)
        else:
            raise RuntimeError("need normal_vol_ticker or atm_straddle_ticker")
        return MarketInputs(F0, sigma_N0, implied_event_sd, days_to_expiry, "bql")
    except Exception as e:
        print(f"[data] bql unavailable ({type(e).__name__}); trying xbbg...")

    # --- attempt 2: xbbg ---
    try:
        from xbbg import blp                          # noqa
        F0 = float(blp.bdp(future_ticker, "PX_LAST").iloc[0, 0])
        if normal_vol_ticker:
            sigma_N0 = float(blp.bdp(normal_vol_ticker, "PX_LAST").iloc[0, 0])
        elif atm_straddle_ticker:
            straddle_px = float(blp.bdp(atm_straddle_ticker, atm_straddle_px_field).iloc[0, 0])
            K = round(F0 * 4) / 4
            sigma_N0 = _normal_iv_from_price(straddle_px, F0, K, days_to_expiry / TRADING_DAYS)
        else:
            raise RuntimeError("need normal_vol_ticker or atm_straddle_ticker")
        return MarketInputs(F0, sigma_N0, implied_event_sd, days_to_expiry, "xbbg")
    except Exception as e:
        print(f"[data] xbbg unavailable ({type(e).__name__}); using manual inputs.")

    # --- attempt 3: manual ---
    if manual is None:
        manual = MarketInputs(F0=112.00, sigma_N0=0.95, implied_event_sd=implied_event_sd,
                              days_to_expiry=days_to_expiry, source="manual-default")
    return manual


# ============================================================================
# LAYER B — MONTE CARLO ENGINE
# ============================================================================
@dataclass
class MCConfig:
    contracts: int = 1000
    band_pts: float = 0.05          # delta band (straddle-delta units) for "band" hedging
    cost_ticks: float = 0.5         # round-trip future hedge cost, ticks
    auction_mult: float = 1.6       # auction-day move sd as multiple of diffusive daily sd
    vol_crush: float = 0.28         # absolute sigma_N drop applied at/after the FOMC mark
    signal_mult: float = 1.3        # NLP edge: realized FOMC jump sd = signal_mult * implied_event_sd
    n_paths: int = 5000
    seed: int = 11
    # calendar: each trading day from entry+1 to EXIT (we exit end of FOMC day, pre-expiry)
    # kinds: 'diffuse' (band hedge), 'auction' (bracket hedge), 'fomc' (let run, then exit)
    calendar: list = field(default_factory=lambda: [
        ("Jul17", "diffuse"), ("Jul20", "diffuse"), ("Jul21", "diffuse"),
        ("Jul22", "diffuse"), ("Jul23", "diffuse"), ("Jul24", "diffuse"),
        ("Jul27", "auction"), ("Jul28", "auction"), ("Jul29", "fomc"),
    ])


def simulate(mkt: MarketInputs, cfg: MCConfig):
    """Monte Carlo of delta-hedged straddle P&L. Exit at end of FOMC day, with
    residual time value marked at the crushed vol (this is where vega/crush bites)."""
    rng = np.random.default_rng(cfg.seed)
    n = cfg.contracts
    F0 = mkt.F0
    K = round(F0 * 4) / 4
    T0 = mkt.days_to_expiry / TRADING_DAYS
    diffuse_sd = mkt.sigma_N0 * np.sqrt(1.0 / TRADING_DAYS)        # daily diffusive sd (points)
    fomc_sd = cfg.signal_mult * mkt.implied_event_sd               # realized FOMC jump sd
    n_days = len(cfg.calendar)

    entry_px = straddle(F0, K, T0, mkt.sigma_N0)["price"] * n * PRICE_PER_PT

    totals = np.empty(cfg.n_paths)
    opt_pnls = np.empty(cfg.n_paths)
    hedge_pnls = np.empty(cfg.n_paths)

    for p in range(cfg.n_paths):
        F = F0
        sig = mkt.sigma_N0
        T = T0
        # initial delta hedge to flat
        g = straddle(F, K, T, sig)
        fut = -g["delta"] * n
        costs = abs(fut) * cfg.cost_ticks * TICK_VALUE
        hedge_pnl = 0.0
        prev_F = F

        for k, (label, kind) in enumerate(cfg.calendar):
            # draw the day's move
            if kind == "diffuse":
                move = rng.normal(0.0, diffuse_sd)
            elif kind == "auction":
                move = rng.normal(0.0, cfg.auction_mult * diffuse_sd)
            else:  # fomc
                move = rng.normal(0.0, fomc_sd)
            F += move
            T = max(T - 1.0 / TRADING_DAYS, 0.0)

            # hedge P&L from futures held INTO the move
            hedge_pnl += fut * (F - prev_F) * PRICE_PER_PT
            prev_F = F

            # crush implied at/after the FOMC mark
            if kind == "fomc":
                sig = max(sig - cfg.vol_crush, 1e-6)

            # reprice and hedge per policy
            g = straddle(F, K, T, sig)
            net_delta = g["delta"] * n + fut
            if kind == "auction":               # bracket: flatten after the auction move
                traded = -net_delta
            elif kind == "diffuse":             # band
                traded = -net_delta if abs(net_delta) > cfg.band_pts * n else 0.0
            else:                               # fomc: let it run, no re-hedge (we exit next)
                traded = 0.0
            fut += traded
            costs += abs(traded) * cfg.cost_ticks * TICK_VALUE

        # EXIT at end of FOMC day: mark residual time value at crushed vol
        exit_opt = straddle(F, K, T, sig)["price"] * n * PRICE_PER_PT
        opt_pnl = exit_opt - entry_px
        total = opt_pnl + hedge_pnl - costs
        totals[p] = total
        opt_pnls[p] = opt_pnl
        hedge_pnls[p] = hedge_pnl

    return dict(entry_premium=entry_px, totals=totals, opt_pnls=opt_pnls,
                hedge_pnls=hedge_pnls, K=K, fomc_sd=fomc_sd, diffuse_sd=diffuse_sd)


def summarize(res, label=""):
    t = res["totals"]
    out = dict(
        label=label,
        mean=float(np.mean(t)),
        median=float(np.median(t)),
        p5=float(np.percentile(t, 5)),
        p95=float(np.percentile(t, 95)),
        prob_profit=float(np.mean(t > 0)),
        prob_lose_premium=float(np.mean(t < -0.9 * res["entry_premium"])),
        worst=float(np.min(t)),
        best=float(np.max(t)),
        expected_shortfall_5=float(np.mean(t[t <= np.percentile(t, 5)])),
    )
    return out


def signal_sweep(mkt: MarketInputs, cfg: MCConfig, mults=(0.8, 1.0, 1.2, 1.5, 2.0)):
    """P(profit) and mean P&L vs NLP signal strength."""
    rows = []
    for m in mults:
        c = MCConfig(**{**cfg.__dict__, "signal_mult": m})
        res = simulate(mkt, c)
        s = summarize(res, f"signal_mult={m}")
        rows.append((m, s))
    return rows


# ============================================================================
# LAYER C — BACKTEST DATA INTEGRATION + BLOOMBERG vs NLP COMPARISON
# ============================================================================

@dataclass
class BacktestInputs:
    """
    NLP/backtest signal inputs derived from fomc_features.parquet +
    fomc_backtest_results.parquet.  All vol quantities in price points (pts).
    """
    meeting_date:        str
    gk_vol_actual_pp:    float          # realized GK vol on event day (pp annualised)
    nlp_predicted_pp:    float          # Ridge model predicted vol (pp annualised); 0 if N/A
    word_predicted_pp:   float          # word-bag model predicted vol (pp); 0 if N/A
    gk_actual_pts:       float          # gk_vol_actual  → TY price pts (1-sigma event move)
    nlp_predicted_pts:   float          # nlp_predicted  → TY price pts
    word_predicted_pts:  float          # word_predicted → TY price pts
    yield_pct:           float          # 10Y yield used for conversion
    dv01_per_bp:         float          # TY DV01 per bp used for conversion
    nlp_features:        dict = None    # raw NLP features dict (for display)


def load_backtest_inputs(nlp_parquet:      str,
                          meeting_date:     str,
                          backtest_parquet: str  = None,
                          yield_pct:        float = 4.5,
                          dv01_per_bp:      float = 0.065) -> BacktestInputs:
    """
    Load a meeting from parquets and return BacktestInputs with all vol
    quantities already converted to TY price points.
    """
    data = FOMCDataLayer.load_meeting(nlp_parquet, meeting_date, backtest_parquet)

    gk_pp   = float(data.get("gk_vol_10y", 0) or 0)
    nlp_pp  = float(data.get("nlp_pred",   0) or 0)
    word_pp = float(data.get("word_pred",  0) or 0)

    def to_pts(pp):
        return FOMCDataLayer.gk_to_ty_pts(pp, yield_pct, dv01_per_bp) if pp > 0 else 0.0

    nlp_feat_cols = ["word_count_zscore", "novelty_prev", "novelty_window",
                     "guidance_change", "uncertainty_density", "disagree_density",
                     "polarity_hd", "gk_vol_10y"]
    nlp_features = {k: data[k] for k in nlp_feat_cols if k in data}

    return BacktestInputs(
        meeting_date       = meeting_date,
        gk_vol_actual_pp   = gk_pp,
        nlp_predicted_pp   = nlp_pp,
        word_predicted_pp  = word_pp,
        gk_actual_pts      = to_pts(gk_pp),
        nlp_predicted_pts  = to_pts(nlp_pp),
        word_predicted_pts = to_pts(word_pp),
        yield_pct          = yield_pct,
        dv01_per_bp        = dv01_per_bp,
        nlp_features       = nlp_features,
    )


def signal_mult_from_backtest(bt: BacktestInputs,
                               bbg_implied_event_sd: float,
                               use: str = "word") -> tuple:
    """
    Compute signal_mult for the MC from backtest data vs Bloomberg implied.

    use: 'word'    → word-bag prediction (best model)
         'nlp'     → NLP feature model prediction
         'actual'  → actual realized (backtest/hindsight only)

    Returns (signal_mult, predicted_pts, label).
    """
    if use == "word" and bt.word_predicted_pts > 0:
        pts   = bt.word_predicted_pts
        pp    = bt.word_predicted_pp
        label = "word-bag model"
    elif use == "nlp" and bt.nlp_predicted_pts > 0:
        pts   = bt.nlp_predicted_pts
        pp    = bt.nlp_predicted_pp
        label = "NLP feature model"
    else:
        pts   = bt.gk_actual_pts
        pp    = bt.gk_vol_actual_pp
        label = "actual realized (hindsight)"

    mult = pts / bbg_implied_event_sd if bbg_implied_event_sd > 0 else 1.0
    return mult, pts, pp, label


def compare_bloomberg_vs_nlp(mkt_bbg:    MarketInputs,
                              bt:         BacktestInputs,
                              cfg:        MCConfig,
                              use_signal: str  = "word") -> dict:
    """
    Run two MCs side by side:
      Bloomberg (signal_mult=1.0) — market is correctly priced
      NLP signal (signal_mult=X)  — our model's edge

    Returns dict with both result sets + comparison metrics.
    """
    # ── Bloomberg baseline ──────────────────────────────────────────────────
    cfg_bbg = MCConfig(**{**cfg.__dict__, "signal_mult": 1.0})
    res_bbg = simulate(mkt_bbg, cfg_bbg)
    s_bbg   = summarize(res_bbg, "Bloomberg (mkt-implied, signal_mult=1.0)")

    # ── NLP signal scenario ─────────────────────────────────────────────────
    mult, pred_pts, pred_pp, sig_label = signal_mult_from_backtest(
        bt, mkt_bbg.implied_event_sd, use_signal)
    cfg_nlp = MCConfig(**{**cfg.__dict__, "signal_mult": mult})
    res_nlp = simulate(mkt_bbg, cfg_nlp)
    s_nlp   = summarize(res_nlp, f"NLP signal ({sig_label}, signal_mult={mult:.2f})")

    # ── Actual realized (hindsight check, if gk_actual available) ───────────
    res_actual = None
    s_actual   = None
    if bt.gk_actual_pts > 0:
        actual_mult = bt.gk_actual_pts / mkt_bbg.implied_event_sd
        cfg_act     = MCConfig(**{**cfg.__dict__, "signal_mult": actual_mult})
        res_actual  = simulate(mkt_bbg, cfg_act)
        s_actual    = summarize(res_actual,
                                f"Actual realized (hindsight, signal_mult={actual_mult:.2f})")

    edge = {
        "mean_pnl_lift":        s_nlp["mean"] - s_bbg["mean"],
        "prob_profit_lift":     s_nlp["prob_profit"] - s_bbg["prob_profit"],
        "signal_mult":          mult,
        "bbg_implied_event_sd": mkt_bbg.implied_event_sd,
        "nlp_predicted_pts":    pred_pts,
        "nlp_predicted_pp":     pred_pp,
        "signal_label":         sig_label,
    }

    return dict(bbg=s_bbg, nlp=s_nlp, actual=s_actual,
                res_bbg=res_bbg, res_nlp=res_nlp, res_actual=res_actual,
                edge=edge, bt=bt, mkt=mkt_bbg)


def print_comparison(comp: dict) -> None:
    """Pretty-print the Bloomberg vs NLP comparison results."""
    bt    = comp["bt"]
    mkt   = comp["mkt"]
    edge  = comp["edge"]
    print("=" * 74)
    print(f"  BLOOMBERG vs NLP SIGNAL COMPARISON  ·  {bt.meeting_date}")
    print("=" * 74)
    print(f"\n  Market inputs [{mkt.source}]:")
    print(f"    F0={mkt.F0}  σ_N={mkt.sigma_N0:.3f}pts/yr  "
          f"implied event σ={mkt.implied_event_sd:.3f}pts  DTE={mkt.days_to_expiry}")
    print(f"\n  NLP signal [{edge['signal_label']}]:")
    print(f"    predicted vol  = {edge['nlp_predicted_pp']:.1f}pp  "
          f"→  {edge['nlp_predicted_pts']:.3f}pts  "
          f"(signal_mult={edge['signal_mult']:.2f}x)")
    print(f"    actual GK vol  = {bt.gk_vol_actual_pp:.1f}pp  "
          f"→  {bt.gk_actual_pts:.3f}pts"
          + ("  ← hindsight only" if bt.gk_actual_pts else ""))
    print(f"\n  {'Scenario':42s}  {'E[P&L]':>10s}  {'P(profit)':>10s}  "
          f"{'p5':>10s}  {'p95':>10s}")
    print(f"  {'─'*80}")
    for key in ("bbg", "nlp", "actual"):
        s = comp[key]
        if s is None:
            continue
        print(f"  {s['label']:42s}  ${s['mean']:>9,.0f}  "
              f"{s['prob_profit']:>9.1%}  ${s['p5']:>9,.0f}  ${s['p95']:>9,.0f}")
    print(f"\n  Edge (NLP vs Bloomberg baseline):")
    print(f"    E[P&L] lift     = ${edge['mean_pnl_lift']:>+12,.0f}")
    print(f"    P(profit) lift  = {edge['prob_profit_lift']:>+11.1%}")
    print("=" * 74)


def main():
    print("=" * 74)
    print("FOMC EVENT-STRADDLE — MONTE CARLO (Bloomberg inputs -> in-house pricer)")
    print("=" * 74)

    # LAYER A: get inputs (Bloomberg if available, else manual). On BQuant, pass
    # the real future/option tickers; here it falls back to manual defaults.
    mkt = fetch_bloomberg_inputs(future_ticker="TYU6 Comdty",
                                 atm_straddle_ticker=None,  # e.g. build the ATM weekly straddle ticker
                                 days_to_expiry=11,
                                 implied_event_sd=0.45)
    print(f"\ninputs [{mkt.source}]: F0={mkt.F0}  sigma_N0={mkt.sigma_N0:.3f} pts/yr  "
          f"implied_event_sd={mkt.implied_event_sd} pts  DTE={mkt.days_to_expiry}")

    cfg = MCConfig()
    res = simulate(mkt, cfg)
    print(f"\nentry premium (max loss): ${res['entry_premium']:,.0f}   "
          f"strike K={res['K']}   realized FOMC sd={res['fomc_sd']:.3f} pts "
          f"(= {cfg.signal_mult}x implied)")

    s = summarize(res, "base")
    print("\n--- base-case P&L distribution (signal_mult=%.1f, %d paths) ---" % (cfg.signal_mult, cfg.n_paths))
    for k in ("mean", "median", "p5", "p95", "prob_profit", "prob_lose_premium",
              "expected_shortfall_5", "worst", "best"):
        v = s[k]
        print(f"   {k:>20}: {v:,.3f}" if "prob" in k else f"   {k:>20}: ${v:,.0f}")

    print("\n--- SIGNAL SWEEP: P(profit) and mean P&L vs NLP signal strength ---")
    print(f"   {'signal_mult':>12} {'mean P&L':>14} {'P(profit)':>11} {'p5':>12} {'p95':>12}")
    for m, s in signal_sweep(mkt, cfg):
        print(f"   {m:>12.1f} {s['mean']:>14,.0f} {s['prob_profit']:>11.2%} "
              f"{s['p5']:>12,.0f} {s['p95']:>12,.0f}")

    print("\nReads: at signal_mult=1.0 (market is right) the trade ~ loses costs/decay;")
    print("as the NLP-predicted realized move exceeds implied (mult>1), mean P&L and")
    print("P(profit) climb — that lift IS the tradeable edge, quantified.")
    print("\nNOTE: F0/vol are manual fallbacks here. In BQuant, pass real tickers to")
    print("fetch_bloomberg_inputs(); set implied_event_sd from your vol-kink calc;")
    print("confirm contract multipliers and option ticker strings on the terminal.")


if __name__ == "__main__":
    main()
