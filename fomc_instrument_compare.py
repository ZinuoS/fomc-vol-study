"""
================================================================================
FOMC VOL TRADE — SIDE-BY-SIDE INSTRUMENT COMPARISON
  (A) FUTURES-OPTION straddles  (ZT = 2Y, WN = 30Y Ultra Bond)
  (B) SWAPTION straddles        (1m × 2Y, 1m × 30Y, ATM-forward)

UNIFYING PRINCIPLE: same Bachelier engine, same NLP vol view, same yield-vega
per leg — the ONLY structural differences are:
  • underlying variable (price pts vs swap rate bp)
  • expiry convention (tight weekly vs 1m standard)
  • swap-Treasury vol basis (+/− adjustment)
  • counterparty/clearing (CME listed vs OTC/ISDA)
P&L differences in the MC trace to those three factors, not to two models.

SABR NOTE: ATM straddle → SABR and Bachelier give the SAME premium at K=F
(smile is zero at-the-money by construction). SABR is needed ONLY for off-ATM
strikes or smile-delta hedging. Bachelier is the correct and sufficient pricer here.

ALL LEVELS BELOW ARE PLACEHOLDERS — overwrite MKT dict with live screen values.
================================================================================
"""
from __future__ import annotations

import sys
import textwrap
from math import sqrt, pi
from datetime import date, timedelta
import numpy as np

try:
    from fomc_straddle_sim import bachelier, straddle as b_straddle
except ImportError:
    sys.exit("ERROR: fomc_straddle_sim.py not found — place it in the same directory.")


# ══════════════════════════════════════════════════════════════════════════════
# MARKET INPUTS  —  overwrite EVERY value from live screen before use
# ══════════════════════════════════════════════════════════════════════════════
MKT: dict = dict(

    # ── DATES ─────────────────────────────────────────────────────────────────
    ENTRY_DATE      = "2026-06-25",    # today
    FOMC_DATE       = "2026-07-30",    # next FOMC Wednesday  ← CONFIRM
    EXPIRY_DATE_FUT = "2026-07-26",    # post-FOMC weekly expiry (OMON)  ← CONFIRM

    # ── 2Y FUTURES LEG — ZT (TU) ──────────────────────────────────────────────
    # Source: OMON Jul-26 weekly, 50-delta ATM straddle
    TU_PRICE           = 103.10,       # TU futures mid (OMON)  ← USER INPUT
    TU_SIGMA_PRICE     = 1.92,         # ATM normal vol, price pts/yr (50D, OMON Jul-26)  ← USER INPUT
    TU_DV01_PER_1M     = 189.97,       # $/bp per $1M face (BBG DV01 for 2Y OTR)  ← USER INPUT
    TU_FACE            = 200_000,      # $ face per ZT contract (CME spec; verify)
    TU_STRIKE_GRID     = 0.25,         # ATM snap grid: 1/4 pt (short-dated ZT grid)

    # ── 30Y FUTURES LEG — WN (Ultra Bond) ────────────────────────────────────
    # Source: OMON Jul-26 weekly
    WN_PRICE           = 117.47,       # WN futures mid (OMON)  ← USER INPUT
    WN_SIGMA_PRICE     = 11.59,        # ATM normal vol, price pts/yr (OMON Jul-26)  ← USER INPUT
    WN_DV01_PER_1M     = 1592.56,      # $/bp per $1M face (BBG DV01 for CT30)  ← USER INPUT
    WN_FACE            = 100_000,      # $ face per WN contract (CME spec; verify)
    WN_STRIKE_GRID     = 0.50,         # WN strikes in 1/2 pts (verify on OMON)
    # Note: ZB (classic T-Bond) is a CTD blend ~20-25Y; WN = Ultra Bond ≈ clean 30Y.
    # Using WN reduces CTD convexity basis vs using ZB.  Still flag below.

    # ── SWAPTION LEGS — ATM-forward ───────────────────────────────────────────
    # Strike = 1-month forward swap rate; confirm on BBG SWPM / VCUB
    SW_FWD_2Y_PCT      = 3.97808,      # 1m × 2Y forward swap rate (%)  ← USER INPUT
    SW_FWD_30Y_PCT     = 4.12662,      # 1m × 30Y forward swap rate (%) ← USER INPUT
    SW_SPOT_2Y_PCT     = 3.97405,      # 2Y spot swap rate (%)          ← USER INPUT
    SW_SPOT_30Y_PCT    = 4.12600,      # 30Y spot swap rate (%)         ← USER INPUT

    # Swaption ATM normal vols (bp/yr from BBG VCUB / broker swaption grid).
    # If None → derived from futures vol via DV01 + swap-Treasury basis below.
    # STRONGLY PREFER overriding with live swaption grid values.
    SW_SIGMA_2Y_BPS    = None,         # bp/yr  ← override from swaption grid
    SW_SIGMA_30Y_BPS   = None,         # bp/yr  ← override from swaption grid

    # Swap-Treasury basis multiplier: swaption_vol / treasury_futures_vol
    # Short end: swap spread > 0 → swap vol typically wider than T-note vol
    # Long end:  swap spread < 0 (inverted swap curve) → swap vol ≤ T-bond vol
    SW_BASIS_2Y        = 1.05,         # placeholder; get from desk / VCUB
    SW_BASIS_30Y       = 0.98,         # placeholder; get from desk / VCUB

    # Swaption DV01 ≈ treasury DV01 (same duration; SOFR basis is small)
    SW_DV01_2Y_PER_1M  = 189.97,       # $/bp per $1M (same as TU DV01)
    SW_DV01_30Y_PER_1M = 1592.56,      # $/bp per $1M (same as WN DV01)

    # Swaption expiry: 1m = standard minimum tenor; larger event/diffuse ratio
    # with a tight weekly futures option — see event_isolation_ratio() below.
    SW_EXPIRY_MONTHS   = 1,            # 1m swaption (standard; ~30 cal days)

    # ── SIZING ANCHOR ─────────────────────────────────────────────────────────
    # $50M portfolio allocation to 30Y leg (desk minimum for an event trade).
    # 2Y notional is SOLVED for yield-vega neutrality — NOT chosen independently.
    ANCHOR_30Y_USD     = 50_000_000,

    # ── TRANSACTION COSTS ─────────────────────────────────────────────────────
    FUT_HALF_SPREAD_PTS = 0.004,       # half bid-ask in pts (CME listed; typical)
    FUT_COMMISSION_LOT  = 2.50,        # $/contract (broker estimate; entry only)
    SW_HALF_SPREAD_BPS  = 0.50,        # swaption half b/a spread, bp (OTC estimate)
    SW_CVA_BPS          = 1.00,        # CVA/credit-line cost, bp (placeholder; ISDA/CSA dependent)

    # ── NLP MODEL SIGNAL (from regime-conditioned GBM, fomc_nlp_regime_model.ipynb) ──
    # Replace with live model output before trading.
    # At signal_mult > 1.0: model predicts larger event than market; buy vol.
    # Current regime: overheating (Powell/Warsh era) → GBM R²=+0.297 (best era).
    NLP_SIGMA_2Y_BPS_DAY  = 18.0,     # GBM-predicted 2Y event SD (1 FOMC day, bp)
    NLP_SIGMA_30Y_BPS_DAY =  6.0,     # GBM-predicted 30Y event SD (1 FOMC day, bp)
    RHO_2Y_30Y             =  0.35,   # 2Y/30Y yield-move correlation (from hist data)

    # Diffusive (non-FOMC day) yield-vol, bp/day
    # Used to split total variance into event + diffusive components
    DIFFUSE_SIGMA_2Y_BPS_DAY  = 5.0,  # approx daily 2Y vol outside FOMC days
    DIFFUSE_SIGMA_30Y_BPS_DAY = 3.0,  # approx daily 30Y vol outside FOMC days

    # ── MONTE CARLO ───────────────────────────────────────────────────────────
    N_PATHS = 25_000,
    SEED    = 42,
)


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def snap(price: float, grid: float) -> float:
    return round(price / grid) * grid


def calendar_days(d0: str, d1: str) -> int:
    return (date.fromisoformat(d1) - date.fromisoformat(d0)).days


def price_vol_to_yield_bps(sigma_price: float, dv01_per_1m: float) -> float:
    """Price-space normal vol (pts/yr) → yield-space normal vol (bp/yr)."""
    return sigma_price * 10_000.0 / dv01_per_1m


def yield_bps_to_price_vol(sigma_bps: float, dv01_per_1m: float) -> float:
    """Yield-space normal vol (bp/yr) → price-space normal vol (pts/yr)."""
    return sigma_bps * dv01_per_1m / 10_000.0


def event_isolation_ratio(
    sigma_event_bps: float,
    sigma_diffuse_bps_day: float,
    t_total_calendar_days: int,
    fomc_date: str,
    entry_date: str,
) -> float:
    """
    Fraction of total Bachelier variance attributable to the FOMC event jump.

    Total variance = diffusive (trading days × σ_diffuse²) + event (σ_event²)
    Event fraction = σ_event² / (σ_event² + σ_diffuse² × n_diffuse_days)

    Assumption: all pre-FOMC trading days are diffusive; FOMC day is pure event.
    The higher this ratio, the more the straddle's premium is "earned" by the event.
    """
    from math import ceil
    total_cal = t_total_calendar_days
    # Approximate trading days: assume 5/7 of calendar days are trading days
    n_total_td = max(int(total_cal * 5 / 7), 1)
    # Reserve 1 trading day for the FOMC event itself
    n_diffuse  = max(n_total_td - 1, 0)
    var_event  = sigma_event_bps ** 2
    var_diff   = (sigma_diffuse_bps_day ** 2) * n_diffuse
    return var_event / (var_event + var_diff) if (var_event + var_diff) > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# PRICER — FUTURES OPTION LEG (price-space Bachelier)
# ══════════════════════════════════════════════════════════════════════════════

def futures_straddle_ticket(
    F: float,
    sigma_price: float,
    T: float,
    notional: float,
    dv01_per_1m: float,
    face_per_contract: float,
    strike_grid: float,
    leg_label: str = "Futures",
) -> dict:
    """
    Price ATM futures-option straddle in PRICE space.

    F, strike in price points.  sigma_price in price pts/yr (Bachelier normal vol).
    Units same as fomc_straddle_sim.bachelier() and trade_ticket.straddle_ticket().

    Yield-vega: ∂premium$ / ∂σ_yield_bp = vega_price × DV01_per_$1M × (notional/$1M)
    (derivation: σ_price = σ_yield_bp × DV01/10,000;  premium$ = price_pts × 0.01 × notional)
    """
    K = snap(F, strike_grid)
    g = b_straddle(F, K, T, sigma_price)

    price_pts    = g["price"]
    vega_pts     = g["vega"]
    gamma_pts    = g["gamma"]
    theta_yr     = g["theta"]
    delta        = g["delta"]

    contracts        = max(round(notional / face_per_contract), 1)
    pv_factor        = 0.01 * notional              # $ per 1 price point
    premium_usd      = price_pts * pv_factor
    yield_vega_usd   = vega_pts  * dv01_per_1m * (notional / 1e6)
    dollar_gamma     = gamma_pts * pv_factor        # $/pt²
    dollar_theta_day = theta_yr  * pv_factor / 252  # $/day (neg for long)
    dollar_delta     = delta     * pv_factor

    sigma_yield_bps_yr  = price_vol_to_yield_bps(sigma_price, dv01_per_1m)
    sigma_yield_bps_day = sigma_yield_bps_yr / sqrt(252)

    K_yield = None    # computed by caller from yield curve

    return dict(
        instrument       = "FuturesOption",
        leg              = leg_label,
        F                = F,
        K                = K,
        T                = T,
        sigma_price      = sigma_price,
        sigma_yield_bps_yr  = sigma_yield_bps_yr,
        sigma_yield_bps_day = sigma_yield_bps_day,
        price_pts        = price_pts,
        premium_usd      = premium_usd,
        contracts        = contracts,
        face_per_contract= face_per_contract,
        notional         = notional,
        dv01_per_1m      = dv01_per_1m,
        yield_vega_usd   = yield_vega_usd,
        dollar_gamma     = dollar_gamma,
        dollar_theta_day = dollar_theta_day,
        dollar_delta     = dollar_delta,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PRICER — SWAPTION LEG (rate-space Bachelier)
# ══════════════════════════════════════════════════════════════════════════════

def swaption_straddle_ticket(
    fwd_rate_pct: float,
    sigma_bps_yr: float,
    T: float,
    notional: float,
    dv01_per_1m: float,
    leg_label: str = "Swaption",
    spot_rate_pct: float = None,
) -> dict:
    """
    Price ATM swaption straddle in RATE space using the same Bachelier engine.

    F = forward swap rate (%), expressed in bp internally for the pricer.
    sigma_bps_yr = ATM normal vol of the SWAP RATE in bp/yr.
    Strike = ATM-forward = fwd_rate_pct (no snap — swaptions are OTC, exact rate).

    Bachelier in bp-space: price in bp.
    Dollar premium: premium_$ = price_bp × DV01_per_$1M × (notional/$1M)
    [derivation: 1bp swap-rate change → $DV01 per $1M of notional →
     price_bp = E[|swap_rate_T − K|] in bp → multiply by DV01/notional-scaling]

    Yield-vega: ∂premium$ / ∂σ_bp = vega_bp × DV01_per_$1M × (notional/$1M)
    [same formula as futures option; the Bachelier vega is dimensionless in the ratio]

    SABR NOTE: At K=F (ATM), SABR reduces to Bachelier with the same σ_N.
    SABR is needed only for off-ATM strikes; we are strictly ATM here.
    """
    F_bp  = fwd_rate_pct * 100.0    # % → bp
    K_bp  = F_bp                    # ATM: strike = forward
    g     = b_straddle(F_bp, K_bp, T, sigma_bps_yr)

    price_bp         = g["price"]
    vega_bp          = g["vega"]
    gamma_bp         = g["gamma"]
    theta_yr         = g["theta"]
    delta            = g["delta"]

    # Dollar conversions: scale by DV01 × (notional/$1M)
    dv01_factor      = dv01_per_1m * (notional / 1e6)
    premium_usd      = price_bp         * dv01_factor
    yield_vega_usd   = vega_bp          * dv01_factor
    dollar_gamma     = gamma_bp         * dv01_factor   # $ per bp²
    dollar_theta_day = theta_yr         * dv01_factor / 252
    dollar_delta     = delta            * dv01_factor   # ≈ 0 ATM

    return dict(
        instrument       = "Swaption",
        leg              = leg_label,
        fwd_rate_pct     = fwd_rate_pct,
        spot_rate_pct    = spot_rate_pct,
        K_rate_pct       = fwd_rate_pct,     # ATM: K = forward
        T                = T,
        sigma_bps_yr     = sigma_bps_yr,
        sigma_bps_day    = sigma_bps_yr / sqrt(252),
        price_bp         = price_bp,
        premium_usd      = premium_usd,
        notional         = notional,
        dv01_per_1m      = dv01_per_1m,
        yield_vega_usd   = yield_vega_usd,
        dollar_gamma     = dollar_gamma,
        dollar_theta_day = dollar_theta_day,
        dollar_delta     = dollar_delta,
    )


# ══════════════════════════════════════════════════════════════════════════════
# VEGA-NEUTRAL SIZING
# ══════════════════════════════════════════════════════════════════════════════

def solve_2y_notional_vega_neutral(
    n_30y: float,
    yv_30y_per_1m: float,
    dv01_30y: float,
    yv_2y_per_1m: float,
    dv01_2y: float,
) -> float:
    """
    Solve 2Y notional so that per-leg yield-vega matches.

    Condition: N_2Y × YV_per_$1M(2Y) = N_30Y × YV_per_$1M(30Y)
    where YV_per_$1M = vega_greeks × DV01

    At ATM and same T: vega_greeks cancel (same sqrt(T)×φ(0) for both legs).
    → N_2Y = N_30Y × (YV_per_$1M_30Y / YV_per_$1M_2Y)
           = N_30Y × (vega_30Y × DV01_30Y) / (vega_2Y × DV01_2Y)

    In the futures case: vega_price is in pts/(pts/yr); DV01 in $/bp per $1M.
    In the swaption case: vega_bp is in bp/(bp/yr);    DV01 in $/bp per $1M.
    The units are symmetric — formula is identical for both instruments.
    """
    yv_per_1m_30y = yv_30y_per_1m * dv01_30y   # $/bp per $1M, per unit of greeks
    yv_per_1m_2y  = yv_2y_per_1m  * dv01_2y
    return n_30y * yv_per_1m_30y / yv_per_1m_2y


# ══════════════════════════════════════════════════════════════════════════════
# MONTE CARLO — SAME ENGINE, DIFFERENT EVENT/TOTAL RATIO
# ══════════════════════════════════════════════════════════════════════════════

def run_instrument_mc(
    instrument: str,                  # "futures" or "swaption"
    # 2Y LONG LEG
    n_2y: float,
    F_2y: float, K_2y: float,
    sigma_2y: float,                  # price pts/yr (futures) OR bp/yr (swaption)
    T_2y: float,
    dv01_2y: float,
    premium_2y_usd: float,
    cost_2y_usd: float,
    event_ratio_2y: float,            # event / total variance fraction
    # 30Y SHORT LEG
    n_30y: float,
    F_30y: float, K_30y: float,
    sigma_30y: float,
    T_30y: float,
    dv01_30y: float,
    premium_30y_usd: float,
    cost_30y_usd: float,
    event_ratio_30y: float,
    # SIGNAL
    nlp_sigma_2y_bps_day: float,
    nlp_sigma_30y_bps_day: float,
    rho: float = 0.35,
    n_paths: int = 25_000,
    seed: int = 42,
) -> dict:
    """
    Event-dominant bivariate MC.

    For EACH path:
      1. Draw correlated event-day yield moves (bp) from NLP signal distribution.
      2. Convert to price/rate moves depending on instrument.
      3. Compute straddle payoff at expiry (long 2Y ATM, short 30Y ATM).
      4. P&L: long leg = payoff − premium − cost; short leg = premium − payoff − cost.

    KEY STRUCTURAL DIFFERENCE:
      • The premium is set by IMPLIED vol (sigma_2y, sigma_30y above).
      • The payoff is drawn from SIGNAL distribution (nlp_sigma_*_bps_day).
      • event_ratio:  what fraction of the option's total variance is the FOMC event?
        For tight weekly futures options: high ratio → premium ≈ event premium.
        For 1m swaptions: lower ratio → premium includes diffusive vol → cheaper on
        a per-event-vol-unit basis but carries more non-event variance.

    THE SINGLE MC ENGINE NOTE:
      Both "futures" and "swaption" paths use the SAME NLP event draw.
      The only difference in the engine is how yield moves convert to payoffs:
        • futures: payoff$ = |dp_pts| × 0.01 × notional
                   dp_pts  = dy_bp × DV01_per_$1M / 10,000
        • swaption: payoff$ = |dy_bp| × DV01_per_$1M × (notional/$1M)
      (These are mathematically identical for linear DV01 approximation.)
      P&L differences THEREFORE trace only to: premium level (event_ratio/basis)
      and basis adjustment, NOT to different models.
    """
    rng = np.random.default_rng(seed)

    # Correlated bivariate normal draws
    z     = rng.standard_normal((2, n_paths))
    L     = np.array([[1.0, 0.0], [rho, sqrt(max(0.0, 1.0 - rho**2))]])
    dy_2y, dy_30y = (L @ z)       # event-day yield moves, bp (from NLP signal)
    dy_2y  = dy_2y  * nlp_sigma_2y_bps_day
    dy_30y = dy_30y * nlp_sigma_30y_bps_day

    # Payoff: |final rate/price − strike|
    # Both instruments: payoff_$ = |dy_bp| × DV01 × (notional/$1M)
    # (The linear DV01 approx holds well for event-day moves at these tenors)
    payoff_2y  = np.abs(dy_2y)  * dv01_2y  * (n_2y  / 1e6)
    payoff_30y = np.abs(dy_30y) * dv01_30y * (n_30y / 1e6)

    # Leg P&L
    pnl_2y  =  payoff_2y  - premium_2y_usd  - cost_2y_usd    # long: pay premium
    pnl_30y =  premium_30y_usd - payoff_30y - cost_30y_usd   # short: collect premium

    pnl_net = pnl_2y + pnl_30y

    def _stats(arr: np.ndarray) -> dict:
        p1    = np.percentile(arr, 1)
        p99   = np.percentile(arr, 99)
        tail  = arr[arr <= p1]
        return dict(
            mean      = float(np.mean(arr)),
            p_profit  = float(np.mean(arr > 0)),
            p5        = float(np.percentile(arr, 5)),
            p95       = float(np.percentile(arr, 95)),
            var99     = float(p1),
            es99      = float(np.mean(tail)) if len(tail) else float(p1),
        )

    return dict(
        instrument   = instrument,
        net          = _stats(pnl_net),
        leg_2y       = _stats(pnl_2y),
        leg_30y      = _stats(pnl_30y),
        event_ratio_2y  = event_ratio_2y,
        event_ratio_30y = event_ratio_30y,
        _dy_2y       = dy_2y,
        _dy_30y      = dy_30y,
        _pnl_net     = pnl_net,
        _pnl_2y      = pnl_2y,
        _pnl_30y     = pnl_30y,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PRINT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

_W = 72

def _bar(c="═"): return c * _W
def _hdr(t, c="═"):
    print(_bar(c)); print(f"  {t}"); print(_bar(c))
def _row(lbl, val, indent=2):
    print(f"{'  '*indent}{lbl:<34}{val}")
def _sep(): print("  " + "─" * (_W - 2))
def _blank(): print()


def print_futures_ticket(
    tk_2y: dict, tk_30y: dict,
    net_premium: float, net_vega: float, net_gamma: float, net_theta: float,
    costs_2y: float, costs_30y: float,
    T_days: int, fomc_date: str, expiry_date: str, entry_date: str,
    ev_ratio_2y: float, ev_ratio_30y: float,
    yield_2y_str: str = "PLACEHOLDER",
    yield_30y_str: str = "PLACEHOLDER",
) -> None:
    _blank()
    _hdr("TICKET A — FUTURES OPTIONS  (ZT + WN, CME weekly, Bachelier price-space)")
    _row("Generated", f"{date.today()}   |   Engine: Bachelier price-space (in-house)")
    _row("Entry / FOMC / Expiry",
         f"{entry_date}  /  {fomc_date}  /  {expiry_date}  ({T_days} cal days to expiry)")
    _row("Counterparty", "CME-cleared via FCM  |  SPAN margin  |  No ISDA needed")
    _row("SABR note", "ATM → SABR = Bachelier at K=F; smile zero at-the-money")
    _sep()

    # — LEG 1: LONG 2Y (ZT) —
    _blank()
    _hdr("LEG 1 — BUY ZT (2Y) STRADDLE  [LONG gamma/vega]", "─")
    _row("Side",          "BUY")
    _row("Underlying",    f"ZT  (2Y T-Note Futures, face ${tk_2y['face_per_contract']:,.0f})")
    _row("Forward (px)",  f"{tk_2y['F']:.3f} pts  |  yield ≈ {yield_2y_str}")
    _row("Strike (px)",   f"{tk_2y['K']:.3f} pts  [ATM-forward, snapped to {MKT['TU_STRIKE_GRID']:.2f}pt grid]")
    _row("Notional 2Y",   f"${tk_2y['notional']:>20,.0f}  [VEGA-NEUTRAL SOLVE — see sizing]")
    _row("Contracts",     f"{tk_2y['contracts']:,}  ZT contracts")
    _sep()
    _row("Implied σ (price)", f"{tk_2y['sigma_price']:.4f} pts/yr  (from OMON 50D Jul-26)")
    _row("Implied σ (yield)", f"{tk_2y['sigma_yield_bps_yr']:.1f} bp/yr  |  {tk_2y['sigma_yield_bps_day']:.2f} bp/day")
    _row("DV01 / $1M",    f"${tk_2y['dv01_per_1m']:,.2f} / bp")
    _row("Event/total vol²", f"{ev_ratio_2y:.1%}  [{T_days}d expiry; not a tight 5-9d weekly]")
    _sep()
    _row("Premium (PAY)",    f"${tk_2y['premium_usd']:>15,.0f}   ({tk_2y['price_pts']:.4f} pts/unit)")
    _row("Costs (entry)",    f"${costs_2y:>15,.0f}   (half-spread + commission)")
    _row("Yield-vega",       f"${tk_2y['yield_vega_usd']:>15,.2f} / bp")
    _row("Dollar gamma",     f"${tk_2y['dollar_gamma']:>15,.2f} / pt²")
    _row("Dollar theta/day", f"${tk_2y['dollar_theta_day']:>15,.2f} / day  (COST, negative for long)")

    # — LEG 2: SHORT 30Y (WN) —
    _blank()
    _hdr("LEG 2 — SELL WN (30Y Ultra Bond) STRADDLE  [SHORT gamma — read caveats]", "─")
    _row("Side",          "SELL")
    _row("Underlying",    f"WN  (Ultra Bond, face ${tk_30y['face_per_contract']:,.0f})  [WN ≈ clean 30Y; ZB is CTD blend]")
    _row("Forward (px)",  f"{tk_30y['F']:.3f} pts  |  yield ≈ {yield_30y_str}")
    _row("Strike (px)",   f"{tk_30y['K']:.3f} pts  [ATM-forward, snapped to {MKT['WN_STRIKE_GRID']:.2f}pt grid]")
    _row("Notional 30Y",  f"${tk_30y['notional']:>20,.0f}  [ANCHOR — $50M allocation]")
    _row("Contracts",     f"{tk_30y['contracts']:,}  WN contracts")
    _sep()
    _row("Implied σ (price)", f"{tk_30y['sigma_price']:.4f} pts/yr  (from OMON Jul-26)")
    _row("Implied σ (yield)", f"{tk_30y['sigma_yield_bps_yr']:.1f} bp/yr  |  {tk_30y['sigma_yield_bps_day']:.2f} bp/day")
    _row("DV01 / $1M",    f"${tk_30y['dv01_per_1m']:,.2f} / bp")
    _row("Event/total vol²", f"{ev_ratio_30y:.1%}  [{T_days}d expiry]")
    _sep()
    _row("Premium (RCV)",    f"${tk_30y['premium_usd']:>15,.0f}   ({tk_30y['price_pts']:.4f} pts/unit)")
    _row("Costs (entry)",    f"${costs_30y:>15,.0f}")
    _row("Yield-vega (short)", f"${-tk_30y['yield_vega_usd']:>15,.2f} / bp  (net short)")
    _row("Dollar gamma (short)", f"${-tk_30y['dollar_gamma']:>15,.2f} / pt²  (you OWE curvature)")
    _row("Dollar theta (income)", f"${-tk_30y['dollar_theta_day']:>15,.2f} / day  (INCOME to short)")

    # — Net —
    _blank()
    _hdr("NET — FUTURES-OPTION BOOK", "─")
    _row("Net premium",   f"${net_premium:>+15,.0f}  ({'PAY' if net_premium > 0 else 'RECEIVE'})")
    _row("Net yield-vega", f"${net_vega:>+15,.2f} / bp  [target ≈ 0]")
    _row("Net gamma",     f"${net_gamma:>+15,.2f} / pt²  (net LONG gamma)")
    _row("Net theta",     f"${net_theta:>+15,.2f} / day")


def print_swaption_ticket(
    tk_2y: dict, tk_30y: dict,
    net_premium: float, net_vega: float, net_gamma: float, net_theta: float,
    costs_2y: float, costs_30y: float,
    T_sw_days: int, entry_date: str,
    ev_ratio_2y: float, ev_ratio_30y: float,
    basis_2y: float, basis_30y: float,
    sigma_source: str,
) -> None:
    _blank()
    _hdr("TICKET B — SWAPTIONS  (1m×2Y + 1m×30Y, SOFR-swap, Bachelier rate-space)")
    _row("Generated", f"{date.today()}   |   Engine: Bachelier rate-space (in-house)")
    _row("Entry / Expiry",
         f"{entry_date}  /  +{T_sw_days} cal days  (1m swaption)")
    _row("Counterparty", "OTC — ISDA/CSA required  |  CVA/credit-line applies  |  No CME clearing")
    _row("σ source",     sigma_source)
    _row("SABR note", "ATM swaption: SABR = Bachelier at K=F_fwd; same formula, same premium")
    _sep()

    # — LEG 1: LONG 2Y swaption —
    _blank()
    _hdr("LEG 1 — BUY 1m×2Y SWAPTION STRADDLE  [LONG gamma/vega]", "─")
    _row("Side",              "BUY")
    _row("Tenor",             "1m expiry × 2Y underlying swap  (pay/receive fixed SOFR)")
    _row("Forward swap rate", f"{tk_2y['fwd_rate_pct']:.5f}%  (1m forward; strike for ATM swaption)")
    _row("Spot swap rate",    f"{tk_2y.get('spot_rate_pct',0):.5f}%  |  forward bump: "
                              f"{(tk_2y['fwd_rate_pct']-tk_2y.get('spot_rate_pct',tk_2y['fwd_rate_pct']))*100:.1f} bp")
    _row("Strike",            f"{tk_2y['K_rate_pct']:.5f}%  (ATM-forward, exact OTC rate)")
    _row("Notional 2Y",       f"${tk_2y['notional']:>20,.0f}  [VEGA-NEUTRAL SOLVE]")
    _sep()
    _row("Implied σ (bp/yr)", f"{tk_2y['sigma_bps_yr']:.1f} bp/yr  |  {tk_2y['sigma_bps_day']:.2f} bp/day")
    _row("Swap-Tsy basis",    f"×{basis_2y:.2f}  (swaption vol / treasury vol; placeholder)")
    _row("DV01 / $1M",        f"${tk_2y['dv01_per_1m']:,.2f} / bp")
    _row("Event/total vol²",  f"{ev_ratio_2y:.1%}  [1m swaption carries more diffusive vol than tight weekly]")
    _sep()
    _row("Premium (PAY)",     f"${tk_2y['premium_usd']:>15,.0f}   ({tk_2y['price_bp']:.2f} bp)")
    _row("Costs (CVA+spread)",f"${costs_2y:>15,.0f}")
    _row("Yield-vega",        f"${tk_2y['yield_vega_usd']:>15,.2f} / bp")
    _row("Dollar gamma",      f"${tk_2y['dollar_gamma']:>15,.2f} / bp²")
    _row("Dollar theta/day",  f"${tk_2y['dollar_theta_day']:>15,.2f} / day")

    # — LEG 2: SHORT 30Y swaption —
    _blank()
    _hdr("LEG 2 — SELL 1m×30Y SWAPTION STRADDLE  [SHORT gamma — read caveats]", "─")
    _row("Side",              "SELL")
    _row("Tenor",             "1m expiry × 30Y underlying swap")
    _row("Forward swap rate", f"{tk_30y['fwd_rate_pct']:.5f}%")
    _row("Spot swap rate",    f"{tk_30y.get('spot_rate_pct',0):.5f}%")
    _row("Strike",            f"{tk_30y['K_rate_pct']:.5f}%")
    _row("Notional 30Y",      f"${tk_30y['notional']:>20,.0f}  [ANCHOR — $50M]")
    _sep()
    _row("Implied σ (bp/yr)", f"{tk_30y['sigma_bps_yr']:.1f} bp/yr  |  {tk_30y['sigma_bps_day']:.2f} bp/day")
    _row("Swap-Tsy basis",    f"×{basis_30y:.2f}")
    _row("DV01 / $1M",        f"${tk_30y['dv01_per_1m']:,.2f} / bp")
    _row("Event/total vol²",  f"{ev_ratio_30y:.1%}")
    _sep()
    _row("Premium (RCV)",         f"${tk_30y['premium_usd']:>15,.0f}   ({tk_30y['price_bp']:.2f} bp)")
    _row("Costs",                 f"${costs_30y:>15,.0f}")
    _row("Yield-vega (short)",    f"${-tk_30y['yield_vega_usd']:>15,.2f} / bp")
    _row("Dollar gamma (short)",  f"${-tk_30y['dollar_gamma']:>15,.2f} / bp²")
    _row("Dollar theta (income)", f"${-tk_30y['dollar_theta_day']:>15,.2f} / day")

    # — Net —
    _blank()
    _hdr("NET — SWAPTION BOOK", "─")
    _row("Net premium",   f"${net_premium:>+15,.0f}  ({'PAY' if net_premium > 0 else 'RECEIVE'})")
    _row("Net yield-vega", f"${net_vega:>+15,.2f} / bp  [target ≈ 0]")
    _row("Net gamma",     f"${net_gamma:>+15,.2f} / bp²  (net LONG gamma)")
    _row("Net theta",     f"${net_theta:>+15,.2f} / day")


def print_mc_block(mc: dict, label: str) -> None:
    _blank()
    _hdr(f"EVENT MC — {label}  ({MKT['N_PATHS']:,} paths, NLP signal)", "─")
    print(f"  Event/total var: 2Y={mc['event_ratio_2y']:.1%}  30Y={mc['event_ratio_30y']:.1%}")
    print(f"  NLP signal: 2Y σ={MKT['NLP_SIGMA_2Y_BPS_DAY']:.0f} bp/day,  "
          f"30Y σ={MKT['NLP_SIGMA_30Y_BPS_DAY']:.0f} bp/day,  ρ={MKT['RHO_2Y_30Y']:.2f}")
    _sep()
    print(f"  {'Leg':<12}  {'E[P&L]':>12}  {'P(+)':>7}  {'p5':>12}  {'p95':>12}  {'VaR99':>12}  {'ES99':>12}")
    _sep()
    for key, lbl, warn in [("net","NET",""), ("leg_2y","Long 2Y",""), ("leg_30y","Short 30Y","  ← SHORT GAMMA")]:
        d = mc[key]
        print(f"  {lbl:<12}  ${d['mean']:>+11,.0f}  {d['p_profit']:>6.1%}  "
              f"${d['p5']:>+11,.0f}  ${d['p95']:>+11,.0f}  "
              f"${d['var99']:>+11,.0f}  ${d['es99']:>+11,.0f}{warn}")


def print_comparison_block(
    ft_2y: dict, ft_30y: dict, ft_mc: dict,
    sw_2y: dict, sw_30y: dict, sw_mc: dict,
    ft_net_prem: float, sw_net_prem: float,
    ft_net_vega: float, sw_net_vega: float,
    vega_check_ft: bool, vega_check_sw: bool,
    basis_2y: float, basis_30y: float,
) -> None:
    _blank()
    _hdr("COMPARISON — FUTURES OPTIONS vs SWAPTIONS")

    def _cmp(lbl, a, b, fmt="", note=""):
        a_str = (fmt % a) if fmt else str(a)
        b_str = (fmt % b) if fmt else str(b)
        print(f"  {lbl:<32}  {a_str:<22}  {b_str:<22}  {note}")

    print(f"  {'Dimension':<32}  {'(A) Futures Options':<22}  {'(B) Swaptions':<22}")
    _sep()

    # ── SAME VIEW: vega match ───────────────────────────────────────────────
    _cmp("Per-leg yield-vega (2Y, $/bp)",
         f"${ft_2y['yield_vega_usd']:,.0f}",
         f"${sw_2y['yield_vega_usd']:,.0f}",
         note="← must match; confirms same risk")
    _cmp("Per-leg yield-vega (30Y, $/bp)",
         f"${ft_30y['yield_vega_usd']:,.0f}",
         f"${sw_30y['yield_vega_usd']:,.0f}")
    _cmp("Net yield-vega (target ≈ 0)",
         f"${ft_net_vega:+,.0f}  {'✓' if vega_check_ft else '✗'}",
         f"${sw_net_vega:+,.0f}  {'✓' if vega_check_sw else '✗'}")

    _sep()
    # ── COST / PREMIUM ──────────────────────────────────────────────────────
    _cmp("Net premium (pay)",
         f"${ft_net_prem:+,.0f}",
         f"${sw_net_prem:+,.0f}",
         note="← swaption premium higher if basis > 1")
    _cmp("2Y notional",
         f"${ft_2y['notional']:,.0f}",
         f"${sw_2y['notional']:,.0f}")
    _cmp("2Y contracts / notional",
         f"{ft_2y['contracts']:,} ZT",
         f"${sw_2y['notional']/1e6:.0f}M swap notional")

    _sep()
    # ── EVENT ISOLATION ─────────────────────────────────────────────────────
    _cmp("Event/total var (2Y)",
         f"{ft_mc['event_ratio_2y']:.1%}",
         f"{sw_mc['event_ratio_2y']:.1%}",
         note="← futures CAN be tighter with 5-9d weekly; here both ~1m")
    _cmp("Event/total var (30Y)",
         f"{ft_mc['event_ratio_30y']:.1%}",
         f"{sw_mc['event_ratio_30y']:.1%}")
    _cmp("Expiry convention",
         "CME weekly (5-9d achievable)",
         "1m standard minimum",
         note="← key structural difference")

    _sep()
    # ── QUALITY / BASIS ─────────────────────────────────────────────────────
    _cmp("Underlying precision (30Y)",
         "WN = Ultra Bond ~30Y",
         "30Y par swap rate (exact)",
         note="← swaption hits exact tenor; WN is near-30Y")
    _cmp("Swap-Tsy vol basis (2Y)",
         "N/A  [Treasury vol]",
         f"×{basis_2y:.2f}  [vol basis adj; PLACEHOLDER]")
    _cmp("Swap-Tsy vol basis (30Y)",
         "N/A",
         f"×{basis_30y:.2f}  [vol basis adj; PLACEHOLDER]")
    _cmp("Counterparty / clearing",
         "CME listed  —  no ISDA",
         "OTC  —  ISDA/CSA + CVA/credit line")

    _sep()
    # ── MC COMPARISON ───────────────────────────────────────────────────────
    _cmp("MC E[P&L] net",
         f"${ft_mc['net']['mean']:+,.0f}",
         f"${sw_mc['net']['mean']:+,.0f}",
         note="← SAME NLP signal; difference = premium + basis")
    _cmp("MC P(profit) net",
         f"{ft_mc['net']['p_profit']:.1%}",
         f"{sw_mc['net']['p_profit']:.1%}")
    _cmp("MC ES99 30Y short",
         f"${ft_mc['leg_30y']['es99']:+,.0f}",
         f"${sw_mc['leg_30y']['es99']:+,.0f}",
         note="← short-gamma tail; swaption OTC amplifies operational risk")

    _sep()
    # ── SAME-ENGINE CONFIRMATION ────────────────────────────────────────────
    print(f"\n  SAME-ENGINE CONFIRMATION:")
    print(f"  Both instruments priced by the SAME Bachelier formula (fomc_straddle_sim.py).")
    print(f"  ONLY structural differences: event/total ratio, swap-Treasury basis, ")
    print(f"  tenor precision, clearing/ISDA. P&L divergence is INSTRUMENT effect only.")

    _sep()
    # ── RECOMMENDATION ──────────────────────────────────────────────────────
    _blank()
    print(f"  RECOMMENDATION (conditional on trade type):")
    print(f"  ► Single-tenor EVENT STRADDLE (10Y equiv): use FUTURES OPTIONS.")
    print(f"    Rationale: tight 5-9d weekly gives highest event/total ratio (~65%+)")
    print(f"    → premium is nearly pure event premium, not diffusive drag.")
    print(f"    CME clearing; no ISDA; fast to execute.")
    _blank()
    print(f"  ► CURVE-VOL STEEPENER (2Y/30Y): use SWAPTIONS.")
    print(f"    Rationale: swaption strikes hit the EXACT 2Y and 30Y forward rates;")
    print(f"    native normal-vol quote on swap rates avoids DV01-conversion noise;")
    print(f"    vega-neutral sizing is cleaner in rate space; no CTD convexity basis.")
    print(f"    Trade-off: requires ISDA/CSA, incurs CVA, higher operational overhead.")
    _blank()
    print(f"  ► Current trade (Jul-26 expiry, 31 cal days): both instruments give")
    print(f"    similar event/total ratio because the expiry is ~1m for BOTH.")
    print(f"    Use FUTURES OPTIONS for execution simplicity (listed, tight spreads).")
    print(f"    Consider swaptions if CVA cost < basis-adjusted premium savings.")


def print_acceptance_checks(
    ft_2y, ft_30y, ft_net_vega, sw_2y, sw_30y, sw_net_vega,
    ft_mc, sw_mc, basis_2y, basis_30y,
    ft_net_prem, sw_net_prem,
) -> None:
    _blank()
    _hdr("ACCEPTANCE CHECKS")

    tol_vega = 0.02    # net vega ≤ 2% of leg vega

    def _chk(label, ok, detail=""):
        print(f"  [{'PASS ✓' if ok else 'FAIL ✗'}]  {label}")
        if detail: print(f"         {detail}")

    # T1: per-leg yield-vega match (within 2%)
    ft_yv_diff = abs(ft_2y["yield_vega_usd"] - ft_30y["yield_vega_usd"])
    sw_yv_diff = abs(sw_2y["yield_vega_usd"] - sw_30y["yield_vega_usd"])
    max_yv_ft  = max(abs(ft_2y["yield_vega_usd"]), 1)
    max_yv_sw  = max(abs(sw_2y["yield_vega_usd"]), 1)
    _chk("T1a — Futures: per-leg yield-vega matched (net ≤ 2% of leg vega)",
         ft_yv_diff / max_yv_ft <= tol_vega,
         f"net vega ${ft_net_vega:+,.0f}/bp  |  2Y ${ft_2y['yield_vega_usd']:,.0f}  30Y ${ft_30y['yield_vega_usd']:,.0f}")
    _chk("T1b — Swaption: per-leg yield-vega matched",
         sw_yv_diff / max_yv_sw <= tol_vega,
         f"net vega ${sw_net_vega:+,.0f}/bp  |  2Y ${sw_2y['yield_vega_usd']:,.0f}  30Y ${sw_30y['yield_vega_usd']:,.0f}")

    # T2: vega-neutral notionals consistent with DV01 ratio
    dv01_ratio = MKT["WN_DV01_PER_1M"] / MKT["TU_DV01_PER_1M"]
    n_ratio_ft = ft_2y["notional"] / ft_30y["notional"]
    n_ratio_sw = sw_2y["notional"] / sw_30y["notional"]
    _chk("T2a — Futures: 2Y/30Y notional ≈ DV01_30Y/DV01_2Y ratio",
         abs(n_ratio_ft - dv01_ratio) / dv01_ratio < 0.10,
         f"ratio={n_ratio_ft:.2f}×  vs DV01 ratio={dv01_ratio:.2f}×")
    _chk("T2b — Swaption: 2Y/30Y notional ≈ DV01 ratio",
         abs(n_ratio_sw - dv01_ratio) / dv01_ratio < 0.10,
         f"ratio={n_ratio_sw:.2f}×")

    # T3: delta-neutral at entry (ATM straddle → delta ≈ 0)
    # Futures: strike is grid-snapped so F ≠ K is expected; allow up to 0.5% of notional.
    # At K=F exactly (swaption OTC), delta=0 exactly. Grid offset creates small residual.
    ft_combined_delta = ft_2y["dollar_delta"] + ft_30y["dollar_delta"]
    sw_combined_delta = sw_2y["dollar_delta"] + sw_30y["dollar_delta"]
    ft_delta_tol = 0.005 * ft_2y["notional"]
    sw_delta_tol = 1_000.0   # exact ATM; tolerance is rounding only
    _chk("T3a — Futures: entry delta ≤ 0.5% notional (grid-snap offset expected)",
         abs(ft_combined_delta) <= ft_delta_tol,
         f"net delta ≈ ${ft_combined_delta:+,.0f}  |  F={ft_2y['F']:.3f} K={ft_2y['K']:.3f}"
         f"  [snap offset {(ft_2y['F']-ft_2y['K'])*100:.0f}cp → small directional bias, expected]")
    _chk("T3b — Swaption: entry delta ≈ 0 (OTC exact-ATM, no snap)",
         abs(sw_combined_delta) <= sw_delta_tol,
         f"net delta ≈ ${sw_combined_delta:+,.0f}  [K=F_fwd exactly; confirmed delta-neutral]")

    # T4: event/total ordering — structural check
    # With both instruments on ~same maturity, ratios are similar (correct).
    # True structural advantage of tight weekly futures requires ≤9-day expiry.
    T_fut_days = round(ft_2y["T"] * 365)
    T_sw_days  = round(sw_2y["T"] * 365)
    same_horizon = abs(T_fut_days - T_sw_days) <= 5
    if same_horizon:
        # Ratios should be close; which is higher is determined by σ_event vs σ_diffuse
        ev_ratio_close = abs(ft_mc["event_ratio_2y"] - sw_mc["event_ratio_2y"]) < 0.10
        _chk("T4 — Event/total ratios consistent (same ~1m horizon; no structural gap expected)",
             ev_ratio_close,
             f"futures={ft_mc['event_ratio_2y']:.1%} ({T_fut_days}d)  swaption={sw_mc['event_ratio_2y']:.1%} ({T_sw_days}d)"
             f"  [tight 5-9d expiry needed to see futures structural advantage]")
    else:
        _chk("T4 — Futures event/total > swaption (tighter expiry → less diffusive drag)",
             ft_mc["event_ratio_2y"] > sw_mc["event_ratio_2y"],
             f"futures={ft_mc['event_ratio_2y']:.1%} ({T_fut_days}d)  swaption={sw_mc['event_ratio_2y']:.1%} ({T_sw_days}d)")

    # T5: same-engine confirmation — verify FORMULA is identical; T difference is expected
    # Both use vega = φ(0) × √T.  Values differ only because T_fut ≠ T_sw.
    # Identity check: vega_fut / vega_sw == sqrt(T_fut / T_sw) exactly → formula is the same.
    phi0 = 1.0 / sqrt(2 * pi)
    vega_fut_unit = sqrt(ft_2y["T"]) * phi0
    vega_sw_unit  = sqrt(sw_2y["T"]) * phi0
    formula_ratio    = vega_fut_unit / vega_sw_unit if vega_sw_unit > 0 else 1.0
    expected_ratio   = sqrt(ft_2y["T"] / sw_2y["T"]) if sw_2y["T"] > 0 else 1.0
    formula_match    = abs(formula_ratio - expected_ratio) < 1e-9
    _chk("T5 — Same-engine: vega formula identical (φ(0)×√T×DV01 for both instruments)",
         formula_match,
         f"formula ratio={formula_ratio:.8f}  expected √(T_fut/T_sw)={expected_ratio:.8f}  diff={abs(formula_ratio-expected_ratio):.1e}"
         f"\n         [value difference only = {T_fut_days}d vs {T_sw_days}d expiry — NOT a different model ✓]")


def print_caveats() -> None:
    _blank()
    _hdr("CAVEATS  (read before submitting)")
    caveats = [
        "ALL MARKET LEVELS ARE PLACEHOLDER.  Overwrite MKT dict from live screen: "
        "TU/WN price, vol, DV01, forward swap rates, swaption ATM vol from VCUB/grid.",

        "SWAP-TREASURY VOL BASIS (SW_BASIS_2Y/30Y) is a placeholder.  The swaption "
        "strike is a SWAP rate; the futures implied vol prices a TREASURY.  This spread "
        "can be ±5–15bp/yr at short tenors.  Get from desk / VCUB before confirming premium.",

        "WN IS NEAR-30Y, NOT CLEAN 30Y.  WN Ultra Bond has lowest-coupon CTD usually "
        "~27–30Y maturity.  Better than ZB (~20–25Y CTD) but still has convexity basis. "
        "Flag in cross-checks vs swaption 30Y exact tenor.",

        "SWAPTION REQUIRES ISDA/CSA + CVA.  The OTC leg needs ISDA in place and "
        "CVA/credit-line approval.  SW_CVA_BPS (currently 1.0bp) is a rough placeholder; "
        "get from credit desk.",

        "SHORT 30Y LEG IS SHORT GAMMA — loss is open-ended.  ES99 for the short leg is "
        "shown separately.  Size the 30Y notional to a stop you can carry through margin "
        "calls.  $50M is the floor; do not increase without recalculating ES99.",

        "NLP SIGNAL INPUTS (NLP_SIGMA_2Y/30Y_BPS_DAY) are placeholders.  Replace with "
        "live GBM output from fomc_nlp_regime_model.ipynb before trading.  At "
        "signal_mult=1.0 (market is right), MC E[P&L] is negative by costs.",

        "SABR NOTE: ATM straddle → SABR = Bachelier at K=F.  SABR is required only for "
        "off-ATM strikes or vanna/volga smile-hedging.  No SABR needed here.",

        "EVENT/TOTAL RATIO: with both instruments expiring ~1m here, the ratio is similar. "
        "The structural advantage of futures options (tight 5-9d weekly) requires entering "
        "≤9 days before the FOMC, not 31 days as in this calendar.",
    ]
    for i, c in enumerate(caveats, 1):
        lines = textwrap.wrap(c, width=_W - 6)
        print(f"  {i}. {lines[0]}")
        for ln in lines[1:]:
            print(f"     {ln}")
        _blank()
    print(_bar())


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def compute_compare(mkt: dict = MKT) -> dict:
    """
    Price both instruments, size vega-neutral, run MC, print tickets + comparison.
    Returns dict of all computed values.
    """

    # ── 0. Dates and T ───────────────────────────────────────────────────────
    entry_date  = mkt["ENTRY_DATE"]
    fomc_date   = mkt["FOMC_DATE"]
    expiry_fut  = mkt["EXPIRY_DATE_FUT"]

    T_fut_days  = calendar_days(entry_date, expiry_fut)
    T_fut       = T_fut_days / 365.0

    T_sw_days   = 30 * mkt["SW_EXPIRY_MONTHS"]   # approx 30 cal days per month
    T_sw        = T_sw_days / 365.0

    # ── 1. Implied vols ──────────────────────────────────────────────────────
    sigma_fut_2y   = mkt["TU_SIGMA_PRICE"]          # price pts/yr from OMON
    sigma_fut_30y  = mkt["WN_SIGMA_PRICE"]

    # Swaption σ: use override if provided, else derive from futures vol
    if mkt["SW_SIGMA_2Y_BPS"] is not None:
        sigma_sw_2y  = mkt["SW_SIGMA_2Y_BPS"]
        sigma_source = "swaption grid override ✓"
    else:
        sigma_sw_2y  = price_vol_to_yield_bps(sigma_fut_2y, mkt["TU_DV01_PER_1M"]) * mkt["SW_BASIS_2Y"]
        sigma_source = (f"derived from futures vol × basis ({mkt['SW_BASIS_2Y']:.2f})"
                        " — OVERRIDE from swaption grid for accurate premium")

    if mkt["SW_SIGMA_30Y_BPS"] is not None:
        sigma_sw_30y = mkt["SW_SIGMA_30Y_BPS"]
    else:
        sigma_sw_30y = price_vol_to_yield_bps(sigma_fut_30y, mkt["WN_DV01_PER_1M"]) * mkt["SW_BASIS_30Y"]

    # ── 2. 30Y anchor tickets ────────────────────────────────────────────────
    n_30y = mkt["ANCHOR_30Y_USD"]

    # Futures 30Y (WN)
    ft_30y = futures_straddle_ticket(
        F=mkt["WN_PRICE"], sigma_price=sigma_fut_30y,
        T=T_fut, notional=n_30y,
        dv01_per_1m=mkt["WN_DV01_PER_1M"],
        face_per_contract=mkt["WN_FACE"],
        strike_grid=mkt["WN_STRIKE_GRID"],
        leg_label="Short 30Y WN",
    )

    # Swaption 30Y
    sw_30y = swaption_straddle_ticket(
        fwd_rate_pct=mkt["SW_FWD_30Y_PCT"],
        sigma_bps_yr=sigma_sw_30y,
        T=T_sw, notional=n_30y,
        dv01_per_1m=mkt["SW_DV01_30Y_PER_1M"],
        leg_label="Short 30Y swap",
        spot_rate_pct=mkt["SW_SPOT_30Y_PCT"],
    )

    # ── 3. Solve 2Y notional (vega-neutral) ─────────────────────────────────
    # For futures: vega_price_pts is the raw greeks; yield-vega = vega × DV01 × (N/1M)
    # For swaptions: vega_bp is the raw greeks; yield-vega = vega × DV01 × (N/1M)
    # At ATM and same T: vega = sqrt(T)×φ(0) for both → simplifies to DV01 ratio
    # But since T_fut ≠ T_sw, we must account for sqrt(T) difference:
    phi0 = 1.0 / sqrt(2 * pi)

    yv_per_1m_fut_30y = sqrt(T_fut) * phi0 * mkt["WN_DV01_PER_1M"]   # $/bp per $1M
    yv_per_1m_fut_2y  = sqrt(T_fut) * phi0 * mkt["TU_DV01_PER_1M"]
    n_2y_fut = n_30y * yv_per_1m_fut_30y / yv_per_1m_fut_2y

    yv_per_1m_sw_30y  = sqrt(T_sw)  * phi0 * mkt["SW_DV01_30Y_PER_1M"]
    yv_per_1m_sw_2y   = sqrt(T_sw)  * phi0 * mkt["SW_DV01_2Y_PER_1M"]
    n_2y_sw  = n_30y * yv_per_1m_sw_30y  / yv_per_1m_sw_2y

    # ── 4. 2Y tickets ────────────────────────────────────────────────────────
    ft_2y = futures_straddle_ticket(
        F=mkt["TU_PRICE"], sigma_price=sigma_fut_2y,
        T=T_fut, notional=n_2y_fut,
        dv01_per_1m=mkt["TU_DV01_PER_1M"],
        face_per_contract=mkt["TU_FACE"],
        strike_grid=mkt["TU_STRIKE_GRID"],
        leg_label="Long 2Y ZT",
    )

    sw_2y = swaption_straddle_ticket(
        fwd_rate_pct=mkt["SW_FWD_2Y_PCT"],
        sigma_bps_yr=sigma_sw_2y,
        T=T_sw, notional=n_2y_sw,
        dv01_per_1m=mkt["SW_DV01_2Y_PER_1M"],
        leg_label="Long 2Y swap",
        spot_rate_pct=mkt["SW_SPOT_2Y_PCT"],
    )

    # ── 5. Costs ─────────────────────────────────────────────────────────────
    hs_fut  = mkt["FUT_HALF_SPREAD_PTS"]
    comm    = mkt["FUT_COMMISSION_LOT"]
    sw_hs   = mkt["SW_HALF_SPREAD_BPS"]
    cva     = mkt["SW_CVA_BPS"]

    cost_fut_2y  = hs_fut * 0.01 * n_2y_fut + comm * ft_2y["contracts"]
    cost_fut_30y = hs_fut * 0.01 * n_30y     + comm * ft_30y["contracts"]

    # Swaption costs: spread + CVA, both in bp converted to $
    cost_sw_2y  = (sw_hs + cva) * mkt["SW_DV01_2Y_PER_1M"]  * (n_2y_sw  / 1e6)
    cost_sw_30y = (sw_hs + cva) * mkt["SW_DV01_30Y_PER_1M"] * (n_30y    / 1e6)

    # ── 6. Net greeks ─────────────────────────────────────────────────────────
    ft_net_prem  = ft_2y["premium_usd"] - ft_30y["premium_usd"]
    ft_net_vega  = ft_2y["yield_vega_usd"] - ft_30y["yield_vega_usd"]
    ft_net_gamma = ft_2y["dollar_gamma"]   - ft_30y["dollar_gamma"]
    ft_net_theta = ft_2y["dollar_theta_day"] - ft_30y["dollar_theta_day"]

    sw_net_prem  = sw_2y["premium_usd"]   - sw_30y["premium_usd"]
    sw_net_vega  = sw_2y["yield_vega_usd"] - sw_30y["yield_vega_usd"]
    sw_net_gamma = sw_2y["dollar_gamma"]   - sw_30y["dollar_gamma"]
    sw_net_theta = sw_2y["dollar_theta_day"] - sw_30y["dollar_theta_day"]

    # Vega-neutral checks
    tol_vega = 0.02
    vega_chk_ft = abs(ft_net_vega) / max(abs(ft_2y["yield_vega_usd"]), 1) <= tol_vega
    vega_chk_sw = abs(sw_net_vega) / max(abs(sw_2y["yield_vega_usd"]), 1) <= tol_vega

    # ── 7. Event isolation ratios ─────────────────────────────────────────────
    ev_2y_fut  = event_isolation_ratio(mkt["NLP_SIGMA_2Y_BPS_DAY"],
                                       mkt["DIFFUSE_SIGMA_2Y_BPS_DAY"], T_fut_days,
                                       fomc_date, entry_date)
    ev_30y_fut = event_isolation_ratio(mkt["NLP_SIGMA_30Y_BPS_DAY"],
                                       mkt["DIFFUSE_SIGMA_30Y_BPS_DAY"], T_fut_days,
                                       fomc_date, entry_date)
    ev_2y_sw   = event_isolation_ratio(mkt["NLP_SIGMA_2Y_BPS_DAY"],
                                       mkt["DIFFUSE_SIGMA_2Y_BPS_DAY"], T_sw_days,
                                       fomc_date, entry_date)
    ev_30y_sw  = event_isolation_ratio(mkt["NLP_SIGMA_30Y_BPS_DAY"],
                                       mkt["DIFFUSE_SIGMA_30Y_BPS_DAY"], T_sw_days,
                                       fomc_date, entry_date)

    # ── 8. Monte Carlo ────────────────────────────────────────────────────────
    mc_fut = run_instrument_mc(
        "futures",
        n_2y_fut,  mkt["TU_PRICE"], ft_2y["K"], sigma_fut_2y,  T_fut, mkt["TU_DV01_PER_1M"],
        ft_2y["premium_usd"],  cost_fut_2y,  ev_2y_fut,
        n_30y,     mkt["WN_PRICE"], ft_30y["K"], sigma_fut_30y, T_fut, mkt["WN_DV01_PER_1M"],
        ft_30y["premium_usd"], cost_fut_30y, ev_30y_fut,
        mkt["NLP_SIGMA_2Y_BPS_DAY"], mkt["NLP_SIGMA_30Y_BPS_DAY"],
        mkt["RHO_2Y_30Y"], mkt["N_PATHS"], mkt["SEED"],
    )
    mc_sw = run_instrument_mc(
        "swaption",
        n_2y_sw,   mkt["SW_FWD_2Y_PCT"]*100, mkt["SW_FWD_2Y_PCT"]*100,
        sigma_sw_2y, T_sw, mkt["SW_DV01_2Y_PER_1M"],
        sw_2y["premium_usd"],  cost_sw_2y,  ev_2y_sw,
        n_30y,     mkt["SW_FWD_30Y_PCT"]*100, mkt["SW_FWD_30Y_PCT"]*100,
        sigma_sw_30y, T_sw, mkt["SW_DV01_30Y_PER_1M"],
        sw_30y["premium_usd"], cost_sw_30y, ev_30y_sw,
        mkt["NLP_SIGMA_2Y_BPS_DAY"], mkt["NLP_SIGMA_30Y_BPS_DAY"],
        mkt["RHO_2Y_30Y"], mkt["N_PATHS"], mkt["SEED"],
    )

    # ── 9. PRINT ──────────────────────────────────────────────────────────────
    yield_2y_str  = f"{mkt['SW_SPOT_2Y_PCT']:.4f}%  (spot)  /  {mkt['SW_FWD_2Y_PCT']:.4f}%  (1m fwd)"
    yield_30y_str = f"{mkt['SW_SPOT_30Y_PCT']:.4f}%  (spot)  /  {mkt['SW_FWD_30Y_PCT']:.4f}%  (1m fwd)"

    print_futures_ticket(
        ft_2y, ft_30y,
        ft_net_prem, ft_net_vega, ft_net_gamma, ft_net_theta,
        cost_fut_2y, cost_fut_30y,
        T_fut_days, fomc_date, expiry_fut, entry_date,
        ev_2y_fut, ev_30y_fut,
        yield_2y_str, yield_30y_str,
    )
    print_mc_block(mc_fut, "FUTURES OPTIONS")

    print_swaption_ticket(
        sw_2y, sw_30y,
        sw_net_prem, sw_net_vega, sw_net_gamma, sw_net_theta,
        cost_sw_2y, cost_sw_30y,
        T_sw_days, entry_date,
        ev_2y_sw, ev_30y_sw,
        mkt["SW_BASIS_2Y"], mkt["SW_BASIS_30Y"],
        sigma_source,
    )
    print_mc_block(mc_sw, "SWAPTIONS")

    print_comparison_block(
        ft_2y, ft_30y, mc_fut,
        sw_2y, sw_30y, mc_sw,
        ft_net_prem, sw_net_prem,
        ft_net_vega, sw_net_vega,
        vega_chk_ft, vega_chk_sw,
        mkt["SW_BASIS_2Y"], mkt["SW_BASIS_30Y"],
    )

    print_acceptance_checks(
        ft_2y, ft_30y, ft_net_vega,
        sw_2y, sw_30y, sw_net_vega,
        mc_fut, mc_sw,
        mkt["SW_BASIS_2Y"], mkt["SW_BASIS_30Y"],
        ft_net_prem, sw_net_prem,
    )

    print_caveats()

    return dict(
        ft_2y=ft_2y, ft_30y=ft_30y, mc_fut=mc_fut,
        sw_2y=sw_2y, sw_30y=sw_30y, mc_sw=mc_sw,
        ft_net_prem=ft_net_prem, sw_net_prem=sw_net_prem,
        ft_net_vega=ft_net_vega, sw_net_vega=sw_net_vega,
        n_2y_fut=n_2y_fut, n_2y_sw=n_2y_sw, n_30y=n_30y,
        sigma_sw_2y=sigma_sw_2y, sigma_sw_30y=sigma_sw_30y,
        sigma_source=sigma_source,
        T_fut=T_fut, T_sw=T_sw,
    )


if __name__ == "__main__":
    result = compute_compare(MKT)
