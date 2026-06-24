"""
trade_ticket.py — FOMC Two-Leg Vol Steepener: Price, Size, and MC P&L
=======================================================================
TRADE:  LONG ZT (2Y ATM straddle)  /  SHORT ZB (30Y ATM straddle)
        Same post-FOMC weekly expiry.  Entered after pre-FOMC auctions clear.

PRICER: Bachelier (normal-vol), in-house only.
        Reuses bachelier() and straddle() from fomc_straddle_sim.py.
        DO NOT use any external / company pricer here — that is your cross-check.

ALL LEVELS IN THE CFG DICT BELOW ARE ── PLACEHOLDER ──
Overwrite every field with live OMON / Bloomberg / screen values
before submitting. The derivation — not the numbers — is the deliverable.
"""
from __future__ import annotations
import sys
import textwrap
from math import sqrt, pi
import numpy as np

# ── In-house Bachelier pricer ──────────────────────────────────────────────────
try:
    from fomc_straddle_sim import bachelier, straddle as b_straddle
except ImportError:
    sys.exit("ERROR: fomc_straddle_sim.py not found — place it in the same directory.")


# ══════════════════════════════════════════════════════════════════════════════
# PLACEHOLDER CONFIG — overwrite every value with live screen / OMON data
# ══════════════════════════════════════════════════════════════════════════════
CFG: dict = dict(
    # ── Forward prices (price points; 1 point = 1 % of face) ─────────────────
    F_2Y_PTS           = 102.500,    # ZT front: OMON mid (e.g. 102-16 → 102.500)
    F_30Y_PTS          = 118.000,    # ZB front: OMON mid

    # ── Forward yields — enter directly from OMON/curve, do NOT infer from price
    Y_2Y_PCT           = 4.75,       # 2Y forward yield %
    Y_30Y_PCT          = 4.85,       # 30Y forward yield %

    # ── Strike grid (snap ATM-forward to nearest tick) ─────────────────────────
    STRIKE_GRID_PTS    = 0.250,      # 1/4-pt = typical short-dated OTM grid

    # ── Dates — set all three; T_CALENDAR_DAYS is derived, do NOT set manually ─
    # ENTRY_DATE : when you put the trade on (ISO format YYYY-MM-DD)
    # FOMC_DATE  : FOMC announcement day (Wednesday)
    # EXPIRY_DATE: post-FOMC weekly option expiry (Friday after FOMC week)
    #              Days from FOMC Wed → next-Friday = 9 calendar days.
    #              Days from FOMC Wed → same-week-Friday = 2 calendar days (too short).
    #              Use the LISTED expiry from OMON — do not guess.
    ENTRY_DATE         = "2026-06-24",  # today — actual trade entry date
    FOMC_DATE          = "YYYY-MM-DD",  # PLACEHOLDER — next FOMC Wednesday
    EXPIRY_DATE        = "YYYY-MM-DD",  # PLACEHOLDER — post-FOMC weekly expiry from OMON

    # ── DV01: $ change per 1 bp of yield per $1,000,000 face ──────────────────
    # DV01 ≈ mod_duration × spot_price/100 × 10,000  — GET THESE FROM OMON/BBG
    # ZT (2Y):  mod-dur ≈ 1.75,  price ≈ 102.5  → DV01 ≈ 175  $/bp/$1M
    # ZB (30Y): mod-dur ≈ 17.5,  price ≈ 118.0  → DV01 ≈ 2,065 $/bp/$1M
    DV01_2Y_PER_1M     = 175.0,     # $/bp per $1M face  — OVERRIDE FROM OMON
    DV01_30Y_PER_1M    = 2_065.0,   # $/bp per $1M face  — OVERRIDE FROM OMON

    # ── Event-day yield-move SDs (bp, 1 σ) — GapSpread signal plugs in here ──
    # These are the SIGNAL INPUTS.  Replace with your GapSpread model output.
    # Front-end SD > long-end SD encodes the steepener thesis.
    SIGMA_2Y_YIELD_BPS_DAY  = 18.0, # 2Y event yield SD  (bp, one FOMC event)
    SIGMA_30Y_YIELD_BPS_DAY =  6.0, # 30Y event yield SD (bp)
    RHO_2Y_30Y              =  0.35,# 2Y/30Y yield-move correlation (for MC)

    # ── Normal (Bachelier) vols for PRICING ────────────────────────────────────
    # Derived from event SDs by default (see _sigma_price_from_event below).
    # OVERRIDE with ATM implied vol from OMON for market-price cross-check.
    SIGMA_N_2Y_OVERRIDE    = None,   # pts/yr — set to OMON ATM IV, or leave None
    SIGMA_N_30Y_OVERRIDE   = None,   # pts/yr — set to OMON ATM IV, or leave None

    # ── Notional anchors ───────────────────────────────────────────────────────
    N_30Y_NOTIONAL     = 50_000_000,  # 30Y anchor: desk minimum for an event trade
    # 2Y notional is SOLVED (not chosen) — see vega-neutral and gamma-neutral below

    # ── Contract specs ─────────────────────────────────────────────────────────
    FACE_ZT            = 200_000,    # ZT: $200,000 face per contract
    FACE_ZB            = 100_000,    # ZB: $100,000 face per contract

    # ── Transaction costs ──────────────────────────────────────────────────────
    HALF_SPREAD_PTS    = 0.0040,     # half bid-ask in pts (entry, each leg)
    COMMISSION_PER_LOT = 2.50,       # $ per contract (broker; entry only)

    # ── Monte Carlo ────────────────────────────────────────────────────────────
    N_PATHS            = 15_000,
    RANDOM_SEED        = 42,
)


# ══════════════════════════════════════════════════════════════════════════════
# UNIT UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def snap_to_grid(price: float, grid: float = 0.25) -> float:
    """Round to nearest strike-grid tick."""
    return round(price / grid) * grid


def sigma_price_from_event_sd(sigma_bps_day: float, dv01_per_1m: float) -> float:
    """
    Convert event-day yield SD (bp) → annualised Bachelier normal vol (price pts/yr).

    σ_price_pts_yr = σ_yield_bps_yr × DV01_per_$1M / 10,000
    σ_yield_bps_yr = σ_yield_bps_day × √252   (annualise)

    Derivation: 1 bp yield move → price change = DV01/10,000 pts.
    DV01 is per $1M face; pts are 1% of face.  10,000 = 100 bps per % × 100 pts-per-face.
    """
    sigma_bps_yr = sigma_bps_day * sqrt(252)
    return sigma_bps_yr * dv01_per_1m / 10_000


def straddle_ticket(F: float, K: float, T: float, sigma_n: float,
                    notional: float, dv01_per_1m: float,
                    face_per_contract: float) -> dict:
    """
    Price one straddle leg and return all trader-facing metrics.

    Parameters
    ----------
    F, K       : forward and strike in price points
    T          : time to expiry in years
    sigma_n    : annualised Bachelier normal vol (price pts / yr)
    notional   : face-value notional ($)
    dv01_per_1m: DV01 $ per bp per $1M face
    face_per_contract: face value of one futures contract ($)

    Returns
    -------
    dict of float metrics, all dollar-denominated where appropriate.

    UNIT NOTES
    ----------
    • 1 price point = 1% of face → $ value = pts × 0.01 × notional
    • yield_vega_usd = price_vega_pts × DV01_per_$1M × (notional/$1M)
      Because: ∂σ_price / ∂σ_yield_bps = DV01/10,000 and ∂V$ / ∂σ_price = vega_pts × 0.01 × N
      Combined: ∂V$ / ∂σ_yield_bps = price_vega × DV01/10,000 × 0.01 × N
                                    = price_vega × DV01 × N / 1,000,000
    • dollar_gamma_per_pt_sq = gamma_pts × 0.01 × notional  [$ per pt²]
      P&L from Δpts move ≈ ½ × dollar_gamma × (Δpts)²
    • dollar_theta_per_day = theta_pts_per_yr × 0.01 × notional / 252
      (negative for long straddle)
    """
    g = b_straddle(F, K, T, sigma_n)

    price_pts           = g["price"]
    vega_price_pts      = g["vega"]    # pts / (pt/yr of σ)
    delta_price         = g["delta"]   # ≈ 0 ATM
    gamma_price_pts_sq  = g["gamma"]   # pts / pt²
    theta_price_yr      = g["theta"]   # pts / yr  (negative for long)

    contracts = round(notional / face_per_contract)

    # Dollar conversions — the "× 0.01" converts pts to fraction of face
    pv01_factor        = 0.01 * notional          # $ per 1 price point
    premium_usd        = price_pts * pv01_factor
    yield_vega_usd     = vega_price_pts * dv01_per_1m * (notional / 1e6)
    dollar_gamma       = gamma_price_pts_sq * pv01_factor   # $/pt²
    dollar_theta_day   = theta_price_yr * pv01_factor / 252  # $/day (neg = long)

    # Dollar delta at ATM ≈ 0; report for completeness
    dollar_delta       = delta_price * pv01_factor

    # Yield vol display: sigma_n → back to bp/day for human readability
    sigma_yield_bps_yr  = sigma_n * 10_000 / dv01_per_1m
    sigma_yield_bps_day = sigma_yield_bps_yr / sqrt(252)

    return dict(
        price_pts        = price_pts,
        premium_usd      = premium_usd,
        contracts        = contracts,
        sigma_n          = sigma_n,
        sigma_yield_bps_day = sigma_yield_bps_day,
        vega_price_pts   = vega_price_pts,
        yield_vega_usd   = yield_vega_usd,
        dollar_gamma     = dollar_gamma,
        dollar_theta_day = dollar_theta_day,
        dollar_delta     = dollar_delta,
        gamma_price_pts  = gamma_price_pts_sq,
        T                = T,
    )


def solve_notional_vega_neutral(n_30y: float, vega_pts_30y: float, dv01_30y: float,
                                vega_pts_2y: float,  dv01_2y: float) -> float:
    """
    Yield-vega-neutral 2Y notional.

    Condition: N_2Y × yv_per_$1M(2Y) = N_30Y × yv_per_$1M(30Y)
    where yv_per_$1M = vega_price_pts × DV01_per_$1M

    At same T: vega_price_pts cancels → N_2Y = N_30Y × DV01_30Y / DV01_2Y
    (confirmed by expansion; the ~10× ratio IS CORRECT and not oversized —
    it exactly matches the DV01 ratio so that a 1bp parallel shift produces
    equal $ P&L on both legs.)
    """
    yv_2y  = vega_pts_2y  * dv01_2y
    yv_30y = vega_pts_30y * dv01_30y
    return n_30y * yv_30y / yv_2y


def solve_notional_gamma_neutral(n_30y: float, gamma_pts_30y: float,
                                 gamma_pts_2y: float) -> float:
    """
    Dollar-gamma-neutral 2Y notional.

    Condition: N_2Y × gamma_pts(2Y) × 0.01 = N_30Y × gamma_pts(30Y) × 0.01
    At ATM, gamma_pts ∝ 1/σ_n, so higher σ_n (30Y) → lower gamma per dollar.
    Use when thesis is "realised front-end move exceeds priced level" (event-gamma bet).
    """
    return n_30y * gamma_pts_30y / gamma_pts_2y


# ══════════════════════════════════════════════════════════════════════════════
# EVENT MONTE CARLO  (event-dominant; no diffusive delta-hedge P&L)
# ══════════════════════════════════════════════════════════════════════════════

def run_event_mc(
    # 2Y long leg
    n_2y: float,   k_2y: float, f_2y: float,
    sigma_2y_bps_day: float, dv01_2y: float,
    premium_2y_usd: float,   cost_2y_usd: float,
    # 30Y short leg
    n_30y: float,  k_30y: float, f_30y: float,
    sigma_30y_bps_day: float, dv01_30y: float,
    premium_30y_usd: float,   cost_30y_usd: float,
    # correlation + MC params
    rho: float = 0.35, n_paths: int = 15_000, seed: int = 42,
) -> dict:
    """
    Event-dominant single-jump Monte Carlo.

    Assumptions (print in ticket caveats):
    • One FOMC event jump per path — diffusive/intraday hedge P&L is second-order
      (prior sim showed delta-hedge barely moves Sharpe; event dominates P&L).
    • Yield moves: bivariate normal (σ_2Y, σ_30Y, ρ).  Plug in GapSpread SDs.
    • P&L at expiry: payoff = |F_leg + price_jump − K_leg| × 0.01 × notional.
    • Costs: entry half-spread + commission (no exit cost; options expire at settlement).

    price_jump_pts = yield_move_bp × DV01_per_$1M / 10,000
    (Derivation: 1bp yield Δ → $DV01 per $1M → DV01/10,000 pts per $1M face)
    """
    rng = np.random.default_rng(seed)

    # Correlated draws via Cholesky  [2 × n_paths]
    z  = rng.standard_normal((2, n_paths))
    L  = np.array([[1.0, 0.0],
                   [rho, sqrt(max(0.0, 1.0 - rho**2))]])
    cz = L @ z

    dy_2y  = cz[0] * sigma_2y_bps_day   # bp, event-day yield move for 2Y
    dy_30y = cz[1] * sigma_30y_bps_day  # bp, event-day yield move for 30Y

    # Price jumps: yields rise → prices fall (negative); straddle cares about |jump|
    dp_2y  = dy_2y  * dv01_2y  / 10_000   # pts, may be positive or negative
    dp_30y = dy_30y * dv01_30y / 10_000

    # Payoffs at expiry (straddle = |F_final − K|)
    payoff_2y  = np.abs(f_2y  + dp_2y  - k_2y)  * 0.01 * n_2y
    payoff_30y = np.abs(f_30y + dp_30y - k_30y) * 0.01 * n_30y

    # P&L: long 2Y gains payoff − premium − costs
    #       short 30Y gains premium − payoff − costs
    pnl_2y  = payoff_2y  - premium_2y_usd  - cost_2y_usd
    pnl_30y = premium_30y_usd - payoff_30y - cost_30y_usd
    pnl_net = pnl_2y + pnl_30y

    def _stats(arr: np.ndarray) -> dict:
        p1 = np.percentile(arr, 1)
        tail = arr[arr <= p1]
        return dict(
            mean    = float(np.mean(arr)),
            p_profit= float(np.mean(arr > 0)),
            p5      = float(np.percentile(arr, 5)),
            p95     = float(np.percentile(arr, 95)),
            var99   = float(p1),
            es99    = float(np.mean(tail)) if len(tail) else float(p1),
        )

    return dict(
        net   = _stats(pnl_net),
        leg2y = _stats(pnl_2y),
        leg30y= _stats(pnl_30y),
        # raw arrays for downstream plotting / debugging
        _pnl_net  = pnl_net,
        _pnl_2y   = pnl_2y,
        _pnl_30y  = pnl_30y,
        _dy_2y    = dy_2y,
        _dy_30y   = dy_30y,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PRINT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

_W = 66   # line width

def _bar(char: str = "═") -> str:  return char * _W

def _hdr(title: str, char: str = "═") -> None:
    print(_bar(char))
    print(f"  {title}")
    print(_bar(char))

def _row(label: str, value: str, indent: int = 2) -> None:
    pad = " " * indent
    print(f"{pad}{label:<32}{value}")

def _sep() -> None:
    print("  " + "─" * (_W - 2))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TICKET FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def compute_trade_ticket(cfg: dict = CFG) -> dict:
    """
    Price, size, and MC-simulate the two-leg FOMC vol steepener.
    Returns a dict with all computed values (for programmatic use / notebook display).
    Prints the full trader-facing ticket.
    """
    from datetime import date

    # ── 0. Parse config ─────────────────────────────────────────────────────────
    F2   = cfg["F_2Y_PTS"];       F30  = cfg["F_30Y_PTS"]
    Y2   = cfg["Y_2Y_PCT"];       Y30  = cfg["Y_30Y_PCT"]
    grid = cfg["STRIKE_GRID_PTS"]

    # Derive T from real dates (preferred) or fall back to legacy T_CALENDAR_DAYS
    entry_str  = cfg.get("ENTRY_DATE",  "")
    fomc_str   = cfg.get("FOMC_DATE",   "")
    expiry_str = cfg.get("EXPIRY_DATE", "")

    _date_placeholder = lambda s: not s or s.startswith("YYYY")

    if not _date_placeholder(entry_str) and not _date_placeholder(expiry_str):
        entry_dt  = date.fromisoformat(entry_str)
        expiry_dt = date.fromisoformat(expiry_str)
        if expiry_dt <= entry_dt:
            raise ValueError(f"EXPIRY_DATE ({expiry_str}) must be after ENTRY_DATE ({entry_str})")
        T_cd = (expiry_dt - entry_dt).days
    else:
        # Legacy fallback: use T_CALENDAR_DAYS if dates not set
        T_cd      = cfg.get("T_CALENDAR_DAYS", 9)
        entry_dt  = date.today()
        expiry_dt = None

    fomc_dt = date.fromisoformat(fomc_str) if not _date_placeholder(fomc_str) else None

    T = T_cd / 365.0          # years (calendar-day convention)

    dv01_2y  = cfg["DV01_2Y_PER_1M"]
    dv01_30y = cfg["DV01_30Y_PER_1M"]

    ev_sd_2y  = cfg["SIGMA_2Y_YIELD_BPS_DAY"]
    ev_sd_30y = cfg["SIGMA_30Y_YIELD_BPS_DAY"]
    rho       = cfg["RHO_2Y_30Y"]

    n30 = cfg["N_30Y_NOTIONAL"]
    face_zt = cfg["FACE_ZT"];   face_zb = cfg["FACE_ZB"]
    hs      = cfg["HALF_SPREAD_PTS"]
    comm    = cfg["COMMISSION_PER_LOT"]
    n_paths = cfg["N_PATHS"];   seed = cfg["RANDOM_SEED"]

    # ── 1. Strikes: ATM-forward snapped to grid ─────────────────────────────────
    K2   = snap_to_grid(F2,  grid)
    K30  = snap_to_grid(F30, grid)

    # Approximate yield equivalent of strike (inverse DV01 approximation)
    # ΔP_pts = −DV01 × Δy_bp / 10,000  → Δy_bp = −(K − F) × 10,000 / DV01
    K2_yield  = Y2  - (K2  - F2)  * 10_000 / dv01_2y  / 100   # %
    K30_yield = Y30 - (K30 - F30) * 10_000 / dv01_30y / 100   # %

    # ── 2. Bachelier vols (price pts/yr) ────────────────────────────────────────
    ov2  = cfg.get("SIGMA_N_2Y_OVERRIDE")
    ov30 = cfg.get("SIGMA_N_30Y_OVERRIDE")
    sigma_n_2y  = ov2  if ov2  is not None else sigma_price_from_event_sd(ev_sd_2y,  dv01_2y)
    sigma_n_30y = ov30 if ov30 is not None else sigma_price_from_event_sd(ev_sd_30y, dv01_30y)
    sigma_source = "event-SD derived" if ov2 is None else "OMON override"

    # ── 3. 30Y leg: anchor notional ────────────────────────────────────────────
    tk30 = straddle_ticket(F30, K30, T, sigma_n_30y, n30, dv01_30y, face_zb)

    # ── 4. 2Y leg: vega-neutral sizing ─────────────────────────────────────────
    n2_vega  = solve_notional_vega_neutral(
                   n30, tk30["vega_price_pts"], dv01_30y,
                   b_straddle(F2, K2, T, sigma_n_2y)["vega"], dv01_2y)
    n2_gamma = solve_notional_gamma_neutral(
                   n30, tk30["gamma_price_pts"],
                   b_straddle(F2, K2, T, sigma_n_2y)["gamma"])

    # Lead with vega-neutral
    n2 = n2_vega
    tk2 = straddle_ticket(F2, K2, T, sigma_n_2y, n2, dv01_2y, face_zt)

    # ── 5. Transaction costs ────────────────────────────────────────────────────
    cost_2y  = (hs * 0.01 * n2  + comm * tk2["contracts"])
    cost_30y = (hs * 0.01 * n30 + comm * tk30["contracts"])

    # ── 6. Monte Carlo ──────────────────────────────────────────────────────────
    mc = run_event_mc(
        n2,  K2,  F2,  ev_sd_2y,  dv01_2y,  tk2["premium_usd"],  cost_2y,
        n30, K30, F30, ev_sd_30y, dv01_30y, tk30["premium_usd"], cost_30y,
        rho=rho, n_paths=n_paths, seed=seed,
    )

    # ── 7. Net summary ──────────────────────────────────────────────────────────
    net_premium_usd = tk2["premium_usd"] - tk30["premium_usd"]   # positive = net pay
    net_yv          = tk2["yield_vega_usd"] - tk30["yield_vega_usd"]
    net_gamma       = tk2["dollar_gamma"] - tk30["dollar_gamma"]
    net_theta       = tk2["dollar_theta_day"] - tk30["dollar_theta_day"]
    net_delta       = tk2["dollar_delta"] - tk30["dollar_delta"]  # ≈ 0

    # ── 8. Acceptance checks ────────────────────────────────────────────────────
    tol_vega  = 0.02   # net yield-vega ≤ 2% of leg vega → PASS
    tol_ratio = 0.10   # N2/N30 within 10% of DV01 ratio
    tol_delta = 0.01   # |net delta| ≤ 1% of leg delta → PASS

    yv_check   = abs(net_yv) / max(abs(tk2["yield_vega_usd"]), 1) <= tol_vega
    ratio_check = abs(n2 / n30 - dv01_30y / dv01_2y) / (dv01_30y / dv01_2y) <= tol_ratio
    delta_check = abs(net_delta) <= tol_delta * max(abs(tk2["dollar_delta"]), 1) + 1
    # Thesis check: tested externally (see mc_thesis_check below)

    # ── 9. PRINT THE TICKET ─────────────────────────────────────────────────────
    print()
    _hdr("FOMC TWO-LEG VOL STEEPENER  ·  TRADE TICKET")
    print(f"  Generated : {date.today()}   |   Pricer: Bachelier (normal-vol, in-house)")

    # ── Date timeline ──────────────────────────────────────────────────────────
    fomc_display   = fomc_dt.isoformat()   if fomc_dt   else "YYYY-MM-DD  ← SET FOMC_DATE"
    expiry_display = expiry_dt.isoformat() if expiry_dt else "YYYY-MM-DD  ← SET EXPIRY_DATE"
    days_to_fomc   = (fomc_dt - entry_dt).days if fomc_dt else "?"
    print(f"  Entry date: {entry_str}   │   FOMC date : {fomc_display}  ({days_to_fomc}d away)")
    print(f"  Expiry    : {expiry_display}   │   T to expiry: {T_cd} calendar days  ({T:.5f} yr)")
    print(f"  Vol source: {sigma_source}  |  {'⚠  OVERWRITE LEVELS FROM OMON / SCREEN BEFORE TRADING' if sigma_source != 'OMON override' else 'IV from OMON ✓'}")
    print(_bar())

    # ── LEG 1 — LONG 2Y ────────────────────────────────────────────────────────
    print()
    _hdr("LEG 1 — BUY ZT (2Y) STRADDLE  [LONG GAMMA, LONG VEGA]", "─")
    _row("Side",          "BUY")
    _row("Underlying",    "ZT  (2Y T-Note Futures,  $200k face/contract)")
    _row("Forward (px)",  f"{F2:.3f} pts")
    _row("Strike (px)",   f"{K2:.3f} pts   ≈  {K2_yield:.4f}% yield")
    _row("Notional",      f"${n2:>20,.0f}   [VEGA-NEUTRAL SOLVE ← see below]")
    _row("Contracts",     f"{tk2['contracts']:,}   (@ ${face_zt:,.0f} face/contract)")
    _sep()
    _row("Normal vol σ",  f"{sigma_n_2y:.4f} pts/yr   ({ev_sd_2y:.1f} bp/day yield equiv)")
    _row("DV01 / $1M",    f"${dv01_2y:,.2f} / bp")
    _sep()
    _row("Premium (PAY)", f"${tk2['premium_usd']:>15,.0f}   ({tk2['price_pts']:.4f} pts/unit)")
    _row("Trans. cost",   f"${cost_2y:>15,.0f}   (half-spread + commission, entry only)")
    _row("Yield-vega",    f"${tk2['yield_vega_usd']:>15,.2f} / bp-yield")
    _row("Dollar gamma",  f"${tk2['dollar_gamma']:>15,.2f} / pt²  (½×γ×Δpt² → P&L)")
    # theta_day is negative (long straddle decays); convert to cost so sign is explicit
    _row("Dollar theta",  f"${tk2['dollar_theta_day']:>15,.2f} / day  (cost of carry, negative)")

    # ── LEG 2 — SHORT 30Y ──────────────────────────────────────────────────────
    print()
    _hdr("LEG 2 — SELL ZB (30Y) STRADDLE  [SHORT GAMMA — READ CAVEATS]", "─")
    _row("Side",          "SELL")
    _row("Underlying",    "ZB  (30Y T-Bond Futures, $100k face/contract)")
    _row("Forward (px)",  f"{F30:.3f} pts")
    _row("Strike (px)",   f"{K30:.3f} pts   ≈  {K30_yield:.4f}% yield")
    _row("Notional",      f"${n30:>20,.0f}   [ANCHOR — desk minimum]")
    _row("Contracts",     f"{tk30['contracts']:,}   (@ ${face_zb:,.0f} face/contract)")
    _sep()
    _row("Normal vol σ",  f"{sigma_n_30y:.4f} pts/yr   ({ev_sd_30y:.1f} bp/day yield equiv)")
    _row("DV01 / $1M",    f"${dv01_30y:,.2f} / bp")
    _sep()
    _row("Premium (RCV)", f"${tk30['premium_usd']:>15,.0f}   ({tk30['price_pts']:.4f} pts/unit)")
    _row("Trans. cost",   f"${cost_30y:>15,.0f}   (half-spread + commission, entry only)")
    _row("Yield-vega",    f"${tk30['yield_vega_usd']:>15,.2f} / bp-yield")
    # Short leg: gamma and theta flip sign from the long-perspective computation
    _row("Dollar gamma",  f"${-tk30['dollar_gamma']:>15,.2f} / pt²  (SHORT: negative gamma = you OWE curvature)")
    _row("Dollar theta",  f"${-tk30['dollar_theta_day']:>15,.2f} / day  (theta INCOME to short seller, positive)")

    # ── SIZING RATIONALE ────────────────────────────────────────────────────────
    print()
    _hdr("SIZING RATIONALE", "─")
    dv01_ratio = dv01_30y / dv01_2y
    n_ratio    = n2 / n30
    _row("DV01_30Y / DV01_2Y",  f"{dv01_ratio:.2f}×  (drives vega-neutral ratio)")
    _row("N_2Y / N_30Y",        f"{n_ratio:.2f}×  [should ≈ DV01 ratio → "
                                 f"{'PASS ✓' if ratio_check else 'FAIL ✗'}]")
    _row("Why not oversized?",  f"2Y DV01 is ~{dv01_ratio:.0f}× smaller; more notional needed")
    _row("",                    f"to match the same $ yield-sensitivity as the 30Y leg.")
    _sep()
    _row("Gamma-neutral 2Y N",  f"${n2_gamma:>20,.0f}   (alt sizing)")
    _row("Gamma-neutral ratio", f"{n2_gamma/n30:.2f}×  N_30Y")
    print()
    print("  VEGA-NEUTRAL  (default/lead):  thesis = 'implied vol spread reprices'")
    print(f"    → N_2Y = N_30Y × DV01_30Y/DV01_2Y = ${n2:,.0f}")
    print("  GAMMA-NEUTRAL (alternative):   thesis = 'realised front move > priced'")
    print(f"    → N_2Y = N_30Y × γ_30Y/γ_2Y      = ${n2_gamma:,.0f}")
    print()
    print("  Vega-neutral chosen here because the SIGNAL (GapSpread) measures")
    print("  whether IMPLIED front vol is cheap vs long-end — a spread-repricing bet.")

    # ── NET SUMMARY ─────────────────────────────────────────────────────────────
    print()
    _hdr("NET POSITION SUMMARY", "─")
    _row("Net premium",  f"${net_premium_usd:>+15,.0f}   "
                         f"({'PAY' if net_premium_usd > 0 else 'RECEIVE'})")
    _row("Net yield-vega", f"${net_yv:>+15,.2f} / bp   "
                           f"[VEGA-NEUTRAL CHECK: {'PASS ✓' if yv_check else 'FAIL ✗'}]")
    _row("Net dollar gamma", f"${net_gamma:>+15,.2f} / pt²  (net LONG gamma)")
    _row("Net theta",        f"${net_theta:>+15,.2f} / day  "
                              f"({'INCOME' if net_theta > 0 else 'COST'})")
    _row("Net delta (≈0 ATM)", f"${net_delta:>+15,.2f}  "
                               f"['DELTA-NEUTRAL CHECK: {'PASS ✓' if delta_check else 'note – recheck'}]")

    # ── MONTE CARLO ─────────────────────────────────────────────────────────────
    print()
    _hdr(f"EVENT MONTE CARLO  ({n_paths:,} paths, ρ={rho:.2f})", "─")
    print(f"  FOMC jump: 2Y ~ N(0,{ev_sd_2y}bp/day),  30Y ~ N(0,{ev_sd_30y}bp/day)")
    print(f"  Event-dominant: one jump per path; diffusive/hedge P&L is second-order.")
    if sigma_source == "event-SD derived":
        print(f"  NOTE: premium uses event-SD vol annualised over {T_cd}d window.")
        print(f"  Set SIGMA_N_*_OVERRIDE = OMON ATM IV to get market-implied premium.")
        print(f"  (OMON-priced premium + GapSpread MC SDs → true edge estimate.)")
    print()

    def _mc_row(label: str, d: dict, is_short30=False) -> None:
        warn = "  ← SHORT GAMMA: SIZE TO STOP" if is_short30 else ""
        print(f"  {label:<14}"
              f"  mean ${d['mean']:>+10,.0f}"
              f"  P(+) {d['p_profit']:5.1%}"
              f"  p5 ${d['p5']:>+10,.0f}"
              f"  p95 ${d['p95']:>+10,.0f}"
              f"  ES99 ${d['es99']:>+10,.0f}{warn}")

    print(f"  {'Leg':<14}  {'Mean':>14}  {'P(+)':>7}  {'5th pct':>13}  {'95th pct':>13}  {'ES99':>13}")
    _sep()
    _mc_row("NET",         mc["net"])
    _mc_row("Long 2Y",     mc["leg2y"])
    _mc_row("Short 30Y",   mc["leg30y"], is_short30=True)
    print()
    print(f"  VaR99 (NET): ${mc['net']['var99']:>+,.0f}  (1% of paths lose more than this)")
    print(f"  ES99 30Y short leg: ${mc['leg30y']['es99']:>+,.0f}   "
          f"  MAX LOSS ≠ NET PREMIUM — the short gamma leg has open-ended loss.")
    print(f"  → Size the 30Y notional to a STOP that you can carry + margin.")

    # ── ACCEPTANCE CHECKS ───────────────────────────────────────────────────────
    print()
    _hdr("ACCEPTANCE CHECKS", "─")

    def _chk(label: str, passed: bool, detail: str = "") -> None:
        mark = "PASS ✓" if passed else "FAIL ✗"
        print(f"  [{mark}]  {label}")
        if detail:
            print(f"           {detail}")

    _chk("T1 — Net yield-vega ≈ 0 (vega-neutral sizing)",
         yv_check,
         f"net vega = ${net_yv:+.2f}/bp  (threshold: {tol_vega*100:.0f}% of leg vega)")

    _chk("T2 — N_2Y / N_30Y ≈ DV01_30Y / DV01_2Y  (10× sanity check)",
         ratio_check,
         f"ratio = {n_ratio:.3f}×  vs DV01 ratio = {dv01_ratio:.3f}×")

    _chk("T3 — Both legs delta-neutral at entry (ATM straddle)",
         delta_check,
         f"net delta = ${net_delta:+.2f}  (ATM → Δ_straddle = 0 per Bachelier)")

    # T4 — thesis check: mean P&L rises as σ_2Y rises vs σ_30Y
    sds_test = [sigma_2y * ev_sd_2y for sigma_2y in [0.7, 0.85, 1.0, 1.15, 1.3]]
    means_test = []
    for sd2 in sds_test:
        mc_t = run_event_mc(
            n2, K2, F2, sd2, dv01_2y, tk2["premium_usd"], cost_2y,
            n30, K30, F30, ev_sd_30y, dv01_30y, tk30["premium_usd"], cost_30y,
            rho=rho, n_paths=5_000, seed=seed,
        )
        means_test.append(mc_t["net"]["mean"])
    thesis_ok = all(means_test[i] < means_test[i+1] for i in range(len(means_test)-1))
    _chk("T4 — Mean P&L rises as front-end event SD rises (thesis is directional)",
         thesis_ok,
         "SD_2Y × [0.7,0.85,1,1.15,1.3]:  means = "
         + ", ".join(f"${m:+,.0f}" for m in means_test))

    # ── CROSS-VALIDATION HANDOFF ─────────────────────────────────────────────────
    print()
    _hdr("CROSS-VALIDATION HANDOFF  (feed to company pricer)", "─")
    print("  Enter these exactly into OMON / internal pricer for cross-check.")
    print("  Net premium and ≈0 net vega are what the company system should reproduce.")
    print()
    _row("── LEG 1 (LONG) ──", "")
    _row("Contract",     "ZT  (2Y T-Note Futures)")
    _row("Side",         "BUY ATM straddle (or BUY call + BUY put, same strike)")
    _row("Forward",      f"{F2:.3f} pts  |  {Y2:.4f}% yield")
    _row("Strike",       f"{K2:.3f} pts  |  ≈ {K2_yield:.4f}% yield")
    _row("Expiry",       f"{T_cd} calendar days ({T:.5f} yr)")
    _row("Notional",     f"${n2:,.0f}  ({tk2['contracts']:,} contracts @ ${face_zt:,.0f})")
    _row("Normal vol σ", f"{sigma_n_2y:.4f} pts/yr  →  {ev_sd_2y:.1f} bp/day yield equiv")
    _row("Premium (in)", f"${tk2['premium_usd']:,.0f}")
    print()
    _row("── LEG 2 (SHORT) ──", "")
    _row("Contract",     "ZB  (30Y T-Bond Futures)")
    _row("Side",         "SELL ATM straddle (or SELL call + SELL put, same strike)")
    _row("Forward",      f"{F30:.3f} pts  |  {Y30:.4f}% yield")
    _row("Strike",       f"{K30:.3f} pts  |  ≈ {K30_yield:.4f}% yield")
    _row("Expiry",       f"{T_cd} calendar days ({T:.5f} yr)")
    _row("Notional",     f"${n30:,.0f}  ({tk30['contracts']:,} contracts @ ${face_zb:,.0f})")
    _row("Normal vol σ", f"{sigma_n_30y:.4f} pts/yr  →  {ev_sd_30y:.1f} bp/day yield equiv")
    _row("Premium (out)",f"${tk30['premium_usd']:,.0f}")
    print()
    _row("── NET (company system must reproduce) ──", "")
    _row("Net premium",  f"${net_premium_usd:+,.0f}  ({'PAY' if net_premium_usd > 0 else 'RECEIVE'})")
    _row("Net yield-vega", f"${net_yv:+.2f}/bp  [MUST be ≈ 0 — flag if |net_vega|/$vega_2y > 2%]")
    _row("Net delta",    f"${net_delta:+.2f}  [MUST be ≈ 0  — flag if materially non-zero]")

    # ── CAVEATS ─────────────────────────────────────────────────────────────────
    print()
    _hdr("CAVEATS  (read before submitting)", "─")
    caveats = [
        "ALL MARKET LEVELS ARE PLACEHOLDER.  Overwrite F, DV01, σ, and T from live screen.",
        f"DV01 RATIO drives the 2Y notional ({n_ratio:.1f}× 30Y).  "
         "Get DV01s right — the sizing is exact only when DV01s are exact.",
        "The SHORT 30Y leg is SHORT GAMMA.  'Defined-risk' does NOT apply to it.  "
         "Run the VaR/ES above with actual position size and set a hard stop.",
        "Event-dominant MC ignores diffusive pre-FOMC vol and intraday delta-hedge P&L "
         "by design (prior simulation showed hedge P&L is second-order vs FOMC jump).",
        "GapSpread signal plugs into SIGMA_2Y_YIELD_BPS_DAY and SIGMA_30Y_YIELD_BPS_DAY.  "
         "Both are currently set to placeholder values; replace with model output before trading.",
        "Bachelier (normal-vol) pricer is for internal sizing only.  "
         "Cross-check against company pricer using the HANDOFF block above.",
    ]
    for i, c in enumerate(caveats, 1):
        lines = textwrap.wrap(c, width=_W - 6)
        print(f"  {i}. {lines[0]}")
        for ln in lines[1:]:
            print(f"     {ln}")
        print()

    print(_bar())
    print()

    # ── RETURN DICT for programmatic / notebook use ─────────────────────────────
    return dict(
        # config
        F2=F2, K2=K2, K2_yield=K2_yield, F30=F30, K30=K30, K30_yield=K30_yield,
        T=T, T_days=T_cd,
        sigma_n_2y=sigma_n_2y, sigma_n_30y=sigma_n_30y,
        dv01_2y=dv01_2y, dv01_30y=dv01_30y,
        # sizing
        n2_vega=n2_vega, n2_gamma=n2_gamma, n30=n30,
        n2=n2,   # active sizing (vega-neutral)
        contracts_2y=tk2["contracts"], contracts_30y=tk30["contracts"],
        # greeks (per leg)
        premium_2y=tk2["premium_usd"],  premium_30y=tk30["premium_usd"],
        yield_vega_2y=tk2["yield_vega_usd"], yield_vega_30y=tk30["yield_vega_usd"],
        gamma_2y=tk2["dollar_gamma"],        gamma_30y=tk30["dollar_gamma"],
        theta_2y=tk2["dollar_theta_day"],    theta_30y=tk30["dollar_theta_day"],
        cost_2y=cost_2y, cost_30y=cost_30y,
        # net
        net_premium=net_premium_usd, net_yv=net_yv,
        net_gamma=net_gamma, net_theta=net_theta,
        # MC
        mc=mc,
        # checks
        checks=dict(yv=yv_check, ratio=ratio_check, delta=delta_check, thesis=thesis_ok),
    )


# ══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    result = compute_trade_ticket(CFG)
    all_pass = all(result["checks"].values())
    sys.exit(0 if all_pass else 1)
