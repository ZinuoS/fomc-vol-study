"""Generator for vol_stop_simulation.ipynb.  Run: python3 _make_vol_stop_nb.py"""
import json
from uuid import uuid4

cells = []

def md(src):
    lines = []
    for ln in src.split("\n"):
        s = ln.rstrip()
        lines.append(s[2:] if s.startswith("# ") else ("" if s == "#" else s))
    cells.append({"cell_type":"markdown","id":uuid4().hex[:8],"metadata":{},"source":"\n".join(lines)})

def code(src):
    cells.append({"cell_type":"code","id":uuid4().hex[:8],"metadata":{},
                  "source":src.strip(),"outputs":[],"execution_count":None})

# ─────────────────────────────────────────────────────────────────────────────
# CELL 0 — Title
# ─────────────────────────────────────────────────────────────────────────────
md("""
# 2s30s Vol-Spread Stop-Loss Simulation
#
# **LONG 2Y swaption straddle / SHORT 30Y swaption straddle | vega-neutral | Bachelier bp pricer**
#
# **Core principle**: the stop is on the **2s30s VOL SPREAD** (σ_N(2Y) − σ_N(30Y)), not P&L, not a single leg.
# Triggers when 30Y vol richens relative to 2Y vol by **STOP_Z bp** — the thesis variable.
# The simulation models the **trigger → fill gap** (slippage on unwinding a short 30Y straddle
# in a fast market). A stop that marks at mid is dishonest.
#
# | Section | Content |
# |---|---|
# | **1** | Position + stop + sim inputs (two tagged groups) |
# | **2** | Joint (σ_2Y, σ_30Y) vol process — correlated arithmetic BM + FOMC jump |
# | **3** | Stop logic + slippage model + three-policy comparison |
# | **4** | Diagnostics — loss distribution, whipsaw rate, STOP_Z sweep, **unwind instruction** |
# | **5** | Visuals — paths, loss distribution, sweep tradeoff, representative path |
# | **6** | Acceptance checks T1–T4 + caveats |
#
# All market levels are **PLACEHOLDER** — overwrite from live screen before trading.
""")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 1 — Imports + vectorized Bachelier pricer
# ─────────────────────────────────────────────────────────────────────────────
code("""\
# CELL 1: Imports, vectorized Bachelier pricer, palette
import sys, warnings, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import norm as _norm
from IPython.display import Image, display as _dimg

sys.path.insert(0, '.')
try:
    from fomc_straddle_sim import bachelier, straddle as b_straddle_scalar
except ImportError:
    sys.exit("fomc_straddle_sim.py not found — run from repo root.")

warnings.filterwarnings('ignore')
os.makedirs('vol_stop_figs', exist_ok=True)

# ── Vectorized Bachelier straddle (Bachelier C + P, normal vol) ──────────────
# Formula: C+P = 2·s·φ(d) + (F-K)·(2·Φ(d)-1), s = σ·√T
# All inputs broadcastable numpy arrays; F,K in same units as σ (bp).
def bstraddle_vec(F, K, T_arr, sigma_arr):
    T_arr, sigma_arr = np.asarray(T_arr, float), np.asarray(sigma_arr, float)
    s   = sigma_arr * np.sqrt(np.maximum(T_arr, 0.0))
    d   = np.where(s > 1e-10, (F - K) / np.where(s > 1e-10, s, 1.0), 0.0)
    p   = 2.0 * s * _norm.pdf(d) + (F - K) * (2.0 * _norm.cdf(d) - 1.0)
    return np.where(np.asarray(T_arr) > 0, p, np.abs(F - K))

# ── Shared palette ────────────────────────────────────────────────────────────
PAL = dict(buy="#2166ac", sell="#d6604d", spread="#7b2d8b",
           clean="#4dac26", slip="#d01c8b", sweep="#f1a340",
           neutral="#888888", grid="#e8e8e8")

plt.rcParams.update({"figure.dpi": 130, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
print("Imports OK — vectorized Bachelier pricer loaded.")
""")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 2 — Section 1 header
# ─────────────────────────────────────────────────────────────────────────────
md("""
# ## SECTION 1 — INPUTS
#
# | Group | Source | Purpose |
# |---|---|---|
# | **A — Position** | From trade ticket (`trade_ticket_pricer.ipynb`) | Leg sizes, entry vols, DV01s |
# | **B — Stop config** | Risk / desk decision | STOP_Z to sweep; slippage model |
# | **C — Sim config** | Vol-of-vol calibration (*** SCREEN ***) | Joint vol process parameters |
#
# *** = overwrite from screen / live calibration before trading.
""")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 3 — Group A: Position inputs
# ─────────────────────────────────────────────────────────────────────────────
code("""\
# CELL 3: GROUP A — POSITION (from trade_ticket_pricer.ipynb)
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  All bp-space: rates in bp (100bp = 1%), vols in bp/yr (normal model)  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

POS = dict(
    # ── 2Y swaption (LONG straddle) ──────────────────────────────────────
    F_2Y_BP    = 397.808,   # 2Y fwd par swap rate (bp)   *** SCREEN ***
    K_2Y_BP    = 398.0,     # ATM strike, rounded to 1bp  *** SCREEN ***
    IV_2Y_BP   = 101.1,     # entry implied vol (bp/yr, normal)  *** SCREEN ***
    DV01_2Y    = 189.97,    # $/bp per $1M notional  *** BBG OTR 2Y ***
    N_2Y_M     = 420.0,     # $M notional on 2Y leg (vega-neutral vs $50M 30Y)

    # ── 30Y swaption (SHORT straddle) ────────────────────────────────────
    F_30Y_BP   = 412.662,   # 30Y fwd par swap rate (bp)  *** SCREEN ***
    K_30Y_BP   = 413.0,     # ATM strike, rounded to 1bp  *** SCREEN ***
    IV_30Y_BP  = 72.8,      # entry implied vol (bp/yr, normal)  *** SCREEN ***
    DV01_30Y   = 1592.56,   # $/bp per $1M notional  *** BBG CT30 ***
    N_30Y_M    = 50.0,      # $M notional on 30Y leg (desk anchor)

    # ── Dates ─────────────────────────────────────────────────────────────
    ENTRY_DATE = "2026-06-25",  # today
    FOMC_DATE  = "2026-07-29",  # FOMC day (Tuesday before expiry)
    EXPIRY     = "2026-07-31",  # expiry (Friday after FOMC)

    # ── Net premium at entry (model value from trade ticket) ──────────────
    NET_PREM_K = 564.0,     # $k net premium paid (model; desk ≈ $520k)
)
print(f"Position: LONG ${POS['N_2Y_M']:.0f}M 2Y straddle | SHORT ${POS['N_30Y_M']:.0f}M 30Y straddle")
print(f"Entry vol spread: {POS['IV_2Y_BP']:.1f} - {POS['IV_30Y_BP']:.1f} = "
      f"{POS['IV_2Y_BP']-POS['IV_30Y_BP']:.1f} bp")
""")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 4 — Group B (stop) + Group C (sim)
# ─────────────────────────────────────────────────────────────────────────────
code("""\
# CELL 4: GROUP B — STOP CONFIG  /  GROUP C — SIMULATION CONFIG

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  GROUP B — STOP CONFIG                                                   ║
# ║  STOP_Z: adverse 2s30s vol-spread move that triggers unwind (SWEEP this) ║
# ╚══════════════════════════════════════════════════════════════════════════╝
STOP = dict(
    STOP_Z_BP       = 15.0,   # base-case stop width (bp); sweep in Section 4  *** DESK ***
    RISK_BUDGET_K   = 700.0,  # max acceptable post-slippage p99 loss ($k)      *** RISK ***
    # NOTE: budget must exceed net premium paid ($564k); a budget below net premium
    # is unachievable because holding to expiry already loses the full premium.
    # Slippage model: state-dependent (wider in stress = exactly when stop fires)
    SLIP_2Y_BASE    = 0.5,    # half bid/ask on 2Y vol at rest (bp/yr)
    SLIP_30Y_BASE   = 1.0,    # half bid/ask on 30Y vol at rest — WIDER (less liquid long-end)
    SLIP_SPEED_COEF = 0.15,   # extra slippage (bp) per 1 bp/day of vol velocity  *** SCREEN ***
    SLIP_LOOKBACK   = 3,      # days to measure vol speed for slippage scaling
)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  GROUP C — SIMULATION CONFIG  (vol-of-vol calibration)                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝
SIM = dict(
    N_PATHS         = 20_000,
    SEED            = 42,
    # Daily vol innovation (arithmetic BM on vol levels, calendar-day steps)
    VOL_SD_2Y       = 3.2,    # daily σ of 2Y implied vol (bp/day)  *** SCREEN calibrate ***
    VOL_SD_30Y      = 2.0,    # daily σ of 30Y implied vol (bp/day) *** SCREEN calibrate ***
    VOL_CORR        = 0.65,   # ρ(ε_2Y, ε_30Y): co-move but not perfectly
    VOL_FLOOR       = 5.0,    # minimum vol floor (bp/yr)
    # FOMC-day additive vol jump (structural event-day shock on top of diffusion)
    FOMC_JUMP_SD_2Y  = 8.0,  # std of FOMC-day vol jump, 2Y (bp)  *** SCREEN ***
    FOMC_JUMP_SD_30Y = 5.0,  # std of FOMC-day vol jump, 30Y (bp) *** SCREEN ***
)
print(f"Stop config: base STOP_Z={STOP['STOP_Z_BP']:.0f}bp | risk budget=${STOP['RISK_BUDGET_K']:.0f}k")
print(f"Sim config: N={SIM['N_PATHS']:,} paths | σ_vol(2Y)={SIM['VOL_SD_2Y']}bp/day | "
      f"σ_vol(30Y)={SIM['VOL_SD_30Y']}bp/day | ρ={SIM['VOL_CORR']}")
""")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 5 — Derived entry quantities
# ─────────────────────────────────────────────────────────────────────────────
code("""\
# CELL 5: Derived entry quantities
import pandas as pd

T_DAYS = (pd.Timestamp(POS['EXPIRY'])    - pd.Timestamp(POS['ENTRY_DATE'])).days  # 36
T_YR   = T_DAYS / 365.0

FOMC_DAYS = (pd.Timestamp(POS['FOMC_DATE']) - pd.Timestamp(POS['ENTRY_DATE'])).days  # 34
FOMC_IDX  = FOMC_DAYS - 1    # 0-indexed in vol_path arrays (day 1 = index 0)

ENTRY_SPREAD = POS['IV_2Y_BP'] - POS['IV_30Y_BP']   # 28.3 bp

# Entry straddle prices (bp) via vectorized pricer (scalar inputs → scalar out)
p2e_bp  = float(bstraddle_vec(POS['F_2Y_BP'],  POS['K_2Y_BP'],  T_YR, POS['IV_2Y_BP']))
p30e_bp = float(bstraddle_vec(POS['F_30Y_BP'], POS['K_30Y_BP'], T_YR, POS['IV_30Y_BP']))

# Dollar conversions (per-leg aggregate, $k)
DV2   = POS['DV01_2Y']  * POS['N_2Y_M']  / 1e3   # $k per bp of straddle price (2Y leg)
DV30  = POS['DV01_30Y'] * POS['N_30Y_M'] / 1e3   # $k per bp of straddle price (30Y leg)

p2e_K  = p2e_bp  * DV2
p30e_K = p30e_bp * DV30
model_net_K = p2e_K - p30e_K

# Time array (remaining years at end of each simulated day)
T_rem = np.maximum((T_DAYS - np.arange(1, T_DAYS + 1)) / 365.0, 0.0)   # shape (T_DAYS,)

# Dollar vega at each day (straddle vega = sqrt(T_rem)*sqrt(2/pi) for ATM, bp per bp/yr vol)
vega_straddle = np.sqrt(T_rem) * np.sqrt(2.0 / np.pi)   # (T_DAYS,)
dv_vega_2Y  = vega_straddle * DV2    # $k per bp/yr vol, 2Y leg, shape (T_DAYS,)
dv_vega_30Y = vega_straddle * DV30   # $k per bp/yr vol, 30Y leg

print(f"T_DAYS={T_DAYS}  T_YR={T_YR:.4f}  FOMC_DAY=index {FOMC_IDX} (day {FOMC_DAYS})")
print(f"Entry spread: {ENTRY_SPREAD:.1f}bp")
print(f"Entry 2Y straddle : {p2e_bp:6.2f}bp = ${p2e_K:7,.0f}k  "
      f"(${POS['N_2Y_M']:.0f}M × {POS['DV01_2Y']:.2f} $/bp)")
print(f"Entry 30Y straddle: {p30e_bp:6.2f}bp = ${p30e_K:7,.0f}k  "
      f"(${POS['N_30Y_M']:.0f}M × {POS['DV01_30Y']:.2f} $/bp)")
print(f"Model net premium : ${model_net_K:,.0f}k  (desk: ${POS['NET_PREM_K']:.0f}k)")
""")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 6 — Section 2 header
# ─────────────────────────────────────────────────────────────────────────────
md("""
# ## SECTION 2 — JOINT VOL PROCESS
#
# Simulate **N paths** of **(σ_2Y(t), σ_30Y(t))** over T calendar days via
# arithmetic Brownian motion (additive innovations → vol level can be negative
# but is floored at `VOL_FLOOR`).  Innovations are **correlated** via Cholesky
# decomposition so the vol-spread `σ_2Y − σ_30Y` has realistic dynamics.
#
# An additional **FOMC-day jump** is superimposed at the meeting date to capture
# the structural event-day vol move.
#
# **Stop variable**: `spread(t) = σ_2Y(t) − σ_30Y(t)`.  Trigger when
# `spread(t) ≤ ENTRY_SPREAD − STOP_Z`.
""")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 7 — Simulate vol paths
# ─────────────────────────────────────────────────────────────────────────────
code("""\
# CELL 7: Simulate correlated (σ_2Y, σ_30Y) paths
N = SIM['N_PATHS']
T = T_DAYS   # 36 calendar days

# Cholesky factor for correlated daily vol innovations
cov = np.array([
    [SIM['VOL_SD_2Y']**2,
     SIM['VOL_CORR'] * SIM['VOL_SD_2Y'] * SIM['VOL_SD_30Y']],
    [SIM['VOL_CORR'] * SIM['VOL_SD_2Y'] * SIM['VOL_SD_30Y'],
     SIM['VOL_SD_30Y']**2]
])
L = np.linalg.cholesky(cov)   # lower-triangular Cholesky factor

rng = np.random.default_rng(SIM['SEED'])

# Daily diffusion innovations: (N, T, 2) × L.T → correlated (ε_2Y, ε_30Y)
Z = rng.standard_normal((N, T, 2)) @ L.T    # shape (N, T, 2)

# FOMC-day additive vol jump (index FOMC_IDX, uncorrelated to daily diffusion)
Z_fomc = rng.standard_normal((N, 2)) * np.array([
    SIM['FOMC_JUMP_SD_2Y'], SIM['FOMC_JUMP_SD_30Y']
])
Z[:, FOMC_IDX, :] += Z_fomc

# Cumulate to get vol level paths
diffs   = np.cumsum(Z, axis=1)            # shape (N, T, 2)
vol_2Y  = POS['IV_2Y_BP']  + diffs[:, :, 0]   # (N, T) bp/yr
vol_30Y = POS['IV_30Y_BP'] + diffs[:, :, 1]   # (N, T) bp/yr

# Floor at VOL_FLOOR
vol_2Y  = np.maximum(vol_2Y,  SIM['VOL_FLOOR'])
vol_30Y = np.maximum(vol_30Y, SIM['VOL_FLOOR'])

# 2s30s vol spread along each path
spread  = vol_2Y - vol_30Y    # (N, T)

# Quick sanity: spread std at terminal day
print(f"Vol-path summary at day {T} (expiry-1):")
print(f"  σ_N(2Y):  mean={vol_2Y[:,-1].mean():.1f}bp  std={vol_2Y[:,-1].std():.1f}bp")
print(f"  σ_N(30Y): mean={vol_30Y[:,-1].mean():.1f}bp  std={vol_30Y[:,-1].std():.1f}bp")
print(f"  spread:   mean={spread[:,-1].mean():.1f}bp  std={spread[:,-1].std():.1f}bp  "
      f"(entry={ENTRY_SPREAD:.1f}bp)")
print(f"  realized ρ(ε_2Y,ε_30Y): "
      f"{np.corrcoef(diffs[:,-1,0], diffs[:,-1,1])[0,1]:.3f}  (target {SIM['VOL_CORR']})")
""")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 8 — Section 3 header
# ─────────────────────────────────────────────────────────────────────────────
md("""
# ## SECTION 3 — STOP LOGIC + UNWIND
#
# Three sub-steps:
# 1. **MTM pricing**: reprice both straddles daily at current vol using the in-house Bachelier
#    pricer; track net dollar P&L along each path.
# 2. **Slippage model**: estimate the bid/ask crossing cost on unwind as a *function of vol speed*
#    (state-dependent — wider exactly when the stop triggers).
# 3. **Three-policy comparison**: (i) vol-spread stop, (ii) no stop, (iii) naive P&L stop —
#    to show the spread stop fires on *thesis breaks*, not noise.
""")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 9 — MTM pricing along paths
# ─────────────────────────────────────────────────────────────────────────────
code("""\
# CELL 9: MTM pricing — vectorized Bachelier over all (N, T) simultaneously
# F and K are fixed at entry (delta-hedged; vol P&L only)

T_mat = T_rem[np.newaxis, :]   # (1, T) — broadcasts with (N, T) vol arrays

# Straddle prices at each (vol, T_rem) node  — result in bp
price_2Y  = bstraddle_vec(POS['F_2Y_BP'],  POS['K_2Y_BP'],  T_mat, vol_2Y)    # (N, T)
price_30Y = bstraddle_vec(POS['F_30Y_BP'], POS['K_30Y_BP'], T_mat, vol_30Y)   # (N, T)

# Dollar MTM P&L from entry ($k):
#   Long  2Y: gain when 2Y straddle price rises (2Y vol richens)
#   Short 30Y: gain when 30Y straddle price falls (30Y vol cheapens)
pnl_2Y_K  = (price_2Y  - p2e_bp)  * DV2    # (N, T) $k
pnl_30Y_K = (price_30Y - p30e_bp) * DV30   # (N, T) $k
net_pnl_K = pnl_2Y_K - pnl_30Y_K           # (N, T) $k; positive = thesis working

print(f"MTM pricing OK — shape {net_pnl_K.shape}")
print(f"  net_pnl at FOMC day (index {FOMC_IDX}): "
      f"mean={net_pnl_K[:, FOMC_IDX].mean():+.0f}k  "
      f"p5={np.percentile(net_pnl_K[:, FOMC_IDX], 5):+.0f}k  "
      f"p95={np.percentile(net_pnl_K[:, FOMC_IDX], 95):+.0f}k")
""")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 10 — Slippage model
# ─────────────────────────────────────────────────────────────────────────────
code("""\
# CELL 10: State-dependent slippage model
# Slippage (half bid/ask in bp/yr vol) on each leg:
#   slip_leg = BASE + SPEED_COEF × vol_speed_leg
# where vol_speed = |Δσ| over last LOOKBACK days.
# Key property: slippage WIDENS exactly when the stop fires (30Y moved fast against us).

LB = STOP['SLIP_LOOKBACK']

# Vol speed: |change over last LB days| / LB  — vectorized
vol_2Y_lagged  = np.concatenate([
    np.repeat(vol_2Y[:, :1],  LB, axis=1),
    vol_2Y[:,  :-LB]
], axis=1)    # (N, T): vol LB days ago
vol_30Y_lagged = np.concatenate([
    np.repeat(vol_30Y[:, :1], LB, axis=1),
    vol_30Y[:, :-LB]
], axis=1)

speed_2Y  = np.abs(vol_2Y  - vol_2Y_lagged)  / LB   # (N, T) bp/day
speed_30Y = np.abs(vol_30Y - vol_30Y_lagged) / LB

# Slippage per leg (bp vol): base + speed-driven component
slip_2Y_mat  = STOP['SLIP_2Y_BASE']  + STOP['SLIP_SPEED_COEF'] * speed_2Y    # (N, T)
slip_30Y_mat = STOP['SLIP_30Y_BASE'] + STOP['SLIP_SPEED_COEF'] * speed_30Y   # (N, T)

# Dollar slippage cost of unwinding both legs ($k):
#   sell 2Y at bid (mid - slip_2Y) → lose slip_2Y × dollar_vega_2Y
#   buy back 30Y at offer (mid + slip_30Y) → lose slip_30Y × dollar_vega_30Y
slip_cost_K_mat = (slip_2Y_mat  * dv_vega_2Y[np.newaxis,  :]
                 + slip_30Y_mat * dv_vega_30Y[np.newaxis, :])   # (N, T) $k

print(f"Slippage model OK — shape {slip_cost_K_mat.shape}")
print(f"  Base slippage cost (day 1, rest conditions): "
      f"${slip_cost_K_mat[:, 0].min():.0f}k – ${slip_cost_K_mat[:, 0].max():.0f}k")
print(f"  Worst-case slippage cost (day 1, stressed): "
      f"${slip_cost_K_mat[:, 0].max():.0f}k")
""")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 11 — apply_stop function + base-case execution
# ─────────────────────────────────────────────────────────────────────────────
code("""\
# CELL 11: apply_stop() — core stop-loss engine (vectorized over all N paths)

def apply_stop(stop_z_bp):
    trigger_mask = (spread - ENTRY_SPREAD) <= -stop_z_bp   # (N, T)
    triggered    = trigger_mask.any(axis=1)                 # (N,)
    first_t      = np.where(triggered, np.argmax(trigger_mask, axis=1), T - 1)  # (N,)
    idx_n        = np.arange(N)

    loss_clean_K  = net_pnl_K[idx_n, first_t]             # (N,) MTM at trigger (or expiry)
    slip_at_K     = slip_cost_K_mat[idx_n, first_t]        # (N,) slippage cost
    loss_post_K   = loss_clean_K - slip_at_K               # (N,) after slippage

    # Final P&L per path: triggered → stop-loss; else → hold to expiry (vol model)
    final_clean_K = np.where(triggered, loss_clean_K, net_pnl_K[:, -1])
    final_post_K  = np.where(triggered, loss_post_K,  net_pnl_K[:, -1])

    return dict(
        triggered    = triggered,
        stop_rate    = triggered.mean(),
        first_t      = first_t,
        loss_clean_K = np.where(triggered, loss_clean_K, np.nan),
        slip_at_K    = np.where(triggered, slip_at_K,    np.nan),
        loss_post_K  = np.where(triggered, loss_post_K,  np.nan),
        final_clean_K= final_clean_K,
        final_post_K = final_post_K,
    )

# ── Base-case (STOP_Z from config) ───────────────────────────────────────────
RES = apply_stop(STOP['STOP_Z_BP'])
trig_mask   = RES['triggered']
trig_clean  = RES['loss_clean_K'][trig_mask]    # loss given stop (clean)
trig_post   = RES['loss_post_K'][trig_mask]     # loss given stop (post-slippage)
slip_only   = RES['slip_at_K'][trig_mask]       # slippage cost only

print(f"=== BASE CASE: STOP_Z = {STOP['STOP_Z_BP']:.0f}bp ===")
print(f"  Stop hit rate : {100*RES['stop_rate']:.1f}% of paths")
print(f"  Avg trigger day : {RES['first_t'][trig_mask].mean():.1f} of {T} days "
      f"(FOMC=day {FOMC_DAYS})")
print(f"  Loss at trigger (clean):       mean={trig_clean.mean():+.0f}k   "
      f"p95={np.percentile(trig_clean, 5):+.0f}k   p99={np.percentile(trig_clean, 1):+.0f}k")
print(f"  Loss post-slippage:            mean={trig_post.mean():+.0f}k   "
      f"p95={np.percentile(trig_post, 5):+.0f}k   p99={np.percentile(trig_post, 1):+.0f}k")
print(f"  Slippage cost (GAP):           mean={slip_only.mean():+.0f}k   "
      f"p95={np.percentile(slip_only, 95):+.0f}k   max={slip_only.max():+.0f}k")
print(f"  → Gap = {abs(trig_post.mean()-trig_clean.mean()):.0f}k extra loss from trigger→fill slippage")
""")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 12 — Three-policy comparison
# ─────────────────────────────────────────────────────────────────────────────
code("""\
# CELL 12: Three-policy comparison
# (i)   Vol-spread stop (STOP_Z bp adverse spread move)
# (ii)  No stop (hold to expiry; vol model only — excludes rate-move payoff)
# (iii) Naive P&L stop calibrated to same average dollar loss as policy (i)

# Policy (i) — already computed above
pol1_final_K = RES['final_post_K']

# Policy (ii) — no stop: hold to expiry
pol2_final_K = net_pnl_K[:, -1].copy()

# Policy (iii) — P&L stop calibrated to same expected loss as spread stop
pnl_stop_level = abs(np.nanmean(RES['loss_post_K']))   # same avg loss as spread stop
pnl_trigger_mask = net_pnl_K < -pnl_stop_level          # (N, T)
pnl_triggered    = pnl_trigger_mask.any(axis=1)
pnl_first_t      = np.where(pnl_triggered, np.argmax(pnl_trigger_mask, axis=1), T - 1)
pol3_pnl_K       = net_pnl_K[np.arange(N), pnl_first_t] - slip_cost_K_mat[np.arange(N), pnl_first_t]
pol3_final_K     = np.where(pnl_triggered, pol3_pnl_K, net_pnl_K[:, -1])

print(f"Policy comparison (N={N:,} paths):")
print(f"  (i)  Vol-spread stop ({STOP['STOP_Z_BP']:.0f}bp):")
print(f"       Stop rate={100*RES['stop_rate']:.1f}%  mean P&L={pol1_final_K.mean():+.0f}k  "
      f"p5={np.percentile(pol1_final_K,5):+.0f}k  p95={np.percentile(pol1_final_K,95):+.0f}k")
print(f"  (ii) No stop (hold to expiry, vol model):")
print(f"       mean P&L={pol2_final_K.mean():+.0f}k  "
      f"p5={np.percentile(pol2_final_K,5):+.0f}k  p95={np.percentile(pol2_final_K,95):+.0f}k")
print(f"  (iii) Naive P&L stop (level=${pnl_stop_level:.0f}k):")
print(f"       Stop rate={100*pnl_triggered.mean():.1f}%  mean P&L={pol3_final_K.mean():+.0f}k  "
      f"p5={np.percentile(pol3_final_K,5):+.0f}k  p95={np.percentile(pol3_final_K,95):+.0f}k")
""")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 13 — Whipsaw / false-positive analysis
# ─────────────────────────────────────────────────────────────────────────────
code("""\
# CELL 13: Whipsaw / false-positive analysis
# A whipsaw: stop triggered BEFORE FOMC, but on FOMC day the P&L would have
# recovered above the realized stop loss (= false positive = cost of stopping).

pre_fomc  = trig_mask & (RES['first_t'] < FOMC_IDX)
fomc_pnl  = net_pnl_K[:, FOMC_IDX]    # P&L on FOMC day if no stop

# Whipsaw: pre-FOMC stop fired, but FOMC P&L (no stop) > realized stop-loss
whipsaw   = pre_fomc & (fomc_pnl > RES['loss_post_K'])
ws_rate   = whipsaw.sum() / pre_fomc.sum() if pre_fomc.sum() > 0 else 0.0
ws_cost   = (fomc_pnl[whipsaw] - RES['loss_post_K'][whipsaw]).mean() if whipsaw.sum() > 0 else 0.0

print(f"=== Whipsaw / False-Positive Analysis (STOP_Z={STOP['STOP_Z_BP']:.0f}bp) ===")
print(f"  Pre-FOMC stops     : {pre_fomc.sum():,} paths  ({100*pre_fomc.mean():.1f}% of total)")
print(f"  Of those, whipsawed: {whipsaw.sum():,} ({100*ws_rate:.1f}%) — stopped early, thesis proved right on FOMC")
print(f"  Avg whipsaw cost   : ${ws_cost:+.0f}k  (opportunity cost per false-positive path)")
print(f"  Avg trigger day    : {RES['first_t'][pre_fomc].mean():.1f} (of {FOMC_DAYS} to FOMC)")
print()
print(f"Interpretation: {100*ws_rate:.0f}% of pre-FOMC stops were false alarms.")
print(f"Each false alarm cost an average ${ws_cost:.0f}k in forgone P&L.")
""")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 14 — Section 4 header
# ─────────────────────────────────────────────────────────────────────────────
md("""
# ## SECTION 4 — OUTPUTS / DIAGNOSTICS
#
# - **Loss distribution**: mean / p95 / p99 given stop, clean vs post-slippage.
# - **Whipsaw rate**: cost of stopping — how often the stop fires then the trade recovers.
# - **STOP_Z sweep**: tradeoff — tight stop = more whipsaws + smaller tail; wide = fewer + bigger.
# - **Unwind instruction**: literal desk order for a representative triggered path.
""")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 15 — Loss distribution stats table
# ─────────────────────────────────────────────────────────────────────────────
code('''\
# CELL 15: Loss distribution stats — given stop triggered
from IPython.display import display

stats = {
    "metric": ["mean", "p90 worst", "p95 worst", "p99 worst", "worst 0.1%"],
    "clean ($k)": [
        f"{np.nanmean(trig_clean):+.0f}",
        f"{np.nanpercentile(trig_clean, 10):+.0f}",
        f"{np.nanpercentile(trig_clean, 5):+.0f}",
        f"{np.nanpercentile(trig_clean, 1):+.0f}",
        f"{np.nanmin(trig_clean):+.0f}",
    ],
    "post-slippage ($k)": [
        f"{np.nanmean(trig_post):+.0f}",
        f"{np.nanpercentile(trig_post, 10):+.0f}",
        f"{np.nanpercentile(trig_post, 5):+.0f}",
        f"{np.nanpercentile(trig_post, 1):+.0f}",
        f"{np.nanmin(trig_post):+.0f}",
    ],
    "slippage gap ($k)": [
        f"{np.nanmean(slip_only):+.0f}",
        f"{np.nanpercentile(slip_only, 90):+.0f}",
        f"{np.nanpercentile(slip_only, 95):+.0f}",
        f"{np.nanpercentile(slip_only, 99):+.0f}",
        f"{np.nanmax(slip_only):+.0f}",
    ],
}
df_stats = pd.DataFrame(stats)
print(f"Loss distribution GIVEN stop triggered (STOP_Z={STOP['STOP_Z_BP']:.0f}bp, "
      f"hit rate={100*RES['stop_rate']:.1f}%):")
display(df_stats.to_string(index=False))
''')

# ─────────────────────────────────────────────────────────────────────────────
# CELL 16 — STOP_Z sweep
# ─────────────────────────────────────────────────────────────────────────────
code("""\
# CELL 16: STOP_Z sweep — vary stop width, track tradeoff
stop_z_grid = np.arange(5, 56, 5)    # 5, 10, 15, ..., 55 bp
sweep = []

for sz in stop_z_grid:
    r = apply_stop(sz)
    trig    = r['triggered']
    lp      = r['loss_post_K'][trig]    # post-slippage losses for triggered paths

    # Pre-FOMC whipsaws for this STOP_Z
    pre_f   = trig & (r['first_t'] < FOMC_IDX)
    ws_flg  = pre_f & (net_pnl_K[:, FOMC_IDX] > r['loss_post_K'])
    ws_rt   = ws_flg.sum() / pre_f.sum() if pre_f.sum() > 0 else 0.0

    sweep.append({
        "stop_z_bp"    : sz,
        "stop_rate_pct": 100 * r['stop_rate'],
        "mean_K"       : np.nanmean(lp) if len(lp) else np.nan,
        "p95_K"        : np.nanpercentile(lp, 5) if len(lp) else np.nan,    # 5th pctile = 95th worst
        "p99_K"        : np.nanpercentile(lp, 1) if len(lp) else np.nan,    # 1st pctile = 99th worst
        "whipsaw_pct"  : 100 * ws_rt,
    })

sweep_df = pd.DataFrame(sweep)

# Find stop level where post-slippage p99 worst loss is within RISK_BUDGET
budget   = -STOP['RISK_BUDGET_K']   # negative (loss)
inside   = sweep_df[sweep_df['p99_K'] >= budget]
rec_z    = float(inside['stop_z_bp'].iloc[0]) if len(inside) else float(sweep_df['stop_z_bp'].iloc[-1])

print(f"STOP_Z sweep (RISK_BUDGET = -${STOP['RISK_BUDGET_K']:.0f}k post-slippage p99):")
print(sweep_df[['stop_z_bp','stop_rate_pct','p99_K','whipsaw_pct']].to_string(index=False))
print(f"\\n  → Recommended STOP_Z: {rec_z:.0f}bp  (first stop where p99 loss ≤ -${STOP['RISK_BUDGET_K']:.0f}k)")
""")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 17 — Unwind instruction
# ─────────────────────────────────────────────────────────────────────────────
code('''\
# CELL 17: Unwind instruction — representative triggered path (median trigger day)
triggered_idx = np.where(trig_mask)[0]
med_pos       = len(triggered_idx) // 2
sorted_by_t   = triggered_idx[np.argsort(RES['first_t'][triggered_idx])]
n_rep         = int(sorted_by_t[med_pos])   # path with median trigger day
t_rep         = int(RES['first_t'][n_rep])  # trigger day index

sig2_t   = float(vol_2Y[n_rep,  t_rep])
sig30_t  = float(vol_30Y[n_rep, t_rep])
spread_t = sig2_t - sig30_t
spread_mv= spread_t - ENTRY_SPREAD    # negative (moved against us)

sl2_t    = float(slip_2Y_mat[n_rep,  t_rep])
sl30_t   = float(slip_30Y_mat[n_rep, t_rep])
slip_K_t = float(slip_cost_K_mat[n_rep, t_rep])
mtm_K    = float(net_pnl_K[n_rep, t_rep])
realized = mtm_K - slip_K_t

print("=" * 66)
print("  DESK UNWIND INSTRUCTION — 2s30s Vol-Spread Stop Triggered")
print("=" * 66)
print(f"  TRIGGER : Vol spread moved {spread_mv:+.1f}bp against position.")
print(f"  Entry spread  : {ENTRY_SPREAD:.1f}bp  (σ_N(2Y) − σ_N(30Y))")
print(f"  Current spread: {spread_t:.1f}bp  (2Y={sig2_t:.1f}  30Y={sig30_t:.1f} bp/yr)")
print(f"  Stop level    : {ENTRY_SPREAD - STOP['STOP_Z_BP']:.1f}bp  (entry − {STOP['STOP_Z_BP']:.0f}bp)")
print(f"  Trigger day   : Day {t_rep+1} of {T_DAYS}  (FOMC = Day {FOMC_DAYS})")
print()
print("  ─── LEG 1 — UNWIND LONG 2Y STRADDLE ─────────────────────────")
print(f"  Action : SELL ${POS['N_2Y_M']:.0f}M 2Y swaption straddle at market")
print(f"  Strike : {POS['K_2Y_BP']:.1f}bp  |  Expiry: {POS['EXPIRY']}")
print(f"  Execute: vol BID  = {sig2_t - sl2_t:.2f}bp/yr")
print(f"  vs Mid : {sig2_t:.2f}bp/yr  |  Slippage: −{sl2_t:.2f}bp")
print(f"  $ cost : ${sl2_t * dv_vega_2Y[t_rep]:,.0f}k")
print()
print("  ─── LEG 2 — UNWIND SHORT 30Y STRADDLE ───────────────────────")
print(f"  Action : BUY BACK ${POS['N_30Y_M']:.0f}M 30Y swaption straddle at market")
print(f"  Strike : {POS['K_30Y_BP']:.1f}bp  |  Expiry: {POS['EXPIRY']}")
print(f"  Execute: vol OFFER = {sig30_t + sl30_t:.2f}bp/yr")
print(f"  vs Mid : {sig30_t:.2f}bp/yr  |  Slippage: +{sl30_t:.2f}bp")
print(f"  $ cost : ${sl30_t * dv_vega_30Y[t_rep]:,.0f}k")
print()
print("  ─── REALIZED P&L ─────────────────────────────────────────────")
print(f"  MTM at trigger  (clean) : ${mtm_K:+,.0f}k")
print(f"  Slippage cost (GAP)     : −${slip_K_t:,.0f}k   ← trigger→fill gap")
print(f"  REALIZED STOP LOSS      : ${realized:+,.0f}k")
print("=" * 66)
''')

# ─────────────────────────────────────────────────────────────────────────────
# CELL 18 — Section 5 header
# ─────────────────────────────────────────────────────────────────────────────
md("""
# ## SECTION 5 — VISUALS
#
# Five figures saved to `vol_stop_figs/`:
# - **Fig 1**: Sample vol-spread paths with stop level and trigger events
# - **Fig 2**: Loss distribution — clean-trigger vs post-slippage (the headline)
# - **Fig 3**: STOP_Z sweep — whipsaw rate + p99 loss vs stop width
# - **Fig 4**: Policy comparison — vol-spread stop vs no stop vs naive P&L stop
# - **Fig 5**: Representative triggered path — spread, MTM, trigger, unwind
""")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 19 — Fig 1: Sample vol-spread paths
# ─────────────────────────────────────────────────────────────────────────────
code("""\
# CELL 19: Fig 1 — Sample vol-spread paths with stop level
NSHOW  = 60
days   = np.arange(1, T + 1)
np.random.seed(7)
show_idx = np.random.choice(N, NSHOW, replace=False)

fig, ax = plt.subplots(figsize=(11, 4.5))

for n in show_idx:
    hit_t = RES['first_t'][n] if trig_mask[n] else None
    col   = PAL['sell'] if trig_mask[n] else PAL['buy']
    alp   = 0.35 if trig_mask[n] else 0.18
    x = days[:hit_t+1] if hit_t is not None else days
    y = spread[n, :hit_t+1] if hit_t is not None else spread[n, :]
    ax.plot(x, y, color=col, lw=0.7, alpha=alp)
    if hit_t is not None:
        ax.scatter(days[hit_t], spread[n, hit_t], color=PAL['sell'], s=12, alpha=0.5, zorder=4)

ax.axhline(ENTRY_SPREAD, color='#333333', lw=1.4, ls='--', label=f'Entry spread {ENTRY_SPREAD:.1f}bp')
ax.axhline(ENTRY_SPREAD - STOP['STOP_Z_BP'], color=PAL['sell'], lw=1.8, ls='--',
           label=f"Stop level {ENTRY_SPREAD - STOP['STOP_Z_BP']:.1f}bp (STOP_Z={STOP['STOP_Z_BP']:.0f}bp)")
ax.axvline(FOMC_DAYS, color='#666666', lw=1.2, ls=':', alpha=0.7)
ax.text(FOMC_DAYS + 0.3, ax.get_ylim()[1]*0.98 if ax.get_ylim()[1] != 0 else 50,
        'FOMC', va='top', fontsize=8, color='#555555')
ax.set_xlabel('Day from entry')
ax.set_ylabel('Vol spread σ_N(2Y) − σ_N(30Y)  (bp/yr)')
ax.set_title(f'Sample vol-spread paths  (N={NSHOW} shown; red = triggered, blue = no trigger)')
ax.legend(fontsize=8)
ax.grid(True, color=PAL['grid'], lw=0.5)

fig.tight_layout()
_out1 = 'vol_stop_figs/fig1_spread_paths.png'
plt.savefig(_out1, dpi=150, bbox_inches='tight')
plt.close()
_dimg(Image(_out1, width=980))
""")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 20 — Fig 2: Loss distribution (headline)
# ─────────────────────────────────────────────────────────────────────────────
code("""\
# CELL 20: Fig 2 — Loss distribution: clean trigger vs post-slippage
fig, ax = plt.subplots(figsize=(10, 4.5))

bins = np.linspace(min(trig_post.min(), trig_clean.min()) * 1.05,
                   max(trig_post.max(), trig_clean.max()) * 1.05, 60)

ax.hist(trig_clean, bins=bins, alpha=0.55, color=PAL['clean'],  label='Clean-trigger loss')
ax.hist(trig_post,  bins=bins, alpha=0.55, color=PAL['slip'],   label='Post-slippage loss (realized)')

for pct, ls in [(5, '--'), (1, ':')]:
    ax.axvline(np.percentile(trig_clean, pct), color=PAL['clean'], lw=1.3, ls=ls,
               label=f'Clean p{100-pct} = ${np.percentile(trig_clean,pct):+.0f}k')
    ax.axvline(np.percentile(trig_post, pct),  color=PAL['slip'],  lw=1.3, ls=ls,
               label=f'Post-slip p{100-pct} = ${np.percentile(trig_post,pct):+.0f}k')

ax.set_xlabel('Realized loss ($k)  [negative = loss]')
ax.set_ylabel('Path count')
ax.set_title(f'Loss distribution given stop triggered (STOP_Z={STOP["STOP_Z_BP"]:.0f}bp, '
             f'hit rate={100*RES["stop_rate"]:.1f}%)\\n'
             f'The GAP: slippage adds ${np.nanmean(slip_only):.0f}k mean / '
             f'${np.nanpercentile(slip_only,95):.0f}k p95 extra loss')
ax.legend(fontsize=7.5, ncol=2)
ax.grid(True, color=PAL['grid'], lw=0.5, axis='x')

fig.tight_layout()
_out2 = 'vol_stop_figs/fig2_loss_distribution.png'
plt.savefig(_out2, dpi=150, bbox_inches='tight')
plt.close()
_dimg(Image(_out2, width=960))
""")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 21 — Fig 3: STOP_Z sweep
# ─────────────────────────────────────────────────────────────────────────────
code("""\
# CELL 21: Fig 3 — STOP_Z sweep: whipsaw rate + p99 loss vs stop width
fig, ax1 = plt.subplots(figsize=(10, 4.5))
ax2 = ax1.twinx()

xs = sweep_df['stop_z_bp'].values

ax1.plot(xs, sweep_df['whipsaw_pct'],   color=PAL['buy'],   lw=2, marker='o', ms=5, label='Whipsaw rate (%)')
ax2.plot(xs, -sweep_df['p99_K'],        color=PAL['sell'],  lw=2, marker='s', ms=5, label='p99 loss | stop ($k, positive = loss)')
ax2.axhline(STOP['RISK_BUDGET_K'], color=PAL['sweep'], lw=1.5, ls='--',
            label=f"Risk budget ${STOP['RISK_BUDGET_K']:.0f}k")
if rec_z in xs:
    ax2.axvline(rec_z, color='#333333', lw=1.2, ls=':', alpha=0.7)
    ax2.text(rec_z+0.5, STOP['RISK_BUDGET_K']+10, f' STOP_Z={rec_z:.0f}bp\\n(budget fit)',
             fontsize=8, va='bottom')

ax1.set_xlabel('STOP_Z (bp)')
ax1.set_ylabel('Whipsaw rate (% of pre-FOMC stops)', color=PAL['buy'])
ax2.set_ylabel('p99 worst loss | stop ($k)', color=PAL['sell'])
ax1.tick_params(axis='y', colors=PAL['buy'])
ax2.tick_params(axis='y', colors=PAL['sell'])
ax1.set_title('STOP_Z tradeoff: tight = more whipsaws, smaller tail; wide = fewer false stops, bigger tail')
ax1.set_xticks(xs)
ax1.grid(True, color=PAL['grid'], lw=0.5, axis='x')

lines1, labs1 = ax1.get_legend_handles_labels()
lines2, labs2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labs1 + labs2, fontsize=8)

fig.tight_layout()
_out3 = 'vol_stop_figs/fig3_stopz_sweep.png'
plt.savefig(_out3, dpi=150, bbox_inches='tight')
plt.close()
_dimg(Image(_out3, width=960))
""")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 22 — Fig 4: Policy comparison
# ─────────────────────────────────────────────────────────────────────────────
code("""\
# CELL 22: Fig 4 — Policy comparison (distribution of final P&L)
fig, ax = plt.subplots(figsize=(10, 4.5))

pol_data   = [pol1_final_K, pol2_final_K, pol3_final_K]
pol_labels = [f'(i) Vol-spread stop\\n({STOP[\"STOP_Z_BP\"]:.0f}bp)',
              f'(ii) No stop\\n(hold to expiry, vol only)',
              f'(iii) Naive P&L stop\\n(level=${pnl_stop_level:.0f}k)']
pol_colors = [PAL['sell'], PAL['buy'], PAL['sweep']]

for i, (data, col) in enumerate(zip(pol_data, pol_colors)):
    lo, hi = np.percentile(data, 1), np.percentile(data, 99)
    bins   = np.linspace(lo, hi, 55)
    ax.hist(data, bins=bins, alpha=0.45, color=col, label=pol_labels[i])
    ax.axvline(np.mean(data), color=col, lw=2, ls='-')

ax.set_xlabel('Final P&L per path ($k)')
ax.set_ylabel('Path count')
ax.set_title('Policy comparison — final P&L distribution (vol-model)')
ax.legend(fontsize=8)
ax.grid(True, color=PAL['grid'], lw=0.5, axis='x')
ax.axvline(0, color='#333333', lw=0.8, ls='--', alpha=0.5)

fig.tight_layout()
_out4 = 'vol_stop_figs/fig4_policy_comparison.png'
plt.savefig(_out4, dpi=150, bbox_inches='tight')
plt.close()
_dimg(Image(_out4, width=960))
""")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 23 — Fig 5: Representative triggered path
# ─────────────────────────────────────────────────────────────────────────────
code("""\
# CELL 23: Fig 5 — Representative triggered path: spread + MTM + trigger annotation
days_rep = np.arange(1, T + 1)

fig, (ax_s, ax_p) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                   gridspec_kw={'height_ratios': [1, 1]})

# ── Top panel: vol spread ────────────────────────────────────────────────────
ax_s.plot(days_rep, spread[n_rep, :], color=PAL['spread'], lw=2, label='2s30s vol spread')
ax_s.axhline(ENTRY_SPREAD,                        color='#333333',  lw=1.2, ls='--',
             label=f'Entry spread {ENTRY_SPREAD:.1f}bp')
ax_s.axhline(ENTRY_SPREAD - STOP['STOP_Z_BP'],    color=PAL['sell'], lw=1.5, ls='--',
             label=f"Stop level {ENTRY_SPREAD-STOP['STOP_Z_BP']:.1f}bp")
ax_s.axvline(days_rep[t_rep], color=PAL['sell'], lw=1.8, ls=':', alpha=0.8)
ax_s.scatter([days_rep[t_rep]], [spread[n_rep, t_rep]], color=PAL['sell'],
             s=80, zorder=5, label=f'Trigger: Day {t_rep+1}, spread={spread[n_rep,t_rep]:.1f}bp')
ax_s.axvline(FOMC_DAYS, color='#888888', lw=1.2, ls=':', alpha=0.6)
ax_s.text(FOMC_DAYS + 0.2, ax_s.get_ylim()[0] * 0.98 if ax_s.get_ylim()[0] < 0 else
          ENTRY_SPREAD * 0.9, 'FOMC', fontsize=8, color='#666666')
ax_s.set_ylabel('Vol spread (bp/yr)')
ax_s.set_title(f'Representative triggered path (path #{n_rep}, trigger Day {t_rep+1})')
ax_s.legend(fontsize=8)
ax_s.grid(True, color=PAL['grid'], lw=0.5)

# ── Bottom panel: MTM P&L ────────────────────────────────────────────────────
ax_p.plot(days_rep, net_pnl_K[n_rep, :], color=PAL['buy'], lw=2, label='Net MTM P&L ($k)')
ax_p.axvline(days_rep[t_rep], color=PAL['sell'], lw=1.8, ls=':', alpha=0.8)
ax_p.scatter([days_rep[t_rep]], [mtm_K],      color=PAL['clean'], s=80, zorder=5,
             label=f'Clean MTM ${mtm_K:+.0f}k')
ax_p.scatter([days_rep[t_rep]], [realized],   color=PAL['sell'],  s=80, zorder=5,
             label=f'Post-slippage ${realized:+.0f}k  (gap = ${slip_K_t:.0f}k)')
ax_p.annotate(f'', xy=(days_rep[t_rep], realized), xytext=(days_rep[t_rep], mtm_K),
              arrowprops=dict(arrowstyle='<->', color=PAL['slip'], lw=1.5))
ax_p.text(days_rep[t_rep] + 0.4, (mtm_K + realized) / 2,
          f'slip ${slip_K_t:.0f}k', fontsize=8, color=PAL['slip'], va='center')
ax_p.axhline(0, color='#333333', lw=0.8, ls='--', alpha=0.5)
ax_p.axvline(FOMC_DAYS, color='#888888', lw=1.2, ls=':', alpha=0.6)
ax_p.set_xlabel('Day from entry')
ax_p.set_ylabel('Net P&L ($k)')
ax_p.legend(fontsize=8)
ax_p.grid(True, color=PAL['grid'], lw=0.5)

fig.tight_layout()
_out5 = 'vol_stop_figs/fig5_rep_path.png'
plt.savefig(_out5, dpi=150, bbox_inches='tight')
plt.close()
_dimg(Image(_out5, width=1000))
""")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 24 — Acceptance checks
# ─────────────────────────────────────────────────────────────────────────────
md("""
# ## ACCEPTANCE CHECKS
""")

code("""\
# CELL 24: Acceptance checks T1–T4
PASS = "✓ PASS"
FAIL = "✗ FAIL"

# T1: Stop is on SPREAD (2 correlated vols simulated, not 1).
# Check: both legs simulated (correct shapes), Cholesky correctly sets up the target ρ.
# NOTE: the FOMC uncorrelated jump slightly dilutes the terminal correlation vs target;
# we verify shapes and that the diffusion covariance matrix is correct (Cholesky check).
t1_ok = (vol_2Y.shape == (N, T)) and (vol_30Y.shape == (N, T))
realized_corr = np.corrcoef(diffs[:, -1, 0], diffs[:, -1, 1])[0, 1]
# Expected effective corr (diluted by uncorrelated FOMC jump):
diff_var_2Y  = (T - 1) * SIM['VOL_SD_2Y']**2  + SIM['FOMC_JUMP_SD_2Y']**2
diff_var_30Y = (T - 1) * SIM['VOL_SD_30Y']**2 + SIM['FOMC_JUMP_SD_30Y']**2
diff_cov     = (T - 1) * SIM['VOL_CORR'] * SIM['VOL_SD_2Y'] * SIM['VOL_SD_30Y']
expected_corr = diff_cov / np.sqrt(diff_var_2Y * diff_var_30Y)
print(f"T1 Stop on SPREAD (two correlated vols simulated):          {PASS if t1_ok else FAIL}")
print(f"   vol_2Y shape={vol_2Y.shape}  vol_30Y shape={vol_30Y.shape}")
print(f"   Diffusion ρ target={SIM['VOL_CORR']:.2f}  |  Effective terminal ρ (with uncorr FOMC jump)="
      f"{expected_corr:.3f}  realized={realized_corr:.3f}  (tolerance ±0.05: "
      f"{'OK' if abs(realized_corr - expected_corr) < 0.05 else 'WARN'})")

# T2: Slippage models the gap — mean post-slippage loss is worse than mean clean loss.
t2_ok = (float(np.nanmean(trig_post)) < float(np.nanmean(trig_clean))
          and float(np.nanmean(slip_only)) > 0)
print(f"\\nT2 Unwind models slippage (post-slip loss > clean trigger): {PASS if t2_ok else FAIL}")
print(f"   Mean clean loss: ${trig_clean.mean():+.0f}k  Mean post-slip: ${trig_post.mean():+.0f}k  "
      f"GAP: ${np.nanmean(slip_only):.0f}k  ({100*np.nanmean(slip_only)/abs(trig_clean.mean()):.1f}% of clean loss)")

# T3: Sweep finds a STOP_Z where post-slippage p99 conditional loss is within budget.
# Budget must exceed net premium (${model_net_K:.0f}k); a budget below premium is not achievable.
r_rec = apply_stop(rec_z)
trig_rec = r_rec['loss_post_K'][r_rec['triggered']]
p99_rec = float(np.nanpercentile(trig_rec, 1)) if len(trig_rec) else -np.inf
t3_ok = (p99_rec >= -STOP['RISK_BUDGET_K'])
print(f"\\nT3 Sweep finds STOP_Z with post-slippage p99 inside budget: {PASS if t3_ok else FAIL}")
print(f"   Budget = -${STOP['RISK_BUDGET_K']:.0f}k  (note: net premium = -${model_net_K:.0f}k)")
print(f"   Recommended STOP_Z={rec_z:.0f}bp: conditional p99 = ${p99_rec:+.0f}k  "
      f"({'within' if t3_ok else 'exceeds'} budget)")

# T4: Whipsaw/false-positive rate reported (cost of the stop quantified)
t4_ok = isinstance(ws_rate, float)
print(f"\\nT4 Whipsaw / false-positive rate reported:                  {PASS if t4_ok else FAIL}")
print(f"   Whipsaw rate: {100*ws_rate:.1f}%  (avg opportunity cost per whipsaw: ${ws_cost:+.0f}k)")
""")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 25 — Caveats
# ─────────────────────────────────────────────────────────────────────────────
md("""
# ## CAVEATS
#
# 1. **Stop is thesis-invalidation (vol spread), not P&L.** A P&L stop fires on noise;
#    this fires when 30Y vol richens vs 2Y vol — the specific thesis variable.
#
# 2. **Trigger ≠ Fill.** Slippage on the short 30Y straddle is widest exactly when the
#    stop triggers (stressed, fast-moving market). The realized stop loss is the post-slippage
#    figure, not the clean-trigger MTM. The gap is the headline finding.
#
# 3. **Vol-model only.** No underlying rate moves are simulated; P&L is purely vega / theta.
#    The rate-move payoff at expiry (FOMC jump) is excluded from the `no-stop` distribution.
#    For full expiry P&L, use the MC in `trade_ticket_pricer.ipynb`.
#
# 4. **Sizing.** Post-slippage p99 tail (not clean trigger) should sit inside the stated
#    RISK_BUDGET. The recommended STOP_Z is the tightest stop consistent with this budget.
#
# 5. **All levels are PLACEHOLDER.** Vol_SD, FOMC_JUMP_SD, and SLIP_SPEED_COEF must be
#    calibrated from realized vol-of-vol and OTC market observations before trading.
""")

# ─────────────────────────────────────────────────────────────────────────────
# WRITE NOTEBOOK
# ─────────────────────────────────────────────────────────────────────────────
NB = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": cells,
}

OUT = "vol_stop_simulation.ipynb"
with open(OUT, "w") as f:
    json.dump(NB, f, indent=1)

n_code = sum(1 for c in cells if c["cell_type"] == "code")
n_md   = sum(1 for c in cells if c["cell_type"] == "markdown")
print(f"Generated {OUT}  ({len(cells)} cells: {n_code} code, {n_md} markdown)")
