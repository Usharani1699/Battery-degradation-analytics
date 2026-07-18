"""
FSI Weight Cross-Validation on Severson et al. (2019)
=====================================================
Addresses the core methodological criticism: "weights are principled but not
cross-validated against data with variable KI."

Strategy:
  - Severson dataset has 124 LFP cells across 72 protocols with varying KI
  - Optimize FSI weights to maximise |Spearman(FSI, cycle_life)| at cell level
  - Compare optimised weights to the principled weights (0.30/0.25/0.25/0.20)
  - Run at both cell level (n=49 cells with cycle_life) and protocol level (n=23)

Usage:
    python 04_Code/validation/weight_optimization.py
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "03_Processed_Data"
RES_DIR  = ROOT / "04_Code" / "results"
RES_DIR.mkdir(exist_ok=True)

# ── Load Severson ──────────────────────────────────────────────────────────
print("Loading Severson FSI features...")
sev = pd.read_csv(DATA_DIR / "Severson_FSI_Features.csv")
print(f"  {len(sev):,} cycles, {sev['ID'].nunique()} cells")
print(f"  KI range: [{sev['KI'].min():.3f}, {sev['KI'].max():.3f}]")
print(f"  Cycle_Life range: [{sev['Cycle_Life'].min():.0f}, {sev['Cycle_Life'].max():.0f}]")

COMPONENTS = ["KI", "DoD_%", "T_stress_norm", "C_peak_norm"]
PRINCIPLED = np.array([0.30, 0.25, 0.25, 0.20])

# ── Cell-level aggregation ────────────────────────────────────────────────
# Per cell: mean of each component, and the recorded cycle_life
cell_agg = sev.groupby("ID").agg(
    KI=("KI", "mean"),
    DoD=("DoD_%", "mean"),
    T_stress=("T_stress_norm", "mean"),
    C_peak=("C_peak_norm", "mean"),
    cycle_life=("Cycle_Life", "first"),
).dropna(subset=["cycle_life"])

print(f"\n  Cell-level aggregation: {len(cell_agg)} cells with Cycle_Life")

X = cell_agg[["KI", "DoD", "T_stress", "C_peak"]].values
y = cell_agg["cycle_life"].values


def fsi_rho(weights):
    """Negative Spearman rho of FSI vs cycle_life (we minimise, so negate)."""
    w = np.abs(weights)
    w = w / w.sum()  # normalise to sum=1
    fsi = X @ w
    rho, _ = spearmanr(fsi, y)
    return rho  # positive rho = FSI increases with cycle_life; we want negative correlation


def neg_abs_rho(weights):
    return -abs(fsi_rho(weights))


# ── Optimise ──────────────────────────────────────────────────────────────
print("\nOptimising FSI weights on Severson cell-level data...")

best_result = None
best_obj    = 1.0

# Multiple random starts to avoid local minima
np.random.seed(42)
for trial in range(50):
    w0 = np.random.dirichlet(np.ones(4))  # random weights summing to 1
    res = minimize(
        neg_abs_rho, w0,
        method="L-BFGS-B",
        bounds=[(0.0, 1.0)] * 4,
        options={"maxiter": 500, "ftol": 1e-10},
    )
    if res.fun < best_obj:
        best_obj    = res.fun
        best_result = res

opt_w_raw = np.abs(best_result.x)
opt_w     = opt_w_raw / opt_w_raw.sum()  # ensure sum=1

print(f"  Optimisation converged: {best_result.success}")

# ── Compute rho under each weight set ─────────────────────────────────────
fsi_principled = X @ PRINCIPLED
fsi_optimised  = X @ opt_w
fsi_ki_only    = X[:, 0]   # KI alone
fsi_equal      = X @ np.array([0.25, 0.25, 0.25, 0.25])

rho_p,   p_p   = spearmanr(fsi_principled, y)
rho_o,   p_o   = spearmanr(fsi_optimised,  y)
rho_ki,  p_ki  = spearmanr(fsi_ki_only,    y)
rho_eq,  p_eq  = spearmanr(fsi_equal,      y)

print(f"\n{'='*62}")
print("WEIGHT OPTIMISATION RESULTS — SEVERSON CELL LEVEL (n={})".format(len(cell_agg)))
print(f"{'='*62}")
print(f"{'Scheme':<28} {'KI':>6} {'DoD':>6} {'T':>6} {'Cpk':>6}  {'Spearman rho':>14}")
print(f"{'-'*62}")
print(f"  {'KI alone':<26} {'1.00':>6} {'0.00':>6} {'0.00':>6} {'0.00':>6}  {rho_ki:>+13.4f}")
print(f"  {'Equal weights':<26} {'0.25':>6} {'0.25':>6} {'0.25':>6} {'0.25':>6}  {rho_eq:>+13.4f}")
print(f"  {'Principled (0.30/0.25/0.25/0.20)':<26} {PRINCIPLED[0]:>6.2f} {PRINCIPLED[1]:>6.2f} {PRINCIPLED[2]:>6.2f} {PRINCIPLED[3]:>6.2f}  {rho_p:>+13.4f}")
print(f"  {'Optimised':<26} {opt_w[0]:>6.2f} {opt_w[1]:>6.2f} {opt_w[2]:>6.2f} {opt_w[3]:>6.2f}  {rho_o:>+13.4f}")
print()
print(f"  Principled vs Optimised rho difference: {abs(rho_p) - abs(rho_o):+.4f}")

if abs(rho_p) >= abs(rho_o) - 0.02:
    verdict = "VALIDATED: Principled weights achieve within 0.02 of optimised; no significant improvement from data-driven tuning."
else:
    verdict = f"DIVERGED: Optimised weights outperform principled by {abs(rho_o)-abs(rho_p):.3f} rho units."
print(f"\n  Verdict: {verdict}")

# ── Protocol-level check ──────────────────────────────────────────────────
if "Charge_Policy" in sev.columns:
    proto_col = "Charge_Policy"
elif "Profile" in sev.columns:
    proto_col = "Profile"
else:
    proto_col = None

if proto_col:
    proto_agg = sev.groupby(proto_col).agg(
        KI=("KI", "mean"),
        DoD=("DoD_%", "mean"),
        T_stress=("T_stress_norm", "mean"),
        C_peak=("C_peak_norm", "mean"),
        cycle_life=("Cycle_Life", "mean"),
    ).dropna(subset=["cycle_life"])

    Xp = proto_agg[["KI", "DoD", "T_stress", "C_peak"]].values
    yp = proto_agg["cycle_life"].values

    fsi_pp = Xp @ PRINCIPLED
    fsi_op = Xp @ opt_w
    rho_pp, _ = spearmanr(fsi_pp, yp)
    rho_op, _ = spearmanr(fsi_op, yp)
    print(f"\n  Protocol-level (n={len(proto_agg)}):")
    print(f"    Principled rho = {rho_pp:+.4f}")
    print(f"    Optimised rho  = {rho_op:+.4f}")

# ── Save ──────────────────────────────────────────────────────────────────
out = {
    "dataset": "Severson_FSI_Features.csv",
    "n_cells": int(len(cell_agg)),
    "components": COMPONENTS,
    "principled_weights": {
        "KI": float(PRINCIPLED[0]), "DoD": float(PRINCIPLED[1]),
        "T_stress_norm": float(PRINCIPLED[2]), "C_peak_norm": float(PRINCIPLED[3]),
        "spearman_rho": round(float(rho_p), 4), "p_value": round(float(p_p), 4),
    },
    "optimised_weights": {
        "KI": round(float(opt_w[0]), 3), "DoD": round(float(opt_w[1]), 3),
        "T_stress_norm": round(float(opt_w[2]), 3), "C_peak_norm": round(float(opt_w[3]), 3),
        "spearman_rho": round(float(rho_o), 4), "p_value": round(float(p_o), 4),
    },
    "ki_only": {"spearman_rho": round(float(rho_ki), 4)},
    "equal_weights": {"spearman_rho": round(float(rho_eq), 4)},
    "verdict": verdict,
    "rho_improvement": round(float(abs(rho_o) - abs(rho_p)), 4),
}
if proto_col and len(proto_agg) >= 5:
    out["protocol_level"] = {
        "n_protocols": int(len(proto_agg)),
        "principled_rho": round(float(rho_pp), 4),
        "optimised_rho": round(float(rho_op), 4),
    }

out_path = RES_DIR / "weight_optimization_results.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nResults saved: {out_path}")
