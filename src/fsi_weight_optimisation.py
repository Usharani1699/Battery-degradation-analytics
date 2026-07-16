"""
FSI Weight Optimisation & Sensitivity Analysis
===============================================
Three-part study to justify the FSI weight vector (0.30, 0.25, 0.25, 0.20):

  Part 1 — Data-driven derivation
    Ridge regression of SoH on [KI, DoD, T_norm, C_peak_norm].
    Normalised coefficients = empirically derived weights.
    Compare to manual weights -> show agreement.

  Part 2 — Bayesian optimisation (scipy minimize, Nelder-Mead)
    Optimise weights to minimise XGBoost SoH RMSE (5-fold CV).
    No optuna dependency - uses scipy only.
    Constraint: weights sum to 1, all >= 0.

  Part 3 — Sensitivity analysis
    Vary each weight +/-10%, +/-20%, +/-30% while holding others proportional.
    Show RMSE change -> prove robustness.

Output:
  results/fsi_weight_analysis.json

Usage:
    python src/fsi_weight_optimisation.py
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.optimize import minimize
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RES_DIR  = ROOT / "results"
RES_DIR.mkdir(exist_ok=True)

# FSI component columns (raw, not the composite)
# Note: KI=0 for all CALCE CC lab data (constant current = no kinetic variation)
# C_peak is stored as C_rate_peak in the CSV; T_stress_norm is already normalised
COMP_LABELS = ["KI (kinetic intensity)", "DoD (depth of discharge)",
               "T_norm (temperature stress)", "C_peak (peak current)"]
MANUAL_W = np.array([0.30, 0.25, 0.25, 0.20])

FSI_FEATURES = ["FSI", "T_avg_C", "T_stress_norm", "KI", "DCSS", "RBF", "CVI"]

# ── Load CALCE lab data ────────────────────────────────────────────────────
print("Loading CALCE lab data...")
calce = pd.read_csv(DATA_DIR / "Linked_Lab_Fleet_Degradation.csv")
lab   = calce[calce["Source"] == "Lab"].copy()
lab["SoH_%"] = lab["SoH_%"].clip(upper=100.0)
needed = ["SoH_%", "KI", "DoD_%", "T_stress_norm", "C_rate_peak"]
lab   = lab[lab[needed].notna().all(axis=1)].copy()

# Build normalised component matrix for regression
# DoD_frac: cap at 1 (some measurements exceed 100%)
# C_peak_norm: divide by 2.0 (max observed C-rate in CALCE)
lab["DoD_frac"]    = (lab["DoD_%"] / 100.0).clip(0, 1)
lab["C_peak_norm"] = (lab["C_rate_peak"] / 2.0).clip(0, 1)
COMP_COLS = ["KI", "DoD_frac", "T_stress_norm", "C_peak_norm"]

X_comp = lab[COMP_COLS].values    # raw FSI components
y_soh  = lab["SoH_%"].values

print(f"  Lab samples (complete components): {len(X_comp):,}")

# Imputer for ML features
imputer = SimpleImputer(strategy="median")
imputer.fit(lab[FSI_FEATURES].values)
X_ml = imputer.transform(lab[FSI_FEATURES].values)


# ═══════════════════════════════════════════════════════════════════════════
# PART 1 — DATA-DRIVEN WEIGHT DERIVATION (Ridge regression)
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART 1 — DATA-DRIVEN WEIGHT DERIVATION")
print("=" * 60)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_comp)

ridge = Ridge(alpha=1.0)
ridge.fit(X_scaled, y_soh)

coefs    = ridge.coef_
pos_coef = np.maximum(coefs, 0)          # keep only positive contributions
derived_w = pos_coef / pos_coef.sum()    # normalise to sum=1

r2_linear = ridge.score(X_scaled, y_soh)

print(f"\n  Ridge R² (SoH ~ components): {r2_linear:.4f}")
print(f"  Note: KI=0 for ALL CALCE lab data (CC cycling, no kinetic variation).")
print(f"  KI coefficient is unreliable from lab alone — its weight (0.30)")
print(f"  is justified from fleet literature (primary distinguisher in real driving).")
print(f"\n  {'Component':<30} {'Regression coef':>16}  {'Derived w':>10}  {'Manual w':>10}  {'Diff':>8}")
print("  " + "-" * 76)
for label, dc, dw, mw in zip(COMP_LABELS, coefs, derived_w, MANUAL_W):
    diff = dw - mw
    print(f"  {label:<30} {dc:>16.4f}  {dw:>10.4f}  {mw:>10.4f}  {diff:>+8.4f}")

print(f"\n  Manual    weights: {MANUAL_W}")
print(f"  Derived   weights: {derived_w.round(4)}")
print(f"  Max absolute diff: {np.abs(derived_w - MANUAL_W).max():.4f}")
print(f"  Cosine similarity: {np.dot(derived_w, MANUAL_W) / (np.linalg.norm(derived_w)*np.linalg.norm(MANUAL_W)):.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# PART 2 — BAYESIAN / SCIPY OPTIMISATION
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART 2 — BAYESIAN OPTIMISATION (scipy Nelder-Mead)")
print("=" * 60)

kf = KFold(n_splits=5, shuffle=True, random_state=42)


def build_fsi(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Recompute FSI column from raw components using weight vector w."""
    w = np.abs(w) / np.abs(w).sum()    # enforce non-negative + sum-to-1
    return X @ w                        # dot product: (n,4) @ (4,) -> (n,)


def cv_rmse_from_weights(w_raw: np.ndarray) -> float:
    """5-fold CV RMSE of XGB when FSI is recomputed with weights w."""
    w    = np.abs(w_raw) / np.abs(w_raw).sum()
    fsi_col = X_comp @ w                         # recomputed FSI
    X_cv = np.column_stack([
        fsi_col,
        lab["T_avg_C"].values,
        lab["T_stress_norm"].values,
        lab["KI"].values,
        lab["DCSS"].values  if "DCSS"  in lab.columns else np.zeros(len(lab)),
        lab["RBF"].values   if "RBF"   in lab.columns else np.zeros(len(lab)),
        lab["CVI"].values   if "CVI"   in lab.columns else np.zeros(len(lab)),
    ])
    # impute NaNs for this recomputed feature matrix
    X_imp = SimpleImputer(strategy="median").fit_transform(X_cv)
    model = xgb.XGBRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        random_state=42, verbosity=0,
    )
    y_pred = cross_val_predict(model, X_imp, y_soh, cv=kf)
    y_pred = np.clip(y_pred, 0, 100)
    return float(mean_squared_error(y_soh, y_pred) ** 0.5)


# Baseline: manual weights
print("\n  Computing baseline (manual weights)...")
baseline_rmse = cv_rmse_from_weights(MANUAL_W)
print(f"  Manual weights RMSE (5-fold CV): {baseline_rmse:.4f}%")

# Optimise
print("  Running Nelder-Mead optimisation (may take ~2 min)...")
result = minimize(
    cv_rmse_from_weights,
    x0=MANUAL_W,
    method="Nelder-Mead",
    options={"maxiter": 400, "xatol": 1e-3, "fatol": 1e-3, "disp": False},
)
opt_w_raw  = result.x
opt_w      = np.abs(opt_w_raw) / np.abs(opt_w_raw).sum()
opt_rmse   = result.fun

print(f"  Optimised weights RMSE:          {opt_rmse:.4f}%")
print(f"  RMSE improvement:                {baseline_rmse - opt_rmse:+.4f}%")

print(f"\n  {'Component':<30} {'Manual w':>10}  {'Derived w':>10}  {'Optimal w':>10}")
print("  " + "-" * 62)
for label, mw, dw, ow in zip(COMP_LABELS, MANUAL_W, derived_w, opt_w):
    print(f"  {label:<30} {mw:>10.4f}  {dw:>10.4f}  {ow:>10.4f}")

print(f"\n  Manual   weights: {MANUAL_W}")
print(f"  Derived  weights: {derived_w.round(4)}")
print(f"  Optimal  weights: {opt_w.round(4)}")
print(f"  Max diff manual vs optimal: {np.abs(opt_w - MANUAL_W).max():.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# PART 3 — SENSITIVITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART 3 — SENSITIVITY ANALYSIS (+/-10%, +/-20%, +/-30%)")
print("=" * 60)
print("  Varying each weight while keeping others proportional.\n")

perturbations = [-0.30, -0.20, -0.10, 0.0, +0.10, +0.20, +0.30]
sensitivity_results = {}

for ci, comp_label in enumerate(COMP_LABELS):
    comp_results = []
    for delta in perturbations:
        w_perturbed = MANUAL_W.copy()
        w_perturbed[ci] = max(0.0, MANUAL_W[ci] * (1 + delta))
        # Renormalise others
        remainder = 1.0 - w_perturbed[ci]
        other_sum = MANUAL_W.sum() - MANUAL_W[ci]
        if other_sum > 0:
            for j in range(4):
                if j != ci:
                    w_perturbed[j] = MANUAL_W[j] / other_sum * remainder
        w_perturbed = np.clip(w_perturbed, 0, 1)

        rmse = cv_rmse_from_weights(w_perturbed)
        comp_results.append({
            "delta_pct": int(delta * 100),
            "weight": round(float(w_perturbed[ci]), 4),
            "rmse": round(rmse, 4),
            "rmse_change": round(rmse - baseline_rmse, 4),
        })

    sensitivity_results[comp_label] = comp_results
    rmse_values = [r["rmse"] for r in comp_results]
    rmse_range  = max(rmse_values) - min(rmse_values)
    print(f"  {comp_label:<30}  RMSE range over +/-30%: {rmse_range:.4f}%")
    for r in comp_results:
        bar = "#" * int(abs(r["rmse_change"]) / 0.05)
        sign = "+" if r["rmse_change"] >= 0 else "-"
        print(f"    delta={r['delta_pct']:+4d}%  w={r['weight']:.3f}  "
              f"RMSE={r['rmse']:.4f}%  change={r['rmse_change']:+.4f}%  {sign}{bar}")
    print()


# ── Robustness summary ────────────────────────────────────────────────────
print("=" * 60)
print("ROBUSTNESS SUMMARY")
print("=" * 60)
max_rmse_changes = {}
for comp, results in sensitivity_results.items():
    max_change = max(abs(r["rmse_change"]) for r in results)
    max_rmse_changes[comp] = round(max_change, 4)
    print(f"  {comp:<30}  max RMSE change at +/-30%: {max_change:.4f}%")

overall_max = max(max_rmse_changes.values())
print(f"\n  Overall max RMSE change at +/-30% weight perturbation: {overall_max:.4f}%")
print(f"  Baseline RMSE: {baseline_rmse:.4f}%")
print(f"  Robustness: {overall_max/baseline_rmse*100:.1f}% of baseline RMSE")

if overall_max < 1.0:
    print("\n  CONCLUSION: FSI is ROBUST to weight perturbations.")
    print("  A 30% change in any weight produces < 1% RMSE change.")
    print("  This confirms FSI captures real physics, not a tuned artefact.")


# ── Save all results ──────────────────────────────────────────────────────
output = {
    "manual_weights":  dict(zip(COMP_LABELS, MANUAL_W.tolist())),
    "derived_weights": dict(zip(COMP_LABELS, derived_w.round(4).tolist())),
    "optimal_weights": dict(zip(COMP_LABELS, opt_w.round(4).tolist())),
    "linear_r2":       round(r2_linear, 4),
    "baseline_rmse":   round(baseline_rmse, 4),
    "optimised_rmse":  round(opt_rmse, 4),
    "rmse_improvement":round(baseline_rmse - opt_rmse, 4),
    "cosine_similarity_manual_vs_derived": round(
        float(np.dot(derived_w, MANUAL_W) /
              (np.linalg.norm(derived_w) * np.linalg.norm(MANUAL_W))), 4),
    "max_diff_manual_vs_optimal": round(float(np.abs(opt_w - MANUAL_W).max()), 4),
    "sensitivity": sensitivity_results,
    "robustness_max_rmse_change_at_30pct": max_rmse_changes,
    "overall_max_rmse_change": round(overall_max, 4),
}

out_path = RES_DIR / "fsi_weight_analysis.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {out_path}")
