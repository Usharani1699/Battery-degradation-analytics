"""
Feature Ablation Study
=======================
Addresses the SHAP circular logic criticism:
  FSI contains KI and T_stress_norm as sub-components.
  FSI's 82.4% SHAP importance may partly reflect self-encoding.

Ablation designs:
  A: FSI only (no sub-components)
  B: Sub-components only (KI + DoD + T_norm + C_peak — no composite FSI)
  C: All features (current design — FSI + sub-components)
  D: KI only
  E: DoD only

Each ablation uses CALCE 5-fold CV for RMSE.
Severson cell-level Spearman rho for ordinal validation.

Usage:
    python 04_Code/validation/feature_ablation.py
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
import xgboost as xgb

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "03_Processed_Data"
RES_DIR  = ROOT / "04_Code" / "results"
RES_DIR.mkdir(exist_ok=True)

# ── Load CALCE ─────────────────────────────────────────────────────────────
calce = pd.read_csv(DATA_DIR / "Linked_Lab_Fleet_Degradation.csv")
lab   = calce[calce["Source"] == "Lab"].copy()
lab["SoH_%"] = lab["SoH_%"].clip(upper=100.0)
lab_reg = lab[lab["SoH_%"].notna()].copy()
y_calce = lab_reg["SoH_%"].values

# ── Load Severson ──────────────────────────────────────────────────────────
sev = pd.read_csv(DATA_DIR / "Severson_FSI_Features.csv")
sev_cell = sev.groupby("ID").agg(
    FSI=("FSI", "mean"),
    KI=("KI", "mean"),
    DoD=("DoD_%", "mean"),
    T_stress_norm=("T_stress_norm", "mean"),
    C_peak_norm=("C_peak_norm", "mean"),
    cycle_life=("Cycle_Life", "first"),
).dropna(subset=["cycle_life"])


# ── Ablation configurations ────────────────────────────────────────────────
ABLATIONS = {
    "A — FSI only":           ["FSI"],
    "B — Sub-components only":["KI", "DoD_%", "T_stress_norm", "C_peak_norm", "DCSS", "RBF", "CVI"],
    "C — All (current)":      ["FSI", "T_avg_C", "T_stress_norm", "KI", "DCSS", "RBF", "CVI"],
    "D — KI only":            ["KI"],
    "E — DoD only":           ["DoD_%"],
    "F — FSI + T_avg_C":      ["FSI", "T_avg_C"],
}

XGB_PARAMS = dict(n_estimators=300, max_depth=6, learning_rate=0.05,
                   random_state=42, verbosity=0)
KF = KFold(n_splits=5, shuffle=True, random_state=42)


def run_cv(features: list) -> dict:
    """5-fold CV RMSE on CALCE."""
    cols = [f for f in features if f in lab_reg.columns]
    if not cols:
        return {}
    X   = lab_reg[cols].fillna(lab_reg[cols].median()).values
    rms, mae, r2s = [], [], []
    for tr, va in KF.split(X):
        m = xgb.XGBRegressor(**XGB_PARAMS)
        m.fit(X[tr], y_calce[tr])
        p = np.clip(m.predict(X[va]), 0, 100)
        rms.append(mean_squared_error(y_calce[va], p) ** 0.5)
        mae.append(mean_absolute_error(y_calce[va], p))
        r2s.append(r2_score(y_calce[va], p))
    return {
        "rmse": round(float(np.mean(rms)), 3),
        "rmse_std": round(float(np.std(rms)), 3),
        "mae": round(float(np.mean(mae)), 3),
        "r2": round(float(np.mean(r2s)), 4),
    }


def severson_rho(features: list) -> dict:
    """Spearman rho between mean feature-based FSI and cycle_life, cell level."""
    # Map ablation features to Severson columns
    sev_map = {
        "FSI": "FSI", "KI": "KI", "DoD_%": "DoD",
        "T_stress_norm": "T_stress_norm", "C_peak_norm": "C_peak_norm",
    }
    cols = [sev_map[f] for f in features if f in sev_map and sev_map[f] in sev_cell.columns]
    if not cols:
        return {}
    X_s = sev_cell[cols].values
    # Simple mean of available features as composite score
    score = X_s.mean(axis=1)
    rho, p = spearmanr(score, sev_cell["cycle_life"].values)
    return {"rho": round(float(rho), 4), "p": round(float(p), 4)}


# ── Run ────────────────────────────────────────────────────────────────────
print("Running feature ablation study...")
print("=" * 72)
print(f"{'Ablation':<30} {'RMSE':>8} {'±':>5} {'R²':>8} {'Sev rho':>10} {'p':>8}")
print(f"{'-'*72}")

results = {}
for name, feats in ABLATIONS.items():
    cv  = run_cv(feats)
    sev_r = severson_rho(feats)
    results[name] = {
        "features": feats,
        "n_features": len([f for f in feats if f in lab_reg.columns]),
        "calce_cv": cv,
        "severson_rho": sev_r,
    }
    rmse_s = f"{cv.get('rmse', 0):.3f}%" if cv else "—"
    std_s  = f"{cv.get('rmse_std', 0):.3f}" if cv else "—"
    r2_s   = f"{cv.get('r2', 0):.4f}" if cv else "—"
    rho_s  = f"{sev_r.get('rho', 0):+.4f}" if sev_r else "—"
    p_s    = f"{sev_r.get('p', 0):.4f}" if sev_r else "—"
    print(f"  {name:<28} {rmse_s:>8} {std_s:>5} {r2_s:>8} {rho_s:>10} {p_s:>8}")

print(f"\nKey questions answered:")
print("  Q1: Does adding sub-components to FSI improve CALCE RMSE? (A vs C)")
print("  Q2: Does FSI composite outperform sub-components alone on Severson? (A vs B rho)")
print("  Q3: Does KI alone provide Severson ordinal signal? (D rho)")

# ── Save ──────────────────────────────────────────────────────────────────
out_path = RES_DIR / "feature_ablation_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved: {out_path}")
