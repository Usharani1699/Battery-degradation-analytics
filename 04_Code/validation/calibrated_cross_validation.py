"""
Calibrated Cross-Dataset Validation
=====================================
Extends cross_dataset_validation.py with early-cycle intercept calibration.

Strategy: for each target cell, use the first N_EARLY cycles to compute a
per-cell bias offset (mean actual - mean predicted). Apply this offset to ALL
predictions for that cell. This simulates what a fleet BMS would do: take a
handful of initial capacity measurements when a new pack is installed, then
recalibrate the model's absolute scale.

Key distinction from retraining:
  - No parameters are changed
  - Only the intercept (mean shift) is corrected
  - The model structure (slopes, feature weights) is untouched
  - Calibration data come from the TARGET dataset only (no leakage from training)

Also tests asymmetric T_norm: cold temperatures receive half the stress
penalty of equivalent hot temperatures (Arrhenius-motivated).

Usage:
    python 04_Code/validation/calibrated_cross_validation.py
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).resolve().parent.parent.parent   # Term 3 root
DATA_DIR = ROOT / "03_Processed_Data"
CODE_DIR = ROOT / "04_Code"
RES_DIR  = CODE_DIR / "results"
RES_DIR.mkdir(exist_ok=True)

FEATURES    = ["FSI", "T_avg_C", "T_stress_norm", "KI", "DCSS", "RBF", "CVI"]
N_EARLY     = 5    # calibration cycles per cell

# ── Train fresh XGBoost model on CALCE (matches cross_dataset_validation.py) ─
print("Loading CALCE data and training XGBoost...")
calce = pd.read_csv(DATA_DIR / "Linked_Lab_Fleet_Degradation.csv")
lab   = calce[calce["Source"] == "Lab"].copy()
lab["SoH_%"] = lab["SoH_%"].clip(upper=100.0)
lab_reg = lab[lab["SoH_%"].notna()].copy()

imputer = SimpleImputer(strategy="median")
imputer.fit(lab[FEATURES].values)

X_train = imputer.transform(lab_reg[FEATURES].values)
y_train = lab_reg["SoH_%"].values

import xgboost as xgb
model = xgb.XGBRegressor(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    random_state=42, verbosity=0,
)
model.fit(X_train, y_train)
print(f"  Trained on {len(X_train):,} CALCE lab cycles")


def health_label(soh: float) -> str:
    return "Healthy" if soh >= 90 else ("Degraded" if soh >= 80 else "End_of_Life")


def asymmetric_tnorm(t_avg_c: pd.Series) -> pd.Series:
    """Arrhenius-motivated T_norm: hot=full penalty, cold=half penalty."""
    return np.where(
        t_avg_c >= 25,
        (t_avg_c - 25) / 25,
        0.5 * (25 - t_avg_c) / 25
    )


def early_cycle_calibration(df: pd.DataFrame, y_pred: np.ndarray,
                             id_col: str = "ID") -> tuple:
    """
    Per-cell intercept calibration using first N_EARLY cycles.
    Returns calibrated predictions and per-cell offsets dict.
    """
    offsets = {}
    tmp = df.copy().reset_index(drop=True)
    tmp["_pred"] = y_pred

    for cell_id, grp in tmp.groupby(id_col, sort=False):
        early = grp.head(N_EARLY)
        if len(early) < 2:
            offsets[cell_id] = 0.0
        else:
            offsets[cell_id] = float((early["SoH_%"] - early["_pred"]).mean())

    y_calib = np.array([
        tmp["_pred"].iloc[i] + offsets.get(tmp[id_col].iloc[i], 0.0)
        for i in range(len(tmp))
    ])
    return np.clip(y_calib, 0, 100), offsets


def evaluate_dataset(name: str, csv_path: Path) -> dict:
    if not csv_path.exists():
        print(f"  SKIP — file not found: {csv_path}")
        return {}

    df = pd.read_csv(csv_path)
    df = df[(df["SoH_%"] > 10) & (df["SoH_%"] <= 100)].copy()
    y_true = df["SoH_%"].values

    # ── Base predictions ───────────────────────────────────────────────────
    X = imputer.transform(df[FEATURES].values)
    y_pred = np.clip(model.predict(X), 0, 100)

    rmse_raw = mean_squared_error(y_true, y_pred) ** 0.5
    mae_raw  = mean_absolute_error(y_true, y_pred)
    r2_raw   = r2_score(y_true, y_pred)

    # ── Early-cycle calibration ────────────────────────────────────────────
    id_col = "ID" if "ID" in df.columns else df.columns[0]
    y_calib, offsets = early_cycle_calibration(df, y_pred, id_col=id_col)

    rmse_cal = mean_squared_error(y_true, y_calib) ** 0.5
    mae_cal  = mean_absolute_error(y_true, y_calib)
    r2_cal   = r2_score(y_true, y_calib)

    # ── Asymmetric T_norm experiment (if T_avg_C available) ───────────────
    asym_result = {}
    if "T_avg_C" in df.columns:
        df2 = df.copy()
        df2["T_stress_norm"] = asymmetric_tnorm(df2["T_avg_C"])
        # Recompute FSI with asymmetric T_norm
        # FSI = 0.30*KI + 0.25*DoD + 0.25*T_norm_asym + 0.20*C_peak_norm
        # We can proxy via: FSI_asym = FSI - 0.25*T_stress_norm_old + 0.25*T_stress_norm_new
        if "T_stress_norm" in df.columns:
            fsi_delta = 0.25 * (df2["T_stress_norm"] - df["T_stress_norm"])
            df2["FSI"] = (df["FSI"] + fsi_delta).clip(lower=0)
        X2 = imputer.transform(df2[FEATURES].values)
        y_asym = np.clip(model.predict(X2), 0, 100)
        y_asym_cal, _ = early_cycle_calibration(df, y_asym, id_col=id_col)

        asym_result = {
            "rmse_uncalib": round(float(mean_squared_error(y_true, y_asym) ** 0.5), 3),
            "r2_uncalib":   round(float(r2_score(y_true, y_asym)), 4),
            "rmse_calib":   round(float(mean_squared_error(y_true, y_asym_cal) ** 0.5), 3),
            "r2_calib":     round(float(r2_score(y_true, y_asym_cal)), 4),
        }

    # ── Classifier accuracy ────────────────────────────────────────────────
    clf_raw  = sum(health_label(p) == health_label(t) for p, t in zip(y_pred, y_true))
    clf_cal  = sum(health_label(p) == health_label(t) for p, t in zip(y_calib, y_true))

    n_cells = df[id_col].nunique() if id_col in df.columns else "?"
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"  n={len(y_true)} cycles, {n_cells} cells, {N_EARLY} calib cycles per cell")
    print(f"  {'Metric':<20} {'Uncalibrated':>14} {'Calibrated':>12} {'Change':>8}")
    print(f"  {'-'*56}")
    print(f"  {'RMSE (%)':<20} {rmse_raw:>13.3f}% {rmse_cal:>11.3f}% {rmse_cal-rmse_raw:>+8.3f}%")
    print(f"  {'MAE (%)':<20} {mae_raw:>13.3f}% {mae_cal:>11.3f}% {mae_cal-mae_raw:>+8.3f}%")
    print(f"  {'R²':<20} {r2_raw:>14.4f} {r2_cal:>12.4f} {r2_cal-r2_raw:>+8.4f}")
    print(f"  {'Clf Acc (%)':<20} {clf_raw/len(y_true)*100:>13.1f}% {clf_cal/len(y_true)*100:>11.1f}%")
    if asym_result:
        print(f"\n  Asymmetric T_norm (Arrhenius-weighted):")
        print(f"  {'RMSE uncalib':<20} {asym_result['rmse_uncalib']:>13.3f}%")
        print(f"  {'RMSE calibrated':<20} {asym_result['rmse_calib']:>13.3f}%")
        print(f"  {'R² calibrated':<20} {asym_result['r2_calib']:>14.4f}")

    return {
        "name": name,
        "n_cycles": int(len(y_true)),
        "n_cells": int(n_cells) if isinstance(n_cells, (int, np.integer)) else n_cells,
        "n_calib_per_cell": N_EARLY,
        "uncalibrated": {
            "rmse": round(float(rmse_raw), 3),
            "mae":  round(float(mae_raw),  3),
            "r2":   round(float(r2_raw),   4),
            "clf_accuracy": round(float(clf_raw / len(y_true) * 100), 2),
        },
        "calibrated": {
            "rmse": round(float(rmse_cal), 3),
            "mae":  round(float(mae_cal),  3),
            "r2":   round(float(r2_cal),   4),
            "clf_accuracy": round(float(clf_cal / len(y_true) * 100), 2),
            "mean_offset_pct": round(float(np.mean(list(offsets.values()))), 3),
            "std_offset_pct":  round(float(np.std(list(offsets.values()))),  3),
        },
        "asymmetric_tnorm": asym_result,
    }


# ── Run ────────────────────────────────────────────────────────────────────
print("\nRunning calibrated cross-dataset validation...")

results = {}

oxford = evaluate_dataset(
    "Oxford NMC BMP Drive-Cycle",
    DATA_DIR / "Oxford_FSI_Features.csv"
)
results["oxford"] = oxford

nasa = evaluate_dataset(
    "NASA LiCoO2 CC Multi-Temperature",
    DATA_DIR / "NASA_FSI_Features.csv"
)
results["nasa"] = nasa

# BLAST validation
blast_csv = DATA_DIR / "BLAST_FSI_Features.csv"
if not blast_csv.exists():
    # BLAST features may be embedded in the main CSV under source=="Fleet"
    blast_df = calce[calce["Source"] == "Fleet"].copy()
    blast_df = blast_df[(blast_df["SoH_%"] > 10) & (blast_df["SoH_%"] <= 100)].copy()
    if len(blast_df) > 0:
        blast_csv_tmp = RES_DIR / "_blast_tmp.csv"
        blast_df.to_csv(blast_csv_tmp, index=False)
        blast = evaluate_dataset("NREL BLAST Simulated Fleet", blast_csv_tmp)
        blast_csv_tmp.unlink(missing_ok=True)
    else:
        blast = {}
        print("\n  BLAST: no fleet data found in main CSV")
else:
    blast = evaluate_dataset("NREL BLAST Simulated Fleet", blast_csv)
results["blast"] = blast

# ── Summary table ──────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("CALIBRATION IMPROVEMENT SUMMARY")
print(f"{'='*70}")
print(f"{'Dataset':<35} {'Raw RMSE':>9} {'Cal RMSE':>9} {'Raw R²':>8} {'Cal R²':>8}")
print(f"{'-'*70}")

for key, r in results.items():
    if not r:
        continue
    u = r.get("uncalibrated", {})
    c = r.get("calibrated", {})
    print(f"  {r['name']:<33} {u.get('rmse',0):>8.3f}% {c.get('rmse',0):>8.3f}%"
          f" {u.get('r2',0):>8.4f} {c.get('r2',0):>8.4f}")

# ── Save ──────────────────────────────────────────────────────────────────
results["calce_reference"] = {
    "name": "CALCE (in-distribution, 5-fold CV)",
    "uncalibrated": {"rmse": 3.73, "mae": 2.14, "r2": 0.9839, "clf_accuracy": 98.48},
    "calibrated": {"rmse": 3.73, "mae": 2.14, "r2": 0.9839, "clf_accuracy": 98.48},
}
results["n_early_cycles_used"] = N_EARLY
results["method"] = (
    "Per-cell intercept calibration: offset = mean(y_true_first_N - y_pred_first_N). "
    "No model parameters changed. Simulates initial BMS capacity measurement."
)

out = RES_DIR / "calibrated_validation_results.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved: {out}")
