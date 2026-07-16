"""
Cross-Dataset Validation of the FSI-Only Model
================================================
Trains the XGBoost FSI model on CALCE lab data only,
then evaluates on two unseen external datasets:
  - Oxford NMC (Birkl & Howey 2017) — different chemistry + variable current
  - NASA PCoE LiCoO2 (Saha & Goebel 2007) — same chemistry, different lab + multi-temperature

This is the key generalizability test:
  CALCE (LiCoO2, CC, 0-45C) -> Oxford (NMC, drive cycle, 40C)
  CALCE (LiCoO2, CC, 0-45C) -> NASA   (LiCoO2, CC, 4-55C)

Usage:
    python src/cross_dataset_validation.py
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT / "data"
RES_DIR   = ROOT / "results"
RES_DIR.mkdir(exist_ok=True)

FEATURES     = ["FSI", "T_avg_C", "T_stress_norm", "KI", "DCSS", "RBF", "CVI"]
LABEL_ORDER  = ["Healthy", "Degraded", "End_of_Life"]

def health_label(soh: float) -> str:
    return "Healthy" if soh >= 90 else ("Degraded" if soh >= 80 else "End_of_Life")


# ── Load CALCE training data ──────────────────────────────────────────────
print("Loading CALCE training data...")
calce = pd.read_csv(DATA_DIR / "Linked_Lab_Fleet_Degradation.csv")
lab   = calce[calce["Source"] == "Lab"].copy()
lab["SoH_%"] = lab["SoH_%"].clip(upper=100.0)
lab_reg = lab[lab["SoH_%"].notna()].copy()

le = LabelEncoder()
le.fit(LABEL_ORDER)

imputer = SimpleImputer(strategy="median")
imputer.fit(lab[FEATURES].values)

X_train = imputer.transform(lab_reg[FEATURES].values)
y_train = lab_reg["SoH_%"].values

print(f"  CALCE training samples: {len(X_train):,}")
print(f"  SoH range (train):      {y_train.min():.1f} - {y_train.max():.1f}%")


# ── Train XGBoost regressor on CALCE ─────────────────────────────────────
print("\nTraining XGBoost regressor on CALCE (LiCoO2, CC cycling)...")
model = xgb.XGBRegressor(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    random_state=42, verbosity=0,
)
model.fit(X_train, y_train)


# ── Evaluate on one external dataset ─────────────────────────────────────
def evaluate(name: str, csv_path: Path) -> dict:
    if not csv_path.exists():
        print(f"\n  {name}: file not found at {csv_path}")
        return {}

    ext_df = pd.read_csv(csv_path)
    # Drop rows with very anomalous SoH (calibration artifacts in some NASA cells)
    ext_df = ext_df[(ext_df["SoH_%"] > 10) & (ext_df["SoH_%"] <= 100)].copy()
    ext_df["Health_Actual"] = ext_df["SoH_%"].apply(health_label)

    X_ext = imputer.transform(ext_df[FEATURES].values)  # same imputer as train
    y_true = ext_df["SoH_%"].values

    y_pred = np.clip(model.predict(X_ext), 0, 100)
    rmse   = mean_squared_error(y_true, y_pred) ** 0.5
    mae    = mean_absolute_error(y_true, y_pred)
    r2     = r2_score(y_true, y_pred)

    # Classification from predicted SoH
    y_pred_label  = [health_label(v) for v in y_pred]
    y_true_label  = ext_df["Health_Actual"].tolist()
    correct       = sum(p == t for p, t in zip(y_pred_label, y_true_label))
    clf_acc       = correct / len(y_true) * 100

    # Per-dataset breakdown
    per_ds = ext_df.copy()
    per_ds["SoH_pred"] = y_pred
    per_ds["error"]    = np.abs(y_pred - y_true)
    summary = per_ds.groupby("Dataset").agg(
        n=("SoH_%", "count"),
        SoH_actual_mean=("SoH_%", "mean"),
        SoH_pred_mean=("SoH_pred", "mean"),
        MAE=("error", "mean"),
        R2_group=("SoH_%", lambda x: r2_score(x, per_ds.loc[x.index, "SoH_pred"]) if len(x) > 1 else 0),
    ).round(2)

    print(f"\n=== {name} ===")
    print(f"  Chemistry : {ext_df['Chemistry'].iloc[0]}  |  "
          f"Profile: {ext_df['Profile'].iloc[0]}")
    print(f"  Samples   : {len(y_true)} cycles / {ext_df['ID'].nunique()} cells")
    print(f"  RMSE      : {rmse:.2f}%")
    print(f"  MAE       : {mae:.2f}%")
    print(f"  R2        : {r2:.4f}")
    print(f"  Clf Acc   : {clf_acc:.1f}%  (Healthy/Degraded/EOL from predicted SoH)")
    print(f"\n  Per-dataset breakdown:")
    print(summary.to_string())

    return {
        "name": name,
        "n_cycles": int(len(y_true)),
        "n_cells": int(ext_df["ID"].nunique()),
        "chemistry": ext_df["Chemistry"].iloc[0],
        "profile": ext_df["Profile"].iloc[0],
        "rmse": round(rmse, 3),
        "mae":  round(mae, 3),
        "r2":   round(r2, 4),
        "clf_accuracy": round(clf_acc, 2),
    }


# ── Run evaluations ───────────────────────────────────────────────────────
oxford_result = evaluate("Oxford NMC (variable-current drive cycles)",
                         DATA_DIR / "Oxford_FSI_Features.csv")

nasa_result   = evaluate("NASA PCoE LiCoO2 (CC, multi-temperature)",
                         DATA_DIR / "NASA_FSI_Features.csv")

# CALCE in-distribution performance (5-fold CV reference)
calce_ref = {
    "name": "CALCE (in-distribution, 5-fold CV)",
    "n_cycles": int(len(X_train)),
    "chemistry": "LiCoO2",
    "profile": "CC",
    "rmse": 3.73,   # from ml_fsi_model.py
    "mae":  2.14,
    "r2":   0.9839,
    "clf_accuracy": 98.48,
}


# ── Print comparison table ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print("CROSS-DATASET GENERALIZATION SUMMARY")
print("Model trained on CALCE (LiCoO2, CC) only")
print("=" * 70)
print(f"\n{'Dataset':<40} {'Chemistry':<12} {'Profile':<14} {'RMSE':>6} {'R2':>7} {'ClfAcc':>8}")
print("-" * 70)
for r in [calce_ref, oxford_result, nasa_result]:
    if r:
        print(f"  {r['name']:<38} {r['chemistry']:<12} {r['profile']:<14} "
              f"{r['rmse']:>5.2f}%  {r['r2']:>6.4f}  {r['clf_accuracy']:>6.1f}%")

print("\nKey insight:")
print("  - Oxford (NMC, drive cycle) tests cross-CHEMISTRY and cross-PROFILE")
print("    generalization simultaneously — the strongest validation possible.")
print("  - NASA (LiCoO2, multi-temp) tests cross-INSTITUTION and cross-TEMPERATURE")
print("    generalization — different lab, same chemistry.")
print("  - Performance degradation from CALCE baseline quantifies the")
print("    domain-shift cost, which is expected and scientifically honest.")


# ── Save ─────────────────────────────────────────────────────────────────
results = {
    "training_dataset": "CALCE",
    "calce_reference": calce_ref,
    "oxford_validation": oxford_result,
    "nasa_validation": nasa_result,
}
out = RES_DIR / "cross_dataset_validation.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to: {out}")
