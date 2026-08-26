"""
Cross-Dataset / Cross-Chemistry Validation of the FSI-Only Model — CORRECTED
==============================================================================
Fixes two verified issues found in the original src/cross_dataset_validation.py
and 04_Code/validation/cross_dataset_validation.py:

  BUG 1 (leakage): the original script trained on ALL rows where Source=="Lab",
  which is NASA + CALCE + Oxford COMBINED (9,031 rows) — including the exact
  same Oxford cells it then "validated" on as if they were unseen. This
  contradicts the dissertation's stated design (Methodology 3.2/3.6): the
  model must be fit on NASA+CALCE (LCO) ONLY, with Oxford (NMC) held out
  completely and never touched during training/imputer-fitting.

  BUG 2 (mislabeling): the original script hardcoded a hop-in reference dict
  labelled "CALCE (in-distribution, 5-fold CV)" with r2=0.9839, rmse=3.73,
  mae=2.14 — but that number ("from ml_fsi_model.py") was actually computed
  by an UNGROUPED, ROW-LEVEL KFold over the full NASA+CALCE+Oxford combined
  set, not CALCE alone, and its mae doesn't match any of the three models'
  real MAE values in ml_fsi_results.json. This script replaces it with a
  properly battery/cell-grouped (GroupKFold) in-distribution reference,
  fit and evaluated on CALCE only, so cells never leak across folds.

This script is the corrected replacement for the "R2 = -4.35" figure cited
in Appendix A.3 / Discussion 5.5 of main(prism).pdf, which was found to be
unreproducible anywhere in the codebase.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent   # .../Term 3
DATA_DIR = ROOT / "data"
RES_DIR = ROOT / "04_Code" / "results"
RES_DIR.mkdir(exist_ok=True)

FEATURES = ["FSI", "T_avg_C", "T_stress_norm", "KI", "DCSS", "RBF", "CVI"]


def health_label(soh: float) -> str:
    return "Healthy" if soh >= 90 else ("Degraded" if soh >= 80 else "End_of_Life")


print("=" * 78)
print("CORRECTED cross-dataset / cross-chemistry validation")
print("=" * 78)

df = pd.read_csv(DATA_DIR / "Linked_Lab_Fleet_Degradation.csv", encoding="utf-8-sig")
lab = df[df["Source"] == "Lab"].copy()
lab["SoH_%"] = lab["SoH_%"].clip(upper=100.0)
lab = lab[lab["SoH_%"].notna()].copy()

print("\nRows in 'Lab' by Dataset (sanity check):")
print(lab["Dataset"].value_counts().to_string())

# ── Correct split: LCO-only training set (NASA + CALCE), Oxford held out ──
lco_mask = lab["Dataset"].isin(["NASA", "CALCE_CS2"])
train_lco = lab[lco_mask].copy()
oxford_ext = lab[lab["Dataset"] == "Oxford"].copy()

print(f"\nLCO training pool (NASA+CALCE): {len(train_lco):,} cycles")
print(f"Oxford (NMC) held-out pool     : {len(oxford_ext):,} cycles "
      f"(NEVER included in training/imputer fitting)")

# ── 1) In-distribution reference: CALCE-only, battery-GROUPED 5-fold CV ────
calce_only = train_lco[train_lco["Dataset"] == "CALCE_CS2"].copy()
imputer_ref = SimpleImputer(strategy="median")
X_ref_all = imputer_ref.fit_transform(calce_only[FEATURES].values)
y_ref_all = calce_only["SoH_%"].values
groups_ref = calce_only["ID"].values

n_calce_cells = calce_only["ID"].nunique()
n_splits_ref = min(5, n_calce_cells)  # CALCE_CS2 in this master file has only
                                       # a handful of physical cells (checked at
                                       # runtime) -> true leave-one/few-cell(s)-out
print(f"\nNOTE: CALCE_CS2 subset has {n_calce_cells} unique physical cell(s) "
      f"({sorted(calce_only['ID'].unique())}) -> using GroupKFold(n_splits={n_splits_ref})")
gkf = GroupKFold(n_splits=n_splits_ref)
oof_pred = np.zeros_like(y_ref_all, dtype=float)
for tr_idx, te_idx in gkf.split(X_ref_all, y_ref_all, groups_ref):
    m = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                          random_state=42, verbosity=0)
    m.fit(X_ref_all[tr_idx], y_ref_all[tr_idx])
    oof_pred[te_idx] = np.clip(m.predict(X_ref_all[te_idx]), 0, 100)

calce_rmse = mean_squared_error(y_ref_all, oof_pred) ** 0.5
calce_mae = mean_absolute_error(y_ref_all, oof_pred)
calce_r2 = r2_score(y_ref_all, oof_pred)

calce_ref = {
    "name": f"CALCE only, battery-GROUPED {n_splits_ref}-fold CV (GroupKFold by cell ID, {n_calce_cells} cells)",
    "n_cycles": int(len(calce_only)),
    "n_cells": int(calce_only["ID"].nunique()),
    "chemistry": "LiCoO2",
    "profile": "CC",
    "rmse": round(calce_rmse, 3),
    "mae": round(calce_mae, 3),
    "r2": round(calce_r2, 4),
}
print(f"\n=== {calce_ref['name']} ===")
print(f"  n_cycles={calce_ref['n_cycles']}  n_cells={calce_ref['n_cells']}")
print(f"  RMSE={calce_ref['rmse']}  MAE={calce_ref['mae']}  R2={calce_ref['r2']}")

# ── 2) Train the deployed model on NASA+CALCE (LCO) only ──────────────────
imputer = SimpleImputer(strategy="median")
imputer.fit(train_lco[FEATURES].values)
X_train = imputer.transform(train_lco[FEATURES].values)
y_train = train_lco["SoH_%"].values

model = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                          random_state=42, verbosity=0)
model.fit(X_train, y_train)


def evaluate(name, ext_df):
    X_ext = imputer.transform(ext_df[FEATURES].values)
    y_true = ext_df["SoH_%"].values
    y_pred = np.clip(model.predict(X_ext), 0, 100)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    result = {
        "name": name,
        "n_cycles": int(len(y_true)),
        "n_cells": int(ext_df["ID"].nunique()),
        "rmse": round(rmse, 3),
        "mae": round(mae, 3),
        "r2": round(r2, 4),
    }
    print(f"\n=== {name} ===")
    print(f"  n_cycles={result['n_cycles']}  n_cells={result['n_cells']}")
    print(f"  RMSE={result['rmse']}  MAE={result['mae']}  R2={result['r2']}")
    return result


oxford_result = evaluate(
    "Oxford NMC — TRUE held-out cross-chemistry transfer "
    "(model trained ONLY on NASA+CALCE, Oxford never seen)",
    oxford_ext,
)

# ── Save ────────────────────────────────────────────────────────────────
results = {
    "note": "Corrected version: training restricted to NASA+CALCE (LCO) only; "
            "Oxford (NMC) fully held out; in-distribution reference uses "
            "battery-GROUPED GroupKFold (no cell leakage). Supersedes the "
            "'R2 = -4.35' figure in main(prism).pdf Appendix A.3 / Discussion 5.5, "
            "which was not reproducible from any script in this repository.",
    "training_dataset": "NASA + CALCE (LCO only)",
    "calce_in_distribution_grouped": calce_ref,
    "oxford_true_holdout": oxford_result,
}
out = RES_DIR / "cross_dataset_validation_CORRECTED.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out}")
