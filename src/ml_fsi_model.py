"""
Battery Degradation ML — FSI-Only Model
========================================
Trains Random Forest, XGBoost, and Gradient Boosting models to:
  - classify battery health (Healthy / Degraded / End_of_Life)
  - predict State of Health (SoH %) via regression

Features used: FSI, T_avg_C, T_stress_norm, KI, DCSS, RBF, CVI
DoD_% and EFC are deliberately excluded — they co-move with SoH
in constant-current lab tests and would cause data leakage.

FSI (Fleet Severity Index) = 0.30×KI + 0.25×DoD + 0.25×T_norm + 0.20×C_peak_norm

Usage:
    python src/ml_fsi_model.py
    python src/ml_fsi_model.py --csv data/Linked_Lab_Fleet_Degradation.csv
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "data" / "Linked_Lab_Fleet_Degradation.csv"
RESULTS_DIR = ROOT / "results"

# ── Feature set ────────────────────────────────────────────────────────────
FEATURES = ["FSI", "T_avg_C", "T_stress_norm", "KI", "DCSS", "RBF", "CVI"]
LABEL_ORDER = ["Healthy", "Degraded", "End_of_Life"]


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def load_data(csv_path: Path):
    df = pd.read_csv(csv_path)
    lab = df[df["Source"] == "Lab"].copy()
    fleet = df[df["Source"] == "Fleet"].copy()
    lab["SoH_%"] = lab["SoH_%"].clip(upper=100.0)
    return lab, fleet


def run(csv_path: Path):
    print(f"Loading data from: {csv_path}")
    lab, fleet = load_data(csv_path)

    lab_clf = lab[lab["Health_Label"].isin(LABEL_ORDER)].copy()
    lab_reg = lab[lab["SoH_%"].notna()].copy()

    le = LabelEncoder()
    le.fit(LABEL_ORDER)

    # Imputer fit on lab only — avoids data leakage from fleet
    imputer = SimpleImputer(strategy="median")
    imputer.fit(lab[FEATURES].values)

    X_clf = imputer.transform(lab_clf[FEATURES].values)
    y_clf = le.transform(lab_clf["Health_Label"])
    X_reg = imputer.transform(lab_reg[FEATURES].values)
    y_reg = lab_reg["SoH_%"].values
    X_fl = imputer.transform(fleet[FEATURES].values)

    print(f"\nDataset: {len(X_clf):,} lab cycles (clf), "
          f"{len(X_reg):,} (reg), {len(X_fl)} fleet trips")

    # ── Classification ────────────────────────────────────────────────────
    clf_models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            eval_metric="mlogloss", random_state=42, verbosity=0,
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=300, max_depth=6, learning_rate=0.05,
            random_state=42, class_weight="balanced",
        ),
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf_results = {}

    print("\n=== CLASSIFICATION (5-fold CV) ===")
    for name, clf in clf_models.items():
        y_pred = cross_val_predict(clf, X_clf, y_clf, cv=skf, n_jobs=1)
        acc = accuracy_score(y_clf, y_pred)
        f1m = f1_score(y_clf, y_pred, average="macro")
        f1w = f1_score(y_clf, y_pred, average="weighted")
        cm = confusion_matrix(y_clf, y_pred)
        # BUGFIX: LabelEncoder.fit() always sorts classes ALPHABETICALLY
        # regardless of the order LABEL_ORDER lists them in, so numeric
        # class 0/1/2 do NOT correspond to LABEL_ORDER[0]/[1]/[2]. Passing
        # target_names=LABEL_ORDER here silently mislabelled every per-class
        # precision/recall/F1 (Healthy<->Degraded<->End_of_Life swapped).
        # Use le.classes_ (the encoder's real, alphabetically-sorted order)
        # instead so labels match the actual numeric classes.
        cr = classification_report(y_clf, y_pred, target_names=list(le.classes_), output_dict=True)
        clf_results[name] = {
            "accuracy": round(acc * 100, 2),
            "f1_macro": round(f1m, 4),
            "f1_weighted": round(f1w, 4),
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_label_order": list(le.classes_),
            "per_class": {
                c: {"precision": round(cr[c]["precision"], 3),
                    "recall": round(cr[c]["recall"], 3),
                    "f1": round(cr[c]["f1-score"], 3)}
                for c in le.classes_
            },
        }
        print(f"  {name:20s}: Acc={acc*100:.2f}%  F1-macro={f1m:.4f}")

    # ── Regression ────────────────────────────────────────────────────────
    reg_models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=300, random_state=42, n_jobs=-1
        ),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            random_state=42, verbosity=0,
        ),
        "Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=300, max_depth=6, learning_rate=0.05, random_state=42
        ),
    }
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    reg_results = {}

    print("\n=== REGRESSION — SoH % (5-fold CV) ===")
    for name, reg in reg_models.items():
        y_pred = cross_val_predict(reg, X_reg, y_reg, cv=kf, n_jobs=1)
        y_pred = np.clip(y_pred, 0, 100)
        rmse = mean_squared_error(y_reg, y_pred) ** 0.5
        mae = mean_absolute_error(y_reg, y_pred)
        r2 = r2_score(y_reg, y_pred)
        reg_results[name] = {"rmse": round(rmse, 3), "mae": round(mae, 3), "r2": round(r2, 4)}
        print(f"  {name:20s}: RMSE={rmse:.2f}%  MAE={mae:.2f}%  R²={r2:.4f}")

    # ── Fit final models on full lab data ────────────────────────────────
    rf_clf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf_clf.fit(X_clf, y_clf)

    xgb_reg = xgb.XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42, verbosity=0
    )
    xgb_reg.fit(X_reg, y_reg)

    # Save model for cross-dataset validation (Severson, BLAST, etc.)
    import pickle, os
    mod_dir = ROOT / "models"
    mod_dir.mkdir(exist_ok=True)
    with open(mod_dir / "xgb_model.pkl", "wb") as _f:
        pickle.dump(xgb_reg, _f)
    print(f"Model saved: models/xgb_model.pkl")

    rf_imp = dict(zip(FEATURES, rf_clf.feature_importances_.round(4)))
    xgb_imp = dict(zip(FEATURES, xgb_reg.feature_importances_.round(4)))
    print("\n=== FEATURE IMPORTANCE ===")
    print("  RF  (clf):", {k: v for k, v in sorted(rf_imp.items(), key=lambda x: -x[1])})
    print("  XGB (reg):", {k: v for k, v in sorted(xgb_imp.items(), key=lambda x: -x[1])})

    # ── Fleet predictions ─────────────────────────────────────────────────
    fleet = fleet.copy()
    fleet["Predicted_Health"] = le.inverse_transform(rf_clf.predict(X_fl))
    fleet["Predicted_SoH_%"] = np.clip(xgb_reg.predict(X_fl), 0, 100).round(2)
    proba = rf_clf.predict_proba(X_fl)
    for i, cls in enumerate(le.classes_):
        fleet[f"P_{cls}"] = proba[:, i].round(3)

    per_vehicle = (
        fleet.groupby(["Dataset", "ID"])
        .agg(
            trips=("Predicted_SoH_%", "count"),
            FSI_mean=("FSI", "mean"),
            KI_mean=("KI", "mean"),
            SoH_pred=("Predicted_SoH_%", "mean"),
            P_Healthy=("P_Healthy", "mean"),
            P_Degraded=("P_Degraded", "mean"),
            P_End_of_Life=("P_End_of_Life", "mean"),
        )
        .reset_index()
        .round(3)
    )
    per_vehicle["Health_Pred"] = per_vehicle.apply(
        lambda r: "Healthy" if r.P_Healthy >= 0.5
        else "Degraded" if r.P_Degraded >= r.P_End_of_Life
        else "End_of_Life",
        axis=1,
    )

    print(f"\n=== FLEET PREDICTIONS ({len(per_vehicle)} vehicles) ===")
    print("  SoH range:", round(per_vehicle["SoH_pred"].min(), 1),
          "–", round(per_vehicle["SoH_pred"].max(), 1), "%")
    print("  Health:\n", per_vehicle["Health_Pred"].value_counts().to_string())

    # ── Save results ──────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(exist_ok=True)
    payload = {
        "features": FEATURES,
        "label_order": LABEL_ORDER,
        "clf_results": clf_results,
        "reg_results": reg_results,
        "rf_importance": rf_imp,
        "xgb_importance": xgb_imp,
        "fleet_vehicle": per_vehicle.to_dict(orient="list"),
    }
    out = RESULTS_DIR / "ml_fsi_results.json"
    with open(out, "w") as f:
        json.dump(payload, f, cls=NpEncoder, indent=2)
    print(f"\nResults saved to: {out}")

    fleet_out = RESULTS_DIR / "Fleet_ML_Predictions.csv"
    fleet[["Dataset", "ID", "FSI", "KI", "Predicted_SoH_%", "Predicted_Health",
           "P_Healthy", "P_Degraded", "P_End_of_Life"]].to_csv(fleet_out, index=False)
    print(f"Fleet predictions saved to: {fleet_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FSI-only battery health ML model")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                        help="Path to Linked_Lab_Fleet_Degradation.csv")
    args = parser.parse_args()
    run(args.csv)
