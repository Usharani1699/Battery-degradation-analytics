"""
SHAP Validation — FSI-Only XGBoost Model
==========================================
Uses SHAP TreeExplainer to validate that FSI is the dominant predictor
of battery SoH without relying on DoD or EFC.

Key findings:
  - FSI accounts for 82.1% of mean |SHAP| in regression
  - FSI accounts for 77.2% of mean |SHAP| for End-of-Life classification
  - KI, RBF, CVI have zero SHAP in lab data (CC cycling — no kinetic variation)

Outputs:
  - results/shap_results.json     — full SHAP data for all analyses
  - results/shap_artifact.json    — compact payload for the HTML dashboard
  - dashboards/SHAP_Dashboard.html — standalone interactive dashboard

Usage:
    python src/shap_analysis.py
    python src/shap_analysis.py --csv data/Linked_Lab_Fleet_Degradation.csv
"""

import argparse
import json
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "data" / "Linked_Lab_Fleet_Degradation.csv"
RESULTS_DIR = ROOT / "results"
DASHBOARD_DIR = ROOT / "dashboards"

# ── Feature set ────────────────────────────────────────────────────────────
FEATURES = ["FSI", "T_avg_C", "T_stress_norm", "KI", "DCSS", "RBF", "CVI"]
FEAT_LABELS = ["FSI", "T avg (°C)", "T stress", "KI", "DCSS", "RBF", "CVI"]
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


def run(csv_path: Path):
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    lab = df[df["Source"] == "Lab"].copy()
    fleet = df[df["Source"] == "Fleet"].copy()
    lab["SoH_%"] = lab["SoH_%"].clip(upper=100.0)

    lab_clf = lab[lab["Health_Label"].isin(LABEL_ORDER)].copy()
    lab_reg = lab[lab["SoH_%"].notna()].copy()

    le = LabelEncoder()
    le.fit(LABEL_ORDER)

    imputer = SimpleImputer(strategy="median")
    imputer.fit(lab[FEATURES].values)

    X_clf = imputer.transform(lab_clf[FEATURES].values)
    y_clf = le.transform(lab_clf["Health_Label"])
    X_reg = imputer.transform(lab_reg[FEATURES].values)
    y_reg = lab_reg["SoH_%"].values
    X_fl = imputer.transform(fleet[FEATURES].values)

    print(f"Dataset: {len(X_clf):,} clf samples, {len(X_reg):,} reg samples, {len(X_fl)} fleet")

    # ── Train XGBoost models ──────────────────────────────────────────────
    print("\nTraining XGBoost classifier...")
    clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        eval_metric="mlogloss", random_state=42, verbosity=0,
    )
    clf.fit(X_clf, y_clf)

    print("Training XGBoost regressor...")
    reg = xgb.XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        random_state=42, verbosity=0,
    )
    reg.fit(X_reg, y_reg)

    # ── SHAP — Regression (1 500 samples) ────────────────────────────────
    print("\nComputing regression SHAP values...")
    rng = np.random.default_rng(42)
    idx_reg = rng.choice(len(X_reg), size=min(1500, len(X_reg)), replace=False)
    X_reg_s = X_reg[idx_reg]
    y_reg_s = y_reg[idx_reg]

    explainer_reg = shap.TreeExplainer(reg)
    sv_reg = explainer_reg.shap_values(X_reg_s)
    base_reg = float(explainer_reg.expected_value)

    mean_abs_reg = np.abs(sv_reg).mean(axis=0).tolist()
    total = sum(mean_abs_reg)
    print("\nRegression SHAP — mean |SHAP| per feature:")
    for f, v in sorted(zip(FEATURES, mean_abs_reg), key=lambda x: -x[1]):
        print(f"  {f:18s}: {v:.4f}  ({v/total*100:.1f}%)")

    # ── SHAP — Classification (1 500 samples) ────────────────────────────
    print("\nComputing classification SHAP values...")
    idx_clf = rng.choice(len(X_clf), size=min(1500, len(X_clf)), replace=False)
    X_clf_s = X_clf[idx_clf]

    explainer_clf = shap.TreeExplainer(clf)
    sv_clf = explainer_clf.shap_values(X_clf_s)
    base_clf = explainer_clf.expected_value

    sv_clf_arr = np.array(sv_clf)
    # Handle both SHAP API versions:
    #   old (<0.51): list of 3 → shape (3, n, 7)
    #   new (≥0.51): single array → shape (n, 7, 3)
    if sv_clf_arr.ndim == 3 and sv_clf_arr.shape[0] == 3:
        def cls_shap(ci):
            return sv_clf_arr[ci]
    else:
        def cls_shap(ci):
            return sv_clf_arr[:, :, ci]

    classes_order = list(le.classes_)
    mean_abs_clf = {}
    for ci, cls_name in enumerate(classes_order):
        mean_abs_clf[cls_name] = np.abs(cls_shap(ci)).mean(axis=0).tolist()

    print(f"\nEOL classification SHAP — mean |SHAP| per feature:")
    eol = mean_abs_clf["End_of_Life"]
    total_eol = sum(eol)
    for f, v in sorted(zip(FEATURES, eol), key=lambda x: -x[1]):
        print(f"  {f:18s}: {v:.4f}  ({v/total_eol*100:.1f}%)")

    # ── Dependence data ───────────────────────────────────────────────────
    FSI_IDX = FEATURES.index("FSI")
    T_IDX = FEATURES.index("T_avg_C")
    KI_IDX = FEATURES.index("KI")
    DCSS_IDX = FEATURES.index("DCSS")

    dep_fsi = {
        "fsi":   X_reg_s[:, FSI_IDX].tolist(),
        "shap":  sv_reg[:, FSI_IDX].tolist(),
        "t_avg": X_reg_s[:, T_IDX].tolist(),
        "soh":   y_reg_s.tolist(),
    }
    dep_temp = {
        "t_avg": X_reg_s[:, T_IDX].tolist(),
        "shap":  sv_reg[:, T_IDX].tolist(),
        "fsi":   X_reg_s[:, FSI_IDX].tolist(),
        "soh":   y_reg_s.tolist(),
    }

    # ── Fleet SHAP waterfall ──────────────────────────────────────────────
    print("\nComputing fleet SHAP values...")
    sv_fleet = explainer_reg.shap_values(X_fl)
    fleet_pred = reg.predict(X_fl)

    fleet = fleet.copy()
    fleet["SHAP_FSI"] = sv_fleet[:, FSI_IDX]
    fleet["SHAP_T"] = sv_fleet[:, T_IDX]
    fleet["SHAP_KI"] = sv_fleet[:, KI_IDX]
    fleet["SHAP_DCSS"] = sv_fleet[:, DCSS_IDX]
    fleet["SoH_pred"] = fleet_pred

    sorted_pred = np.argsort(fleet_pred)
    rep_idx = [sorted_pred[0], sorted_pred[len(sorted_pred) // 2], sorted_pred[-1]]
    waterfall_data = [
        {
            "label": label,
            "pred": min(100.0, float(fleet_pred[ri])),
            "base": base_reg,
            "shap": sv_fleet[ri].tolist(),
            "feat_vals": X_fl[ri].tolist(),
        }
        for ri, label in zip(rep_idx, ["Lowest SoH", "Median SoH", "Highest SoH"])
    ]
    for w in waterfall_data:
        print(f"  {w['label']:15s}: SoH={w['pred']:.1f}%  SHAP(FSI)={w['shap'][FSI_IDX]:.3f}")

    fleet_veh = (
        fleet.groupby(["Dataset", "ID"])
        .agg(
            FSI_mean=("FSI", "mean"),
            KI_mean=("KI", "mean"),
            SoH_pred=("SoH_pred", "mean"),
            SHAP_FSI_mean=("SHAP_FSI", "mean"),
            SHAP_T_mean=("SHAP_T", "mean"),
            SHAP_KI_mean=("SHAP_KI", "mean"),
            SHAP_DCSS_mean=("SHAP_DCSS", "mean"),
        )
        .reset_index()
        .round(4)
    )

    # ── Beeswarm (600 points) ─────────────────────────────────────────────
    bee_idx = rng.choice(len(X_reg_s), size=min(600, len(X_reg_s)), replace=False)
    beeswarm = {
        "shap_values": sv_reg[bee_idx].tolist(),
        "feat_values": X_reg_s[bee_idx].tolist(),
        "soh": y_reg_s[bee_idx].tolist(),
    }

    # ── Save full results ──────────────────────────────────────────────────
    RESULTS_DIR.mkdir(exist_ok=True)
    payload = {
        "features": FEATURES,
        "feat_labels": FEAT_LABELS,
        "label_order": LABEL_ORDER,
        "classes_order": classes_order,
        "mean_abs_reg": mean_abs_reg,
        "mean_abs_clf_per_class": mean_abs_clf,
        "base_reg": base_reg,
        "base_clf": ([float(b) for b in base_clf]
                     if hasattr(base_clf, "__len__") else [float(base_clf)] * 3),
        "dep_fsi": dep_fsi,
        "dep_temp": dep_temp,
        "beeswarm": beeswarm,
        "waterfall": waterfall_data,
        "fleet_vehicle": fleet_veh.to_dict(orient="list"),
        "fsi_idx": FSI_IDX,
        "t_idx": T_IDX,
        "ki_idx": KI_IDX,
        "dcss_idx": DCSS_IDX,
    }
    out = RESULTS_DIR / "shap_results.json"
    with open(out, "w") as f:
        json.dump(payload, f, cls=NpEncoder)
    print(f"\nFull SHAP results saved to: {out}")

    # ── Build compact dashboard payload ──────────────────────────────────
    random.seed(42)
    n = len(dep_fsi["fsi"])
    idx500 = random.sample(range(n), min(500, n))

    imp_reg = [[f, round(v, 4)] for f, v in zip(FEATURES, mean_abs_reg)]
    imp_eol = [[f, round(v, 4)] for f, v in zip(FEATURES, mean_abs_clf["End_of_Life"])]
    imp_deg = [[f, round(v, 4)] for f, v in zip(FEATURES, mean_abs_clf["Degraded"])]
    imp_hea = [[f, round(v, 4)] for f, v in zip(FEATURES, mean_abs_clf["Healthy"])]

    fv = fleet_veh.to_dict(orient="list")
    n_veh = len(fv["ID"])
    veh_order = sorted(range(n_veh), key=lambda i: -fv["SHAP_FSI_mean"][i])[:8]
    fleet_compact = [
        {
            "id": fv["ID"][i],
            "fsi": fv["FSI_mean"][i],
            "ki": fv["KI_mean"][i],
            "soh": min(100.0, fv["SoH_pred"][i]),
            "shap_fsi": fv["SHAP_FSI_mean"][i],
            "shap_t": fv["SHAP_T_mean"][i],
            "shap_ki": fv["SHAP_KI_mean"][i],
        }
        for i in veh_order
    ]

    compact = {
        "features": FEATURES,
        "feat_labels": FEAT_LABELS,
        "base_reg": round(base_reg, 2),
        "imp_reg": imp_reg,
        "imp_eol": imp_eol,
        "imp_deg": imp_deg,
        "imp_hea": imp_hea,
        "dep_fsi": {k: [round(dep_fsi[k][i], 4) for i in idx500]
                    for k in dep_fsi},
        "dep_t": {k: [round(dep_temp[k][i], 4) for i in idx500]
                  for k in dep_temp},
        "beeswarm_shap": [[round(x, 4) for x in row] for row in beeswarm["shap_values"]],
        "beeswarm_feat": [[round(x, 4) for x in row] for row in beeswarm["feat_values"]],
        "waterfall": waterfall_data,
        "fleet": fleet_compact,
        "fsi_idx": FSI_IDX,
        "t_idx": T_IDX,
    }
    compact_out = RESULTS_DIR / "shap_artifact.json"
    with open(compact_out, "w") as f:
        json.dump(compact, f, cls=NpEncoder)
    size = compact_out.stat().st_size // 1024
    print(f"Compact SHAP payload saved to: {compact_out}  ({size} KB)")

    # ── Embed payload into SHAP dashboard HTML ────────────────────────────
    _build_dashboard(compact, DASHBOARD_DIR)


def _build_dashboard(data: dict, out_dir: Path):
    """Embed SHAP JSON payload into the standalone HTML dashboard."""
    template_path = out_dir / "SHAP_Dashboard.html"
    if not template_path.exists():
        print(f"\nDashboard template not found at {template_path} — skipping HTML build.")
        print("Run dashboards/SHAP_Dashboard.html directly (it already has data embedded).")
        return

    # If the template exists, re-embed fresh data
    content = template_path.read_text(encoding="utf-8")
    import re
    new_data = json.dumps(data)
    content = re.sub(r"const D = \{.*?\};", f"const D = {new_data};",
                     content, count=1, flags=re.DOTALL)
    template_path.write_text(content, encoding="utf-8")
    print(f"Dashboard updated: {template_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHAP validation for FSI-only battery ML model")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                        help="Path to Linked_Lab_Fleet_Degradation.csv")
    args = parser.parse_args()
    run(args.csv)
