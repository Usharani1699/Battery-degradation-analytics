"""
Bootstrap Confidence Intervals for CALCE XGBoost FSI Model
============================================================
Runs 1000 bootstrap resamplings of the CALCE training data to compute
95% CIs for RMSE, MAE, and R² under 5-fold cross-validation.

Also computes Spearman ρ CI for the Severson ordinal validation.

Usage:
    python 04_Code/ml/bootstrap_ci.py
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "03_Processed_Data"
CODE_DIR = ROOT / "04_Code"
RES_DIR  = CODE_DIR / "results"
RES_DIR.mkdir(exist_ok=True)

FEATURES   = ["FSI", "T_avg_C", "T_stress_norm", "KI", "DCSS", "RBF", "CVI"]
N_BOOT     = 1000
SEED       = 42
rng        = np.random.default_rng(SEED)

# ── Load CALCE ─────────────────────────────────────────────────────────────
print("Loading CALCE training data...")
df    = pd.read_csv(DATA_DIR / "Linked_Lab_Fleet_Degradation.csv")
lab   = df[df["Source"] == "Lab"].copy()
lab["SoH_%"] = lab["SoH_%"].clip(upper=100.0)
lab   = lab[lab["SoH_%"].notna()].copy()

imputer = SimpleImputer(strategy="median")
X_all = imputer.fit_transform(lab[FEATURES].values)
y_all = lab["SoH_%"].values
print(f"  {len(y_all):,} CALCE lab regression samples")


def cv_metrics(X, y, seed=0):
    """5-fold CV RMSE / MAE / R² for XGBoost on given data."""
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    preds = np.zeros(len(y))
    for train_idx, val_idx in kf.split(X):
        m = xgb.XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            random_state=seed, verbosity=0
        )
        m.fit(X[train_idx], y[train_idx])
        preds[val_idx] = np.clip(m.predict(X[val_idx]), 0, 100)
    rmse = float(mean_squared_error(y, preds) ** 0.5)
    mae  = float(mean_absolute_error(y, preds))
    r2   = float(r2_score(y, preds))
    return rmse, mae, r2


# ── Bootstrap on CALCE ────────────────────────────────────────────────────
print(f"\nRunning {N_BOOT} bootstrap iterations on CALCE...")
print("  (this takes a few minutes — XGBoost x 5-fold x 1000)")

rmse_boot, mae_boot, r2_boot = [], [], []
n = len(y_all)

for i in range(N_BOOT):
    idx = rng.integers(0, n, size=n)
    X_b = X_all[idx]
    y_b = y_all[idx]
    try:
        rmse, mae, r2 = cv_metrics(X_b, y_b, seed=int(i))
        rmse_boot.append(rmse)
        mae_boot.append(mae)
        r2_boot.append(r2)
    except Exception:
        pass

    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{N_BOOT}  RMSE mean={np.mean(rmse_boot):.2f}%  "
              f"95%CI=[{np.percentile(rmse_boot,2.5):.2f},{np.percentile(rmse_boot,97.5):.2f}]")

rmse_arr = np.array(rmse_boot)
mae_arr  = np.array(mae_boot)
r2_arr   = np.array(r2_boot)


def ci95(arr):
    return {
        "mean":  round(float(np.mean(arr)), 4),
        "std":   round(float(np.std(arr)),  4),
        "ci_lo": round(float(np.percentile(arr,  2.5)), 4),
        "ci_hi": round(float(np.percentile(arr, 97.5)), 4),
        "median":round(float(np.median(arr)), 4),
    }


calce_ci = {
    "rmse_pct": ci95(rmse_arr),
    "mae_pct":  ci95(mae_arr),
    "r2":       ci95(r2_arr),
    "n_bootstrap": len(rmse_boot),
    "n_bootstrap_requested": N_BOOT,
}

print(f"\nCALCE Bootstrap Results ({len(rmse_boot)} valid iterations):")
print(f"  RMSE: {calce_ci['rmse_pct']['mean']:.3f}% "
      f"[{calce_ci['rmse_pct']['ci_lo']:.3f}, {calce_ci['rmse_pct']['ci_hi']:.3f}]")
print(f"  MAE:  {calce_ci['mae_pct']['mean']:.3f}% "
      f"[{calce_ci['mae_pct']['ci_lo']:.3f}, {calce_ci['mae_pct']['ci_hi']:.3f}]")
print(f"  R²:   {calce_ci['r2']['mean']:.4f} "
      f"[{calce_ci['r2']['ci_lo']:.4f}, {calce_ci['r2']['ci_hi']:.4f}]")


# ── Bootstrap Spearman ρ CI — Severson ordinal test ───────────────────────
print("\nBootstrapping Severson Spearman ρ confidence interval...")
sev_path = DATA_DIR / "Severson_FSI_Features.csv"
rho_boot = []

if sev_path.exists():
    sev = pd.read_csv(sev_path)
    # Protocol-level: group by protocol, get mean FSI and cycle_life
    if "cycle_life" in sev.columns and "FSI" in sev.columns:
        proto = sev.groupby("protocol_id" if "protocol_id" in sev.columns
                             else "ID").agg(
            FSI=("FSI", "mean"),
            cycle_life=("cycle_life", "mean") if "cycle_life" in sev.columns else ("SoH_%","count"),
        ).dropna()

        fsi_v   = proto["FSI"].values
        cl_v    = proto["cycle_life"].values
        n_proto = len(proto)
        print(f"  {n_proto} Severson protocols for bootstrap")

        for _ in range(N_BOOT):
            idx = rng.integers(0, n_proto, size=n_proto)
            if len(np.unique(idx)) < 3:
                continue
            rho_val, _ = stats.spearmanr(fsi_v[idx], -cl_v[idx])
            if not np.isnan(rho_val):
                rho_boot.append(rho_val)

severson_ci = {}
if rho_boot:
    rho_arr = np.array(rho_boot)
    severson_ci = {
        "spearman_rho_fsi_neg_cyclelife": ci95(rho_arr),
        "n_bootstrap": len(rho_boot),
    }
    print(f"  Spearman ρ: {severson_ci['spearman_rho_fsi_neg_cyclelife']['mean']:.3f} "
          f"[{severson_ci['spearman_rho_fsi_neg_cyclelife']['ci_lo']:.3f}, "
          f"{severson_ci['spearman_rho_fsi_neg_cyclelife']['ci_hi']:.3f}]")
else:
    # Use known point estimate and Fisher z-transform CI for n=23
    rho_point = 0.807
    n_proto   = 23
    z = np.arctanh(rho_point)
    se = 1 / np.sqrt(n_proto - 3)
    z_lo, z_hi = z - 1.96 * se, z + 1.96 * se
    severson_ci = {
        "spearman_rho_fsi_neg_cyclelife": {
            "mean":   round(rho_point, 4),
            "ci_lo":  round(float(np.tanh(z_lo)), 4),
            "ci_hi":  round(float(np.tanh(z_hi)), 4),
            "method": "Fisher z-transform (n=23 protocols)",
        }
    }
    print(f"  Severson CSV not found — using Fisher z-transform:")
    print(f"  ρ = {rho_point} [{severson_ci['spearman_rho_fsi_neg_cyclelife']['ci_lo']:.3f}, "
          f"{severson_ci['spearman_rho_fsi_neg_cyclelife']['ci_hi']:.3f}]")


# ── Propagate CI to BLAST (analytical) ───────────────────────────────────
# For large-n datasets: CI ≈ RMSE ± 1.96 * RMSE / sqrt(2*(n-1))
# This is the asymptotic CI for RMSE under normality of errors
def rmse_analytical_ci(rmse, n):
    se = rmse / np.sqrt(2 * (n - 1))
    return {"ci_lo": round(rmse - 1.96 * se, 3), "ci_hi": round(rmse + 1.96 * se, 3)}

blast_analytical_ci = {
    "rmse_pct": {"mean": 20.439, **rmse_analytical_ci(20.439, 1932)},
    "oxford_rmse_pct": {"mean": 4.748, **rmse_analytical_ci(4.748, 72)},
    "nasa_rmse_pct": {"mean": 15.295, **rmse_analytical_ci(15.295, 2010)},
    "method": "Asymptotic CI: RMSE ± 1.96 * RMSE / sqrt(2*(n-1))",
}


# ── Save ──────────────────────────────────────────────────────────────────
payload = {
    "calce_bootstrap": calce_ci,
    "severson_bootstrap": severson_ci,
    "cross_dataset_analytical_ci": blast_analytical_ci,
    "interpretation": (
        "95% CI for CALCE from 1000 bootstrap resamplings. "
        "Severson ρ CI from bootstrap or Fisher z-transform. "
        "Cross-dataset CIs from asymptotic normal approximation."
    ),
}

out = RES_DIR / "bootstrap_ci_results.json"
with open(out, "w") as f:
    json.dump(payload, f, indent=2)
print(f"\nBootstrap CI results saved: {out}")
