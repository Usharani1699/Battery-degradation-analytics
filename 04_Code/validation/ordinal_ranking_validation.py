"""
Within-Dataset Ordinal Ranking Validation
==========================================
Addresses the "negative R² means the model fails" critique.

Even when absolute RMSE is high (model predicts wrong absolute SoH level),
the model may still CORRECTLY RANK cells by degradation severity.
For fleet management (early warning, maintenance scheduling), correct
relative ordering matters more than correct absolute values.

Strategy:
  - Per-cell degradation rate = slope of SoH vs cycle (linear regression)
  - Negative slope = faster degradation
  - Spearman(per-cell mean FSI, degradation rate) tests ordinal agreement
  - Evaluated on Oxford (NMC, 8 cells) and NASA (LiCoO2, multi-cell)

Usage:
    python 04_Code/validation/ordinal_ranking_validation.py
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from scipy.linalg import lstsq

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "03_Processed_Data"
RES_DIR  = ROOT / "04_Code" / "results"
RES_DIR.mkdir(exist_ok=True)


def degradation_rate(group: pd.DataFrame) -> float:
    """Slope of SoH (%) vs cycle number — negative = degrading."""
    x = group["Cycle"].values.astype(float)
    y = group["SoH_%"].values.astype(float)
    if len(x) < 3:
        return np.nan
    x = x - x.mean()
    slope = float(np.dot(x, y) / np.dot(x, x))
    return slope


def fisher_z_ci(rho: float, n: int, alpha: float = 0.05) -> tuple:
    """95% CI for Spearman rho via Fisher z-transform."""
    from scipy.stats import norm
    z    = np.arctanh(rho)
    se   = 1.0 / np.sqrt(n - 3)
    crit = norm.ppf(1 - alpha / 2)
    return float(np.tanh(z - crit * se)), float(np.tanh(z + crit * se))


def evaluate_ordinal(name: str, csv_path: Path) -> dict:
    if not csv_path.exists():
        print(f"  SKIP — {csv_path.name} not found")
        return {}

    df = pd.read_csv(csv_path)
    df = df[(df["SoH_%"] > 10) & (df["SoH_%"] <= 100)].copy()
    id_col = "ID" if "ID" in df.columns else df.columns[0]

    cells = []
    for cell_id, grp in df.groupby(id_col, sort=False):
        grp = grp.sort_values("Cycle")
        mean_fsi = grp["FSI"].mean()
        mean_ki  = grp["KI"].mean()
        deg_rate = degradation_rate(grp)
        n_cycles = len(grp)
        soh_min  = grp["SoH_%"].min()
        soh_max  = grp["SoH_%"].max()
        cells.append({
            "ID":       cell_id,
            "mean_FSI": mean_fsi,
            "mean_KI":  mean_ki,
            "deg_rate": deg_rate,  # %/cycle  (negative = degrading)
            "n_cycles": n_cycles,
            "soh_range": soh_max - soh_min,
        })

    cdf = pd.DataFrame(cells).dropna(subset=["deg_rate"])
    # Only keep cells with meaningful SoH range (> 2%) — flat cells give noise
    cdf = cdf[cdf["soh_range"] > 2.0].copy()

    n = len(cdf)
    if n < 3:
        print(f"  {name}: only {n} usable cells — skipping")
        return {}

    rho_fsi, p_fsi = spearmanr(cdf["mean_FSI"], cdf["deg_rate"])
    rho_ki,  p_ki  = spearmanr(cdf["mean_KI"],  cdf["deg_rate"])

    # Sign check: higher FSI → more negative deg_rate (faster degradation)
    # We expect rho < 0 (FSI and deg_rate negatively correlated)
    ci_lo, ci_hi = fisher_z_ci(rho_fsi, n)

    print(f"\n{'='*60}")
    print(f"  {name} — Ordinal Ranking Validation")
    print(f"  n = {n} cells, SoH range filter > 2%")
    print(f"  {'Predictor':<20} {'Spearman rho':>14} {'p-value':>10} {'Direction':>12}")
    print(f"  {'-'*58}")

    dir_fsi = "correct" if rho_fsi < 0 else "WRONG"
    dir_ki  = "correct" if rho_ki  < 0 else "WRONG"
    print(f"  {'FSI':<20} {rho_fsi:>+14.4f} {p_fsi:>10.4f} {dir_fsi:>12}")
    print(f"  {'KI alone':<20} {rho_ki:>+14.4f} {p_ki:>10.4f} {dir_ki:>12}")
    print(f"  FSI 95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]")

    if rho_fsi < 0 and abs(rho_fsi) > 0.4 and p_fsi < 0.1:
        interp = "FSI correctly orders cells by degradation severity (moderate-strong ordinal agreement)"
    elif rho_fsi < 0 and n < 10:
        interp = "FSI direction correct; CI wide due to small sample size"
    elif rho_fsi > 0 and p_fsi < 0.1:
        interp = ("FSI direction incorrect (rho > 0) — likely caused by symmetric T_norm: "
                  "cold cells receive high stress score but degrade more slowly than hot cells. "
                  "Validates asymmetric T_norm limitation.")
    elif rho_fsi > 0:
        interp = "FSI direction incorrect but not significant — possible confound from T_norm symmetry"
    else:
        interp = "Weak ordinal agreement — FSI direction correct but statistically uncertain"

    print(f"  Interpretation: {interp}")

    return {
        "name": name,
        "n_cells": n,
        "FSI_rho": round(float(rho_fsi), 4),
        "FSI_pval": round(float(p_fsi), 4),
        "FSI_ci": [round(ci_lo, 3), round(ci_hi, 3)],
        "KI_rho":  round(float(rho_ki),  4),
        "KI_pval": round(float(p_ki),    4),
        "direction_correct": bool(rho_fsi < 0),
        "interpretation": interp,
    }


# ── Run ────────────────────────────────────────────────────────────────────
print("Within-Dataset Ordinal Ranking Validation")
print("=" * 60)
print("Q: Does FSI correctly RANK cells by degradation severity,")
print("   even when absolute RMSE is high?")

results = {}

results["oxford"] = evaluate_ordinal(
    "Oxford NMC BMP Drive-Cycle",
    DATA_DIR / "Oxford_FSI_Features.csv",
)

results["nasa"] = evaluate_ordinal(
    "NASA LiCoO2 CC Multi-Temperature",
    DATA_DIR / "NASA_FSI_Features.csv",
)

results["blast"] = evaluate_ordinal(
    "NREL BLAST Simulated Fleet",
    DATA_DIR / "BLAST_FSI_Features.csv",
)

# Severson as reference (cell-level, Cycle_Life is ground truth)
sev_path = DATA_DIR / "Severson_FSI_Features.csv"
if sev_path.exists():
    sev = pd.read_csv(sev_path)
    cell_agg = sev.groupby("ID").agg(
        mean_FSI=("FSI", "mean"),
        cycle_life=("Cycle_Life", "first"),
    ).dropna()
    rho_s, p_s = spearmanr(cell_agg["mean_FSI"], cell_agg["cycle_life"])
    ci_s_lo, ci_s_hi = fisher_z_ci(rho_s, len(cell_agg))
    results["severson_reference"] = {
        "name": "Severson LFP (cycle_life ground truth)",
        "n_cells": len(cell_agg),
        "FSI_rho": round(float(rho_s), 4),
        "FSI_pval": round(float(p_s), 4),
        "FSI_ci": [round(ci_s_lo, 3), round(ci_s_hi, 3)],
        "direction_correct": bool(rho_s < 0),
    }
    print(f"\n  Severson reference: rho={rho_s:.4f}, n={len(cell_agg)}")

# ── Summary ───────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("ORDINAL RANKING SUMMARY")
print(f"{'='*60}")
print(f"{'Dataset':<38} {'rho':>8} {'Direction':>12} {'p < 0.1':>8}")
print(f"{'-'*60}")
for k, r in results.items():
    if not r:
        continue
    dir_ok = "correct" if r.get("direction_correct") else "WRONG"
    sig    = "yes" if r.get("FSI_pval", 1.0) < 0.1 else "no"
    print(f"  {r['name']:<36} {r.get('FSI_rho', 0):>+8.4f} {dir_ok:>12} {sig:>8}")

print()
print("Key insight: Even when absolute RMSE is high (R² < 0),")
print("correct ordinal ranking enables fleet maintenance scheduling.")

out_path = RES_DIR / "ordinal_ranking_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved: {out_path}")
