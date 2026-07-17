"""
Severson et al. (2019) LFP Dataset Extraction
===============================================
Extracts per-cycle FSI features and SoH from the Severson battery dataset.

Reference:
  Severson, K.A. et al. (2019). Data-driven prediction of battery cycle life
  before capacity degradation. Nature Energy, 4(5), pp.383-391.
  doi:10.1038/s41560-019-0356-8

Dataset:
  124 commercial LFP/graphite A123 APR18650M1A cells (1.1 Ah nominal)
  - Tested at 30°C in a temperature-controlled environment
  - Different multi-step fast-charging protocols (variable CC steps, 1C to 8C)
  - Discharged at 4C constant current to 2.0V
  - All cells cycled to end of life (80% SoH threshold or failure)

Why this fills the gap:
  - Real LFP chemistry (vs CALCE LiCoO2) — second real chemistry
  - Variable charging current → real KI > 0 (unlike CALCE CC)
  - 124 cells with different protocols → tests whether FSI predicts which
    protocol causes faster degradation (the core fleet stress claim)

Input:
  Severson_data/batch1.pkl   (~50 MB, 41 cells)
  Severson_data/batch2.pkl   (~90 MB, 43 cells, optional)

Output:
  data/Severson_FSI_Features.csv

Download data from:
  https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation

Usage:
  python src/extract_severson.py
"""

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).resolve().parent.parent
SEV_DIR   = ROOT / "Severson_data"
OUT_DIR   = ROOT / "data"
OUT_DIR.mkdir(exist_ok=True)

Q_NOM   = 1.1    # Ah nominal capacity for A123 APR18650M1A
T_REF   = 25.0   # reference temperature
T_TEST  = 30.0   # Severson test temperature (controlled chamber)

# ── FSI component functions ───────────────────────────────────────────────

def ki(current: np.ndarray) -> float:
    i = np.abs(current)
    m = i.mean()
    return float(i.std() / m) if m > 1e-6 else 0.0

def dod_frac(q_discharge: float, q_nom: float = Q_NOM) -> float:
    return min(float(q_discharge / q_nom), 1.0)

def t_norm(t_c: float) -> float:
    return abs(t_c - T_REF) / T_REF

def c_peak_norm(current: np.ndarray, i_nom: float) -> float:
    return float(np.max(np.abs(current)) / i_nom)

def fsi(ki_v, dod_v, tn_v, cp_v) -> float:
    return 0.30*ki_v + 0.25*dod_v + 0.25*tn_v + 0.20*cp_v

def health_label(soh: float) -> str:
    return "Healthy" if soh >= 90 else ("Degraded" if soh >= 80 else "End_of_Life")


# ── Load one batch file ───────────────────────────────────────────────────

def load_batch(pkl_path: Path) -> dict:
    with open(pkl_path, "rb") as f:
        try:
            batch = pickle.load(f)
        except UnicodeDecodeError:
            # Some batch files saved with Python 2 need latin-1 encoding
            f.seek(0)
            batch = pickle.load(f, encoding="latin1")
    return batch


# ── Extract FSI features for one cell ────────────────────────────────────

def extract_cell(cell_key: str, cell: dict, batch_num: int) -> list:
    rows = []

    # Summary data (per cycle)
    summary = cell.get("summary", {})
    if not summary:
        return []

    cycle_numbers = np.array(summary.get("cycle", []))
    qd_arr        = np.array(summary.get("QD", []))          # discharge capacity (Ah)
    temp_arr_sum  = np.array(summary.get("Tdlin", []))       # avg temperature
    ir_arr        = np.array(summary.get("IR", []))          # internal resistance

    if len(qd_arr) < 5:
        return []

    # Nominal capacity from first few full cycles (avoid initial conditioning)
    q_initial = float(np.median(qd_arr[2:min(10, len(qd_arr))]))
    if q_initial < 0.5:
        q_initial = Q_NOM

    # Charging policy (determines KI)
    charge_policy = cell.get("charge_policy", "unknown")

    # Cycle-level data for KI and C_peak
    cycles_data = cell.get("cycles", {})

    # I_nom = 1C = Q_nom Ah
    i_nom = Q_NOM   # 1C in amps

    prev_rows_len = 0

    for idx, cyc_num in enumerate(cycle_numbers):
        if idx >= len(qd_arr):
            break
        q_this = float(qd_arr[idx])
        if q_this < 0.3:     # skip clearly bad cycles
            continue

        soh_v = min(q_this / q_initial * 100.0, 100.0)
        if soh_v < 60:
            break   # stop after severe degradation

        # Temperature: use summary avg if available
        t_avg = float(temp_arr_sum[idx]) if idx < len(temp_arr_sum) and temp_arr_sum[idx] > 0 else T_TEST

        # Get per-cycle current profile for KI and C_peak
        cyc_str = str(int(cyc_num))
        cyc_data = cycles_data.get(cyc_str, {})

        if cyc_data and "I" in cyc_data and len(cyc_data["I"]) > 10:
            curr = np.array(cyc_data["I"])
            temp = np.array(cyc_data.get("T", [t_avg] * len(curr)))
            volt = np.array(cyc_data.get("V", []))

            # Charge current (positive in Severson convention)
            charge_mask = curr > 0.05
            if charge_mask.sum() < 5:
                charge_mask = np.abs(curr) > 0.05

            ki_v  = ki(curr[charge_mask]) if charge_mask.sum() > 5 else 0.0
            cp_v  = c_peak_norm(curr, i_nom)
            t_avg = float(np.mean(temp)) if len(temp) > 0 else T_TEST
            tn_v  = t_norm(t_avg)
            dcss  = float(np.std(curr[charge_mask])) if charge_mask.sum() > 5 else 0.0
            rbf   = float(np.mean(np.abs(curr[charge_mask]))) if charge_mask.sum() > 5 else 0.0
            cvi   = float(np.std(volt)) if len(volt) > 5 else 0.0
        else:
            # Fall back to policy-derived approximation
            ki_v  = _ki_from_policy(charge_policy)
            cp_v  = _cpeak_from_policy(charge_policy, i_nom)
            tn_v  = t_norm(T_TEST)
            t_avg = T_TEST
            dcss  = ki_v * Q_NOM   # rough proxy
            rbf   = Q_NOM
            cvi   = 0.1

        dod_v = dod_frac(q_this)
        fsi_v = fsi(ki_v, dod_v, tn_v, cp_v)

        rows.append({
            "Source":        "External",
            "Dataset":       f"Severson_batch{batch_num}",
            "ID":            f"Severson_{cell_key}",
            "Cycle":         int(cyc_num),
            "Chemistry":     "LFP",
            "Profile":       "MultiStep_FastCharge",
            "Charge_Policy": charge_policy,
            "FSI":           round(fsi_v,  4),
            "KI":            round(ki_v,   4),
            "DoD_%":         round(dod_v * 100, 2),
            "T_avg_C":       round(t_avg,  2),
            "T_stress_norm": round(tn_v,   4),
            "C_peak_norm":   round(cp_v,   4),
            "DCSS":          round(dcss,   4),
            "RBF":           round(rbf,    4),
            "CVI":           round(cvi,    4),
            "SoH_%":         round(soh_v,  2),
            "Q_Ah":          round(q_this, 4),
            "Cycle_Life":    int(cell.get("cycle_life", 0)),
            "Health_Label":  health_label(soh_v),
        })

    return rows


def _ki_from_policy(policy: str) -> float:
    """Estimate KI from charge policy string if cycle-level I not available."""
    if not policy or policy == "unknown":
        return 0.1
    # Policies like "3.6C(80%)-4C" have multiple C-rate steps → higher KI
    parts = [p for p in policy.replace("(", "-").replace(")", "").split("-")
             if "C" in p.upper()]
    if len(parts) <= 1:
        return 0.05   # single step ≈ near CC
    c_rates = []
    for p in parts:
        try:
            c_rates.append(float(p.upper().replace("C", "").replace("%", "")))
        except ValueError:
            pass
    if len(c_rates) < 2:
        return 0.05
    c_arr = np.array(c_rates)
    m = c_arr.mean()
    return float(c_arr.std() / m) if m > 0 else 0.1


def _cpeak_from_policy(policy: str, i_nom: float) -> float:
    """Estimate peak C-rate from policy string."""
    if not policy or policy == "unknown":
        return 1.0
    import re
    c_vals = re.findall(r"(\d+\.?\d*)C", policy.upper())
    if not c_vals:
        return 1.0
    return min(float(max(c_vals, key=float)) / i_nom, 5.0)


# ── Main extraction ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 62)
    print("Severson et al. (2019) LFP Dataset — FSI Feature Extraction")
    print("=" * 62)

    batch_files = [
        (SEV_DIR / "batch1.pkl", 1),
        (SEV_DIR / "batch2.pkl", 2),
        (SEV_DIR / "batch3.pkl", 3),
    ]

    all_rows = []

    for pkl_path, batch_num in batch_files:
        if not pkl_path.exists():
            print(f"\n  batch{batch_num}.pkl not found — skipping.")
            continue

        print(f"\nLoading batch{batch_num}.pkl ...")
        batch = load_batch(pkl_path)
        print(f"  Cells in batch: {len(batch)}")

        for cell_key, cell in batch.items():
            rows = extract_cell(cell_key, cell, batch_num)
            if rows:
                cyc_life = rows[-1]["Cycle"] if rows else 0
                ki_mean  = np.mean([r["KI"] for r in rows])
                fsi_mean = np.mean([r["FSI"] for r in rows])
                print(f"  {cell_key:<12} policy={cell.get('charge_policy','?'):<18} "
                      f"cycles={len(rows):>4}  KI_mean={ki_mean:.4f}  "
                      f"FSI_mean={fsi_mean:.4f}  SoH_final={rows[-1]['SoH_%']:.1f}%")
            all_rows.extend(rows)

    if not all_rows:
        print("\nERROR: No data extracted.")
        print("Check that batch1.pkl (or batch2.pkl) exists in Severson_data/")
        raise SystemExit(1)

    df = pd.DataFrame(all_rows)
    print(f"\n{'='*62}")
    print(f"Total extracted: {len(df):,} cycle records from {df['ID'].nunique()} cells")
    print(f"SoH range:  {df['SoH_%'].min():.1f} – {df['SoH_%'].max():.1f}%")
    print(f"KI range:   {df['KI'].min():.4f} – {df['KI'].max():.4f}")
    print(f"FSI range:  {df['FSI'].min():.4f} – {df['FSI'].max():.4f}")
    print(f"Charge policies: {df['Charge_Policy'].nunique()} unique protocols")
    print(f"\nKI by charge policy (key fleet-relevance test):")
    ki_by_policy = df.groupby("Charge_Policy")["KI"].agg(["mean","std","count"])
    print(ki_by_policy.sort_values("mean", ascending=False).to_string())

    out = OUT_DIR / "Severson_FSI_Features.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved: data/Severson_FSI_Features.csv ({len(df):,} rows)")
