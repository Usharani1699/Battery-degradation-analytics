"""
External Dataset Extraction — Oxford & NASA PCoE
=================================================
Extracts per-cycle FSI features and SoH from two external datasets
for cross-institution, cross-chemistry validation of the FSI model.

Oxford Battery Degradation Dataset (Birkl & Howey, 2017)
  - 6 NMC pouch cells, 40 degC, Artemis Urban drive cycle (variable current)
  - Profiles: BMP (max power), BMR (mixed random), SPM (power minimisation)
  - Chemistry: NMC — cross-chemistry validation vs CALCE LiCoO2

NASA PCoE Battery Dataset (Saha & Goebel, 2007)
  - 18650 LiCoO2 cells, multiple temperature conditions (4 / 24 / 43 degC)
  - CC discharge: validates FSI in low-KI constant-current regime

Outputs:
  data/Oxford_FSI_Features.csv
  data/NASA_FSI_Features.csv

Usage:
    python src/extract_oxford_nasa.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

warnings.filterwarnings("ignore")

ROOT       = Path(__file__).resolve().parent.parent   # Term 3/
NASA_BASE  = ROOT / "NASA Prognostics (PCoE) Battery Dataset" / "5. Battery Data Set"
OXFORD_DIR = ROOT / "NASA Prognostics (PCoE) Battery Dataset" / "Oxford"
OUT_DIR    = ROOT / "data"
OUT_DIR.mkdir(exist_ok=True)

Q_NOMINAL_NASA = 2.0    # rated capacity (Ah) for NASA 18650
T_REF          = 25.0   # reference temperature (degC)

# ── FSI component functions ────────────────────────────────────────────────

def ki(current: np.ndarray) -> float:
    """Kinetic Intensity = std(|I|) / mean(|I|). 0 for CC, >0 for variable."""
    i = np.abs(current)
    m = i.mean()
    return float(i.std() / m) if m > 1e-6 else 0.0

def dod_frac(q_ah: float, q_nom: float) -> float:
    return min(float(q_ah / q_nom), 1.0)

def t_norm(t_array: np.ndarray) -> float:
    return abs(float(np.mean(t_array)) - T_REF) / T_REF

def c_peak_norm(current: np.ndarray, i_nom: float) -> float:
    return float(np.max(np.abs(current)) / i_nom)

def fsi(ki_v, dod_v, tn_v, cp_v) -> float:
    return 0.30*ki_v + 0.25*dod_v + 0.25*tn_v + 0.20*cp_v

def health_label(soh: float) -> str:
    return "Healthy" if soh >= 90 else ("Degraded" if soh >= 80 else "End_of_Life")


# ═══════════════════════════════════════════════════════════════════════════
# OXFORD EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

def extract_oxford() -> pd.DataFrame:
    """
    6 Oxford cells across 3 drive-cycle profiles.
    capacityData.csv  → Q at each of 13 checkpoints → SoH
    profileData.csv   → I(t), V(t), T(t) for each checkpoint interval → FSI
    Variable current → KI > 0 (key distinction from CALCE CC cycling)
    """
    cells = [("BMP",1),("BMP",2),("BMR",1),("BMR",2),("SPM",1),("SPM",2)]
    rows  = []

    for profile, cid in cells:
        cap_f  = OXFORD_DIR / f"{profile}_cell{cid}_capacityData.csv"
        prof_f = OXFORD_DIR / f"{profile}_cell{cid}_profileData.csv"
        if not cap_f.exists() or not prof_f.exists():
            print(f"  Missing: {cap_f.name}")
            continue

        cap_df  = pd.read_csv(cap_f)   # time_s, profile_time_s, capacity_Ah
        prof_df = pd.read_csv(prof_f)  # time_s, current_A, voltage_V, cell_temperature_C, ...

        q_initial = float(cap_df["capacity_Ah"].iloc[0])
        i_nominal = q_initial   # 1C current (A) = capacity (Ah) for correct c_peak_norm

        t_ckpts = cap_df["time_s"].values
        q_ckpts = cap_df["capacity_Ah"].values

        print(f"  Oxford {profile}_C{cid}: {len(cap_df)} checkpoints | "
              f"Q0={q_initial:.3f} Ah | profile rows={len(prof_df):,}")

        for i in range(1, len(t_ckpts)):
            t0, t1 = t_ckpts[i-1], t_ckpts[i]
            q_this = q_ckpts[i]

            seg = prof_df[(prof_df["time_s"] >= t0) & (prof_df["time_s"] < t1)]
            if len(seg) < 20:
                continue

            curr = seg["current_A"].values
            temp = seg["cell_temperature_C"].values
            volt = seg["voltage_V"].values

            # Discharge = positive current in this dataset
            disc = curr > 0.1
            if disc.sum() < 5:
                continue

            ki_v  = ki(curr[disc])
            dod_v = dod_frac(q_this, q_initial)
            tn_v  = t_norm(temp)
            cp_v  = c_peak_norm(curr[disc], i_nominal)
            fsi_v = fsi(ki_v, dod_v, tn_v, cp_v)
            soh_v = min(q_this / q_initial * 100.0, 100.0)

            rows.append({
                "Source":        "External",
                "Dataset":       f"Oxford_{profile}",
                "ID":            f"Oxford_{profile}_C{cid}",
                "Cycle":         i,
                "Chemistry":     "NMC",
                "Profile":       profile,
                "FSI":           round(fsi_v,  4),
                "KI":            round(ki_v,   4),
                "DoD_%":         round(dod_v*100, 2),
                "T_avg_C":       round(float(np.mean(temp)), 2),
                "T_stress_norm": round(tn_v,   4),
                "C_peak_norm":   round(cp_v,   4),
                "DCSS":          round(float(np.std(curr[disc])), 4),
                "RBF":           round(float(np.mean(np.abs(curr[disc]))), 4),
                "CVI":           round(float(np.std(volt)), 4),
                "SoH_%":         round(soh_v,  2),
                "Q_Ah":          round(q_this, 4),
                "Health_Label":  health_label(soh_v),
            })

    df = pd.DataFrame(rows)
    print(f"\nOxford total: {len(df)} rows | {df['ID'].nunique()} cells")
    print(f"  SoH:   {df['SoH_%'].min():.1f} - {df['SoH_%'].max():.1f}%")
    print(f"  KI:    {df['KI'].min():.3f} - {df['KI'].max():.3f}  (non-zero = fleet-like)")
    print(f"  FSI:   {df['FSI'].min():.3f} - {df['FSI'].max():.3f}")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# NASA EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

NASA_SETS = {
    "1. BatteryAgingARC-FY08Q4":           ["B0005","B0006","B0007","B0018"],
    "2. BatteryAgingARC_25_26_27_28_P1":   ["B0025","B0026","B0027","B0028"],
    "3. BatteryAgingARC_25-44":            ["B0029","B0030","B0031","B0032",
                                             "B0033","B0034","B0035","B0036"],
    "4. BatteryAgingARC_45_46_47_48":      ["B0045","B0046","B0047","B0048"],
    "5. BatteryAgingARC_49_50_51_52":      ["B0049","B0050","B0051","B0052"],
    "6. BatteryAgingARC_53_54_55_56":      ["B0053","B0054","B0055","B0056"],
}


def extract_nasa_cell(mat_path: Path, batt_id: str) -> list:
    try:
        mat = sio.loadmat(str(mat_path), simplify_cells=True)
    except Exception as e:
        print(f"    Cannot read {mat_path.name}: {e}")
        return []

    if batt_id not in mat:
        return []

    cycles    = mat[batt_id]["cycle"]
    i_nominal = 2.0   # 2A CC discharge current

    # Initial capacity from first discharge
    q_initial = None
    for c in cycles:
        if c["type"] == "discharge" and "Capacity" in c["data"]:
            q_initial = float(np.max(np.atleast_1d(c["data"]["Capacity"])))
            break
    if not q_initial or q_initial < 0.5:
        return []

    rows     = []
    cycle_n  = 0
    for c in cycles:
        if c["type"] != "discharge" or "Capacity" not in c["data"]:
            continue

        cap_arr  = np.atleast_1d(c["data"]["Capacity"])
        curr_arr = np.atleast_1d(c["data"]["Current_measured"])
        temp_arr = np.atleast_1d(c["data"]["Temperature_measured"])
        volt_arr = np.atleast_1d(c["data"]["Voltage_measured"])
        t_amb    = float(c["ambient_temperature"])
        cycle_n += 1

        if cap_arr.size == 0 or curr_arr.size == 0:
            continue
        q_this = float(np.max(cap_arr))
        ki_v   = ki(curr_arr)
        dod_v  = dod_frac(q_this, Q_NOMINAL_NASA)
        tn_v   = t_norm(np.array([t_amb]))
        cp_v   = c_peak_norm(curr_arr, i_nominal)
        fsi_v  = fsi(ki_v, dod_v, tn_v, cp_v)
        soh_v  = min(q_this / q_initial * 100.0, 100.0)
        t_avg  = float(np.mean(temp_arr))

        rows.append({
            "Source":        "External",
            "Dataset":       f"NASA_{batt_id}",
            "ID":            batt_id,
            "Cycle":         cycle_n,
            "Chemistry":     "LiCoO2",
            "Profile":       "CC",
            "FSI":           round(fsi_v,  4),
            "KI":            round(ki_v,   4),
            "DoD_%":         round(dod_v*100, 2),
            "T_avg_C":       round(t_avg,   2),
            "T_stress_norm": round(tn_v,    4),
            "C_peak_norm":   round(cp_v,    4),
            "DCSS":          round(float(np.std(curr_arr)), 4),
            "RBF":           round(float(np.mean(np.abs(curr_arr))), 4),
            "CVI":           round(float(np.std(volt_arr)), 4),
            "SoH_%":         round(soh_v,   2),
            "Q_Ah":          round(q_this,  4),
            "Health_Label":  health_label(soh_v),
        })
    return rows


def extract_nasa() -> pd.DataFrame:
    all_rows = []
    for subfolder, batteries in NASA_SETS.items():
        folder = NASA_BASE / subfolder
        if not folder.exists():
            print(f"  Folder missing: {subfolder}")
            continue
        for batt_id in batteries:
            mat_path = folder / f"{batt_id}.mat"
            if not mat_path.exists():
                continue
            rows = extract_nasa_cell(mat_path, batt_id)
            if rows:
                print(f"  {batt_id}: {len(rows):3d} cycles | "
                      f"SoH {rows[0]['SoH_%']:.0f}->{rows[-1]['SoH_%']:.0f}% | "
                      f"T={rows[0]['T_avg_C']:.0f}C | KI~{np.mean([r['KI'] for r in rows]):.4f}")
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    print(f"\nNASA total: {len(df)} discharge cycles | {df['ID'].nunique()} batteries")
    print(f"  Temperatures: {sorted(df['T_avg_C'].round(0).unique())}")
    print(f"  SoH range:    {df['SoH_%'].min():.1f} - {df['SoH_%'].max():.1f}%")
    print(f"  KI range:     {df['KI'].min():.4f} - {df['KI'].max():.4f}  (CC = near-zero)")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 62)
    print("OXFORD — NMC, Variable-current, 40°C (fleet-like drive cycles)")
    print("=" * 62)
    oxford_df = extract_oxford()

    print("\n" + "=" * 62)
    print("NASA PCoE — LiCoO2, CC cycling, multi-temperature")
    print("=" * 62)
    nasa_df = extract_nasa()

    oxford_df.to_csv(OUT_DIR / "Oxford_FSI_Features.csv", index=False)
    nasa_df.to_csv(OUT_DIR  / "NASA_FSI_Features.csv",    index=False)

    print(f"\nSaved:  data/Oxford_FSI_Features.csv  ({len(oxford_df)} rows)")
    print(f"Saved:  data/NASA_FSI_Features.csv    ({len(nasa_df)} rows)")

    # Summary comparison
    combined = pd.concat([oxford_df, nasa_df], ignore_index=True)
    print("\n=== CROSS-DATASET FSI SUMMARY ===")
    print(combined.groupby("Dataset")[["SoH_%","FSI","KI","T_avg_C"]].mean().round(3).to_string())
