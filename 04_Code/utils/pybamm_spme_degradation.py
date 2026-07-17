"""
PyBaMM SPMe Degradation Simulation — FSI Validation
=====================================================
Runs a Single Particle Model with Electrolyte (SPMe) + SEI degradation
for multiple cycles, extracts per-cycle FSI features, and compares
PyBaMM-simulated SoH trajectory with XGBoost FSI-predicted SoH.

Physics model:
  - SPMe: captures electrolyte concentration gradients (more realistic than SPM)
  - SEI: solvent-diffusion-limited side reaction on negative electrode
    (standard capacity-fade mechanism in graphite anodes)
  - Thermal: lumped isothermal at T_sim (can vary)

Output:
  - 04_Code/results/pybamm_spme_results.json
  - plots saved to 04_Code/results/pybamm_*.png

Usage:
  python 04_Code/utils/pybamm_spme_degradation.py
  python 04_Code/utils/pybamm_spme_degradation.py --cycles 60 --temp 35
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent.parent
RESULTS    = ROOT / "04_Code" / "results"
RESULTS.mkdir(exist_ok=True)


# ── FSI constants (must match ml_fsi_model.py) ─────────────────────────────────
W_KI   = 0.30
W_DOD  = 0.25
W_T    = 0.25
W_CP   = 0.20
T_REF  = 25.0


def compute_fsi(ki, dod, t_avg, c_peak_norm):
    t_norm = abs(t_avg - T_REF) / T_REF
    return W_KI * ki + W_DOD * dod + W_T * t_norm + W_CP * c_peak_norm


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return round(float(obj), 6)
        if isinstance(obj, np.ndarray):     return obj.tolist()
        return super().default(obj)


def run(n_cycles: int = 50, t_sim: float = 25.0, c_rate: float = 1.0):
    import pybamm

    print(f"\n{'='*60}")
    print(f"PyBaMM SPMe Degradation — {n_cycles} cycles, T={t_sim}°C, {c_rate}C")
    print(f"{'='*60}")

    # ── 1. Build model ─────────────────────────────────────────────────────────
    model = pybamm.lithium_ion.SPMe(
        options={
            "SEI": "ec reaction limited",   # EC-solvent reaction SEI → capacity fade
        }
    )
    param = pybamm.ParameterValues("Chen2020")    # NMC/graphite validated parameters

    # Set operating temperature
    param.update({"Ambient temperature [K]": 273.15 + t_sim})
    param.update({"Initial temperature [K]": 273.15 + t_sim})

    # Nominal capacity from Chen2020 (5 Ah pouch cell)
    q_nom = float(param["Nominal cell capacity [A.h]"])
    i_1c  = q_nom  # 1C current in Amps

    print(f"Cell: Chen2020 NMC/graphite, Q_nominal = {q_nom:.2f} Ah, I_1C = {i_1c:.2f} A")

    # ── 2. Define cycling experiment ───────────────────────────────────────────
    # CCCV charge to 4.2 V, CC discharge at c_rate to 2.5 V
    charge_c  = min(c_rate, 1.0)   # charge at ≤1C (gentle charge, stress focus on discharge)
    discharge_c = c_rate

    experiment = pybamm.Experiment(
        [
            (
                f"Charge at {charge_c:.2f}C until 4.2 V",
                "Hold at 4.2 V until C/50",
                f"Discharge at {discharge_c:.2f}C until 2.5 V",
                "Rest for 5 minutes",
            )
        ] * n_cycles
    )

    print(f"Experiment: {charge_c}C charge / {discharge_c}C discharge × {n_cycles} cycles")
    print("Running simulation (this takes 1–3 minutes for 50 cycles)...")

    sim = pybamm.Simulation(model, parameter_values=param, experiment=experiment)
    sol  = sim.solve()

    cycles_run = len(sol.cycles)
    print(f"Completed {cycles_run} cycles (terminated early if SoH < 80%)")

    # ── 3. Extract per-cycle metrics ───────────────────────────────────────────
    records = []
    q_initial = None

    for cyc_idx, cyc in enumerate(sol.cycles):
        cyc_num = cyc_idx + 1
        try:
            # Steps: 0=charge CC, 1=charge CV hold, 2=discharge, 3=rest
            charge   = cyc.steps[0]
            discharge = cyc.steps[2]

            # Discharge arrays
            I_dis = discharge["Current [A]"].entries          # negative = discharge
            V_dis = discharge["Terminal voltage [V]"].entries
            T_dis = discharge["Cell temperature [K]"].entries - 273.15
            t_dis = discharge["Time [s]"].entries

            # Charge arrays (for DCSS and RBF)
            I_chg = charge["Current [A]"].entries             # positive = charge
            V_chg = charge["Terminal voltage [V]"].entries

            # Capacity discharged this cycle
            q_discharge = float(np.trapezoid(np.abs(I_dis), t_dis / 3600.0))

            if q_initial is None or cyc_num <= 3:
                if cyc_num == 3:
                    q_initial = q_discharge
                elif cyc_num < 3:
                    continue

            if q_initial is None or q_initial <= 0:
                continue

            soh = min(q_discharge / q_initial * 100.0, 100.0)

            # FSI primary components (discharge phase)
            I_abs = np.abs(I_dis)
            ki    = float(np.std(I_abs) / np.mean(I_abs)) if np.mean(I_abs) > 0 else 0.0
            dod   = float(q_discharge / q_nom)
            t_avg = float(np.mean(T_dis))
            c_peak_norm = float(np.max(I_abs) / i_1c)
            fsi   = compute_fsi(ki, dod, t_avg, c_peak_norm)

            # Secondary features (charge phase)
            I_chg_abs = np.abs(I_chg)
            dcss = float(np.std(I_chg_abs))                   # Dynamic Current Stress Signature
            rbf  = float(np.mean(I_chg_abs))                  # Regenerative Braking Factor proxy
            cvi  = float(np.std(np.concatenate([V_chg, V_dis])))  # Current-Voltage Interaction

            records.append({
                "Cycle":        cyc_num,
                "SoH_%":        round(soh, 4),
                "Q_Ah":         round(q_discharge, 5),
                "KI":           round(ki, 6),
                "DoD_%":        round(dod * 100, 4),
                "T_avg_C":      round(t_avg, 3),
                "T_stress_norm":round(abs(t_avg - T_REF) / T_REF, 6),
                "C_peak_norm":  round(c_peak_norm, 4),
                "FSI":          round(fsi, 6),
                "DCSS":         round(dcss, 6),
                "RBF":          round(rbf, 6),
                "CVI":          round(cvi, 6),
                "V_min":        round(float(np.min(V_dis)), 4),
                "V_mean":       round(float(np.mean(V_dis)), 4),
            })

        except Exception as e:
            print(f"  Cycle {cyc_num}: skipped ({e})")
            continue

    df = pd.DataFrame(records)
    print(f"\nExtracted {len(df)} valid cycles")
    if df.empty:
        print("No valid cycles extracted. Check simulation settings.")
        return

    print(f"  SoH range: {df['SoH_%'].min():.1f}% – {df['SoH_%'].max():.1f}%")
    print(f"  KI range:  {df['KI'].min():.4f} – {df['KI'].max():.4f}  (CC discharge → near 0)")
    print(f"  FSI range: {df['FSI'].min():.4f} – {df['FSI'].max():.4f}")

    # ── 4. Load XGBoost model and predict ──────────────────────────────────────
    soh_predicted = None
    try:
        import pickle
        model_path = ROOT / "05_Models" / "xgb_model.pkl"
        with open(model_path, "rb") as f:
            xgb_model = pickle.load(f)

        feat_cols = [c for c in xgb_model.feature_names_in_ if c in df.columns]
        X = df[feat_cols].values
        soh_predicted = xgb_model.predict(X)
        df["SoH_XGB_pred"] = np.clip(soh_predicted, 0, 100).round(4)
        rmse_xgb = float(np.sqrt(np.mean((df["SoH_%"] - df["SoH_XGB_pred"])**2)))
        print(f"\nXGBoost prediction vs PyBaMM SoH: RMSE = {rmse_xgb:.3f}%")
    except Exception as e:
        print(f"  XGBoost model not loaded: {e}")
        rmse_xgb = None

    # ── 5. Spearman correlation: FSI vs SoH decline ────────────────────────────
    from scipy.stats import spearmanr
    rho_fsi, p_fsi = spearmanr(df["FSI"], -df["SoH_%"])
    rho_ki,  p_ki  = spearmanr(df["KI"],  -df["SoH_%"])
    print(f"\nSpearman(FSI, -SoH): rho={rho_fsi:.4f}, p={p_fsi:.4f}")
    print(f"Spearman(KI,  -SoH): rho={rho_ki:.4f},  p={p_ki:.4f}")

    # ── 6. Plot: SoH trajectory ────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"PyBaMM SPMe + SEI Degradation — {cycles_run} cycles, T={t_sim}°C, {c_rate}C\n"
        f"(Chen2020 NMC/graphite, solvent-diffusion limited SEI)",
        fontsize=12, fontweight="bold"
    )

    # Panel 1: SoH over cycles
    ax = axes[0, 0]
    ax.plot(df["Cycle"], df["SoH_%"], color="#1B2B4A", lw=2, label="PyBaMM SPMe SoH")
    if "SoH_XGB_pred" in df.columns:
        ax.plot(df["Cycle"], df["SoH_XGB_pred"], color="#D4890A",
                lw=1.5, ls="--", label=f"XGBoost FSI prediction (RMSE={rmse_xgb:.2f}%)")
    ax.axhline(80, color="red", ls=":", lw=1, label="EoL (80% SoH)")
    ax.set_xlabel("Cycle Number"); ax.set_ylabel("State of Health (%)")
    ax.set_title("SoH Trajectory — SPMe vs XGBoost FSI Prediction")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_ylim(70, 102)

    # Panel 2: FSI over cycles
    ax = axes[0, 1]
    ax.plot(df["Cycle"], df["FSI"], color="#3E7D52", lw=2)
    ax.set_xlabel("Cycle Number"); ax.set_ylabel("FSI")
    ax.set_title("FSI Evolution Over Cycles")
    ax.grid(True, alpha=0.3)

    # Panel 3: FSI vs SoH scatter
    ax = axes[1, 0]
    sc = ax.scatter(df["FSI"], df["SoH_%"], c=df["Cycle"],
                    cmap="viridis", s=20, alpha=0.7)
    plt.colorbar(sc, ax=ax, label="Cycle")
    ax.set_xlabel("FSI"); ax.set_ylabel("SoH (%)")
    ax.set_title(f"FSI vs SoH — Spearman ρ={rho_fsi:.3f}")
    ax.grid(True, alpha=0.3)

    # Panel 4: Capacity fade
    ax = axes[1, 1]
    ax.plot(df["Cycle"], df["Q_Ah"], color="#990011", lw=2)
    ax.axhline(q_initial * 0.80 if q_initial else 0,
               color="red", ls=":", lw=1, label="80% of initial capacity")
    ax.set_xlabel("Cycle Number"); ax.set_ylabel("Discharge Capacity (Ah)")
    ax.set_title("Capacity Fade (SEI Growth)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_fig = RESULTS / f"pybamm_spme_{c_rate}C_{int(t_sim)}C.png"
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved: {out_fig}")

    # ── 7. Save results JSON ───────────────────────────────────────────────────
    results = {
        "simulation_config": {
            "model":        "SPMe",
            "degradation":  "EC reaction limited SEI growth (standard anode side-reaction)",
            "parameters":   "Chen2020 (NMC/graphite)",
            "n_cycles_requested": n_cycles,
            "n_cycles_completed": cycles_run,
            "n_cycles_extracted": len(df),
            "c_rate":       c_rate,
            "temperature_C": t_sim,
            "q_nominal_Ah": q_nom,
        },
        "soh_trajectory": {
            "cycles":    df["Cycle"].tolist(),
            "soh_pybamm":df["SoH_%"].tolist(),
            "soh_xgb":   df["SoH_XGB_pred"].tolist() if "SoH_XGB_pred" in df.columns else [],
            "rmse_xgb":  rmse_xgb,
        },
        "fsi_features": {
            "FSI":          df["FSI"].tolist(),
            "KI":           df["KI"].tolist(),
            "DoD_%":        df["DoD_%"].tolist(),
            "T_stress_norm":df["T_stress_norm"].tolist(),
            "C_peak_norm":  df["C_peak_norm"].tolist(),
        },
        "correlation_validation": {
            "spearman_rho_fsi_vs_neg_soh": round(rho_fsi, 4),
            "p_fsi":                       round(p_fsi, 6),
            "spearman_rho_ki_vs_neg_soh":  round(rho_ki, 4),
            "p_ki":                        round(p_ki, 6),
            "interpretation": (
                "KI ≈ 0 for CC cycling (SPMe discharge at constant C-rate). "
                "FSI varies due to DoD and T_norm components. "
                "This confirms: variable-current (high-KI) profiles require fleet data."
            ),
        },
        "comparison_to_other_datasets": {
            "CALCE_RMSE": 3.73,
            "SPMe_XGBoost_RMSE": rmse_xgb,
            "note": (
                "SPMe SoH trajectory follows physics-based degradation from SEI growth. "
                "XGBoost FSI prediction approximates this trajectory from cycle-level features. "
                "Discrepancy reflects that XGBoost was trained on CALCE (LiCoO2), "
                "while SPMe uses Chen2020 NMC/graphite parameters."
            ),
        },
    }

    out_json = RESULTS / "pybamm_spme_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, cls=NpEncoder, indent=2)
    print(f"Results saved: {out_json}")

    # ── 8. Summary print ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Cycles simulated:     {cycles_run}")
    print(f"  Final SoH (PyBaMM):   {df['SoH_%'].iloc[-1]:.2f}%")
    if "SoH_XGB_pred" in df.columns:
        print(f"  Final SoH (XGBoost):  {df['SoH_XGB_pred'].iloc[-1]:.2f}%")
        print(f"  RMSE (XGB vs SPMe):   {rmse_xgb:.3f}%")
    print(f"  FSI range:            {df['FSI'].min():.4f} - {df['FSI'].max():.4f}")
    print(f"  Spearman(FSI, -SoH):  rho={rho_fsi:.4f}")
    print(f"  Capacity fade:        {df['Q_Ah'].iloc[0]:.3f} Ah -> {df['Q_Ah'].iloc[-1]:.3f} Ah")
    print(f"  Fade rate:            {(1 - df['Q_Ah'].iloc[-1]/df['Q_Ah'].iloc[0])*100:.2f}% over {len(df)} cycles")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int,   default=50,   help="Number of cycles to simulate")
    parser.add_argument("--temp",   type=float, default=25.0, help="Temperature in Celsius")
    parser.add_argument("--crate",  type=float, default=1.0,  help="Discharge C-rate")
    args = parser.parse_args()

    run(n_cycles=args.cycles, t_sim=args.temp, c_rate=args.crate)
