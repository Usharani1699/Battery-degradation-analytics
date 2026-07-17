# Full extraction script for Oxford, NASA, and CALCE battery datasets.

import os
import zipfile
import glob
import shutil
import traceback
import numpy as np
import pandas as pd
import scipy.io as sio

BASE = os.path.join(os.path.expanduser("~"), "Music", "Term 3")
OUT  = os.path.join(BASE, "Extracted_CSV_Data")

NASA_DIR   = os.path.join(BASE, "NASA Prognostics (PCoE) Battery Dataset", "5. Battery Data Set")
OXFORD_DIR = os.path.join(BASE, "NASA Prognostics (PCoE) Battery Dataset", "Oxford")
CALCE_DIR  = os.path.join(BASE, "CALCE Battery Dataset")

os.makedirs(os.path.join(OUT, "NASA"),   exist_ok=True)
os.makedirs(os.path.join(OUT, "Oxford"), exist_ok=True)
os.makedirs(os.path.join(OUT, "CALCE"),  exist_ok=True)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def safe_array(val):
    """Flatten any nested numpy array to a plain Python list."""
    if val is None:
        return []
    arr = np.array(val).flatten()
    return arr.tolist()

def mat_struct_to_dict(s):
    """Recursively convert scipy mat_struct to plain dict."""
    result = {}
    if hasattr(s, '_fieldnames'):
        for f in s._fieldnames:
            v = getattr(s, f)
            if hasattr(v, '_fieldnames'):
                result[f] = mat_struct_to_dict(v)
            else:
                result[f] = v
    return result


# ─────────────────────────────────────────────
# 1. NASA
# ─────────────────────────────────────────────

def extract_nasa():
    print("\n=== NASA ===")
    mat_files = glob.glob(os.path.join(NASA_DIR, "**", "B*.mat"), recursive=True)
    # De-duplicate by filename (same cell in multiple subfolders — take last/longest run)
    by_name = {}
    for p in mat_files:
        name = os.path.basename(p)
        if name not in by_name:
            by_name[name] = p

    cycle_rows = []
    impedance_rows = []

    for name, path in sorted(by_name.items()):
        cell_id = name.replace(".mat", "")
        print(f"  Processing {cell_id}...", end="", flush=True)
        try:
            mat = sio.loadmat(path, simplify_cells=True)
            battery = mat.get("B", mat.get(cell_id, None))
            if battery is None:
                # Try first non-dunder key
                keys = [k for k in mat if not k.startswith("_")]
                battery = mat[keys[0]] if keys else None
            if battery is None:
                print(" SKIP (no data key)")
                continue

            cycles = battery.get("cycle", []) if isinstance(battery, dict) else []
            if not hasattr(cycles, '__len__'):
                cycles = [cycles]

            discharge_rows = []
            for cyc_idx, cyc in enumerate(cycles):
                if not isinstance(cyc, dict):
                    cyc = mat_struct_to_dict(cyc)
                cyc_type = str(cyc.get("type", "")).strip()
                data = cyc.get("data", {})
                if not isinstance(data, dict):
                    data = mat_struct_to_dict(data)

                if cyc_type == "discharge":
                    voltage   = safe_array(data.get("Voltage_measured", []))
                    current   = safe_array(data.get("Current_measured", []))
                    temp      = safe_array(data.get("Temperature_measured", []))
                    time_     = safe_array(data.get("Time", []))
                    capacity  = safe_array(data.get("Capacity", []))
                    n = max(len(voltage), len(current), len(temp), len(time_))
                    for i in range(n):
                        discharge_rows.append({
                            "cell_id":     cell_id,
                            "cycle_index": cyc_idx,
                            "time_s":      time_[i]     if i < len(time_)    else np.nan,
                            "voltage_V":   voltage[i]   if i < len(voltage)  else np.nan,
                            "current_A":   current[i]   if i < len(current)  else np.nan,
                            "temp_C":      temp[i]      if i < len(temp)     else np.nan,
                            "capacity_Ah": capacity[i]  if i < len(capacity) else np.nan,
                        })

                elif cyc_type == "impedance":
                    re  = safe_array(data.get("Re", []))
                    rct = safe_array(data.get("Rct", []))
                    impedance_rows.append({
                        "cell_id":     cell_id,
                        "cycle_index": cyc_idx,
                        "Re_ohm":      re[0]  if re  else np.nan,
                        "Rct_ohm":     rct[0] if rct else np.nan,
                    })

                # Cycle-level capacity summary
                if cyc_type == "discharge":
                    cap_arr = safe_array(data.get("Capacity", []))
                    cap_val = cap_arr[-1] if cap_arr else np.nan
                    cycle_rows.append({
                        "cell_id":       cell_id,
                        "cycle_index":   cyc_idx,
                        "cycle_type":    cyc_type,
                        "capacity_Ah":   cap_val,
                    })

            if discharge_rows:
                df = pd.DataFrame(discharge_rows)
                df.to_csv(os.path.join(OUT, "NASA", f"{cell_id}_DischargeSeries.csv"), index=False)
                print(f" {len(discharge_rows)} rows saved", end="")
            print()

        except Exception:
            print(f" ERROR")
            traceback.print_exc()

    if cycle_rows:
        pd.DataFrame(cycle_rows).to_csv(os.path.join(OUT, "NASA", "NASA_CycleSummary.csv"), index=False)
        print(f"  NASA_CycleSummary.csv saved ({len(cycle_rows)} cycle entries)")
    if impedance_rows:
        pd.DataFrame(impedance_rows).to_csv(os.path.join(OUT, "NASA", "NASA_ImpedanceSummary.csv"), index=False)
        print(f"  NASA_ImpedanceSummary.csv saved ({len(impedance_rows)} impedance entries)")


# ─────────────────────────────────────────────
# 2. OXFORD
# ─────────────────────────────────────────────

def extract_oxford():
    print("\n=== Oxford ===")
    out_dir = os.path.join(OUT, "Oxford")

    # The main dataset .mat file
    main_mat = os.path.join(OXFORD_DIR, "Oxford_Battery_Degradation_Dataset_1.mat")
    if not os.path.exists(main_mat):
        # try alternate name
        alts = glob.glob(os.path.join(OXFORD_DIR, "Oxford_Battery_Degradation_Dataset_1*.mat"))
        main_mat = alts[0] if alts else None

    if main_mat:
        print(f"  Loading {os.path.basename(main_mat)}...")
        try:
            mat = sio.loadmat(main_mat, simplify_cells=True)
            keys = [k for k in mat if not k.startswith("_")]
            print(f"  Top-level keys: {keys}")

            for key in keys:
                data = mat[key]
                # Oxford dataset stores cell arrays of structs
                if isinstance(data, np.ndarray):
                    cells_flat = data.flatten()
                    for cell_idx, cell_data in enumerate(cells_flat):
                        cell_name = f"Oxford_cell{cell_idx+1}_{key}"
                        if isinstance(cell_data, dict):
                            _save_oxford_cell(cell_data, cell_name, out_dir)
                        elif hasattr(cell_data, '_fieldnames'):
                            _save_oxford_cell(mat_struct_to_dict(cell_data), cell_name, out_dir)
                elif isinstance(data, dict):
                    _save_oxford_cell(data, key, out_dir)
        except Exception:
            print("  ERROR loading main mat")
            traceback.print_exc()

    # Also copy any already-extracted CSVs from the Oxford folder
    existing_csvs = glob.glob(os.path.join(OXFORD_DIR, "*.csv"))
    for src in existing_csvs:
        dst = os.path.join(out_dir, os.path.basename(src))
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
            print(f"  Copied {os.path.basename(src)}")

    # ExampleDC .mat files
    for dc_mat in glob.glob(os.path.join(OXFORD_DIR, "ExampleDC*.mat")):
        name = os.path.basename(dc_mat).replace(".mat", "").replace(" ", "_")
        print(f"  Loading {name}...")
        try:
            mat = sio.loadmat(dc_mat, simplify_cells=True)
            keys = [k for k in mat if not k.startswith("_")]
            for key in keys:
                data = mat[key]
                if isinstance(data, np.ndarray) and data.ndim >= 1:
                    arr = data.flatten()
                    if arr.dtype.kind in ('f', 'i', 'u'):
                        df = pd.DataFrame(arr, columns=[key])
                        df.to_csv(os.path.join(out_dir, f"{name}_{key}.csv"), index=False)
                        print(f"    Saved {name}_{key}.csv")
        except Exception:
            traceback.print_exc()

    print("  Oxford extraction complete.")


def _save_oxford_cell(data_dict, cell_name, out_dir):
    rows = []
    # Try to identify time-series columns
    arrays = {}
    for k, v in data_dict.items():
        try:
            arr = np.array(v).flatten()
            if arr.dtype.kind in ('f', 'i', 'u') and len(arr) > 1:
                arrays[k] = arr
        except Exception:
            pass
    if not arrays:
        return
    max_len = max(len(a) for a in arrays.values())
    df_dict = {}
    for k, arr in arrays.items():
        if len(arr) == max_len:
            df_dict[k] = arr
    if df_dict:
        df = pd.DataFrame(df_dict)
        out_path = os.path.join(out_dir, f"{cell_name}.csv")
        df.to_csv(out_path, index=False)
        print(f"    Saved {cell_name}.csv ({len(df)} rows, cols: {list(df.columns)})")


# ─────────────────────────────────────────────
# 3. CALCE
# ─────────────────────────────────────────────

def extract_calce():
    print("\n=== CALCE ===")
    tmp_dir = os.path.join(BASE, "_calce_unzip_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    out_dir = os.path.join(OUT, "CALCE")
    os.makedirs(out_dir, exist_ok=True)

    zip_files = glob.glob(os.path.join(CALCE_DIR, "**", "*.zip"), recursive=True)
    print(f"  Found {len(zip_files)} zip files")

    for zpath in sorted(zip_files):
        zip_name = os.path.splitext(os.path.basename(zpath))[0]
        # Derive a clean subfolder name from path context
        rel = os.path.relpath(zpath, CALCE_DIR)
        parts = rel.replace("\\", "/").split("/")
        category = parts[0] if len(parts) > 1 else "misc"
        label = f"{category}_{zip_name}".replace(" ", "_")

        extract_to = os.path.join(tmp_dir, label)
        os.makedirs(extract_to, exist_ok=True)

        try:
            with zipfile.ZipFile(zpath, 'r') as zf:
                zf.extractall(extract_to)
        except Exception as e:
            print(f"  SKIP {zip_name}: zip error — {e}")
            continue

        # Find Excel files inside
        xlsx_files = glob.glob(os.path.join(extract_to, "**", "*.xlsx"), recursive=True)
        xlsx_files += glob.glob(os.path.join(extract_to, "**", "*.xls"), recursive=True)

        if xlsx_files:
            cell_dfs = []
            for xf in sorted(xlsx_files):
                try:
                    df = pd.read_excel(xf, engine="openpyxl")
                    # Add source filename as metadata column
                    df.insert(0, "source_file", os.path.basename(xf))
                    cell_dfs.append(df)
                except Exception as e:
                    print(f"    Could not read {os.path.basename(xf)}: {e}")
            if cell_dfs:
                combined = pd.concat(cell_dfs, ignore_index=True)
                out_csv = os.path.join(out_dir, f"{label}.csv")
                combined.to_csv(out_csv, index=False)
                print(f"  {label}: {len(combined)} rows from {len(cell_dfs)} xlsx -> {os.path.basename(out_csv)}")
            continue

        # Find mat files inside
        mat_files = glob.glob(os.path.join(extract_to, "**", "*.mat"), recursive=True)
        if mat_files:
            for mf in mat_files:
                try:
                    mat = sio.loadmat(mf, simplify_cells=True)
                    keys = [k for k in mat if not k.startswith("_")]
                    for key in keys:
                        data = mat[key]
                        if isinstance(data, np.ndarray) and data.dtype.kind in ('f','i','u'):
                            arr = data
                            if arr.ndim == 1:
                                df = pd.DataFrame(arr, columns=[key])
                            elif arr.ndim == 2:
                                df = pd.DataFrame(arr)
                            else:
                                continue
                            mat_label = f"{label}_{os.path.splitext(os.path.basename(mf))[0]}_{key}"
                            out_csv = os.path.join(out_dir, f"{mat_label}.csv")
                            df.to_csv(out_csv, index=False)
                            print(f"  Saved mat→csv: {os.path.basename(out_csv)} ({len(df)} rows)")
                except Exception as e:
                    print(f"    mat error {os.path.basename(mf)}: {e}")
            continue

        # Find csv files inside
        csv_files = glob.glob(os.path.join(extract_to, "**", "*.csv"), recursive=True)
        if csv_files:
            dfs = []
            for cf in sorted(csv_files):
                try:
                    df = pd.read_csv(cf)
                    df.insert(0, "source_file", os.path.basename(cf))
                    dfs.append(df)
                except Exception:
                    pass
            if dfs:
                combined = pd.concat(dfs, ignore_index=True)
                out_csv = os.path.join(out_dir, f"{label}.csv")
                combined.to_csv(out_csv, index=False)
                print(f"  {label}: {len(combined)} rows from {len(dfs)} csv files")
            continue

        print(f"  {label}: no xlsx/mat/csv found inside zip")

    print(f"\n  CALCE extraction complete. Cleaning up temp folder...")
    shutil.rmtree(tmp_dir, ignore_errors=True)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Output directory: {OUT}")
    extract_nasa()
    extract_oxford()
    extract_calce()
    print("\n=== ALL DONE ===")
    print(f"Check your files at: {OUT}")
