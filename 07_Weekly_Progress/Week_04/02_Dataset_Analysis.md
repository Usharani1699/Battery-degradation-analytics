# Dataset Analysis — NASA · CALCE · Oxford
## EG7030 Dissertation — Week 4 Detailed Notes

---

## 1. NASA Prognostics Center of Excellence (PCoE) Battery Dataset

### Source & Access
- **Organisation**: NASA Ames Research Center — Prognostics Center of Excellence (PCoE)
- **Dataset page**: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
- **Direct dataset**: "Battery Data Set" (#5 in PCoE repository)
- **Local path**: `C:\Users\pure ev\Music\Term 3\NASA Prognostics (PCoE) Battery Dataset\5. Battery Data Set\`

### What's in the NASA Dataset?

**Battery Cells Used:**
| Battery ID | Chemistry | Type | Notes |
|-----------|-----------|------|-------|
| B0005 | LCO (LiCoO₂) | 18650 cylindrical | 2 Ah nominal |
| B0006 | LCO | 18650 cylindrical | 2 Ah nominal |
| B0007 | LCO | 18650 cylindrical | 2 Ah nominal |
| B0018 | LCO | 18650 cylindrical | 2 Ah nominal |

**End-of-Life (EoL) Criterion**: Capacity drops to 1.4 Ah (30% below 2 Ah nominal) = SoH = 70%

**Data Files**: Stored as .mat (MATLAB) format, named `B0005.mat`, `B0006.mat`, etc.

### Three Types of Experiments Per Battery

1. **Charging**: Constant Current (CC) at 1.5A until voltage reaches 4.2V, then Constant Voltage (CV) at 4.2V until current drops to 20mA
2. **Discharging**: Constant Current (CC) at 2A until voltage reaches 2.7V (B0005, B0007) or 2.5V (B0006, B0018)
3. **Impedance (EIS)**: Electrochemical Impedance Spectroscopy at 0.1Hz–5kHz at various states of charge

### Variables Recorded Per Cycle
| Variable | Description | Unit |
|----------|-------------|------|
| `Voltage_measured` | Terminal voltage during operation | V |
| `Current_measured` | Current during operation | A |
| `Temperature_measured` | Case temperature | °C |
| `Current_charge` | Charge current | A |
| `Voltage_charge` | Charge voltage | V |
| `Time` | Timestamp | s |
| `Capacity` | Discharge capacity | Ah |

### EIS Data Structure
Each EIS test contains:
- `Battery_impedance.Re` — Real part of impedance vs. frequency
- `Battery_impedance.Rct` — Charge transfer resistance
- `Battery_impedance.Frequency` — Frequency points (Hz)

**Why EIS Matters**: Paper D2 (Si et al. 2025) uses EIS features (Zreal, Zimag at specific frequencies) as XGBoost features to predict capacity. The NASA dataset is the ONLY one of the three that includes EIS measurements — making it essential for replicating D2 methodology.

### How Many Cycles?
- B0005: ~167 cycles until EoL
- B0006: ~168 cycles until EoL
- B0007: ~168 cycles until EoL
- B0018: ~132 cycles until EoL

---

## 2. CALCE Battery Research Group Dataset

### Source & Access
- **Organisation**: Center for Advanced Life Cycle Engineering (CALCE), University of Maryland
- **Dataset page**: https://calce.umd.edu/battery-data
- **Local path**: `C:\Users\pure ev\Music\Term 3\CALCE Battery Dataset\`

### What's in the CALCE Dataset?

The CALCE dataset has **multiple battery series** with different chemistries:

#### CS2 Series (LCO — Lithium Cobalt Oxide)
- **Cell**: CS2 prismatic cells
- **Chemistry**: LiCoO₂ (LCO) / graphite
- **Capacity**: 1.1 Ah nominal
- **Local**: `CS\` folder (zip files: CS2_33.zip, CS2_35.zip, etc.)
- **Experiments**: Cycle aging under constant charge/discharge rates

#### CX2 Series (LCO)
- **Cell**: CX2 prismatic cells
- **Chemistry**: LiCoO₂ (LCO)
- **Local**: `CX\` folder
- **Experiments**: Similar to CS2 series

#### A123 Battery (LFP — Lithium Iron Phosphate)
- **Cell**: ANR26650M1A cylindrical
- **Chemistry**: LiFePO₄ (LFP) / graphite
- **Capacity**: 2.3 Ah
- **Local**: `A123 Battery\` folder
- **Experiments**: Various charge rates

#### INR 18650-20R — THE NMC CELL (Most Important for Your Study)
- **Cell**: Samsung INR 18650-20R
- **Chemistry**: NMC (Nickel Manganese Cobalt Oxide)
- **Capacity**: 2.0 Ah nominal
- **Local**: `INR 18650-20R Battery\` folder
- **Why Important**: This is the **NMC cell** in the CALCE dataset — your selected chemistry

#### Pouch Cells
- **Local**: `Pouch Cells\` folder
- Various chemistries

#### Dynamic Test Profile
- **Local**: `Dynamic Test Profile\` folder
- Includes **DST (Dynamic Stress Test)**, **FUDS (Federal Urban Driving Schedule)**, **US06** profiles
- These dynamic profiles simulate real-world driving more closely than constant CC-CV tests

### Variables in CALCE CSV Files
| Column | Description |
|--------|-------------|
| Cycle_Index | Cycle number |
| Start_Time | Cycle start timestamp |
| End_Time | Cycle end timestamp |
| Start_Voltage | Voltage at cycle start (V) |
| End_Voltage | Voltage at cycle end (V) |
| Start_Current | Current at start (A) |
| Charge_Capacity | Capacity charged (Ah) |
| Discharge_Capacity | Capacity discharged (Ah) |
| Charge_Energy | Energy charged (Wh) |
| Discharge_Energy | Energy discharged (Wh) |

### How to Access INR 18650-20R Data
The NMC cell data is in: `C:\Users\pure ev\Music\Term 3\CALCE Battery Dataset\INR 18650-20R Battery\`

This should contain CSV or Excel files with cycle-by-cycle capacity data showing degradation from ~2.0 Ah down to ~1.4 Ah over hundreds of cycles.

---

## 3. Oxford Battery Degradation Dataset

### Source & Access
- **Organisation**: University of Oxford, Department of Engineering Science
- **Dataset paper**: Birkl et al. (2017), "Degradation diagnostics for lithium ion cells"
- **Dataset DOI**: https://doi.org/10.5287/ora-p22a6e67s
- **Local path**: `C:\Users\pure ev\Music\Term 3\NASA Prognostics (PCoE) Battery Dataset\Oxford\`

*(Note: The Oxford dataset is stored inside the NASA folder in your local machine)*

### What's in the Oxford Dataset?

**Cell Families:**
| Series | Full Name | Chemistry | Format |
|--------|-----------|-----------|--------|
| BMP | Battery Modelling Pouch | NMC-type pouch | .mat + .csv |
| BMR | Battery Modelling Round | Similar NMC pouch | .mat + .csv |
| SPM | Single Particle Model test | NMC pouch | .mat + .csv |

**Key File**: `Oxford_Battery_Degradation_Dataset_1.mat` — contains the main dataset

### Variables in Oxford Dataset
- Voltage (V)
- Current (A)
- Temperature (°C)
- Capacity (Ah)
- State of Health (SoH) — directly computed and stored
- State of Charge (SoC) — estimated and stored
- Cycle number

### How Oxford Data Was Collected
- **Check-up tests**: Every 100 cycles, batteries were discharged at low C-rate (C/25) to measure true capacity
- **Aging cycles**: High-rate cycling between check-ups
- **Temperature-controlled**: Experiments done at 25°C (standard) and other temperatures

---

## Common Points Across All Three Datasets

| Feature | NASA | CALCE | Oxford |
|---------|------|-------|--------|
| Voltage measurements | Yes | Yes | Yes |
| Current measurements | Yes | Yes | Yes |
| Temperature measurements | Yes | Yes | Yes |
| Capacity per cycle | Yes | Yes | Yes |
| SoH derivable | Yes | Yes | Yes (direct) |
| SoC derivable | Yes | Yes | Yes (direct) |
| RUL calculable | Yes | Yes | Yes |
| EIS / Impedance data | Yes | No | No |
| Dynamic drive profiles | No | Yes (DST/FUDS) | No |
| NMC chemistry available | No (LCO only) | Yes (INR 18650-20R) | Yes (pouch cells) |
| Multi-rate experiments | Yes | Yes | Yes |
| Temperature variation | Limited | Limited | Yes |

### Three Key Common Points (For Presentation)

**1. Core Measurement Variables (V, I, T, Capacity)**
All three datasets record voltage, current, temperature, and capacity every cycle. These 4 variables are the input features for every ML model you will train.

**2. Capacity Fade as SoH Indicator**
All three show gradual capacity fade from nominal to End-of-Life (70–80% of nominal). SoH = Q_current / Q_nominal (×100%). This is derivable from all three datasets identically.

**3. Controlled Cycling Protocols Enable Comparison**
All use CC-CV (Constant Current – Constant Voltage) charging, which allows direct cross-dataset comparison of aging rates despite different chemistries.

---

## Why These Datasets Work Together

```
NASA PCoE  ──┐
              ├──► Common Features ──► ML Training ──► SoH/RUL Model
CALCE NMC  ──┤   (V, I, T, Capacity)   (NMC cells)    (Explainable via SHAP)
              │                                              │
Oxford NMC ──┘                                              ▼
                                                     Fleet Severity Index (FSI)
                                                     Applied to Fleet DNA data
```

- **NASA**: Provides EIS features for Paper D2 methodology replication
- **CALCE NMC + Oxford**: Provide NMC-chemistry training data for ML models
- **CALCE Dynamic profiles**: Provide duty-cycle-like data for bridging lab→fleet gap
- **Fleet DNA**: Provides real-world EV fleet data to apply FSI concept

---

## Data Preprocessing Plan (Stage 1–2 of Methodology)

### Step 1: Extract Raw Data
```python
# NASA - read MATLAB files
import scipy.io
data = scipy.io.loadmat('B0005.mat')

# CALCE - read CSV
import pandas as pd
calce = pd.read_csv('INR18650_20R_Cycle1.csv')

# Oxford - read MATLAB
oxford = scipy.io.loadmat('Oxford_Battery_Degradation_Dataset_1.mat')
```

### Step 2: Compute SoH
```python
# SoH formula
Q_nominal = 2.0  # Ah for CALCE NMC cell
soh = discharge_capacity / Q_nominal * 100  # As percentage
```

### Step 3: Feature Engineering
From each cycle, extract:
- Mean discharge voltage
- Voltage variance
- Capacity (Ah)
- Energy (Wh)
- Temperature (mean, max, min)
- C-rate (current / capacity)
- Cycle number
- Delta-capacity (change from previous cycle)

### Step 4: SoC Estimation
```python
# Coulomb counting for SoC
dt = 1  # time step in seconds
I = current_array  # discharge current
Q_total = 2.0  # Ah
soc = 1.0 - (np.cumsum(I * dt) / 3600) / Q_total
```
