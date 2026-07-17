# Research Methodology — 7-Stage Framework
## EG7030 Dissertation — Week 4 Detailed Notes

---

## Overview

The dissertation follows a 7-stage methodology designed to move from raw data to a deployable, explainable battery degradation prediction tool for EV fleet applications.

**Timeline (from Dissertation Brief — Phase Plan):**
| Phase | Months | Tasks |
|-------|--------|-------|
| Phase 1 | May–June | Literature review, data collection |
| Phase 2 | June | Dataset preprocessing, feature engineering |
| Phase 3 | July | ML model development |
| Phase 4 | August | SHAP explainability, PyBaMM simulation |
| Phase 5 | August–September | FSI development, validation |
| Phase 6 | September | Dissertation writing, submission |

---

## Stage 1: Data Collection & Preprocessing ✅ (In Progress)

### What Has Been Done
- Downloaded CALCE Battery Dataset (CS, CX, A123, INR 18650-20R, Pouch)
- Downloaded NASA PCoE Battery Dataset (B0005, B0006, B0007, B0018 + Oxford)
- Obtained Fleet DNA delivery truck duty cycle data

### What Needs to Be Done (Week 5)
1. Extract CALCE zip files and load INR 18650-20R NMC cycle data
2. Parse NASA .mat files using `scipy.io.loadmat()`
3. Load Oxford .mat files and .csv files
4. Standardise column names across all datasets
5. Handle missing values and outliers
6. Split into train/validation/test sets (70/15/15)

### Code Template
```python
import pandas as pd
import numpy as np
import scipy.io

# Load CALCE NMC
calce_nmc = pd.read_csv('INR18650_20R_Cycle1.csv')
calce_nmc['SoH'] = calce_nmc['Discharge_Capacity'] / 2.0 * 100

# Load NASA
nasa_data = scipy.io.loadmat('B0005.mat')
cycles = nasa_data['B0005']['cycle'][0][0][0]

# Load Oxford
oxford = scipy.io.loadmat('Oxford_Battery_Degradation_Dataset_1.mat')
```

---

## Stage 2: Duty Cycle Feature Extraction ✅ (Starting Week 5)

### Objective
Extract quantitative metrics that characterise how "hard" each duty cycle is on the battery. These metrics become features in the ML model and the inputs for the Fleet Severity Index (FSI).

### Features to Extract

**From Battery Cycling Data (CALCE/NASA/Oxford):**
| Feature | Formula | Units | Significance |
|---------|---------|-------|-------------|
| Charge capacity | Direct measurement | Ah | Input to SoH |
| Discharge capacity | Direct measurement | Ah | Primary SoH indicator |
| C-rate | I / Q_nominal | h⁻¹ | Stress intensity |
| Coulombic efficiency | Q_discharge / Q_charge | % | Degradation health check |
| Mean discharge voltage | ∫V dt / T | V | State indicator |
| Voltage variance | Var(V) | V² | Stress homogeneity |
| Internal resistance | ΔV / ΔI | Ω | SEI growth proxy |
| Temperature mean/max | Direct measurement | °C | Thermal stress |
| Delta-capacity (ΔQ) | Q_n - Q_{n-1} | Ah | Rate of fade |

**From Fleet DNA Data:**
| Feature | Formula | Units | Significance |
|---------|---------|-------|-------------|
| KI | Σ|a| / d | m/s²/km | Overall aggressiveness |
| RPA | Σ(v×a_pos) / d | — | Acceleration intensity |
| PKE | Σ(0.5 × Δv²_pos) / d | J/kg/km | Energy-based severity |
| Idle fraction | t_idle / t_total | % | Stop-go indicator |
| Max speed | max(v) | mph | Highway vs. urban flag |
| Speed standard deviation | Std(v) | mph | Speed variability |

---

## Stage 3: Exploratory Data Analysis (EDA) — Week 5

### Tasks
1. **Capacity fade curves**: Plot SoH vs. cycle number for all batteries
2. **Correlation heatmap**: Which features correlate with SoH?
3. **Comparison across datasets**: Side-by-side capacity fade for CALCE NMC vs. Oxford NMC
4. **Temperature effect**: Does higher temperature → faster fade? (scatter plot)
5. **C-rate effect**: Higher C-rate → faster degradation? (boxplot by C-rate group)

### Key Expected Findings
- CALCE NMC cells with DST (dynamic) profiles should show faster degradation than CC profiles
- Higher C-rate experiments should show steeper capacity fade
- Temperature > 35°C should accelerate fade (Arrhenius relationship)

---

## Stage 4: ML Model Development — July

### Five Models Planned

| Model | Type | Strengths | When to Use |
|-------|------|-----------|-------------|
| **Random Forest** | Ensemble (bagging) | Robust, handles non-linear, interpretable via feature importance | Baseline |
| **XGBoost** | Ensemble (boosting) | High accuracy, handles missing values | Main model (Paper D2 validated) |
| **LightGBM** | Ensemble (boosting) | Fastest, good for large data | Main model (Paper D3 validated) |
| **SVM (RBF kernel)** | Kernel method | Good for small datasets | Comparison |
| **LSTM** | Deep learning (RNN) | Captures temporal sequences | Sequence prediction |
| **Bi-LSTM** | Deep learning | Bidirectional temporal context | Best for time series |

### Evaluation Metrics
| Metric | Formula | Target |
|--------|---------|--------|
| RMSE | √(Σ(ŷ-y)²/n) | < 2% SoH |
| MAE | Σ|ŷ-y|/n | < 1.5% SoH |
| R² | 1 - SS_res/SS_tot | > 0.95 |
| MAPE | Σ|(ŷ-y)/y|/n × 100 | < 5% |

---

## Stage 5: SHAP Explainability Layer — August

### What is SHAP?
SHAP (SHapley Additive exPlanations) is a framework from game theory that explains individual model predictions by computing the marginal contribution of each feature to the output.

**Reference paper**: Lundberg & Lee (2017) — "A Unified Approach to Interpreting Model Predictions" — NeurIPS
- GitHub: https://github.com/slundberg/shap
- Documentation: https://shap.readthedocs.io/

### Implementation
```python
import shap

# Train your XGBoost model first
model = xgboost.train(params, dtrain)

# Create SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot — shows feature importance
shap.summary_plot(shap_values, X_test, feature_names=feature_names)

# Dependence plot — how one feature affects prediction
shap.dependence_plot("C_rate", shap_values, X_test)
```

### Expected Insights
From SHAP analysis on battery data, you expect to find:
- **C-rate** and **temperature** are high-importance features (supported by Mulpuri et al.)
- **Cycle number** increases in importance as degradation accelerates past knee point
- **KI** and **RPA** (fleet duty cycle features) will have measurable SHAP contributions
- Different features dominate at different life stages (early vs. late degradation)

---

## Stage 6: PyBaMM Simulation — August

### What PyBaMM Does
PyBaMM validates your data-driven ML predictions with physics-based electrochemical simulation. It answers: *"Can the physics explain what the data shows?"*

### Setup
```python
pip install pybamm
import pybamm

# Use SPMe model (Single Particle Model with electrolyte)
model = pybamm.lithium_ion.SPMe()

# Use NMC-811 parameters (matches Mulpuri et al. LGM50 cell)
param = pybamm.ParameterValues("Chen2020")

# Add degradation submodels
model = pybamm.lithium_ion.SPMe(
    options={
        "SEI": "ec reaction limited",
        "lithium plating": "irreversible",
        "particle mechanics": "swelling and cracking"
    }
)
```

### Validation Plan
1. Run PyBaMM simulation with NMC-811 parameters at **low C-rate** (C/5)
2. Compare simulated capacity fade curve against CALCE NMC real data
3. Run simulation at **high C-rate** (1C, 2C) — simulate aggressive fleet duty cycle
4. Compare accelerated fade prediction against ML model prediction
5. **If PyBaMM and ML agree** → confidence in ML model mechanism
6. **If they disagree** → investigate data quality issues or model limitations

### Expected Output
A side-by-side comparison plot showing:
- PyBaMM simulated capacity fade (physics-based)
- ML model predicted capacity fade (data-driven)
- Real CALCE NMC measured capacity fade (ground truth)

---

## Stage 7: Fleet Severity Index (FSI) — Novel Contribution

### What is the FSI?
The Fleet Severity Index is a **single numerical score** (0 to 1) that quantifies how degrading a fleet vehicle's duty cycle is for its battery. It is your dissertation's primary novel contribution.

### Why the FSI Matters
Current fleet battery management uses:
- Fixed replacement schedules (e.g., every 3 years regardless of actual health)
- Simple odometer-based rules (replace at X miles)

The FSI enables:
- **Condition-based replacement**: Replace when FSI × cycles exceeds threshold
- **Route optimisation**: Assign lower-FSI routes to newer batteries
- **Procurement guidance**: Buy batteries rated for the fleet's specific FSI profile

### FSI Formula (Draft)
```
FSI(v, t) = α × KI_norm(v,t) + β × T_factor(v,t) + γ × CR_factor(v,t) + δ × idle_factor(v,t)
```

Where:
- `KI_norm` = normalised Kinetic Intensity (0–1)
- `T_factor` = temperature stress factor (0–1, increases above 35°C)
- `CR_factor` = charging rate stress factor (0–1)
- `idle_factor` = fraction of micro-cycling events (0–1)
- α, β, γ, δ = weights learned from SHAP importance values of each feature

### How FSI Will Be Validated
1. Compute FSI for each Fleet DNA delivery truck route
2. Group routes by FSI quartile (low, medium-low, medium-high, high severity)
3. Apply ML model to predict remaining life of NMC battery under each FSI quartile
4. Compare predictions: high-FSI routes should predict 1.5–2× shorter battery life than low-FSI routes
5. Cross-validate against Mulpuri et al.'s finding that urban driving (high FSI) = 2× faster degradation than highway (low FSI)

---

## Dissertation Chapter Plan

| Chapter | Content | Target Words | Deadline |
|---------|---------|-------------|----------|
| 1. Introduction | Background, problem statement, objectives, scope | 3,000 | July |
| 2. Literature Review | 5 confirmed papers + 5–8 additional | 6,000 | July |
| 3. Methodology | 7-stage framework in detail | 4,000 | August |
| 4. Dataset Analysis | NASA/CALCE/Oxford preprocessing, EDA | 4,000 | August |
| 5. ML Model Results | Performance metrics, training details | 5,000 | September |
| 6. SHAP + PyBaMM | Explainability results, simulation validation | 4,000 | September |
| 7. FSI Framework | Novel contribution, fleet case study | 3,000 | September |
| 8. Discussion | Limitations, implications, comparison to literature | 3,000 | September |
| 9. Conclusion | Summary, contributions, future work | 1,500 | September |
| **Total** | | **~33,500** | **Sept** |

---

## 9 Dissertation Objectives (From Brief)

1. Review literature on battery degradation modelling and EV fleet applications
2. Collect and preprocess battery datasets (NASA, CALCE, Oxford)
3. Extract and quantify duty cycle characteristics from Fleet DNA data
4. Develop and compare ML models for SoH/RUL prediction
5. Apply SHAP for explainability and feature attribution
6. Validate results using PyBaMM electrochemical simulation
7. Develop the Fleet Severity Index (FSI) as a novel metric
8. Apply FSI to Fleet DNA delivery truck data
9. Discuss implications for UK fleet electrification policy (ZEV Mandate, Faraday Challenge)
