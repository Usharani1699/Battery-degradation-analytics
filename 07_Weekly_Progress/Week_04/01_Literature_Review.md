# Literature Review — 5 Key Papers
## EG7030 Dissertation — Week 4 Detailed Notes

---

## Paper 1 (MOST RELEVANT — from uploaded PDF)

### Citation
Mulpuri, S., Sah, B., & Kumar, A. (2025). **Beyond drive cycles: mapping the intricacies of electric vehicle battery health in diverse environments and driving conditions.** *RSC Advances*, 15, 30980–31004.

- **DOI**: https://doi.org/10.1039/d5ra04379d
- **Journal**: RSC Advances (Royal Society of Chemistry) — open access
- **Impact Factor**: ~4.0
- **File**: D1_RA-015-D5RA04379D.pdf (in your Research papers folder)

### What This Paper Is About
This paper studies how real-world driving conditions affect lithium-ion battery degradation in electric vehicles. The authors used an NMC-811 LGM50 21700 cylindrical cell (5 Ah capacity, made by LG) and applied the **Single Particle Model with Electrolyte (SPMe)** — an electrochemical model — to simulate battery aging under three different Indian regional driving patterns.

### Battery Cell Used
- **Cell**: LGM50 21700 cylindrical lithium-ion cell
- **Chemistry**: NMC-811 (LiNi₀.₈Mn₀.₁Co₀.₁O₂ positive electrode / graphite negative electrode)
- **Capacity**: 5 Ah
- **Manufacturer**: LG Energy Solution

### Three Driving Patterns Studied
1. **Pattern 1 — Urban stop-go** (city driving): Frequent acceleration and braking, short distances, high idle time. Causes the FASTEST battery degradation due to micro-cycling and high average C-rates.
2. **Pattern 2 — Mixed urban-highway**: Moderate speed variation. Intermediate degradation rate.
3. **Pattern 3 — Highway cruising**: Sustained high speed with fewer acceleration events. Slowest degradation rate.

### Degradation Mechanisms Identified
- **SEI (Solid Electrolyte Interphase) layer growth**: Forms on the negative electrode (graphite) surface during cycling. Consumes lithium and increases internal resistance. Accelerated by high temperature and high C-rate.
- **Lithium plating**: Occurs when charging rate exceeds lithium intercalation capacity. Deposits metallic lithium on graphite surface. Risk is highest during fast charging or low temperatures.
- **Particle cracking**: Mechanical fracture of electrode particles due to repeated expansion/contraction. Creates new surface area → more SEI → accelerated capacity loss. Worse under high C-rate cycling.

### Drive Cycle Metrics Used
- **KI (Kinetic Intensity)**: Measures how aggressively the vehicle accelerates and decelerates. Higher KI = higher battery stress.
- **RPA (Relative Positive Acceleration)**: Ratio of positive acceleration work to total distance. Indicates how much energy is used for acceleration vs. constant speed.
- **PKE (Positive Kinetic Energy)**: Total kinetic energy gained per unit distance. Related to C-rate demand on the battery.

### Key Findings
- Urban driving (Pattern 1) caused capacity fade ~2× faster than highway driving (Pattern 3)
- High C-rate + elevated temperature (above 35°C in Indian urban environments) compounds degradation multiplicatively
- Lithium plating was identified as the primary degradation mode under high-rate urban charging
- SPMe model successfully captured the non-linear nature of capacity fade under variable duty cycles

### Relevance to Your Dissertation (Why This Paper Matters)
1. **Core thesis validation**: Directly proves that duty cycle characteristics determine degradation rate — the central argument of your dissertation.
2. **Chemistry alignment**: Uses NMC-811, which you are also using (CALCE INR 18650-20R is NMC).
3. **Model alignment**: Uses SPMe which is implemented in PyBaMM — your Stage 6 simulation tool.
4. **Fleet Severity Index (FSI) concept**: The pattern-based degradation difference supports the idea of scoring fleet routes by severity.
5. **Metric adoption**: KI, RPA, PKE are the drive cycle features you will extract from the Fleet DNA dataset.

---

## Paper 2 (RELEVANT — from uploaded PDF)

### Citation
Si, X., Matsuda, K., et al. (2025). **Capacity Estimation and Knee Point Prediction Using Electrochemical Impedance Spectroscopy for Lithium Metal Battery Degradation via Machine Learning.** *Advanced Science*, 12, 2502336.

- **DOI**: https://doi.org/10.1002/advs.202502336
- **Journal**: Advanced Science (Wiley) — high impact, open access
- **Impact Factor**: ~15.1
- **File**: D2_ADVS-12-2502336.pdf (in your Research papers folder)

### What This Paper Is About
This paper uses **Electrochemical Impedance Spectroscopy (EIS)** measurements as input features to a machine learning model (XGBoost) to predict battery capacity and detect the "knee point" — the inflection point in the capacity-fade curve where degradation suddenly accelerates. SHAP (SHapley Additive exPlanations) is used to explain which EIS features drive the predictions.

### Battery Cell Used
- **Chemistry**: LiNi₀.₈Mn₀.₁Co₀.₁O₂ positive electrode (NMC-811)
- **Type**: Lithium metal battery (LMB) configuration

### What is EIS?
Electrochemical Impedance Spectroscopy measures how a battery responds to small AC signals at different frequencies. It produces a Nyquist plot showing:
- **Zreal** (resistance in ohms — real part): Related to ohmic resistance and SEI layer thickness
- **Zimag** (imaginary part): Related to charge transfer resistance and double-layer capacitance
- **Frequency sweep**: From 0.01 Hz to 100,000 Hz reveals different internal processes

The NASA PCoE dataset includes EIS measurements — making this paper methodology directly applicable to your NASA data.

### Machine Learning Approach
- **Model**: XGBoost (extreme gradient boosting)
- **Features**: EIS parameters at multiple frequencies (Zreal, Zimag, |Z|, phase angle)
- **Targets**: Capacity estimation + knee point location
- **Explainability**: SHAP values show which EIS frequencies correlate most strongly with capacity loss

### What is the Knee Point?
The capacity-cycle curve typically shows:
1. Initial stable region (gradual, linear fade)
2. **Knee point**: Sudden acceleration in capacity loss
3. Rapid degradation region → End of Life (EoL)

Detecting the knee point early = actionable warning for fleet operators (typically at 70–80% SoH = time to replace in EV fleets).

### Key Findings
- EIS features measured at medium frequencies (1–100 Hz) are most predictive of capacity
- XGBoost outperforms linear models for knee point prediction
- SHAP reveals that Zreal at ~10 Hz is the dominant degradation indicator
- Capacity can be estimated with high accuracy (R²>0.95) purely from impedance measurements — no need to run full charge-discharge cycles

### Relevance to Your Dissertation
1. **XGBoost is one of your planned ML models** — this paper validates its use for battery degradation
2. **SHAP framework**: You are adopting SHAP for explainability in Stage 5 of your methodology
3. **EIS data**: NASA PCoE dataset has impedance measurements — enables direct comparison
4. **Knee point**: Maps to the SoH threshold (70–80%) used for EV fleet replacement decisions
5. **NMC-811**: Same chemistry as your selected cell — confirms generalisability

---

## Paper 3 (RELEVANT — from uploaded PDF)

### Citation
İnan, A., et al. (2025). **Descriptive proximity-based modeling for accurate and explainable battery state-of-charge estimation.** *Journal of Energy Storage*, 136, 118150.

- **DOI**: https://doi.org/10.1016/j.est.2025.118150
- **Journal**: Journal of Energy Storage (Elsevier)
- **Impact Factor**: ~9.4
- **File**: 1-s2.0-S2352152X25028634-main.pdf (in your Research papers folder)

### What This Paper Is About
This paper proposes a hybrid method combining **DBSCAN clustering** (an unsupervised ML algorithm) with **LightGBM** (a gradient boosting tree model) to estimate battery State of Charge (SoC) accurately and explainably. SHAP is used to identify which battery features are most important for SoC prediction.

### What is DBSCAN?
DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is an unsupervised algorithm that:
- Groups data points that are close together in feature space
- Labels outliers as noise
- In this paper: groups battery data by similar charging/discharging behaviour patterns
- Allows training separate LightGBM models for each cluster = better accuracy

### What is LightGBM?
LightGBM (Light Gradient Boosting Machine) is a gradient boosting framework developed by Microsoft. Key advantages:
- Much faster than XGBoost for large datasets
- Handles categorical features natively
- Leaf-wise tree growth (better accuracy than level-wise)
- Ideal for tabular battery data (V, I, T, capacity, cycle count)

### Model Performance
- **R² ≈ 1.00** (essentially perfect fit)
- **RMSE ≈ 0.13%** SoC estimation error (extremely accurate)
- Outperforms LSTM and SVM on dynamic drive cycle data

### Input Features Used
- Voltage (V)
- Current (A)
- Capacity (Ah)
- Energy (Wh)
- Cycle count
- Temperature (°C)

These are exactly the features available in the CALCE and NASA datasets.

### Relevance to Your Dissertation
1. **LightGBM is a core model** in your planned ML ensemble (alongside XGBoost, Random Forest, LSTM)
2. **SHAP for explainability**: The same SHAP framework you are adopting in Stage 5
3. **Feature alignment**: The 6 features (V, I, capacity, energy, cycles, T) are all present in CALCE/NASA/Oxford
4. **DBSCAN for clustering**: Can be applied to cluster fleet duty cycle patterns in your Fleet DNA dataset
5. **SoC estimation**: Direct application to your dissertation's SoC modelling objective

---

## Paper 4 (FOUNDATIONAL — from literature knowledge)

### Citation
Severson, K. A., Attia, P. M., Jin, N., et al. (2019). **Data-driven prediction of battery cycle life before capacity degradation.** *Nature Energy*, 4, 383–391.

- **DOI**: https://doi.org/10.1038/s41560-019-0356-8
- **Journal**: Nature Energy (Nature Publishing Group)
- **Impact Factor**: ~60+ (one of the highest in the field)
- **Dataset**: Publicly available at https://data.matr.io/1/

### What This Paper Is About
This landmark 2019 paper from Stanford, MIT, and Toyota Research Institute demonstrated that machine learning can predict a battery's **total cycle life** using only data from the **first 100 cycles** — long before any significant capacity degradation has occurred.

### What Was Done
- **124 LFP (LiFePO₄) cells** (A123 Systems, 1.1 Ah) were cycled under varied fast-charging protocols until end of life
- End of life = 80% of nominal capacity (same as SoH = 80%)
- **Features**: Differences in discharge voltage curve between cycles 10 and 100, variance of discharge capacity over first 100 cycles, minimum, mean, skewness, etc.
- **Models**: Elastic Net regression + Random Forest — tested on held-out cells
- **Result**: Predicted cycle life within 9.1% error on unseen test cells

### Why This Paper is Foundational
1. **Established ML viability**: First high-quality demonstration that ML can predict battery lifetime before degradation is visible
2. **Feature engineering approach**: Shows that derived features (differences, variances) from charge/discharge curves outperform raw measurements
3. **Public dataset**: The 124-cell dataset is publicly available and used as a benchmark by many subsequent papers
4. **Benchmark comparison**: Your dissertation results should be compared against this Severson 2019 baseline

### Connection to Your Work
- **Chemistry difference**: They used LFP; you are using NMC — NMC has more complex degradation, which is a research gap your dissertation addresses
- **Duty cycle gap**: Their cells were charged under different CC protocols but not real EV drive cycles — your work extends this to fleet duty cycles
- **FSI concept**: Their finding that charging protocol determines lifetime supports the Fleet Severity Index concept

---

## Paper 5 (METHODOLOGY — from literature knowledge)

### Citation
Sulzer, V., Marquis, S. G., Timms, R., Hasan, M., & Chapman, S. J. (2021). **Python Battery Mathematical Modelling (PyBaMM).** *Journal of Open Research Software*, 9(1), 14.

- **DOI**: https://doi.org/10.5334/jors.309
- **Journal**: Journal of Open Research Software (Ubiquity Press)
- **GitHub**: https://github.com/pybamm-team/PyBaMM
- **Documentation**: https://pybamm.readthedocs.io/
- **Developed at**: University of Oxford Mathematical Institute

### What PyBaMM Is
PyBaMM is an open-source Python library for electrochemical battery modelling. It implements physically-grounded models that simulate battery internal chemistry mathematically. This is different from data-driven ML — PyBaMM is **physics-based simulation**.

### Models Implemented in PyBaMM
| Model | Full Name | Complexity | Use Case |
|-------|-----------|------------|----------|
| SPM | Single Particle Model | Low | Fast screening |
| SPMe | SPM with electrolyte | Medium | Realistic simulation |
| DFN | Doyle-Fuller-Newman | High | Research-grade accuracy |
| MPM | Many Particle Model | High | Electrode heterogeneity |

### Degradation Submodels (Relevant to Your Dissertation)
- **SEI growth**: Models the electrochemical reactions that form the SEI layer; outputs capacity loss and resistance increase per cycle
- **Particle cracking**: Models mechanical fracture from diffusion-induced stress; outputs new surface area exposed to electrolyte
- **Lithium plating**: Models the conditions under which lithium deposits as metal instead of intercalating; outputs irreversible capacity loss

### Why This Directly Applies to Your Dissertation
The dissertation brief (Stage 6 of your methodology) explicitly states:
> "Validation using PyBaMM simulation framework (SPMe model)"

This means you will:
1. Load NMC-811 parameters into PyBaMM
2. Simulate capacity fade under Fleet DNA duty cycle stress profiles
3. Compare PyBaMM simulation results against real CALCE/Oxford NMC data
4. Use the comparison to validate your ML model predictions

### Getting Started with PyBaMM
```python
pip install pybamm
import pybamm

model = pybamm.lithium_ion.SPMe()
geometry = model.default_geometry
param = pybamm.ParameterValues("Chen2020")  # NMC-811 LGM50 cell parameters
# These are the EXACT parameters for the LGM50 21700 cell from Paper D1!
```

The `Chen2020` parameter set in PyBaMM is for the **LGM50 NMC-811 cell** — exactly the same cell used in Mulpuri et al. (2025, Paper D1). This means you have validated electrochemical parameters ready to use.

---

## Summary Table — All 5 Papers

| # | Paper | Journal | Year | Chemistry | ML Method | Explainability | Relevance |
|---|-------|---------|------|-----------|-----------|---------------|-----------|
| 1 | Mulpuri et al. | RSC Advances | 2025 | NMC-811 | SPMe (physics) | Mechanistic | Highest — duty cycle degradation |
| 2 | Si et al. | Advanced Science | 2025 | NMC-811 | XGBoost | SHAP | High — EIS + ML for RUL |
| 3 | İnan et al. | J. Energy Storage | 2025 | Li-ion | LightGBM + DBSCAN | SHAP | High — SoC estimation framework |
| 4 | Severson et al. | Nature Energy | 2019 | LFP | Elastic Net + RF | None | Foundational — benchmark |
| 5 | Sulzer et al. | J. Open Res. Sw. | 2021 | Universal | PyBaMM (physics) | Physics-based | Direct — dissertation Stage 6 tool |

---

## Additional Reading Recommendations for Week 5–6

When you sit down to read more papers, prioritise these:

1. **Chen et al. (2020)** — "Development of Experimental Techniques for Parameterization of Multi-scale Lithium-ion Battery Models" — *J. Electrochem. Soc.* — Provides the NMC-811 LGM50 parameters used in PyBaMM
   - DOI: https://doi.org/10.1149/1945-7111/ab9050

2. **Attia et al. (2022)** — "Closed-loop optimization of fast-charging protocols for batteries with machine learning" — *Nature* — Extends Severson 2019 approach with active learning
   - DOI: https://doi.org/10.1038/s41586-020-1994-5

3. **Stiaszny et al. (2014)** — "Electrochemical characterization and post-mortem analysis of aged LiFePO4–Li[Ni0.5Co0.2Mn0.3]O2 lithium ion batteries" — Degradation mechanisms reference
   - Journal of Power Sources, DOI: 10.1016/j.jpowsour.2014.08.060

4. **Ng et al. (2020)** — "Predicting the State of Charge and Health of Batteries using Data-Driven Machine Learning" — *Nature Machine Intelligence*
   - DOI: https://doi.org/10.1038/s42256-020-0156-7
