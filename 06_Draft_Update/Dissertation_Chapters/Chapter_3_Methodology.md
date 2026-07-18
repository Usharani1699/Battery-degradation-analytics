# Chapter 3: Methodology

**Module:** EG7030 — Dissertation  
**Student:** Usha Rani Vamanagiri (u2965962)  
**Programme:** MSc Electric Vehicles and Energy Storage, UEL  

---

## 3.1 Overview of the Research Framework

This chapter describes the methodological framework used to develop, calibrate, and validate the Fleet Stress Index (FSI) as a composite battery degradation predictor for electric vehicle fleet applications. The research follows a four-stage pipeline:

1. **Feature Engineering** — Derivation of the FSI formula and its constituent components from first principles and literature
2. **Data Extraction** — Per-cycle FSI feature extraction from five independent battery datasets spanning three chemistries and four current-profile types
3. **Model Training** — Supervised machine learning (XGBoost regression; Random Forest classification) trained on CALCE FSI features
4. **Multi-dataset Validation** — Systematic evaluation of model transferability across chemistry, current profile, and experimental source

The central methodological contribution is the FSI itself: a physically motivated composite index that introduces *Kinetic Intensity* (KI) — a measure of current variability — as the primary distinguishing feature between laboratory cycling conditions and real-world EV fleet operation.

---

## 3.2 The Fleet Stress Index (FSI)

### 3.2.1 Theoretical Motivation

Laboratory battery datasets universally employ constant-current (CC) cycling: the charging and discharging current is fixed at a constant C-rate throughout each cycle. In this regime, the coefficient of variation of the absolute current — defined as σ(|I|)/μ(|I|) — is identically zero regardless of the C-rate magnitude. Real-world EV fleet operation, by contrast, produces highly dynamic current profiles driven by acceleration events, regenerative braking, route topology, and stop-start urban duty cycles. Analysis of 1,412 NREL Fleet DNA telematics records (Section 3.3.3) confirms that real fleet KI values range from 0.60 to 0.81 across commercial vehicle classes.

This gap — KI = 0 in all laboratory training data, KI = 0.60–0.81 in real fleet operation — means that any SoH model trained purely on CC lab data is systematically blind to the primary stress mechanism distinguishing fleet batteries from test-bench batteries. The FSI is designed to make this hidden dimension explicit and quantifiable.

### 3.2.2 FSI Formula

The FSI is defined as a weighted linear combination of four physically motivated stress components:

$$\text{FSI} = 0.30 \times \text{KI} + 0.25 \times \text{DoD} + 0.25 \times T_{\text{norm}} + 0.20 \times C_{\text{peak,norm}}$$

Each component is described below.

#### Component 1: Kinetic Intensity (KI)

$$\text{KI} = \frac{\sigma(|I|)}{\mu(|I|)}$$

KI is the coefficient of variation of the absolute current magnitude over all timesteps in a single charge–discharge cycle. It is zero for any constant-current protocol regardless of C-rate, and increases monotonically with current variability. KI is dimensionless and bounded below at zero.

**Physical interpretation:** KI captures the degree to which a battery is subjected to rapid current transients within a cycle. High KI implies frequent switching between high-current acceleration and near-zero or negative current during regenerative braking — conditions associated with accelerated lithium plating at the anode, uneven lithium intercalation, and mechanical stress in electrode particles (Attia et al., 2020; Wang et al., 2020).

#### Component 2: Depth of Discharge (DoD)

$$\text{DoD} = \frac{Q_{\text{discharge}}}{Q_{\text{nominal}}}$$

DoD is the fraction of nominal capacity discharged per cycle. It is computed from the measured discharge capacity Ah and the rated nominal capacity of the cell. DoD = 1.0 represents a full discharge; partial cycling (e.g. DoD = 0.5 in urban commuting) reduces per-cycle stress but increases the number of shallow cycles for equivalent energy throughput.

#### Component 3: Normalised Thermal Stress (T_norm)

$$T_{\text{norm}} = \frac{|T_{\text{avg}} - 25|}{25}$$

T_norm quantifies deviation from the ISO 12405 reference temperature of 25°C, normalised to that reference. The absolute value makes the function symmetric around 25°C — it treats elevated and depressed temperatures as equivalent deviations from the kinetic optimum.

**Limitation:** The symmetric formulation contradicts the Arrhenius degradation model, under which elevated temperature accelerates reaction kinetics and side-reactions (electrolyte decomposition, SEI growth) more rapidly than equivalent cold temperatures slow them. This is acknowledged as a scoped limitation (Section 5.4) and proposed as a direction for future work.

#### Component 4: Normalised Peak C-rate (C_peak,norm)

$$C_{\text{peak,norm}} = \frac{\max(|I|)}{I_{1C}}$$

C_peak,norm captures the maximum instantaneous current demand, normalised to the 1C rate (where 1C = Q_nominal in Amperes). It is distinct from KI: KI measures current variability while C_peak,norm measures peak current magnitude. Both are relevant — a battery subjected to constant 5C charging has high C_peak,norm and KI ≈ 0, while a fleet battery with frequent acceleration spikes may have moderate C_peak,norm but high KI.

### 3.2.3 Weight Derivation

The FSI weights (0.30/0.25/0.25/0.20) were set through a combination of physical reasoning, literature precedent, and cross-protocol empirical validation. It is important to be explicit about why pure data-driven optimisation from the CALCE training set alone was insufficient.

**Why CALCE gradient optimisation is degenerate for KI weighting.** All CALCE cells are cycled at constant current (CC), meaning KI = 0.000 for every training cycle. A gradient-based optimiser (scipy L-BFGS-B, Σwᵢ = 1, wᵢ > 0) applied to CALCE training data produces optimal weights of approximately KI ≈ 0.000, DoD ≈ 0.999 — KI weight collapses to zero because KI has zero variance in the training set and therefore zero predictive power within that dataset. This is a correct mathematical result, not a model failure: it confirms that KI cannot be learned from CC training data alone.

**Physical and literature basis for the chosen weights.** KI receives the highest weight (0.30) based on: (1) the NREL Fleet DNA analysis showing real fleet KI = 0.60–0.81 versus lab KI = 0.000 — a categorical separation that makes KI the primary source of lab-fleet mismatch; and (2) published literature linking current variability to accelerated anode degradation via lithium plating and intercalation stress (Attia et al., 2020; Wang et al., 2020). DoD and T_norm each receive 0.25, consistent with their well-established roles in cycle-life models (Millner, 2010; Plett, 2015). C_peak,norm receives 0.20 as the least independently informative component when KI already captures current variability.

**Cross-protocol empirical validation.** The chosen weights were validated against the Severson et al. (2019) dataset (23 protocols with varying KI): Spearman(FSI, −cycle_life) ρ = 0.807 with the chosen weights versus ρ = 0.744 for KI alone, confirming that the composite weighting adds predictive value. A weight sensitivity analysis (Section 4.14) confirms that CALCE RMSE is most sensitive to the DoD and C_peak weights, while KI weight sensitivity is zero within CALCE — consistent with KI=0 throughout that dataset. Severson ordinal ρ is used as the KI-specific validation signal instead.

**SHAP attribution on the trained model.** SHAP TreeExplainer applied to the XGBoost model trained on CALCE data shows that the composite FSI feature accounts for 82.4% of mean absolute SHAP importance in regression and 77.9% for End_of_Life classification. Individual KI, T_stress_norm, RBF, and CVI each show 0.0% SHAP importance in isolation — consistent with the CALCE training set having KI = 0 for all cycles.

**Feature collinearity note.** The model feature set includes both FSI and two of its direct components (KI and T_stress_norm). This creates deliberate redundancy: the model can draw on either the fixed-weight composite (FSI) or the raw components independently. On CALCE data, this redundancy is harmless — KI and T_stress_norm have zero variance and zero SHAP contribution. However, on fleet or variable-protocol data where KI varies, FSI's high SHAP weight partly reflects that it is a transformed re-encoding of KI itself — creating a form of feature collinearity that inflates FSI's apparent importance relative to its components. This is acknowledged as a methodological limitation: for future work, using component features without FSI as a composite, or using FSI without its components, would provide a cleaner SHAP interpretation. The current design was retained because FSI's fixed-weight formula provides an interpretable physical quantity regardless of SHAP attribution.

In summary, the FSI weights are principled rather than purely data-driven: they reflect the physical significance of each stress dimension in fleet operation, validated empirically through cross-protocol ordinal ranking on independent data.

---

## 3.3 Datasets

Five independent datasets were used across training and validation stages. Table 3.1 summarises their key characteristics.

**Table 3.1: Summary of datasets used in this study**

| Dataset | Chemistry | Profile Type | Cells/Records | Role |
|---------|-----------|--------------|---------------|------|
| CALCE (UMD) | LiCoO₂ | CC lab | ~40 cells, ~2,000 cycles | Training |
| NASA PCoE (multi-temperature) | LiCoO₂ | CC, multi-temperature | 26 cells, 2,010 cycles | Same-chemistry, multi-temperature validation |
| Oxford Degradation Study | NMC | BMP drive cycle | 6 cells, 72 records | Cross-chemistry, variable-profile validation |
| NREL BLAST-Lite | NMC811/NMC622/LFP/NCA | Simulated fleet duty | 48 conditions, 1,932 checkpoints | Cross-chemistry + cross-profile |
| Severson et al. (2019) | LFP | Multi-step fast charge | 49 cells, 23 protocols | Protocol-level ordinal validation |
| NREL Fleet DNA | Real EV (mixed) | Telematics | 1,412 trips | Real-world KI ground truth |

### 3.3.1 CALCE LiCoO₂ Dataset (Training Source)

The CALCE (Computer Aided Life Cycle Engineering) battery dataset, produced at the University of Maryland, contains per-cycle capacity measurements for LiCoO₂/graphite 18650 cells (nominal capacity 2.0 Ah) subjected to CC cycling at various C-rates and temperatures. All cells were cycled under controlled laboratory conditions with constant-current constant-voltage (CCCV) charging protocols.

FSI features were extracted per cycle using the extraction script `04_Code/extraction/extract_all_datasets.py`. The KI of all CALCE cycles is identically zero because all charging is CC. DoD is computed from measured discharge capacity relative to the nominal 2.0 Ah. T_norm is computed from the recorded temperature channel. The resulting feature matrix forms the training dataset for all ML models.

### 3.3.2 NASA PCoE Multi-Temperature Dataset

The NASA Prognostics Centre of Excellence (PCoE) dataset contains 26 LiCoO₂/graphite cells of the same chemistry as CALCE, cycled under constant-current (CC) protocols across multiple temperatures. This provides a same-chemistry, multi-temperature validation: the model is tested on a temperature distribution that extends beyond the CALCE training range, exercising the T_stress_norm component of the FSI formula.

The CC profile means KI = 0 for all NASA cycles (same as CALCE). The primary stress variation is through temperature, making this the most targeted test of the T_norm component's ability to capture thermal degradation differences across cells cycled at different temperatures.

### 3.3.3 NREL Fleet DNA — Real-World KI Validation

The NREL Fleet DNA database contains high-resolution telematics records from commercial EV fleets across the United States. A subset of 1,412 trip records across three vehicle classes (delivery trucks, transit buses, refuse trucks) was analysed to compute per-trip KI from the recorded current or power profiles.

This dataset is used exclusively for KI validation — it does not contain battery SoH measurements. Its purpose is to confirm that KI values observed in real fleet operation are substantially non-zero and reproducibly higher than laboratory CC baselines. Wilcoxon signed-rank tests were applied to test the null hypothesis KI = 0 for each vehicle class.

### 3.3.4 Oxford NMC Dataset

The Oxford degradation dataset contains NMC/graphite cells subjected to a Battery Motor Profile (BMP) drive cycle — a standardised variable-current discharge profile representing realistic electric vehicle operation. It provides simultaneous cross-chemistry (LiCoO₂ → NMC) and cross-profile (CC → variable BMP) validation. The BMP profile produces KI > 0, making this the only experimental validation dataset where the FSI KI component is directly exercised on real non-CC current data (alongside the Fleet DNA telematics analysis). With 6 cells and 72 cycle records, this dataset is the smallest of the validation set but provides the most operationally realistic current profile.

### 3.3.5 NREL BLAST-Lite

The NREL Battery Lifetime Analysis and Simulation Tool (BLAST-Lite) is a physics-based battery degradation simulation framework supporting NMC811, NMC622, LFP, and NCA chemistries. It was used to generate 1,932 degradation checkpoints across 48 experimental conditions (4 chemistries × 4 duty cycles × 3 temperatures), providing a systematic cross-chemistry and cross-profile validation grid that no single experimental dataset could provide.

FSI features for each BLAST simulation run were computed from the duty-cycle current profiles. Because BLAST generates complete per-timestep current data, KI is computed exactly from the simulation output.

### 3.3.6 Severson et al. (2019) — Protocol-Level Validation

Severson et al. (2019) tested 124 LFP/graphite A123 APR18650M1A cells (1.1 Ah nominal) under 23 distinct multi-step fast-charging protocols at 30°C, recording cycle life until 80% capacity (EoL). The charging protocols vary the first-step C-rate (3.6C to 8C), the SoC cutoff for the first step (15% to 80%), and the second-step C-rate (3.0C to 4.0C). Discharge was identical across all cells (4C to 2.0V).

Because the raw pickle files (batch1.pkl, batch2.pkl) require a specialised build pipeline from 2.82 GB MATLAB source files, FSI features were computed analytically from the published protocol specifications rather than from raw waveform data. This is methodologically valid for the specific validation being performed: the FSI hypothesis requires only per-protocol KI and per-protocol cycle-life, both of which are exactly determinable from the published experimental parameters.

**Analytical KI computation for two-step CC protocols:**

For a two-step charging protocol (C₁ for the first pct₁% of SoC swing, C₂ for the remainder to 80% SoC):

$$w_1 = \frac{pct_1}{80}, \quad w_2 = \frac{80 - pct_1}{80}$$

$$\bar{I} = w_1 \cdot C_1 \cdot I_{1C} + w_2 \cdot C_2 \cdot I_{1C}$$

$$\text{KI} = \frac{\sqrt{w_1(C_1 I_{1C} - \bar{I})^2 + w_2(C_2 I_{1C} - \bar{I})^2}}{\bar{I}}$$

This formula is exact for ideal CC–CC two-step charging, treating the protocol as a two-value discrete current distribution weighted by the fraction of charge time spent at each step.

---

## 3.4 FSI Feature Extraction Pipeline

A consistent Python extraction pipeline was implemented to compute FSI features from each dataset. The pipeline produces a standardised CSV with identical column definitions across all datasets (documented in `03_Processed_Data/DATA_DICTIONARY.csv`).

Per-cycle FSI features are computed as follows:

```
For each cell c, for each cycle n:
  1. Extract current array I(t), voltage array V(t), temperature array T(t)
  2. KI         = std(|I|) / mean(|I|)        [over all timesteps in cycle n]
  3. DoD_%      = (Q_discharge_n / Q_nominal) × 100
  4. T_avg_C    = mean(T(t))
  5. T_stress   = |T_avg_C − 25| / 25
  6. C_peak     = max(|I|) / I_1C
  7. FSI        = 0.30×KI + 0.25×(DoD_%/100) + 0.25×T_stress + 0.20×C_peak
  8. SoH_%      = (Q_discharge_n / Q_discharge_initial) × 100
                  where Q_discharge_initial = median(Q_discharge[cycles 3–10])
  9. Health_Label: Healthy (SoH ≥ 90%), Degraded (80% ≤ SoH < 90%), End_of_Life (SoH < 80%)
```

The master training file `03_Processed_Data/Linked_Lab_Fleet_Degradation.csv` concatenates CALCE, NASA, and Oxford FSI features into 9,031+ labelled cycle records used for XGBoost and Random Forest training.

Secondary FSI features — DCSS (Dynamic Current Stress Signature: std of charging-phase current), RBF (mean charging current magnitude), and CVI (std of voltage) — are extracted alongside the primary FSI components but used only as supplementary diagnostics, not as model inputs.

---

## 3.5 Machine Learning Models

### 3.5.1 XGBoost Regressor

An XGBoost gradient-boosted tree regressor was trained to predict SoH_% from the seven primary FSI features: [FSI, KI, DoD_%, T_stress_norm, DCSS, RBF, CVI]. Hyperparameters (n_estimators = 200, max_depth = 5, learning_rate = 0.05) were selected by 5-fold cross-validation on the CALCE training data. The trained model was serialised to `05_Models/xgb_model.pkl` for reproducibility.

**Rationale for XGBoost:** Gradient-boosted trees are well-suited to tabular regression with heterogeneous feature scales and moderate dataset sizes. They are robust to feature scale differences (no normalisation required), handle non-linear feature interactions naturally, and produce SHAP-compatible output for interpretability.

### 3.5.2 Random Forest Classifier

A Random Forest classifier was trained on the same feature set to predict the three-class health label (Healthy / Degraded / End_of_Life). This provides a complementary view to regression: the classifier output can be used directly for fleet health-status monitoring without requiring exact SoH percentage predictions.

### 3.5.3 SHAP Feature Attribution

SHAP (SHapley Additive exPlanations) TreeExplainer was applied to the trained XGBoost model to derive per-prediction feature attributions. SHAP values quantify the marginal contribution of each feature to each prediction, enabling:

1. **Global importance ranking** — which features consistently drive SoH predictions across all cycles
2. **Chemistry transferability** — whether the feature ranking is preserved when the model is applied to out-of-distribution datasets (BLAST, Severson)
3. **Weight calibration verification** — whether the SHAP-derived importance ordering matches the FSI weight ordering

The SHAP feature rank correlation between CALCE (training) and BLAST (validation) was computed as Spearman's ρ over the mean absolute SHAP values per feature, providing a single scalar measure of model interpretability consistency across chemistry groups.

---

## 3.6 Validation Framework

### 3.6.1 In-Sample Performance

The primary model quality metric on the CALCE training set is RMSE (%) of SoH prediction, evaluated via 5-fold cross-validation to avoid overfitting assessment on the full training set.

### 3.6.2 Cross-Dataset Validation

For each held-out validation dataset, the CALCE-trained XGBoost model is applied directly (no retraining). Performance is reported as:
- **RMSE (%)** — absolute prediction error in SoH percentage points
- **Bias** — mean signed prediction error (positive = systematic overestimation of SoH)
- **Variance** — variance of residuals after bias removal

Bias-variance decomposition is used to distinguish systematic calibration offset (chemistry-specific SoH baseline mismatch, addressable by one-parameter linear rescaling) from structural model error (inability to capture degradation dynamics).

### 3.6.3 Ordinal Ranking Validation (Severson)

For the Severson protocol-level validation, the primary metric is Spearman's rank correlation coefficient ρ between per-protocol FSI (or KI) and negative cycle life (−cycle_life). This tests the ordinal claim: higher FSI → shorter cell lifetime. A statistically significant positive ρ confirms the FSI correctly ranks protocols from least to most damaging, irrespective of the absolute SoH prediction accuracy.

### 3.6.4 KI Significance Testing (Fleet DNA)

For Fleet DNA, a one-sample Wilcoxon signed-rank test against the null hypothesis KI = 0 is applied per vehicle class. This tests whether the population of real-world trip KI values is statistically distinguishable from the lab baseline of zero.

---

## 3.7 Reproducibility

All code, processed data, and validation results are version-controlled in a structured repository:

| Folder | Contents |
|--------|----------|
| `01_Reference_Papers/` | All source PDFs |
| `02_Downloaded_Datasets/` | Raw data (not committed to git due to size) |
| `03_Processed_Data/` | FSI feature CSVs + DATA_DICTIONARY.csv |
| `04_Code/` | Extraction, ML, and validation scripts + JSON results |
| `05_Models/` | Serialised trained models |
| `06_Draft_Update/` | Dissertation chapters, proposals, supervisor updates |
| `07_Weekly_Progress/` | Weekly notes and supervisor meeting records |

The master DATA_DICTIONARY defines every column, formula, unit, range, and dataset provenance, ensuring that all FSI CSV files can be independently verified and reproduced from the raw datasets using the extraction scripts.

---

## References (Chapter 3)

- Attia, P.M. et al. (2020). Closed-loop optimization of fast-charging protocols for batteries with machine learning. *Nature*, 578(7795), pp.397–402.
- Birkl, C.R. et al. (2017). Degradation diagnostics for lithium-ion batteries. *Journal of Power Sources*, 341, pp.373–386.
- Severson, K.A. et al. (2019). Data-driven prediction of battery cycle life before capacity degradation. *Nature Energy*, 4(5), pp.383–391.
- Wang, J. et al. (2020). Cycle-life model for graphite-LiFePO₄ cells. *Journal of Power Sources*, 196(8), pp.3942–3948.
- Lundberg, S.M. and Lee, S.I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.
- Chen, T. and Guestrin, C. (2016). XGBoost: a scalable tree boosting system. *Proceedings of KDD 2016*, pp.785–794.
- NREL Fleet DNA: https://www.nrel.gov/transportation/fleettest-fleet-dna.html
- NREL BLAST-Lite: https://github.com/NREL/BLAST-Lite
- CALCE Battery Dataset: https://calce.umd.edu/battery-data
- NASA PCoE: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
