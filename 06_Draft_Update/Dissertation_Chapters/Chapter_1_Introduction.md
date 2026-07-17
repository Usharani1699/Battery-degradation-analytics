# Chapter 1: Introduction

**Module:** EG7030 — Dissertation  
**Student:** Usha Rani Vamanagiri (u2965962)  
**Programme:** MSc Electric Vehicles and Energy Storage, UEL  

---

## 1.1 Background and Motivation

The electrification of road transport has accelerated dramatically over the past decade, driven by regulatory pressure, declining battery costs, and growing awareness of urban air quality impacts. Commercial electric vehicle (EV) fleets — encompassing delivery vehicles, transit buses, refuse trucks, and logistics vans — represent a rapidly growing segment of this transition. Unlike privately owned passenger EVs, commercial fleet vehicles are subjected to intensive, continuous operation: multiple full charge-discharge cycles per day, heavy loads, frequent stop-start urban driving, and exposure to a wide range of ambient temperatures. The batteries powering these vehicles experience stress conditions fundamentally different from those encountered in laboratory testing or even typical passenger car use.

The economic viability of commercial EV fleets depends critically on accurate battery State-of-Health (SoH) prediction. SoH — defined as the ratio of current discharge capacity to rated nominal capacity — determines remaining useful life, maintenance scheduling requirements, and second-life repurposing potential. Inaccurate SoH estimates lead directly to operational disruptions: premature battery replacement (unnecessary capital expenditure), unexpected mid-route failures (operational risk), and suboptimal charging strategies (accelerated degradation).

Despite the practical urgency of this challenge, current SoH prediction methods suffer from a fundamental methodological gap: they are developed and validated using laboratory battery datasets, yet deployed in real-world fleet environments whose operating conditions differ substantially from the laboratory. This gap is the central motivation for the work presented in this dissertation.

---

## 1.2 The Laboratory–Fleet Gap

Laboratory battery testing employs constant-current (CC) cycling protocols: a fixed current is applied during both charging and discharging, producing a perfectly uniform current waveform. This approach enables controlled, reproducible experiments and is the foundation of virtually all published battery degradation datasets (CALCE, NASA PCoE, Oxford, Severson, and others). However, CC cycling produces a current profile that is categorically different from what a fleet battery experiences.

A real EV fleet vehicle's current profile is shaped by:
- **Acceleration events**: high instantaneous discharge currents lasting seconds to tens of seconds
- **Regenerative braking**: polarity reversal — current switches from discharge to charge mid-trip
- **Stop-start urban operation**: repeated transitions between near-zero and full-current states
- **Route-dependent variation**: different trips produce different current statistical distributions

The result is a highly dynamic, variable current profile whose statistical properties — particularly the coefficient of variation of current magnitude — differ fundamentally from any CC laboratory protocol.

This dissertation introduces and quantifies this distinction through a new metric: **Kinetic Intensity (KI)**, defined as the coefficient of variation of the absolute current magnitude over a cycle:

$$\text{KI} = \frac{\sigma(|I|)}{\mu(|I|)}$$

KI = 0 by definition for any constant-current protocol, regardless of C-rate. Analysis of 1,412 NREL Fleet DNA commercial vehicle trip records (presented in Chapter 4) confirms that real fleet KI values range from 0.60 (delivery trucks) to 0.81 (refuse trucks) — a categorical difference from laboratory CC cycling that has not previously been operationalised as a model input.

---

## 1.3 Research Gap

Prior work on data-driven battery SoH prediction has developed sophisticated models — long short-term memory networks, Gaussian process regression, gradient-boosted trees — that achieve high accuracy on held-out test data from the same laboratory source as training data. However, cross-dataset generalisation across different current profiles and chemistries remains an open problem.

Several specific gaps are identified in the literature:

1. **No standard feature encoding current variability.** Published FSI-adjacent features (DoD, C-rate, temperature, cycle count) do not include a coefficient of variation of current. The implicit assumption is that C-rate magnitude is a sufficient proxy for current stress — yet two protocols with identical mean C-rate but different variability patterns produce different degradation trajectories (demonstrated in Chapter 4 via the Severson et al. dataset).

2. **No quantitative bridge between lab and fleet current statistics.** Fleet DNA databases exist and contain high-resolution telematics data, but prior work has not systematically extracted KI from these records and compared it to laboratory protocol statistics.

3. **Cross-chemistry calibration gap not decomposed.** When models trained on one chemistry are applied to another, reported performance drops are typically attributed to "chemistry mismatch" without decomposing how much error is structural (the model fails to capture degradation dynamics) versus calibration (the absolute SoH scale differs between chemistries). The bias-variance decomposition presented in Chapter 4 addresses this gap.

---

## 1.4 Research Question and Objectives

### Primary Research Question

> *Can a composite index encoding kinetic current variability, thermal stress, depth of discharge, and peak C-rate — derived from both laboratory battery datasets and real-world fleet telematics — improve SoH prediction accuracy and provide early degradation warning for commercial EV fleets?*

### Research Objectives

1. **Define and derive** the Fleet Stress Index (FSI) from physical first principles, with weights grounded in literature and validated through cross-protocol empirical testing.

2. **Extract and process** FSI features from five independent battery datasets (CALCE, NASA PCoE, Oxford, NREL BLAST-Lite, Severson et al. 2019) spanning three chemistries, four current profile types, and three temperature ranges.

3. **Quantify the laboratory–fleet KI gap** using NREL Fleet DNA commercial vehicle telematics data, establishing the empirical basis for prioritising KI in the FSI formula.

4. **Train and validate** XGBoost regression and Random Forest classification models on CALCE FSI features, evaluating cross-dataset transferability using systematic bias-variance decomposition.

5. **Validate the FSI ordinal hypothesis** — that higher FSI predicts shorter cell lifetime — using the Severson et al. (2019) multi-protocol dataset as an independent ordinal test.

6. **Benchmark the FSI framework against physics-based simulation** using a PyBaMM Single Particle Model with Electrolyte (SPMe) and EC-reaction limited SEI degradation.

7. **Identify and honestly characterise limitations**, including the T_norm symmetry assumption, cross-chemistry calibration requirements, and the absence of real fleet battery SoH labels.

---

## 1.5 Scope and Boundaries

This dissertation is scoped as a feature engineering and model validation study, not a deployment engineering study. Specifically:

**In scope:**
- FSI formula derivation and weight justification
- Per-cycle feature extraction from public battery datasets
- ML model training and multi-dataset validation
- Physics-based benchmark comparison (PyBaMM SPMe)
- Honest characterisation of limitations and generalisation boundaries

**Out of scope:**
- Real-time BMS hardware implementation
- Prospective instrumented fleet trials with SoH measurement
- Online learning or adaptive calibration systems
- Economic optimisation of maintenance scheduling

The validation pipeline uses exclusively publicly available datasets. No proprietary fleet battery data was accessed.

---

## 1.6 Dissertation Structure

The remainder of this dissertation is organised as follows:

- **Chapter 2 — Literature Review**: Reviews published battery degradation models, stress feature engineering approaches, and cross-dataset generalisation challenges. Identifies the specific gap that the FSI framework addresses.

- **Chapter 3 — Methodology**: Derives the FSI formula and its components, describes the five datasets and extraction pipeline, and specifies the ML training and validation protocol.

- **Chapter 4 — Results**: Presents results from all validation stages — in-sample performance, SHAP attribution, Fleet DNA KI analysis, cross-dataset validation (NASA, Oxford, BLAST), Severson ordinal ranking, PyBaMM SPMe benchmark, and weight sensitivity analysis.

- **Chapter 5 — Discussion**: Interprets results in the context of the research question, addresses the calibration gap, evaluates limitations with supporting evidence, and considers practical implications for fleet BMS deployment.

---

## References (Chapter 1)

- Severson, K.A. et al. (2019). Data-driven prediction of battery cycle life before capacity degradation. *Nature Energy*, 4(5), pp.383–391.
- Attia, P.M. et al. (2020). Closed-loop optimization of fast-charging protocols for batteries with machine learning. *Nature*, 578(7795), pp.397–402.
- Wang, J. et al. (2020). Cycle-life model for graphite-LiFePO₄ cells. *Journal of Power Sources*, 196(8), pp.3942–3948.
- NREL Fleet DNA Database: https://www.nrel.gov/transportation/fleettest-fleet-dna.html
- CALCE Battery Dataset: https://calce.umd.edu/battery-data
- NASA PCoE Prognostic Data Repository: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
