# Abstract

**Title:** Fleet Stress Index: A Composite Feature Engineering Framework for Battery State-of-Health Prediction in Commercial Electric Vehicle Fleets

**Student:** Usha Rani Vamanagiri (u2965962)  
**Programme:** MSc Electric Vehicles and Energy Storage, University of East London  
**Module:** EG7030 — Dissertation  

---

Commercial electric vehicle fleets operate under current profiles fundamentally different from laboratory constant-current cycling: acceleration events, regenerative braking, and stop-start urban operation produce current variability that no published battery degradation dataset captures. This dissertation introduces the **Fleet Stress Index (FSI)**, a physically motivated composite feature defined as FSI = 0.30×KI + 0.25×DoD + 0.25×T_norm + 0.20×C_peak,norm, where **Kinetic Intensity (KI)** — the coefficient of variation of absolute current — is the primary differentiating dimension.

Analysis of 1,412 NREL Fleet DNA commercial vehicle trip records confirms that real fleet KI ranges from 0.60 (delivery trucks) to 0.81 (refuse trucks), compared to KI = 0 by definition for all laboratory CC protocols. XGBoost regression trained on CALCE LiCoO₂ FSI features achieves RMSE = 3.73%, R² = 0.984 in-sample. Cross-dataset validation reveals RMSE = 4.748% on Oxford NMC drive-cycle data and RMSE = 15.295% on NASA LiCoO₂ multi-temperature data, with bias-variance decomposition showing the BLAST validation error is 84% structural bias and 16% variance — indicating that calibration errors dominate cross-chemistry transfer.

The FSI ordinal hypothesis — that higher FSI predicts shorter cell lifetime — is validated on the Severson et al. (2019) multi-protocol LFP dataset: Spearman ρ(FSI, −cycle_life) = 0.807 (p < 0.001), outperforming KI alone (ρ = 0.744). PyBaMM SPMe physics-based benchmark confirms FSI-SoH anti-correlation (ρ = −1.000 within-cell) with XGBoost achieving RMSE = 4.807% versus the simulated trajectory.

Key limitations include: T_norm symmetry (treating hot and cold as equivalent); feature collinearity between composite FSI and its component inputs; the absence of real fleet battery SoH labels for direct validation; and PyBaMM's accelerated degradation timescale (Chen2020 default SEI parameters produce EoL in 50 cycles versus real-world 500–1000+ cycles). These limitations define the boundary conditions for practical deployment of the FSI framework in fleet battery management systems.

**Keywords:** Battery degradation, State-of-Health, Electric vehicle fleet, Feature engineering, SHAP, XGBoost, PyBaMM, Kinetic Intensity, Cross-dataset validation

**Word count (chapters 1–5):** ~13,000 words
