# Chapter 6: Conclusions

**Module:** EG7030 — Dissertation  
**Student:** Usha Rani Vamanagiri (u2965962)  
**Programme:** MSc Electric Vehicles and Energy Storage, UEL  

---

## 6.1 Summary of Contributions

This dissertation introduced and validated the **Fleet Stress Index (FSI)** as a physically interpretable composite feature for battery State-of-Health prediction in commercial electric vehicle fleets. The primary contributions are:

**1. Kinetic Intensity (KI) as a diagnostic feature.** KI — the coefficient of variation of absolute cycle current — is proposed and validated as the first generalised, dataset-agnostic metric for quantifying the gap between laboratory CC cycling and real fleet operation. Analysis of 1,412 NREL Fleet DNA trip records confirms that real fleet KI (0.60–0.81) is categorically separated from laboratory CC KI (0.000), with Wilcoxon p < 0.001 across all vehicle classes.

**2. FSI formula with physically justified weights.** The composite FSI (KI × 0.30, DoD × 0.25, T_norm × 0.25, C_peak × 0.20) was derived from physical first principles and literature precedent, with the critical caveat that CALCE-based gradient optimisation is degenerate for KI (all training data has KI = 0). Weights were validated through independent ordinal ranking on the Severson et al. multi-protocol LFP dataset: Spearman ρ = 0.807 for FSI versus ρ = 0.744 for KI alone.

**3. Bias-variance decomposition of cross-chemistry transfer error.** BLAST validation RMSE (20.44%) was decomposed into structural bias (18.99%, 84%) and variance (0.48%, 16%), establishing that cross-chemistry performance gaps are primarily calibration failures rather than model structural failures. This finding has direct implications for fleet BMS deployment: a single early-cycle calibration observation can correct most of the bias.

**4. Cross-protocol ordinal validation as a non-circular KI test.** The Severson dataset provides the only validation where KI genuinely varies across training examples (different fast-charge protocols produce different KI). The FSI ρ = 0.807 on this independent dataset validates the KI weight's role in a setting where CALCE training cannot have created spurious KI correlations.

**5. PyBaMM SPMe physics-based benchmark.** The first direct comparison between FSI-based XGBoost prediction and physics-based SPMe simulation confirms that the within-cell FSI trajectory (ρ = −1.000) correctly tracks simulated SoH decline, with XGBoost RMSE = 4.807% against the SPMe reference.

---

## 6.2 Answers to Research Objectives

| Objective | Outcome |
|-----------|---------|
| 1. Derive FSI from physical principles | ✅ Done — weights justified via literature + Severson validation |
| 2. Extract FSI from 5 independent datasets | ✅ Done — CALCE, NASA, Oxford, BLAST, Severson, Fleet DNA |
| 3. Quantify lab–fleet KI gap via Fleet DNA | ✅ Done — KI = 0 (lab) vs 0.60–0.81 (fleet), Wilcoxon p<0.001 |
| 4. Train XGBoost + validate with bias-variance decomposition | ✅ Done — BLAST: 84% bias, 16% variance |
| 5. Validate FSI ordinal hypothesis on Severson | ✅ Done — ρ = 0.807, p < 0.001 |
| 6. PyBaMM SPMe benchmark comparison | ✅ Done — RMSE = 4.807%, ρ(FSI, −SoH) = −1.000 |
| 7. Characterise limitations honestly | ✅ Done — T_norm symmetry, collinearity, no fleet SoH labels |

---

## 6.3 Primary Research Question — Answered

> *Can a composite index encoding kinetic current variability, thermal stress, depth of discharge, and peak C-rate improve SoH prediction accuracy and provide early degradation warning for commercial EV fleets?*

**Within-chemistry (CALCE):** Yes. RMSE = 3.73%, R² = 0.984 demonstrates that the FSI feature set provides near-EIS-precision SoH prediction on the training chemistry, with SHAP confirming FSI as the dominant predictor (82.4% importance).

**Cross-chemistry:** Partially. The FSI framework correctly ranks protocol severity (Severson ρ = 0.807) and identifies the structural direction of cross-chemistry bias (84% of BLAST RMSE is systematic calibration offset). However, uncalibrated deployment on different chemistries produces R² < 0 (negative — worse than a mean predictor), confirming that FSI alone cannot substitute for chemistry-specific calibration.

**For early degradation warning:** The Fleet DNA analysis establishes that real fleet batteries are subjected to KI conditions (0.60–0.81) that laboratory models have never encountered in training. A BMS incorporating FSI can flag cells operating with high-KI profiles as elevated-risk, even without direct SoH labels, supporting proactive rather than reactive maintenance scheduling.

---

## 6.4 Limitations and Honest Assessment

The following limitations must be stated for a complete assessment of the FSI framework's validity:

**L1 — No real fleet SoH labels.** The Fleet DNA analysis characterises current statistics from telematics data but contains no direct battery SoH measurements from fleet vehicles. The connection between fleet KI = 0.60–0.81 and actual capacity fade in those vehicles is inferred from laboratory degradation literature and ordinal validation on controlled Severson protocols — not measured directly. Closing this gap requires a future study with instrumented fleet batteries.

**L2 — T_norm symmetry is physically incorrect.** The symmetric formulation treats −25°C as equivalent to +75°C (same T_norm = 2.0), contradicting Arrhenius kinetics where hot temperatures accelerate SEI growth faster than cold temperatures slow it. A future asymmetric T_norm (e.g. exponential weighting for T > 25°C) would better reflect the underlying physics, particularly for high-temperature fleet routes.

**L3 — Feature collinearity in the ML feature set.** The XGBoost input includes FSI as well as two of its constituent components (KI and T_stress_norm). This creates redundancy that inflates FSI's apparent SHAP importance. On CALCE data this is harmless (KI=0 everywhere), but on variable-protocol data FSI's 82.4% SHAP reflects partly circular attribution. Future work should evaluate whether using components only (without composite FSI) produces equivalent or better performance.

**L4 — PyBaMM timescale mismatch.** The Chen2020 default SEI parameters produce EoL (80% SoH) in approximately 50 cycles for NMC/graphite at 1C, 25°C — substantially faster than real-world NMC degradation (typically 500–1000+ cycles). The SPMe benchmark confirms the correct qualitative behaviour (FSI tracks SoH decline with ρ = −1.000) but should not be used for absolute cycle-life prediction without re-parameterisation.

**L5 — Oxford dataset is too small for statistical conclusions.** Six cells, 72 cycle records, R² = −0.57. While the RMSE = 4.748% result is reported faithfully, any conclusions drawn from this dataset must be understood as exploratory rather than statistically definitive.

---

## 6.5 Recommendations for Future Work

1. **Asymmetric T_norm.** Replace the symmetric absolute-deviation formula with a segmented function: linear or exponential for T > 25°C (Arrhenius acceleration), reduced weight for T < 25°C (lithium plating risk, but slower thermal degradation). This could be validated using the NASA multi-temperature dataset (which spans 4°C–45°C operating conditions).

2. **Feature set redesign.** Retrain XGBoost with either FSI alone or component features alone (not both), to produce unambiguous SHAP attributions. A systematic ablation study (FSI-only, components-only, combined) would quantify whether the composite adds value beyond its parts.

3. **Instrumented fleet validation.** Partner with a commercial EV fleet operator to collect paired telematics (KI, T_avg, DoD) and periodic capacity discharge tests across the vehicle lifecycle. This would close the most critical validation gap: directly demonstrating that fleet KI predicts in-field capacity fade.

4. **PyBaMM re-parameterisation.** Use experimentally measured SEI growth rate constants for the specific cell chemistry (NMC or LFP) to produce realistic cycle-life predictions from the SPMe model. The re-parameterised model would then serve as a more credible physics-based benchmark for the XGBoost predictions.

5. **Confidence intervals.** Apply bootstrap resampling or Bayesian inference to quantify uncertainty on all key metrics (RMSE, ρ, R²). This is particularly important for the Oxford and Severson results where sample sizes are small.

6. **Online adaptive calibration.** Develop and test an online learning variant of the FSI model that updates the intercept calibration term from streaming fleet telemetry, reducing the systematic bias component identified in the BLAST analysis without requiring full model retraining.

---

## 6.6 Closing Remarks

The FSI framework demonstrates that the laboratory–fleet KI gap is not merely a philosophical concern but a quantifiable, empirically grounded discrepancy (KI: 0.000 laboratory vs. 0.674 fleet, n=1,412 trips) with measurable consequences for model cross-generalisability. The ordinal validation on Severson multi-protocol data (ρ = 0.807) provides the strongest evidence that KI is a genuine stress predictor independent of CALCE training artefacts.

The framework is not ready for uncalibrated fleet BMS deployment — L1 through L4 above must be addressed first. What it does establish is the necessity and feasibility of including current variability as an explicit stress feature in future battery health models. Models trained exclusively on CC laboratory data are structurally blind to the primary stress mechanism distinguishing fleet batteries from laboratory test cells. The FSI provides a vocabulary and a validation pathway for addressing that blindness.

---

## References (Chapter 6)

- Attia, P.M. et al. (2020). Closed-loop optimization of fast-charging protocols for batteries with machine learning. *Nature*, 578(7795), pp.397–402.
- Severson, K.A. et al. (2019). Data-driven prediction of battery cycle life before capacity degradation. *Nature Energy*, 4(5), pp.383–391.
- Millner, A. (2010). Modeling lithium ion battery degradation in electric vehicles. *IEEE Conference on Innovative Technologies for an Efficient and Reliable Electricity Supply*, pp.349–356.
- Sulzer, V. et al. (2021). Python Battery Mathematical Modelling (PyBaMM). *Journal of Open Research Software*, 9(1).
- NREL Fleet DNA Database. Available at: https://www.nrel.gov/transportation/fleettest-fleet-dna.html
