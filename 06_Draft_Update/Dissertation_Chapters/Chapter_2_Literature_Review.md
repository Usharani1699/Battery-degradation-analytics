# Chapter 2: Literature Review

**Module:** EG7030 — Dissertation  
**Student:** Usha Rani Vamanagiri (u2965962)  
**Programme:** MSc Electric Vehicles and Energy Storage, UEL  

---

## 2.1 Overview

This chapter reviews the literature relevant to battery State-of-Health prediction, feature engineering for degradation modelling, and cross-dataset generalisation in battery machine learning. The review is structured to identify the specific gap that the Fleet Stress Index (FSI) addresses: the absence of a physically grounded composite feature that bridges laboratory CC testing conditions and real-world variable-current fleet operation.

---

## 2.2 Battery Degradation Mechanisms

### 2.2.1 Capacity Fade in Lithium-Ion Cells

Lithium-ion battery capacity fade arises from several concurrent electrochemical mechanisms whose relative contributions depend on operating conditions (Birkl et al., 2017; Reniers et al., 2019):

**Solid Electrolyte Interface (SEI) growth** occurs primarily at the graphite anode, where electrolyte solvents react with lithium at low potentials to form a passivation layer. SEI growth is thermally activated (Arrhenius kinetics) and consumes cyclable lithium irreversibly. It is the dominant capacity-fade mechanism at moderate temperatures and rates, and is the mechanism modelled in the PyBaMM SPMe simulation (Chapter 4).

**Lithium plating** occurs when lithium deposition on the anode surface is kinetically preferred over intercalation — a condition favoured by high charge rates (high C-rate), low temperatures, and deep states of charge. Plated lithium may be partially reversible (stripping) but a portion becomes electrochemically inactive ("dead lithium"), permanently reducing capacity. High-KI profiles with frequent high-current spikes are particularly conducive to transient lithium plating (Attia et al., 2020).

**Particle cracking and mechanical fatigue** result from the volumetric expansion and contraction of electrode particles during lithiation and delithiation. Repeated cycling causes fatigue-crack propagation, increasing tortuosity and reducing accessible surface area. High DoD cycling maximises the volume swing per cycle and therefore accelerates mechanical fatigue.

**Electrolyte decomposition** is accelerated at elevated temperatures and high voltages, producing resistive films on particle surfaces and consuming electrolyte, increasing cell impedance and reducing power delivery.

### 2.2.2 Temperature Dependence — Arrhenius Kinetics

All electrochemical reaction rates follow Arrhenius temperature dependence:

$$k(T) = A \cdot e^{-E_a / RT}$$

where Eₐ is the activation energy and R is the gas constant. For SEI growth, typical Eₐ values in the literature are 50–70 kJ/mol (Ecker et al., 2012), implying that a 10°C increase in temperature approximately doubles the SEI growth rate (Q10 ≈ 2). Cold temperatures slow SEI growth but accelerate lithium plating during charging — these are directionally asymmetric effects, motivating the discussion of the T_norm symmetry limitation in Chapter 5.

---

## 2.3 State-of-Health Estimation Methods

### 2.3.1 Model-Based Methods

Electrochemical models — from the Doyle–Fuller–Newman (DFN) model to the Single Particle Model (SPM) and its electrolyte-enhanced variant (SPMe) — simulate degradation from first principles (Doyle et al., 1993; Plett, 2015). These models provide physically interpretable degradation trajectories and can be parameterised for specific cell chemistries. However, they require cell-specific parameterisation, high-resolution measurement inputs, and substantial computational resources — impractical for real-time BMS deployment across large heterogeneous fleets.

Equivalent Circuit Models (ECMs) offer a computationally tractable alternative, representing the cell as a resistor-capacitor network whose parameters are estimated from impedance measurements (Plett, 2015). ECMs can be deployed on embedded BMS hardware but require periodic recalibration and do not extrapolate well across operating conditions not observed during parameterisation.

### 2.3.2 Data-Driven Methods

Data-driven approaches bypass explicit physical modelling by learning the SoH–feature mapping directly from measured data. Key published approaches include:

**Feature engineering from discharge curves.** Severson et al. (2019) demonstrated that features extracted from early-cycle discharge voltage curves (particularly the variance of the discharge voltage difference between cycle 2 and cycle 100) predict full cycle life with test RMSE of 9.1% using a regularised linear model. This approach requires only discharge curve measurements but depends on access to early-cycle data — a constraint that limits its applicability to in-service fleet batteries that did not have standardised early-cycle measurements recorded.

**LSTM and recurrent models.** Zhang et al. (2018) applied Long Short-Term Memory (LSTM) networks to per-cycle charge/discharge sequences, achieving RMSE < 1% on held-out test cells from the same laboratory source. However, LSTMs require sequence data of consistent length and temporal resolution — conditions rarely met by heterogeneous fleet telemetry systems that record at variable sample rates.

**Gradient-boosted trees.** XGBoost and Random Forest regressors applied to per-cycle aggregate features (capacity, energy, internal resistance) have shown RMSE of 3–8% within a single dataset (Richardson et al., 2019). These models are interpretable, computationally efficient, and compatible with tabular BMS telemetry data — motivating their use in this dissertation.

**Gaussian process regression.** Richardson et al. (2017) applied GP regression to capacity trajectories, providing prediction intervals alongside point estimates. GPs are well-suited to small datasets but scale poorly to the multi-thousand-cycle records in fleet applications.

### 2.3.3 Limitations of Existing Methods

A common limitation across all published data-driven approaches is their reliance on training data from a single laboratory source, typically constant-current cycling. Cross-dataset evaluation — applying a model trained on one dataset to a different chemistry, protocol, or temperature — is rarely reported, and when reported, performance typically degrades substantially (Bhatt et al., 2022; Paulson et al., 2022). The specific contribution of current profile variability (KI) to this cross-dataset performance gap has not been isolated in prior work.

---

## 2.4 Stress Feature Engineering

### 2.4.1 Published Stress-Based Cycle Life Models

Empirical cycle-life models in the battery literature typically incorporate a limited set of stress factors:

**Arrhenius-weighted temperature models** (Ecker et al., 2012; Calendar ageing models) express capacity fade rate as:

$$Q_{\text{loss}} = A \cdot e^{-E_a/RT} \cdot f(C_{\text{rate}}, \text{DoD}, \text{SoC})$$

where f(·) is a power-law or polynomial function of cycling parameters. These models are chemistry-specific and typically calibrated on CC test data, making them inapplicable to variable-current profiles.

**DoD-focused models.** Millner (2010) and Plett (2015) emphasise DoD as the primary cycle-life driver in lithium iron phosphate batteries, with shallower cycling significantly extending cycle life. These models do not include current variability.

**C-rate stress models.** Ecker et al. (2012) show that increasing charge C-rate accelerates capacity fade through enhanced lithium plating and internal resistance growth. However, C-rate magnitude is treated as a scalar — the variability of current *within* a cycle is not considered.

### 2.4.2 The Missing Dimension: Current Variability

No published composite feature explicitly encodes the within-cycle coefficient of variation of current as a stress component. The closest published metrics are:

- **Dynamic stress test (DST) vs CC comparison** (Anseán et al., 2016): cells cycled under dynamic current profiles degrade faster than CC-cycled cells at the same mean C-rate, confirming that current variability per se (not just C-rate magnitude) is a stress driver. However, DST comparisons are protocol-specific and do not produce a generalised scalar feature.

- **Current ripple in fast-charging** (Amanor-Boadu et al., 2018): high-frequency current ripple from power converter switching is associated with accelerated degradation. This is a specific high-frequency case of current variability, not the broader statistical variability captured by KI.

- **Discharge variability in standardised cycles** (Saxena et al., 2016): drive-cycle tests (UDDS, HWFET, LA92) produce different current distributions and different degradation rates from CC. The ratio of standard deviation to mean current in these cycles is conceptually equivalent to KI but was not extracted or used as a model feature.

This review confirms that KI — the coefficient of variation of current magnitude — has not previously been proposed as a generalised, dataset-agnostic stress feature for SoH prediction models.

### 2.4.3 Composite Stress Indices in Adjacent Fields

Composite stress indices are standard tools in other reliability engineering domains. The Palmgren-Miner rule for mechanical fatigue cumulates damage fractions across load amplitudes. The Rain-flow counting algorithm in vibration fatigue analysis decomposes complex load sequences into equivalent simple cycles. The closest battery analogue is the coulomb-counting approach to SoH tracking (integrating charge throughput) — but this is a single-dimensional counter, not a composite multi-physical index. The FSI framework applies the composite index concept to battery electrochemistry for the first time, combining four physically distinct stress dimensions into a single predictive feature.

---

## 2.5 Cross-Dataset Generalisation

### 2.5.1 Chemistry Transferability

Battery ML models trained on one chemistry and applied to another typically show substantial RMSE increases. Bhatt et al. (2022) survey cross-chemistry transfer learning approaches and report that domain adaptation methods (instance reweighting, feature alignment) can recover 40–60% of the in-domain performance, but require some labelled data from the target chemistry. The per-chemistry calibration approach used in Chapter 4 (single intercept shift from early-cycle observations) is consistent with this literature as a minimal-data adaptation strategy.

### 2.5.2 Profile Transferability

Cross-profile generalisation — from CC to variable-current — is less studied than cross-chemistry transfer. The fundamental challenge identified by this dissertation is that models trained on CC data have KI = 0 for all training examples, meaning they cannot have learned the relationship between KI and SoH from training data alone. The FSI framework addresses this by making KI an explicit input feature whose weight is set from physical reasoning and cross-protocol validation (Severson ordinal test) rather than from CALCE training data regression.

### 2.5.3 Interpretability and SHAP

SHAP (SHapley Additive exPlanations), introduced by Lundberg and Lee (2017), provides a game-theoretic framework for attributing model predictions to individual features. For battery ML models, SHAP has been used to verify that physically meaningful features dominate model decisions (Yang et al., 2021; Li et al., 2022). This dissertation uses SHAP in two roles: (1) to verify internal consistency between FSI weights and model-attributed importance on CALCE training data; and (2) to assess whether feature importance rankings transfer across chemistries (SHAP rank correlation between CALCE and BLAST validation sets).

---

## 2.6 PyBaMM and Physics-Based Benchmarking

The Python Battery Mathematical Modelling (PyBaMM) framework (Sulzer et al., 2021) provides open-source implementations of electrochemical battery models ranging from the simple SPM to the full DFN model, with validated parameter sets for commercial cell chemistries including NMC/graphite (Chen2020). PyBaMM has been used as a benchmark tool for comparing data-driven models with physics-based predictions (Timms et al., 2021; Marquis et al., 2019).

The SPMe model used in this dissertation adds electrolyte concentration dynamics to the SPM, improving accuracy for moderate-to-high C-rates and multi-cycle simulations. The EC-reaction limited SEI submodel represents anode SEI growth as a rate-limited reduction of ethylene carbonate (EC) solvent, producing realistic capacity fade trajectories without requiring parameterisation beyond the Chen2020 default set (Marquis et al., 2019).

---

## 2.7 Summary and Research Gap

The reviewed literature reveals that:

1. **Battery degradation is well understood mechanistically**, with SEI growth, lithium plating, and mechanical fatigue as the primary capacity-fade drivers — all accelerated by current variability, high temperatures, and deep cycling.

2. **Data-driven SoH models achieve high accuracy within a single dataset** but transfer poorly across different current profiles, chemistries, and temperatures. Cross-dataset performance gaps are reported but not systematically decomposed into structural versus calibration components.

3. **No published composite feature explicitly encodes current variability** (coefficient of variation of |I|) as a stress dimension. C-rate magnitude is used as a proxy, but within-cycle variability — which distinguishes fleet operation from CC lab cycling — is not captured.

4. **Real-world fleet current statistics have not been linked quantitatively to laboratory protocol statistics** in the degradation modelling literature. Fleet DNA databases exist but have not been mined for KI statistics in connection with battery SoH models.

The Fleet Stress Index framework developed in this dissertation directly addresses gaps 3 and 4, using KI as the primary feature distinguishing fleet operation from laboratory baselines, and providing the first quantitative measurement of this gap using NREL Fleet DNA commercial vehicle trip data.

---

## References (Chapter 2)

- Anseán, D. et al. (2016). Fast charging technique for high power lithium iron phosphate batteries: a mechanistic analysis of aging. *Journal of Power Sources*, 321, pp.201–209.
- Attia, P.M. et al. (2020). Closed-loop optimization of fast-charging protocols for batteries with machine learning. *Nature*, 578(7795), pp.397–402.
- Bhatt, A. et al. (2022). Machine learning for battery degradation prediction: a review. *Journal of The Electrochemical Society*, 169(6).
- Birkl, C.R. et al. (2017). Degradation diagnostics for lithium-ion batteries. *Journal of Power Sources*, 341, pp.373–386.
- Chen, C.H. et al. (2020). Development of experimental techniques for parameterization of multi-scale lithium-ion battery models. *Journal of The Electrochemical Society*, 167(8).
- Doyle, M., Fuller, T.F. and Newman, J. (1993). Modeling of galvanostatic charge and discharge of the lithium/polymer/insertion cell. *Journal of The Electrochemical Society*, 140(6), pp.1526–1533.
- Ecker, M. et al. (2012). Development of a lifetime prediction model for lithium-ion batteries based on extended accelerated aging test data. *Journal of Power Sources*, 215, pp.248–257.
- Li, W. et al. (2022). Interpretable machine learning model for battery degradation. *Applied Energy*, 316, 119093.
- Lundberg, S.M. and Lee, S.I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.
- Marquis, S.G. et al. (2019). An asymptotic derivation of a single particle model with electrolyte. *Journal of The Electrochemical Society*, 166(15).
- Millner, A. (2010). Modeling lithium ion battery degradation in electric vehicles. *IEEE Conference on Innovative Technologies for an Efficient and Reliable Electricity Supply*, pp.349–356.
- Paulson, N.H. et al. (2022). Feature engineering for machine learning enabled early prediction of battery lifetime. *Journal of Power Sources*, 527, 231127.
- Plett, G.L. (2015). *Battery Management Systems, Volume I: Battery Modeling*. Artech House.
- Reniers, J.M., Mulder, G. and Howey, D.A. (2019). Review and performance comparison of mechanical-chemical degradation models for lithium-ion batteries. *Journal of The Electrochemical Society*, 166(14).
- Richardson, R.R. et al. (2017). Gaussian process regression for in situ capacity estimation of lithium-ion batteries. *IEEE Transactions on Industrial Informatics*, 15(1), pp.127–138.
- Richardson, R.R. et al. (2019). Battery health prediction under generalized conditions using a Gaussian process transition model. *Journal of Energy Storage*, 23, pp.320–328.
- Saxena, S. et al. (2016). Quantifying EV battery end-of-life through analysis of travel needs with vehicle powertrain models. *Journal of Power Sources*, 282, pp.265–276.
- Severson, K.A. et al. (2019). Data-driven prediction of battery cycle life before capacity degradation. *Nature Energy*, 4(5), pp.383–391.
- Sulzer, V. et al. (2021). Python Battery Mathematical Modelling (PyBaMM). *Journal of Open Research Software*, 9(1).
- Timms, R. et al. (2021). Asymptotic reduction of a lithium-ion pouch cell model. *SIAM Journal on Applied Mathematics*, 81(3), pp.765–788.
- Yang, J. et al. (2021). Interpretable machine learning for battery capacities prediction and coating process parameters analysis. *Control Engineering Practice*, 107, 104679.
- Zhang, Y. et al. (2018). Remaining useful life prediction for lithium-ion batteries based on exponential model and particle filter. *IEEE Access*, 6, pp.17729–17740.
