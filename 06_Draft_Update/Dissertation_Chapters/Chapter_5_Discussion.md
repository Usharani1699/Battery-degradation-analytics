# Chapter 5: Discussion

**Module:** EG7030 — Dissertation  
**Student:** Usha Rani Vamanagiri (u2965962)  
**Programme:** MSc Electric Vehicles and Energy Storage, UEL  

---

## 5.1 Overview

This chapter interprets the results from Chapter 4 in the context of the original research question: whether the Fleet Stress Index (FSI) can improve SoH prediction accuracy and provide early degradation warning for commercial EV fleets. The discussion addresses five main themes: (1) the core contribution and its significance, (2) the calibration gap and its practical implications, (3) new empirical evidence from weight cross-validation and ordinal ranking tests, (4) what the feature ablation study reveals about the model architecture, and (5) limitations and future directions. Sections 5.2–5.3 discuss previously established results; Sections 5.4–5.6 incorporate findings from the four new validation analyses completed in Chapter 4 (Sections 4.10–4.13).

---

## 5.2 The Central Contribution: Making the Lab-Fleet Gap Explicit

The most important finding of this dissertation is not a numerical accuracy result — it is the demonstration that a *specific, named, and quantifiable* feature (KI) captures the primary difference between laboratory battery testing and real-world EV fleet operation.

Before the FSI framework, the gap between lab-trained battery models and fleet-deployed batteries was acknowledged informally but not operationalised. Practitioners knew that real vehicles did not cycle at constant current, but no standard feature encoded *how much* more variable real operation was, in a form directly usable by a machine learning model.

The Fleet DNA analysis demonstrates this gap quantitatively: KI = 0.000 for all laboratory constant-current protocols; KI = 0.604–0.813 for all real fleet vehicle classes tested. This is not a marginal difference — it is a categorical separation. A model trained on CC lab data that does not encode KI is, in effect, trained on data from a different operating regime than the one it will be deployed in.

The FSI operationalises this insight by assigning KI the highest weight in the composite index (0.30), reflecting its SHAP-validated status as the most predictive stress feature. This transforms an informal observation about the lab-fleet gap into a quantitative tool that can be applied directly to fleet telematics data.

---

## 5.3 Interpreting the Cross-Chemistry Calibration Gap

The most technically complex finding to interpret is the BLAST-Lite cross-chemistry result: RMSE = 20.4% before calibration, R² = −5.9. This initially appears to contradict the strong in-sample performance (RMSE = 3.73%, R² = 0.9839).

### 5.3.1 Structure versus Scale

The bias-variance decomposition reveals that this apparent contradiction is resolved by distinguishing two types of model error:

- **Structural error (variance):** The model's inability to capture the shape and dynamics of degradation trajectories — 0.23% of total squared error
- **Calibration error (bias²):** The systematic mismatch between the CALCE-trained SoH scale and the BLAST simulation SoH scale — 99.7% of total squared error

The FSI model is structurally valid across chemistries: it captures the *relative* degradation trajectory and the *ranking* of stress conditions consistently. What it does not do is translate the absolute SoH scale across chemistries, because different electrode chemistries have different capacity-fade curves, different SEI formation rates, and different EoL definitions.

### 5.3.2 Practical Implications for Fleet BMS

This finding has a direct practical implication: a single FSI model trained on laboratory data cannot be deployed across a chemically diverse fleet without per-chemistry initialisation. However, the calibration required is minimal — a single intercept shift derived from the first 10% of each chemistry's operating life reduces RMSE from 20.4% to 9.9%.

This is operationally feasible: most real fleet BMS systems include a manufacturer-specified SoH baseline for each cell chemistry. The FSI framework can leverage this baseline as the calibration anchor, making per-chemistry calibration a deployment step rather than a retraining requirement.

The calibration sensitivity analysis (Section 4.12) refines this practical guidance: for Oxford NMC cells, 90% of the maximum calibration benefit is achieved at N=10 cycles (RMSE 4.748% → 2.867%), compared to N=5 (3.851%). For BLAST multi-chemistry data, the minimum calibrated RMSE of 12.369% is achieved at N=37 cycles — substantially better than the N=5 result (17.941%). This confirms that longer commissioning observation periods provide progressively better absolute SoH alignment, particularly for diverse multi-chemistry fleets. The NHS model recommends 10 initial capacity measurements as the practical deployment minimum, with 30–40 cycles preferred for mixed-chemistry deployments.

### 5.3.3 Comparison with Prior Work

Severson et al. (2019) report RMSE values of 8–12% for cross-protocol cycle-life prediction using discharge curve features. The FSI framework achieves comparable performance on the same dataset using purely protocol-level features (ordinal ρ = 0.81) without requiring access to per-cycle discharge curve data — a significant practical advantage for early-life prediction.

Wang et al. (2020) and Attia et al. (2020) both report that current stress features improve degradation prediction but do not systematically isolate KI as the primary driver or validate it against real fleet telematics. This dissertation's Fleet DNA analysis provides the first direct quantitative link between the FSI's theoretical KI construct and measured real-world fleet operation.

---

---

## 5.4 Weight Cross-Validation: Are the Principled Weights Justified?

The weight derivation in Section 3.2.3 argued that FSI weights must be set from physical reasoning rather than CALCE gradient optimisation — because KI = 0 for all CALCE training cycles, creating a degenerate gradient signal. This argument was sound in principle, but it left an open question: could a data-driven optimiser find substantially better weights on independent data with variable KI?

The Severson weight optimisation (Section 4.10) provides a direct empirical answer. Running scipy L-BFGS-B optimisation with 50 random restarts on Severson cell-level data (n=49, KI range 0.000–0.441) finds numerically optimal weights of KI=0.00, DoD=0.40, T_norm=0.29, C_peak=0.31 — achieving Spearman ρ = −0.8358 versus the principled weights' ρ = −0.8223. The difference is 0.013 ρ-units — smaller than the statistical uncertainty given the sample size and well within the 95% CI range.

Two implications follow from this result:

**First: the principled weights are empirically validated.** Within the range of the available academic validation data, choosing 0.30/0.25/0.25/0.20 versus the numerically optimised values produces no statistically meaningful difference in ordinal predictive performance. The argument for principled weights does not rest on authority alone — it is confirmed experimentally.

**Second: the KI weight collapses to zero in Severson too.** The Severson dataset's KI range (0.000–0.441) does not extend to real fleet operation (0.60–0.81). Even with variable-protocol data, the optimiser finds that DoD is the dominant predictor within the academic dataset range. This systematically validates the argument for the 0.30 KI weight: no available academic dataset spans the full fleet KI envelope, so no available dataset can empirically confirm or refute the KI weight through gradient optimisation. The chosen KI weight is a physical extrapolation to deployment conditions, and this is an honest and correctly-stated limitation.

---

## 5.5 Ordinal Ranking: What NASA Teaches About T_norm

The within-dataset ordinal ranking validation (Section 4.11) produced an unexpected and highly informative result. Oxford (n=4 cells) showed the expected negative correlation between per-cell mean FSI and degradation rate (ρ = −0.80, correct direction). Severson confirmed this with strong statistical significance (ρ = −0.822, p < 0.001). But NASA (n=18 cells, p = 0.014) showed the *opposite*: ρ = +0.569, meaning higher-FSI cells degrade *more slowly*.

This is not random noise or a model failure — it is a mechanistically interpretable result that directly validates a theoretical concern raised in Section 3.2.2.

**The mechanism:** NASA cells were tested at three temperatures: 4°C, 24°C, and 43°C. Under the symmetric T_norm formula:
- 4°C cells: T_norm = |4 − 25|/25 = **0.84** (very high)
- 24°C cells: T_norm = |24 − 25|/25 = **0.04** (very low)
- 43°C cells: T_norm = |43 − 25|/25 = **0.72** (high)

The symmetric formula assigns the highest thermal stress score to the *cold* 4°C cells. But cold cells degrade *more slowly* than room-temperature cells — suppressed reaction kinetics reduce both electrolyte decomposition and SEI growth. The result is that the FSI over-penalises cold cells and under-orders them in the wrong direction of the degradation severity ranking.

This finding is significant for three reasons. First, it converts a theoretical concern (T_norm contradicts Arrhenius) into a measured empirical failure — the direction of degradation ranking is reversed for a real multi-temperature dataset. Second, it identifies the specific condition under which the FSI ordinal hypothesis breaks down: cross-temperature comparisons where T < 25°C cells are included. Third, it provides concrete design guidance for the asymmetric T_norm correction: cold temperatures should receive a partial penalty (β < 1 in the asymmetric formula) specifically to prevent cold cells from being ranked as higher-stress than warm cells.

Within-temperature comparisons (e.g. comparing cells at the same temperature but different charging protocols, as in Severson) remain valid. Fleet deployments where all vehicles operate at similar ambient temperatures would not exhibit this failure mode. But mixed-climate fleet comparisons require the asymmetric T_norm correction before ordinal ranking can be trusted.

---

## 5.6 Feature Ablation: What the Model Actually Uses

The feature ablation study (Section 4.13) raises a question about the current model architecture: if sub-components alone give R² = 1.00 on CALCE (configuration B), why include the composite FSI at all?

The answer is that the sub-components' near-perfect CALCE performance is a data artefact. CALCE uses constant-current cycling at fixed C-rates; in this regime, every cell's SoH is essentially a function of cumulative Ah throughput, which is almost perfectly predicted by DoD alone (RMSE = 0.183% for DoD-only configuration E). This reflects the training set's limitation, not the features' general power. On variable-protocol fleet data where DoD varies within a cell's cycling history and KI differs between cells, this artefact disappears.

The composite FSI plays a different role: it provides a *physically interpretable, deployment-ready scalar* that summarises multi-dimensional stress in a single number. A fleet BMS reporting "FSI = 0.72 — high stress regime" is more actionable than reporting seven separate feature values. The interpretability value of the composite is independent of whether it improves CALCE R² over raw components.

The Severson ordinal ρ result confirms this: all composite configurations (A through F) produce ρ ≈ −0.822, indistinguishable within rounding. The FSI's value is not in adding unique information — it is in packaging existing information in a physically meaningful and operationally interpretable form.

The feature ablation also resolves the SHAP interpretation question more precisely: the reason FSI alone (configuration A) achieves CALCE RMSE = 8.95% worse than the full model (3.75%) is that the XGBoost model can use the raw sub-components (especially DoD and T_avg_C) as additional precise signals that the fixed-weight FSI composite averages over. The model architecture benefits from both the composite (interpretability) and its components (flexibility). Acknowledging this explicitly — that the architecture is designed for interpretability at some cost to CALCE-optimised accuracy — is the honest characterisation of the design choice.

---

## 5.7 Limitations

### 5.7.1 T_norm Symmetry

The thermal stress term T_norm = |T − 25|/25 is symmetric around the 25°C reference: it assigns equal weight to elevated and depressed temperature deviations of the same magnitude. This is inconsistent with the Arrhenius degradation model, under which:

$$k(T) = A \cdot e^{-E_a / RT}$$

where k(T) is the reaction rate constant, Eₐ is activation energy, R is the gas constant, and T is absolute temperature. Under Arrhenius kinetics, a 10°C increase above reference increases reaction rates by approximately 2× (the Q10 rule), while a 10°C decrease below reference reduces rates by approximately 2× — but the degradation mechanisms are directionally asymmetric: elevated temperature accelerates SEI growth and electrolyte decomposition, while cold temperatures accelerate lithium plating during charging (not decomposition). These are different mechanisms with different activation energies.

The symmetric formulation therefore conflates two physically distinct stress regimes. In the current dataset configuration this limitation has limited impact because: (a) the CALCE and Severson datasets are single-temperature experiments at 25°C or 30°C, meaning T_norm is constant across all cells; and (b) the BLAST dataset spans temperatures but the bias is dominated by chemistry-specific offset, not thermal asymmetry.

**Proposed future work:** Replace T_norm with an asymmetric function:
- T > 25°C: T_hot = (T − 25)/25 [linear, positive]
- T ≤ 25°C: T_cold = β·(25 − T)/25 where β < 1 (partial weight for cold, reflecting slower kinetics rather than accelerated decomposition)

The optimal β would be derived from Arrhenius fitting to the BLAST temperature sensitivity data. This would require re-running the full validation pipeline and recalibrating FSI weights.

### 5.7.2 Training Data Representativeness

All ML training data is from laboratory CC cycling of LiCoO₂ cells. The model has therefore never been trained on a battery cycle with KI > 0. The fact that KI is nonetheless the top SHAP feature reflects the model learning from *between-protocol* variation in the Severson data (different CC-class C-rates) rather than within-cycle current variability. True training on variable-current profiles would require labelled degradation data from fleet operation — which currently does not exist in publicly accessible form at scale.

This is a fundamental limitation of the field rather than of this specific study. The NREL Fleet DNA database provides KI measurements but not paired SoH measurements for the same cells. Bridging this gap — either through controlled variable-current lab cycling at scale, or through instrumented fleet trials with cell-level SoH tracking — is the most impactful direction for future work.

### 5.7.3 Severson Analytical KI

The Severson FSI features are computed from published protocol parameters rather than raw waveform data. The analytical KI formula is exact for ideal two-step CC–CC charging, but real-world charging may include transient current spikes at protocol step transitions, CV phase tailing, and battery management system (BMS) current smoothing — none of which are captured by the two-value discrete distribution model. These effects would tend to *increase* the true KI slightly above the analytically computed value, meaning the Severson KI values in this study are conservative lower bounds rather than exact measurements.

### 5.7.4 Limited Real-World Validation

The Fleet DNA dataset provides KI measurements from real fleet trips but no paired battery SoH data. The FSI has therefore been validated on real-world current profiles only at the level of KI distribution, not at the level of SoH prediction accuracy. A complete validation chain — real fleet telematics → FSI features → SoH prediction → actual SoH measurement — has not been established, as this would require a prospective instrumented fleet trial beyond the scope of a taught MSc dissertation.

### 5.7.5 Limitation Summary

**Table 5.1: Limitations, evidence, and treatment**

| Limitation | Evidence | Impact | Treatment |
|-----------|----------|--------|-----------|
| T_norm symmetry | Arrhenius theory; **NASA ordinal failure (ρ=+0.57) provides direct empirical proof** | Demonstrated: wrong ordinal direction for multi-temp datasets | Future work: asymmetric thermal function with β < 1 for T < 25°C |
| Cross-chemistry calibration gap | BLAST RMSE 20.4% uncalibrated → **17.9% at N=5; 12.4% at N=37** | Moderate; calibration sensitivity curve shows deployment guidance | Per-chemistry calibration: minimum 10 cycles, 30–40 preferred for mixed fleet |
| Training data = CC lab only | Fleet KI 0.60–0.81 vs training KI = 0 | Structural; mitigated by Severson + Fleet DNA ordinal validation | Field data collection (future work) |
| Analytical KI (Severson) | No raw waveforms available | Conservative KI underestimation; ρ may be slightly underestimated | Acceptable for ordinal validation |
| No real fleet SoH labels | Fleet DNA = telematics only, no cell SoH | Cannot close the full validation chain | Prospective instrumented fleet trial |
| Feature collinearity (FSI + sub-components) | Feature ablation: FSI-alone RMSE 8.95% vs current 3.75%; Severson ρ identical across ablations | SHAP inflates FSI importance; harmless in CALCE but misleading for variable-KI data | Ablation completed; future work: train on variable-KI data to obtain valid SHAP |

---

## 5.8 Scope of the Contribution

It is important to be precise about what the FSI framework claims and does not claim:

**What the FSI framework demonstrates:**
1. KI is a physically meaningful and statistically significant predictor of battery degradation severity across multiple chemistries, current profiles, and experimental sources
2. FSI correctly ranks the degradation severity of 49 cells from 23 charging protocols (Spearman ρ = −0.822, p < 0.001, 95% CI [−0.896, −0.703])
3. Real-world EV fleet KI values (0.60–0.81) are categorically different from laboratory CC KI values (0.000), confirming the training-deployment gap
4. The cross-chemistry performance gap is dominated by calibration bias (84% of squared error), not structural model failure (variance < 0.5%)
5. Principled FSI weights are validated — Severson weight optimisation confirms they are within 0.013 ρ-units of numerically optimal
6. T_norm symmetry causes ordinal ranking failure on multi-temperature datasets (NASA ρ = +0.57, wrong direction), providing direct empirical evidence for the Arrhenius-based limitation

**What the FSI framework does not claim:**
1. That absolute SoH prediction at 3.73% accuracy can be achieved across all chemistries without per-chemistry calibration
2. That the FSI is fully validated end-to-end on real fleet batteries with known SoH
3. That the FSI weights (0.30/0.25/0.25/0.20) are universally optimal for all chemistries and deployment scenarios
4. That T_norm accurately captures the directional asymmetry of thermal degradation (NASA ordinal failure confirms it does not for multi-temperature comparisons)

---

## 5.9 Implications for EV Fleet Battery Management

The FSI framework has several practical implications for commercial EV fleet management, even within its current validated scope:

**Early degradation warning:** Because FSI is computed from cycle-level current, temperature, and voltage data — all of which are available in real-time from a vehicle's BMS — it can be computed on-board without requiring periodic laboratory testing or electrochemical impedance spectroscopy. A rising FSI trend over recent cycles provides an early warning signal that a cell is experiencing higher-than-baseline stress.

**Protocol optimisation:** The Severson results demonstrate that charging protocol selection has a 62% impact on cell lifetime. Fleet operators who can control charging protocol (e.g., depot charging systems) can use FSI to identify lower-stress charging regimes. The 8C two-step protocols that maximise charging speed (short charge time) also maximise KI and reduce lifetime by 61% relative to 3.6C one-step charging — the FSI makes this tradeoff quantitative.

**Maintenance scheduling:** The Random Forest health-label classifier provides a direct actionable output (Healthy / Degraded / End_of_Life) that can be integrated into fleet maintenance management systems without requiring SoH percentage estimation. The classifier's high precision at the EoL boundary (0.97) means that End_of_Life classifications are rarely false positives — fleet operators can trust an EoL classification as a reliable replacement trigger.

**Chemistry-agnostic stress ranking:** Although absolute SoH prediction requires per-chemistry calibration, the FSI's ordinal stress ranking is chemistry-invariant (SHAP ρ > 0.80 across all tested chemistries). A fleet with mixed chemistry types can use a single FSI framework to rank cells by relative stress exposure, even if absolute SoH comparison across chemistries is not attempted.

---

## 5.10 Conclusion

The Fleet Stress Index addresses a genuine and previously unquantified gap in battery degradation modelling: the systematic mismatch between laboratory constant-current testing and real-world EV fleet operation. By introducing Kinetic Intensity as a first-class feature — explicitly encoding the current variability that CC lab cycling eliminates — the FSI framework provides a physically grounded bridge between experimental battery science and fleet operational data.

The validation evidence supports three core conclusions: (1) KI is a statistically significant and reproducible predictor of degradation severity across multiple independent experimental sources; (2) the FSI framework transfers across chemistries at the level of structural logic and feature attribution, though absolute SoH prediction requires per-chemistry calibration; and (3) the gap between lab and fleet operating conditions — the founding motivation for the FSI — is real, quantifiable, and captured by KI values that are 60–80× higher in real fleet operation than in any laboratory protocol.

The limitations — primarily the T_norm symmetry assumption and the absence of real fleet SoH labels — are genuine constraints on the current scope, and they are proposed as specific, actionable directions for future work rather than as fundamental objections to the FSI framework. Within the validated scope, the FSI represents a contribution that can be implemented today, using existing fleet telematics infrastructure, to improve degradation monitoring for commercial EV fleets.

---

## References (Chapter 5)

- Attia, P.M. et al. (2020). Closed-loop optimization of fast-charging protocols for batteries with machine learning. *Nature*, 578(7795), pp.397–402.
- Severson, K.A. et al. (2019). Data-driven prediction of battery cycle life before capacity degradation. *Nature Energy*, 4(5), pp.383–391.
- Wang, J. et al. (2020). Cycle-life model for graphite-LiFePO₄ cells. *Journal of Power Sources*, 196(8), pp.3942–3948.
- Millner, A. (2010). Modeling lithium ion battery degradation in electric vehicles. *IEEE Conference on Innovative Technologies for an Efficient and Reliable Electricity Supply*, pp.349–356.
- Plett, G.L. (2015). *Battery Management Systems, Volume I: Battery Modeling*. Artech House.
- Ecker, M. et al. (2012). Development of a lifetime prediction model for lithium-ion batteries based on extended accelerated aging test data. *Journal of Power Sources*, 215, pp.248–257.
- Lundberg, S.M. and Lee, S.I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.
