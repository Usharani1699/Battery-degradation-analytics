# Why NMC Chemistry Was Selected
## EG7030 Dissertation — Week 4 Detailed Notes

---

## What Is NMC?

NMC stands for **Nickel Manganese Cobalt Oxide** — the positive electrode (cathode) material in lithium-ion batteries. Full chemical formula: **LiNixMnyCozO2**, where x + y + z = 1.

### Common NMC Variants

| Variant | Ni : Mn : Co | Energy Density | Stability | EV Use |
|---------|-------------|----------------|-----------|--------|
| NMC-111 | 1:1:1 | Moderate | Good | Older EVs |
| NMC-523 | 5:2:3 | Good | Good | BMW i3 (early) |
| NMC-622 | 6:2:2 | Good | Moderate | Nissan Leaf Gen 2 |
| NMC-811 | 8:1:1 | Highest | Lower | Tesla Model 3 LR, BMW i4 |

**NMC-811 is the current industry standard for high-energy EV batteries** and is the variant used in both Paper D1 (Mulpuri et al.) and Paper D2 (Si et al.).

The CALCE INR 18650-20R (Samsung NMC cell) is likely NMC-622 or NMC-532 formulation — still NMC family, enabling direct chemistry comparison.

---

## Four Reasons NMC Was Selected

### Reason 1: Dataset Availability (Cross-Study Consistency)

Your three datasets provide NMC data as follows:

| Dataset | NMC Cell | Details |
|---------|---------|---------|
| CALCE | INR 18650-20R | Samsung 18650 cylindrical, 2.0 Ah, NMC cathode |
| Oxford | BMP/BMR/SPM pouch cells | NMC-type pouch cells, tested by Oxford Engineering |
| NASA | NOT available | NASA cells are LCO (Lithium Cobalt Oxide) |

**Two out of three datasets have NMC cells** → sufficient training data for cross-validated ML models.

If you had chosen LFP (Lithium Iron Phosphate): only the CALCE A123 cell is available — one dataset = insufficient for robust ML training.
If you had chosen LCO: NASA + CALCE CS2/CX2 cells available, but LCO is declining in EV use (mainly in older phones and early EVs).

**NMC gives the best balance of dataset coverage and industry relevance.**

### Reason 2: Literature Alignment (Benchmarking Possible)

| Paper | Cell | Chemistry |
|-------|------|-----------|
| Mulpuri et al. 2025 (D1) | LGM50 21700 | NMC-811 |
| Si et al. 2025 (D2) | Research cell | NMC-811 (LiNi₀.₈Mn₀.₁Co₀.₁O₂) |

Because both your most directly relevant papers use **NMC-811**, you can:
- Compare your capacity fade rates against their published results
- Validate your ML model accuracy against their reported metrics
- Use the same PyBaMM parameter set (Chen2020 = LGM50 NMC-811 parameters)

If you had used LFP, you could only benchmark against Severson 2019 — a less relevant comparison for EV applications.

### Reason 3: EV Market Dominance

NMC chemistry powers the majority of electric vehicles sold globally:

| Vehicle | Battery Chemistry | Battery Supplier |
|---------|-----------------|-----------------|
| Tesla Model 3 Long Range | NMC-811 | Panasonic (2170 cell) |
| Tesla Model Y | NMC-811 | LG Energy (2170 cell) |
| BMW i4 | NMC | Samsung SDI |
| Volkswagen ID.4 | NMC | LG Energy / Samsung SDI |
| Hyundai Ioniq 5 | NMC | SK Innovation |
| Nissan Leaf e+ | NMC-622 | AESC |
| Audi Q4 e-tron | NMC | Samsung SDI |
| Rivian R1T | NMC-811 | Samsung SDI |

**UK market context**: Under the Zero Emission Vehicle (ZEV) Mandate (2024), 22% of new cars sold in the UK in 2024 must be zero emission, rising to 80% by 2030. The majority of these will be NMC-battery EVs — making NMC degradation research directly policy-relevant.

**Market share**: NMC accounts for approximately 45–60% of global EV battery production by capacity (GWh). LFP is second (~30%) but growing mainly in lower-cost segments (base Tesla Model 3/Y in China, BYD vehicles).

**Energy density comparison:**
- NMC-811: ~200–250 Wh/kg (cell level)
- LFP: ~120–160 Wh/kg (cell level)
- LCO: ~180–200 Wh/kg (cell level)

Higher energy density of NMC means longer range EVs — precisely the segment where battery degradation management matters most for fleet operators.

### Reason 4: Rich Degradation Signature for SHAP Analysis

NMC batteries degrade through **multiple concurrent mechanisms**, making them ideal for explainability analysis (SHAP):

| Degradation Mode | Mechanism | What It Affects | SHAP Feature |
|-----------------|-----------|-----------------|-------------|
| **SEI growth** | Electrolyte reduction at graphite anode forms a growing passivation layer | Capacity loss, resistance increase | Charge capacity ratio, internal resistance |
| **Particle cracking** | Volume change (~5–8%) during lithiation/delithiation causes mechanical fracture | New SEI on exposed surfaces → accelerated loss | Voltage variance, coulombic efficiency |
| **Lithium plating** | Li deposits as metal instead of intercalating at high rates or low temperatures | Irreversible capacity loss, safety risk | Charging voltage spike, differential capacity |
| **LLI (Loss of Lithium Inventory)** | Lithium trapped in SEI, dead lithium, or plated Li no longer participates in reactions | Direct capacity reduction | Discharge capacity fade |
| **LAM (Loss of Active Material)** | Cracked or isolated electrode particles no longer participate | Capacity and power fade | High-frequency impedance |
| **Electrolyte decomposition** | Oxidation of electrolyte at high voltage NMC cathode | Gas generation, resistance increase | Temperature rise, voltage plateau shift |

**Why this matters for SHAP**: Because NMC has multiple active degradation modes that respond differently to different stress factors (temperature, C-rate, SoC window), SHAP analysis will reveal **which features correspond to which degradation mode** at different stages of battery life. This gives mechanistic insight that purely data-driven approaches cannot provide.

**LFP comparison**: LFP has mainly SEI growth and lithium plating — fewer modes, less rich for SHAP analysis. Its flat discharge voltage curve also makes feature extraction harder.

---

## NMC Chemistry Parameters for PyBaMM

When you reach Stage 6 (PyBaMM simulation), you will use these parameters:

```python
import pybamm

# Chen2020 parameters are specifically for the LGM50 NMC-811 cell
# (same cell as Paper D1 by Mulpuri et al.)
param = pybamm.ParameterValues("Chen2020")

# Key parameters
param["Nominal cell capacity [A.h]"]  # = 5 Ah
param["Upper voltage cut-off [V]"]    # = 4.2 V
param["Lower voltage cut-off [V]"]    # = 2.5 V
param["Positive electrode SOC_0"]     # NMC initial stoichiometry
```

The Chen2020 parameter set was published in:
- **Chen, M. et al. (2020)** — "Development of Experimental Techniques for Parameterization of Multi-scale Lithium-ion Battery Models" — *J. Electrochem. Soc.*
- DOI: https://doi.org/10.1149/1945-7111/ab9050

---

## Summary: NMC vs. Alternatives Comparison Table

| Criterion | NMC | LFP | LCO |
|-----------|-----|-----|-----|
| Available in CALCE | Yes (INR 18650-20R) | Yes (A123) | Yes (CS2, CX2) |
| Available in Oxford | Yes | No | No |
| Available in NASA | No | No | Yes |
| Used in D1 paper | Yes (NMC-811) | No | No |
| Used in D2 paper | Yes (NMC-811) | No | No |
| EV market share 2024 | ~50% | ~30% | <5% |
| Energy density | Highest | Lowest | Medium |
| Degradation modes | Many (SEI, crack, plating, LLI, LAM) | Few (SEI dominant) | Medium |
| SHAP richness | Highest | Lower | Medium |
| PyBaMM parameters | Chen2020 (validated) | Chen2020 (LFP variant) | Available |
| **VERDICT** | **SELECTED** | Alternative | Not suitable |
