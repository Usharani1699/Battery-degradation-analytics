# Week 4 Dissertation Progress — Detailed Notes
**EG7030 · Meeting with Dr. Abdulrahman Albar · 23 June 2026**

Student: Usha Rani V. | ID: u2965962 | MSc Electric Vehicle Engineering | UEL

Dissertation: *"Duty Cycle-Aware Battery Degradation Analytics for Electric Vehicle Fleet Applications"*

---

## Contents

| File | Topic |
|------|-------|
| [01_Literature_Review.md](01_Literature_Review.md) | Full summaries of all 5 papers with DOI links |
| [02_Dataset_Analysis.md](02_Dataset_Analysis.md) | NASA, CALCE & Oxford datasets — structure, files, common points |
| [03_NMC_Chemistry.md](03_NMC_Chemistry.md) | Why NMC was selected — detailed reasoning |
| [04_EV_Fleet_Data.md](04_EV_Fleet_Data.md) | Fleet DNA dataset analysis & related articles |
| [05_Research_Methodology.md](05_Research_Methodology.md) | 7-stage methodology framework with explanations |

---

## Quick Summary of Week 4 Achievements

| Task | Status |
|------|--------|
| Research papers reviewed (3 uploaded PDFs confirmed relevant) | Done |
| Two additional foundational papers identified | Done |
| CALCE, NASA, Oxford datasets downloaded & organised | Done |
| Common dataset features identified | Done |
| NMC chemistry selected with justification | Done |
| Fleet DNA EV data located | Done |
| Week 4 presentation created | Done |

---

## Three Datasets in One Line Each

- **NASA PCoE**: 4 LCO 18650 cells cycled to 30% capacity fade; includes EIS impedance data. Good for RUL benchmarks.
- **CALCE**: Multi-chemistry dataset (LCO/LFP/NMC). INR 18650-20R is the **NMC cell** (Samsung, 2.0 Ah). Has dynamic profiles (DST, FUDS).
- **Oxford**: NMC pouch cells in BMP/BMR/SPM families. .csv and .mat format. High-fidelity SoH data.

## NMC in One Paragraph

NMC (Nickel Manganese Cobalt Oxide — LiNixMnyCozO2) is selected because: (1) CALCE's INR 18650-20R and Oxford pouch cells are NMC, giving us cross-dataset NMC data; (2) both our most relevant papers (Mulpuri 2025, Si 2025) use NMC-811 cells, enabling direct benchmarking; (3) NMC dominates commercial EVs (Tesla, BMW, VW, Hyundai) representing ~50% of EV market share; (4) NMC's multi-mode degradation signature (SEI, particle cracking, Li-plating, LLI, LAM) is richest for SHAP feature attribution analysis.
