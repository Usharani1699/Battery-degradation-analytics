# Battery Degradation Analytics for EV Fleet Applications

**MSc Electric Vehicle Engineering Dissertation**  
University of East London | Distinction track | 2025-2026  
**Author:** Usha Rani Vamanagiri

---

## Project Overview

This repository contains the data processing, feature engineering, and machine learning pipeline for my MSc dissertation:

**"Duty Cycle-Aware Battery Degradation Analytics for EV Fleet Applications"**

The core idea: battery degradation is not just about age or total charge cycles. The *type* of driving (urban stop-start vs motorway) has a large impact on how fast a battery degrades. This project builds a Fleet Severity Index (FSI) that quantifies how hard a fleet is working its batteries, then predicts remaining useful life (RUL) using ML models.

---

## Research Questions

1. How much does drive cycle profile (WLTP, UDDS, FTP-75, HWFET) affect NMC battery degradation rate?
2. Can a Fleet Severity Index (FSI) reliably rank fleet duty cycle harshness?
3. Which ML approach (XGBoost, LSTM, Random Forest) gives the best SOH prediction with SHAP explainability?

---

## Datasets Used

- **CALCE Battery Dataset** — University of Maryland. NMC cells under various charge/discharge conditions.
- **NASA PCoE Battery Dataset** — Li-ion cells aged under different operating conditions.
- **Oxford Battery Degradation Dataset** — Cells cycled to 80% capacity retention.

> Raw dataset files are not included in this repository (too large). Download links are in `data/README_data_sources.md`.

---

## ML Pipeline

| Stage | Method | Script |
|-------|--------|--------|
| Data extraction | Pandas, MATLAB | `extract_all_datasets.py` |
| Feature engineering | Capacity fade, IR growth, dQ/dV | `features/` |
| Drive cycle severity | FSI calculation across WLTP/UDDS/FTP-75/HWFET | `fsi/` |
| SOH prediction | XGBoost, Random Forest | `models/xgb_model.py` |
| Sequence modelling | LSTM (PyTorch) | `models/lstm_model.py` |
| Explainability | SHAP waterfall + summary plots | `explainability/shap_analysis.py` |

---

## Tools and Libraries

- **Python:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, PyTorch, SHAP
- **MATLAB:** Data extraction from Oxford and NASA `.mat` files
- **Simscape Battery:** Battery equivalent circuit modelling (coursework)
- **MathWorks Certifications (all 100%):** MATLAB, Simulink, Power Electronics, Control Design, Simscape Battery

---

## Key Results (Preliminary)

> Results will be updated as the dissertation progresses (submission: September 2026)

- FSI successfully differentiates between aggressive (UDDS urban) and mild (HWFET highway) duty cycles
- XGBoost achieves [X]% RMSE on SOH prediction
- SHAP identifies internal resistance growth as the strongest predictor of capacity fade

---

## Author

**Usha Rani Vamanagiri**  
MSc Electric Vehicle Engineering, University of East London  
4+ years battery engineering experience (Amara Raja Advanced Cell Technologies, PUR Energy)  
LinkedIn: [linkedin.com/in/usharaniv3](https://linkedin.com/in/usharaniv3)  
Email: vamanagiriusharani16.99@gmail.com
