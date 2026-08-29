# Live demo notebooks

Two Colab-ready notebooks that re-run this dissertation's own code, live, from
this public repository — for use during the viva when asked "how did you get
this number?".

| Notebook | Reproduces | Open in Colab |
|---|---|---|
| `live_demo_cross_chemistry.ipynb` | Outcome 4 — R² = −0.57 cross-chemistry limit | [Open](https://colab.research.google.com/github/Usharani1699/battery-degradation-analytics/blob/main/08_Live_Demo_Notebooks/live_demo_cross_chemistry.ipynb) |
| `live_demo_shap_outcome2.ipynb` | Outcome 2 — SHAP confirms mileage/duty-cycle dominate over FSI | [Open](https://colab.research.google.com/github/Usharani1699/battery-degradation-analytics/blob/main/08_Live_Demo_Notebooks/live_demo_shap_outcome2.ipynb) |

Both notebooks download the actual scripts and data from this repo at runtime
and execute them unmodified — nothing is pre-computed or faked. Each cell
takes 10–60 seconds in Colab; run "Run all" a few minutes before you need it.

`evbattery_vehicle_fsi.csv` in this folder is a copy of the EV-fleet feature
table (100 real vehicles) used by the SHAP notebook.
