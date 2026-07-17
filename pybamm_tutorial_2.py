import pybamm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── TUTORIAL 2: C-rate Comparison ──
# Question: What happens inside an NMC cell when you discharge faster?
# This is directly relevant to your Fleet Severity Index (FSI):
# Urban buses = high C-rate (frequent stop/start acceleration)
# Motorway vehicles = low C-rate (steady cruise)

model = pybamm.lithium_ion.SPM()
param = pybamm.ParameterValues("Chen2020")

# Define 4 different C-rates to compare
c_rates = [0.5, 1, 2, 3]
colors  = ["green", "navy", "orange", "red"]
labels  = ["0.5C (gentle - motorway)", "1C (normal)", "2C (aggressive - urban)", "3C (very aggressive)"]

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("PyBaMM Tutorial 2: Effect of C-rate on NMC/Graphite Cell Discharge Behaviour", fontsize=12, fontweight="bold")

ax_voltage  = axes[0, 0]
ax_capacity = axes[0, 1]
ax_temp     = axes[1, 0]
ax_summary  = axes[1, 1]

summary_capacity = []
summary_end_v    = []

for c_rate, color, label in zip(c_rates, colors, labels):

    # Calculate current for this C-rate
    # Chen2020 cell = 5 Ah, so 1C = 5A, 2C = 10A, 3C = 15A
    current = c_rate * 5.0  # Amps

    # Set the discharge current in parameters
    param_copy = param.copy()
    param_copy["Current function [A]"] = current

    # Solve - time window shrinks as C-rate increases
    # 0.5C takes 2 hours (7200s), 3C takes only 20 min (1200s)
    t_end = int(3600 / c_rate)
    sim = pybamm.Simulation(model, parameter_values=param_copy)
    sol = sim.solve([0, t_end])

    # Extract results
    time     = sol["Time [s]"].entries
    voltage  = sol["Terminal voltage [V]"].entries
    capacity = sol["Discharge capacity [A.h]"].entries

    # Plot voltage vs time
    ax_voltage.plot(time / 3600, voltage, color=color, linewidth=2, label=label)

    # Plot voltage vs capacity (more useful — shows energy delivered)
    ax_capacity.plot(capacity, voltage, color=color, linewidth=2, label=label)

    # Store for summary bar chart
    summary_capacity.append(capacity[-1])
    summary_end_v.append(voltage[-1])

    print(f"{label}")
    print(f"  End voltage : {voltage[-1]:.3f} V")
    print(f"  Capacity    : {capacity[-1]:.3f} Ah")
    print(f"  Energy      : {np.trapezoid(voltage, capacity):.3f} Wh")
    print()

# Voltage vs Time
ax_voltage.axhline(y=3.0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
ax_voltage.set_xlabel("Time (hours)")
ax_voltage.set_ylabel("Voltage (V)")
ax_voltage.set_title("Voltage vs Time\n(Higher C-rate = shorter discharge time)")
ax_voltage.legend(fontsize=8)
ax_voltage.grid(True, alpha=0.3)

# Voltage vs Capacity
ax_capacity.axhline(y=3.0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
ax_capacity.set_xlabel("Discharge Capacity (Ah)")
ax_capacity.set_ylabel("Voltage (V)")
ax_capacity.set_title("Voltage vs Capacity\n(Higher C-rate = less usable capacity)")
ax_capacity.legend(fontsize=8)
ax_capacity.grid(True, alpha=0.3)

# Summary bar - usable capacity at each C-rate
bars = ax_summary.bar([f"{c}C" for c in c_rates], summary_capacity,
                       color=colors, alpha=0.8, edgecolor="black")
ax_summary.set_xlabel("C-rate")
ax_summary.set_ylabel("Usable Capacity (Ah)")
ax_summary.set_title("Usable Capacity vs C-rate")
ax_summary.set_ylim(0, 6)
ax_summary.grid(True, alpha=0.3, axis="y")
for bar, cap in zip(bars, summary_capacity):
    ax_summary.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f"{cap:.2f} Ah", ha="center", va="bottom", fontsize=10, fontweight="bold")

# Hide unused subplot
ax_temp.axis("off")
ax_temp.text(0.5, 0.6, "Key Observations:", ha="center", va="center",
             fontsize=11, fontweight="bold", transform=ax_temp.transAxes)
ax_temp.text(0.5, 0.45, "Higher C-rate = lower usable capacity", ha="center", va="center",
             fontsize=10, color="red", transform=ax_temp.transAxes)
ax_temp.text(0.5, 0.32, "Urban driving (high C) stresses cells more", ha="center", va="center",
             fontsize=10, color="darkorange", transform=ax_temp.transAxes)
ax_temp.text(0.5, 0.19, "than motorway driving (low C)", ha="center", va="center",
             fontsize=10, color="darkorange", transform=ax_temp.transAxes)
ax_temp.text(0.5, 0.06, "C-rate effect underpins the Fleet Severity Index metric", ha="center", va="center",
             fontsize=10, color="navy", transform=ax_temp.transAxes)

plt.tight_layout()
plt.savefig("tutorial_2_output.png", dpi=150, bbox_inches="tight")
print("Plot saved: tutorial_2_output.png")
print("\nKey takeaway:")
print("Capacity at 0.5C vs 3C shows how much energy is LOST due to aggressive driving.")
print("This energy loss per cycle is what accelerates battery degradation in urban fleets.")
