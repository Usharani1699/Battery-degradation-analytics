import pybamm
import matplotlib
matplotlib.use("Agg")  # Save to file instead of opening a window
import matplotlib.pyplot as plt

# ── TUTORIAL 1: Your first battery simulation ──
# SPM = Single Particle Model
# Simplest lithium-ion model: one particle per electrode
# Chen2020 = NMC/graphite cell parameters (same chemistry as Amara Raja)

model = pybamm.lithium_ion.SPM()
param = pybamm.ParameterValues("Chen2020")

# Simulate a 1C discharge (full discharge in 1 hour = 3600 seconds)
sim = pybamm.Simulation(model, parameter_values=param)
sol = sim.solve([0, 3600])

# Extract results
time     = sol["Time [s]"].entries
voltage  = sol["Terminal voltage [V]"].entries
current  = sol["Current [A]"].entries
soc      = sol["Discharge capacity [A.h]"].entries

# Plot 3 subplots
fig, axes = plt.subplots(3, 1, figsize=(10, 8))
fig.suptitle("PyBaMM Tutorial 1 — NMC/Graphite Cell, 1C Discharge (SPM)", fontsize=13, fontweight="bold")

axes[0].plot(time, voltage, color="navy", linewidth=2)
axes[0].set_ylabel("Voltage (V)")
axes[0].set_title("Terminal Voltage drops as cell discharges")
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=3.0, color="red", linestyle="--", alpha=0.5, label="Cutoff ~3.0V")
axes[0].legend()

axes[1].plot(time, current, color="darkorange", linewidth=2)
axes[1].set_ylabel("Current (A)")
axes[1].set_title("Discharge Current (constant 1C)")
axes[1].grid(True, alpha=0.3)

axes[2].plot(time, soc, color="green", linewidth=2)
axes[2].set_ylabel("Discharge Capacity (Ah)")
axes[2].set_xlabel("Time (seconds)")
axes[2].set_title("Capacity delivered over time")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("tutorial_1_output.png", dpi=150, bbox_inches="tight")
print("Plot saved: tutorial_1_output.png")

# Print some key values
print(f"\nSimulation Summary:")
print(f"  Start voltage : {voltage[0]:.3f} V")
print(f"  End voltage   : {voltage[-1]:.3f} V")
print(f"  Voltage drop  : {voltage[0] - voltage[-1]:.3f} V")
print(f"  Total capacity: {soc[-1]:.3f} Ah")
print(f"\nThis is what happens inside every NMC cell during discharge.")
print("Lithium ions move from negative electrode to positive electrode.")
print("Voltage drops as the electrodes reach their limits.")
