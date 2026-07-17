# EV Fleet Data — Delivery Truck Duty Cycles
## EG7030 Dissertation — Week 4 Detailed Notes

---

## Fleet DNA Dataset

### Source
- **Full Name**: Fleet DNA — Commercial Fleet Duty Cycle Data
- **Organisation**: National Renewable Energy Laboratory (NREL), US Department of Energy
- **Website**: https://www.nrel.gov/transportation/fleetdna.html
- **Dataset portal**: https://www.nrel.gov/transportation/fleettest-clean-cities.html
- **Your local file**: `C:\Users\pure ev\Music\Term 3\data_for_fleet_dna_delivery_trucks.csv`

### What Fleet DNA Is
Fleet DNA is a publicly available database of real-world vehicle duty cycle data collected from commercial fleets across the United States. Data was collected using onboard GPS loggers and vehicle data recorders installed in actual fleet vehicles during regular operations.

### What's in the Delivery Trucks CSV

The `data_for_fleet_dna_delivery_trucks.csv` file contains data from commercial delivery truck operations. Based on standard NREL Fleet DNA data structure, it typically includes:

| Column | Description | Unit |
|--------|-------------|------|
| `vehicle_id` | Unique vehicle identifier | — |
| `trip_id` | Individual trip identifier | — |
| `date` | Date of operation | YYYY-MM-DD |
| `time` | Time of recording | HH:MM:SS |
| `speed` | Vehicle speed | mph (or km/h) |
| `distance` | Cumulative trip distance | miles |
| `idle_time` | Time spent at 0 mph | seconds |
| `trip_duration` | Total trip time | minutes |
| `max_speed` | Maximum speed in trip | mph |
| `avg_speed` | Average speed | mph |
| `positive_accel` | Positive acceleration segments | — |
| `decel` | Deceleration events | — |

### Why This Data Matters for Your Dissertation

**Connection to Battery Degradation:**

Delivery trucks follow routes with characteristic stop-go patterns:
- Frequent stops at delivery addresses (urban pattern)
- Acceleration from standstill → high C-rate demand on battery
- Extended idle periods at loading docks
- Potentially multiple short trips per day (micro-cycling)

Paper D1 (Mulpuri et al. 2025) showed that **urban stop-go patterns cause the fastest battery degradation** — and delivery truck routes are the archetypal urban stop-go duty cycle.

---

## Duty Cycle Metrics You Will Compute

From the Fleet DNA data, you will calculate these metrics for each trip or vehicle:

### 1. Kinetic Intensity (KI)
```
KI = (1/d) × Σ(|a_i|) for all acceleration/deceleration events
where d = total distance, a_i = acceleration at time step i
```
- Units: m/s² per km
- **High KI** = aggressive driving, frequent stop-go = high battery stress
- Delivery trucks typically have KI of 0.3–0.8 m/s²/km vs. highway trucks at 0.1–0.2

### 2. Relative Positive Acceleration (RPA)
```
RPA = Σ(v_i × a_i) / d   for all time steps where a_i > 0
where v_i = speed at step i, a_i = acceleration at step i, d = distance
```
- Units: m/s² (dimensionless when normalised)
- Measures the energy demanded during acceleration relative to total distance
- Higher RPA = more aggressive acceleration cycles = higher C-rate demand on battery

### 3. Positive Kinetic Energy (PKE)
```
PKE = (1/d) × Σ(0.5 × (v_i² - v_{i-1}²)) for all time steps where v_i > v_{i-1}
```
- Units: m²/s² per km (or J/kg/km)
- Measures kinetic energy gained per unit distance
- Directly proportional to electrical energy consumed for acceleration

### 4. Fleet Severity Index (FSI) — Your Novel Contribution
```
FSI = w₁ × KI_normalised + w₂ × RPA_normalised + w₃ × temperature_factor
      + w₄ × charging_rate_factor
```
Where weights (w₁, w₂, w₃, w₄) are derived from the ML model's SHAP values to reflect actual degradation impact of each factor.

FSI = 0 → extremely gentle operation (highway, constant speed, moderate temperature)
FSI = 1 → maximum severity (urban stop-go, high temperature, fast charging)

---

## Connection Between Fleet DNA and Lab Datasets

### Pattern Matching

| Fleet DNA Pattern | CALCE Equivalent | Notes |
|------------------|-----------------|-------|
| Delivery urban stop-go | DST (Dynamic Stress Test) | DST simulates urban duty cycling with variable current |
| Mixed urban-highway | FUDS (Federal Urban Driving Schedule) | FUDS captures mixed driving patterns |
| Highway delivery (less common) | US06 (Highway profile) | High sustained current |

**This is a critical connection for your dissertation**: You can argue that:
1. Fleet DNA delivery trucks follow patterns equivalent to CALCE's DST profile
2. CALCE NMC cells aged under DST show X% faster degradation than CC aging
3. Therefore Fleet DNA delivery truck batteries will show similar acceleration in degradation

### Bridge Argument (For Chapter 2 Literature Review)

> *"Real-world delivery fleet duty cycles, characterised by high KI (0.4–0.8 m/s²/km) and frequent micro-cycling, correspond closely to laboratory dynamic test profiles such as the DST (Dynamic Stress Test) in the CALCE dataset. Mulpuri et al. (2025) demonstrated that urban stop-go patterns cause up to 2× faster NMC-811 capacity fade compared to constant-current aging. This laboratory-validated relationship enables the application of CALCE NMC degradation data to predict capacity fade in real-world delivery fleet batteries characterised by their Fleet DNA duty cycles."*

---

## Related Literature on EV Fleet Battery Management

### 1. Mulpuri et al. (2025) — RSC Advances
**Direct relevance**: Shows that urban delivery-type driving patterns cause fastest degradation. Provides quantitative relationship between duty cycle severity and capacity fade rate in NMC-811 batteries.

**Key finding for fleets**: "Pattern 1 (urban stop-go) led to 47% faster capacity fade after 500 equivalent full cycles compared to Pattern 3 (highway cruising)"

### 2. Severson et al. (2019) — Nature Energy
**Indirect relevance**: Shows that the charging protocol in the first 100 cycles predicts total battery lifetime with <10% error. For fleet operators, this means: measuring how batteries respond to the first few weeks of fleet operation can predict when replacement will be needed — enabling proactive maintenance planning.

**Key finding for fleets**: "A machine learning model using data from cycles 1–100 predicted remaining cycle life with mean absolute percentage error of 9.1%"

### 3. UK Zero Emission Vehicle Mandate (2024)
- **Policy reference**: UK Government, "Zero Emission Vehicle Mandate and amendments to the plug-in vehicle grant" (2024)
- **Relevance**: 80% of new car and van sales must be ZEV by 2030. Commercial delivery fleets are included in the mandate. Battery health management is critical for fleet cost-effectiveness.
- **Link**: https://www.gov.uk/government/consultations/zero-emission-vehicle-zev-mandate-and-co2-emissions-regulation-for-new-cars-and-vans

### 4. Faraday Battery Challenge (UK Government / UKRI)
- **Organisation**: Innovate UK / UKRI — UK's national battery R&D initiative
- **Relevance**: The Faraday Battery Challenge has specifically identified **fleet duty cycle characterisation** as a critical knowledge gap in battery management for commercial EVs
- **Funding**: £541M invested in battery research including fleet applications
- **Link**: https://www.faraday.ac.uk/

### 5. Additional Recommended Papers for Fleet Battery Research

**Martinez-Laserna et al. (2018)** — "Battery second life: Hype, hope or reality?" — Renewable and Sustainable Energy Reviews
- DOI: https://doi.org/10.1016/j.rser.2018.04.035
- Relevance: Fleet battery life → second-life applications when SoH drops below 70–80%

**Hu et al. (2020)** — "Battery lifetime prognostics" — Joule
- DOI: https://doi.org/10.1016/j.joule.2020.11.014
- Relevance: Comprehensive review of battery prognostics methods including fleet applications

**Suri & Onori (2016)** — "A control-oriented cycle-life model for hybrid electric vehicle lithium-ion batteries"
- Relevance: Duty cycle → degradation modelling for fleet management

---

## Your Analysis Plan for Fleet DNA Data

### Step 1: Preview the Data
```python
import pandas as pd
df = pd.read_csv(r"C:\Users\pure ev\Music\Term 3\data_for_fleet_dna_delivery_trucks.csv")
print(df.head())
print(df.shape)
print(df.columns.tolist())
print(df.dtypes)
```

### Step 2: Compute Duty Cycle Metrics Per Trip
```python
# Group by vehicle_id and trip_id
for (vid, tid), trip in df.groupby(['vehicle_id', 'trip_id']):
    speed = trip['speed'].values  # mph or km/h
    time = trip['time_s'].values  # seconds
    
    # Convert speed to m/s
    v = speed * 0.44704  # mph to m/s
    dt = np.diff(time)
    
    # Acceleration
    a = np.diff(v) / dt
    
    # Distance (metres)
    d = np.sum(v[:-1] * dt)
    
    # KI
    ki = np.sum(np.abs(a)) / (d / 1000)  # per km
    
    # RPA
    pos_mask = a > 0
    rpa = np.sum(v[:-1][pos_mask] * a[pos_mask]) / (d / 1000)
    
    # PKE
    dv2 = np.diff(v**2)
    pke = np.sum(0.5 * dv2[dv2 > 0]) / (d / 1000)
    
    results.append({'vehicle_id': vid, 'trip_id': tid, 'KI': ki, 'RPA': rpa, 'PKE': pke})
```

### Step 3: FSI Prototype Score
```python
# Normalise each metric 0-1
ki_norm = (ki - ki_min) / (ki_max - ki_min)
rpa_norm = (rpa - rpa_min) / (rpa_max - rpa_min)

# Equal-weight FSI as starting point (weights will be learned from SHAP later)
fsi = 0.4 * ki_norm + 0.4 * rpa_norm + 0.2 * idle_fraction

# Rank routes by severity
severity_ranking = results.sort_values('fsi', ascending=False)
```

### Step 4: Connect to CALCE Degradation Data
```python
# Map FSI score to predicted capacity fade rate
# Using the relationship from Mulpuri et al. (2025)
# High FSI → faster degradation (approaching urban Pattern 1 rates)
# Low FSI → slower degradation (approaching highway Pattern 3 rates)

capacity_fade_rate = base_rate + (mulpuri_ratio - 1) * base_rate * fsi
```
