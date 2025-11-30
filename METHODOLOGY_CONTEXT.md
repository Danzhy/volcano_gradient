# Experimental Methodology - Complete Context

## Paper Abstract Context
Bio-inspired, gradient-following algorithms have shown success in source-seeking tasks. However, their performance in time-critical navigation along constrained, curvilinear paths—a scenario critical for tasks like disaster evacuation—is not well understood. This paper benchmarks a known gradient-following algorithm in this new "race" context.

---

## 1. SIMULATION PLATFORM

### 1.1 Physics Engine
- **Software**: PyBullet physics engine
- **Physics Update Frequency**: 240 Hz (default) or 120 Hz (performance mode)
  - Each physics step = 1/240 = 0.0042 seconds of simulated time
  - Handles collision detection, aerodynamics, thrust dynamics

### 1.2 Agent Model: Crazyflie 2.1 Nano-Quadrotor
**Physical Parameters** (from `cf2x.urdf`, validated from research literature):
- **Mass**: 27 grams (0.027 kg)
- **Arm Length**: 3.97 cm (0.0397 m)
- **Propeller Radius**: 2.31 cm
- **Thrust-to-Weight Ratio**: 2.25
- **Max Speed**: 30 km/h
- **Moments of Inertia**:
  - Ixx = 1.4×10⁻⁵ kg·m²
  - Iyy = 1.4×10⁻⁵ kg·m²
  - Izz = 2.17×10⁻⁵ kg·m²
- **Thrust Coefficient (kf)**: 3.16×10⁻¹⁰
- **Torque Coefficient (km)**: 7.94×10⁻¹²
- **Motor Configuration**: X-configuration quadrotor
- **PWM Range**: 20,000 to 65,535 (motor control resolution)

**References for Parameters**:
- Julian Forster (2015): *"System Identification of the Crazyflie 2.0 Nano Quadrocopter"*, ETH Zurich
- Research collection: https://www.research-collection.ethz.ch/handle/20.500.11850/214143

### 1.3 Motion Constraints
- **Dimensionality**: 2D planar navigation
- **Fixed Altitude**: 1.0 meter above ground
- **Control Space**: Forward velocity (u) and yaw rate (w)
- **Control Frequency**: 48 Hz (default) or 24 Hz (performance mode)
  - Agents make decisions every 1/48 = 0.021 seconds
  - Each control decision is applied for 5 physics steps (240Hz/48Hz = 5)

---

## 2. ENVIRONMENT SETUP

### 2.1 World Dimensions
- **Length (X-axis)**: 20.0 meters
- **Width (Y-axis)**: 4.0 meters
- **Height (Z-axis)**: Fixed at 1.0 meter

### 2.2 Gradient Field Representation
- **Format**: Grayscale PNG image
- **Resolution**: 500 × 100 pixels
- **Pixel-to-World Mapping**: 0.04 m/pixel (step size from research code)
  - X-direction: 500 pixels / 20.0m = 25 pixels/meter
  - Y-direction: 100 pixels / 4.0m = 25 pixels/meter
- **Gradient Encoding**: Pixel intensity (0-255)
  - 0 = Black (low gradient, goal region)
  - 255 = White (high gradient, avoid region)
- **Sensing Mechanism**: Simulated light intensity sensor at drone position

### 2.3 Path Geometries Tested

#### **Sinusoidal Paths**
Mathematical definition:
```
y(x) = A × sin(f × π × x / L) + H/2
```
Where:
- A = amplitude = H/4 = 1.0m (height/4)
- f = frequency (number of complete cycles)
- L = path length = 20.0m
- H = world height = 4.0m

**Path Characteristics**:
- **Centerline**: Oscillates ±1.0m from center (y = 2.0m)
- **Wavelength (λ)**: λ = L/f = 20.0m/f
  - freq2: λ = 10.0m per cycle
  - freq4: λ = 5.0m per cycle
  - freq8: λ = 2.5m per cycle

#### **Path Width Control (Thickness Parameter)**
Gradient falloff follows exponential decay:
```
intensity(d) = exp(-k × d)
```
Where:
- k = thickness parameter (controls path width)
- d = perpendicular distance from path centerline

**Effective Width** (at half-maximum intensity):
```
width_effective = ln(2) / k ≈ 0.693 / k
```

**Three Thickness Levels**:
1. **thick0.9** (narrow): k = 0.9
   - Effective width ≈ 0.77m (3.1 cm at half-max)
   - Very narrow corridor
   
2. **thick001** (moderate): k = 0.01
   - Effective width ≈ 69m (2.8m visible gradient)
   - Balanced navigation corridor
   
3. **thick000001** (wide): k = 0.000001
   - Effective width ≈ 693,000m (effectively unlimited)
   - Broad, easily detectable gradient

---

## 3. GRADIENT-FOLLOWING ALGORITHM

### 3.1 Algorithm Origin
Based on distributed swarm control algorithms from:
- `swarm_vu.c` (Crazyflie firmware implementation)
- `dm_ds_v2.py` (Dynamic simulation research code)

### 3.2 Control Parameters
**Flocking Forces**:
- **α (alpha)**: 2.0 - Weight for proximal force (attraction/repulsion)
- **β (beta)**: 1.0 (ON) or 0.0 (OFF) - Weight for alignment force
- **ε (epsilon)**: 12.0 - Lennard-Jones potential parameter
- **sb**: 0.3 - Base spacing (meters)
- **sv**: 0.5 - Variable spacing (meters)
- **Dp**: 2.0 - Sensing range for neighbor detection (meters)

**Velocity Control**:
- **K1**: 0.08 - Proportional gain for linear velocity
- **K2**: 0.2 - Proportional gain for angular velocity
- **u_add**: 0.05 - Constant forward velocity bias (m/s)
- **umax**: 0.15 - Maximum linear velocity (m/s)
- **wmax**: 1.5708/3 ≈ 0.524 - Maximum angular velocity (rad/s ≈ 30°/s)

### 3.3 Algorithm Steps (per agent, per control cycle)

#### Step 1: Adaptive Spacing Calculation
```python
light_intensity = read_sensor(x, y)  # Range: 0-255
light_normalized = (255 - light_intensity) / 255.0  # INVERTED
su = sb + light_normalized^0.1 × sv
```
- **High light (255)** → small su → agents aggregate (tight formation)
- **Low light (0)** → large su → agents spread out (loose formation)
- **Result**: Swarm follows gradient from BRIGHT to DARK regions

#### Step 2: Proximal Force (Lennard-Jones Potential)
For each neighbor j within distance Dp:
```python
distance = sqrt((xj - xi)² + (yj - yi)²)
angle_ij = atan2(yj - yi, xj - xi)
force_magnitude = -ε × (2×su⁴/distance⁵ - su²/distance³)
px += force_magnitude × cos(angle_ij)
py += force_magnitude × sin(angle_ij)
```

#### Step 3: Alignment Force (Optional, β=1.0 when enabled)
```python
sum_cos = Σ cos(heading_j) for all neighbors j
sum_sin = Σ sin(heading_j) for all neighbors j
magnitude = sqrt(sum_cos² + sum_sin²)
hx = sum_cos / magnitude
hy = sum_sin / magnitude
```

#### Step 4: Total Force Calculation
```python
fx_world = α × px + β × hx
fy_world = α × py + β × hy

# Transform to body frame
fx_body = fx_world × cos(-θi) - fy_world × sin(-θi)
fy_body = fx_world × sin(-θi) + fy_world × cos(-θi)
```

#### Step 5: Velocity Commands
```python
u = clip(K1 × fx_body + u_add, 0, umax)
w = clip(K2 × fy_body, -wmax, wmax)
```

#### Step 6: State Update
```python
heading += w × dt  # dt = 0.042s (from research code)
vx = u × cos(heading)
vy = u × sin(heading)
```

---

## 4. EXPERIMENTAL DESIGN

### 4.1 Swarm Initialization
**Spatial Layout**: Square grid formation
- **Grid Spacing**: 0.6 meters
- **Position Calculation**:
  ```python
  init_area = 0.6 × sqrt(num_drones)
  grid_x = [center_x + init_area/2 : center_x - init_area/2 : -0.6]
  grid_y = [center_y + init_area/2 : center_y - init_area/2 : -0.6]
  ```
- **Center Position**: (x=0.5m, y=2.0m, z=1.0m)
- **Initial Heading**: Random uniform in [-15°, +15°] (near-aligned start)

### 4.2 Success Criteria

#### **Finish Line Calculation**
Dynamic finish line based on swarm size to account for spatial extent:
```python
# Hexagonal packing estimation
n_rings = ceil((sqrt(12 × num_drones - 3) - 3) / 6)
swarm_radius = n_rings × d_des_max

# Safety tolerance
tolerance = 0.9 × swarm_radius

# Finish line position
finish_line_x = map_length - swarm_radius - tolerance
```

**Example Finish Lines**:
- 7 drones: finish_line_x = 18.67m
- 10 drones: finish_line_x = 17.54m
- 19 drones: finish_line_x = 17.54m
- 37 drones: finish_line_x = 16.41m

#### **Success Definition**
Run is successful if:
```
(final_x_position + tolerance) ≥ finish_line_x
```
within maximum duration

### 4.3 Parameter Sweep Design

#### **Independent Variables**:
1. **Swarm Size (n)**: 7, 10, 19, 37 drones
2. **Path Width (thickness)**: 
   - 0.9 (narrow ≈3cm)
   - 0.01 (moderate ≈2.8m)
   - 0.000001 (wide ≈unlimited)
3. **Path Complexity (frequency)**:
   - freq2 (λ=10.0m, simple)
   - freq4 (λ=5.0m, moderate)
   - freq8 (λ=2.5m, complex)
4. **Alignment**: ON (β=1.0) or OFF (β=0.0)

#### **Dependent Variables**:
- **Success Rate**: Percentage of runs reaching finish line
- **Completion Time**: Time to reach finish line (successful runs only)
- **Time Statistics**: Mean, standard deviation, median
- **Final Position**: Average x-position at simulation end

### 4.4 Experimental Protocol

#### **Baseline Configuration**:
- **Swarm Size**: 7 drones
- **Path**: sine_curve_thick001_freq2
- **Alignment**: ON (β=1.0)
- **Duration**: 300 seconds base (5 minutes) - **auto-scaled by swarm size**
- **Repetitions**: 50 runs per configuration

#### **Adaptive Duration Scaling**:
Larger swarms require more time due to increased coordination overhead and slower consensus formation. Duration is automatically scaled using:

```python
max_duration = base_duration + (num_drones - 7) × scaling_factor
```

Where:
- `base_duration` = 300 seconds (baseline for 7 drones)
- `scaling_factor` = 5 seconds per additional drone
- `min_drones` = 7 (reference swarm size)

**Duration by Swarm Size**:
- **7 drones**: 300s (baseline, 100%)
- **10 drones**: 315s (+5% time allowance)
- **19 drones**: 360s (+20% time allowance)
- **37 drones**: 450s (+50% time allowance)

This ensures fair comparison across swarm sizes - larger swarms get proportionally more time to complete the task.

#### **Statistical Rigor**:
- **Runs per Configuration**: 50
- **Seed Management**: 
  - Base seed: 42
  - Run-specific seed: base_seed + run_number
  - Ensures reproducibility while capturing natural variability
- **Total Experiments**: 
  - Baseline: 1 configuration × 50 runs = 50
  - Swarm size sweep: 4 sizes × 50 runs = 200
  - Thickness sweep: 3 levels × 50 runs = 150
  - Frequency sweep: 3 levels × 50 runs = 150
  - **Total**: ~550 simulation runs

#### **Performance Optimization**:
Two simulation modes:
1. **Headless Accurate** (production):
   - No GUI rendering
   - Physics: 240 Hz
   - Control: 48 Hz
   - Used for batch experiments
   
2. **Balanced** (testing):
   - GUI enabled
   - Physics: 120 Hz
   - Control: 24 Hz
   - Used for visualization/debugging

---

## 5. DATA COLLECTION

### 5.1 Per-Run Metrics
**Recorded every 5 seconds** (snapshot interval):
- Drone positions (x, y, z)
- Drone velocities (vx, vy, vz)
- Heading angles
- Light intensity readings
- Timestamp

### 5.2 Batch Statistics
**Aggregated over 50 runs per configuration**:
- Success rate (percentage)
- Completion time: mean, std, median (successful runs only)
- Final x-position: mean across all runs
- Finish line position (configuration-dependent)

### 5.3 Output Files
**Per Run**:
- `metadata.json`: Configuration and parameters
- `positions.csv`: Time-series position data
- `snapshot_*.png`: Visualization frames (if enabled)

**Per Batch**:
- `batch_summary.json`: Aggregate statistics
- `batch_summary.png`: Distribution plots
- Individual run folders

**Consolidated**:
- `consolidated_results_[date].csv`: All batch results
- `sweep_summary_[date].json`: Parameter sweep metadata

---

## 6. KEY FINDINGS (Sample Data)

### 6.1 Baseline Performance
**Configuration**: 7 drones, thick001, freq2, alignment ON
- **Success Rate**: 100%
- **Mean Time**: 174.33 ± 17.14 seconds
- **Median Time**: 169.85 seconds

### 6.2 Swarm Size Effect
| Size | Duration | Success | Mean Time | Std Dev |
|------|----------|---------|-----------|---------|
| 7    | 300s     | 100%    | 174.33s   | 17.14s  |
| 10   | 315s     | 100%    | 180.17s   | 6.06s   |
| 19   | 360s     | 100%    | 205.10s   | 5.56s   |
| 37   | 450s     | 74%     | 226.24s   | 14.93s  |

**Observation**: Despite longer allowed time, larger swarms are still slower and less reliable. The 37-drone swarm has 50% more time (450s vs 300s) but still fails 26% of runs.

### 6.3 Path Width Effect (thickness)
| Width      | k value   | Success | Mean Time |
|------------|-----------|---------|-----------|
| Narrow     | 0.9       | 8%      | 263.99s   |
| Moderate   | 0.01      | 100%    | 174.33s   |
| Wide       | 0.000001  | 100%    | 173.51s   |

**Observation**: Very narrow paths cause failure

### 6.4 Path Complexity Effect (frequency)
| Complexity | λ (m)  | Success | Mean Time |
|------------|--------|---------|-----------|
| Simple     | 10.0   | 100%    | 174.33s   |
| Moderate   | 5.0    | 100%    | 190.80s   |
| Complex    | 2.5    | 6%      | 262.08s   |

**Observation**: High-frequency paths (sharp turns) cause failure

---

## 7. COMPUTATIONAL REQUIREMENTS

### 7.1 Single Run Performance
- **Simulation Duration**: 300-450 seconds (5-7.5 minutes) depending on swarm size
  - 7 drones: 300s
  - 37 drones: 450s (scaled for coordination overhead)
- **Wall Clock Time** (headless): ~5-10 minutes per run
- **Wall Clock Time** (GUI): ~10-20 minutes per run

### 7.2 Batch Experiment Performance
- **50 runs** (single config): ~6-12 hours depending on swarm size
- **Full sweep** (550+ runs): ~4-7 days
- **Note**: Larger swarms take longer both in simulation time (scaled duration) and computation (more inter-agent calculations)

### 7.3 Hardware Used
- CPU-based physics simulation (PyBullet)
- No GPU required
- Multi-core beneficial for parallel batch runs (not implemented)

---

## 8. REPRODUCIBILITY INFORMATION

### 8.1 Software Versions
- **Python**: 3.x
- **PyBullet**: Latest stable
- **NumPy**: Latest stable
- **Pillow (PIL)**: For image processing
- **Matplotlib**: For visualization

### 8.2 Random Seed Strategy
```python
base_seed = 42
run_seed = base_seed + run_number  # e.g., 42, 43, 44, ...
np.random.seed(run_seed)
```
- Ensures different runs have natural variability
- Allows exact reproduction of any specific run
- All 50 runs in a batch are reproducible

### 8.3 Code Structure
```
gym_pybullet_drones/
├── envs/
│   ├── BaseAviary.py          # Core physics simulation
│   ├── CtrlAviary.py          # Control interface
│   └── CFAviary.py            # Crazyflie-specific
├── assets/
│   └── cf2x.urdf              # Crazyflie physical parameters
├── examples/
│   ├── 2d_flocking_with_real_utils.py    # Main simulation
│   ├── batch_experiments.py              # Batch runner
│   ├── run_parameter_sweep.py            # Sweep orchestrator
│   └── generate_gradients.py             # Path generation
└── maps_gradient/
    └── path_example_20.0_sine_curve_*.png  # Gradient maps
```

---

## 9. LIMITATIONS AND ASSUMPTIONS

### 9.1 Simplifications
1. **2D Constraint**: No altitude variation (fixed at 1.0m)
2. **Perfect Sensors**: No sensor noise in gradient readings (optional noise available)
3. **Collision Detection**: PyBullet default (no custom collision handling)
4. **Communication**: Assumed perfect (no packet loss or latency)

### 9.2 Model Fidelity
- Physics validated for Crazyflie 2.0/2.1 from system ID studies
- Control parameters from actual swarm deployment code
- Gradient following logic directly from research implementations

### 9.3 Environmental Idealization
- No wind or disturbances
- No obstacles (except boundary repulsion)
- Uniform lighting (gradient only)
- Flat ground plane

---

## 10. CITATION CONTEXT

This experimental setup builds upon:
1. **Crazyflie System ID**: Forster (2015) - ETH Zurich
2. **Swarm Control**: `swarm_vu.c` firmware implementation
3. **Gradient Following**: `dm_ds_v2.py` research code
4. **Physics Engine**: PyBullet open-source simulator

Algorithm parameters and implementation directly replicate published swarm control strategies, adapted for the specific "race along gradient path" scenario.

---

## USAGE FOR LLM

When writing the methodology section, use this context to:

1. **Describe the simulation platform** → Section 1
2. **Explain agent modeling** → Section 1.2
3. **Define the environment** → Section 2
4. **Detail the algorithm** → Section 3
5. **Describe experimental design** → Section 4
6. **Explain data collection** → Section 5
7. **Report findings** → Section 6
8. **Discuss reproducibility** → Section 8
9. **Acknowledge limitations** → Section 9

This provides complete technical detail while maintaining scientific rigor and reproducibility standards.
