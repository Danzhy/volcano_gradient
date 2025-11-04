"""
2D Flocking with Goal Target - Built on REAL FlockingUtils foundation
Extended to add goal-seeking behavior while keeping all the proven research algorithms intact!
"""
import time
import argparse
import numpy as np
import math
import sys
import pybullet as p

# Import existing gym-pybullet-drones components
sys.path.append('ants_2024/')
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.Logger import Logger
from ants_2024.flocking_utils import FlockingUtils
import glob

# Gradient Map Configuration - Step 1: Infrastructure Setup
# Inspired by: DynamicSimulationGradFollow/Dynamic Simulaton/dm_ds_v2.py
# This adds gradient map loading without changing any existing behavior
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Import visualization and experiment management modules
from swarm_visualization import create_drone_position_overlay, create_analysis_dashboard, print_simulation_summary
from experiment_data import ExperimentConfig, ExperimentData, check_success

# Gradient map settings (matching dm_ds_v2.py approach)
# GRADIENT_MAP_PATH = "/Users/kiandrew/Desktop/Capstone/PyBullet/gym-pybullet-drones-3DAE/maps_gradient/linear_gradient.png"
# GRADIENT_MAP_PATH = "gym-pybullet-drones-3DAE/maps_gradient/parabolic_funnel.png"
GRADIENT_MAP_PATH = "/Users/kiandrew/Desktop/Capstone/PyBullet/gym-pybullet-drones-3DAE/maps_gradient/parabolic_funnel.png"
GRADIENT_MAP_PATH = "/Users/kiandrew/Desktop/Capstone/PyBullet/gym-pybullet-drones-3DAE/maps_gradient/sine_wave_ramped_with_banks.png"
GRADIENT_MAP_PATH = "/Users/kiandrew/Desktop/Capstone/PyBullet/gym-pybullet-drones-3DAE/maps_gradient/path_example_exponential.png"
GRADIENT_MAP_PATH = "/Users/kiandrew/Desktop/Capstone/PyBullet/gym-pybullet-drones-3DAE/maps_gradient/linear_4x65.png"
GRADIENT_MAP_PATH = "/Users/kiandrew/Desktop/Capstone/PyBullet/gym-pybullet-drones-3DAE/maps_gradient/path_example_sine_curve.png"
# GRADIENT_MAP_PATH = "gym-pybullet-drones-3DAE/maps_gradient/sine_wave_nice_inverted.png"
WORLD_SIZE_X = 6.5  # meters (matches dm_ds_v2.py)
WORLD_SIZE_Y = 4.0  # meters (matches dm_ds_v2.py)

# Load gradient map (will be used in future steps)
try:
    # .convert('L') converts image to grayscale (L = Luminance)
    # This gives us a 2D array where each pixel is a single intensity value (0-255)
    # Perfect for gradient following since we only need scalar light intensity readings
    gradient_map = np.array(Image.open(GRADIENT_MAP_PATH).convert('L')) # gray scale
    print(f"[INFO] Gradient map loaded: {gradient_map.shape} pixels")
    print(f"[INFO] World size: {WORLD_SIZE_X}m x {WORLD_SIZE_Y}m")
except FileNotFoundError:
    print(f"[ERROR] Gradient map not found at {GRADIENT_MAP_PATH}")
    print("[WARNING] Using dummy gradient map for now")
    gradient_map = np.zeros((100, 100))  # Dummy map

# Light Sensor Simulation - Step 2: Infrastructure Setup
# Inspired by: DynamicSimulationGradFollow/Dynamic Simulaton/dm_ds_v2.py
# Simulates light intensity readings from gradient map at drone positions

def read_light_intensity(pybullet_x, pybullet_y, add_noise=True):
    """
    Read light intensity from gradient map at given PyBullet coordinates
    Supports both single drone and multiple drones (arrays)
    
    Args:
        pybullet_x: X coordinate(s) in PyBullet world (meters) - single value or array
        pybullet_y: Y coordinate(s) in PyBullet world (meters) - single value or array
        add_noise: Whether to add realistic sensor noise (default: True)
    
    Returns:
        intensity: Light intensity value(s) (0-255) - single value or array
    """
    # Calculate mapping constants (based on dm_ds_v2.py: step=0.04)
    step_size = 0.04
    grad_const_x = (len(np.arange(start=0.00, stop=WORLD_SIZE_X, step=step_size))) / WORLD_SIZE_X
    grad_const_y = (len(np.arange(start=0.00, stop=WORLD_SIZE_Y, step=step_size))) / WORLD_SIZE_Y
    
    # Convert PyBullet coordinates to map indices (matching dm_ds_v2.py coordinate swap)
    # This coordinate swap IS CORRECT and matches the original research implementation
    map_y = np.ceil(pybullet_x * grad_const_y)  # x->y mapping from dm_ds_v2.py
    map_x = np.ceil(pybullet_y * grad_const_x)  # y->x mapping from dm_ds_v2.py
    
    # Invert Y-axis: PyBullet Y increases upward, but image rows increase downward
    map_x = gradient_map.shape[0] - 1 - map_x
    
    # Check if coordinates are outside map bounds BEFORE clipping
    out_of_bounds = ((map_x < 0) | (map_x >= gradient_map.shape[0]) | 
                     (map_y < 0) | (map_y >= gradient_map.shape[1]))
    
    # Clip to map bounds for reading valid pixels
    map_x_clipped = np.clip(map_x.astype(int), 0, gradient_map.shape[0] - 1)
    map_y_clipped = np.clip(map_y.astype(int), 0, gradient_map.shape[1] - 1)
    
    # Read gradient values from map
    grad_vals = gradient_map[map_x_clipped, map_y_clipped]
    
    # Replace out-of-bounds values with maximum brightness (255 = repulsive for bright→dark swarm)
    grad_vals = np.where(out_of_bounds, 255.0, grad_vals)
    
    # Add realistic sensor noise (inspired by dm_ds_v2.py: g_noise_mag = 0.5)
    if add_noise:
        g_noise_mag = 0.5  # Same noise magnitude as dm_ds_v2.py
        g_noises = -g_noise_mag + 2 * g_noise_mag * np.random.rand(*grad_vals.shape)
        grad_vals = grad_vals + g_noises
    
    return grad_vals

# Test coordinate mapping (will be used in future steps)
print(f"[INFO] Coordinate mapping functions ready")
print(f"[INFO] World bounds: (0,0) to ({WORLD_SIZE_X},{WORLD_SIZE_Y}) meters")
print(f"[INFO] Gradient map bounds: (0,0) to {gradient_map.shape} pixels")


# Finish Line Calculation - Dynamic based on swarm size
def calculate_finish_line(num_drones, sb=0.3, sv=0.5, map_length=6.5, 
                         tolerance=0.20, packing_efficiency=0.85):
    """
    Calculate finish line position based on swarm size and spacing parameters.
    
    Uses hexagonal packing geometry to estimate maximum swarm radius,
    then adds safety margin (tolerance) to ensure swarm doesn't hit walls.
    
    Args:
        num_drones (int): Number of drones in swarm (e.g., 5, 7, 19)
        sb (float): Base spacing parameter (m)
        sv (float): Variable spacing range (m)
        map_length (float): Total map length in X direction (m)
        tolerance (float): Safety buffer beyond swarm radius (m)
        packing_efficiency (float): Real vs theoretical packing (default 0.85)
    
    Returns:
        tuple: (finish_line_x, swarm_radius, n_rings)
            - finish_line_x: X-coordinate for finish line (m)
            - swarm_radius: Predicted maximum swarm radius (m)
            - n_rings: Number of hexagonal rings
    
    Example:
        >>> finish_x, radius, rings = calculate_finish_line(7)
        >>> print(f"7 drones: finish at {finish_x:.2f}m, radius {radius:.2f}m")
        7 drones: finish at 5.29m, radius 0.96m
    """
    # Calculate maximum d_des (in darkest region where swarm is loosest)
    sigma_max = sb + sv  # 0.3 + 0.5 = 0.8m
    d_des_max = sigma_max * math.sqrt(2)  # ≈ 1.13m
    
    # Calculate number of hexagonal rings for this swarm size
    if num_drones == 1:
        n_rings = 0
        swarm_radius = 0.1  # Single drone, minimal radius
    else:
        # Solve 3n² + 3n + 1 = N for n (hexagonal packing formula)
        n = (-3 + math.sqrt(9 + 12*(num_drones - 1))) / 6
        n_rings = math.ceil(n)  # Round up for incomplete outer ring
        
        # # Calculate radius with packing efficiency correction
        # swarm_radius = n_rings * d_des_max * packing_efficiency
        # # Calculate radius without packing efficiency correction
        swarm_radius = n_rings * d_des_max 
    
    # Calculate finish line: map_length - radius - tolerance
    finish_line_x = map_length - swarm_radius - tolerance
    
    # Validation: Ensure finish line is reasonable
    if finish_line_x < map_length * 0.75:
        print(f"⚠️  WARNING: Finish line at {finish_line_x:.2f}m is less than halfway!")
        print(f"    Consider: smaller swarm, larger map, or smaller tolerance")
    
    # Debug output
    print(f"📐 Finish Line Calculation:")
    print(f"   Swarm: {num_drones} drones in {n_rings} hexagonal rings")
    print(f"   d_des_max: {d_des_max:.3f}m (σ_max={sigma_max:.1f}m)")
    print(f"   Predicted radius: {swarm_radius:.3f}m (theoretical)")
    print(f"   Safety tolerance: {tolerance:.3f}m")
    print(f"   Finish line: X = {finish_line_x:.3f}m")
    
    return finish_line_x, swarm_radius, n_rings


# Configuration
DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_PLOT = False
DEFAULT_USER_DEBUG_GUI = False
# PyBullet calculates physics 240 times per second
# Each physics step = 1/240 = 0.0042 seconds of simulated time
# Higher frequency = more accurate physics (collision detection, aerodynamics)
DEFAULT_SIMULATION_FREQ_HZ = 240 
# Your controller runs 48 times per second
# Drones make decisions every 1/48 = 0.021 seconds
# This is when they:
# Read light intensity
# Calculate flocking forces
# Update motor commands
DEFAULT_CONTROL_FREQ_HZ = 48
# 48 Hz control on 240 Hz simulation:
# - Every 5 physics steps = 1 control update
# - Step 0-4: Apply same motor commands
# - Step 5: NEW control decision + motor update
# - Step 6-9: Apply same motor commands
# - Step 10: NEW control decision + motor update

# Check if running as part of batch experiment
DEFAULT_OUTPUT_FOLDER = os.environ.get('BATCH_OUTPUT_FOLDER', 'results_data_stored')
# DURATION_SEC = 120
DURATION_SEC = 240
# DURATION_SEC = 480

# Performance optimization configuration
ENABLE_HEADLESS_MODE = True   # Set to True for maximum speed (no GUI)
SIMULATION_SPEEDUP = 1.0      # Keep at 1.0 - we'll optimize differently

# Frequency optimization (can be adjusted for speed vs accuracy tradeoff)
OPTIMIZED_SIMULATION_FREQ_HZ = 120  # Reduced from 240 for better performance
OPTIMIZED_CONTROL_FREQ_HZ = 24      # Reduced from 48 for better performance

# Performance mode configuration
PERFORMANCE_MODES = {
    "accurate": {
        "headless": False,
        "sim_freq": 240,
        "ctrl_freq": 48
    },
    "balanced": {
        "headless": False,
        "sim_freq": 120,
        "ctrl_freq": 24
    },
    "fast": {
        "headless": True,
        "sim_freq": 120,
        "ctrl_freq": 24
    },
    "headless_accurate": {
        "headless": True,
        "sim_freq": 240,
        "ctrl_freq": 48
    }
}

# Current performance mode settings (will be set by set_performance_mode())
CURRENT_MODE_SETTINGS = PERFORMANCE_MODES["accurate"]  # Default

def set_performance_mode(mode="accurate"):
    """
    Set performance mode for simulation.
    
    Args:
        mode: "accurate" (GUI, high freq), "balanced" (GUI, lower freq), or "fast" (headless, lower freq)
    """
    global CURRENT_MODE_SETTINGS, ENABLE_HEADLESS_MODE, OPTIMIZED_SIMULATION_FREQ_HZ, OPTIMIZED_CONTROL_FREQ_HZ
    
    if mode not in PERFORMANCE_MODES:
        print(f"⚠️  Unknown mode '{mode}', using 'accurate'")
        mode = "accurate"
    
    CURRENT_MODE_SETTINGS = PERFORMANCE_MODES[mode]
    ENABLE_HEADLESS_MODE = CURRENT_MODE_SETTINGS["headless"]
    OPTIMIZED_SIMULATION_FREQ_HZ = CURRENT_MODE_SETTINGS["sim_freq"]
    OPTIMIZED_CONTROL_FREQ_HZ = CURRENT_MODE_SETTINGS["ctrl_freq"]
    
    print(f"🎯 Performance mode set to: {mode.upper()}")
    print(f"   - Headless: {ENABLE_HEADLESS_MODE}")
    print(f"   - Simulation freq: {OPTIMIZED_SIMULATION_FREQ_HZ} Hz")
    print(f"   - Control freq: {OPTIMIZED_CONTROL_FREQ_HZ} Hz")

# NUM_DRONES = 5
NUM_DRONES = 19
FIXED_HEIGHT = 1.0  # All drones stay at this Z height

# p.setRealTimeSumiulation(0)

# Starting position for swarm
init_center_x = 0.1
init_center_y = 1.8
init_center_z = FIXED_HEIGHT  # Use our fixed height
spacing = 0.8

# No goal settings needed - using gradient following instead

class FlockingUtils2DWithLightSensor:
    """
    2D Flocking with Gradient Following - Re-implemented from original research code.
    This class now contains the flocking logic from swarm_vu.c and dm_ds_v2.py,
    removing the dependency on the incorrect ants_2024.flocking_utils.
    """
    
    def __init__(self, n_agents, center_x, center_y, center_z, spacing):
        self.n_agents = n_agents
        self.center_x = center_x
        self.center_y = center_y
        self.fixed_z = center_z
        self.spacing = spacing

        # --- Parameters from swarm_vu.c and dm_ds_v2.py ---
        self.alpha = 2.0     # Weight for proximal force
        self.beta = 1.0      # Weight for alignment force
        self.gama = 1.0      # Weight for boundary repulsion
        self.epsilon = 12.0  # Lennard-Jones potential parameter
        self.sb = 0.3        # Base spacing for su
        self.sv = 0.5        # Variable spacing for su
        self.K1 = 0.08       # Proportional gain for linear velocity u
        self.K2 = 0.2        # Proportional gain for angular velocity w
        self.u_add = 0.05    # Constant forward velocity push
        self.umax = 0.15     # Max linear velocity
        self.wmax = 1.5708/3 # Max angular velocity
        self.Dp = 2.0        # Sensing range for neighbor interaction

        # Drone state variables
        # self.headings = np.random.rand(n_agents) * 2 * np.pi # Initialize with random headings
        # Initialize with aligned headings
        self.headings = np.random.uniform(-np.pi/12, np.pi/12, n_agents)

        print(f"💡 Created 2D FlockingUtils with Research-Aligned Gradient Following")
        print(f"   - Re-implementing logic from swarm_vu.c and dm_ds_v2.py")
        print(f"   - Constraining all drones to Z = {self.fixed_z}")

    def initialize_positions(self):
        """Initialize positions but ensure Z is fixed"""

        # --- FIX: Replace old dependency with direct initialization ---
        # This logic is inspired by the initialization in dm_ds_v2.py
        
        num_agents = self.n_agents
        init_area = 0.6 * np.sqrt(num_agents)
        mem = 0
        finish = 0
        
        ii = np.arange(self.center_x + init_area / 2, self.center_x - init_area / 2 - 0.1, -0.6)
        jj = np.arange(self.center_y + init_area / 2, self.center_y - init_area / 2 - 0.1, -0.6)

        pos_xs = np.zeros(num_agents)
        pos_ys = np.zeros(num_agents)

        for i in range(len(ii)):
            for j in range(len(jj)):
                if not finish:
                    pos_xs[mem] = ii[i]
                    pos_ys[mem] = jj[j]
                    mem += 1
                if mem == num_agents:
                    finish = 1
        
        pos_zs = np.full(num_agents, self.fixed_z)
        pos_h_xc = np.zeros(num_agents) # Not used in 2D flocking
        pos_h_yc = np.zeros(num_agents) # Not used in 2D flocking
        pos_h_zc = np.zeros(num_agents) # Not used in 2D flocking
        
        # Force all Z positions to be constant
        pos_zs.fill(self.fixed_z)
        
        print(f"📍 Initialized {len(pos_xs)} drones in 2D:")
        for i in range(len(pos_xs)):
            print(f"   Drone {i}: ({pos_xs[i]:.2f}, {pos_ys[i]:.2f}, {pos_zs[i]:.2f})")
        
        return pos_xs, pos_ys, pos_zs, pos_h_xc, pos_h_yc, pos_h_zc
    
    def compute_2d_flocking_forces_with_light_sensor(self, pos_xs, pos_ys, pos_zs):
        """
        Computes flocking forces using the exact logic from swarm_vu.c.
        This replaces the dependency on the external FlockingUtils class.
        """
        # Final velocities to be returned
        velocities_2d = np.zeros((self.n_agents, 3))
        
        # --- Re-implementation of research code logic ---
        for i in range(self.n_agents):
            # Reset forces for the current agent
            px = 0.0  # Proximal force x
            py = 0.0  # Proximal force y
            
            # --- 1. Calculate Adaptive Spacing 'su' ---
            light_intensity = read_light_intensity(pos_xs[i], pos_ys[i], add_noise=True)
            light_capped = np.clip(light_intensity, 0.0, 255.0)
            # light_normalized = (light_capped - 0.0) / (255.0 - 0.0)
            # INVERTED: High light → low normalized value → small su → drones aggregate
            # Low light → high normalized value → large su → drones spread out
            # Result: Swarm follows from BRIGHT to DARK
            light_normalized = (255.0 - light_capped) / 255.0
            # calculate adaptive spacing: lower light = larger spacing
            su = self.sb + np.power(light_normalized, 0.1) * self.sv

            # --- 2. Calculate Proximal & Alignment Forces (Combined Loop) ---
            # Initialize alignment force accumulators
            sum_cosh = np.cos(self.headings[i])
            sum_sinh = np.sin(self.headings[i])
            
            # Single loop through all neighbors for BOTH proximal and alignment forces
            for j in range(self.n_agents):
                if i == j:
                    continue

                # Calculate distance and angle between agent i and agent j
                dist_x = pos_xs[j] - pos_xs[i]
                dist_y = pos_ys[j] - pos_ys[i]
                distance = np.sqrt(dist_x**2 + dist_y**2)
                
                # Consider only neighbors within the sensing range Dp
                if distance < self.Dp:
                    # ---- Proximal Force Calculation ----
                    ij_ang = np.arctan2(dist_y, dist_x)
                    
                    # Equation (1) from swarm_vu.c: Lennard-Jones potential
                    force_magnitude = -self.epsilon * (
                        (2 * (su**4 / distance**5)) - (su**2 / distance**3)
                    )
                    
                    # Accumulate proximal force components
                    px += force_magnitude * np.cos(ij_ang)
                    py += force_magnitude * np.sin(ij_ang)
                    
                    # ---- Alignment Force Calculation ----
                    # Accumulate neighbor headings for alignment
                    sum_cosh += np.cos(self.headings[j])
                    sum_sinh += np.sin(self.headings[j])
            
            # Calculate average heading direction (unit vector)
            heading_magnitude = np.sqrt(sum_cosh**2 + sum_sinh**2)
            if heading_magnitude > 0:
                hx = sum_cosh / heading_magnitude
                hy = sum_sinh / heading_magnitude
            else:
                hx = 0.0
                hy = 0.0
            
            # Boundary repulsion not needed (using bright outside boundaries)
            rx = 0.0
            ry = 0.0

            # --- 4. Calculate Total Force ---
            # Equation (10) from swarm_vu.c
            fx_raw = self.alpha * px + self.beta * hx
            fy_raw = self.alpha * py + self.beta * hy

            # Transform force to the agent's body frame
            fx = fx_raw * np.cos(-self.headings[i]) - fy_raw * np.sin(-self.headings[i])
            fy = fx_raw * np.sin(-self.headings[i]) + fy_raw * np.cos(-self.headings[i])
            
            # --- 5. Calculate Linear and Angular Velocity ---
            # Equation (11) from swarm_vu.c
            u = self.K1 * fx + self.u_add
            w = self.K2 * fy
            
            # Clip velocities to their maximum values
            u = np.clip(u, 0, self.umax)
            w = np.clip(w, -self.wmax, self.wmax)

            # --- 6. Update Agent State ---
            # Update heading based on angular velocity
            self.headings[i] += w * 0.042 # Using dt from research code
            
            # Convert linear and angular velocity to world frame velocity vector
            base_vx = u * np.cos(self.headings[i])
            base_vy = u * np.sin(self.headings[i])

            velocities_2d[i, 0] = base_vx
            velocities_2d[i, 1] = base_vy
        
        return velocities_2d

    def update_heading(self):
        """
        Heading is now updated internally during the force calculation.
        This method is kept for API compatibility but does nothing.
        """
        pass
    
    def get_heading(self):
        """
        Returns the current headings of the drones.
        """
        # This function needs to return 3 values for compatibility, returning dummy values for hz
        return self.headings, np.zeros(self.n_agents), np.zeros(self.n_agents)

def run(duration_sec=DURATION_SEC, seed=None, run_number=None, base_seed=42, 
        num_drones=NUM_DRONES, map_length=WORLD_SIZE_X):
    """
    Run 2D flocking simulation with gradient following.
    
    Args:
        duration_sec: Simulation duration in seconds
        seed: Explicit random seed (overrides base_seed + run_number)
        run_number: Run number for batch experiments (combined with base_seed)
        base_seed: Base seed for all experiments (default: 42)
        num_drones: Number of drones in swarm (default: NUM_DRONES)
        map_length: Map length in X direction for finish line calculation (default: WORLD_SIZE_X)
    """
    print("=== 2D Flocking with Gradient Following (Built on REAL FlockingUtils) ===")
    print("This extends the proven FlockingUtils foundation")
    print("by adding light sensor simulation and gradient following behavior!")

    # ============================================
    # SEED MANAGEMENT FOR REPRODUCIBILITY
    # ============================================
    
    # Determine actual seed (Option 3: explicit > run_number > base_seed)
    if seed is not None:
        # Explicit seed provided (for reproducing specific run)
        actual_seed = seed
        print(f"🎲 Using explicit seed: {actual_seed}")
    elif run_number is not None:
        # Batch experiment: derive from base_seed + run_number
        # This gives each run a different (but reproducible) seed for natural variability
        actual_seed = base_seed + run_number
        print(f"🎲 Batch run {run_number}: seed = {actual_seed} (base={base_seed})")
    else:
        # Single run: use base_seed
        actual_seed = base_seed
        print(f"🎲 Using base seed: {actual_seed}")
    
    # Set numpy random seed for reproducibility
    np.random.seed(actual_seed)
    print(f"   → All random operations (initial headings, sensor noise) will be reproducible")
    
    # ============================================
    # FINISH LINE CALCULATION (Dynamic based on swarm size)
    # ============================================
    
    # Calculate finish line position based on swarm geometry
    FINISH_LINE_X, predicted_swarm_radius, n_rings = calculate_finish_line(
        num_drones=num_drones,
        sb=0.3,  # Base spacing (matches FlockingUtils2DWithLightSensor)
        sv=0.5,  # Variable spacing range
        map_length=map_length,
        tolerance=0.20,  # Fixed tolerance for now (can be updated after baseline runs)
        packing_efficiency=0.85  # Empirical correction for real (non-perfect) swarms
    )
    print(f"   → Finish line dynamically set for {num_drones} drones")
    print()
    
    # Set performance mode (change this to "fast" for maximum speed!)
    # set_performance_mode("fast")  # Options: "fast", "balanced", "accurate"
    set_performance_mode("headless_accurate")
    # set_performance_mode("accurate")

    # Create 2D wrapper with gradient following capability
    f_util = FlockingUtils2DWithLightSensor(
        n_agents=num_drones,
        center_x=init_center_x, center_y=init_center_y, center_z=init_center_z, 
        spacing=spacing
    )
    pos_xs, pos_ys, pos_zs, pos_h_xc, pos_h_yc, pos_h_zc = f_util.initialize_positions()

    # Create 3D positions for environment (but Z will be constrained)
    INIT_XYZ = np.zeros([num_drones, 3])
    INIT_XYZ[:, 0] = pos_xs
    INIT_XYZ[:, 1] = pos_ys  
    INIT_XYZ[:, 2] = pos_zs  # All should be FIXED_HEIGHT
    INIT_RPY = np.array([[.0, .0, .0] for _ in range(num_drones)])

    # Create environment with performance optimizations
    env = CtrlAviary(
        drone_model=DEFAULT_DRONES,
        num_drones=num_drones,
        initial_xyzs=INIT_XYZ,
        initial_rpys=INIT_RPY,
        physics=DEFAULT_PHYSICS,
        neighbourhood_radius=10,
        pyb_freq=OPTIMIZED_SIMULATION_FREQ_HZ,  # Use optimized frequency
        ctrl_freq=OPTIMIZED_CONTROL_FREQ_HZ,    # Use optimized frequency
        gui=DEFAULT_GUI and not ENABLE_HEADLESS_MODE,  # Disable GUI in headless mode
        user_debug_gui=False
    )

    # PyBullet performance optimizations
    if not ENABLE_HEADLESS_MODE:
        # Remove shadows for better performance
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        
        # Disable unnecessary visualizations
        p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        
        # Set top-down camera view for 2D visualization
        p.resetDebugVisualizerCamera(
            cameraDistance=8,
            cameraYaw=0,
            cameraPitch=-89,  # Look straight down
            cameraTargetPosition=[3, 2.5, FIXED_HEIGHT]
        )
    else:
        print("🚀 Running in HEADLESS mode for maximum speed!")

    # Create controllers (same as always!)
    ctrl = [DSLPIDControl(drone_model=DEFAULT_DRONES) for i in range(num_drones)]

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=OPTIMIZED_CONTROL_FREQ_HZ,
                    num_drones=num_drones,
                    output_folder=DEFAULT_OUTPUT_FOLDER,
                    )

    print("\n💡 Controls:")
    print("- Q: Quit simulation")
    print("🔍 Watch how drones balance flocking behavior with gradient following!")
    print("📈 Drones will naturally aggregate in areas with higher light intensity")
    if ENABLE_HEADLESS_MODE:
        print("🚀 HEADLESS MODE: Maximum speed, no GUI rendering")
    else:
        print("🖥️  GUI MODE: Visual rendering enabled")

    START = time.time()
    action = np.zeros((num_drones, 4))
    
    # Calculate expected simulation time
    expected_simulation_time = duration_sec
    print(f"⏱️  Expected simulation duration: {expected_simulation_time} seconds")
    
    # ============================================
    # EXPERIMENT DATA TRACKING SETUP
    # ============================================
    
    # Create experiment configuration
    exp_config = ExperimentConfig(
        num_drones=NUM_DRONES,
        init_xyzs=INIT_XYZ,
        alignment_enabled=True,  # Currently always enabled
        desired_spacing=spacing,
        gradient_map_path=GRADIENT_MAP_PATH,
        world_size_x=WORLD_SIZE_X,
        world_size_y=WORLD_SIZE_Y,
        duration_sec=duration_sec,
        snapshot_interval=5,
        gui=DEFAULT_GUI and not ENABLE_HEADLESS_MODE,
        performance_mode="headless_accurate" if ENABLE_HEADLESS_MODE else "accurate",
        finish_line_x=FINISH_LINE_X,  # Dynamically calculated based on swarm size
        finish_line_enabled=True,
        experiment_name="",  # Will be auto-generated
        notes="",
        # Random seed management (for reproducibility)
        random_seed=actual_seed,
        base_seed=base_seed,
        run_number=run_number
    )
    
    # Create experiment data container
    exp_data = ExperimentData(exp_config)
    print(f"📋 Experiment ID: {exp_data.experiment_id}")
    
    # Create experiment-specific folder (all files for this run go here)
    experiment_folder = os.path.join(DEFAULT_OUTPUT_FOLDER, exp_data.experiment_id)
    os.makedirs(experiment_folder, exist_ok=True)
    print(f"📁 Experiment folder: {experiment_folder}")
    
    # Legacy data collection (keep for backward compatibility)
    time_data = []
    centroid_x_data = []
    centroid_y_data = []
    avg_light_intensity_data = []
    distance_from_start_data = []
    speed_data = []
    swarm_radius_data = []
    
    # Initialize tracking variables for distance from start
    initial_centroid_x = None
    initial_centroid_y = None
    
    # Success tracking
    finish_line_crossed = False
    time_to_finish = None
    
    # Check if running in batch mode (skip visualizations for speed)
    BATCH_MODE = os.environ.get('BATCH_MODE', '0') == '1'
    if BATCH_MODE:
        print("🚀 BATCH MODE: Skipping visualization generation for speed")
        print("   (Use regenerate_visualizations.py later to create plots)")
    
    # Create subfolder for position snapshots inside experiment folder (only if not batch mode)
    snapshots_folder = None
    if not BATCH_MODE:
        snapshots_folder = os.path.join(experiment_folder, "position_snapshots")
        os.makedirs(snapshots_folder, exist_ok=True)
        print(f"📁 Snapshots will be saved to: {snapshots_folder}")
    
    # Snapshot interval (every 5 seconds)
    SNAPSHOT_INTERVAL = 5  # seconds
    last_snapshot_time = -SNAPSHOT_INTERVAL  # Force first snapshot at t=0

    # Main simulation loop - gradient following
    for i in range(0, int(duration_sec * env.CTRL_FREQ)):
        # Clear any visual artifacts (same as before)
        p.removeAllUserDebugItems()
        
        # Handle user input for simulation control
        keys = p.getKeyboardEvents()
        
        if ord('q') in keys:
            break

        # Step simulation (same as always!)
        obs, reward, done, info, _ = env.step(action)
        
        # Get current positions from observations
        pos_x = np.zeros(num_drones)
        pos_y = np.zeros(num_drones)
        pos_z = np.zeros(num_drones)

        for j in range(num_drones):
            states = env._getDroneStateVector(j)
            pos_x[j] = states[0]
            pos_y[j] = states[1]
            pos_z[j] = FIXED_HEIGHT  # Force Z to be constant!

        # NEW: Compute 2D flocking forces WITH LIGHT SENSOR SIMULATION
        velocities_2d = f_util.compute_2d_flocking_forces_with_light_sensor(pos_x, pos_y, pos_z)
        pos_hxs, pos_hys, pos_hzs = f_util.get_heading()
        f_util.update_heading()
        
        # Collect comprehensive metrics every simulation step
        current_time = i / env.CTRL_FREQ
        
        # 1. Light intensity
        light_readings = [read_light_intensity(pos_x[j], pos_y[j], add_noise=False) for j in range(num_drones)]
        avg_light_intensity = np.mean(light_readings)
        
        # 2. Swarm centroid position
        centroid_x = np.mean(pos_x)
        centroid_y = np.mean(pos_y)
        
        # Store initial centroid position (first iteration only)
        if initial_centroid_x is None:
            initial_centroid_x = centroid_x
            initial_centroid_y = centroid_y
        
        # 3. Distance from starting point (X-axis displacement to track progression along gradient)
        distance_from_start = centroid_x - initial_centroid_x
        
        # 4. Swarm speed (from velocities)
        speeds = [np.sqrt(velocities_2d[j][0]**2 + velocities_2d[j][1]**2) for j in range(num_drones)]
        avg_speed = np.mean(speeds)
        
        # 5. Swarm cohesion (radius from centroid)
        distances_from_centroid = [np.sqrt((pos_x[j] - centroid_x)**2 + (pos_y[j] - centroid_y)**2) 
                                   for j in range(num_drones)]
        swarm_radius = np.mean(distances_from_centroid)
        
        # Store all metrics (legacy lists)
        time_data.append(current_time)
        centroid_x_data.append(centroid_x)
        centroid_y_data.append(centroid_y)
        avg_light_intensity_data.append(avg_light_intensity)
        distance_from_start_data.append(distance_from_start)
        speed_data.append(avg_speed)
        swarm_radius_data.append(swarm_radius)
        
        # Take snapshot every 5 seconds AND store in experiment data
        if current_time - last_snapshot_time >= SNAPSHOT_INTERVAL:
            timestamp_str = f"{int(current_time)}s"
            
            # Only create visualization if not in batch mode
            if not BATCH_MODE:
                create_drone_position_overlay(
                    pos_x, pos_y, GRADIENT_MAP_PATH, WORLD_SIZE_X, WORLD_SIZE_Y,
                    snapshots_folder, timestamp=timestamp_str, show_plot=False
                )
            
            # Store datapoint in experiment data (with full positions for this snapshot)
            current_positions = np.column_stack([pos_x, pos_y, pos_z])
            exp_data.add_datapoint(
                time=current_time,
                centroid_x=centroid_x,
                centroid_y=centroid_y,
                light_intensity=avg_light_intensity,
                distance_from_start=distance_from_start,
                speed=avg_speed,
                swarm_radius=swarm_radius,
                positions=current_positions
            )
            
            last_snapshot_time = current_time
        
        # Check if finish line crossed (for success metrics)
        if not finish_line_crossed and check_success(centroid_x, exp_config.finish_line_x):
            finish_line_crossed = True
            time_to_finish = current_time
            print(f"\n🎉 FINISH LINE CROSSED at t={time_to_finish:.1f}s! X={centroid_x:.2f}m")
            print(f"🏁 Terminating simulation early (success!)")
            break  # Exit simulation loop immediately
        
        # Show progress every 3 seconds
        if i % (env.CTRL_FREQ * 3) == 0:
            # Print comprehensive metrics for debugging
            print(f"⏱️  Time: {current_time:.1f}s | "
                  f"💡 Light: {avg_light_intensity:.1f} | "
                  f"📍 Centroid: ({centroid_x:.2f}, {centroid_y:.2f}) | "
                  f"📏 Distance: {distance_from_start:.2f}m | "
                  f"🚀 Speed: {avg_speed:.3f}m/s | "
                  f"🎯 Radius: {swarm_radius:.2f}m")

        # Apply control for each drone (same structure as before)
        for j in range(num_drones):
            # Target position: current XY + fixed Z (same as before)
            target_pos = np.array([
                pos_x[j],      # Current X (will be changed by velocity)
                pos_y[j],      # Current Y (will be changed by velocity)
                FIXED_HEIGHT   # Fixed Z height
            ])
            
            # NEW: Velocity command from flocking + goal (X, Y, 0)
            vel_cmd = velocities_2d[j]
            
            # Generate control action (same as always!)
            action[j], _, _ = ctrl[j].computeControlFromState(
                control_timestep=env.CTRL_TIMESTEP,
                state=obs[j],
                target_pos=target_pos,
                target_vel=vel_cmd,
                target_rpy=np.array([0, 0, 0])
            )

        # Render (NO SYNC - maximum speed!)
        env.render()
        if DEFAULT_GUI and not ENABLE_HEADLESS_MODE:
            sync(i, START, env.CTRL_TIMESTEP)

    # Cleanup
    env.close()
    
    # Calculate timing statistics
    END = time.time()
    actual_real_time = END - START
    real_time_factor = expected_simulation_time / actual_real_time if actual_real_time > 0 else 0
    
    # Finalize experiment data with one last snapshot at final time
    final_positions = np.column_stack([pos_x, pos_y, [FIXED_HEIGHT] * num_drones])
    exp_data.add_datapoint(
        time=current_time,
        centroid_x=centroid_x,
        centroid_y=centroid_y,
        light_intensity=avg_light_intensity,
        distance_from_start=distance_from_start,
        speed=avg_speed,
        swarm_radius=swarm_radius,
        positions=final_positions
    )
    
    exp_data.finalize(
        actual_duration=actual_real_time,
        real_time_factor=real_time_factor,
        success=finish_line_crossed,
        time_to_finish=time_to_finish
    )
    
    # Save experiment data to disk (in experiment folder)
    print(f"\n💾 Saving experiment data...")
    exp_data.save(output_dir=experiment_folder)
    
    print(f"\n⏱️  TIMING ANALYSIS:")
    print(f"   Simulation time: {expected_simulation_time:.1f} seconds")
    print(f"   Real-world time: {actual_real_time:.1f} seconds")
    print(f"   Real-time factor: {real_time_factor:.2f}x")
    
    # Skip all visualization creation in batch mode (can regenerate later)
    if not BATCH_MODE:
        # Create overlay of final positions (save to experiment folder)
        print("\n📸 Creating final position overlays...")
        create_drone_position_overlay(
            pos_x, pos_y, GRADIENT_MAP_PATH, WORLD_SIZE_X, WORLD_SIZE_Y, 
            experiment_folder, timestamp="final", show_plot=False
        )
        create_drone_position_overlay(
            pos_x, pos_y, GRADIENT_MAP_PATH, WORLD_SIZE_X, WORLD_SIZE_Y,
            snapshots_folder, timestamp="final", show_plot=True
        )
        
        # Create comprehensive analysis dashboard (save to experiment folder)
        print(f"\n📊 Creating comprehensive analysis dashboard...")
        dashboard_path = create_analysis_dashboard(
            time_data=time_data,
            centroid_x_data=centroid_x_data,
            centroid_y_data=centroid_y_data,
            light_intensity_data=avg_light_intensity_data,
            distance_from_start_data=distance_from_start_data,
            speed_data=speed_data,
            swarm_radius_data=swarm_radius_data,
            gradient_map_path=GRADIENT_MAP_PATH,
            world_size_x=WORLD_SIZE_X,
            world_size_y=WORLD_SIZE_Y,
            output_folder=experiment_folder
        )
        print(f"📊 Dashboard saved to: {dashboard_path}")
    else:
        print("\n⚡ Skipped visualization generation (batch mode)")
        print("   Use regenerate_visualizations.py to create plots later")
    
    # Print comprehensive simulation summary
    print_simulation_summary(
        simulation_time=expected_simulation_time,
        real_time=actual_real_time,
        final_light_intensity=avg_light_intensity_data[-1] if avg_light_intensity_data else 0,
        final_distance=distance_from_start_data[-1] if distance_from_start_data else 0,
        avg_speed=np.mean(speed_data) if speed_data else 0,
        avg_radius=np.mean(swarm_radius_data) if swarm_radius_data else 0
    )
    
    # Print experiment summary with success metrics
    print(exp_data.get_summary())

if __name__ == "__main__":
    # Command line argument parsing
    parser = argparse.ArgumentParser(
        description='2D Flocking with Gradient Following',
        epilog='Examples:\n'
               '  Single run (default seed):     python %(prog)s --duration 240\n'
               '  Specific seed:                 python %(prog)s --duration 240 --seed 123\n'
               '  Batch run (from batch_experiments): python %(prog)s --duration 240 --run-number 5\n'
               '  Reproduce specific run:        python %(prog)s --duration 240 --seed 47',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--duration', type=int, default=DURATION_SEC,
                       help=f'Simulation duration in seconds (default: {DURATION_SEC})')
    parser.add_argument('--seed', type=int, default=None,
                       help='Explicit random seed for reproducibility (overrides --run-number and base seed)')
    parser.add_argument('--run-number', type=int, default=None,
                       help='Run number for batch experiments (seed = base_seed + run_number)')
    parser.add_argument('--base-seed', type=int, default=42,
                       help='Base seed for batch experiments (default: 42)')
    parser.add_argument('--num-drones', type=int, default=NUM_DRONES,
                       help=f'Number of drones in swarm (default: {NUM_DRONES})')
    parser.add_argument('--map-length', type=float, default=WORLD_SIZE_X,
                       help=f'Map length in X direction for finish line calculation (default: {WORLD_SIZE_X}m)')
    
    args = parser.parse_args()
    
    run(duration_sec=args.duration, 
        seed=args.seed,
        run_number=args.run_number,
        base_seed=args.base_seed,
        num_drones=args.num_drones,
        map_length=args.map_length)
