"""
2D Flocking with Goal Target - Built on REAL FlockingUtils foundation
Extended to add goal-seeking behavior while keeping all the proven research algorithms intact!
"""
import time
import argparse
import numpy as np
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
    
    # Convert to integers and clip to map bounds
    map_x = np.clip(map_x.astype(int), 0, gradient_map.shape[0] - 1)
    map_y = np.clip(map_y.astype(int), 0, gradient_map.shape[1] - 1)
    
    # Read gradient values from map
    grad_vals = gradient_map[map_x, map_y]
    
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

# Configuration
DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_PLOT = False
DEFAULT_USER_DEBUG_GUI = False
# PyBullet calculates physics 120 times per second
# Each physics step = 1/120 = 0.0083 seconds of simulated time
# Higher frequency = more accurate physics (collision detection, aerodynamics)
DEFAULT_SIMULATION_FREQ_HZ = 240 
# Your controller runs 24 times per second
# Drones make decisions every 1/24 = 0.042 seconds
# This is when they:
# Read light intensity
# Calculate flocking forces
# Update motor commands
DEFAULT_CONTROL_FREQ_HZ = 48
# 24 Hz control on 120 Hz simulation:
# - Every 5 physics steps = 1 control update
# - Step 0-4: Apply same motor commands
# - Step 5: NEW control decision + motor update
# - Step 6-9: Apply same motor commands
# - Step 10: NEW control decision + motor update

DEFAULT_OUTPUT_FOLDER = '/Users/kiandrew/Desktop/Capstone/PyBullet/results_2d_1'
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

NUM_DRONES = 5
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

            # --- 2. Calculate Proximal (Separation) Forces ---
            # Loop through all other agents to calculate pair-wise forces
            for j in range(self.n_agents):
                if i == j:
                    continue

                # Calculate distance and angle between agent i and agent j
                dist_x = pos_xs[j] - pos_xs[i]
                dist_y = pos_ys[j] - pos_ys[i]
                distance = np.sqrt(dist_x**2 + dist_y**2)
                
                # Consider only neighbors within the sensing range Dp
                if distance < self.Dp:
                    ij_ang = np.arctan2(dist_y, dist_x)
                    
                    # Equation (1) from swarm_vu.c: Proximal force calculation
                    # This is the Lennard-Jones potential-based force
                    force_magnitude = -self.epsilon * (
                        (2 * (su**4 / distance**5)) - (su**2 / distance**3)
                    )
                    
                    # Accumulate the force components
                    px += force_magnitude * np.cos(ij_ang)
                    py += force_magnitude * np.sin(ij_ang)

            # --- 3. Calculate Alignment and other forces (Simplified for now) ---
            # For this step, we are focusing on the proximal forces which are driven by 'su'.
            # A full implementation would include alignment (beta*hx) and boundary (gama*rx) forces.
            hx = 0.0 # Placeholder
            hy = 0.0 # Placeholder
            rx = 0.0 # Placeholder
            ry = 0.0 # Placeholder

            # --- 4. Calculate Total Force ---
            # Equation (10) from swarm_vu.c (simplified)
            fx_raw = self.alpha * px #+ self.beta * hx + self.gama * rx
            fy_raw = self.alpha * py #+ self.beta * hy + self.gama * ry

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

def create_drone_position_overlay(final_pos_x, final_pos_y, output_folder, timestamp=None, show_plot=False):
    """
    Overlays drone positions on the gradient map.
    
    Args:
        final_pos_x: Array of X positions of drones.
        final_pos_y: Array of Y positions of drones.
        output_folder: Directory to save the output image.
        timestamp: Optional timestamp string for filename (e.g., "10s", "20s", "final")
        show_plot: Whether to display the plot (default: False)
    """
    if timestamp is None:
        print("\n📸 Creating overlay of final drone positions on gradient map...")
        filename = "final_positions_overlay.png"
    else:
        print(f"📸 Snapshot at {timestamp}...")
        filename = f"positions_at_{timestamp}.png"
    
    try:
        # Load the gradient map image
        img = Image.open(GRADIENT_MAP_PATH).convert("RGB")
        draw = ImageDraw.Draw(img)
        
        # Calculate coordinate mapping constants (same as in read_light_intensity)
        step_size = 0.04
        grad_const_x = (len(np.arange(start=0.00, stop=WORLD_SIZE_X, step=step_size))) / WORLD_SIZE_X
        grad_const_y = (len(np.arange(start=0.00, stop=WORLD_SIZE_Y, step=step_size))) / WORLD_SIZE_Y
        
        # Draw each drone's position
        for i in range(len(final_pos_x)):
            pybullet_x = final_pos_x[i]
            pybullet_y = final_pos_y[i]
            
            # Convert PyBullet coords to image pixel coords (matching read_light_intensity)
            # Use the same coordinate swap as the original research implementation
            map_y = int(np.ceil(pybullet_x * grad_const_y))
            map_x = int(np.ceil(pybullet_y * grad_const_x))
            
            # Invert Y-axis: PyBullet Y increases upward, but image rows increase downward
            map_x = img.height - 1 - map_x
            
            # Clip to image bounds to be safe
            map_x_clipped = np.clip(map_x, 0, img.height - 1)
            map_y_clipped = np.clip(map_y, 0, img.width - 1)
            
            # Draw a circle for the drone
            radius = 5
            # Note the coordinate swap: Pillow uses (x,y) which is (width, height)
            draw.ellipse(
                (map_y_clipped - radius, map_x_clipped - radius, map_y_clipped + radius, map_x_clipped + radius),
                fill='red',
                outline='white'
            )
            
        # Save the image
        output_path = os.path.join(output_folder, filename)
        img.save(output_path)
        print(f"✅ Saved to: {output_path}")
        
        # Display the image only if requested
        if show_plot:
            plt.imshow(img)
            plt.title(f"Drone Positions on Gradient Map ({timestamp or 'Final'})")
            plt.xlabel("Image Pixels (PyBullet Y -> Image X)")
            plt.ylabel("Image Pixels (PyBullet X -> Image Y)")
            plt.show()
        
    except FileNotFoundError:
        print(f"[ERROR] Could not create overlay. Gradient map not found at {GRADIENT_MAP_PATH}")
    except Exception as e:
        print(f"[ERROR] An error occurred while creating the overlay: {e}")
        

def run(duration_sec=DURATION_SEC):
    print("=== 2D Flocking with Gradient Following (Built on REAL FlockingUtils) ===")
    print("This extends the proven FlockingUtils foundation")
    print("by adding light sensor simulation and gradient following behavior!")

    
    # Set performance mode (change this to "fast" for maximum speed!)
    # set_performance_mode("fast")  # Options: "fast", "balanced", "accurate"
    set_performance_mode("headless_accurate")
    # set_performance_mode("accurate")

    # Create 2D wrapper with gradient following capability
    f_util = FlockingUtils2DWithLightSensor(
        n_agents=NUM_DRONES,
        center_x=init_center_x, center_y=init_center_y, center_z=init_center_z, 
        spacing=spacing
    )
    pos_xs, pos_ys, pos_zs, pos_h_xc, pos_h_yc, pos_h_zc = f_util.initialize_positions()

    # Create 3D positions for environment (but Z will be constrained)
    INIT_XYZ = np.zeros([NUM_DRONES, 3])
    INIT_XYZ[:, 0] = pos_xs
    INIT_XYZ[:, 1] = pos_ys  
    INIT_XYZ[:, 2] = pos_zs  # All should be FIXED_HEIGHT
    INIT_RPY = np.array([[.0, .0, .0] for _ in range(NUM_DRONES)])

    # Create environment with performance optimizations
    env = CtrlAviary(
        drone_model=DEFAULT_DRONES,
        num_drones=NUM_DRONES,
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
    ctrl = [DSLPIDControl(drone_model=DEFAULT_DRONES) for i in range(NUM_DRONES)]

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=OPTIMIZED_CONTROL_FREQ_HZ,
                    num_drones=NUM_DRONES,
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
    action = np.zeros((NUM_DRONES, 4))
    
    # Calculate expected simulation time
    expected_simulation_time = duration_sec
    print(f"⏱️  Expected simulation duration: {expected_simulation_time} seconds")
    
    # Simple data collection - just time and average light intensity
    time_data = []
    avg_light_intensity_data = []
    
    # Create subfolder for position snapshots
    snapshots_folder = os.path.join(DEFAULT_OUTPUT_FOLDER, "position_snapshots")
    os.makedirs(snapshots_folder, exist_ok=True)
    print(f"📁 Snapshots will be saved to: {snapshots_folder}")
    
    # Snapshot interval (every 10 seconds)
    SNAPSHOT_INTERVAL = 10  # seconds
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
        pos_x = np.zeros(NUM_DRONES)
        pos_y = np.zeros(NUM_DRONES)
        pos_z = np.zeros(NUM_DRONES)

        for j in range(NUM_DRONES):
            states = env._getDroneStateVector(j)
            pos_x[j] = states[0]
            pos_y[j] = states[1]
            pos_z[j] = FIXED_HEIGHT  # Force Z to be constant!

        # NEW: Compute 2D flocking forces WITH LIGHT SENSOR SIMULATION
        velocities_2d = f_util.compute_2d_flocking_forces_with_light_sensor(pos_x, pos_y, pos_z)
        pos_hxs, pos_hys, pos_hzs = f_util.get_heading()
        f_util.update_heading()
        
        # Collect light intensity data every simulation step
        current_time = i / env.CTRL_FREQ
        light_readings = [read_light_intensity(pos_x[j], pos_y[j], add_noise=False) for j in range(NUM_DRONES)]
        avg_light_intensity = np.mean(light_readings)
        
        # Store data
        time_data.append(current_time)
        avg_light_intensity_data.append(avg_light_intensity)
        
        # Take snapshot every 10 seconds
        if current_time - last_snapshot_time >= SNAPSHOT_INTERVAL:
            timestamp_str = f"{int(current_time)}s"
            create_drone_position_overlay(pos_x, pos_y, snapshots_folder, timestamp=timestamp_str, show_plot=False)
            last_snapshot_time = current_time
        
        # Show progress every 3 seconds
        if i % (env.CTRL_FREQ * 3) == 0:
# f_util.plot_swarm(pos_x, pos_y, pos_z, pos_hxs, pos_hys, pos_hzs)
            # Print light intensity readings for debugging
            print(f"⏱️  Time: {current_time:.1f}s | 💡 Avg light intensity: {avg_light_intensity:.1f}")

        # Apply control for each drone (same structure as before)
        for j in range(NUM_DRONES):
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
    
    print(f"\n⏱️  TIMING ANALYSIS:")
    print(f"   Simulation time: {expected_simulation_time:.1f} seconds")
    print(f"   Real-world time: {actual_real_time:.1f} seconds")
    
    # Create overlay of final positions (save to both locations)
    print("\n📸 Creating final position overlays...")
    create_drone_position_overlay(pos_x, pos_y, DEFAULT_OUTPUT_FOLDER, timestamp="final", show_plot=False)
    create_drone_position_overlay(pos_x, pos_y, snapshots_folder, timestamp="final", show_plot=True)
    
    # Plot average light intensity over time
    print(f"\n📊 Creating light intensity plot...")
    plt.figure(figsize=(10, 6))
    plt.plot(time_data, avg_light_intensity_data, linewidth=2, color='blue')
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Average Light Intensity', fontsize=12)
    plt.title('Swarm Average Light Intensity Over Time', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot with absolute path displayed
    os.makedirs(DEFAULT_OUTPUT_FOLDER, exist_ok=True)
    plot_path = os.path.join(DEFAULT_OUTPUT_FOLDER, "bright_to_dark_intensity_plot.png")
    plot_path_absolute = os.path.abspath(plot_path)  # Get absolute path
    plt.savefig(plot_path_absolute, dpi=150)
    print(f"📊 Plot saved to: {plot_path_absolute}")
    
    # Display the plot
    plt.show()
    
    # Final statistics
    final_light_readings = [read_light_intensity(pos_x[j], pos_y[j], add_noise=False) for j in range(NUM_DRONES)]
    avg_final_light = np.mean(final_light_readings)
    print(f"\n💡 Gradient following simulation completed!")
    print(f"📈 Average final light intensity: {avg_final_light:.1f}")

if __name__ == "__main__":
    # Simple command line argument parsing
    parser = argparse.ArgumentParser(description='2D Flocking with Gradient Following')
    parser.add_argument('--duration', type=int, default=DURATION_SEC,
                       help=f'Simulation duration in seconds (default: {DURATION_SEC})')
    
    args = parser.parse_args()
    
    run(duration_sec=args.duration)
