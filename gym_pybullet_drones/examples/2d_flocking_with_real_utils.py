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

# Gradient Map Configuration - Step 1: Infrastructure Setup
# Inspired by: DynamicSimulationGradFollow/Dynamic Simulaton/dm_ds_v2.py
# This adds gradient map loading without changing any existing behavior
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Gradient map settings (matching dm_ds_v2.py approach)
GRADIENT_MAP_PATH = "/Users/kiandrew/Desktop/Capstone/Tugay_Gradient_Pybullet/DynamicSimulationGradFollow/Dynamic Simulaton/linear_4x65.png"
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
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_OUTPUT_FOLDER = 'results_2d_1'
DURATION_SEC = 15

NUM_DRONES = 5
FIXED_HEIGHT = 1.0  # All drones stay at this Z height

# p.setRealTimeSumiulation(0)

# Starting position for swarm
init_center_x = 1.5
init_center_y = 1.5
init_center_z = FIXED_HEIGHT  # Use our fixed height
spacing = 0.8

# No goal settings needed - using gradient following instead

class FlockingUtils2DWithLightSensor:
    """2D Flocking with Gradient Following - Built on proven FlockingUtils foundation"""
    
    def __init__(self, n_agents, center_x, center_y, center_z, spacing):
        # Create the real FlockingUtils (same as before)
        self.flocking_3d = FlockingUtils(n_agents, center_x, center_y, center_z, spacing)
        self.fixed_z = center_z
        
        print(f"💡 Created 2D FlockingUtils with Gradient Following")
        print(f"   - Using proven research algorithms from FlockingUtils")
        print(f"   - Constraining all drones to Z = {self.fixed_z}")
        print(f"   - Added light sensor simulation and adaptive spacing")
    
    def initialize_positions(self):
        """Initialize positions but ensure Z is fixed"""

        # TODO: add a comment here. what are we affecting? is this the drone controller only? and not something else 
        # that might crash?
        pos_xs, pos_ys, pos_zs, pos_h_xc, pos_h_yc, pos_h_zc = self.flocking_3d.initialize_positions() 
        
        # Force all Z positions to be constant
        pos_zs.fill(self.fixed_z)
        pos_h_zc.fill(0.0)  # No Z-axis heading component
        
        print(f"📍 Initialized {len(pos_xs)} drones in 2D:")
        for i in range(len(pos_xs)):
            print(f"   Drone {i}: ({pos_xs[i]:.2f}, {pos_ys[i]:.2f}, {pos_zs[i]:.2f})")
        
        return pos_xs, pos_ys, pos_zs, pos_h_xc, pos_h_yc, pos_h_zc
    
    def compute_2d_flocking_forces_with_light_sensor(self, pos_xs, pos_ys, pos_zs):
        """Compute flocking forces + goal attraction (EXTENDED VERSION)"""
        
        # Force all Z positions to be constant before calculation
        pos_zs_constrained = np.full_like(pos_zs, self.fixed_z)
        
        # --- NEW: Integrate Adaptive Spacing ---
        # Calculate the adaptive spacing 'su' for each drone based on light intensity
        # and update the flocking utility's separation parameter ('sigmas').
        
        # Gradient following parameters (from swarm_vu.c firmware)
        sb = 0.3  # Base spacing
        sv = 0.5  # Variable spacing component
        lmn = 0.0   # Minimum light intensity (grayscale image: 0-255)
        lmx = 255.0 # Maximum light intensity (grayscale image: 0-255)

        # Calculate 'su' for each drone
        adaptive_sigmas = np.zeros(len(pos_xs))
        for i in range(len(pos_xs)):
            light_intensity = read_light_intensity(pos_xs[i], pos_ys[i], add_noise=True)
            light_capped = np.clip(light_intensity, lmn, lmx)
            light_normalized = (light_capped - lmn) / (lmx - lmn)
            su = sb + np.power(light_normalized, 0.1) * sv
            adaptive_sigmas[i] = su
            
        # Update the separation parameter in FlockingUtils with our new values
        self.flocking_3d.update_sigmas(adaptive_sigmas)

        # Use the real FlockingUtils calculations with constrained Z
        self.flocking_3d.calc_dij(pos_xs, pos_ys, pos_zs_constrained)
        self.flocking_3d.calc_ang_ij(pos_xs, pos_ys, pos_zs_constrained)
        self.flocking_3d.calc_grad_vals(pos_xs, pos_ys, pos_zs_constrained)
        self.flocking_3d.calc_p_forces() # separation
        self.flocking_3d.calc_alignment_forces() # alignment
        self.flocking_3d.calc_boun_rep(pos_xs, pos_ys, pos_zs_constrained) # boundary 
        
        # Get velocities but zero out Z component
        u = self.flocking_3d.calc_u_w()
        
        # Convert to 2D velocity commands 
        velocities_2d = np.zeros((len(pos_xs), 3))
        
        for i in range(len(pos_xs)):
            # Get heading from FlockingUtils
            hx, hy, hz = self.flocking_3d.get_heading()
            
            # Base flocking velocities (same as before)
            base_vx = u[i] * np.cos(hx[i])  # X velocity from flocking
            base_vy = u[i] * np.cos(hy[i])  # Y velocity from flocking
            
            # The gradient following is now handled by the adaptive spacing in FlockingUtils,
            # so we don't need any additional velocity components here.
            total_vx = base_vx
            total_vy = base_vy
            
            # Store combined velocities
            velocities_2d[i, 0] = total_vx  # X velocity (flocking only)
            velocities_2d[i, 1] = total_vy  # Y velocity (flocking only)
            velocities_2d[i, 2] = 0.0       # Z velocity = 0 (stay at fixed height)
        
        return velocities_2d
    
    def update_heading(self):
        """Update heading but constrain Z component"""
        self.flocking_3d.update_heading()
        
        # Force Z heading component to zero to maintain 2D behavior
        hx, hy, hz = self.flocking_3d.get_heading()
        hz.fill(0.0)  # No Z-axis rotation
    
    def get_heading(self):
        """Get heading but ensure Z component is zero"""
        hx, hy, hz = self.flocking_3d.get_heading()
        hz.fill(0.0)  # Force Z heading to zero
        return hx, hy, hz

def create_drone_position_overlay(final_pos_x, final_pos_y, output_folder):
    """
    Overlays final drone positions on the gradient map.
    
    Args:
        final_pos_x: Array of final X positions of drones.
        final_pos_y: Array of final Y positions of drones.
        output_folder: Directory to save the output image.
    """
    print("\n📸 Creating overlay of final drone positions on gradient map...")
    
    try:
        # Load the gradient map image
        img = Image.open(GRADIENT_MAP_PATH).convert("RGB")
        draw = ImageDraw.Draw(img)
        
        # Calculate coordinate mapping constants (same as in read_light_intensity)
        step_size = 0.04
        grad_const_x = (len(np.arange(start=0.00, stop=WORLD_SIZE_X, step=step_size))) / WORLD_SIZE_X
        grad_const_y = (len(np.arange(start=0.00, stop=WORLD_SIZE_Y, step=step_size))) / WORLD_SIZE_Y
        
        # Draw each drone's final position
        for i in range(len(final_pos_x)):
            pybullet_x = final_pos_x[i]
            pybullet_y = final_pos_y[i]
            
            # Convert PyBullet coords to image pixel coords (matching read_light_intensity)
            # Use the same coordinate swap as the original research implementation
            map_y = int(np.ceil(pybullet_x * grad_const_y))
            map_x = int(np.ceil(pybullet_y * grad_const_x))
            
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
            
        # Save and show the image
        output_path = os.path.join(output_folder, "final_positions_overlay.png")
        img.save(output_path)
        print(f"✅ Overlay saved to: {output_path}")
        
        # Display the image with correct orientation matching the coordinate system
        plt.imshow(img)
        plt.title("Final Drone Positions on Gradient Map")
        plt.xlabel("Image Pixels (PyBullet Y -> Image X)")
        plt.ylabel("Image Pixels (PyBullet X -> Image Y)")
        plt.gca().invert_yaxis()  # Invert Y-axis to match PyBullet's view
        plt.show()
        
    except FileNotFoundError:
        print(f"[ERROR] Could not create overlay. Gradient map not found at {GRADIENT_MAP_PATH}")
    except Exception as e:
        print(f"[ERROR] An error occurred while creating the overlay: {e}")
        

def run(duration_sec=DURATION_SEC):
    print("=== 2D Flocking with Gradient Following (Built on REAL FlockingUtils) ===")
    print("This extends the proven FlockingUtils foundation")
    print("by adding light sensor simulation and gradient following behavior!")
    
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

    # Create environment (same as 3D version!)
    env = CtrlAviary(
        drone_model=DEFAULT_DRONES,
        num_drones=NUM_DRONES,
        initial_xyzs=INIT_XYZ,
        initial_rpys=INIT_RPY,
        physics=DEFAULT_PHYSICS,
        neighbourhood_radius=10,
        pyb_freq=DEFAULT_SIMULATION_FREQ_HZ,
        ctrl_freq=DEFAULT_CONTROL_FREQ_HZ,
        gui=DEFAULT_GUI,
        user_debug_gui=False
    )

    # to remove the shadows
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)

    # Set top-down camera view for 2D visualization
    p.resetDebugVisualizerCamera(
        cameraDistance=8,
        cameraYaw=0,
        cameraPitch=-89,  # Look straight down
        cameraTargetPosition=[3, 2.5, FIXED_HEIGHT]
    )

    # Create controllers (same as always!)
    ctrl = [DSLPIDControl(drone_model=DEFAULT_DRONES) for i in range(NUM_DRONES)]

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
                    num_drones=NUM_DRONES,
                    output_folder=DEFAULT_OUTPUT_FOLDER,
                    )

    print("\n💡 Controls:")
    print("- Q: Quit simulation")
    print("🔍 Watch how drones balance flocking behavior with gradient following!")
    print("📈 Drones will naturally aggregate in areas with higher light intensity")

    START = time.time()
    action = np.zeros((NUM_DRONES, 4))

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
        
        # Show progress every 3 seconds
        if i % (env.CTRL_FREQ * 3) == 0:
            # f_util.plot_swarm(pos_x, pos_y, pos_z, pos_hxs, pos_hys, pos_hzs)
            # Print light intensity readings for debugging
            light_readings = [read_light_intensity(pos_x[j], pos_y[j], add_noise=False) for j in range(NUM_DRONES)]
            avg_light_intensity = np.mean(light_readings)
            
            print(f"⏱️  Time: {i/env.CTRL_FREQ:.1f}s | 💡 Avg light intensity: {avg_light_intensity:.1f}")
            print(f"    💡 Individual readings: {[f'{reading:.1f}' for reading in light_readings]}")

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

        # Render and sync (same as always!)
        env.render()
        if DEFAULT_GUI:
            sync(i, START, env.CTRL_TIMESTEP)

    # Cleanup
    env.close()
    
    # Create overlay of final positions
    create_drone_position_overlay(pos_x, pos_y, DEFAULT_OUTPUT_FOLDER)
    
    # Final statistics
    final_light_readings = [read_light_intensity(pos_x[j], pos_y[j], add_noise=False) for j in range(NUM_DRONES)]
    avg_final_light = np.mean(final_light_readings)
    
    print(f"\n💡 Gradient following simulation completed!")
    print(f"📊 Final light intensities: {[f'{reading:.1f}' for reading in final_light_readings]}")
    print(f"📈 Average final light intensity: {avg_final_light:.1f}")
    print(f"✅ Successfully combined FlockingUtils research algorithms with gradient following!")
    print(f"🚀 Drones naturally aggregated based on local light intensity readings!")

if __name__ == "__main__":
    run()
