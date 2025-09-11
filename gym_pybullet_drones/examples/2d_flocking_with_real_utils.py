"""
2D Flocking using REAL FlockingUtils - just constrain Z axis!
This is much better - reuses the proven research code and just makes Z constant.
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
from ants_2024.flocking_utils import FlockingUtils

# Configuration
DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DURATION_SEC = 15

NUM_DRONES = 5
FIXED_HEIGHT = 1.0  # All drones stay at this Z height

# Starting position for swarm
init_center_x = 3.0
init_center_y = 3.0
init_center_z = FIXED_HEIGHT  # Use our fixed height
spacing = 1.0

class FlockingUtils2D:
    """Wrapper around FlockingUtils that constrains Z-axis to be constant"""
    
    def __init__(self, n_agents, center_x, center_y, center_z, spacing):
        # Create the real FlockingUtils
        self.flocking_3d = FlockingUtils(n_agents, center_x, center_y, center_z, spacing)
        self.fixed_z = center_z
        
        print(f"🎯 Created 2D wrapper around FlockingUtils")
        print(f"   - Using proven research algorithms")
        print(f"   - Constraining all drones to Z = {self.fixed_z}")
        print(f"   - Real separation/alignment/cohesion forces")
    
    def initialize_positions(self):
        """Initialize positions but ensure Z is fixed"""
        pos_xs, pos_ys, pos_zs, pos_h_xc, pos_h_yc, pos_h_zc = self.flocking_3d.initialize_positions()
        
        # Force all Z positions to be constant
        pos_zs.fill(self.fixed_z)
        pos_h_zc.fill(0.0)  # No Z-axis heading component
        
        print(f"📍 Initialized {len(pos_xs)} drones in 2D:")
        for i in range(len(pos_xs)):
            print(f"   Drone {i}: ({pos_xs[i]:.2f}, {pos_ys[i]:.2f}, {pos_zs[i]:.2f})")
        
        return pos_xs, pos_ys, pos_zs, pos_h_xc, pos_h_yc, pos_h_zc
    
    def compute_2d_flocking_forces(self, pos_xs, pos_ys, pos_zs):
        """Compute flocking forces but constrain Z to be constant"""
        
        # Force all Z positions to be constant before calculation
        pos_zs_constrained = np.full_like(pos_zs, self.fixed_z)
        
        # Use the real FlockingUtils calculations with constrained Z
        self.flocking_3d.calc_dij(pos_xs, pos_ys, pos_zs_constrained)
        self.flocking_3d.calc_ang_ij(pos_xs, pos_ys, pos_zs_constrained)
        self.flocking_3d.calc_grad_vals(pos_xs, pos_ys, pos_zs_constrained)
        self.flocking_3d.calc_p_forces()
        self.flocking_3d.calc_alignment_forces()
        self.flocking_3d.calc_boun_rep(pos_xs, pos_ys, pos_zs_constrained)
        
        # Get velocities but zero out Z component
        u = self.flocking_3d.calc_u_w()
        
        # Convert to 2D velocity commands (X, Y only, Z=0)
        velocities_2d = np.zeros((len(pos_xs), 3))
        
        for i in range(len(pos_xs)):
            # Get heading from FlockingUtils
            hx, hy, hz = self.flocking_3d.get_heading()
            
            # Convert to 2D velocities (ignore Z component)
            velocities_2d[i, 0] = u[i] * np.cos(hx[i])  # X velocity
            velocities_2d[i, 1] = u[i] * np.cos(hy[i])  # Y velocity  
            velocities_2d[i, 2] = 0.0                   # Z velocity = 0 (stay at fixed height)
        
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

def run(duration_sec=DURATION_SEC):
    print("=== 2D Flocking using REAL FlockingUtils (Z-constrained) ===")
    print("This uses the same proven research algorithms from 3d_flocking_v0.py")
    print("but constrains movement to 2D plane for easier learning!")
    
    # Create 2D wrapper around FlockingUtils
    f_util = FlockingUtils2D(NUM_DRONES, init_center_x, init_center_y, init_center_z, spacing)
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
        cameraTargetPosition=[init_center_x, init_center_y, FIXED_HEIGHT]
    )

    # Create controllers (same as always!)
    ctrl = [DSLPIDControl(drone_model=DEFAULT_DRONES) for i in range(NUM_DRONES)]

    print("\nControls: Press Q to quit early")
    print("🔍 Watch the 2D flocking behavior from above!")
    print("📊 This uses the SAME algorithms as the research code!")

    START = time.time()
    action = np.zeros((NUM_DRONES, 4))

    # Main simulation loop (similar to 3d_flocking_v0.py but with Z constraint)
    for i in range(0, int(duration_sec * env.CTRL_FREQ)):
        # Clear any visual artifacts that might cause shadow drones
        p.removeAllUserDebugItems()
        
        # Handle user input
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

        # Compute 2D flocking forces using REAL FlockingUtils
        velocities_2d = f_util.compute_2d_flocking_forces(pos_x, pos_y, pos_z)
        f_util.update_heading()

        # Apply control for each drone
        for j in range(NUM_DRONES):
            # Target position: current XY + fixed Z
            target_pos = np.array([
                pos_x[j],      # Current X (will be changed by velocity)
                pos_y[j],      # Current Y (will be changed by velocity)
                FIXED_HEIGHT   # Fixed Z height
            ])
            
            # Velocity command from flocking (X, Y, 0)
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
    print("\n✅ 2D Flocking completed using REAL FlockingUtils!")
    print("🎯 Same proven algorithms, just constrained to 2D!")
    print("🚀 Ready to add gradient following to this solid foundation!")

if __name__ == "__main__":
    run()
