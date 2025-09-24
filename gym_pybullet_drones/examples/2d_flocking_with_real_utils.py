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

# Goal settings - NEW!
GOAL_POSITION = [4.0, 3.0]  # Target location (X, Y)
GOAL_ATTRACTION_STRENGTH = 0.4  # How strongly drones are attracted to goal

class FlockingUtils2DWithGoal:
    # TODO: change this wrapper term because prof doesnt like it. 
    """Wrapper around FlockingUtils with goal-seeking behavior added"""
    
    def __init__(self, n_agents, center_x, center_y, center_z, spacing, goal_pos, goal_strength):
        # Create the real FlockingUtils (same as before)
        self.flocking_3d = FlockingUtils(n_agents, center_x, center_y, center_z, spacing)
        self.fixed_z = center_z
        
        # NEW: Goal-seeking parameters
        self.goal_position = np.array(goal_pos)
        self.goal_strength = goal_strength
        
        print(f"🎯 Created 2D FlockingUtils with Goal-Seeking")
        print(f"   - Using proven research algorithms (same as before)")
        print(f"   - Constraining all drones to Z = {self.fixed_z}")
        print(f"   - NEW: Goal target at ({goal_pos[0]}, {goal_pos[1]})")
        print(f"   - NEW: Goal attraction strength = {goal_strength}")
    
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
    
    def compute_2d_flocking_forces_with_goal(self, pos_xs, pos_ys, pos_zs):
        """Compute flocking forces + goal attraction (EXTENDED VERSION)"""
        
        # Force all Z positions to be constant before calculation
        pos_zs_constrained = np.full_like(pos_zs, self.fixed_z)
        
        # Use the real FlockingUtils calculations with constrained Z
        self.flocking_3d.calc_dij(pos_xs, pos_ys, pos_zs_constrained)
        self.flocking_3d.calc_ang_ij(pos_xs, pos_ys, pos_zs_constrained)
        self.flocking_3d.calc_grad_vals(pos_xs, pos_ys, pos_zs_constrained)
        self.flocking_3d.calc_p_forces() # separation
        self.flocking_3d.calc_alignment_forces() # alignment
        self.flocking_3d.calc_boun_rep(pos_xs, pos_ys, pos_zs_constrained) # boundary 
        
        # Get velocities but zero out Z component
        u = self.flocking_3d.calc_u_w()
        
        # Convert to 2D velocity commands with GOAL FORCES ADDED
        velocities_2d = np.zeros((len(pos_xs), 3))
        
        for i in range(len(pos_xs)):
            # Get heading from FlockingUtils
            hx, hy, hz = self.flocking_3d.get_heading()
            
            # Base flocking velocities (same as before)
            base_vx = u[i] * np.cos(hx[i])  # X velocity from flocking
            base_vy = u[i] * np.cos(hy[i])  # Y velocity from flocking
            
            # NEW: Goal attraction force calculation
            drone_pos = np.array([pos_xs[i], pos_ys[i]])
            goal_direction = self.goal_position - drone_pos
            goal_distance = np.linalg.norm(goal_direction)
            
            if goal_distance > 0.1:  # Avoid division by zero
                goal_direction = goal_direction / goal_distance  # Normalize
                # Stronger attraction when farther away (up to max distance)
                goal_force = self.goal_strength * min(goal_distance, 2.0)
                goal_vx = goal_direction[0] * goal_force
                goal_vy = goal_direction[1] * goal_force
            else:
                goal_vx = goal_vy = 0.0
            
            # NEW: Combine flocking + goal forces
            total_vx = base_vx + goal_vx
            total_vy = base_vy + goal_vy
            
            # Store combined velocities
            velocities_2d[i, 0] = total_vx  # X velocity (flocking + goal)
            velocities_2d[i, 1] = total_vy  # Y velocity (flocking + goal)
            velocities_2d[i, 2] = 0.0       # Z velocity = 0 (stay at fixed height)
        
        return velocities_2d
    
    def update_goal_position(self, new_goal_pos):
        """NEW: Update goal position for interactive control"""
        self.goal_position = np.array(new_goal_pos)
        print(f"🎯 Goal moved to: ({new_goal_pos[0]:.1f}, {new_goal_pos[1]:.1f})")
    
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

class GoalVisualizer:
    """NEW: Manages the visual goal target"""
    
    def __init__(self, goal_position):
        self.goal_position = goal_position
        
        # Create red circle target
        visual_shape = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.3,
            length=0.05,
            rgbaColor=[1, 0, 0, 0.8]  # Red, semi-transparent
        )
        
        self.goal_body_id = p.createMultiBody(
            baseMass=0,  # Static
            baseVisualShapeIndex=visual_shape,
            basePosition=[goal_position[0], goal_position[1], 0.025]
        )
        
        # Add text label
        self.text_id = p.addUserDebugText(
            "GOAL",
            [goal_position[0], goal_position[1], 0.5],
            textColorRGB=[1, 0, 0],
            textSize=2.0
        )
        
        print(f"🔴 Created visual goal target at ({goal_position[0]}, {goal_position[1]})")
    
    def update_goal_position(self, new_position):
        """Move the goal to a new position"""
        self.goal_position = new_position
        
        # Update visual position
        p.resetBasePositionAndOrientation(
            self.goal_body_id,
            [new_position[0], new_position[1], 0.025],
            [0, 0, 0, 1]
        )
        
        # Update text (remove old, create new)
        p.removeUserDebugItem(self.text_id)
        self.text_id = p.addUserDebugText(
            "GOAL",
            [new_position[0], new_position[1], 0.5],
            textColorRGB=[1, 0, 0],
            textSize=2.0
        )

def run(duration_sec=DURATION_SEC):
    print("=== 2D Flocking with Goal Target (Built on REAL FlockingUtils) ===")
    print("This extends the proven 2d_flocking_with_real_utils.py foundation")
    print("by adding goal-seeking behavior while keeping all research algorithms!")
    
    # Create 2D wrapper with goal-seeking capability
    f_util = FlockingUtils2DWithGoal(
        n_agents=NUM_DRONES,
        center_x=init_center_x, center_y=init_center_y, center_z=init_center_z, 
        spacing=spacing,
        goal_pos=GOAL_POSITION,
        goal_strength=GOAL_ATTRACTION_STRENGTH
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

    
    # NEW: Create goal visualizer
    goal_viz = GoalVisualizer(GOAL_POSITION)

    print("\n🎯 Controls:")
    print("- Arrow keys (WASD): Move the goal target around")
    print("- Q: Quit simulation")
    print("🔍 Watch how drones balance flocking behavior with goal seeking!")
    print("🔴 Red circle = GOAL TARGET that drones will seek")

    START = time.time()
    action = np.zeros((NUM_DRONES, 4))

    # Track goal for progress monitoring
    current_goal = GOAL_POSITION.copy()

    # Main simulation loop (extended from original)
    for i in range(0, int(duration_sec * env.CTRL_FREQ)):
        # Clear any visual artifacts (same as before)
        p.removeAllUserDebugItems()
        
        # NEW: Handle user input for goal movement
        keys = p.getKeyboardEvents()
        goal_moved = False
        
        if p.B3G_LEFT_ARROW in keys or ord('a') in keys:
            current_goal[0] -= 0.1
            goal_moved = True
        if p.B3G_RIGHT_ARROW in keys or ord('d') in keys:
            current_goal[0] += 0.1
            goal_moved = True
        if p.B3G_UP_ARROW in keys or ord('w') in keys:
            current_goal[1] += 0.1
            goal_moved = True
        if p.B3G_DOWN_ARROW in keys or ord('s') in keys:
            current_goal[1] -= 0.1
            goal_moved = True
        
        if goal_moved:
            goal_viz.update_goal_position(current_goal)
            f_util.update_goal_position(current_goal)
        
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

        # NEW: Compute 2D flocking forces WITH GOAL ATTRACTION
        velocities_2d = f_util.compute_2d_flocking_forces_with_goal(pos_x, pos_y, pos_z)
        pos_hxs, pos_hys, pos_hzs = f_util.get_heading()
        f_util.update_heading()
        
        # Show progress every 3 seconds
        if i % (env.CTRL_FREQ * 3) == 0:
            # f_util.plot_swarm(pos_x, pos_y, pos_z, pos_hxs, pos_hys, pos_hzs)
            avg_dist_to_goal = np.mean([
                np.linalg.norm([pos_x[j] - current_goal[0], pos_y[j] - current_goal[1]])
                for j in range(NUM_DRONES)
            ])
            print(f"⏱️  Time: {i/env.CTRL_FREQ:.1f}s | 📏 Avg distance to goal: {avg_dist_to_goal:.2f}m")

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
    
    # NEW: Final statistics
    final_distances = [
        np.linalg.norm([pos_x[j] - current_goal[0], pos_y[j] - current_goal[1]])
        for j in range(NUM_DRONES)
    ]
    print(f"\n🎯 Goal-seeking flocking simulation completed!")
    print(f"📊 Final distances to goal: {[f'{d:.2f}m' for d in final_distances]}")
    print(f"📈 Average final distance: {np.mean(final_distances):.2f}m")
    print(f"✅ Successfully combined FlockingUtils research algorithms with goal-seeking!")
    print(f"🚀 Ready for next step: Replace goal with light source!")

if __name__ == "__main__":
    run()
