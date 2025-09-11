"""
2D Flocking in PyBullet - Simplified from 3d_flocking_v0.py
Key changes: 
1. All drones at fixed height (Z=1.0)
2. Only X,Y forces calculated
3. Top-down camera view for easy 2D visualization
4. Same architecture as 3D but much easier to understand!
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

# Configuration
DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_PLOT = False
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_OUTPUT_FOLDER = 'results_2d'
DURATION_SEC = 10


NUM_DRONES = 7
FIXED_HEIGHT = 1.0  # All drones stay at this height

class Simple2DFlocking:
    """Simplified 2D flocking - easier to understand than 3D version"""
    
    def __init__(self, n_agents, center_x, center_y, spacing):
        self.n_agents = n_agents
        self.center_x = center_x
        self.center_y = center_y
        self.spacing = spacing
        
        # 2D Flocking parameters (simplified from FlockingUtils)
        self.sensing_range = 2.0  # How far drones "see" each other
        self.separation_weight = 0.5
        self.alignment_weight = 0.3
        self.cohesion_weight = 0.2
        self.max_speed = 0.5
        
        # 2D state variables (no Z components!)
        self.positions_2d = np.zeros((n_agents, 2))  # Only X, Y
        self.velocities_2d = np.zeros((n_agents, 2))  # Only X, Y
        self.headings = np.zeros(n_agents)
    
    def initialize_positions(self):
        """Initialize drones in 2D grid pattern"""
        positions = []
        
        # Create simple grid layout
        grid_size = int(np.ceil(np.sqrt(self.n_agents)))
        for i in range(self.n_agents):
            row = i // grid_size
            col = i % grid_size
            x = self.center_x + (col - grid_size/2) * self.spacing
            y = self.center_y + (row - grid_size/2) * self.spacing
            positions.append([x, y])
        
        self.positions_2d = np.array(positions)
        print(f"2D Starting positions:")
        for i, pos in enumerate(self.positions_2d):
            print(f"  Drone {i}: ({pos[0]:.2f}, {pos[1]:.2f})")
        
        return self.positions_2d
    
    def update_positions(self, drone_states):
        """Update 2D positions from drone states"""
        for i in range(self.n_agents):
            self.positions_2d[i, 0] = drone_states[i][0]  # X
            self.positions_2d[i, 1] = drone_states[i][1]  # Y
    
    def calc_2d_flocking_forces(self):
        """Calculate 2D flocking forces - much simpler than 3D!"""
        forces = np.zeros((self.n_agents, 2))
        
        for i in range(self.n_agents):
            # Find neighbors within sensing range
            neighbors = []
            for j in range(self.n_agents):
                if i != j:
                    distance = np.linalg.norm(self.positions_2d[i] - self.positions_2d[j])
                    if distance < self.sensing_range:
                        neighbors.append(j)
            
            if len(neighbors) > 0:
                # 1. Separation: move away from close neighbors
                separation = np.zeros(2)
                for j in neighbors:
                    diff = self.positions_2d[i] - self.positions_2d[j]
                    distance = np.linalg.norm(diff)
                    if distance > 0.1:  # Avoid division by zero
                        separation += diff / (distance * distance)  # Inverse square
                
                # 2. Alignment: match neighbor velocities
                alignment = np.zeros(2)
                for j in neighbors:
                    alignment += self.velocities_2d[j]
                alignment /= len(neighbors)
                
                # 3. Cohesion: move toward center of neighbors
                cohesion = np.zeros(2)
                for j in neighbors:
                    cohesion += self.positions_2d[j]
                cohesion /= len(neighbors)
                cohesion = cohesion - self.positions_2d[i]  # Direction to center
                
                # Combine forces with weights
                forces[i] = (self.separation_weight * separation + 
                           self.alignment_weight * alignment + 
                           self.cohesion_weight * cohesion)
        
        return forces
    
    def calc_velocity_commands(self):
        """Convert 2D forces to velocity commands"""
        forces = self.calc_2d_flocking_forces()
        
        # Update velocities
        self.velocities_2d += forces * 0.1  # Small step size
        
        # Limit speed
        speeds = np.linalg.norm(self.velocities_2d, axis=1)
        for i in range(self.n_agents):
            if speeds[i] > self.max_speed:
                self.velocities_2d[i] *= self.max_speed / speeds[i]
        
        return self.velocities_2d

def run(duration_sec=DURATION_SEC):
    print("=== 2D Flocking in PyBullet ===")
    print("This is much easier to understand than 3D!")
    print("Watch from above - pure 2D bird's eye view")
    
    # Initialize 2D flocking system
    flocking_2d = Simple2DFlocking(NUM_DRONES, 3.0, 3.0, 1.0)
    positions_2d = flocking_2d.initialize_positions()

    # Create 3D positions (but fix Z coordinate)
    INIT_XYZ = np.zeros([NUM_DRONES, 3])
    INIT_XYZ[:, 0] = positions_2d[:, 0]  # X from 2D
    INIT_XYZ[:, 1] = positions_2d[:, 1]  # Y from 2D  
    INIT_XYZ[:, 2] = FIXED_HEIGHT        # Fixed Z for all drones
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

    # Set top-down camera view for 2D visualization
    p.resetDebugVisualizerCamera(
        cameraDistance=8,
        cameraYaw=0,
        cameraPitch=-89,  # Look straight down
        cameraTargetPosition=[3, 3, FIXED_HEIGHT]
    )

    # Create controllers (same as 3D!)
    ctrl = [DSLPIDControl(drone_model=DEFAULT_DRONES) for i in range(NUM_DRONES)]

    print("\nControls: Press Q to quit early")
    print("Watch the 2D flocking behavior from above!")

    START = time.time()
    action = np.zeros((NUM_DRONES, 4))

    # Main simulation loop (similar to 3D but simpler!)
    for i in range(0, int(duration_sec * env.CTRL_FREQ)):
        # Handle user input
        keys = p.getKeyboardEvents()
        if ord('q') in keys:
            break

        # Step simulation (same as always!)
        obs, reward, done, info, _ = env.step(action)
        
        # Update 2D positions from observations
        flocking_2d.update_positions([obs[j] for j in range(NUM_DRONES)])
        
        # Calculate 2D flocking velocities
        velocities_2d = flocking_2d.calc_velocity_commands()

        # Convert 2D velocities to 3D commands
        for j in range(NUM_DRONES):
            # 2D velocity + fixed height
            vel_cmd_3d = np.array([
                velocities_2d[j, 0],  # X velocity from 2D flocking
                velocities_2d[j, 1],  # Y velocity from 2D flocking  
                0.0                   # No Z velocity - stay at fixed height
            ])
            
            # Current position but keep Z fixed
            current_pos = obs[j][:3]
            target_pos = np.array([
                current_pos[0],       # Current X (will be changed by velocity)
                current_pos[1],       # Current Y (will be changed by velocity)
                FIXED_HEIGHT          # Fixed Z
            ])
            
            # Generate control action (same as always!)
            action[j], _, _ = ctrl[j].computeControlFromState(
                control_timestep=env.CTRL_TIMESTEP,
                state=obs[j],
                target_pos=target_pos,
                target_vel=vel_cmd_3d,
                target_rpy=np.array([0, 0, 0])
            )

        # Render and sync (same as always!)
        env.render()
        if DEFAULT_GUI:
            sync(i, START, env.CTRL_TIMESTEP)

    # Cleanup
    env.close()
    print("\n✅ 2D Flocking completed!")
    print("Notice how much easier it was to understand the 2D movement!")

if __name__ == "__main__":
    run()
