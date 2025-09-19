"""
2D Flocking Verification - Check if the behavior is correct
Adds metrics to verify proper flocking behavior
"""
import time
import numpy as np
import sys
import pybullet as p
import matplotlib.pyplot as plt

sys.path.append('ants_2024/')
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.utils import sync, str2bool
from ants_2024.flocking_utils import FlockingUtils

# Configuration
NUM_DRONES = 5
FIXED_HEIGHT = 1.0
DURATION_SEC = 20

class FlockingVerifier:
    """Verify that flocking behavior is working correctly"""
    
    def __init__(self, num_drones):
        self.num_drones = num_drones
        
        # Metrics storage
        self.distances_history = []
        self.velocities_history = []
        self.cohesion_history = []
        self.separation_history = []
        
    def calculate_metrics(self, positions, velocities):
        """Calculate flocking metrics"""
        positions = np.array(positions)
        velocities = np.array(velocities)
        
        # 1. Average inter-drone distance
        distances = []
        for i in range(self.num_drones):
            for j in range(i+1, self.num_drones):
                dist = np.linalg.norm(positions[i] - positions[j])
                distances.append(dist)
        avg_distance = np.mean(distances)
        
        # 2. Velocity alignment (how similar are the velocities?)
        velocity_magnitudes = np.linalg.norm(velocities, axis=1)
        avg_velocity = np.mean(velocity_magnitudes)
        
        # 3. Cohesion (how tight is the group?)
        center = np.mean(positions, axis=0)
        distances_to_center = [np.linalg.norm(pos - center) for pos in positions]
        cohesion = np.mean(distances_to_center)
        
        # 4. Separation (minimum distance - should not be too small)
        min_distance = np.min(distances) if distances else 0
        
        return {
            'avg_distance': avg_distance,
            'avg_velocity': avg_velocity,
            'cohesion': cohesion,
            'min_distance': min_distance,
            'velocity_std': np.std(velocity_magnitudes)
        }
    
    def log_metrics(self, metrics):
        """Store metrics for analysis"""
        self.distances_history.append(metrics['avg_distance'])
        self.velocities_history.append(metrics['avg_velocity'])
        self.cohesion_history.append(metrics['cohesion'])
        self.separation_history.append(metrics['min_distance'])
    
    def print_realtime_metrics(self, metrics, iteration):
        """Print real-time metrics"""
        if iteration % 50 == 0:  # Print every ~1 second
            print(f"\n--- Flocking Metrics (t={iteration/48:.1f}s) ---")
            print(f"🔄 Avg Distance: {metrics['avg_distance']:.2f}m")
            print(f"🚀 Avg Velocity: {metrics['avg_velocity']:.2f}m/s")
            print(f"🎯 Cohesion: {metrics['cohesion']:.2f}m")
            print(f"⚠️  Min Distance: {metrics['min_distance']:.2f}m")
            print(f"📊 Vel Std: {metrics['velocity_std']:.3f}")
            
            # Health checks
            if metrics['min_distance'] < 0.3:
                print("⚠️  WARNING: Drones too close!")
            if metrics['avg_velocity'] < 0.01:
                print("⚠️  WARNING: Drones barely moving!")
            if metrics['cohesion'] > 5.0:
                print("⚠️  WARNING: Swarm too spread out!")
    
    def plot_final_analysis(self):
        """Plot analysis of the flocking behavior"""
        if len(self.distances_history) < 10:
            print("Not enough data for analysis")
            return
            
        time_steps = np.arange(len(self.distances_history)) / 48  # Convert to seconds
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('2D Flocking Behavior Analysis')
        
        # 1. Average distances over time
        ax1.plot(time_steps, self.distances_history)
        ax1.set_title('Average Inter-drone Distance')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Distance (m)')
        ax1.grid(True)
        
        # 2. Velocities over time
        ax2.plot(time_steps, self.velocities_history)
        ax2.set_title('Average Velocity')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Velocity (m/s)')
        ax2.grid(True)
        
        # 3. Cohesion over time
        ax3.plot(time_steps, self.cohesion_history)
        ax3.set_title('Group Cohesion')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Avg Distance to Center (m)')
        ax3.grid(True)
        
        # 4. Separation over time
        ax4.plot(time_steps, self.separation_history)
        ax4.set_title('Minimum Separation')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Min Distance (m)')
        ax4.grid(True)
        ax4.axhline(y=0.3, color='r', linestyle='--', label='Safety Threshold')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig('flocking_analysis.png')
        print("\n📊 Analysis saved as 'flocking_analysis.png'")
        plt.show()
    
    def final_verdict(self):
        """Give final assessment of flocking quality"""
        if len(self.distances_history) < 10:
            return
            
        avg_dist = np.mean(self.distances_history[-100:])  # Last 100 samples
        avg_vel = np.mean(self.velocities_history[-100:])
        avg_cohesion = np.mean(self.cohesion_history[-100:])
        min_sep = np.min(self.separation_history)
        
        print("\n" + "="*50)
        print("🎯 FINAL FLOCKING ASSESSMENT")
        print("="*50)
        
        score = 0
        
        # Check distance (should be reasonable, not too tight or loose)
        if 1.0 <= avg_dist <= 3.0:
            print("✅ Distance: GOOD (balanced separation)")
            score += 25
        else:
            print(f"❌ Distance: POOR ({avg_dist:.2f}m - should be 1-3m)")
        
        # Check velocity (should be moving)
        if avg_vel > 0.05:
            print("✅ Movement: GOOD (active flocking)")
            score += 25
        else:
            print(f"❌ Movement: POOR ({avg_vel:.3f}m/s - too slow)")
        
        # Check cohesion (should stay reasonably together)
        if avg_cohesion < 2.5:
            print("✅ Cohesion: GOOD (tight group)")
            score += 25
        else:
            print(f"❌ Cohesion: POOR ({avg_cohesion:.2f}m - too spread)")
        
        # Check separation (should never be too close)
        if min_sep > 0.2:
            print("✅ Safety: GOOD (no collisions)")
            score += 25
        else:
            print(f"❌ Safety: POOR ({min_sep:.2f}m - too close)")
        
        print(f"\n🏆 OVERALL SCORE: {score}/100")
        
        if score >= 75:
            print("🎉 EXCELLENT! Your 2D flocking is working correctly!")
        elif score >= 50:
            print("👍 GOOD! Minor tweaks needed but mostly working.")
        else:
            print("⚠️  NEEDS WORK! Check your flocking parameters.")

# Modified FlockingUtils2D with same logic as before
class FlockingUtils2D:
    def __init__(self, n_agents, center_x, center_y, center_z, spacing):
        self.flocking_3d = FlockingUtils(n_agents, center_x, center_y, center_z, spacing)
        self.fixed_z = center_z
    
    def initialize_positions(self):
        pos_xs, pos_ys, pos_zs, pos_h_xc, pos_h_yc, pos_h_zc = self.flocking_3d.initialize_positions()
        pos_zs.fill(self.fixed_z)
        pos_h_zc.fill(0.0)
        return pos_xs, pos_ys, pos_zs, pos_h_xc, pos_h_yc, pos_h_zc
    
    def compute_2d_flocking_forces(self, pos_xs, pos_ys, pos_zs):
        pos_zs_constrained = np.full_like(pos_zs, self.fixed_z)
        
        self.flocking_3d.calc_dij(pos_xs, pos_ys, pos_zs_constrained)
        self.flocking_3d.calc_ang_ij(pos_xs, pos_ys, pos_zs_constrained)
        self.flocking_3d.calc_grad_vals(pos_xs, pos_ys, pos_zs_constrained)
        self.flocking_3d.calc_p_forces()
        self.flocking_3d.calc_alignment_forces()
        self.flocking_3d.calc_boun_rep(pos_xs, pos_ys, pos_zs_constrained)
        
        u = self.flocking_3d.calc_u_w()
        velocities_2d = np.zeros((len(pos_xs), 3))
        
        for i in range(len(pos_xs)):
            hx, hy, hz = self.flocking_3d.get_heading()
            velocities_2d[i, 0] = u[i] * np.cos(hx[i])
            velocities_2d[i, 1] = u[i] * np.cos(hy[i])
            velocities_2d[i, 2] = 0.0
        
        return velocities_2d
    
    def update_heading(self):
        self.flocking_3d.update_heading()

def run():
    print("🔍 2D Flocking Verification Test")
    print("This will analyze the quality of your flocking behavior")
    
    # Create verifier
    verifier = FlockingVerifier(NUM_DRONES)
    
    # Initialize flocking
    f_util = FlockingUtils2D(NUM_DRONES, 3.0, 3.0, FIXED_HEIGHT, 1.0)
    positions_2d = f_util.initialize_positions()

    INIT_XYZ = np.zeros([NUM_DRONES, 3])
    INIT_XYZ[:, 0] = positions_2d[0]
    INIT_XYZ[:, 1] = positions_2d[1]
    INIT_XYZ[:, 2] = FIXED_HEIGHT
    INIT_RPY = np.array([[.0, .0, .0] for _ in range(NUM_DRONES)])

    # Create environment
    env = CtrlAviary(
        drone_model=DroneModel("cf2x"),
        num_drones=NUM_DRONES,
        initial_xyzs=INIT_XYZ,
        initial_rpys=INIT_RPY,
        physics=Physics("pyb"),
        gui=True,
        user_debug_gui=False
    )

    # Disable shadows
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
    
    # Set top-down view
    p.resetDebugVisualizerCamera(
        cameraDistance=8,
        cameraYaw=0,
        cameraPitch=-89,
        cameraTargetPosition=[3, 3, FIXED_HEIGHT]
    )

    ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for i in range(NUM_DRONES)]

    print("\n🎯 Watch the metrics! Press Q to quit early.")
    print("Look for: balanced distances, smooth movement, group cohesion")

    START = time.time()
    action = np.zeros((NUM_DRONES, 4))

    # Main loop with verification
    for i in range(0, int(DURATION_SEC * env.CTRL_FREQ)):
        p.removeAllUserDebugItems()
        
        keys = p.getKeyboardEvents()
        if ord('q') in keys:
            break

        obs, reward, done, info, _ = env.step(action)
        
        # Get positions and velocities
        positions = []
        velocities = []
        pos_x = np.zeros(NUM_DRONES)
        pos_y = np.zeros(NUM_DRONES)
        pos_z = np.zeros(NUM_DRONES)

        for j in range(NUM_DRONES):
            states = env._getDroneStateVector(j)
            pos_x[j] = states[0]
            pos_y[j] = states[1]
            pos_z[j] = FIXED_HEIGHT
            
            positions.append([states[0], states[1]])
            velocities.append([states[10], states[11]])  # vx, vy

        # Calculate flocking forces
        velocities_2d = f_util.compute_2d_flocking_forces(pos_x, pos_y, pos_z)
        f_util.update_heading()

        # Apply control
        for j in range(NUM_DRONES):
            target_pos = np.array([pos_x[j], pos_y[j], FIXED_HEIGHT])
            vel_cmd = velocities_2d[j]
            
            action[j], _, _ = ctrl[j].computeControlFromState(
                control_timestep=env.CTRL_TIMESTEP,
                state=obs[j],
                target_pos=target_pos,
                target_vel=vel_cmd,
                target_rpy=np.array([0, 0, 0])
            )

        # Verify behavior
        metrics = verifier.calculate_metrics(positions, velocities)
        verifier.log_metrics(metrics)
        verifier.print_realtime_metrics(metrics, i)

        env.render()
        if env.GUI:
            sync(i, START, env.CTRL_TIMESTEP)

    env.close()
    
    # Final analysis
    verifier.plot_final_analysis()
    verifier.final_verdict()

if __name__ == "__main__":
    run()
