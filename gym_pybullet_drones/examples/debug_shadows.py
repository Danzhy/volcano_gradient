"""
Quick debug version to identify the shadow drones
"""
import time
import numpy as np
import sys
import pybullet as p

sys.path.append('ants_2024/')
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

# Simple test with minimal setup
NUM_DRONES = 3
FIXED_HEIGHT = 1.0

# Create environment
env = CtrlAviary(
    drone_model=DroneModel("cf2x"),
    num_drones=NUM_DRONES,
    initial_xyzs=np.array([[0, 0, FIXED_HEIGHT], [1, 0, FIXED_HEIGHT], [2, 0, FIXED_HEIGHT]]),
    physics=Physics("pyb"),
    gui=True,
    user_debug_gui=False
)

# Clear any visual artifacts
p.removeAllUserDebugItems()
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)

print(f"Number of actual drones: {len(env.getDroneIds())}")
print(f"Drone IDs: {env.getDroneIds()}")

# Check what objects exist in the simulation
print("\nAll objects in simulation:")
for i in range(p.getNumBodies()):
    body_info = p.getBodyInfo(i)
    print(f"Body {i}: {body_info}")

# Simple hover test
ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for i in range(NUM_DRONES)]
action = np.zeros((NUM_DRONES, 4))

for i in range(200):  # Run for a few seconds
    obs, reward, done, info, _ = env.step(action)
    
    # Simple hover control
    for j in range(NUM_DRONES):
        target_pos = np.array([j, 0, FIXED_HEIGHT])
        action[j], _, _ = ctrl[j].computeControlFromState(
            control_timestep=env.CTRL_TIMESTEP,
            state=obs[j],
            target_pos=target_pos,
            target_vel=np.array([0, 0, 0])
        )
    
    env.render()
    time.sleep(0.02)

env.close()
print("Debug complete - did you see shadow drones?")
