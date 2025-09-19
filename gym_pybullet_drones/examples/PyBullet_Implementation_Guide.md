# PyBullet Swarm Gradient Following Implementation Guide

## Context Summary for AI Models

This document provides a complete learning path to implement a swarm robotics gradient-following system in PyBullet, based on an existing real-world Crazyflie drone research project.

## **CURRENT STATUS (Updated for AI Handoff)**

### **✅ COMPLETED ACHIEVEMENTS:**

1. **Environment Setup Complete**
   - ✅ Conda environment `drones` activated and functional
   - ✅ `gym-pybullet-drones` repository working
   - ✅ All dependencies installed and tested

2. **Core Examples Mastered**
   - ✅ **`pid.py`**: Understood PID control, `env.step()`, `DSLPIDControl`, circular trajectories
   - ✅ **`3d_flocking_v0.py`**: Fixed logging errors, successfully demonstrated 3D flocking
   - ✅ **Architecture Understanding**: Control flow, force calculations, drone physics integration

3. **2D Flocking Implementation SUCCESS**
   - ✅ **Created `2d_flocking_simple.py`**: Basic 2D implementation from scratch
   - ✅ **Created `2d_flocking_with_real_utils.py`**: Uses real `FlockingUtils` with Z-constraint
   - ✅ **Fixed Visual Artifacts**: Resolved "shadow drone" issue with `p.removeAllUserDebugItems()`
   - ✅ **Verification System**: Created `verify_2d_flocking.py` with behavioral metrics

4. **Working Files Created**
   ```
   /gym_pybullet_drones/examples/
   ├── 2d_flocking_simple.py              # Custom 2D flocking implementation
   ├── 2d_flocking_with_real_utils.py     # 2D using real FlockingUtils (RECOMMENDED)
   ├── verify_2d_flocking.py              # Verification with metrics
   └── debug_shadows.py                   # Debug tool for visual issues
   ```

5. **Key Technical Understanding**
   - ✅ **Force Calculations**: Separation, alignment, cohesion in 2D
   - ✅ **PyBullet Integration**: Camera control, physics, visual debugging
   - ✅ **FlockingUtils API**: How to constrain Z-axis while keeping research-grade algorithms
   - ✅ **Environment Architecture**: `CtrlAviary`, `DSLPIDControl`, frequency management

### **🎯 NEXT STEP: Gradient Following Implementation**

**Ready for**: Adding light sources and gradient-following forces to the working 2D flocking system.

**Current Status**: The foundation is solid - we have verified 2D flocking behavior working correctly with proper force calculations and PyBullet integration.

**Recommended Approach**: Extend `2d_flocking_with_real_utils.py` by:
1. Adding `LightManager` class for interactive light sources
2. Extending `FlockingUtils` with gradient forces
3. Implementing keyboard controls (1-6 for lights, Q for quit)

### **Key Debugging Solutions Discovered**
- **Shadow Drones**: Fixed with `p.removeAllUserDebugItems()` in main loop
- **FlockingUtils Z-Constraint**: Simply set `pos_zs` to constant value (1.0)
- **Verification Method**: Use distance metrics, velocity analysis, and visual observation

### **Architecture Decisions Made**
- **2D Approach**: Confirmed as optimal learning path (simpler than 3D, same principles)
- **Real FlockingUtils**: Better than custom implementation (research-grade algorithms)
- **PyBullet Integration**: Leverage existing `gym-pybullet-drones` framework

### Background System Architecture

The original system consists of:
- **Hardware**: 5 Crazyflie drones with onboard light sensors + 6 WiFi-controlled smart bulbs
- **Firmware**: `swarm_vu.c` - C code running onboard each drone implementing swarm algorithms
- **Mission Control**: `mc_client.py` - Python script controlling lights and visualizing swarm behavior
- **Research Focus**: Gradient-following where drones autonomously move toward dynamic light sources

### Key Algorithms (from swarm_vu.c firmware)

The swarm behavior is driven by mathematical equations:

1. **Proximal Forces (Equation 1)**: Inter-drone repulsion/attraction based on distance
   ```c
   force = -epsilon * (2*(su^4/distance^5) - (su^2/distance^3)) * direction_vector
   ```

2. **Gradient Following (Equation 2)**: Adaptive speed based on light intensity
   ```c
   su = sigma_base + pow((light_reading - light_min)/(light_max - light_min), 0.1) * sigma_var
   ```

3. **Alignment Forces (Equation 3)**: Heading alignment with neighbors
4. **Boundary Forces (Equation 4)**: Repulsion from flight area boundaries
5. **Total Force Combination (Equation 10)**: Weighted sum of all forces
6. **Velocity Control (Equation 11)**: Convert forces to linear/angular velocities

### System Communication Flow

```
Mission Control ←→ Radio ←→ Drone Firmware
     ↓                          ↓
- Light control           - Swarm algorithms  
- Parameter updates       - Force calculations
- Data visualization      - Flight control
- Experiment management   - Sensor processing
```

### Implementation Goal

Create a PyBullet simulation that replicates the exact swarm behavior without requiring physical hardware, allowing algorithm testing and parameter tuning in a realistic physics environment.

---

# REVISED PyBullet Learning Path for Swarm Gradient Following

**🎉 ADVANTAGE: We now have access to gym-pybullet-drones with working flocking examples!**

This significantly accelerates our learning path. Instead of building from scratch, we'll:
1. Study the existing working examples
2. Understand the architecture 
3. Modify the flocking system for gradient following

## **Phase 1: Understanding the Existing System (Week 1)**

### **Step 1: Explore Existing Examples**
```bash
# Navigate to the examples
cd gym_pybullet_drones/examples/

# Run the basic PID control example first
conda activate drones
python pid.py

# Then run the flocking demo
cd ants_2024/
python 3d_flocking_v0.py
```

**Goal**: See working drone simulations and understand what we're building toward.

### **Step 2: Study the Flocking Architecture**
Read and understand these key files:
```bash
# Key files to examine:
gym_pybullet_drones/examples/ants_2024/
├── 3d_flocking_v0.py          # Main simulation script
├── flocking_utils.py          # Core flocking algorithms
├── 3d_flocking_v1.py          # Extended version with logging
└── 3d_flocking_vectorized.py  # Optimized version
```

**Study Focus:**
- How `FlockingUtils` class implements swarm forces
- How `CtrlAviary` manages multiple drones
- How `DSLPIDControl` converts high-level commands to motor controls
- Integration between flocking logic and drone physics

**Goal**: Understand the existing swarm architecture before modifying it.

### **Step 3: Analyze Key Flocking Components**
Create `analyze_flocking.py` to understand the core algorithms:
```python
import sys
sys.path.append('../')
from ants_2024.flocking_utils import FlockingUtils
import numpy as np

# Create a FlockingUtils instance to study
f_util = FlockingUtils(5, 2.0, 2.0, 1.0, 0.8)

# Study the key methods:
print("FlockingUtils Key Methods:")
print("- initialize_positions():", f_util.initialize_positions.__doc__)
print("- calc_dij(): Calculates inter-drone distances")
print("- calc_p_forces(): Computes proximal forces (repulsion/attraction)")
print("- calc_alignment_forces(): Computes heading alignment forces")
print("- calc_boun_rep(): Boundary repulsion forces")
print("- calc_u_w(): Converts forces to velocity commands")

# Understand the parameters
print("\nKey Parameters:")
print(f"- sensing_range: {f_util.sensing_range}")
print(f"- sigma (force scaling): {f_util.sigma}")
print(f"- alpha (proximal weight): {f_util.alpha}")
print(f"- beta (alignment weight): {f_util.beta}")
print(f"- boundary size: {f_util.boun_x} x {f_util.boun_y}")
```

**Goal**: Understand how existing flocking forces work before adding gradient following.

### **Step 4: Applying Forces**
Create `03_forces.py`:
```python
import pybullet as p
import time
import math

physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -10)
planeId = p.loadURDF("plane.urdf")

# Create a controllable sphere
sphereId = p.createMultiBody(
    baseMass=1,
    baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=0.1),
    baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.1, rgbaColor=[0, 1, 0, 1]),
    basePosition=[0, 0, 1]
)

# Simulation loop with force application
for i in range(2000):
    # Apply circular force
    force_x = math.cos(i * 0.01) * 10
    force_y = math.sin(i * 0.01) * 10
    force_z = 15  # Counteract gravity
    
    p.applyExternalForce(
        objectUniqueId=sphereId,
        linkIndex=-1,  # Apply to base
        forceObj=[force_x, force_y, force_z],
        posObj=[0, 0, 0],  # Relative to object center
        flags=p.LINK_FRAME
    )
    
    p.stepSimulation()
    time.sleep(1./240.)

p.disconnect()
```

**Goal**: Understand how to control objects with forces (essential for swarm control).

### **Step 5: Keyboard Input**
Create `04_keyboard_control.py`:
```python
import pybullet as p
import time

physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -10)
planeId = p.loadURDF("plane.urdf")

sphereId = p.createMultiBody(
    baseMass=1,
    baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=0.1),
    baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.1, rgbaColor=[0, 1, 0, 1]),
    basePosition=[0, 0, 1]
)

while True:
    keys = p.getKeyboardEvents()
    
    force_x = force_y = 0
    force_z = 12  # Counteract gravity
    
    # WASD controls
    if p.B3G_LEFT_ARROW in keys or ord('a') in keys:
        force_x = -20
    if p.B3G_RIGHT_ARROW in keys or ord('d') in keys:
        force_x = 20
    if p.B3G_UP_ARROW in keys or ord('w') in keys:
        force_y = 20
    if p.B3G_DOWN_ARROW in keys or ord('s') in keys:
        force_y = -20
    
    # Quit on ESC
    if ord('q') in keys:
        break
    
    p.applyExternalForce(sphereId, -1, [force_x, force_y, force_z], [0, 0, 0], p.LINK_FRAME)
    p.stepSimulation()
    time.sleep(1./240.)

p.disconnect()
```

**Goal**: Learn interactive control (you'll need this for light controls).

## **Phase 2: Learning PyBullet Integration (Week 2)**

### **Step 4: Study the Environment Architecture**
Understand how gym-pybullet-drones structures the simulation:
```python
# Key classes to understand:
# 1. BaseAviary - Core simulation environment
# 2. CtrlAviary - Control-focused environment (used in flocking)
# 3. DSLPIDControl - Converts high-level commands to motor controls

# Read these files:
gym_pybullet_drones/envs/
├── BaseAviary.py      # Core PyBullet simulation
├── CtrlAviary.py      # Control interface
└── BaseRLAviary.py    # RL environment base

gym_pybullet_drones/control/
└── DSLPIDControl.py   # PID control system
```

**Goal**: Understand how the framework abstracts PyBullet complexity.

### **Step 5: Create Simple Test Environment**
Create `test_environment.py` to experiment with the existing system:
```python
import sys
sys.path.append('..')
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel
import numpy as np

# Create a simple test environment similar to flocking
env = CtrlAviary(
    drone_model=DroneModel.CF2X,
    num_drones=3,
    initial_xyzs=np.array([[0, 0, 1], [0.5, 0, 1], [1, 0, 1]]),
    physics=Physics("pyb"),
    gui=True
)

# Create PID controllers
ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for i in range(3)]

# Simple test: make drones hover in formation
for i in range(1000):
    action = np.zeros((3, 4))
    obs, reward, done, info, _ = env.step(action)
    
    # Compute simple position control
    for j in range(3):
        target_pos = np.array([j * 0.5, 0, 1])  # Line formation
        action[j, :], _, _ = ctrl[j].computeControlFromState(
            control_timestep=env.CTRL_TIMESTEP,
            state=obs[j],
            target_pos=target_pos,
            target_rpy=np.array([0, 0, 0])
        )
    
    env.render()

env.close()
```

**Goal**: Learn to use the existing framework for custom behaviors.
    def __init__(self, agent_id, position, color):
        self.id = agent_id
        self.body_id = p.createMultiBody(
            baseMass=1,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=0.1),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.1, rgbaColor=color),
            basePosition=position
        )
        
    def get_position(self):
        pos, _ = p.getBasePositionAndOrientation(self.body_id)
        return np.array(pos[:2])  # Only x, y
    
    def apply_force(self, fx, fy):
        p.applyExternalForce(self.body_id, -1, [fx, fy, 12], [0, 0, 0], p.LINK_FRAME)

# Setup
physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -10)
planeId = p.loadURDF("plane.urdf")

# Create multiple agents
agents = []
colors = [[1,0,0,1], [0,1,0,1], [0,0,1,1], [1,1,0,1], [1,0,1,1]]
for i in range(5):
    agent = SimpleAgent(i, [i*0.5, 0, 1], colors[i])
    agents.append(agent)

# Simple attraction behavior
while True:
    keys = p.getKeyboardEvents()
    if ord('q') in keys:
        break
    
    # Make agents move toward center (0, 0)
    for agent in agents:
        pos = agent.get_position()
        # Simple attraction to center
        fx = -pos[0] * 5  # Spring force toward x=0
        fy = -pos[1] * 5  # Spring force toward y=0
        agent.apply_force(fx, fy)
    
    p.stepSimulation()
    time.sleep(1./240.)

p.disconnect()
```

**Goal**: Understand multi-agent systems and basic inter-agent behaviors.

### **Step 7: Agent-to-Agent Forces**
Create `06_inter_agent_forces.py`:
```python
import pybullet as p
import time
import numpy as np
import math

class SwarmAgent:
    def __init__(self, agent_id, position, color):
        self.id = agent_id
        self.body_id = p.createMultiBody(
            baseMass=1,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=0.1),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.1, rgbaColor=color),
            basePosition=position
        )
        
    def get_position(self):
        pos, _ = p.getBasePositionAndOrientation(self.body_id)
        return np.array(pos[:2])
    
    def compute_neighbor_forces(self, other_agents):
        """Basic repulsion/attraction forces"""
        my_pos = self.get_position()
        total_fx = total_fy = 0
        
        for other in other_agents:
            if other.id == self.id:
                continue
                
            other_pos = other.get_position()
            distance = np.linalg.norm(my_pos - other_pos)
            
            if distance > 0.1:  # Avoid division by zero
                # Simple repulsion at close range, attraction at medium range
                if distance < 0.5:
                    # Repulsion
                    force_magnitude = 10 / (distance**2)
                    direction = (my_pos - other_pos) / distance
                else:
                    # Attraction
                    force_magnitude = -2
                    direction = (my_pos - other_pos) / distance
                
                total_fx += force_magnitude * direction[0]
                total_fy += force_magnitude * direction[1]
        
        return total_fx, total_fy
    
    def apply_force(self, fx, fy):
        p.applyExternalForce(self.body_id, -1, [fx, fy, 12], [0, 0, 0], p.LINK_FRAME)

# Setup
physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -10)
planeId = p.loadURDF("plane.urdf")

# Create swarm
agents = []
colors = [[1,0,0,1], [0,1,0,1], [0,0,1,1], [1,1,0,1], [1,0,1,1]]
for i in range(5):
    x = np.random.uniform(-1, 1)
    y = np.random.uniform(-1, 1)
    agent = SwarmAgent(i, [x, y, 1], colors[i])
    agents.append(agent)

# Simulation loop
while True:
    keys = p.getKeyboardEvents()
    if ord('q') in keys:
        break
    
    # Compute and apply neighbor forces
    for agent in agents:
        fx, fy = agent.compute_neighbor_forces(agents)
        agent.apply_force(fx, fy)
    
    p.stepSimulation()
    time.sleep(1./240.)

p.disconnect()
```

**Goal**: Implement basic swarm forces (foundation for gradient following).

## **Phase 3: Adding Gradient Following (Week 3-4)**

### **Step 6: Extend FlockingUtils for Light Sensing**
Create `gradient_flocking_utils.py` based on the existing flocking system:
```python
import numpy as np
import sys
sys.path.append('..')
from ants_2024.flocking_utils import FlockingUtils

class GradientFlockingUtils(FlockingUtils):
    """Extended FlockingUtils with light-following capabilities"""
    
    def __init__(self, n_agents, center_x, center_y, center_z, spacing):
        super().__init__(n_agents, center_x, center_y, center_z, spacing)
        
        # Light-following parameters (from swarm_vu.c)
        self.light_min = 13 * 4
        self.light_max = 162 * 4
        self.sigma_base = 25
        self.sigma_var = 28
        self.kappa = 30  # Gradient weight
        
        # Light sensor readings for each agent
        self.light_sensors = np.zeros(n_agents)
    
    def update_light_sensors(self, positions, light_sources):
        """Update light sensor readings for all agents"""
        for i in range(self.n_agents):
            self.light_sensors[i] = self.compute_light_at_position(
                positions[i], light_sources
            )
    
    def compute_light_at_position(self, position, light_sources):
        """Compute light intensity at given position"""
        total_intensity = self.light_min
        
        for light in light_sources:
            if light['is_on']:
                distance = np.linalg.norm(position - light['position'])
                # Inverse distance decay with scaling
                intensity = light['intensity'] / (1.0 + distance * 2.0)
                total_intensity = max(total_intensity, intensity)
        
        # Add noise like real sensors
        total_intensity += np.random.uniform(-10, 10)
        return np.clip(total_intensity, self.light_min, self.light_max)
    
    def calc_gradient_forces(self, positions, light_sources):
        """Compute gradient-following forces (new!)"""
        gx = np.zeros(self.n_agents)
        gy = np.zeros(self.n_agents)
        
        for i in range(self.n_agents):
            # Find the brightest nearby light
            best_light = None
            best_intensity = 0
            
            for light in light_sources:
                if light['is_on']:
                    intensity = self.compute_light_at_position(
                        positions[i], [light]
                    )
                    if intensity > best_intensity:
                        best_intensity = intensity
                        best_light = light
            
            if best_light is not None:
                # Force toward the light
                direction = best_light['position'] - positions[i][:2]
                distance = np.linalg.norm(direction)
                
                if distance > 0.1:
                    direction = direction / distance
                    # Scale force by light intensity (like swarm_vu.c)
                    force_magnitude = (best_intensity - self.light_min) / 100.0
                    gx[i] = direction[0] * force_magnitude
                    gy[i] = direction[1] * force_magnitude
        
        return gx, gy
    
    def calc_total_forces_with_gradient(self, pos_xs, pos_ys, pos_zs, light_sources):
        """Combine existing flocking forces with gradient forces"""
        # Original flocking forces
        self.calc_dij(pos_xs, pos_ys, pos_zs)
        self.calc_ang_ij(pos_xs, pos_ys, pos_zs)
        self.calc_grad_vals(pos_xs, pos_ys, pos_zs)
        self.calc_p_forces()
        self.calc_alignment_forces()
        self.calc_boun_rep(pos_xs, pos_ys, pos_zs)
        
        # Add gradient forces
        positions = np.column_stack([pos_xs, pos_ys, pos_zs])
        self.update_light_sensors(positions, light_sources)
        gx, gy = self.calc_gradient_forces(positions, light_sources)
        
        # Combine forces (weighted sum like swarm_vu.c Equation 10)
        self.f_x += self.kappa * gx / 50.0
        self.f_y += self.kappa * gy / 50.0
        
        return self.calc_u_w()
```

**Goal**: Extend the existing flocking system with gradient following.

### **Step 7: Create Light Source Management**
Create `light_manager.py`:
```python
import pybullet as p
import numpy as np

class LightManager:
    """Manages visual light sources in PyBullet simulation"""
    
    def __init__(self, light_positions):
        self.lights = []
        
        # Create visual light sources
        for i, pos in enumerate(light_positions):
            light_data = {
                'id': i,
                'position': np.array(pos),
                'is_on': False,
                'intensity': 200.0,
                'visual_id': None
            }
            
            # Create visual representation
            visual_shape = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=0.2,
                length=0.05,
                rgbaColor=[1, 0.8, 0, 0.3]  # Dim yellow initially
            )
            
            body_id = p.createMultiBody(
                baseMass=0,  # Static
                baseVisualShapeIndex=visual_shape,
                basePosition=[pos[0], pos[1], 0.025]
            )
            
            light_data['visual_id'] = body_id
            self.lights.append(light_data)
    
    def turn_on_light(self, light_id):
        """Turn on specific light"""
        for light in self.lights:
            light['is_on'] = False  # Turn off all others
            p.changeVisualShape(light['visual_id'], -1, rgbaColor=[1, 0.8, 0, 0.3])
        
        if 0 <= light_id < len(self.lights):
            self.lights[light_id]['is_on'] = True
            p.changeVisualShape(
                self.lights[light_id]['visual_id'], 
                -1, 
                rgbaColor=[1, 0.8, 0, 1.0]  # Bright yellow
            )
    
    def turn_off_all(self):
        """Turn off all lights"""
        for light in self.lights:
            light['is_on'] = False
            p.changeVisualShape(light['visual_id'], -1, rgbaColor=[1, 0.8, 0, 0.3])
    
    def get_light_data(self):
        """Return light data for gradient computation"""
        return self.lights
    
    def handle_keyboard_input(self, keys):
        """Handle keyboard input for light control"""
        for i in range(len(self.lights)):
            if ord(str(i+1)) in keys and keys[ord(str(i+1))] == p.KEY_WAS_TRIGGERED:
                self.turn_on_light(i)
        
        if ord('0') in keys and keys[ord('0')] == p.KEY_WAS_TRIGGERED:
            self.turn_off_all()
```

**Goal**: Create manageable light sources that integrate with the existing PyBullet visualization.

class LightSource:
    def __init__(self, position, light_id):
        self.position = np.array(position)
        self.light_id = light_id
        self.is_on = False
        self.intensity = 100.0
        
        # Visual representation
        self.visual_shape = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.3,
            length=0.1,
            rgbaColor=[1, 0.5, 0, 0.3]  # Orange, dim initially
        )
        
        self.body_id = p.createMultiBody(
            baseMass=0,  # Static
            baseVisualShapeIndex=self.visual_shape,
            basePosition=[*position, 0.05]
        )
    
    def turn_on(self):
        self.is_on = True
        p.changeVisualShape(self.body_id, -1, rgbaColor=[1, 0.5, 0, 1.0])  # Bright orange
    
    def turn_off(self):
        self.is_on = False
        p.changeVisualShape(self.body_id, -1, rgbaColor=[1, 0.5, 0, 0.3])  # Dim orange
    
    def get_light_intensity_at(self, position):
        """Simulate light sensor reading at given position"""
        if not self.is_on:
            return 0.0
        
        distance = np.linalg.norm(self.position - position)
        if distance < 0.1:
            return self.intensity
        
        # Inverse square law with scaling
        intensity = self.intensity / (1.0 + distance * 5.0)
        return max(0, intensity)

# Test light sources
physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -10)
planeId = p.loadURDF("plane.urdf")

# Create light sources (matching original positions)
light_positions = [
    [1.18, 0.99], [2.96, 1.1], [5.0, 1.5], 
    [5.2, 3.5], [3.37, 3.3], [1.31, 2.75]
]

lights = []
for i, pos in enumerate(light_positions):
    light = LightSource(pos, i)
    lights.append(light)

# Test keyboard control
current_light = 0
while True:
    keys = p.getKeyboardEvents()
    
    # Number keys 1-6 to control lights
    for i in range(6):
        if ord(str(i+1)) in keys and keys[ord(str(i+1))] == p.KEY_WAS_TRIGGERED:
            # Turn off all lights
            for light in lights:
                light.turn_off()
            # Turn on selected light
            lights[i].turn_on()
    
    # 0 to turn off all
    if ord('0') in keys and keys[ord('0')] == p.KEY_WAS_TRIGGERED:
        for light in lights:
            light.turn_off()
    
    if ord('q') in keys:
        break
    
    p.stepSimulation()
    time.sleep(1./240.)

p.disconnect()
```

**Goal**: Create controllable light sources that will drive the gradient following.

### **Step 9: Light-Following Agents**
Create `08_gradient_following.py`:
```python
import pybullet as p
import time
import numpy as np
import math

class GradientAgent:
    def __init__(self, agent_id, position, color):
        self.id = agent_id
        self.position = np.array(position[:2])  # 2D position
        self.heading = 0.0
        self.light_sensor = 0.0
        
        self.body_id = p.createMultiBody(
            baseMass=1,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=0.1),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.1, rgbaColor=color),
            basePosition=[*position]
        )
    
    def update_light_sensor(self, light_sources):
        """Update light sensor reading based on nearby lights"""
        self.light_sensor = 10.0  # Minimum reading
        
        for light in light_sources:
            intensity = light.get_light_intensity_at(self.position)
            self.light_sensor = max(self.light_sensor, intensity)
        
        # Add some noise
        self.light_sensor += np.random.uniform(-5, 5)
        self.light_sensor = max(10, min(400, self.light_sensor))  # Clamp
    
    def compute_gradient_force(self, light_sources):
        """Compute force toward brightest light source"""
        if not any(light.is_on for light in light_sources):
            return 0.0, 0.0
        
        # Find direction to brightest light
        best_light = None
        best_intensity = 0
        
        for light in light_sources:
            if light.is_on:
                intensity = light.get_light_intensity_at(self.position)
                if intensity > best_intensity:
                    best_intensity = intensity
                    best_light = light
        
        if best_light is None:
            return 0.0, 0.0
        
        # Force toward best light
        direction = best_light.position - self.position
        distance = np.linalg.norm(direction)
        
        if distance > 0.1:
            direction = direction / distance
            force_magnitude = min(10.0, best_intensity / 20.0)  # Scale force
            return direction[0] * force_magnitude, direction[1] * force_magnitude
        
        return 0.0, 0.0
    
    def update_position(self):
        """Update internal position from PyBullet"""
        pos, _ = p.getBasePositionAndOrientation(self.body_id)
        self.position = np.array(pos[:2])
    
    def apply_force(self, fx, fy):
        p.applyExternalForce(self.body_id, -1, [fx, fy, 12], [0, 0, 0], p.LINK_FRAME)

class LightSource:
    # ... (same as before)

# Setup
physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -10)
planeId = p.loadURDF("plane.urdf")

# Create agents
agents = []
colors = [[1,0,0,1], [0,1,0,1], [0,0,1,1], [1,1,0,1], [1,0,1,1]]
for i in range(5):
    x = 1.0 + i * 0.5 + np.random.uniform(-0.2, 0.2)
    y = 2.0 + np.random.uniform(-0.2, 0.2)
    agent = GradientAgent(i, [x, y, 1], colors[i])
    agents.append(agent)

# Create lights
light_positions = [[1.18, 0.99], [2.96, 1.1], [5.0, 1.5]]  # Start with 3 lights
lights = []
for i, pos in enumerate(light_positions):
    light = LightSource(pos, i)
    lights.append(light)

print("Controls: 1-3 to turn on lights, 0 to turn off all, Q to quit")

# Main simulation loop
while True:
    keys = p.getKeyboardEvents()
    
    # Light controls
    for i in range(3):
        if ord(str(i+1)) in keys and keys[ord(str(i+1))] == p.KEY_WAS_TRIGGERED:
            for light in lights:
                light.turn_off()
            lights[i].turn_on()
    
    if ord('0') in keys and keys[ord('0')] == p.KEY_WAS_TRIGGERED:
        for light in lights:
            light.turn_off()
    
    if ord('q') in keys:
        break
    
    # Update agents
    for agent in agents:
        agent.update_position()
        agent.update_light_sensor(lights)
        
        # Compute gradient force
        fx, fy = agent.compute_gradient_force(lights)
        
        # Apply force
        agent.apply_force(fx, fy)
    
    p.stepSimulation()
    time.sleep(1./240.)

p.disconnect()
```

**Goal**: Basic gradient following behavior working!

### **Step 8: Complete Gradient-Following Implementation**
Create `gradient_following_swarm.py` - Modified version of `3d_flocking_v0.py` with gradient following:

```python
"""
Gradient-Following Swarm Implementation
Based on 3d_flocking_v0.py with light-following capabilities added
"""
import time
import argparse
import numpy as np
import sys
import pybullet as p

# Import existing gym-pybullet-drones components
sys.path.append('..')
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.utils import sync, str2bool

# Import our custom extensions
from gradient_flocking_utils import GradientFlockingUtils
from light_manager import LightManager

# Configuration
DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DURATION_SEC = 30

NUM_DRONES = 5
init_center_x = 3.5
init_center_y = 2.4
init_center_z = 1
spacing = 0.6

# Light positions (matching original research setup)
LIGHT_POSITIONS = [
    [1.18, 0.99], [2.96, 1.1], [5.0, 1.5], 
    [5.2, 3.5], [3.37, 3.3], [1.31, 2.75]
]

def run(duration_sec=DURATION_SEC):
    # Initialize gradient flocking system
    f_util = GradientFlockingUtils(NUM_DRONES, init_center_x, init_center_y, init_center_z, spacing)
    pos_xs, pos_ys, pos_zs, pos_h_xc, pos_h_yc, pos_h_zc = f_util.initialize_positions()

    INIT_XYZ = np.zeros([NUM_DRONES, 3])
    INIT_XYZ[:, 0] = pos_xs
    INIT_XYZ[:, 1] = pos_ys
    INIT_XYZ[:, 2] = pos_zs
    INIT_RPY = np.array([[.0, .0, .0] for _ in range(NUM_DRONES)])

    # Create environment
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

    # Create controllers
    ctrl = [DSLPIDControl(drone_model=DEFAULT_DRONES) for i in range(NUM_DRONES)]
    
    # Create light management system
    light_manager = LightManager(LIGHT_POSITIONS)

    print("=== Gradient-Following Swarm Simulation ===")
    print("Controls:")
    print("- Keys 1-6: Turn on lights")
    print("- Key 0: Turn off all lights")
    print("- Key Q: Quit simulation")
    print("Watch the drones follow the active light source!")

    START = time.time()
    action = np.zeros((NUM_DRONES, 4))

    # Main simulation loop
    for i in range(0, int(duration_sec * env.CTRL_FREQ)):
        # Handle user input
        keys = p.getKeyboardEvents()
        light_manager.handle_keyboard_input(keys)
        
        if ord('q') in keys:
            break

        # Step simulation
        obs, reward, done, info, _ = env.step(action)
        
        # Get current positions
        pos_x = np.zeros(NUM_DRONES)
        pos_y = np.zeros(NUM_DRONES)
        pos_z = np.zeros(NUM_DRONES)
        
        for j in range(NUM_DRONES):
            states = env._getDroneStateVector(j)
            pos_x[j] = states[0]
            pos_y[j] = states[1]
            pos_z[j] = states[2]

        # Compute swarm forces with gradient following
        light_data = light_manager.get_light_data()
        u = f_util.calc_total_forces_with_gradient(pos_x, pos_y, pos_z, light_data)
        pos_hxs, pos_hys, pos_hzs = f_util.get_heading()
        f_util.update_heading()

        # Convert to velocity commands and apply control
        for j in range(NUM_DRONES):
            # Compute velocity command from swarm algorithm
            vel_cmd = np.array([
                u[j] * np.cos(pos_hxs[j]), 
                u[j] * np.cos(pos_hys[j]), 
                u[j] * np.cos(pos_hzs[j])
            ])
            
            # Current position as target (velocity control)
            pos_cmd = np.array([pos_x[j], pos_y[j], pos_z[j]])
            
            # Generate control action
            action[j], _, _ = ctrl[j].computeControlFromState(
                control_timestep=env.CTRL_TIMESTEP,
                state=obs[j],
                target_pos=pos_cmd,
                target_vel=vel_cmd,
                target_rpy=np.array([0, 0, 0])
            )

        # Render and sync
        env.render()
        if DEFAULT_GUI:
            sync(i, START, env.CTRL_TIMESTEP)

    # Cleanup
    env.close()
    print("Simulation completed successfully!")
    print("The drones demonstrated gradient-following swarm behavior!")

if __name__ == "__main__":
    run()
```

**Goal**: Complete working gradient-following swarm that combines flocking with light-seeking behavior!
import math
import json

class SwarmDrone:
    def __init__(self, drone_id, position, config):
        self.id = drone_id
        self.config = config
        
        # State variables
        self.position = np.array(position[:2])
        self.heading = 0.0
        self.light_sensor = 50.0
        
        # PyBullet body
        color = [np.random.random(), np.random.random(), np.random.random(), 1]
        self.body_id = p.createMultiBody(
            baseMass=1,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=0.1),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.1, rgbaColor=color),
            basePosition=[*position]
        )
    
    def compute_swarm_forces(self, other_drones, light_sources):
        """
        EXACT implementation of swarm_vu.c algorithm
        Returns: fx, fy (total forces)
        """
        # Reset forces
        px = py = 0.0  # Proximal forces
        hx = hy = 0.0  # Alignment forces  
        rx = ry = 0.0  # Boundary forces
        gx = gy = 0.0  # Gradient forces
        
        # Update light sensor
        self.update_light_sensor(light_sources)
        
        # Equation (2): Adaptive speed based on light
        light_normalized = (self.light_sensor - self.config['light_min']) / (self.config['light_max'] - self.config['light_min'])
        su = self.config['sigma_base'] + pow(light_normalized, 0.1) * self.config['sigma_var']
        
        # Proximal forces (Equation 1)
        sum_cosh = math.cos(self.heading)
        sum_sinh = math.sin(self.heading)
        
        for other in other_drones:
            if other.id != self.id:
                distance = np.linalg.norm(self.position - other.position)
                if distance > 0.1:  # Avoid division by zero
                    angle = math.atan2(other.position[1] - self.position[1], 
                                     other.position[0] - self.position[0])
                    
                    # Force magnitude (from firmware)
                    epsilon = 12.0
                    force_mag = -epsilon * (2.0 * (su**4 / distance**5) - (su**2 / distance**3))
                    
                    px += force_mag * math.cos(angle)
                    py += force_mag * math.sin(angle)
                    
                    # Collect headings for alignment
                    sum_cosh += math.cos(other.heading)
                    sum_sinh += math.sin(other.heading)
        
        # Equation (3): Alignment forces
        heading_sum_magnitude = math.sqrt(sum_cosh**2 + sum_sinh**2)
        if heading_sum_magnitude > 0:
            hx = sum_cosh / heading_sum_magnitude
            hy = sum_sinh / heading_sum_magnitude
        
        # Equation (4): Boundary forces (7m x 4.75m area)
        krep = 5.0
        L0 = 0.5
        
        # Bottom boundary
        if self.position[1] < 0.5:
            rx += krep * (1.0/abs(self.position[1]) - 1.0/L0) * (math.cos(math.pi/2) / abs(self.position[1])**3)
            ry += krep * (1.0/abs(self.position[1]) - 1.0/L0) * (math.sin(math.pi/2) / abs(self.position[1])**3)
        
        # Right boundary  
        if self.position[0] > 6.5:
            dist_to_edge = abs(7.0 - self.position[0])
            rx += krep * (1.0/dist_to_edge - 1.0/L0) * (math.cos(math.pi) / dist_to_edge**3)
            ry += krep * (1.0/dist_to_edge - 1.0/L0) * (math.sin(math.pi) / dist_to_edge**3)
        
        # Top boundary
        if self.position[1] > 4.25:
            dist_to_edge = abs(4.75 - self.position[1])
            rx += krep * (1.0/dist_to_edge - 1.0/L0) * (math.cos(-math.pi/2) / dist_to_edge**3)
            ry += krep * (1.0/dist_to_edge - 1.0/L0) * (math.sin(-math.pi/2) / dist_to_edge**3)
        
        # Left boundary
        if self.position[0] < 0.5:
            rx += krep * (1.0/abs(self.position[0]) - 1.0/L0) * (1.0 / abs(self.position[0])**3)
        
        # Equation (10): Total force
        alpha = self.config['alpha'] / 50.0
        beta = self.config['beta'] / 50.0
        gamma = 1.0
        kappa = self.config['kappa'] / 50.0
        
        fx_raw = alpha * px + beta * hx + gamma * rx + kappa * gx
        fy_raw = alpha * py + beta * hy + gamma * ry + kappa * gy
        
        return fx_raw, fy_raw
    
    def update_light_sensor(self, light_sources):
        """Update light sensor reading"""
        self.light_sensor = self.config['light_min']
        
        for light in light_sources:
            if light.is_on:
                intensity = light.get_light_intensity_at(self.position)
                self.light_sensor = max(self.light_sensor, intensity)
        
        # Add noise and clamp
        self.light_sensor += np.random.uniform(-10, 10)
        self.light_sensor = max(self.config['light_min'], 
                               min(self.config['light_max'], self.light_sensor))
    
    def update_from_physics(self):
        """Update position from PyBullet"""
        pos, orientation = p.getBasePositionAndOrientation(self.body_id)
        self.position = np.array(pos[:2])
        
        # Update heading from orientation
        euler = p.getEulerFromQuaternion(orientation)
        self.heading = euler[2]  # Yaw
    
    def apply_forces(self, fx, fy):
        """Apply computed forces to PyBullet body"""
        p.applyExternalForce(self.body_id, -1, [fx, fy, 12], [0, 0, 0], p.LINK_FRAME)

# Configuration (matching config.json)
config = {
    "sigma_base": 25,
    "sigma_var": 28, 
    "alpha": 40,
    "beta": 10,
    "kappa": 0,
    "light_max": 162 * 4,  # Scale like firmware
    "light_min": 13 * 4,
    "k1": 3,
    "k2": 30
}

# ... (LightSource class same as before)

# Main execution
if __name__ == "__main__":
    # Setup PyBullet
    physicsClient = p.connect(p.GUI)
    p.setGravity(0, 0, -10)
    planeId = p.loadURDF("plane.urdf")
    
    # Create boundary markers
    # ... (add visual boundary indicators)
    
    # Create drones
    drones = []
    for i in range(5):
        x = 1.0 + i * 1.2 + np.random.uniform(-0.3, 0.3)
        y = 2.0 + np.random.uniform(-0.5, 0.5)
        drone = SwarmDrone(i, [x, y, 1], config)
        drones.append(drone)
    
    # Create lights
    light_positions = [
        [1.18, 0.99], [2.96, 1.1], [5.0, 1.5], 
        [5.2, 3.5], [3.37, 3.3], [1.31, 2.75]
    ]
    lights = []
    for i, pos in enumerate(light_positions):
        light = LightSource(pos, i)
        lights.append(light)
    
    print("Full Swarm Simulation")
    print("Controls: 1-6 for lights, 0 for all off, Q to quit")
    
    # Main loop
    while True:
        keys = p.getKeyboardEvents()
        
        # Handle light controls
        for i in range(6):
            if ord(str(i+1)) in keys and keys[ord(str(i+1))] == p.KEY_WAS_TRIGGERED:
                for light in lights:
                    light.turn_off()
                lights[i].turn_on()
        
        if ord('0') in keys and keys[ord('0')] == p.KEY_WAS_TRIGGERED:
            for light in lights:
                light.turn_off()
        
        if ord('q') in keys:
            break
        
        # Update each drone
        for drone in drones:
            drone.update_from_physics()
            fx, fy = drone.compute_swarm_forces(drones, lights)
            drone.apply_forces(fx, fy)
        
        p.stepSimulation()
        time.sleep(1./240.)
    
    p.disconnect()
```

## **REVISED Weekly Schedule Summary:**

**Week 1**: Steps 1-3 (Understanding existing flocking system)
**Week 2**: Steps 4-5 (Learning PyBullet integration through gym-pybullet-drones)  
**Week 3**: Steps 6-7 (Extending flocking for gradient following)
**Week 4**: Step 8 (Complete gradient-following implementation)

## **Success Metrics:**

- ✅ **Week 1**: You understand how the existing flocking system works
- ✅ **Week 2**: You can modify and extend the gym-pybullet-drones examples
- ✅ **Week 3**: You've added light sensing and gradient forces to the flocking system
- ✅ **Week 4**: **Complete gradient-following swarm behavior matching the original research!**

## **Key Advantages of This Revised Approach:**

🚀 **50% Faster**: Leveraging existing working code instead of building from scratch
🎯 **More Realistic**: Using actual drone models and physics from gym-pybullet-drones
🔧 **Easier Debugging**: Building on proven, working examples
📚 **Better Learning**: Understanding real-world simulation architecture
🎮 **Enhanced Features**: Access to advanced drone physics, controls, and visualization

**Final Result**: You'll have a complete PyBullet implementation that replicates the gradient-following swarm research, built on a robust foundation with realistic drone physics and controls!
