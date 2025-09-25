# note: every time we do planning, update this memory bank
# also make it a point to have incremental changes such that code reviews stay manageable
# Gradient Following Implementation Strategy

## Core Understanding ✅

**Drones have NO knowledge of light source position!**

### What Drones DON'T Know:
- ❌ Light source position (X, Y coordinates)
- ❌ Distance to light source
- ❌ Direction to light source

### What Drones DO Know:
- ✅ Light intensity reading (scalar value from their sensor)
- ✅ Local light intensity affects their flocking behavior

## How Gradient Following ACTUALLY Works:

🚨 **CRITICAL CORRECTION**: The gradient following algorithm does NOT use explicit gradient ascent!

### The REAL Algorithm:
1. **Drone reads light intensity** at its current position
2. **Adaptive spacing calculation** using Equation (2): `su = sb + pow((light_log-lmn)/(lmx - lmn), 0.1)*sv`
3. **Modified flocking behavior** where spacing changes based on light intensity
4. **Natural aggregation** in bright areas due to larger spacing when light is bright

### What We Learned from Debugging:
- ❌ **WRONG**: Explicit gradient ascent (sampling nearby points, calculating direction)
- ✅ **CORRECT**: Adaptive spacing modification in flocking algorithm
- ❌ **WRONG**: Direct coordinate mapping (x->x, y->y)
- ✅ **CORRECT**: Coordinate swap from research (x->y, y->x) is intentional and necessary

## Real Research Implementation:

From Capstone firmware (`swarm_vu.c`):

```c
// Drone reads light sensor
a_read = analogRead(DECK_GPIO_TX2);
light_log = a_read;

// Light intensity bounds
if (light_log < lmn) {light_log=lmn;}
if (light_log >lmx) {light_log=lmx;}

// Equation (2) - Adaptive spacing based on light intensity
su = sb + pow((light_log-lmn)/(lmx - lmn), 0.1)*sv;
```

From original research (`dm_ds_v2.py`):
```python
# Coordinate mapping (CORRECT - do not change!)
self.grad_y = np.ceil(np.multiply(self.cf_x[0],self.grad_const_y))  # x->y
self.grad_x = np.ceil(np.multiply(self.cf_y[0],self.grad_const_x))  # y->x
```

## Implementation Strategy:

### ❌ WRONG Approach (Gradient Ascent):
```python
# Calculate gradient by sampling nearby points
gradient_x = intensity_at_x_plus - intensity_at_x_minus
gradient_y = intensity_at_y_plus - intensity_at_y_minus
# Move in gradient direction
```

### ✅ CORRECT Approach (Adaptive Spacing):
```python
# Read light intensity
light_intensity = read_light_intensity(pos_x, pos_y)
# Calculate adaptive spacing
su = sb + pow((light_capped-lmn)/(lmx-lmn), 0.1)*sv
# Modify flocking behavior based on spacing
```

## Key Benefits:

1. **Realistic**: Matches real-world sensor limitations
2. **Robust**: Works even if light source moves
3. **Research-Accurate**: Implements the actual algorithms from the paper
4. **Emergent Behavior**: Drones naturally converge on light sources through modified flocking

## Debugging Lessons Learned:

1. **Coordinate Mapping**: The x->y, y->x swap from `dm_ds_v2.py` IS CORRECT - do not "fix" it
2. **Algorithm Type**: Adaptive spacing modification, NOT explicit gradient ascent
3. **Parameter Values**: Use firmware values (lmn=50, lmx=450, not 0-255)
4. **Integration**: Modify flocking forces, don't add separate gradient forces

## Status: ✅ CORRECTED AND VERIFIED

---

## 📋 GRADIENT MAP IMPLEMENTATION PLAN

### ✅ **Approach Confirmed:**

**Physical Dimensions:**
- **World Size**: `6.5m x 4.0m` (matches `linear_4x65.png`)
- **Coordinate System**: `(0,0)` to `(6.5, 4.0)`
- **Gradient Map**: Use existing `linear_4x65.png`

**Implementation Details:**
```python
# Gradient Map Configuration
GRADIENT_MAP_PATH = "linear_4x65.png"
WORLD_SIZE_X = 6.5  # meters
WORLD_SIZE_Y = 4.0  # meters

# Coordinate Mapping (from dm_ds_v2.py)
grad_const_x = (len(np.arange(start=0.00, stop=WORLD_SIZE_X, step=0.04))) / WORLD_SIZE_X
grad_const_y = (len(np.arange(start=0.00, stop=WORLD_SIZE_Y, step=0.04))) / WORLD_SIZE_Y

# Drone Initialization (around center like dm_ds_v2.py)
init_center_x = 3.25  # WORLD_SIZE_X / 2
init_center_y = 2.0   # WORLD_SIZE_Y / 2
```

**Key Features:**
- ✅ **Scalar light intensity readings** from gradient map
- ✅ **Realistic sensor noise** (like dm_ds_v2.py)
- ✅ **Coordinate mapping** from PyBullet to gradient map
- ✅ **Linear gradient pattern** for testing

### 📝 **Documentation Requirement:**
**IMPORTANT**: Always include comments in code indicating:
- **Source/Inspiration**: Which of the three repos this comes from
- **Specific files**: Reference the exact files (e.g., `dm_ds_v2.py`, `swarm_vu.c`)
- **Algorithm details**: Which equations/approaches are being used

**Example:**
```python
# Gradient following algorithm inspired by:
# - Capstone_gradient_following/cf-firmware-gradient-vu/examples/app_gradient/src/swarm_vu.c
# - DynamicSimulationGradFollow/Dynamic Simulaton/dm_ds_v2.py
# Uses Equation (2) for adaptive spacing: su = sb + pow((light_log-lmn)/(lmx - lmn), 0.1)*sv
```

### 🔄 **Incremental Development Strategy:**
**IMPORTANT**: Make small, manageable changes for easier code review:

1. **One feature at a time** - Don't change multiple things simultaneously
2. **Test each step** - Ensure each change works before moving to the next
3. **Clear commit messages** - Document what each change does
4. **Backup working versions** - Keep previous working states
5. **Small pull requests** - Easier to review and debug

**Development Order:**
1. ✅ **Add gradient map loading** (no behavior change yet)
2. ✅ **Add coordinate mapping** (no behavior change yet)  
3. ✅ **Add light sensor simulation** (replace goal-seeking)
4. ✅ **Add gradient following algorithm** (Capstone equations)
5. ✅ **Test and tune parameters**
