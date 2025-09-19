# Gradient Following in PyBullet: Implementation Plan

## High-Level Goals

Our primary objective is to replicate the emergent gradient-following behavior from your supervisor's research within the `gym-pybullet-drones` simulation environment. The key features of this implementation will be:

1.  **Image-Based Gradient:** The environment's gradient will be defined by a 2D PNG image, where pixel brightness corresponds to gradient strength.
2.  **Emergent Swarming:** Drones will not directly "climb" the gradient. Instead, the gradient's strength will dynamically alter the desired spacing between drones. This will cause the swarm to naturally move towards areas of higher gradient as an emergent behavior.
3.  **Modular & Testable:** We will build the system in distinct, logical stages, ensuring each component works correctly before integrating it into the final simulation.

---

## Detailed Implementation Steps

### Phase 1: Foundational Setup

*   **Task 1: Create the Simulation File**
    *   **Action:** Duplicate `2d_flocking_with_real_utils.py` and rename the copy to `gradient_following.py`.
    *   **Rationale:** This provides a working, multi-drone simulation structure that you are already comfortable with, including environment setup and PID control.

*   **Task 2: Prepare the Gradient Map**
    *   **Action:** Copy the `linear_4x65.png` file from your supervisor's `DynamicSimulationGradFollow` repository into the `gym_pybullet_drones/assets/` directory.
    *   **Rationale:** This gives us a known gradient map to test against, ensuring our results are comparable to the original research.

### Phase 2: Gradient Sensing & Verification ("Sensor Check")

*   **Task 3: Create the `GradientMap` Class**
    *   **Action:** In `gradient_following.py`, create a new class named `GradientMap`.
    *   **Functionality:**
        *   The constructor (`__init__`) will load the PNG file from the assets folder and convert it into a NumPy array.
        *   It will store the mapping between the simulation's world coordinates (in meters) and the image's pixel coordinates.
        *   It will have a method, `get_gradient_at_position(x, y)`, that takes a drone's world coordinates and returns the corresponding gradient value (pixel brightness).

*   **Task 4: Implement a "Hover and Report" Algorithm**
    *   **Action:** In the main simulation loop of `gradient_following.py`, remove the existing flocking logic.
    *   **New Logic:**
        *   For each drone, the target position will be its own current position (to make it hover).
        *   In each simulation step, each drone will call the `get_gradient_at_position()` method from our `GradientMap` instance to read the value underneath it.
        *   Print the drone's ID and its sensed gradient value to the console.
    *   **Goal:** Verify that our coordinate mapping is correct and that the drones can accurately "sense" the environment before we add complex movement.

### Phase 3: Implement the Gradient Swarm Algorithm

*   **Task 5: Create the `GradientSwarm` Controller**
    *   **Action:** Create a new class named `GradientSwarm`. This class will contain the core logic adapted from your supervisor's `dm.py`.
    *   **Functionality:**
        *   It will take the drone states and the `GradientMap` as input.
        *   It will contain a method to calculate the gradient-influenced desired spacing between drones (adapting the logic from `calc_proximal`).
        *   It will contain a method to calculate the forces and resulting velocity commands based on the difference between actual and desired spacing (adapting the logic from `calc_force`).

### Phase 4: Integration and Final Simulation

*   **Task 6: Integrate the Swarm Controller**
    *   **Action:** In the main simulation loop, remove the "Hover and Report" code.
    *   **New Logic:**
        *   Instantiate the `GradientSwarm` controller.
        *   In each step, pass the current drone states to the controller.
        *   Get the calculated velocity commands from the controller and send them to the drones.

*   **Task 7: Run and Observe**
    *   **Action:** Run the final `gradient_following.py` script.
    *   **Expected Outcome:** The drone swarm should exhibit the desired emergent behavior, moving collectively towards the areas of higher gradient strength on the map.
