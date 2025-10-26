"""
Swarm Visualization Module
==========================
Handles all visualization and plotting for swarm gradient following simulations.

Functions:
- create_drone_position_overlay(): Overlay drone positions on gradient map
- create_analysis_dashboard(): Generate comprehensive 5-panel analysis figure
- print_simulation_summary(): Print formatted terminal summary statistics
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


def create_drone_position_overlay(
    drone_positions_x, 
    drone_positions_y, 
    gradient_map_path,
    world_size_x,
    world_size_y,
    output_folder, 
    timestamp=None, 
    show_plot=False
):
    """
    Overlays drone positions on the gradient map.
    
    Args:
        drone_positions_x: Array of X positions of drones (meters)
        drone_positions_y: Array of Y positions of drones (meters)
        gradient_map_path: Path to gradient map image file
        world_size_x: World width in meters
        world_size_y: World height in meters
        output_folder: Directory to save the output image
        timestamp: Optional timestamp string for filename (e.g., "10s", "20s", "final")
        show_plot: Whether to display the plot (default: False)
    
    Returns:
        str: Path to saved image file
    """
    if timestamp is None:
        print("\n📸 Creating overlay of final drone positions on gradient map...")
        filename = "final_positions_overlay.png"
    else:
        print(f"📸 Snapshot at {timestamp}...")
        filename = f"positions_at_{timestamp}.png"
    
    try:
        # Load the gradient map image
        img = Image.open(gradient_map_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        
        # Calculate coordinate mapping constants
        step_size = 0.04
        grad_const_x = (len(np.arange(start=0.00, stop=world_size_x, step=step_size))) / world_size_x
        grad_const_y = (len(np.arange(start=0.00, stop=world_size_y, step=step_size))) / world_size_y
        
        # Draw each drone's position
        for i in range(len(drone_positions_x)):
            pybullet_x = drone_positions_x[i]
            pybullet_y = drone_positions_y[i]
            
            # Convert PyBullet coords to image pixel coords
            map_y = int(np.ceil(pybullet_x * grad_const_y))
            map_x = int(np.ceil(pybullet_y * grad_const_x))
            
            # Invert Y-axis: PyBullet Y increases upward, but image rows increase downward
            map_x = img.height - 1 - map_x
            
            # Clip to image bounds
            map_x_clipped = np.clip(map_x, 0, img.height - 1)
            map_y_clipped = np.clip(map_y, 0, img.width - 1)
            
            # Draw a circle for the drone
            radius = 5
            draw.ellipse(
                (map_y_clipped - radius, map_x_clipped - radius, 
                 map_y_clipped + radius, map_x_clipped + radius),
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
        
        return output_path
        
    except FileNotFoundError:
        print(f"[ERROR] Could not create overlay. Gradient map not found at {gradient_map_path}")
        return None
    except Exception as e:
        print(f"[ERROR] An error occurred while creating the overlay: {e}")
        return None


def create_analysis_dashboard(
    time_data,
    centroid_x_data,
    centroid_y_data,
    light_intensity_data,
    distance_from_start_data,
    speed_data,
    swarm_radius_data,
    gradient_map_path,
    world_size_x,
    world_size_y,
    output_folder
):
    """
    Creates a comprehensive 5-panel analysis dashboard.
    
    Args:
        time_data: Array of time values (seconds)
        centroid_x_data: Array of swarm centroid X positions over time
        centroid_y_data: Array of swarm centroid Y positions over time
        light_intensity_data: Array of average light intensity values
        distance_from_start_data: Array of cumulative distance traveled
        speed_data: Array of swarm speed values
        swarm_radius_data: Array of swarm cohesion radius values
        gradient_map_path: Path to gradient map image file
        world_size_x: World width in meters
        world_size_y: World height in meters
        output_folder: Directory to save the dashboard
    
    Returns:
        str: Path to saved dashboard image
    """
    print(f"\n📊 Creating comprehensive analysis dashboard...")
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.5, wspace=0.4)
    
    # ========================================
    # PANEL 1: Trajectory on Gradient Map (LARGE - spans 2 rows, 2 columns)
    # ========================================
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    
    # Load and display gradient map as background
    try:
        img = Image.open(gradient_map_path).convert("RGB")
        ax1.imshow(img, extent=[0, world_size_x, 0, world_size_y], origin='upper', alpha=0.6)
    except:
        print("⚠️  Could not load gradient map for trajectory plot")
    
    # Plot trajectory with time-based color coding
    num_points = len(centroid_x_data)
    colors = plt.cm.coolwarm(np.linspace(0, 1, num_points))  # Blue (start) -> Red (end)
    
    for i in range(1, num_points):
        ax1.plot(centroid_x_data[i-1:i+1], centroid_y_data[i-1:i+1], 
                color=colors[i], linewidth=2, alpha=0.8)
    
    # Mark start and end positions
    ax1.scatter(centroid_x_data[0], centroid_y_data[0], 
               s=200, c='blue', marker='o', edgecolors='white', linewidths=2, 
               label='START', zorder=5)
    ax1.scatter(centroid_x_data[-1], centroid_y_data[-1], 
               s=200, c='red', marker='s', edgecolors='white', linewidths=2, 
               label='END', zorder=5)
    
    ax1.set_xlabel('X Position (m)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Y Position (m)', fontsize=11, fontweight='bold')
    ax1.set_title('Swarm Trajectory on Gradient Map', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, world_size_x)
    ax1.set_ylim(0, world_size_y)
    
    # Add colorbar for time
    sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, 
                               norm=plt.Normalize(vmin=0, vmax=time_data[-1]))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label('Time (s)', fontsize=10, fontweight='bold')
    
    # ========================================
    # PANEL 2: Light Intensity Over Time (Top Right)
    # ========================================
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(time_data, light_intensity_data, linewidth=2.5, color='#2E86AB')
    ax2.fill_between(time_data, light_intensity_data, alpha=0.3, color='#2E86AB')
    ax2.set_xlabel('Time (s)', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Light Intensity', fontsize=10, fontweight='bold')
    ax2.set_title('Average Swarm Light Intensity\n(Lower = Darker)', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, time_data[-1])
    
    # ========================================
    # PANEL 3: Distance from Start (Middle Right)
    # ========================================
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.plot(time_data, distance_from_start_data, linewidth=2.5, color='#A23B72')
    ax3.fill_between(time_data, distance_from_start_data, alpha=0.3, color='#A23B72')
    ax3.set_xlabel('Time (s)', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Distance (m)', fontsize=10, fontweight='bold')
    ax3.set_title('Distance from Start \n(along x-axis)', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, time_data[-1])
    
    # ========================================
    # PANEL 4: Speed Over Time (Bottom Left)
    # ========================================
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(time_data, speed_data, linewidth=2.5, color='#F18F01')
    ax4.fill_between(time_data, speed_data, alpha=0.3, color='#F18F01')
    ax4.set_xlabel('Time (s)', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Speed (m/s)', fontsize=10, fontweight='bold')
    ax4.set_title('Swarm Speed', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, time_data[-1])
    ax4.set_ylim(0, max(speed_data) * 1.1 if max(speed_data) > 0 else 0.1)
    
    # ========================================
    # PANEL 5: Swarm Cohesion (Bottom Center & Right - spans 2 columns)
    # ========================================
    ax5 = fig.add_subplot(gs[2, 1:])
    ax5.plot(time_data, swarm_radius_data, linewidth=2.5, color='#06A77D')
    ax5.fill_between(time_data, swarm_radius_data, alpha=0.3, color='#06A77D')
    ax5.set_xlabel('Time (s)', fontsize=10, fontweight='bold')
    ax5.set_ylabel('Radius (m)', fontsize=10, fontweight='bold')
    ax5.set_title('Swarm Cohesion (Average Radius)', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(0, time_data[-1])
    
    # Overall title
    fig.suptitle('Swarm Gradient Following - Comprehensive Analysis', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Save the dashboard
    os.makedirs(output_folder, exist_ok=True)
    dashboard_path = os.path.join(output_folder, "swarm_analysis_dashboard.png")
    dashboard_path_absolute = os.path.abspath(dashboard_path)
    plt.savefig(dashboard_path_absolute, dpi=150, bbox_inches='tight')
    print(f"📊 Dashboard saved to: {dashboard_path_absolute}")
    
    # Display the dashboard
    plt.show()
    
    return dashboard_path_absolute


def print_simulation_summary(
    simulation_time,
    real_time,
    final_light_intensity,
    final_distance,
    avg_speed,
    avg_radius,
    cohesion_threshold=1.5
):
    """
    Prints a formatted summary of simulation results to terminal.
    
    Args:
        simulation_time: Total simulated time (seconds)
        real_time: Actual wall-clock time (seconds)
        final_light_intensity: Final average light intensity value
        final_distance: Total distance traveled (meters)
        avg_speed: Average speed during simulation (m/s)
        avg_radius: Average swarm radius (meters)
        cohesion_threshold: Threshold for good cohesion (default: 1.5m)
    """
    print(f"\n⏱️  TIMING ANALYSIS:")
    print(f"   Simulation time: {simulation_time:.1f} seconds")
    print(f"   Real-world time: {real_time:.1f} seconds")
    print(f"   Speed factor: {simulation_time/real_time:.2f}x")
    
    cohesion_status = "✅ Maintained" if avg_radius < cohesion_threshold else "⚠️  Loose"
    
    print(f"\n╔════════════════════════════════════════════════════╗")
    print(f"║          SIMULATION SUMMARY                        ║")
    print(f"╠════════════════════════════════════════════════════╣")
    print(f"║  💡 Final Light Intensity: {final_light_intensity:6.1f}              ║")
    print(f"║  📏 Total Distance Traveled: {final_distance:5.2f}m             ║")
    print(f"║  🚀 Average Speed: {avg_speed:5.3f} m/s                  ║")
    print(f"║  🎯 Average Swarm Radius: {avg_radius:5.2f}m              ║")
    print(f"║  🤝 Swarm Cohesion: {cohesion_status:<20}       ║")
    print(f"╚════════════════════════════════════════════════════╝")
