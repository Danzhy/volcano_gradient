"""
Regenerate Visualizations from Saved Experiment Data

This script demonstrates how to load saved experiment data and recreate
all visualizations WITHOUT re-running the simulation.

Perfect for:
- Recreating graphs with different styles
- Generating figures for papers/presentations
- Analyzing old experiments
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from experiment_data import ExperimentData
from swarm_visualization import create_drone_position_overlay, create_analysis_dashboard


def regenerate_all_visualizations(experiment_id: str, output_dir: str = "regeneration_files"):
    """
    Load an experiment and regenerate all visualizations.
    
    Args:
        experiment_id: The experiment ID to load (e.g., "exp_4bf7bdf6_10.31.2025_03.00.10")
        output_dir: Directory containing the experiment data
    """
    print(f"\n{'='*60}")
    print(f"REGENERATING VISUALIZATIONS FROM SAVED DATA")
    print(f"{'='*60}\n")
    
    # Load experiment data
    print(f"📂 Loading experiment: {experiment_id}")
    exp_data = ExperimentData.load(experiment_id, output_dir)
    
    # Print summary
    print(exp_data.get_summary())
    
    # Create output subfolder for regenerated files
    regen_folder = Path(output_dir) / f"{experiment_id}_regenerated"
    regen_folder.mkdir(exist_ok=True)
    print(f"📁 Saving regenerated files to: {regen_folder}")
    
    # ============================================
    # 1. REGENERATE COMPREHENSIVE DASHBOARD
    # ============================================
    print(f"\n📊 Regenerating comprehensive dashboard...")
    dashboard_path = create_analysis_dashboard(
        time_data=exp_data.time_data,
        centroid_x_data=exp_data.centroid_x_data,
        centroid_y_data=exp_data.centroid_y_data,
        light_intensity_data=exp_data.light_intensity_data,
        distance_from_start_data=exp_data.distance_from_start_data,
        speed_data=exp_data.speed_data,
        swarm_radius_data=exp_data.swarm_radius_data,
        gradient_map_path=exp_data.config.gradient_map_path,
        world_size_x=exp_data.config.world_size_x,
        world_size_y=exp_data.config.world_size_y,
        output_folder=str(regen_folder)
    )
    print(f"✅ Dashboard saved: {dashboard_path}")
    
    # ============================================
    # 2. REGENERATE POSITION OVERLAYS
    # ============================================
    print(f"\n📸 Regenerating position overlays...")
    
    # Create subfolder for position snapshots
    snapshots_folder = regen_folder / "position_snapshots"
    snapshots_folder.mkdir(exist_ok=True)
    
    # Regenerate overlay for each saved snapshot
    for snapshot_time, positions in exp_data.position_snapshots.items():
        # Extract X and Y positions
        pos_x = positions[:, 0]
        pos_y = positions[:, 1]
        
        timestamp_str = f"{int(snapshot_time)}s"
        
        try:
            create_drone_position_overlay(
                pos_x, pos_y,
                exp_data.config.gradient_map_path,
                exp_data.config.world_size_x,
                exp_data.config.world_size_y,
                str(snapshots_folder),
                timestamp=timestamp_str,
                show_plot=False
            )
            print(f"  ✅ Created overlay at t={timestamp_str}")
        except Exception as e:
            print(f"  ⚠️ Could not create overlay at t={timestamp_str}: {e}")
    
    print(f"✅ All position overlays saved to: {snapshots_folder}")
    
    # ============================================
    # 3. PRINT SUMMARY
    # ============================================
    print(f"\n📋 Simulation Summary:")
    
    avg_speed = np.mean(exp_data.speed_data) if exp_data.speed_data else 0
    avg_radius = np.mean(exp_data.swarm_radius_data) if exp_data.swarm_radius_data else 0
    final_distance = exp_data.distance_from_start_data[-1] if exp_data.distance_from_start_data else 0
    
    print(f"  Simulation Time: {exp_data.config.duration_sec}s")
    print(f"  Real Time: {exp_data.actual_duration:.1f}s ({exp_data.real_time_factor:.2f}x real-time)")
    print(f"  Final Light Intensity: {exp_data.final_light_intensity:.1f}")
    print(f"  Final Distance: {final_distance:.2f}m")
    print(f"  Average Speed: {avg_speed:.3f} m/s")
    print(f"  Average Radius: {avg_radius:.2f}m")
    
    print(f"\n{'='*60}")
    print(f"✅ ALL VISUALIZATIONS REGENERATED!")
    print(f"{'='*60}")
    print(f"\nOutput location: {regen_folder}")
    print(f"\nFiles created:")
    print(f"  - swarm_analysis_dashboard.png")
    print(f"  - position_snapshots/positions_at_*.png ({len(exp_data.position_snapshots)} files)")
    print(f"\n💡 You can now:")
    print(f"  1. Open the dashboard to see all metrics")
    print(f"  2. View position snapshots to see swarm movement")
    print(f"  3. Use this data for papers/presentations")
    print(f"  4. Modify visualization code and regenerate anytime!")
    
    return str(regen_folder)


def main():
    """Interactive script to regenerate visualizations."""
    
    print("="*60)
    print("VISUALIZATION REGENERATION TOOL")
    print("="*60)
    print("\nThis tool loads saved experiment data and recreates all")
    print("visualizations WITHOUT re-running the simulation.\n")
    
    # List available experiments
    output_dir = "regeneration_files"
    experiments = ExperimentData.list_experiments(output_dir)
    
    if not experiments:
        print("❌ No saved experiments found!")
        print(f"   Looking in: {Path(output_dir).absolute()}")
        print("\n💡 Run a simulation first to generate data:")
        print("   python 2d_flocking_with_real_utils.py --duration 40")
        return
    
    print(f"Found {len(experiments)} saved experiments:\n")
    
    # Show recent experiments
    recent_experiments = experiments[-10:]  # Last 10
    for i, exp_id in enumerate(recent_experiments, 1):
        # Try to load and show basic info
        try:
            exp_data = ExperimentData.load(exp_id, output_dir)
            success_icon = "✅" if exp_data.success else "❌"
            gradient_name = Path(exp_data.config.gradient_map_path).stem[:30]
            print(f"{i:2d}. {exp_id}")
            print(f"    {success_icon} {gradient_name} | "
                  f"Drones: {exp_data.config.num_drones} | "
                  f"Final X: {exp_data.final_centroid_x:.2f}m")
        except Exception as e:
            print(f"{i:2d}. {exp_id} (could not load: {e})")
    
    if len(experiments) > 10:
        print(f"\n... and {len(experiments) - 10} more experiments")
    
    print("\n" + "="*60)
    
    # Get user choice
    print("\nOptions:")
    print("1. Enter experiment number (1-{})".format(len(recent_experiments)))
    print("2. Enter full experiment ID")
    print("3. Regenerate ALL recent experiments")
    print("q. Quit")
    
    choice = input("\nYour choice: ").strip()
    
    if choice.lower() == 'q':
        return
    
    if choice == '3':
        print("\n🔄 Regenerating all recent experiments...")
        for exp_id in recent_experiments:
            try:
                print(f"\n{'='*60}")
                regenerate_all_visualizations(exp_id, output_dir)
            except Exception as e:
                print(f"❌ Failed to regenerate {exp_id}: {e}")
        print("\n✅ Batch regeneration complete!")
        return
    
    # Select experiment
    try:
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(recent_experiments):
                exp_id = recent_experiments[idx]
            else:
                print(f"❌ Invalid number. Choose 1-{len(recent_experiments)}")
                return
        else:
            exp_id = choice
    except:
        print("❌ Invalid choice")
        return
    
    # Regenerate visualizations
    try:
        regenerate_all_visualizations(exp_id, output_dir)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Check if experiment ID provided as command-line argument
    if len(sys.argv) > 1:
        exp_id = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "regeneration_files"
        regenerate_all_visualizations(exp_id, output_dir)
    else:
        # Interactive mode
        main()
