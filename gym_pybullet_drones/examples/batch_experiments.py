"""
Batch Experiment Runner for Swarm Gradient Following

Runs multiple experiments sequentially to gather statistics on:
- Success rate (% of runs reaching finish line)
- Time to completion (for successful runs)
- Performance variability

Creates comprehensive summary statistics and visualizations.
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import subprocess

# Add parent directory to path for imports
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

# Import the experiment data module
from experiment_data import ExperimentData


def run_single_experiment(run_number: int, batch_folder: str, duration_sec: int = 240) -> Dict[str, Any]:
    """
    Run a single experiment and return results.
    
    Args:
        run_number: Experiment number in the batch
        batch_folder: Folder to store this run's data
        duration_sec: Maximum simulation duration
        
    Returns:
        Dictionary with run results (success, time_to_finish, etc.)
    """
    print(f"\n{'='*60}")
    print(f"🧪 RUN {run_number}")
    print(f"{'='*60}")
    
    # Import using importlib to handle filename with numbers
    import importlib.util
    import subprocess
    
    # Get path to the simulation script
    script_dir = Path(__file__).parent
    sim_script = script_dir / "2d_flocking_with_real_utils.py"
    
    # Run as subprocess (cleaner than importlib for module with numbers in name)
    start_time = time.time()
    try:
        # Set environment variables to override output folder and skip visualizations
        env = os.environ.copy()
        env['BATCH_OUTPUT_FOLDER'] = batch_folder
        env['BATCH_MODE'] = '1'  # Signal to skip visualization generation
        
        result = subprocess.run(
            [sys.executable, str(sim_script), 
             '--duration', str(duration_sec),
             '--run-number', str(run_number)],  # Pass run number for seed generation
            cwd=str(script_dir.parent.parent),  # Run from project root
            capture_output=True,
            text=True,
            timeout=duration_sec + 60,  # Add buffer for setup/teardown
            env=env
        )
        
        end_time = time.time()
        
        # Print simulation output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        run_success = (result.returncode == 0)
        
    except subprocess.TimeoutExpired:
        print(f"❌ Run {run_number} timed out")
        end_time = time.time()
        run_success = False
    except Exception as e:
        print(f"❌ Run {run_number} failed with error: {e}")
        end_time = time.time()
        run_success = False
    
    # Find the most recent experiment folder in batch_folder
    experiment_folders = sorted([d for d in Path(batch_folder).iterdir() if d.is_dir()])
    if not experiment_folders:
        print(f"⚠️  Warning: No experiment folder found for run {run_number}")
        return {
            "run_number": run_number,
            "success": False,
            "error": "No experiment folder created"
        }
    
    latest_exp_folder = experiment_folders[-1]
    
    # Load experiment data
    try:
        exp_data = ExperimentData.load(latest_exp_folder.name, batch_folder)
        
        result = {
            "run_number": run_number,
            "experiment_id": exp_data.experiment_id,
            "success": exp_data.success,
            "time_to_finish": exp_data.time_to_finish,
            "actual_duration": exp_data.actual_duration,
            "real_time_factor": exp_data.real_time_factor,
            "final_centroid_x": exp_data.final_centroid_x,
            "final_light_intensity": exp_data.final_light_intensity,
            "wall_clock_time": end_time - start_time,
        }
        
        print(f"\n✅ Run {run_number} complete:")
        print(f"   Success: {'YES' if result['success'] else 'NO'}")
        if result['success']:
            print(f"   Time to finish: {result['time_to_finish']:.1f}s")
        print(f"   Wall clock time: {result['wall_clock_time']:.1f}s")
        
        return result
        
    except Exception as e:
        print(f"❌ Error loading results for run {run_number}: {e}")
        return {
            "run_number": run_number,
            "success": False,
            "error": str(e)
        }


def create_batch_summary_visualization(results: List[Dict[str, Any]], output_path: str):
    """
    Create comprehensive visualization of batch results.
    
    Shows:
    - Success rate bar
    - Histogram of completion times
    - Box plot of completion times
    """
    # Filter successful runs
    successful_runs = [r for r in results if r.get('success', False)]
    completion_times = [r['time_to_finish'] for r in successful_runs]
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Batch Experiment Results', fontsize=16, fontweight='bold')
    
    # 1. Success Rate Bar Chart
    ax1 = axes[0]
    success_count = len(successful_runs)
    failure_count = len(results) - success_count
    success_rate = (success_count / len(results)) * 100 if results else 0
    
    bars = ax1.bar(['Success', 'Failure'], [success_count, failure_count], 
                   color=['#2ecc71', '#e74c3c'], alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Number of Runs', fontsize=12)
    ax1.set_title(f'Success Rate: {success_rate:.1f}%\n({success_count}/{len(results)} runs)', 
                  fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')
    
    # 2. Histogram of Completion Times
    ax2 = axes[1]
    if completion_times:
        ax2.hist(completion_times, bins=10, color='#3498db', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Time to Finish (seconds)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Distribution of Completion Times', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # Add mean line
        mean_time = np.mean(completion_times)
        ax2.axvline(mean_time, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_time:.1f}s')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'No successful runs', ha='center', va='center', fontsize=14)
        ax2.set_title('Distribution of Completion Times', fontsize=12, fontweight='bold')
    
    # 3. Box Plot
    ax3 = axes[2]
    if completion_times:
        box = ax3.boxplot(completion_times, vert=True, patch_artist=True)
        box['boxes'][0].set_facecolor('#9b59b6')
        box['boxes'][0].set_alpha(0.7)
        
        ax3.set_ylabel('Time to Finish (seconds)', fontsize=12)
        ax3.set_title('Completion Time Statistics', fontsize=12, fontweight='bold')
        ax3.set_xticklabels(['All Runs'])
        ax3.grid(axis='y', alpha=0.3)
        
        # Add statistics text
        stats_text = (
            f"Min: {np.min(completion_times):.1f}s\n"
            f"Q1: {np.percentile(completion_times, 25):.1f}s\n"
            f"Median: {np.median(completion_times):.1f}s\n"
            f"Q3: {np.percentile(completion_times, 75):.1f}s\n"
            f"Max: {np.max(completion_times):.1f}s"
        )
        ax3.text(1.3, np.median(completion_times), stats_text, 
                fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax3.text(0.5, 0.5, 'No successful runs', ha='center', va='center', fontsize=14)
        ax3.set_title('Completion Time Statistics', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"📊 Batch summary visualization saved to: {output_path}")
    plt.close()


def run_batch_experiments(
    num_runs: int = 50,
    duration_sec: int = 240,
    base_output_folder: str = "results_batch"
):
    """
    Run multiple experiments sequentially and gather statistics.
    
    Args:
        num_runs: Number of experiments to run
        duration_sec: Maximum duration for each experiment
        base_output_folder: Base folder for batch results
    """
    print("="*70)
    print("🚀 BATCH EXPERIMENT RUNNER")
    print("="*70)
    print(f"Configuration:")
    print(f"  - Number of runs: {num_runs}")
    print(f"  - Max duration per run: {duration_sec}s")
    print(f"  - Mode: Sequential")
    print("="*70)
    
    # Create batch folder with timestamp
    timestamp = datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
    batch_id = f"batch_{timestamp}"
    batch_folder = os.path.join(base_output_folder, batch_id)
    os.makedirs(batch_folder, exist_ok=True)
    
    print(f"\n📁 Batch folder: {batch_folder}")
    print(f"⏱️  Estimated time: {num_runs * 6}-{num_runs * 20} seconds ({num_runs * 0.1:.1f}-{num_runs * 0.33:.1f} minutes)")
    print(f"\nStarting experiments...\n")
    
    # Run all experiments
    batch_start_time = time.time()
    results = []
    
    for run_num in range(1, num_runs + 1):
        result = run_single_experiment(run_num, batch_folder, duration_sec)
        results.append(result)
        
        # Print progress
        elapsed = time.time() - batch_start_time
        avg_time_per_run = elapsed / run_num
        remaining_runs = num_runs - run_num
        estimated_remaining = avg_time_per_run * remaining_runs
        
        print(f"\n📊 Progress: {run_num}/{num_runs} complete")
        print(f"   Elapsed: {elapsed:.1f}s | Est. remaining: {estimated_remaining:.1f}s")
    
    batch_end_time = time.time()
    total_time = batch_end_time - batch_start_time
    
    # Calculate statistics
    successful_runs = [r for r in results if r.get('success', False)]
    success_count = len(successful_runs)
    success_rate = (success_count / num_runs) * 100
    
    completion_times = [r['time_to_finish'] for r in successful_runs]
    
    # Extract configuration from first experiment's metadata
    configuration = {}
    if results:
        first_exp_id = results[0].get('experiment_id', '')
        if first_exp_id:
            metadata_path = os.path.join(batch_folder, first_exp_id, 'metadata.json')
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    config = metadata.get('config', {})
                    
                    # Extract key configuration parameters
                    configuration = {
                        "num_drones": config.get('num_drones'),
                        "alignment_enabled": config.get('alignment_enabled'),
                        "desired_spacing": config.get('desired_spacing'),
                        "attraction_strength": config.get('attraction_strength'),
                        "repulsion_strength": config.get('repulsion_strength'),
                        "alignment_strength": config.get('alignment_strength'),
                        "light_influence": config.get('light_influence'),
                        "gradient_map": os.path.basename(config.get('gradient_map_path', '')),
                        "world_size_x": config.get('world_size_x'),
                        "world_size_y": config.get('world_size_y'),
                        "finish_line_x": config.get('finish_line_x'),
                        "finish_line_enabled": config.get('finish_line_enabled'),
                        "max_duration": config.get('duration_sec'),
                        "performance_mode": config.get('performance_mode'),
                        "base_seed": config.get('base_seed'),
                    }
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"⚠️  Warning: Could not extract configuration from metadata: {e}")
    
    summary = {
        "batch_id": batch_id,
        "timestamp": timestamp,
        "configuration": configuration,
        "num_runs": num_runs,
        "success_count": success_count,
        "failure_count": num_runs - success_count,
        "success_rate_percent": success_rate,
        "total_wall_clock_time": total_time,
        "avg_time_per_run": total_time / num_runs,
    }
    
    if completion_times:
        summary.update({
            "completion_time_mean": float(np.mean(completion_times)),
            "completion_time_std": float(np.std(completion_times)),
            "completion_time_min": float(np.min(completion_times)),
            "completion_time_max": float(np.max(completion_times)),
            "completion_time_median": float(np.median(completion_times)),
            "completion_time_q1": float(np.percentile(completion_times, 25)),
            "completion_time_q3": float(np.percentile(completion_times, 75)),
        })
    
    summary["results"] = results
    
    # Save summary JSON
    summary_path = os.path.join(batch_folder, "batch_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*70}")
    print("✅ BATCH COMPLETE!")
    print(f"{'='*70}")
    
    # Print configuration if available
    if configuration:
        print(f"\n⚙️  CONFIGURATION:")
        print(f"   Swarm size: {configuration.get('num_drones', 'N/A')} drones")
        print(f"   World size: {configuration.get('world_size_x', 'N/A')}m × {configuration.get('world_size_y', 'N/A')}m")
        print(f"   Finish line: {configuration.get('finish_line_x', 'N/A'):.2f}m")
        print(f"   Alignment: {'Enabled' if configuration.get('alignment_enabled') else 'Disabled'}")
        print(f"   Spacing: {configuration.get('desired_spacing', 'N/A')}m")
        print(f"   Gradient map: {configuration.get('gradient_map', 'N/A')}")
        print(f"   Base seed: {configuration.get('base_seed', 'N/A')}")
    
    print(f"\n📊 SUMMARY STATISTICS:")
    print(f"   Total runs: {num_runs}")
    print(f"   Successful: {success_count} ({success_rate:.1f}%)")
    print(f"   Failed: {num_runs - success_count}")
    print(f"   Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"   Avg time per run: {total_time/num_runs:.1f}s")
    
    if completion_times:
        print(f"\n⏱️  COMPLETION TIME STATISTICS (successful runs only):")
        print(f"   Mean: {summary['completion_time_mean']:.1f}s")
        print(f"   Std Dev: {summary['completion_time_std']:.1f}s")
        print(f"   Min: {summary['completion_time_min']:.1f}s")
        print(f"   Median: {summary['completion_time_median']:.1f}s")
        print(f"   Max: {summary['completion_time_max']:.1f}s")
    
    print(f"\n💾 Results saved to:")
    print(f"   - Summary: {summary_path}")
    
    # Create visualization
    viz_path = os.path.join(batch_folder, "batch_summary_visualization.png")
    create_batch_summary_visualization(results, viz_path)
    print(f"   - Visualization: {viz_path}")
    
    print(f"\n📁 All experiment data in: {batch_folder}")
    print(f"{'='*70}\n")
    
    return summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run batch experiments with reproducible random seeds',
        epilog='Each run will use seed = base_seed + run_number for reproducibility.\n'
               'Example: --runs 5 --base-seed 42 will use seeds 43, 44, 45, 46, 47'
    )
    parser.add_argument('--runs', type=int, default=50,
                       help='Number of experiments to run (default: 50)')
    parser.add_argument('--duration', type=int, default=240,
                       help='Maximum duration per experiment in seconds (default: 240)')
    parser.add_argument('--output', type=str, default='results_batch',
                       help='Base output folder (default: results_batch)')
    parser.add_argument('--base-seed', type=int, default=42,
                       help='Base random seed for batch (default: 42). Run N uses seed base_seed+N')
    
    args = parser.parse_args()
    
    print(f"🎲 Random Seed Configuration:")
    print(f"   Base seed: {args.base_seed}")
    print(f"   Run seeds: {args.base_seed + 1} to {args.base_seed + args.runs}")
    print(f"   All experiments are reproducible!\n")
    
    run_batch_experiments(
        num_runs=args.runs,
        duration_sec=args.duration,
        base_output_folder=args.output
    )
