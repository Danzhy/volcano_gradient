#!/usr/bin/env python3
"""
Flexible Parameter Sweep Script
================================
Data-driven parameter exploration system.

Define parameter spaces and this script automatically generates all combinations
and runs batch experiments for each configuration.

Usage:
    python run_parameter_sweep.py
    python run_parameter_sweep.py --quick  # Run with fewer iterations for testing
"""

import subprocess
import time
import os
from datetime import datetime
import json
import itertools
from typing import Dict, List, Any, Optional


# ============================================================================
# PARAMETER SPACE CONFIGURATION
# ============================================================================
# Define which parameters to sweep and their values
# Add/remove parameters here to customize your sweep!

PARAMETER_SPACE = {
    # Primary parameters (high impact)
    'num_drones': [5, 7, 10, 19],                    # Swarm size
    'alignment': [True, False],                # Alignment on/off
    'gradient_map': ['sine_curve_thick1_freq2', 
                     'sine_curve_thick01_freq2',
                     'sine_curve_thick01_freq4',
                     'sine_curve_thick000001_freq4'
                     ],  # Path type
                    #  'sine_curve_thick000001_freq2',
                    #  'sine_curve_thick000001_freq8'                     
    
    # Secondary parameters (uncomment to explore)
    # 'max_velocity': [0.10, 0.15, 0.20],      # Max linear velocity
    # 'alignment_weight': [0.5, 1.0, 1.5],     # Beta parameter
    # 'gradient_map': ['sine', 'funnel'],      # Path type
}
PARAMETER_SPACE = {
    # Primary parameters (high impact)
    'num_drones': [7, 10, 19, 37],                  # Swarm size
    'alignment': [True],                # Alignment on/off
    'gradient_map': ['sine_curve_thick01_freq2'
                    #  'sine_curve_thick1_freq2',
                    #  'sine_curve_thick01_freq4',
    #                  'sine_curve_thick000001_freq4'
    #                  ],  # Path type
                    #  'sine_curve_thick000001_freq2',
                    #  'sine_curve_thick000001_freq8'                     
                    ]
    
    # Secondary parameters (uncomment to explore)
    # 'max_velocity': [0.10, 0.15, 0.20],      # Max linear velocity
    # 'alignment_weight': [0.5, 1.0, 1.5],     # Beta parameter
    # 'gradient_map': ['sine', 'funnel'],      # Path type
}



# Experiment configuration
NUM_RUNS_PER_CONFIG = 50      # Number of repetitions per configuration
DURATION_SEC = 240            # Max duration per run (seconds)
BASE_SEED = 42                # Base random seed for reproducibility

# Quick mode (for testing)
QUICK_MODE_RUNS = 3           # Reduced runs for quick testing


# ============================================================================
# BATCH RUNNER
# ============================================================================

def run_batch(config: Dict[str, Any], num_runs: int, duration: int) -> Dict[str, Any]:
    """
    Run a single batch experiment with specified configuration.
    
    Args:
        config: Dictionary of parameter name -> value
        num_runs: Number of runs in the batch
        duration: Max duration per run (seconds)
    
    Returns:
        Dictionary with batch results
    """
    # Generate human-readable config name
    config_parts = []
    for key, value in config.items():
        if isinstance(value, bool):
            config_parts.append(f"{key}={'on' if value else 'off'}")
        else:
            config_parts.append(f"{key}={value}")
    config_name = "_".join(config_parts)
    
    print(f"\n{'='*70}")
    print(f"🚀 STARTING BATCH: {config_name}")
    print(f"{'='*70}")
    print(f"Configuration:")
    print(f"  - Runs: {num_runs}")
    print(f"  - Duration: {duration}s")
    for key, value in config.items():
        print(f"  - {key}: {value}")
    print(f"  - Started: {datetime.now().strftime('%I:%M:%S %p')}")
    print(f"{'='*70}\n")
    
    # Build command
    cmd = [
        "python",
        "gym_pybullet_drones/examples/batch_experiments.py",
        "--runs", str(num_runs),
        "--duration", str(duration)
    ]
    
    # Map parameter names to command-line arguments
    param_mapping = {
        'num_drones': '--num-drones',
        'alignment': '--alignment',
        'max_velocity': '--max-velocity',
        'alignment_weight': '--alignment-weight',
        'gradient_map': '--gradient-map',
    }
    
    # Add parameters to command
    for param_name, param_value in config.items():
        if param_name in param_mapping:
            arg_name = param_mapping[param_name]
            
            # Handle boolean parameters
            if isinstance(param_value, bool):
                cmd.extend([arg_name, 'true' if param_value else 'false'])
            else:
                cmd.extend([arg_name, str(param_value)])
    
    # Run the batch
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # Show output in real-time
            text=True
        )
        elapsed_time = time.time() - start_time
        
        print(f"\n{'='*70}")
        print(f"✅ BATCH COMPLETE: {config_name}")
        print(f"⏱️  Total time: {elapsed_time/60:.1f} minutes ({elapsed_time:.0f}s)")
        print(f"   Finished: {datetime.now().strftime('%I:%M:%S %p')}")
        print(f"{'='*70}\n")
        
        return {
            "config_name": config_name,
            "config": config,
            "success": True,
            "elapsed_time": elapsed_time,
            "num_runs": num_runs,
            "duration": duration
        }
        
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"❌ BATCH FAILED: {config_name}")
        print(f"   Error: {e}")
        print(f"   Time before failure: {elapsed_time/60:.1f} minutes")
        print(f"{'='*70}\n")
        
        return {
            "config_name": config_name,
            "config": config,
            "success": False,
            "elapsed_time": elapsed_time,
            "error": str(e)
        }


# ============================================================================
# PARAMETER SPACE GENERATOR
# ============================================================================

def generate_configurations(param_space: Dict[str, List]) -> List[Dict[str, Any]]:
    """
    Generate all combinations of parameters from the parameter space.
    
    Args:
        param_space: Dictionary mapping parameter names to list of values
    
    Returns:
        List of configuration dictionaries
    """
    # Get parameter names and their value lists
    param_names = list(param_space.keys())
    param_values = list(param_space.values())
    
    # Generate all combinations using itertools.product
    configurations = []
    for combo in itertools.product(*param_values):
        config = dict(zip(param_names, combo))
        configurations.append(config)
    
    return configurations


# ============================================================================
# MAIN SWEEP RUNNER
# ============================================================================

def main():
    """Run the parameter sweep experiments."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Flexible Parameter Sweep for Drone Swarm Experiments'
    )
    parser.add_argument('--quick', action='store_true',
                       help=f'Quick mode: run only {QUICK_MODE_RUNS} iterations per config')
    args = parser.parse_args()
    
    # Determine number of runs
    num_runs = QUICK_MODE_RUNS if args.quick else NUM_RUNS_PER_CONFIG
    mode_str = "QUICK TEST" if args.quick else "FULL SWEEP"
    
    # Generate all configurations
    configurations = generate_configurations(PARAMETER_SPACE)
    
    print("\n" + "="*70)
    print(f"🔬 PARAMETER SWEEP - {mode_str}")
    print("="*70)
    print(f"\nParameter Space:")
    for param, values in PARAMETER_SPACE.items():
        print(f"  - {param}: {values}")
    print(f"\nTotal configurations: {len(configurations)}")
    print(f"Runs per configuration: {num_runs}")
    print(f"Total experiments: {len(configurations) * num_runs}")
    print(f"Duration per run: {DURATION_SEC}s")
    
    # Estimate time
    avg_time_per_run = 30  # seconds (rough estimate)
    total_estimated_seconds = len(configurations) * num_runs * avg_time_per_run
    print(f"\n⏱️  Estimated total time: {total_estimated_seconds/3600:.1f} hours")
    print(f"Started at: {datetime.now().strftime('%I:%M:%S %p')}")
    print("="*70 + "\n")
    
    # Confirm before starting (unless quick mode)
    if not args.quick:
        # response = input("🤔 Ready to start? This will take a while. (yes/no): ")
        response = 'y'
        if response.lower() not in ['yes', 'y']:
            print("❌ Sweep cancelled.")
            return
    
    # Run all configurations
    sweep_start_time = time.time()
    results = []
    
    for i, config in enumerate(configurations, 1):
        print(f"\n{'#'*70}")
        print(f"# Configuration {i}/{len(configurations)}")
        print(f"{'#'*70}")
        
        result = run_batch(config, num_runs, DURATION_SEC)
        results.append(result)
        
        # Progress update
        elapsed = time.time() - sweep_start_time
        avg_time_per_config = elapsed / i
        remaining_configs = len(configurations) - i
        estimated_remaining = avg_time_per_config * remaining_configs
        
        print(f"\n📊 Progress: {i}/{len(configurations)} configurations complete")
        print(f"   Elapsed: {elapsed/60:.1f} minutes")
        print(f"   Est. remaining: {estimated_remaining/60:.1f} minutes")
        
        # Short pause between batches (except after last one)
        if i < len(configurations):
            time.sleep(5)
    
    # Final summary
    sweep_end_time = time.time()
    total_time = sweep_end_time - sweep_start_time
    
    print("\n" + "="*70)
    print("🎉 PARAMETER SWEEP COMPLETE!")
    print("="*70)
    print(f"\n⏱️  Total Time: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)")
    print(f"   Started: {datetime.fromtimestamp(sweep_start_time).strftime('%I:%M:%S %p')}")
    print(f"   Ended: {datetime.fromtimestamp(sweep_end_time).strftime('%I:%M:%S %p')}")
    
    print(f"\n📊 Batch Results:")
    successful = sum(1 for r in results if r.get('success', False))
    for i, result in enumerate(results, 1):
        status = "✅" if result.get('success', False) else "❌"
        print(f"   {status} Config {i}: {result['config_name']}")
        print(f"      Time: {result['elapsed_time']/60:.1f} minutes")
        if not result.get('success', False):
            print(f"      Error: {result.get('error', 'Unknown')}")
    
    print(f"\n   Success Rate: {successful}/{len(results)} configurations")
    print(f"\n💾 All results saved to: results_batch/")
    print("="*70 + "\n")
    
    # Save sweep summary
    sweep_summary = {
        "mode": mode_str,
        "parameter_space": PARAMETER_SPACE,
        "num_configurations": len(configurations),
        "runs_per_config": num_runs,
        "sweep_start_time": datetime.fromtimestamp(sweep_start_time).isoformat(),
        "sweep_end_time": datetime.fromtimestamp(sweep_end_time).isoformat(),
        "total_time_seconds": total_time,
        "total_time_hours": total_time / 3600,
        "successful_batches": successful,
        "results": results
    }
    
    summary_path = f"results_batch/sweep_summary_{datetime.now().strftime('%m.%d.%Y_%H.%M.%S')}.json"
    os.makedirs("results_batch", exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(sweep_summary, f, indent=2)
    
    print(f"📄 Sweep summary saved to: {summary_path}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Sweep interrupted by user!")
        print("Partial results are saved in results_batch/\n")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        print("Check results_batch/ for any completed batches\n")
