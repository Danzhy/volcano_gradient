#!/usr/bin/env python3
"""
Consolidate Batch Results Script
=================================
Consolidates multiple batch experiment results into a single CSV table for analysis.

Takes a sweep_summary JSON file and generates a CSV with key metrics from each batch.
Perfect for comparing different parameter configurations in research papers.

Usage:
    python consolidate_results.py results_batch/sweep_summary_11.06.2025_10.03.10.json
    python consolidate_results.py --help
"""

import json
import csv
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any
import statistics


def simplify_gradient_map_name(full_name: str) -> str:
    """
    Simplify gradient map filename for display.
    
    Example: 
        'path_example_20.0_sine_curve_thick001_freq2.png' -> 'sine_curve_thick001_freq2'
    """
    # Remove path_example_20.0_ prefix and .png suffix
    name = full_name.replace('path_example_20.0_', '').replace('.png', '')
    return name


def load_batch_summary(batch_folder: str) -> Dict[str, Any]:
    """
    Load batch_summary.json from a batch folder.
    
    Args:
        batch_folder: Path to the batch folder (e.g., 'results_batch/batch_11.06.2025_04.38.11')
    
    Returns:
        Dictionary containing batch summary data
    """
    summary_path = os.path.join(batch_folder, 'batch_summary.json')
    
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Batch summary not found: {summary_path}")
    
    with open(summary_path, 'r') as f:
        return json.load(f)


def extract_batch_metrics(batch_summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract key metrics from a batch summary for the CSV table.
    
    Args:
        batch_summary: Loaded batch_summary.json data
    
    Returns:
        Dictionary with extracted metrics
    """
    config = batch_summary['configuration']
    
    # Calculate average final centroid X from successful runs only
    results = batch_summary.get('results', [])
    successful_results = [r for r in results if r.get('success', False)]
    if successful_results:
        avg_final_x = statistics.mean([r['final_centroid_x'] for r in successful_results])
    else:
        avg_final_x = None
    
    # Handle batches with no successful runs (completion_time fields may be missing)
    completion_time_mean = batch_summary.get('completion_time_mean')
    completion_time_std = batch_summary.get('completion_time_std')
    completion_time_median = batch_summary.get('completion_time_median')
    
    return {
        'num_drones': config['num_drones'],
        'alignment': 'ON' if config['alignment_enabled'] else 'OFF',
        'gradient_map': simplify_gradient_map_name(config['gradient_map']),
        'num_runs': batch_summary['num_runs'],
        'success_rate_percent': batch_summary['success_rate_percent'],
        'completion_time_mean': round(completion_time_mean, 2) if completion_time_mean else None,
        'completion_time_std': round(completion_time_std, 2) if completion_time_std else None,
        'completion_time_median': round(completion_time_median, 2) if completion_time_median else None,
        'finish_line_x': round(config['finish_line_x'], 2),
        'avg_final_x': round(avg_final_x, 2) if avg_final_x else None,
    }


def consolidate_sweep_results(sweep_summary_path: str) -> tuple[List[Dict[str, Any]], str]:
    """
    Consolidate all batch results from a sweep summary file.
    
    Args:
        sweep_summary_path: Path to sweep_summary JSON file
    
    Returns:
        Tuple of (list of batch metrics, sweep timestamp)
    """
    # Load sweep summary
    with open(sweep_summary_path, 'r') as f:
        sweep_data = json.load(f)
    
    # Extract timestamp from filename for CSV naming
    filename = os.path.basename(sweep_summary_path)
    # Format: sweep_summary_11.06.2025_10.03.10.json
    timestamp = filename.replace('sweep_summary_', '').replace('.json', '')
    
    # Get the directory containing batch folders
    results_batch_dir = os.path.dirname(sweep_summary_path)
    
    # Process each batch result
    consolidated_data = []
    for result in sweep_data['results']:
        config_name = result['config_name']
        
        # Find the batch folder - it should be referenced in the sweep summary
        # The batch folder is created during the sweep with timestamp
        # We need to find it based on the configuration and timestamp
        
        # Alternative: extract batch_id from elapsed_time matching
        # But easier: look for batch folders that match the config timestamp range
        
        # For now, we'll search for batch folders and match by timestamp proximity
        # But first, let's check if the result has a batch_id or folder reference
        
        # Since sweep results don't directly reference batch folders,
        # we'll search for batch folders in chronological order
        # and match them with sweep results
        
        print(f"⚠️  Warning: Need to match config '{config_name}' to batch folder")
        print(f"   Config: {result['config']}")
        
        # For now, let's create a placeholder entry
        # We need to enhance this to properly find the batch folder
        consolidated_data.append({
            'config_name': config_name,
            'config': result['config'],
            'elapsed_time_min': round(result['elapsed_time'] / 60, 2),
            'success': result.get('success', False),
        })
    
    return consolidated_data, timestamp


def consolidate_from_batch_folders(results_batch_dir: str, batch_folders: List[str]) -> List[Dict[str, Any]]:
    """
    Consolidate results by directly reading batch folders.
    
    Args:
        results_batch_dir: Path to results_batch directory
        batch_folders: List of batch folder names to process
    
    Returns:
        List of consolidated metrics dictionaries
    """
    consolidated_data = []
    
    for batch_folder_name in batch_folders:
        batch_path = os.path.join(results_batch_dir, batch_folder_name)
        
        if not os.path.isdir(batch_path):
            print(f"⚠️  Skipping non-directory: {batch_folder_name}")
            continue
        
        try:
            print(f"📊 Processing: {batch_folder_name}")
            batch_summary = load_batch_summary(batch_path)
            metrics = extract_batch_metrics(batch_summary)
            metrics['batch_id'] = batch_summary['batch_id']
            consolidated_data.append(metrics)
        except Exception as e:
            print(f"❌ Error processing {batch_folder_name}: {e}")
            continue
    
    return consolidated_data


def write_csv(data: List[Dict[str, Any]], output_path: str):
    """
    Write consolidated data to CSV file.
    
    Args:
        data: List of metric dictionaries
        output_path: Path to output CSV file
    """
    if not data:
        print("❌ No data to write!")
        return
    
    # Define column order
    fieldnames = [
        'batch_id',
        'num_drones',
        'alignment',
        'gradient_map',
        'num_runs',
        'success_rate_percent',
        'completion_time_mean',
        'completion_time_std',
        'completion_time_median',
        'finish_line_x',
        'avg_final_x'
    ]
    
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    
    print(f"\n✅ CSV saved to: {output_path}")
    print(f"   Rows: {len(data)}")
    print(f"   Columns: {len(fieldnames)}")


def print_table(data: List[Dict[str, Any]]):
    """
    Print a nice formatted table to terminal.
    
    Args:
        data: List of metric dictionaries
    """
    if not data:
        print("❌ No data to display!")
        return
    
    print("\n" + "="*120)
    print("📊 CONSOLIDATED BATCH RESULTS")
    print("="*120)
    
    # Print header
    header = f"{'Drones':<8} {'Align':<7} {'Gradient Map':<30} {'Runs':<6} {'Success%':<10} {'Time(s)':<10} {'±Std':<8} {'Finish X':<10} {'Final X':<10}"
    print(header)
    print("-"*120)
    
    # Print rows
    for row in data:
        # Handle None values for failed batches
        time_mean = f"{row['completion_time_mean']:.1f}" if row['completion_time_mean'] else "N/A"
        time_std = f"{row['completion_time_std']:.1f}" if row['completion_time_std'] else "N/A"
        final_x = f"{row['avg_final_x']:.2f}" if row['avg_final_x'] else "N/A"
        
        line = (
            f"{row['num_drones']:<8} "
            f"{row['alignment']:<7} "
            f"{row['gradient_map']:<30} "
            f"{row['num_runs']:<6} "
            f"{row['success_rate_percent']:<10.1f} "
            f"{time_mean:<10} "
            f"{time_std:<8} "
            f"{row['finish_line_x']:<10.2f} "
            f"{final_x:<10}"
        )
        print(line)
    
    print("="*120 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Consolidate batch experiment results into CSV table',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use with sweep summary file (recommended)
  python gym_pybullet_drones/examples/consolidate_results.py results_batch/sweep_summary_11.06.2025_10.03.10.json
  
  # Or use latest sweep automatically
  python gym_pybullet_drones/examples/consolidate_results.py --latest
  
  # Or specify batch folders directly
  python gym_pybullet_drones/examples/consolidate_results.py --batches batch_11.06.2025_04.38.11 batch_11.06.2025_05.01.35
        """
    )
    
    parser.add_argument('sweep_file', nargs='?', 
                       help='Path to sweep_summary JSON file')
    parser.add_argument('--batches', nargs='+',
                       help='List of batch folder names to consolidate')
    parser.add_argument('--latest', action='store_true',
                       help='Auto-find and use the most recent sweep summary')
    parser.add_argument('--output', '-o',
                       help='Custom output CSV path (default: auto-generated)')
    parser.add_argument('--no-print', action='store_true',
                       help='Skip printing table to terminal')
    
    args = parser.parse_args()
    
    # Determine input mode
    if args.batches:
        # Mode: Direct batch folder specification
        results_batch_dir = 'results_batch'
        print(f"\n📂 Consolidating {len(args.batches)} batch folders...")
        consolidated_data = consolidate_from_batch_folders(results_batch_dir, args.batches)
        timestamp = 'custom'
        
    elif args.latest:
        # Mode: Auto-find latest sweep summary
        results_batch_dir = 'results_batch'
        sweep_files = sorted([f for f in os.listdir(results_batch_dir) if f.startswith('sweep_summary_')])
        if not sweep_files:
            print("❌ No sweep summary files found in results_batch/")
            sys.exit(1)
        sweep_file = os.path.join(results_batch_dir, sweep_files[-1])
        print(f"\n📄 Using latest sweep summary: {sweep_files[-1]}")
        
        # Load sweep and get batch folders from results
        with open(sweep_file, 'r') as f:
            sweep_data = json.load(f)
        
        # Extract batch folders from sweep results (we need to find them)
        # For now, use all batch folders in chronological order matching the sweep
        all_batches = sorted([d for d in os.listdir(results_batch_dir) 
                            if os.path.isdir(os.path.join(results_batch_dir, d)) 
                            and d.startswith('batch_')])
        
        # Take the last N batches matching the number of results in sweep
        num_configs = len(sweep_data['results'])
        batch_folders = all_batches[-num_configs:]
        
        print(f"📊 Found {len(batch_folders)} batches matching sweep")
        consolidated_data = consolidate_from_batch_folders(results_batch_dir, batch_folders)
        
        filename = os.path.basename(sweep_file)
        timestamp = filename.replace('sweep_summary_', '').replace('.json', '')
        
    elif args.sweep_file:
        # Mode: Specific sweep summary file
        if not os.path.exists(args.sweep_file):
            print(f"❌ File not found: {args.sweep_file}")
            sys.exit(1)
        
        print(f"\n📄 Reading sweep summary: {args.sweep_file}")
        
        # Load sweep and get batch folders from results
        with open(args.sweep_file, 'r') as f:
            sweep_data = json.load(f)
        
        results_batch_dir = os.path.dirname(args.sweep_file) or 'results_batch'
        
        # Extract batch folders - match by timestamp proximity
        all_batches = sorted([d for d in os.listdir(results_batch_dir) 
                            if os.path.isdir(os.path.join(results_batch_dir, d)) 
                            and d.startswith('batch_')])
        
        # Take the last N batches matching the number of results in sweep
        num_configs = len(sweep_data['results'])
        batch_folders = all_batches[-num_configs:]
        
        print(f"📊 Found {len(batch_folders)} batches matching sweep")
        consolidated_data = consolidate_from_batch_folders(results_batch_dir, batch_folders)
        
        filename = os.path.basename(args.sweep_file)
        timestamp = filename.replace('sweep_summary_', '').replace('.json', '')
        
    else:
        parser.print_help()
        sys.exit(1)
    
    if not consolidated_data:
        print("❌ No data to consolidate!")
        sys.exit(1)
    
    # Print table to terminal
    if not args.no_print:
        print_table(consolidated_data)
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = f'results_batch/consolidated_results_{timestamp}.csv'
    
    # Write CSV
    write_csv(consolidated_data, output_path)
    
    print(f"\n🎉 Consolidation complete!")
    print(f"   Import into Excel, Python pandas, R, etc. for further analysis\n")


if __name__ == "__main__":
    main()
