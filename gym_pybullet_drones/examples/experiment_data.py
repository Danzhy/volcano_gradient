"""
Experiment Data Management Module

Handles saving, loading, and managing experimental data for swarm simulations.
Stores metrics, parameters, and metadata for reproducibility and analysis.
"""

import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import hashlib


class ExperimentConfig:
    """Configuration for a single experiment run."""
    
    def __init__(
        self,
        # Swarm parameters
        num_drones: int = 10,
        init_xyzs: Optional[np.ndarray] = None,
        
        # Flocking parameters
        alignment_enabled: bool = True,
        desired_spacing: float = 0.8,
        attraction_strength: float = 0.5,
        repulsion_strength: float = 1.0,
        alignment_strength: float = 0.3,
        light_influence: float = 0.2,
        
        # Environment parameters
        gradient_map_path: str = "",
        world_size_x: float = 6.5,
        world_size_y: float = 4.0,
        
        # Simulation parameters
        duration_sec: int = 240,
        snapshot_interval: int = 10,
        gui: bool = False,
        performance_mode: str = "headless_accurate",
        
        # Success criteria
        finish_line_x: float = 5.5,
        finish_line_enabled: bool = True,
        
        # Metadata
        experiment_name: str = "",
        notes: str = "",
        
        # Random seed management (for reproducibility)
        random_seed: Optional[int] = None,
        base_seed: int = 42,
        run_number: Optional[int] = None
    ):
        self.num_drones = num_drones
        self.init_xyzs = init_xyzs
        self.alignment_enabled = alignment_enabled
        self.desired_spacing = desired_spacing
        self.attraction_strength = attraction_strength
        self.repulsion_strength = repulsion_strength
        self.alignment_strength = alignment_strength
        self.light_influence = light_influence
        self.gradient_map_path = gradient_map_path
        self.world_size_x = world_size_x
        self.world_size_y = world_size_y
        self.duration_sec = duration_sec
        self.snapshot_interval = snapshot_interval
        self.gui = gui
        self.performance_mode = performance_mode
        self.finish_line_x = finish_line_x
        self.finish_line_enabled = finish_line_enabled
        self.experiment_name = experiment_name
        self.notes = notes
        self.random_seed = random_seed
        self.base_seed = base_seed
        self.run_number = run_number
        self.timestamp = datetime.now().isoformat()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary (for JSON serialization)."""
        config_dict = self.__dict__.copy()
        # Convert numpy arrays to lists for JSON serialization
        if self.init_xyzs is not None:
            config_dict['init_xyzs'] = self.init_xyzs.tolist()
        return config_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        """Create config from dictionary."""
        # Convert lists back to numpy arrays
        if 'init_xyzs' in data and data['init_xyzs'] is not None:
            data['init_xyzs'] = np.array(data['init_xyzs'])
        
        config = cls()
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
    
    def get_hash(self) -> str:
        """Generate unique hash for this configuration."""
        # Create string representation of key parameters
        key_params = f"{self.num_drones}_{self.alignment_enabled}_{self.desired_spacing}_" \
                    f"{self.gradient_map_path}_{self.alignment_strength}_{self.light_influence}"
        return hashlib.md5(key_params.encode()).hexdigest()[:8]
    
    def get_experiment_id(self) -> str:
        """Generate unique experiment ID."""
        timestamp = datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
        config_hash = self.get_hash()
        if self.experiment_name:
            return f"{self.experiment_name}_{config_hash}_{timestamp}"
        return f"exp_{config_hash}_{timestamp}"


class ExperimentData:
    """Container for experiment results and metadata."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.experiment_id = config.get_experiment_id()
        
        # Time series data (sampled every snapshot_interval)
        self.time_data: List[float] = []
        self.centroid_x_data: List[float] = []
        self.centroid_y_data: List[float] = []
        self.light_intensity_data: List[float] = []
        self.distance_from_start_data: List[float] = []
        self.speed_data: List[float] = []
        self.swarm_radius_data: List[float] = []
        
        # Position snapshots (full drone positions)
        self.position_snapshots: Dict[float, np.ndarray] = {}
        
        # Success metrics
        self.success: bool = False
        self.time_to_finish: Optional[float] = None
        self.final_centroid_x: float = 0.0
        self.final_light_intensity: float = 0.0
        
        # Simulation metadata
        self.actual_duration: float = 0.0
        self.real_time_factor: float = 0.0
        self.completed: bool = False
        
    def add_datapoint(
        self,
        time: float,
        centroid_x: float,
        centroid_y: float,
        light_intensity: float,
        distance_from_start: float,
        speed: float,
        swarm_radius: float,
        positions: Optional[np.ndarray] = None
    ):
        """Add a single datapoint to the time series."""
        self.time_data.append(time)
        self.centroid_x_data.append(centroid_x)
        self.centroid_y_data.append(centroid_y)
        self.light_intensity_data.append(light_intensity)
        self.distance_from_start_data.append(distance_from_start)
        self.speed_data.append(speed)
        self.swarm_radius_data.append(swarm_radius)
        
        # Store position snapshot if provided
        if positions is not None:
            self.position_snapshots[time] = positions.copy()
    
    def finalize(
        self,
        actual_duration: float,
        real_time_factor: float,
        success: bool = False,
        time_to_finish: Optional[float] = None
    ):
        """Finalize the experiment with completion metadata."""
        self.actual_duration = actual_duration
        self.real_time_factor = real_time_factor
        self.success = success
        self.time_to_finish = time_to_finish
        self.completed = True
        
        # Store final metrics
        if len(self.centroid_x_data) > 0:
            self.final_centroid_x = self.centroid_x_data[-1]
        if len(self.light_intensity_data) > 0:
            self.final_light_intensity = self.light_intensity_data[-1]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'experiment_id': self.experiment_id,
            'config': self.config.to_dict(),
            'time_data': self.time_data,
            'centroid_x_data': self.centroid_x_data,
            'centroid_y_data': self.centroid_y_data,
            'light_intensity_data': self.light_intensity_data,
            'distance_from_start_data': self.distance_from_start_data,
            'speed_data': self.speed_data,
            'swarm_radius_data': self.swarm_radius_data,
            # Position snapshots stored separately (too large for JSON)
            'snapshot_times': list(self.position_snapshots.keys()),
            'success': self.success,
            'time_to_finish': self.time_to_finish,
            'final_centroid_x': self.final_centroid_x,
            'final_light_intensity': self.final_light_intensity,
            'actual_duration': self.actual_duration,
            'real_time_factor': self.real_time_factor,
            'completed': self.completed
        }
    
    def save(self, output_dir: str = "results_data_stored"):
        """
        Save experiment data to disk.
        
        Creates two files in the output_dir:
        1. metadata.json - Configuration and metrics
        2. positions.pkl - Full position snapshots
        
        Note: output_dir is expected to be the experiment-specific folder
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save metadata as JSON (without experiment_id prefix since folder name is the ID)
        metadata_file = output_path / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        # Save position snapshots as pickle (more compact for numpy arrays)
        positions_file = output_path / "positions.pkl"
        with open(positions_file, 'wb') as f:
            pickle.dump(self.position_snapshots, f)
        
        print(f"💾 Experiment data saved:")
        print(f"   Metadata: {metadata_file}")
        print(f"   Positions: {positions_file}")
        
        return str(metadata_file), str(positions_file)
    
    @classmethod
    def load(cls, experiment_id: str, base_dir: str = "results_data_stored") -> 'ExperimentData':
        """
        Load experiment data from disk.
        
        Args:
            experiment_id: The experiment ID (folder name)
            base_dir: Base directory containing experiment folders (default: "results_data_stored")
        """
        # Experiment folder path
        experiment_folder = Path(base_dir) / experiment_id
        
        # Load metadata from experiment folder
        metadata_file = experiment_folder / "metadata.json"
        with open(metadata_file, 'r') as f:
            data_dict = json.load(f)
        
        # Reconstruct ExperimentData
        config = ExperimentConfig.from_dict(data_dict['config'])
        exp_data = cls(config)
        exp_data.experiment_id = data_dict['experiment_id']
        exp_data.time_data = data_dict['time_data']
        exp_data.centroid_x_data = data_dict['centroid_x_data']
        exp_data.centroid_y_data = data_dict['centroid_y_data']
        exp_data.light_intensity_data = data_dict['light_intensity_data']
        exp_data.distance_from_start_data = data_dict['distance_from_start_data']
        exp_data.speed_data = data_dict['speed_data']
        exp_data.swarm_radius_data = data_dict['swarm_radius_data']
        exp_data.success = data_dict['success']
        exp_data.time_to_finish = data_dict['time_to_finish']
        exp_data.final_centroid_x = data_dict['final_centroid_x']
        exp_data.final_light_intensity = data_dict['final_light_intensity']
        exp_data.actual_duration = data_dict['actual_duration']
        exp_data.real_time_factor = data_dict['real_time_factor']
        exp_data.completed = data_dict['completed']
        
        # Load position snapshots from experiment folder
        positions_file = experiment_folder / "positions.pkl"
        if positions_file.exists():
            with open(positions_file, 'rb') as f:
                exp_data.position_snapshots = pickle.load(f)
        
        print(f"📂 Experiment data loaded: {experiment_id}")
        return exp_data
    
    @classmethod
    def list_experiments(cls, base_dir: str = "results_data_stored") -> List[str]:
        """
        List all available experiment IDs (folder names) in the base directory.
        
        Each experiment is stored in its own folder containing:
        - metadata.json
        - positions.pkl
        - position_snapshots/
        - swarm_analysis_dashboard.png
        """
        base_path = Path(base_dir)
        if not base_path.exists():
            return []
        
        # Find all folders that contain a metadata.json file
        experiment_ids = []
        for item in base_path.iterdir():
            if item.is_dir() and (item / "metadata.json").exists():
                experiment_ids.append(item.name)
        
        return sorted(experiment_ids)
    
    def get_summary(self) -> str:
        """Get a formatted summary of the experiment."""
        summary = f"\n{'='*60}\n"
        summary += f"  EXPERIMENT: {self.experiment_id}\n"
        summary += f"{'='*60}\n\n"
        
        summary += f"Configuration:\n"
        summary += f"  Drones: {self.config.num_drones}\n"
        summary += f"  Alignment: {'Enabled' if self.config.alignment_enabled else 'Disabled'}\n"
        summary += f"  Desired Spacing: {self.config.desired_spacing}m\n"
        summary += f"  Gradient Map: {Path(self.config.gradient_map_path).name}\n"
        summary += f"  Duration: {self.config.duration_sec}s\n\n"
        
        summary += f"Results:\n"
        summary += f"  Success: {'✓ YES' if self.success else '✗ NO'}\n"
        if self.time_to_finish:
            summary += f"  Time to Finish: {self.time_to_finish:.1f}s\n"
        summary += f"  Final X Position: {self.final_centroid_x:.2f}m\n"
        summary += f"  Final Light Intensity: {self.final_light_intensity:.1f}\n"
        summary += f"  Real-Time Factor: {self.real_time_factor:.2f}x\n\n"
        
        summary += f"Data Points: {len(self.time_data)} snapshots\n"
        summary += f"Position Snapshots: {len(self.position_snapshots)}\n"
        
        summary += f"\n{'='*60}\n"
        return summary


def check_success(
    centroid_x: float,
    finish_line_x: float = 5.5,
    tolerance: float = 0.1
) -> bool:
    """
    Check if swarm has crossed the finish line.
    
    Args:
        centroid_x: Current X position of swarm centroid
        finish_line_x: X coordinate of finish line
        tolerance: Tolerance for finish line crossing
    
    Returns:
        True if swarm crossed finish line, False otherwise
    """
    return centroid_x >= (finish_line_x - tolerance)


if __name__ == "__main__":
    # Example usage
    print("Experiment Data Management Module")
    print("==================================\n")
    
    # Create example config
    config = ExperimentConfig(
        num_drones=10,
        alignment_enabled=True,
        gradient_map_path="maps_gradient/sine_wave_nice.png",
        experiment_name="test_sine_wave",
        notes="Testing data persistence system"
    )
    
    print(f"Experiment ID: {config.get_experiment_id()}")
    print(f"Config Hash: {config.get_hash()}")
    
    # Create example experiment data
    exp_data = ExperimentData(config)
    
    # Simulate adding data points
    for i in range(5):
        exp_data.add_datapoint(
            time=i * 10.0,
            centroid_x=i * 0.5,
            centroid_y=2.0,
            light_intensity=100 + i * 10,
            distance_from_start=i * 0.5,
            speed=0.1,
            swarm_radius=0.8,
            positions=np.random.rand(10, 3)
        )
    
    # Finalize
    exp_data.finalize(
        actual_duration=50.0,
        real_time_factor=10.5,
        success=True,
        time_to_finish=45.0
    )
    
    print("\n" + exp_data.get_summary())
    
    # List available experiments
    experiments = ExperimentData.list_experiments()
    if experiments:
        print(f"\nAvailable experiments: {len(experiments)}")
        for exp_id in experiments[:5]:  # Show first 5
            print(f"  - {exp_id}")
