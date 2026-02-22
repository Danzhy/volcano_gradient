"""
Pursuit-Evasion (Catch-Run) Vision Utilities

Provides distance-scaling functions for pursuit-evasion perception in flocking simulations.
Creates a "catch-run" interaction between agents:
- Agents AHEAD of you appear MUCH FARTHER (reduced influence → you want to catch up)
- Agents to the SIDES have neutral perceived distance
- Agents BEHIND you appear VERY CLOSE (strong influence → you feel pressure to run away)

This is achieved by scaling the perceived distance based on relative angle,
which modulates the Lennard-Jones force strength accordingly.
"""
import numpy as np


def compute_distance_scale(theta, method="cosine", **kwargs):
    """
    Compute distance scale factor for a neighbor at relative angle theta.
    The actual distance is multiplied by this factor for force calculations.
    
    Args:
        theta: Relative angle (radians) in [-π, π]
               0 = ahead, π/2 = left, -π/2 = right, ±π = behind
        method: Scaling method ("cosine", "linear", "gaussian")
        **kwargs: Method-specific parameters:
            - max_scale: Scale when neighbor is ahead (default: 3.0) - appear farther
            - min_scale: Scale when neighbor is behind (default: 0.3) - appear closer
            - sigma: Transition sharpness for gaussian (default: π/3)
    
    Returns:
        scale: Float > 0. Scale > 1 = appear farther, scale < 1 = appear closer
    """
    max_scale = kwargs.get("max_scale", 3.0)
    min_scale = kwargs.get("min_scale", 0.3)
    
    if method == "cosine":
        # Smooth interpolation: theta=0 (ahead) → max_scale, theta=±π (behind) → min_scale
        # cos(0)=1, cos(π)=-1 → map [1,-1] to [max_scale, min_scale]
        cos_theta = np.cos(theta)
        # (1 + cos_theta) / 2 goes from 1 (ahead) to 0 (behind)
        t = (1.0 + cos_theta) / 2.0
        scale = min_scale + t * (max_scale - min_scale)
        return float(np.clip(scale, 1e-6, 1e6))  # Avoid division by zero
    
    elif method == "linear":
        # Linear interpolation based on angle magnitude from π (behind)
        # |theta| = 0 → ahead → max_scale
        # |theta| = π → behind → min_scale
        abs_theta = np.abs(theta)
        t = abs_theta / np.pi  # 0 when ahead, 1 when behind
        scale = max_scale + t * (min_scale - max_scale)
        return float(np.clip(scale, 1e-6, 1e6))
    
    elif method == "gaussian":
        sigma = kwargs.get("sigma", np.pi / 4)
        sigma = max(sigma, 1e-10)
        # Gaussian: ahead (theta≈0) gets max_scale, behind (theta≈±π) gets min_scale
        # Use 1 - gaussian so that theta=0 gives high value
        g = np.exp(-0.5 * (theta / sigma) ** 2)
        scale = min_scale + g * (max_scale - min_scale)
        return float(np.clip(scale, 1e-6, 1e6))
    
    else:
        raise ValueError(f"Unknown method: {method}. Choose from: cosine, linear, gaussian")


def compute_distance_scale_2d(theta_i, theta_ij, method="cosine", **kwargs):
    """
    2D version: compute distance scale for agent i perceiving neighbor at angle theta_ij.
    
    Args:
        theta_i: Agent i's heading (scalar, radians), 0 = +X
        theta_ij: Angle from i toward j (scalar, radians) in world frame
        method: Scaling method (see compute_distance_scale)
        **kwargs: Method-specific parameters
    
    Returns:
        scale: Float > 0. Multiply neighbor distance by this for force calculation.
    """
    # Compute relative angle: where is j relative to i's forward direction?
    relative_angle = theta_ij - theta_i
    
    # Wrap to [-π, π]
    relative_angle = (relative_angle + np.pi) % (2 * np.pi) - np.pi
    
    return compute_distance_scale(relative_angle, method=method, **kwargs)
