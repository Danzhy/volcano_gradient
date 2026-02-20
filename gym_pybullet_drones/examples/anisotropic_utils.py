"""
Anisotropic Vision Utilities

Provides angle-based weighting functions for anisotropic perception in flocking simulations.
Agents perceive neighbors differently based on relative angle:
- Neighbors ahead (in front of heading): full weight
- Neighbors to the side: reduced weight  
- Neighbors behind: minimal or zero weight
"""
import numpy as np


def compute_angle_weight(theta, method="gaussian_normal", **kwargs):
    """
    Compute weight for a neighbor at relative angle theta.
    
    Args:
        theta: Relative angle (radians) in [-π, π]
               0 = ahead, π/2 = left, -π/2 = right, ±π = behind
        method: Weighting method ("gaussian_normal", "cosine", "step", "uniform")
        **kwargs: Method-specific parameters:
            - sigma: FOV width for gaussian (default: π/4)
            - mu: Center direction (default: 0.0 = forward)
            - min_weight: Minimum weight for behind (default: 0.0)
            - power: Sharpness for cosine (default: 2)
            - clip_value: Max angle for cosine (default: π/2)
    
    Returns:
        weight: Float in [0, 1] (or [min_weight, 1])
    """
    if method == "gaussian_normal":
        sigma = kwargs.get("sigma", np.pi / 4)
        mu = kwargs.get("mu", 0.0)
        min_weight = kwargs.get("min_weight", 0.0)
        
        # Ensure sigma > 0 to avoid division by zero
        sigma = max(sigma, 1e-10)
        
        # Gaussian weight centered at mu
        weight = np.exp(-0.5 * ((theta - mu) / sigma) ** 2)
        
        # Apply minimum weight floor
        weight = max(weight, min_weight)
        
        return float(weight)
    
    elif method == "cosine":
        power = kwargs.get("power", 2)
        clip_value = kwargs.get("clip_value", np.pi / 2)
        
        # Clip angle to [-clip_value, clip_value]
        theta_clipped = np.clip(theta, -clip_value, clip_value)
        
        # Cosine weight: cos^power(theta_clipped / clip_value * π/2)
        # At theta=0: cos(0) = 1 → weight = 1
        # At theta=±clip_value: cos(π/2) = 0 → weight = 0
        normalized_angle = theta_clipped / clip_value * (np.pi / 2)
        weight = np.cos(normalized_angle) ** power
        
        return float(max(weight, 0.0))
    
    elif method == "step":
        fov_half = kwargs.get("fov_half", np.pi / 4)  # Half FOV angle
        min_weight = kwargs.get("min_weight", 0.0)
        
        if abs(theta) <= fov_half:
            return 1.0
        else:
            return min_weight
    
    elif method == "uniform":
        # No angle weighting (isotropic)
        return 1.0
    
    else:
        raise ValueError(f"Unknown method: {method}. Choose from: gaussian_normal, cosine, step, uniform")


def compute_angle_weight_2d(theta_i, theta_ij, method="gaussian_normal", **kwargs):
    """
    2D version: compute weight for agent i perceiving neighbor at angle theta_ij.
    
    Args:
        theta_i: Agent i's heading (scalar, radians), 0 = +X
        theta_ij: Angle from i toward j (scalar, radians) in world frame
        method: Weighting method (see compute_angle_weight)
        **kwargs: Method-specific parameters
    
    Returns:
        weight: Float in [0, 1]
    """
    # Compute relative angle: where is j relative to i's forward direction?
    relative_angle = theta_ij - theta_i
    
    # Wrap to [-π, π]
    relative_angle = (relative_angle + np.pi) % (2 * np.pi) - np.pi
    
    return compute_angle_weight(relative_angle, method=method, **kwargs)
