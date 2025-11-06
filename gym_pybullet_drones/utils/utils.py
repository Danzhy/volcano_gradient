"""General use functions.
"""
import time
import argparse
import numpy as np
from scipy.optimize import nnls

################################################################################

def sync(i, start_time, timestep):
    """Syncs the stepped simulation with the wall-clock.

    Function `sync` calls time.sleep() to pause a for-loop
    running faster than the expected timestep.

    Parameters
    ----------
    i : int
        Current simulation iteration.
    start_time : timestamp
        Timestamp of the simulation start.
    timestep : float
        Desired, wall-clock step of the simulation's rendering.

    """
    if timestep > .04 or i%(int(1/(24*timestep))) == 0:
        elapsed = time.time() - start_time
        if elapsed < (i*timestep):
            time.sleep(timestep*i - elapsed)

################################################################################

def str2bool(val):
    """Converts a string into a boolean.

    Parameters
    ----------
    val : str | bool
        Input value (possibly string) to interpret as boolean.

    Returns
    -------
    bool
        Interpretation of `val` as True or False.

    """
    if isinstance(val, bool):
        return val
    elif val.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif val.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError("[ERROR] in str2bool(), a Boolean value is expected")

################################################################################

def get_max_duration(num_drones, base_duration=300, scaling_factor=5):
    """Calculate appropriate maximum duration based on swarm size.
    
    Larger swarms require more time due to increased coordination overhead
    and slower consensus formation. This function provides a conservative
    linear scaling to ensure sufficient time for task completion across
    different swarm sizes.
    
    Parameters
    ----------
    num_drones : int
        Number of drones in the swarm.
    base_duration : int, optional
        Base duration in seconds for the smallest swarm size (default: 300).
    scaling_factor : int, optional
        Additional seconds per drone beyond minimum size (default: 5).
    
    Returns
    -------
    int
        Maximum simulation duration in seconds.
    
    Examples
    --------
    >>> get_max_duration(7)
    300
    >>> get_max_duration(10)
    315
    >>> get_max_duration(19)
    360
    >>> get_max_duration(37)
    450
    
    Notes
    -----
    Scaling formula: max_duration = base_duration + (num_drones - 7) * scaling_factor
    
    This ensures:
    - 7 drones:  300s (baseline)
    - 10 drones: 315s (+5% time allowance)
    - 19 drones: 360s (+20% time allowance)
    - 37 drones: 450s (+50% time allowance)
    
    """
    min_drones = 7  # Reference minimum swarm size
    max_duration = base_duration + (num_drones - min_drones) * scaling_factor
    return max_duration
