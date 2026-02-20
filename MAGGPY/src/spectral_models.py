"""
spectral_models.py - Spectral Model Functions

This module contains spectral model functions used across the simulation.
"""

import numpy as np

# Default spectral parameters
DEFAULT_SPECTRAL_PARAMS = {
    "alpha"         : -0.67,    # 2/3 from synchrotron
    "beta_s"        : -2.59,    # Average value from GRBs
    "n"             : 2,        # Smoothly broken power law curvature
    "theta_c"       : 3.4,      # Ghirlanda half-angle of jet core (from GW170817)
    "theta_v_max"   : 10,       # Maximum viewing angle of the jet (in degrees)
}

def broken_power_law(
    E, 
    E_p, 
    alpha   = DEFAULT_SPECTRAL_PARAMS["alpha"], 
    beta_s  = DEFAULT_SPECTRAL_PARAMS["beta_s"], 
    n       = DEFAULT_SPECTRAL_PARAMS["n"]
):
    """
    Broken power law spectrum function.
    
    Args:
        E: Energy values
        E_p: Peak energy
        alpha: Spectral index before the break
        beta_s: Spectral index after the break
        n: Smoothness parameter for the break
        
    Returns:
        Spectral values at given energies
    """
    eps = (-(2 + alpha)/(2 + beta_s))**(1/(n*(alpha - beta_s)))
    y   = E / (E_p/eps)  # Ep = E_break * eps 
    C_n = 2 ** (1/n) # f(1) = 1 
    return C_n*((y ** (-alpha * n) + y ** (-beta_s * n)) ** (-1 / n))