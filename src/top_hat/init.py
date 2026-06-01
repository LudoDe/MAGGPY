"""
init.py - Initialization Module

This module contains all the functions needed to initialize the Monte Carlo simulation. 
"""

import time
import numpy            as np
from pathlib            import Path
from .montecarlo        import SimParams
from typing             import Any, Dict, Tuple
from scipy.integrate    import quad
from scipy.interpolate  import RectBivariateSpline
from ..data_io          import catalogue_prep
from ..init             import load_redshift_data

SEED = 42  # Seed for reproducibility

# =============================================================================
# K-Factor Computation
# =============================================================================
DEFAULT_SPECTRAL_PARAMS = {
    "alpha"     : -0.67,    # 2/3 from synchrotron
    "beta_s"    : -2.59,    # Average value from GRBs
    "n"         : 2,        # Smoothly broken power law curvature
    "z_model"   : None,     # Redshift distribution model (e.g., 'fiducial_Hrad_A1.0')
}

def create_k_interpolator(params = DEFAULT_SPECTRAL_PARAMS, E_p_range=(50, 10_000), z_range=(0, 14)):
    """
    Create a k-factor interpolator for redshift correction.
    
    Parameters:
    -----------
    params : dict
        Must contain 'alpha', 'beta_s', 'n' keys
    """
    def broken_power_law(E, E_p):
        alpha   = params["alpha"]
        beta_s  = params["beta_s"]
        n       = params["n"]
        eps = (-(2 + alpha)/(2 + beta_s))**(1/(n*(alpha - beta_s)))
        y = E / (E_p/eps)
        C_n = 2 ** (1/n)
        return C_n*((y ** (-alpha * n) + y ** (-beta_s * n)) ** (-1 / n))

    def numerator_int(E, E_p):
        return E * broken_power_law(E, E_p)

    def denominator_int(E, E_p):
        return broken_power_law(E, E_p)

    def k_factor(E_p, z):
        numerator, _ = quad(numerator_int, 1 / (1 + z), 10_000 / (1 + z), args=(E_p,))
        denominator, _ = quad(denominator_int, 50, 300, args=(E_p,))
        if denominator == 0:
            return 0.0
        return numerator / denominator

    E_p_grid        = np.logspace(np.log10(E_p_range[0]), np.log10(E_p_range[1]), 60)
    z_grid          = np.linspace(z_range[0], z_range[1], 30)
    log_E_p_grid    = np.log10(E_p_grid)

    k_factor_grid = np.zeros((len(E_p_grid), len(z_grid)))
    for i, E_p in enumerate(E_p_grid):
        for j, z in enumerate(z_grid):
            k_factor_grid[i, j] = k_factor(E_p, z)

    return RectBivariateSpline(log_E_p_grid, z_grid, k_factor_grid, kx=3, ky=3)

def initialize_simulation(
        datafiles   : Path              = Path("datafiles"), 
        params      : Dict[str, Any]    = DEFAULT_SPECTRAL_PARAMS,
    ) -> Tuple[SimParams, Dict[str, np.ndarray]]:
    """
    Initialize the Monte Carlo simulation by loading necessary data and computing integrals.

    Parameters:
        datafiles (Path): Directory containing data files.
        params (dict): Simulation parameters including:
            - alpha, beta_s, n: Spectral parameters
            - theta_c, theta_v_max: Jet geometry
            - z_model (optional): Name of redshift population model

    Returns:
        default_params (SimParams): Simulation parameters including P(z) interpolator.
        data_dict (Dict[str, np.ndarray]): Dictionary of observable data.
    """
    # check if theta_c or theta_v_max exist if it is inside use that otherwise default to 20 max and 3.4 for theta_c
    params_in   = DEFAULT_SPECTRAL_PARAMS.copy()
    params_in.update(params)
    params      = params_in

    rng = np.random.default_rng(SEED)

    z_arr, P_z_interp, total_rate, local_rate, z_grid, P_z_density = load_redshift_data(
        datafiles, params, rng
    )
    
    k_interpolator = create_k_interpolator(params)

    data_dict = catalogue_prep(datafiles=datafiles)

    default_params = SimParams(
        epeak_data          = data_dict["epeak"],
        duration_data       = data_dict["t90"],
        pflux_data          = data_dict["pflux"],
        fluence_data        = data_dict["fluence"],
        yearly_rate         = data_dict["c_det"],
        triggered_years     = data_dict["trigger_years"],
        # MRD-related fields
        z_arr               = z_arr, 
        P_z_interp          = P_z_interp,
        z_grid              = z_grid,
        P_z_density         = P_z_density,
        total_merger_rate   = total_rate,
        local_rate          = local_rate,
        # Lone interpolator needed for top-hat models
        k_interpolator      = k_interpolator,
        # RNG
        rng                 = rng,
    )

    return default_params, data_dict

def create_run_dir(run_name: str = 'run', use_timestamp : bool = False, output_files_default : str = 'Output_files', QUIET_FLAG : bool = False) -> Path:
    """
    Create a directory to store the output files of a given run. The directory is created in the 'Output_files' folder

    Parameters:
    run_name (str): Name of the run
    autoname (bool): If True, append the current date and time to the run name.

    Returns:
    output_files (Path): Path to the output directory
    """

    output_default  = Path(output_files_default)
    output_files    = output_default  / run_name

    if use_timestamp:
        base_name       = f"{run_name}_{time.strftime('_%Y-%m-%d_%H-%M-%S')}"             # Append timestamp to the run name
        output_files    = output_default  / base_name           # Modify the name to include the timestamp

    msg = f"Creating new directory : {output_files}"

    if output_files.exists(): msg = f"Loading existing directory  : {output_files}"
    
    if not QUIET_FLAG: print(msg)

    output_files.mkdir(parents=True, exist_ok=True) # Create the directory if it doesn't exist

    return output_files