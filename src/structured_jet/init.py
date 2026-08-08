"""
init.py - Initialization Module

This module contains all the functions needed to initialize the Monte Carlo simulation. 
"""

import numpy                as np
from pathlib                import Path
from scipy                  import integrate
from typing                 import Any, Callable, Dict, Tuple
from .montecarlo            import SimParams, Interps
from ..spectral_models      import broken_power_law, DEFAULT_SPECTRAL_PARAMS
from ..data_io              import get_Rf_Re, get_alpha_n_alpha_e, catalogue_prep
from ..redshift             import get_redshift_quantities, sample_from_mrd

SEED = 42  # Seed for reproducibility

DEFAULT_PARAMS = DEFAULT_SPECTRAL_PARAMS.copy()
DEFAULT_PARAMS["theta_c"]       = 3.4  # Ghirlanda half-angle of jet core (from GW170817)
DEFAULT_PARAMS["theta_v_max"]   = 10  # Maximum viewing angle to simulate (in degrees) as to not waste time on very off-axis events that are not detectable

def create_integral_interpolators( 
    alpha   : float = DEFAULT_SPECTRAL_PARAMS["alpha"],
    beta_s  : float = DEFAULT_SPECTRAL_PARAMS["beta_s"],
    n       : float = DEFAULT_SPECTRAL_PARAMS["n"],
    E_p_arr : np.ndarray = np.logspace(-3, 5, 200),  # E_p range from 10^1 to 10^4
) -> Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray], 
           Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray], np.ndarray]:
    """
    Calculates model integrals numerically and returns interpolators.
    
    This version directly uses E_p values rather than k=1/E_p and incorporates 
    the E_p scaling into the integrations to avoid additional multiplications.

    Args:
        alpha: Spectral index before the break.
        beta_s: Spectral index after the break.
        n: Smoothness parameter for the break.
        E_p_arr: Array of peak energies (E_p) for which to compute integrals.

    Returns:
        A tuple containing:
        - interp_integral_0: Interpolator for the first integral variant.
        - interp_integral_1: Interpolator for the second integral variant.
        - interp_integral_2: Interpolator for the third integral variant.
        - interp_integral_3: Interpolator for the fourth integral variant.
        - E_p_values: The array of E_p values used for the calculation.
    """
    
    # Calculate integrals for each E_p value
    integral_results = []
    
    bounds = [(1, 1e4), (10, 1e3), (10, 1e3), (50, 300), (50, 300)]
    
    # Define integrand functions for each integral variant
    def integrand_0(E, E_p):
        return E * broken_power_law(E, E_p, alpha=alpha, beta_s=beta_s, n=n)  # E * N(E)
    
    def integrand_1(E, E_p):
        return broken_power_law(E, E_p, alpha=alpha, beta_s=beta_s, n=n)  # N(E)
    
    # Functions corresponding to each integral
    integrand_funcs = [integrand_0, integrand_0, integrand_1, integrand_1, integrand_0] #integral 4 is fluence in BATSE range, as t90 is measured in BATSE range
    
    # Compute each integral for all E_p values
    for func, bounds in zip(integrand_funcs, bounds):
        def integrand_wrapper(E, E_p=E_p_arr, func=func):
            return func(E, E_p)
        
        results, _ = integrate.quad_vec(integrand_wrapper, bounds[0] , bounds[1])
        
        integral_results.append(results)
    
    # Create interpolation functions
    interp_funcs = [
        lambda x, data=data: np.interp(x, E_p_arr, data)
        for data in integral_results
    ]
    
    return (*interp_funcs, E_p_arr)

def initialize_simulation(
        datafiles   : Path              = Path("datafiles"), 
        mrd_path    : Path              = Path("datafiles/MRD_outputs/fiducial_A1_0_BNS.csv"),
        params      : Dict[str, Any]    = DEFAULT_PARAMS,
        size_test   : int = 2_000
    ) -> Tuple[SimParams, Interps, Dict[str, np.ndarray]]:
    """
    Initialize the Monte Carlo simulation by loading necessary data and computing integrals.

    Parameters:
        datafiles (Path): Directory containing data files.
        params (dict): Simulation parameters including:
            - alpha, beta_s, n: Spectral parameters
            - theta_c, theta_v_max: Jet geometry
            - z_model (optional): Name of redshift population model
        size_test (int): Number of viewing angles to generate.

    Returns:
        default_params (SimParams): Simulation parameters including P(z) interpolator.
        default_interpolator (Interps): Interpolator containing integrals and scaling functions.
        data_dict (Dict[str, np.ndarray]): Dictionary of observable data.
    """
    rng         = np.random.default_rng(SEED)

    params_in   = DEFAULT_SPECTRAL_PARAMS.copy() # any user-specified params will override defaults
    params_in.update(params) 

    deg_to_rad          = np.pi / 180
    alpha, beta_s, n    = params_in["alpha"], params_in["beta_s"], params_in["n"]

    total_rate, local_rate, z_grid, P_z_density = get_redshift_quantities(
        mrd_path=mrd_path
    )
    z_arr                   = sample_from_mrd(z_grid, P_z_density, int(total_rate), rng)

    R_F, R_E, _             = get_Rf_Re(datafiles / 'structure_constants' / 'F_Fmax_3.4_s4.0.txt')
    alpha_n, alpha_e, _, _  = get_alpha_n_alpha_e(datafiles / 'structure_constants' / 'alpha.txt', datafiles / 'structure_constants' / 'alpha_e.txt')
    cos_angle_min           = np.cos(params_in["theta_v_max"] * deg_to_rad)
    theta_v                 = np.arccos(rng.uniform(cos_angle_min, 1, size=size_test))
    
    int_0_alt, int_1_alt, int_2_alt, int_3_alt, int_4_alt, _ = create_integral_interpolators(
        alpha=alpha, beta_s=beta_s, n=n
    )

    data_dict       = catalogue_prep(datafiles=datafiles)

    default_params = SimParams(
        theta_c         = params_in["theta_c"] * deg_to_rad, 
        theta_v_max     = params_in["theta_v_max"] * deg_to_rad, 
        z_arr           = z_arr, # sampled redshift values 
        theta_v         = theta_v,
        epeak_data      = data_dict["epeak"],
        duration_data   = data_dict["t90"],
        pflux_data      = data_dict["pflux"],
        fluence_data    = data_dict["fluence"],
        yearly_rate     = data_dict["c_det"],
        triggered_years = data_dict["trigger_years"],
        rng             = rng,
        alpha_n         = alpha_n(theta_v),
        alpha_e         = alpha_e(theta_v),
        R_F             = R_F(theta_v),
        R_E             = R_E(theta_v),

        # New MRD-related fields
        z_grid              = z_grid,
        P_z_density         = P_z_density,
        total_merger_rate   = total_rate,
        local_rate          = local_rate
    )

    default_interpolator = Interps(
        int_0_alt   = int_0_alt,
        int_1_alt   = int_1_alt,
        int_2_alt   = int_2_alt,
        int_3_alt   = int_3_alt,
        int_4_alt   = int_4_alt,
    )

    return default_params, default_interpolator, data_dict