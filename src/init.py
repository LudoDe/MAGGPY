"""
init.py - Initialization Module

This module contains all the functions needed to initialize the Monte Carlo simulation. 
"""

import re
import time
import numpy            as np
from pathlib            import Path
from scipy              import integrate
from .montecarlo        import SimParams, Interps
from .spectral_models   import broken_power_law, DEFAULT_SPECTRAL_PARAMS
from typing             import Any, Callable, Dict, Optional, Tuple
from .data_io           import load_structure_constants, get_redshift_distribution, catalogue_prep
from .redshift          import get_mrd_redshift_distribution
from scipy.interpolate  import interp1d

SEED = 42  # Seed for reproducibility

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

def load_redshift_data(
    data_dir: Path = Path("datafiles"),
    params: Dict[str, Any] = None,
    rng: np.random.Generator = None
) -> Tuple[np.ndarray, Optional[Callable], Optional[float], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Loads redshift distribution and optionally the MRD probability distribution.

    Args:
        data_dir: Path to the directory containing data files.
        params: Simulation parameters dictionary. Can contain:
            - 'z_model': Name of population model (e.g., 'fiducial_Hrad_A1.0')
            - 'mrd_population': MRD population name (default: extracted from z_model or 'fiducial_Hrad')
            - 'mrd_alpha': MRD alpha parameter (default: extracted from z_model or 'A1.0')
        rng: Random number generator for sampling.

    Returns:
        z_arr: Array of redshift samples (1 year of BNS mergers)
        P_z_interp: Interpolator for P(z) probability density (or None if not using MRD)
        total_rate: Total merger rate per year (or None)
        z_grid: Redshift grid for P(z) (or None)
        P_z_density: P(z) density values (or None)
    """
    if rng is None:
        rng = np.random.default_rng(SEED)
    
    P_z_interp = None
    total_rate = None
    z_grid = None
    P_z_density = None
    
    default_catalog_file = data_dir / 'Catalog_co_BNSs_a_3.0_sn_delayed.txt'
    
    if 'z_model' in params and params['z_model'] is not None:
        model_name = params['z_model']
        population_folder = data_dir / "populations" / "samples"
        
        try:
            # Find the file matching the pattern
            model_file = next(population_folder.glob(f'samples*{model_name}*.dat'))
            z_arr = np.loadtxt(model_file)
            print(f"Using redshift model: {model_file.name} with {len(z_arr)} BNSs.")
            
            # Parse model name to get MRD parameters
            # Expected format: 'fiducial_Hrad_A1.0' or similar
            parts = model_name.split('_')
            
            # Try to extract population and alpha from model name
            mrd_population = params.get('mrd_population', None)
            mrd_alpha = params.get('mrd_alpha', None)
            
            #if mrd_population is None and len(parts) >= 2:
            #    mrd_population = f"{parts[0]}_{parts[1]}"  # e.g., 'fiducial_Hrad'
            #if mrd_alpha is None and len(parts) >= 3:
            #    mrd_alpha = parts[2]  # e.g., 'A1.0'
            
            if mrd_population is None or mrd_alpha is None:
                # Use regex to extract alpha (pattern: A followed by digits and optional decimal)
                alpha_match = re.search(r'_(A\d+\.?\d*)$', model_name)
                
                if alpha_match:
                    if mrd_alpha is None:
                        mrd_alpha = alpha_match.group(1)  # e.g., 'A0.5'
                    
                    if mrd_population is None:
                        # Everything before the alpha is the population name
                        mrd_population = model_name[:alpha_match.start()]  # e.g., 'fiducial_Hrad_5M'

            # Load MRD distribution if we have the parameters
            if mrd_population and mrd_alpha:
                try:
                       P_z_interp, total_rate, local_rate, z_grid, P_z_density = get_mrd_redshift_distribution(
                            datafiles=data_dir,
                            population=mrd_population,
                            alpha=mrd_alpha,
                            component="BNSs"
                        )
                except FileNotFoundError as e:
                    print(f"Warning: Could not load MRD distribution: {e}")
                    print("Continuing without P(z) interpolator.")
                    print("Defaulting to values for Fiducial populations")

                    population  = "fiducial_Hrad"   # Population synthesis model
                    alpha       = "A1.0"            # Common envelope efficiency
                    P_z_interp, total_rate, local_rate, z_grid, P_z_density = get_mrd_redshift_distribution(
                        datafiles=data_dir,
                        population=population,
                        alpha=alpha,
                        component="BNSs"
                    )
                    
        except StopIteration:
            raise ValueError(f"No z_model file found matching pattern: '{model_name}' in {population_folder}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Population directory not found: {population_folder}")
    else:
        print(f"Using default redshift distribution: {default_catalog_file.name}")
        if not default_catalog_file.is_file():
            raise FileNotFoundError(f"Default redshift catalog not found: {default_catalog_file}")
        z_arr = get_redshift_distribution(default_catalog_file)
        
        # For default catalog, create P(z) from histogram
        hist_z, bin_edges   = np.histogram(z_arr, bins=100, density=True)
        bin_centers         = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        P_z_interp          = interp1d(bin_centers, hist_z, kind='linear', bounds_error=False, fill_value=0.0)
        z_grid              = bin_centers
        P_z_density         = hist_z
        total_rate          = len(z_arr)  # Assuming the catalog represents 1 year
        local_rate          = 365
    
    return z_arr, P_z_interp, total_rate, local_rate, z_grid, P_z_density

def load_and_filter_redshifts(
    data_dir: Path = Path("datafiles"),
    params: Dict[str, Any] = None
) -> np.ndarray:
    """
    Loads redshift distribution, optionally applies a model, and filters by z_max.
    
    DEPRECATED: Use load_redshift_data() instead for full MRD support.
    """
    z_arr, _, _, _, _ = load_redshift_data(data_dir, params)
    return z_arr

def initialize_simulation(
        datafiles   : Path              = Path("datafiles"), 
        params      : Dict[str, Any]    = DEFAULT_SPECTRAL_PARAMS,
    structure_source : Optional[Path] = None,
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
        structure_source (Path, optional): Directory or CSV file containing
            custom structured-jet constants. If omitted, the legacy split files
            are used.
        size_test (int): Number of viewing angles to generate.

    Returns:
        default_params (SimParams): Simulation parameters including P(z) interpolator.
        default_interpolator (Interps): Interpolator containing integrals and scaling functions.
        data_dict (Dict[str, np.ndarray]): Dictionary of observable data.
    """
    # check if theta_c or theta_v_max exist if it is inside use that otherwise default to 20 max and 3.4 for theta_c
    params_in   = DEFAULT_SPECTRAL_PARAMS.copy()
    params_in.update(params)
    params      = params_in

    rng = np.random.default_rng(SEED)

    deg_to_rad = np.pi / 180
    alpha, beta_s, n = params["alpha"], params["beta_s"], params["n"]

    z_arr, P_z_interp, total_rate, local_rate, z_grid, P_z_density = load_redshift_data(
        datafiles, params, rng
    )

    R_F, R_E, alpha_n, alpha_e, _, _ = load_structure_constants(datafiles, structure_source)
    
    cos_angle_min = np.cos(params["theta_v_max"] * deg_to_rad)
    theta_v = np.arccos(rng.uniform(cos_angle_min, 1, size=size_test))
    
    int_0_alt, int_1_alt, int_2_alt, int_3_alt, int_4_alt, _ = create_integral_interpolators(
        alpha=alpha, beta_s=beta_s, n=n
    )

    data_dict = catalogue_prep(datafiles=datafiles)

    default_params = SimParams(
        theta_c         = params["theta_c"] * deg_to_rad, 
        theta_v_max     = params["theta_v_max"] * deg_to_rad, 
        z_arr           = z_arr, 
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
        # MRD-related fields
        P_z_interp          = P_z_interp,
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
        int_4_alt   = int_4_alt
    )

    return default_params, default_interpolator, data_dict

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