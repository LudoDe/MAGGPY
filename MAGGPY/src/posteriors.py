"""
posteriors.py - Posterior Predictive Checks Module

This module provides helper functions to produce, store and read observables. 
"""

import pickle
import numpy        as np
from dataclasses    import dataclass
from .montecarlo    import make_observations, DEFAULT_LIMITS, N_SIMS, make_observations_with_iso
from typing         import Any, Dict, List

@dataclass
class SimulationResults:
    """Store simulation results with variable length arrays"""
    data: Dict[str, List[np.ndarray]]
    
    def __init__(self, data: Dict[str, List[np.ndarray]]):
        self.data = data

    def save(self, filename: str):
        """Save results to pickle file"""
        with open(filename, 'wb') as f:
            pickle.dump(self.data, f)
    
    def __str__(self):
        return f"SimulationResults({self.data.keys})"

    @classmethod
    def load(cls, filename: str) -> 'SimulationResults':
        """Load results from pickle file"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return cls(data)
    
    @classmethod
    def load_and_flatten(cls, filename: str) -> 'SimulationResults':
        """Load results from pickle file and flatten arrays"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        for key, value in data.items():
            if key == "c_det":
                continue
            data[key] = np.concatenate(value)
        return cls(data)

def extract_samples_and_calculate_cdfs(
    sampler                 : Any,
    default_params          : Any,
    default_interpolator    : Any,
    choices                 : int   = 100,
    chain_kargs             : Dict[str, Any] = None,
    flatten                 : bool  = True,
    discard                 : int   = 3_000,
    limits                  : Any   = DEFAULT_LIMITS,
    years_of_data_simulated : float = 10,
    n_sims                  : int   = N_SIMS,
    use_best                : bool  = False
) -> Dict[str, List[Any]]:
    """
    Extract a subset of samples from an MCMC sampler and calculate corresponding CDFs.

    Parameters:
        sampler: MCMC sampler with a get_chain method.
        generate_observables (Callable): Function that generates observables given a sample.
        default_params: Simulation parameters.
        default_interpolator: Integrals/scaling interpolators.
        choices (int): Number of random samples to choose.
        chain_kargs (Dict[str, Any], optional): Keyword arguments for sampler.get_chain.

    Returns:
        Dict[str, List[Any]]: Dictionary of observables' lists (each key holds several CDF arrays).
    """

    # Use default chain parameters if none provided
    default_chain_kargs = {"discard": discard, "thin": 15, "flat": True}
    if chain_kargs is not None:
        default_chain_kargs.update(chain_kargs)

    # Extract chain samples (shape: (n_samples, n_dim))
    chain           = sampler.get_chain(**default_chain_kargs)
    selected_idx    = np.random.choice(chain.shape[0], choices)
    samples         = chain[selected_idx]

    if use_best:
        log_probs   = sampler.get_log_prob(discard=discard, thin=15, flat=True)
        best_idx    = np.argmax(log_probs)
        best_sample = chain[best_idx]
        samples     = best_sample[np.newaxis, :]  # Make it 2D for consistency
        print(f"Best samples are used for PPC: {best_sample}")

    # Initialize dictionary for observables
    cdfs = {
        "epeak"     :   [],
        "t90"       :   [],
        "pflux"     :   [],
        "fluence"   :   [],
        "c_det"     :   [],
        "z_det"     :   [],
        "theta_det" :   [],
        "total_energy_det" : [],
        "isotropic_energy_det" : [],
        "isotropic_luminosity_det" : [],
        "triggered_events" : [],
    }

    for sample in samples:
        observables = generate_observables(sample, default_params, default_interpolator, limits=limits, years_of_data_simulated=years_of_data_simulated, n_sims=n_sims)
        for key in cdfs.keys():
            if type(observables) is float:
                continue
            if not (key in observables.keys()):
                continue
            # in case of nan observables, skip

            cdfs[key].append(observables[key])

    if flatten:
        for key, value in cdfs.items():
            if key == "c_det":
                continue
            cdfs[key] = np.concatenate(value)

    return cdfs

def generate_observables(
        thetas, 
        default_params, 
        default_interpolator, 
        limits: Dict[str, Any]          = DEFAULT_LIMITS,
        years_of_data_simulated: float  = 10,
        n_sims: int = N_SIMS
    ) -> Dict[str, Any]:
    """
    Calculates the log likelihood based on the collected GRB observables.

    Returns:
        The log likelihood value.
    """
    
    #obs = make_observations(thetas, default_params, default_interpolator, limits=limits)
    obs = make_observations_with_iso(thetas, default_params, default_interpolator, limits=limits, n_events=n_sims)

    if obs is None:
        return -np.inf

    t_det, f_det, Ep_det, Fp_det, z_det, theta_det, triggered_events, e_tot_obs, E_iso, L_iso = obs.values()
    
    generated_grbs          = n_sims
    gbm_efficiency          = 0.6 # 60% efficiency for GBM (duty cycle + SAA + FOV)
    geometric_factor_theta  = 1 - np.cos(default_params.theta_v_max) # Geometric factor for the viewing angle
    expected_yearly_rate    = gbm_efficiency * thetas[-1] * len(default_params.z_arr) * geometric_factor_theta
    generate_years_of_GRB   = generated_grbs / expected_yearly_rate 
    simulated_yearly_rate   = triggered_events / generate_years_of_GRB # This is the rate of GRBs in the simulation, not the expected rate
    expected_events         = int(simulated_yearly_rate * years_of_data_simulated)

    ids                     = np.random.randint(0, len(t_det), size=expected_events)
    t_det                   = t_det[ids]
    f_det                   = f_det[ids]
    Ep_det                  = Ep_det[ids]
    Fp_det                  = Fp_det[ids]
    z_det                   = z_det[ids]
    theta_det               = theta_det[ids]
    e_tot_obs               = e_tot_obs[ids] 
    E_iso                   = E_iso[ids]
    L_iso                   = L_iso[ids]
    
    return {
        "epeak"		: Ep_det,
        "t90"	    : t_det,
        "pflux"		: Fp_det,
        "fluence"	: f_det,
        "c_det"		: simulated_yearly_rate,
        "z_det"		: z_det,
        "theta_det"	: theta_det,
        "total_energy_det" : e_tot_obs,
        "isotropic_energy_det" : E_iso,
        "isotropic_luminosity_det" : L_iso,
        "triggered_events" : triggered_events,
    }

def calculate_cdf_bounds(
    arrays: List[np.ndarray],
    x_grid: np.ndarray,
    percentiles: tuple = (5, 50, 95)
) -> Dict[str, np.ndarray]:
    """
    Calculate CDF percentile bounds from multiple realizations.
    
    Parameters:
        arrays: List of 1D arrays (one per posterior sample)
        x_grid: Common x-axis to evaluate CDFs on
        percentiles: Percentiles to compute (default: 5th, 50th, 95th)
    
    Returns:
        Dictionary with 'x', 'lower', 'median', 'upper' arrays
    """
    n_samples = len(arrays)
    n_points = len(x_grid)
    
    # Compute CDF for each realization on the common grid
    cdf_matrix = np.zeros((n_samples, n_points))
    
    for i, arr in enumerate(arrays):
        if len(arr) == 0:
            cdf_matrix[i, :] = np.nan
            continue
        sorted_arr = np.sort(arr)
        # CDF: fraction of points <= x
        cdf_matrix[i, :] = np.searchsorted(sorted_arr, x_grid, side='right') / len(arr)
    
    # Compute percentiles across samples (ignoring NaN)
    lower   = np.nanpercentile(cdf_matrix, percentiles[0], axis=0)
    median  = np.nanpercentile(cdf_matrix, percentiles[1], axis=0)
    upper   = np.nanpercentile(cdf_matrix, percentiles[2], axis=0)
    
    return {
        'x': x_grid,
        'lower': lower,
        'median': median,
        'upper': upper
    }








