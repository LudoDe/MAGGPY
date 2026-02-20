"""
montecarlo.py - Monte Carlo Simulation Module

This module contains all the functions needed to run the Monte Carlo simulation, including parralelization.
"""

import functools
import numpy 			as np 
from scipy.special 		import gammaln
from astropy.cosmology 	import Planck18, FlatLambdaCDM
from dataclasses 		import dataclass
from scipy.integrate    import cumulative_trapezoid
from scipy.stats        import cramervonmises_2samp, ks_2samp, gengamma
from typing             import Any, Dict, Optional, Callable
from scipy.integrate    import cumulative_trapezoid, quad_vec
from .spectral_models   import broken_power_law
# OPTIMIZED MOVING AVERAGE using np.lib.stride_tricks
from numpy.lib.stride_tricks import sliding_window_view
    
DEFAULT_LIMITS: Dict[str, Any] = {
    "F_LIM"         : 4,         # 4 ph/cm^2/s in 64 ms
    "T90_LIM"       : 2,         # 1 s
    "EP_LIM_UPPER"  : 10_000,    # 10_000 keV
    "EP_LIM_LOWER"  : 50,        # 50 keV
}

N_SIMS          = 20_000 #  Default

@dataclass
class SimParams:
    theta_c                 : float         # jet core angle in radians
    theta_v_max             : float         # minimum viewing angle in radians
    z_arr                   : np.ndarray
    theta_v                 : np.ndarray    

    epeak_data              : np.ndarray
    duration_data           : np.ndarray
    pflux_data              : np.ndarray
    fluence_data            : np.ndarray
    yearly_rate             : float         # expected yearly rate of GRBs
    triggered_years         : float           # number of years with triggered events

    rng                     : np.random.Generator 

    R_E                     : np.ndarray #= None
    R_F                     : np.ndarray #= None
    alpha_e                 : np.ndarray #= None
    alpha_n                 : np.ndarray #= None
        
    # Properties to set in __post_init__
    z_corr                  : float = None
    z_count                 : float = None  
    geometric_factor        : float = None  

    # MRD-related fields
    P_z_interp              : Optional[Callable]    = None
    z_grid                  : Optional[np.ndarray]  = None
    P_z_density             : Optional[np.ndarray]  = None
    total_merger_rate       : Optional[float]       = None
    local_rate              : Optional[float]       = None  # R_0 at z=0 in Gpc^-3 yr^-1
    
    def __post_init__(self):
        # Automatically compute derived parameters
        self.z_count                = len(self.z_arr) * (1 - np.cos(self.theta_v_max)) # "on-axis" BNS
        self.geometric_factors      = 1 / (4 * np.pi * d_l(self.z_arr)**2 * (1 - np.cos(self.theta_c)))
        self.z_corr                 = 1 + self.z_arr

@dataclass
class Interps:
    int_0_alt : callable
    int_1_alt : callable
    int_2_alt : callable
    int_3_alt : callable
    int_4_alt : callable

    interp_t90      : Any = None
    interp_flu      : Any = None
    interp_pf       : Any = None
   
def l_random_new(A, n, rng = None):
    return gengamma.rvs(a = (A - 1)/A, c = -A, size=n, random_state=rng) #! Modified schelchter, based on Salafia et al 2024

# Cosmology assumptions
def d_l(z, cosmology = FlatLambdaCDM(H0 = Planck18.H0.value, Om0 = Planck18.Om0)):
	return cosmology.luminosity_distance(z).cgs.value

def poiss_log(k, mu):
    """
    Calculate the Poisson probability mass function at k given λ.
    By using this way of expressing the distribution we avoid crashes for k too large

    Parameters:
    - k: number of events
    - lam: average number of events
    """	
    return -mu + k * np.log(mu) - gammaln(k + 1)

def generate_macro_properties(thetas : list, params : SimParams, interps :Interps, n_counts_new : int) -> dict:
    """
    Generate macro properties for GRB samples.

    Parameters:
        thetas : list
            List of parameters to generate the macro properties.
        params : SimParams
            Simulation parameters.
        interps : Interps
            Interpolators for integrals and scaling functions.
        n_counts_new : int
            Number of samples to generate. 

    Returns:
        dict: Dictionary containing macro properties for the generated GRB samples
    """
    k_pl, L_L0, L_mu_E_10, sigma_E_10, L_mu_tau_10, sigma_tau_10, _ = thetas # fj unused

    rng         = params.rng

    idx         = rng.integers(low = 0, high = len(params.z_corr), size = n_counts_new) 
    geometry    = params.geometric_factors[idx] # 4 pi DL^2 * (1 - cos(theta_c))
    one_plus_z  = params.z_corr[idx] # 1 + z

    l_10        = np.log(10)
    t_peak      = rng.lognormal(mean=L_mu_tau_10 * l_10, sigma=sigma_tau_10  * l_10, size=n_counts_new) 
    E_p_hat     = rng.lognormal(mean=L_mu_E_10   * l_10, sigma=sigma_E_10    * l_10, size=n_counts_new) 

    idtheta     = rng.integers(low = 0, high = len(params.theta_v), size = n_counts_new) 
    R_E_theta   = params.R_E[idtheta] 
    R_F_theta   = params.R_F[idtheta]

    L_arr       = l_random_new(k_pl, n_counts_new, rng = rng) 

    E_p_obs     = R_E_theta * E_p_hat / one_plus_z
    I1          = 1.15739   * t_peak * interps.int_0_alt(E_p_hat) # Approximate
    N0          = 1e49      * 10**L_L0 * L_arr * geometry / I1
    
    F_0         = N0  * (one_plus_z)**2 * R_F_theta
    F_P_real    = F_0 * interps.int_3_alt( E_p_obs ) * 6.2e8 # peak flux in 50-300 keV (BATSE) 1 erg to keV

    return {
        "t_peak_c_z"        : one_plus_z * t_peak,
        "F_p_real"          : F_P_real,
        "F_0"               : F_0,
        "E_p_obs"           : E_p_obs,
        "R_F_theta"         : R_F_theta,
        "alpha_e"           : params.alpha_e[idtheta], # Can actually select from id_theta at time evolution step,
        "alpha_n"           : params.alpha_n[idtheta],
        "z"                 : one_plus_z - 1, # avoids finding the redshift again, these are big arrays
        "theta_v"           : params.theta_v[idtheta], # This is the viewing angle
        "isotropic_energy"  : 1e49 * 10**L_L0 * L_arr / (1 - np.cos(params.theta_c)) # Total energy in erg
    }

def compute_Fp_64_ms_sliding_window_old(m_prop_m: dict, interps: Interps) -> np.ndarray:
    """Calculate peak flux with optimized moving average."""
    t_peak      = m_prop_m['t_peak_c_z']
    
    n_bins      = 16
    bin_width   = 0.008
    window_size = 8
    
    time_offsets = np.arange(-n_bins / 2 + 0.5, n_bins / 2 + 0.5) * bin_width
    
    # Broadcasting setup
    abs_times = time_offsets[:, np.newaxis] + t_peak[np.newaxis, :]
    time_ratios = np.maximum(abs_times / t_peak[np.newaxis, :], 1e-12)
    
    # Get all needed parameters at once
    alpha_e = m_prop_m['alpha_e'][np.newaxis, :]
    alpha_n = m_prop_m['alpha_n'][np.newaxis, :]
    E_p_obs = m_prop_m["E_p_obs"][np.newaxis, :]
    F_0     = m_prop_m["F_0"][np.newaxis, :]
    
    # Calculate temporal profiles
    mask_before_peak = time_ratios < 1.0
    P_e_mat = np.where(mask_before_peak, 1.0, time_ratios ** (-alpha_e))
    P_n_mat = np.where(mask_before_peak, time_ratios, time_ratios ** (-alpha_n))
    
    # Spectral evolution
    E_p_t = E_p_obs * P_e_mat
    F_0_t = F_0 * P_n_mat
    
    # Calculate flux
    flux_array = F_0_t * interps.int_3_alt(E_p_t) * 6.2e8
    
    # Create sliding windows along axis 0 (time axis)
    flux_windows = sliding_window_view(flux_array, window_shape=window_size, axis=0)
    # Shape: (n_bins - window_size + 1, n_grbs, window_size)
    
    # Average over the window dimension
    moving_averages = flux_windows.mean(axis=-1)
    # Shape: (n_bins - window_size + 1, n_grbs)
    
    # Find maximum along time axis
    F64ms = np.max(moving_averages, axis=0)
    
    return F64ms

def compute_time_evolution_old(m_prop_m: dict, interps: Interps, 
                                      bin_width_ms: float = 16.0,
                                      n_bins: int = 2000) -> tuple:
    """
    Compute the time evolution of the fluence using fixed time bins.
    
    This version uses fixed time intervals (e.g., 16ms) independent of t_peak,
    which is more consistent with how real detectors operate. For 200 bins this is about 3.2s. So safe for most GRBs.
    
    Parameters:
        m_prop_m : dict
            Dictionary containing macro properties for valid GRBs.
        interps : Interps
            Interpolator object containing spectral integration functions.
        bin_width_ms : float
            Width of each time bin in milliseconds (default: 16ms)
        n_bins : int
            Total number of time bins to compute (default: 200)
    
    Returns:
        t_90_array_out : np.ndarray
            Array of t₉₀ values.
        final_fluence : np.ndarray
            Final fluence values.
    """
    t_peak          = m_prop_m['t_peak_c_z']
    
    # Convert bin width to seconds
    bin_width       = bin_width_ms / 1000.0
    
    # Create time grid: start from 0, go out to n_bins * bin_width
    # Use bin centers for evaluation
    time_edges      = np.arange(n_bins + 1) * bin_width
    time_centers    = (time_edges[:-1] + time_edges[1:]) / 2.0  # Shape: (n_bins,)
    
    # Broadcast time centers for all GRBs
    # Shape: (n_bins, n_grbs)
    time_grid = time_centers[:, np.newaxis]  # (n_bins, 1)
    t_peak_grid = t_peak[np.newaxis, :]      # (1, n_grbs)
    
    # Calculate time ratios t/t_peak
    time_ratios = time_grid / t_peak_grid
    safe_time_ratios = np.maximum(time_ratios, 1e-12)
    
    # Calculate P_e(t) and P_n(t) using the same prescription
    P_e_mat = np.where(safe_time_ratios < 1.0, 1.0, np.power(safe_time_ratios, -m_prop_m['alpha_e']))
    P_n_mat = np.where(safe_time_ratios < 1.0, safe_time_ratios, np.power(safe_time_ratios, -m_prop_m['alpha_n']))

    # Calculate evolving spectral properties
    E_p_obs_t = m_prop_m["E_p_obs"] * P_e_mat
    
    # Calculate fluence rate at each time bin (erg/cm²/s in 50-300 keV)
    fluence_rate = m_prop_m["F_0"] * P_n_mat * interps.int_4_alt(E_p_obs_t)
    
    # Integrate fluence over time using trapezoidal rule
    # Note: using time_centers as x-values
    fluence_time_mat = cumulative_trapezoid(
        fluence_rate, 
        x=time_centers, 
        axis=0, 
        initial=0
    )
    
    # Final fluence at last time bin
    total_fluence = fluence_time_mat[-1]
    
    # Find t₉₀: time when fluence reaches 90% of total
    # Need to handle edge cases where 90% is never reached
    threshold_fluence = 0.9 * total_fluence
    
    # Find first index where fluence >= 90% threshold
    above_threshold = fluence_time_mat >= threshold_fluence
    t_90_indices = np.argmax(above_threshold, axis=0)
    
    # Handle cases where threshold is never reached (use last bin)
    never_reached = ~np.any(above_threshold, axis=0)
    t_90_indices[never_reached] = n_bins - 1
    
    # Get actual t₉₀ times from the time grid
    t_90_array_out = time_centers[t_90_indices]
    
    return t_90_array_out, total_fluence

def make_observations(thetas, params : SimParams, interps : Interps, 
                      limits : Dict[str, Any] = DEFAULT_LIMITS, n_events : int = N_SIMS):
    """
    Accumulates GRB observables until the number of counts reaches min_count.

    Returns:
        t_det, f_det, Ep_det, Fp_det: Lists with aggregated observables.
        count_real_grbs: Total count of base valid GRBs.
        current_n: Total samples drawn.
        count: Total count of observed catalogue GRBs.
    """

    m_prop          = generate_macro_properties(thetas, params, interps, n_events)

    trigger_mask = (
        (m_prop['t_peak_c_z'] < limits["T90_LIM"]) &
        (m_prop['F_p_real'] > limits["F_LIM"])
    )

    if np.sum(trigger_mask) <= 5:
        return None  # Avoid very small samples

    # Filter properties to triggered events only
    m_prop_triggered = {k: v[trigger_mask] for k, v in m_prop.items()}

    # Compare to sliding window version for consistency
    #P_F_64ms_50_300         = compute_Fp_64_ms_sliding_window(m_prop_triggered, interps)
    P_F_64ms_50_300         = compute_Fp_64_ms_optimized(m_prop_triggered, interps)
    t_90_array, f_det_in    = compute_time_evolution(m_prop_triggered, interps)

    # Compare to interps
    #P_F_64ms_50_300         = compute_Fp_64_ms_sliding_window_2(m_prop_triggered, interps)

    #t_90_array, f_det_in    = compute_time_evolution_2(m_prop_triggered, interps)

    # Additional detection criteria on computed observables
    detection_mask = (
        (t_90_array < limits["T90_LIM"]) &
        (P_F_64ms_50_300 > limits["F_LIM"])
    )

    triggered_events = np.sum(detection_mask) # Total triggered events before shape cuts

    if triggered_events == 0:
        return None
    
    # Shape analysis cuts (additional Ep cuts for distribution comparison)
    shape_mask = (
        (m_prop_triggered["E_p_obs"] > limits["EP_LIM_LOWER"]) &
        (m_prop_triggered["E_p_obs"] < limits["EP_LIM_UPPER"])
    )

    # Combine detection and shape masks for final sample
    final_mask = detection_mask & shape_mask

    if np.sum(final_mask) == 0:
        return None

    return {
        "t_det"                 : t_90_array[final_mask],
        "f_det"                 : f_det_in[final_mask],
        "Ep_det"                : m_prop_triggered["E_p_obs"][final_mask],
        "Fp_det"                : P_F_64ms_50_300[final_mask],
        "z_det"                 : m_prop_triggered["z"][final_mask],
        "theta_v_det"           : m_prop_triggered["theta_v"][final_mask],
        "triggered_events"      : triggered_events,
        "isotropic_energy_det"  : m_prop_triggered["isotropic_energy"][final_mask],
    }

def cdf_sample(data, n, rng):
    """
    Generate n random samples from the empirical distribution of data
    using inverse transform sampling.
    
    Parameters:
        data: array-like, input data to sample from
        n: int, number of samples to generate
        
    Returns:
        numpy array of n samples distributed according to data
    """
    # Sort the data to build the empirical CDF
    x_sorted    = np.sort(data)
    u           = rng.uniform(0, 1, n) # Generate n random percentiles
    samples     = np.interp(u, np.linspace(0, 1, len(data)), x_sorted) # Direct inverse transform sampling (map percentiles to data values)     
    return samples

def crammer_score(y_sim, y_obs, rng):
    y_resample   = cdf_sample(y_sim, len(y_obs), rng=rng)  # Resample y_sim to match the length of y_obs, mainly if simulating a small amount of events for 16 years this shouldn't be necessary
    y_in        = np.log10(y_resample)
    y_out       = np.log10(y_obs)
    return np.log(cramervonmises_2samp(y_in, y_out).pvalue) # Use the Cramer von Mises test for shape comparison

def score_func(y_sim, y_obs, rng=None):
    return crammer_score(y_sim, y_obs, rng=rng)

def log_likelihood_(
        thetas              : list, 
        params              : SimParams, 
        interps             : Interps, 
        limits              : Dict[str, Any] = DEFAULT_LIMITS,
        n_years             : float = None
    ):
    
    params.rng = np.random.default_rng(42)

    fj                      = thetas[-1]  # last parameter is f_j
    
    n_years                 = params.triggered_years if n_years is None else n_years # 16 ish years
    GBM_eff                 = 0.6
    geometric_efficiency    = 1 - np.cos(params.theta_v_max)
    #factor to make the sim faster by simulating less years and scaling up later
    factor                  = 1
    n_years                 = n_years / factor
    
    # Total BNS mergers per year in the universe (all-sky)
    total_bns_all_sky       = n_years * len(params.z_arr)
    
    # Total number of GRBs to simulate (already accounts for geometry and time)
    available_events        = total_bns_all_sky * geometric_efficiency * GBM_eff * fj
    n_events                = int(available_events)

    # Run simulation
    obs = make_observations(
        thetas, 
        params, 
        interps,
        limits = limits,
        n_events = n_events
    )

    if obs is None:
        return -np.inf, None, None, None, None, None

    t_det = obs["t_det"]

    detected_events = len(t_det)

    if detected_events < 5:  # Minimum required events for stable comparison
        return -np.inf, None, None, None, None, None 

    #Cramer von Mises test but log scale obs and sim values
    logL_shape_t90     = score_func(t_det           , params.duration_data  , rng=params.rng    )
    logL_shape_epeak   = score_func(obs["Ep_det"]   , params.epeak_data     , rng=params.rng    )
    logL_shape_pflux   = score_func(obs["Fp_det"]   , params.pflux_data     , rng=params.rng    )
    logL_shape_fluence = score_func(obs["f_det"]    , params.fluence_data   , rng=params.rng    )

    total_logL_shape    = (logL_shape_epeak + logL_shape_t90 + logL_shape_pflux + logL_shape_fluence) 

    # --- Calculate Rate Log-Likelihood ---
    total_observed_events   = params.yearly_rate * params.triggered_years # this is the triggered total number of events
    # normalize by factor 
    total_observed_events  = total_observed_events / factor

    detection_efficiency = obs["triggered_events"] / n_events
    expected_detections  = available_events * detection_efficiency

    # 3. Calculate the Poisson log-likelihood
    if expected_detections <= 0:
        return -np.inf, None, None, None, None, None

    logL_norm = poiss_log(k=total_observed_events, mu=expected_detections)

    # For diagnostics, calculate the equivalent yearly rate
    simulated_yearly_rate = expected_detections / n_years

    # Total log-likelihood
    likelihood_total    = total_logL_shape + logL_norm
    
    return likelihood_total, simulated_yearly_rate , logL_shape_epeak, logL_shape_t90, logL_shape_pflux, logL_shape_fluence

def log_likelihood(
        thetas              : list, 
        params              : SimParams, 
        interps             : Interps, 
        limits              : Dict[str, Any]    = DEFAULT_LIMITS,
        n_years             : float             = None,
        n_events_fixed      : int               = 20_000  # NEW: option to fix simulation size
    ):
    
    fj                  = thetas[-1]
    n_years             = params.triggered_years if n_years is None else n_years
    gbm_efficiency      = 0.6
    geo_efficiency      = 1 - np.cos(params.theta_v_max)
    
    # BNS mergers per year (z_arr represents 1 year of mergers)
    bns_per_year        = len(params.z_arr)
    
    # Expected GRB rate in our cone
    grb_rate_per_year   = bns_per_year * geo_efficiency * gbm_efficiency * fj
    
    if grb_rate_per_year <= 5: # no reason to simulate if the rate is too low
        return -np.inf, None, None, None, None, None

    n_events            = n_events_fixed

    # Run simulation
    obs = make_observations(thetas, params, interps, limits=limits, n_events=n_events)

    if obs is None or len(obs["t_det"]) < 10:
        return -np.inf, None, None, None, None, None

    # --- Shape Likelihood ---
    logL_shape_t90          = score_func(obs["t_det"],  params.duration_data, rng=params.rng)
    logL_shape_epeak        = score_func(obs["Ep_det"], params.epeak_data,    rng=params.rng)
    logL_shape_pflux        = score_func(obs["Fp_det"], params.pflux_data,    rng=params.rng)
    logL_shape_fluence      = score_func(obs["f_det"],  params.fluence_data,  rng=params.rng)
    
    total_logL_shape        = logL_shape_epeak + logL_shape_t90 + logL_shape_pflux + logL_shape_fluence

    # --- Rate Likelihood ---
    total_observed          = params.yearly_rate * n_years # this is just the rate 
    
    # Scale detection efficiency properly
    detection_fraction      = obs["triggered_events"] / n_events
    expected_total          = grb_rate_per_year * n_years * detection_fraction

    if expected_total <= 0:
        return -np.inf, None, None, None, None, None

    logL_rate               = poiss_log(k=total_observed, mu=expected_total)
    simulated_yearly_rate   = expected_total / n_years

    return (
        total_logL_shape + logL_rate,
        simulated_yearly_rate,
        logL_shape_epeak,
        logL_shape_t90,
        logL_shape_pflux,
        logL_shape_fluence
    )







TIME_RESOLUTION = 200
MIN_T_RATIO     = 0.0001
MAX_T_RATIO     = 1_000

def compute_Fp_64_ms_optimized(m_prop_m: dict, interps: Interps) -> np.ndarray:
    """
    Calculate the peak flux averaged over 64ms time window (Optimized Version).

    This function vectorizes the flux calculation and moving average.

    Parameters:
        m_prop_m: Dictionary containing GRB macro properties.
                  Requires keys: 't_peak_c_z', 'alpha_e', 'alpha_n',
                                 'E_p_obs', 'F_0'.
        interps: Object with interpolator function 'int_2_alt'.

    Returns:
        np.ndarray: Maximum 64ms-averaged peak flux for each GRB.
                    Returns empty array if no GRBs.
    """
    # --- Setup ---
    t_peak      = m_prop_m['t_peak_c_z']  # Shape: (n_grbs,)
    n_grbs      = len(t_peak)

    # Extract other properties and ensure correct shape for broadcasting
    # Shapes need to be (1, n_grbs) to broadcast against time axis (n_bins, 1)
    alpha_e     = m_prop_m['alpha_e'][np.newaxis, :]     # Shape: (1, n_grbs)
    alpha_n     = m_prop_m['alpha_n'][np.newaxis, :]     # Shape: (1, n_grbs)
    E_p_obs     = m_prop_m["E_p_obs"][np.newaxis, :]     # Shape: (1, n_grbs)
    F_0         = m_prop_m["F_0"][np.newaxis, :]         # Shape: (1, n_grbs)

    n_bins      = 16
    bin_width   = 0.008     # 8ms in seconds
    window_size = 8         # Moving average window size (8 bins * 8ms/bin = 64ms)

    # Time offsets relative to t_peak (centers of the 16 bins)
    # Shape: (n_bins,)
    time_offsets = np.arange(-n_bins / 2 + 0.5, n_bins / 2 + 0.5) * bin_width

    # --- Calculate Fluxes at Bin Centers Vectorially ---

    # Calculate absolute time for each bin center for each GRB
    # Shape: (n_bins, n_grbs) via broadcasting (n_bins, 1) + (1, n_grbs)
    # Calculate time ratios t/t_peak.
    abs_times = time_offsets[:, np.newaxis] + t_peak[np.newaxis, :]
    time_ratios = abs_times / t_peak[np.newaxis, :]

    # Clip time_ratios slightly above zero for numerical stability in power laws
    # Especially if abs_times could become <= 0 for very early bins + small t_peak
    safe_time_ratios = np.maximum(time_ratios, 1e-12) # If tpeak < 64 ms this avoids division by zero

    # Calculate P_e(t) and P_n(t)
    P_e_mat = np.where(safe_time_ratios < 1.0, 1.0, np.power(safe_time_ratios, -alpha_e)) # P_e = 1 if t/tp < 1 else (t/tp)^(-alpha_e)
    P_n_mat = np.where(safe_time_ratios < 1.0, safe_time_ratios, np.power(safe_time_ratios, -alpha_n)) # P_n = t/tp if t/tp < 1 else (t/tp)^(-alpha_n)

    # Calculate E_p(t) = E_p_obs * P_e(t)
    E_p_t = E_p_obs * P_e_mat 

    # Calculate Flux term F_0(t) = F_0 * P_n(t)
    F_0_t = F_0 * P_n_mat   

    # Calculate flux at each time bin center
    flux_array_50_300   = F_0_t * interps.int_3_alt(E_p_t) * 6.2e8 # 50-300 keV range
    
    # --- Calculate 64ms Moving Average for 50-300 keV ---
    flux_cumsum_50_300          = np.cumsum(flux_array_50_300, axis=0)
    window_sums_50_300          = np.zeros((n_bins - window_size + 1, n_grbs))
    window_sums_50_300[0, :]    = flux_cumsum_50_300[window_size - 1, :]
    window_sums_50_300[1:, :]   = flux_cumsum_50_300[window_size:, :] - flux_cumsum_50_300[:-window_size, :]
    moving_averages_50_300      = window_sums_50_300 / window_size
    F64ms_50_300                = np.max(moving_averages_50_300, axis=0)

    return F64ms_50_300

@functools.lru_cache(maxsize=1) # only compute once
def get_time_factors(time_max=20, min_t = 0.1, max_t = 5):
    factor = min_t * ((max_t/min_t)**(1.0/(time_max-1)))**np.arange(time_max)
    return factor

def compute_time_evolution(m_prop_m: dict, interps: Interps) -> tuple:
    """
    Compute the time evolution of the fluence and determine the t₉₀ values.

    Parameters:
        m_prop_m : dict
            Dictionary containing macro properties for valid GRBs (e.g., 't_peak_c_z', 'theta_v',
            'R_E_theta_E_p_hat', 'N0').
        interps : Interps
            Interpolator object containing functions (e.g., int_1, int_3, alpha_e, alpha_n).
    
    Returns:
        t_90_array_out : np.ndarray
            Array of t₉₀ values after evolution.
        final_fluence : np.ndarray
            Final fluence values computed at the last time step.
    """
    # Retrieve the initial time values from the macro properties
    t_array             = m_prop_m['t_peak_c_z']

    # Define the evolution time grid, everything is a function of t/tp ie. only define the ratio and rescale later
    time_max            = TIME_RESOLUTION
    ratio_base          = get_time_factors(time_max, min_t = 0.0001, max_t = 1_000)
    max_m               = np.searchsorted(ratio_base, 1)

    P_e_mat_m           = np.empty((time_max, len(t_array)))
    P_n_mat_m           = np.empty((time_max, len(t_array)))
    P_e_mat_m[:max_m]   = 1 
    P_n_mat_m[:max_m]   = ratio_base[:max_m, None]  
    ratio_after_peak    = ratio_base[max_m:, None]
    np.power(ratio_after_peak, -m_prop_m['alpha_e'], out=P_e_mat_m[max_m:])
    np.power(ratio_after_peak, -m_prop_m['alpha_n'], out=P_n_mat_m[max_m:])

    # Compute common terms and fluence matrices
    E_p_obs             = m_prop_m["E_p_obs"] * P_e_mat_m
    fluence_terms_mat   = m_prop_m["F_0"] * P_n_mat_m * interps.int_4_alt(E_p_obs) # 50-300 keV BATSE range

    fluence_time_mat    = t_array * cumulative_trapezoid(fluence_terms_mat, x = ratio_base, axis=0, initial=0)
    total_fluence       = fluence_time_mat[-1]

    # Determine t₉₀ (time where fluence reaches 90% of the final fluence)
    t_90_indices        = np.argmax(fluence_time_mat >= 0.9 * total_fluence, axis=0) - 1
    t_90_array_out      = t_array * ratio_base[t_90_indices] 

    return t_90_array_out, total_fluence



def compute_time_evolution_(m_prop_m: dict, interps: Interps, 
                                      bin_width_ms: float = 16.0,
                                      n_bins: int = 200) -> tuple:
    """
    Compute the time evolution of the fluence using fixed time bins.
    
    This version uses fixed time intervals (e.g., 16ms) independent of t_peak,
    which is more consistent with how real detectors operate. For 200 bins this is about 3.2s. So safe for most GRBs.
    
    Parameters:
        m_prop_m : dict
            Dictionary containing macro properties for valid GRBs.
        interps : Interps
            Interpolator object containing spectral integration functions.
        bin_width_ms : float
            Width of each time bin in milliseconds (default: 16ms)
        n_bins : int
            Total number of time bins to compute (default: 200)
    
    Returns:
        t_90_array_out : np.ndarray
            Array of t₉₀ values.
        final_fluence : np.ndarray
            Final fluence values.
    """
    t_peak = m_prop_m['t_peak_c_z']
    n_grbs = len(t_peak)
    
    # Convert bin width to seconds
    bin_width = bin_width_ms / 1000.0
    
    # Create time grid: start from 0, go out to n_bins * bin_width
    # Use bin centers for evaluation
    time_edges = np.arange(n_bins + 1) * bin_width
    time_centers = (time_edges[:-1] + time_edges[1:]) / 2.0  # Shape: (n_bins,)
    
    # Broadcast time centers for all GRBs
    # Shape: (n_bins, n_grbs)
    time_grid = time_centers[:, np.newaxis]  # (n_bins, 1)
    t_peak_grid = t_peak[np.newaxis, :]      # (1, n_grbs)
    
    # Calculate time ratios t/t_peak
    time_ratios = time_grid / t_peak_grid
    safe_time_ratios = np.maximum(time_ratios, 1e-12)
    
    # Calculate P_e(t) and P_n(t) using the same prescription
    P_e_mat = np.where(safe_time_ratios < 1.0, 1.0, 
                       np.power(safe_time_ratios, -m_prop_m['alpha_e']))
    P_n_mat = np.where(safe_time_ratios < 1.0, safe_time_ratios, 
                       np.power(safe_time_ratios, -m_prop_m['alpha_n']))
    
    # Calculate evolving spectral properties
    E_p_obs_t = m_prop_m["E_p_obs"] * P_e_mat
    
    # Calculate fluence rate at each time bin (erg/cm²/s in 50-300 keV)
    fluence_rate = m_prop_m["F_0"] * P_n_mat * interps.int_4_alt(E_p_obs_t)
    
    # Integrate fluence over time using trapezoidal rule
    # Note: using time_centers as x-values
    fluence_time_mat = cumulative_trapezoid(
        fluence_rate, 
        x=time_centers, 
        axis=0, 
        initial=0
    )
    
    # Final fluence at last time bin
    total_fluence = fluence_time_mat[-1]
    
    # Find t₉₀: time when fluence reaches 90% of total
    # Need to handle edge cases where 90% is never reached
    threshold_fluence = 0.9 * total_fluence
    
    # Find first index where fluence >= 90% threshold
    above_threshold = fluence_time_mat >= threshold_fluence
    t_90_indices = np.argmax(above_threshold, axis=0)
    
    # Handle cases where threshold is never reached (use last bin)
    never_reached = ~np.any(above_threshold, axis=0)
    t_90_indices[never_reached] = n_bins - 1
    
    # Get actual t₉₀ times from the time grid
    t_90_array_out = time_centers[t_90_indices]
    
    return t_90_array_out, total_fluence




def compute_Fp_64_ms_sliding_window_2(m_prop_m: dict, interps: Interps) -> np.ndarray:
    """Calculate peak flux with optimized moving average using interpolators."""
    t_peak      = m_prop_m['t_peak_c_z']
    E_p_obs     = m_prop_m["E_p_obs"]
    theta_v     = m_prop_m["theta_v"]
    F_0         = m_prop_m["F_0"]

    # Prepare points for interpolation (theta_v, log10(E_p), log10(t_peak))
    # We use log10 for E_p and t_peak as the grid is likely logarithmic
    points = np.column_stack((
        theta_v, 
        np.log10(E_p_obs), 
        np.log10(t_peak)
    ))
    
    # Get correction factor from interpolator
    # The interpolator returns the peak flux factor (integral part * 6.2e8 * shape factor)
    flux_factor = interps.interp_pf(points)
    
    # F_p = F_0 * factor
    F64ms = F_0 * flux_factor
    
    return F64ms

def compute_time_evolution_2(m_prop_m: dict, interps: Interps) -> tuple:
    """
    Compute the time evolution of the fluence using interpolators.
    
    Parameters:
        m_prop_m : dict
            Dictionary containing macro properties for valid GRBs.
        interps : Interps
            Interpolator object containing spectral integration functions.
        bin_width_ms, n_bins: Ignored in this optimized version.
    
    Returns:
        t_90_array_out : np.ndarray
            Array of t₉₀ values.
        final_fluence : np.ndarray
            Final fluence values.
    """
    t_peak      = m_prop_m['t_peak_c_z']
    E_p_obs     = m_prop_m["E_p_obs"]
    theta_v     = m_prop_m["theta_v"]
    F_0         = m_prop_m["F_0"]
    
    # Prepare points for interpolation (theta_v, log10(E_p))
    points = np.column_stack((theta_v, np.log10(E_p_obs)))
    
    # Interpolate normalized t90 and fluence integral
    # t90_norm = t90 / t_peak
    # flu_norm = Fluence / (F_0 * t_peak)
    t90_norm = interps.interp_t90(points)
    flu_norm = interps.interp_flu(points)
    
    t_90_array_out = t90_norm * t_peak
    final_fluence  = flu_norm * F_0 * t_peak 
    
    return t_90_array_out, final_fluence

def calculate_isotropic_luminosity(
    detected_properties: dict, 
    interps: Interps
) -> tuple:
    """
    Calculates the k-correction, isotropic energy (E_iso), and isotropic 
    luminosity (L_iso) for detected GRBs, following the procedure in
    Poolakkil et al. 2021 (arXiv:2103.13528).

    Args:
        detected_properties (dict): A dictionary containing the properties of
            the final, detected GRB sample. Must include 'z_det', 'Ep_det',
            'f_det' (observed fluence), and 't_det' (T90).
        interps (Interps): An object containing the spectral model interpolator,
            specifically `interps.bpl_model`.

    Returns:
        tuple: A tuple containing:
            - E_iso (np.ndarray): The calculated isotropic energy in ergs.
            - L_iso (np.ndarray): The calculated apparent isotropic luminosity
                                  in ergs/s.
    """
    z           = detected_properties['z_det']
    one_plus_z  = 1 + z
    E_p_obs     = detected_properties['Ep_det']
    S_obs       = detected_properties['f_det']
    T90_obs     = detected_properties['t_det']

    # Define the integrand: E * N(E), where N(E) is the spectral model
    def integrand(E, E_p):
        return E * broken_power_law(E, E_p)

    # Numerator of k-correction: Bolometric fluence in rest frame
    # Energy band: 1 keV to 10 MeV (1e4 keV) in the rest frame
    E_p_rest = E_p_obs * one_plus_z
    S_bolo, _ = quad_vec(
        lambda E: integrand(E, E_p_rest),
        #a   =1,
        #b   =1e4
        a = 50,
        b = 300 # 50-300 keV for consistency with observed fluence
    )

    # Denominator of k-correction: Fluence in observer frame bandpass
    # Energy band: 10 keV to 1000 keV in the observer frame
    S_band, _ = quad_vec(
        lambda E: integrand(E, E_p_obs),
        a=10,
        b=1000
    )

    # Calculate the k-correction factor (Eq. 5)
    # Add a small epsilon to the denominator to avoid division by zero
    k_corr = S_bolo / (S_band + 1e-99)

    # Calculate Isotropic Energy (E_iso) (Eq. 6)
    D_L     = d_l(z)
    E_iso   = (4 * np.pi * D_L**2 / one_plus_z) * S_obs * k_corr

    # Calculate Apparent Isotropic Luminosity (L_iso)
    # L_iso = E_iso / T90_rest
    T90_rest    = T90_obs / one_plus_z
    L_iso       = E_iso / (T90_rest + 1e-99)

    return E_iso, L_iso

def make_observations_with_iso(
        thetas, 
        params      : SimParams, 
        interps     : Interps, 
        limits      : Dict[str, Any] = DEFAULT_LIMITS, 
        n_events    : int = N_SIMS
    ):
    """
    Wrapper around make_observations to also compute E_iso and L_iso.

    Returns:
        dict: Dictionary containing observables and isotropic properties.
    """
    obs = make_observations(
        thetas, 
        params, 
        interps,
        limits = limits,
        n_events = n_events
    )

    if obs is None:
        return None

    E_iso_det, L_iso_det = calculate_isotropic_luminosity(obs, interps)

    obs["E_iso_det"] = E_iso_det
    obs["L_iso_det"] = L_iso_det

    return obs


def generate_grb_population(
        thetas: list,
        params: SimParams,
        interps: Interps,
        limits: Dict[str, Any] = DEFAULT_LIMITS,
        n_events: int = N_SIMS,
        seed: int = None
    ) -> dict:
    """
    Generate a synthetic GRB population for posterior predictive checks.
    
    This function generates GRB observables from given parameters, useful for:
    - Posterior predictive checks (PPC)
    - Parameter recovery tests
    - Forward modeling
    
    Parameters:
        thetas : list
            Model parameters [k, L_L0, L_mu_E, sigma_E, L_mu_tau, sigma_tau, f_j]
        params : SimParams
            Simulation parameters containing cosmology and jet structure
        interps : Interps
            Interpolators for spectral calculations
        limits : Dict[str, Any]
            Detection limits dictionary with keys:
            - F_LIM: Peak flux limit (ph/cm²/s)
            - T90_LIM: Duration limit (s)
            - EP_LIM_UPPER/LOWER: Peak energy limits (keV)
        n_events : int
            Number of GRB events to simulate
        seed : int, optional
            Random seed for reproducibility. If None, uses existing RNG state.
    
    Returns:
        dict: Dictionary containing synthetic observables:
            - 't90': Duration array
            - 'epeak': Peak energy array
            - 'pflux': Peak flux array
            - 'fluence': Fluence array
            - 'z_det': Redshift of detected events
            - 'theta_det': Viewing angle of detected events
            - 'n_detected': Number of detected events
            - 'isotropic_energy_det': Isotropic energy of detected events
        Returns None if too few events pass detection cuts.
    """
    # Set random seed if provided
    if seed is not None:
        params.rng = np.random.default_rng(seed)
    
    # Generate observations using existing machinery
    obs = make_observations(
        thetas,
        params,
        interps,
        limits=limits,
        n_events=n_events
    )
    
    if obs is None:
        return None
    
    # Reformat output for easier use in PPC
    return {
        't90'                   : obs['t_det'],
        'epeak'                 : obs['Ep_det'],
        'pflux'                 : obs['Fp_det'],
        'fluence'               : obs['f_det'],
        'z_det'                 : obs['z_det'],
        'theta_det'             : obs['theta_v_det'],
        'n_detected'            : obs['triggered_events'],
        'isotropic_energy_det'  : obs['isotropic_energy_det'],
    }