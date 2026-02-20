"""
Top-Hat GRB Monte Carlo likelihood functions for MCMC inference.
"""

import numpy as np
import scipy.special as sc
from scipy.stats import gengamma, cramervonmises_2samp, lognorm
from scipy.integrate import quad
from scipy.interpolate import RectBivariateSpline
from astropy.cosmology import Planck18, FlatLambdaCDM
from math import inf
import emcee
import multiprocessing
from multiprocessing import Pool
ncpu = multiprocessing.cpu_count()

# =============================================================================
# K-Factor Computation
# =============================================================================
DEFAULT_SPECTRAL_PARAMS = {
    "alpha"     : -0.67,    # 2/3 from synchrotron
    "beta_s"    : -2.59,    # Average value from GRBs
    "n"         : 2,        # Smoothly broken power law curvature
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


# =============================================================================
# Core Monte Carlo Functions
# =============================================================================

def luminosity_gen(A, n, rng=None):
    """Generate luminosities from modified Schechter distribution (Salafia et al 2024)."""
    return gengamma.rvs(a=(A - 1)/A, c=-A, size=n, random_state=rng)

def compute_luminosity_distance(z, cosmology=None):
    """Compute luminosity distance in cm."""
    if cosmology is None:
        cosmology = FlatLambdaCDM(H0=Planck18.H0, Om0=Planck18.Om0)
    return cosmology.luminosity_distance(z).cgs.value

def simplified_montecarlo(thetas, n_events, params_in, distances, k_interpolator, rng=None):
    """
    Core Monte Carlo simulation for GRB observables.
    
    Parameters:
    -----------
    thetas : array-like
        [inv_A_index, L_L0, L_mu_E_10, sigma_E_10, ...]
        Additional parameters are ignored (flexibility for different models)
    """
    A_index, L_L0, L_mu_E_10, sigma_E_10 = thetas[:4]

    if rng is None:
        rng = params_in.rng

    l_10 = np.log(10)
    
    L_obs_iso = luminosity_gen(A_index, n_events, rng=rng) * 10**(L_L0 + 49)
    E_p_rest = rng.lognormal(mean=L_mu_E_10 * l_10, sigma=sigma_E_10 * l_10, size=n_events)

    id_z = rng.integers(low=0, high=len(params_in.z_arr), size=n_events)
    z_arr = params_in.z_arr[id_z]
    d_L_sq = distances[id_z]**2
    E_p_obs = E_p_rest / (1 + z_arr)

    k_corr = k_interpolator.ev(np.log10(E_p_obs), z_arr)
    p_flux = L_obs_iso / (4 * np.pi * d_L_sq * k_corr) * 6.242e8

    return {
        "p_flux": p_flux,
        "E_p_obs": E_p_obs,
        "z_arr": z_arr,
        "L_p_obs": L_obs_iso,
    }

# =============================================================================
# Likelihood Utilities
# =============================================================================

def poiss_log(k, mu):
    """Log Poisson probability (numerically stable)."""
    return -mu + k * np.log(mu) - sc.gammaln(k + 1)


def cdf_sample(data, n, rng):
    """Inverse transform sampling from empirical CDF."""
    x_sorted = np.sort(data)
    u = rng.uniform(0, 1, n)
    return np.interp(u, np.linspace(0, 1, len(data)), x_sorted)


def score_func_cvm(y_sim, y_obs, rng):
    """Cramér-von Mises score function."""
    y_resample  = cdf_sample(y_sim, len(y_obs), rng=rng)
    y_in        = np.log10(y_resample)
    y_out       = np.log10(y_obs)
    return np.log(cramervonmises_2samp(y_in, y_out).pvalue)


def binned_likelihood(simulation_data, observed_data, n_quantiles=10):
    """Binned Poisson likelihood."""
    bins_obs = np.quantile(observed_data, np.linspace(0, 1, n_quantiles + 1))
    hist_sim, _ = np.histogram(simulation_data, bins=bins_obs)
    hist_obs, _ = np.histogram(observed_data, bins=bins_obs)
    hist_sim = hist_sim + 1e-10
    hist_norm = hist_sim * (len(observed_data) / len(simulation_data))
    return np.sum(poiss_log(hist_obs, hist_norm))


def apply_detection_cuts(p_flux, E_p_obs, pflux_min=4, epeak_range=(50, 10_000)):
    """Apply standard detection cuts."""
    trigger_mask = p_flux > pflux_min
    analysis_mask = (
        (p_flux > pflux_min) &
        (E_p_obs > epeak_range[0]) &
        (E_p_obs < epeak_range[1])
    )
    return trigger_mask, analysis_mask

# =============================================================================
# Geometric Efficiency Functions
# =============================================================================

def calculate_geometric_efficiency_flat(theta_c_max, theta_c_min=1):
    """
    Geometric efficiency for flat theta_c distribution from 0.1° to theta_c_max.
    
    Parameters
    ----------
    theta_c_max : float
        Maximum core angle in degrees.
    """
    rad_max = np.deg2rad(theta_c_max)
    rad_min = np.deg2rad(theta_c_min)

    term_1  = (rad_max - np.sin(rad_max))
    term_2  = (rad_min - np.sin(rad_min))
    norm    = rad_max - rad_min

    return (term_1 - term_2) / norm

from scipy.interpolate import interp1d
def create_geometric_efficiency_lognormal_interpolator(sigma_theta_c=0.5, n_points=200):
    """
    Create an interpolator for lognormal geometric efficiency.
    
    Call this once before MCMC, then use the returned function.
    """
    theta_c_med_10_grid = np.linspace(-1, np.log10(25), n_points)
    
    efficiencies = np.array([
        _calculate_geometric_efficiency_lognormal_raw(t, sigma_theta_c) 
        for t in theta_c_med_10_grid
    ])
    
    return interp1d(theta_c_med_10_grid, efficiencies, kind='cubic', 
                    bounds_error=False, fill_value=(efficiencies[0], efficiencies[-1]))

def _calculate_geometric_efficiency_lognormal_raw(theta_c_med_10, sigma_theta_c=0.5):
    """Raw calculation - use interpolator version in MCMC."""
    mu      = theta_c_med_10 * np.log(10)
    sigma   = sigma_theta_c * np.log(10)
    
    shape = sigma
    scale = np.exp(mu)


    theta_c_min = 1
    theta_c_max = 90

    cdf_max = lognorm.cdf(theta_c_max, s=shape, scale=scale)
    cdf_min = lognorm.cdf(theta_c_min, s=shape, scale=scale)
    norm    = cdf_max - cdf_min
    if norm <= 1e-9:
        return 0.0

    def integrand(theta_c_deg):
        # theta_c_deg in degrees, between 1 and 90
        theta_c_rad = np.deg2rad(theta_c_deg)
        detection_prob = 1.0 - np.cos(theta_c_rad)
        # use scipy's lognorm.pdf for numerical stability
        pdf = lognorm.pdf(theta_c_deg, s=shape, scale=scale)
        return detection_prob * pdf

    # integrate ordinary lognormal from small positive to 90 deg
    geometric_eff_raw, _ = quad(integrand, theta_c_min, theta_c_max, epsabs=1e-8, epsrel=1e-8)
    # divide by CDF(90) to get truncated expectation
    return geometric_eff_raw / norm

# =============================================================================
# MCMC Utilities
# =============================================================================

def create_move_strategy():
    """Create emcee move strategy with stretch and DE-snooker moves."""
    return [
        (emcee.moves.StretchMove(), 0.70),
        (emcee.moves.DESnookerMove(), 0.30),
    ]


def create_log_probability_function(log_prior_func, log_likelihood_func, params_in, k_interpolator, distances = None):
    """
    Create a log probability function for MCMC.
    
    Parameters
    ----------
    log_prior_func : callable
        Log prior function.
    log_likelihood_func : callable
        Log likelihood function.
    params_in : object
        Input parameters for Monte Carlo.
    distances : array-like
        Precomputed luminosity distances.
    k_interpolator : callable
        K-factor interpolator.
    n_events : int
        Number of Monte Carlo events.   
    Returns
    -------
    callable
        Log probability function.
    """
    distances = compute_luminosity_distance(params_in.z_arr)

    def log_probability(thetas):
        lp = log_prior_func(thetas)	
        
        if not np.isfinite(lp):
            return -inf, 0, 0, 0, 0
        
        l_out = log_likelihood_func(thetas, params_in, distances, k_interpolator)
        
        if not np.isfinite(l_out[0]):
            return -inf, 0, 0, 0, 0
        
        return lp + l_out[0], l_out[1], l_out[2], l_out[3], l_out[4]
    
    return log_probability


def run_mcmc(log_probability_func, initial_walkers, n_iterations, n_walkers, n_params,
             backend, blobs_dtype=None, moves=None, pool=None, progress=True):
    """
    Run emcee MCMC sampler.
    
    Parameters
    ----------
    log_probability_func : callable
        Log probability function.
    initial_walkers : array-like
        Initial walker positions, shape (n_walkers, n_params).
    n_iterations : int
        Number of MCMC steps.
    n_walkers : int
        Number of walkers.
    n_params : int
        Number of parameters.
    backend : emcee.backends.Backend
        Backend for storing results.
    blobs_dtype : list, optional
        Dtype specification for blobs.
    moves : list, optional
        Move strategy.
    pool : multiprocessing.Pool, optional
        Pool for parallel execution.
    progress : bool
        Show progress bar.
        
    Returns
    -------
    emcee.EnsembleSampler
        The sampler after running.
    """
    if moves is None:
        moves = create_move_strategy()
    
    if blobs_dtype is None:
        blobs_dtype = [
            ("l_pflux", float), 
            ("l_epeak", float), 
            ("l_poiss", float), 
            ("l_eff", float)
        ]
    #with Pool(processes=ncpu) as pool:
    sampler = emcee.EnsembleSampler(
        n_walkers,
        n_params,
        log_probability_func,
        pool=pool,
        blobs_dtype=blobs_dtype,
        backend=backend,
        moves=moves
    )

    sampler.run_mcmc(initial_walkers, n_iterations, progress=progress)
    
    return sampler


def check_and_resume_mcmc(filename, n_steps, initialize_walkers_func, n_walkers):
    backend = emcee.backends.HDFBackend(filename)

    # invert the logic for more readability
    if not filename.exists():
        initial_walkers = initialize_walkers_func(n_walkers)
        print("Starting new run")
        return initial_walkers, n_steps, backend
    
    initial_walkers = backend.get_last_sample()
    if backend.iteration >= n_steps:
        print("Already completed this run")
        return initial_walkers, 0, backend
    
    n_iterations = n_steps - backend.iteration
    print(f"Continuing from iteration {backend.iteration}")
    return initial_walkers, n_iterations, backend