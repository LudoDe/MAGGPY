"""
Top-Hat GRB Monte Carlo likelihood functions for MCMC inference.
"""

import emcee
import numpy            as np
import scipy.special    as sc
from math               import inf
from scipy.integrate    import quad
from dataclasses 		import dataclass
from typing             import Optional, Callable
from astropy.cosmology  import Planck18, FlatLambdaCDM
from scipy.stats        import gengamma, cramervonmises_2samp, lognorm
import  multiprocessing
from    multiprocessing import Pool
ncpu = multiprocessing.cpu_count()

@dataclass
class SimParams:
    epeak_data              : np.ndarray
    duration_data           : np.ndarray
    pflux_data              : np.ndarray
    fluence_data            : np.ndarray
    yearly_rate             : float         # expected yearly rate of GRBs
    triggered_years         : float           # number of years with triggered events

    rng                     : np.random.Generator 
    # K-correction interpolator for spectral corrections
    k_interpolator          : Callable
    
    # MRD-related fields
    z_arr                   : np.ndarray
    P_z_interp              : Optional[Callable]    = None
    z_grid                  : Optional[np.ndarray]  = None
    P_z_density             : Optional[np.ndarray]  = None
    total_merger_rate       : Optional[float]       = None
    local_rate              : Optional[float]       = None  # R_0 at z=0 in Gpc^-3 yr^-1

    # Properties to set in __post_init__ 
    distances               : float = None  
    
    def __post_init__(self): self.distances = compute_luminosity_distance(self.z_arr) 

# =============================================================================
# Core Monte Carlo Functions
# =============================================================================

def luminosity_gen(A, n, rng=None):
    """Generate luminosities from modified Schechter distribution (Salafia et al 2024)."""
    return gengamma.rvs(a=(A - 1)/A, c=-A, size=n, random_state=rng)

def compute_luminosity_distance(z, cosmology=None):
    """Compute luminosity distance in cm."""
    if cosmology is None: cosmology = FlatLambdaCDM(H0=Planck18.H0, Om0=Planck18.Om0)
    return cosmology.luminosity_distance(z).cgs.value

def simplified_montecarlo(thetas, n_events, params_in, rng=None):
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

    id_z    = rng.integers(low=0, high=len(params_in.z_arr), size=n_events)
    z_arr   = params_in.z_arr[id_z]
    d_L_sq  = params_in.distances[id_z]**2
    E_p_obs = E_p_rest / (1 + z_arr)

    k_corr  = params_in.k_interpolator.ev(np.log10(E_p_obs), z_arr)
    p_flux  = L_obs_iso / (4 * np.pi * d_L_sq * k_corr) * 6.242e8

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
# MCMC Utilities
# =============================================================================

def create_move_strategy():
    """Create emcee move strategy with stretch and DE-snooker moves."""
    return [
        (emcee.moves.StretchMove(), 0.70),
        (emcee.moves.DESnookerMove(), 0.30),
    ]


def create_log_probability_function(log_prior_func, log_likelihood_func, params_in):
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

    def log_probability(thetas):
        lp = log_prior_func(thetas)	
        
        if not np.isfinite(lp):
            return -inf, 0, 0, 0, 0
        
        l_out = log_likelihood_func(thetas, params_in)
        
        if not np.isfinite(l_out[0]):
            return -inf, 0, 0, 0, 0
        
        return lp + l_out[0], l_out[1], l_out[2], l_out[3], l_out[4]
    
    return log_probability


def run_mcmc(log_probability_func, initial_walkers, n_iterations,
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
    if moves is None: moves = create_move_strategy()
    
    if blobs_dtype is None:
        blobs_dtype = [
            ("l_pflux", float), 
            ("l_epeak", float), 
            ("l_poiss", float), 
            ("l_eff", float)
        ]
    n_params    = initial_walkers.shape[1]
    n_walkers   = initial_walkers.shape[0]

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