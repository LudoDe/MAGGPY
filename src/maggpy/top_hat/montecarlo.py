"""
Top-Hat GRB Monte Carlo likelihood functions for MCMC inference.
"""

import  numpy as np
import emcee
import multiprocessing
from scipy.integrate    import quad
from scipy.interpolate  import RectBivariateSpline
from ..spectral_models  import broken_power_law
from ..utils            import luminosity_gen
ncpu = multiprocessing.cpu_count()

# =============================================================================
# K-Factor Computation
# =============================================================================

N_PARAMS = 6

def create_k_interpolator(E_p_range=(50, 10_000), z_range=(0, 14)):
    """
    Create a k-factor interpolator for redshift correction.
    
    Parameters:
    -----------
    params : dict
        Must contain 'alpha', 'beta_s', 'n' keys
    """

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

def montecarlo(thetas, n_events, bns_data, rng=np.random.default_rng(42)):
    """
    Core Monte Carlo simulation for GRB observables.
    
    Parameters:
    -----------
    thetas : array-like
        [inv_A_index, L_L0, L_mu_E_10, sigma_E_10, ...]
        Additional parameters are ignored (flexibility for different models)
    """
    A_index, L_L0, L_mu_E_10, sigma_E_10 = thetas[:4]

    l_10        = np.log(10)
    L_obs_iso   = luminosity_gen(A_index, n_events, rng=rng) * 10**(L_L0 + 49)
    E_p_rest    = rng.lognormal(mean=L_mu_E_10 * l_10, sigma=sigma_E_10 * l_10, size=n_events)

    id_z        = rng.integers(low=0, high=len(bns_data.z_arr), size=n_events)
    z_arr       = bns_data.z_arr[id_z]
    d_L_sq      = bns_data.distances[id_z]**2
    E_p_obs     = E_p_rest / (1 + z_arr)

    k_corr      = bns_data.k_interpolator.ev(np.log10(E_p_obs), z_arr)
    p_flux      = L_obs_iso / (4 * np.pi * d_L_sq * k_corr) * 6.242e8

    return {
        "p_flux"    : p_flux,
        "E_p_obs"   : E_p_obs,
        "z_arr"     : z_arr,
        "L_p_obs"   : L_obs_iso,
    }

def apply_detection_cuts(p_flux, E_p_obs, pflux_min=4, epeak_range=(50, 10_000)):
    """Apply standard detection cuts."""
    trigger_mask = p_flux > pflux_min
    analysis_mask = (
        (p_flux > pflux_min) &
        (E_p_obs > epeak_range[0]) &
        (E_p_obs < epeak_range[1])
    )
    return trigger_mask, analysis_mask

def create_move_strategy():
    """Create emcee move strategy with stretch and DE-snooker moves."""
    return [
        (emcee.moves.StretchMove(), 0.70),
        (emcee.moves.DESnookerMove(), 0.30),
    ]

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

def start_mcmc(log_probability_func, initialize_walkers_func, n_iterations, n_walkers, backend_fn, progress=True):

    initial_pos, n_steps_remaining, backend = check_and_resume_mcmc(
        filename                = backend_fn,
        n_steps                 = n_iterations,
        initialize_walkers_func = initialize_walkers_func,
        n_walkers               = n_walkers
    )

    moves = create_move_strategy()
    
    blobs_dtype = [
        ("l_pflux", float), 
        ("l_epeak", float), 
        ("l_poiss", float), 
        ("l_eff", float)
    ]

    sampler = emcee.EnsembleSampler(
        n_walkers,
        N_PARAMS,
        log_probability_func,
        blobs_dtype=blobs_dtype,
        backend=backend,
        moves=moves
    )

    sampler.run_mcmc(initial_pos, n_steps_remaining, progress=progress)
    
    return sampler