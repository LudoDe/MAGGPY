
from math       import  inf
from typing     import  Dict
import          numpy   as np

DEFAULT_PRIOR_BOUNDS = {
    'k': (1.5, 6),
    'L_L0': (-3, 2),
    'L_mu_E': (2.5, 5),
    'sigma_E': (0, 1),
    'L_mu_tau': (-2, 2.5),
    'sigma_tau': (0, 1.8),
    'f_j': (0, 1)
}

def create_log_prior(bounds: Dict[str, tuple] = DEFAULT_PRIOR_BOUNDS) -> callable:
    """
    Factory function to create a log_prior function with hardcoded bounds.
    
    Parameters:
        bounds: Dictionary mapping parameter names to (min, max) tuples
                Expected keys: 'k_inv', 'L_L0', 'L_mu_E', 'sigma_E', 
                              'L_mu_tau', 'sigma_tau', 'f_j'
    
    Returns:
        Compiled log_prior function with hardcoded bounds
    
    Example:
        >>> custom_bounds = {
        ...     'k': (1.5, 5),
        ...     'L_L0': (-3, 2),
        ...     'L_mu_E': (2.5, 5),
        ...     'sigma_E': (0, 1),
        ...     'L_mu_tau': (-2, 2.5),
        ...     'sigma_tau': (0, 1.8),
        ...     'f_j': (0, 1)
        ... }
        >>> my_prior = create_log_prior(custom_bounds)
    """
    # Extract bounds
    k_min, k_max                    = bounds['k']
    L_L0_min, L_L0_max              = bounds['L_L0']
    L_mu_E_min, L_mu_E_max          = bounds['L_mu_E']
    sigma_E_min, sigma_E_max        = bounds['sigma_E']
    L_mu_tau_min, L_mu_tau_max      = bounds['L_mu_tau']
    sigma_tau_min, sigma_tau_max    = bounds['sigma_tau']
    f_j_min, f_j_max                = bounds['f_j']
    
    # Create the function with hardcoded bounds
    def log_prior_custom(thetas):
        k, L_L0, L_mu_E, sigma_E, L_mu_tau, sigma_tau, f_j = thetas
        
        if not (k_min < k < k_max):
            return -inf
        if not (L_L0_min < L_L0 < L_L0_max):
            return -inf
        if not (L_mu_E_min < L_mu_E < L_mu_E_max):
            return -inf
        if not (sigma_E_min < sigma_E < sigma_E_max):
            return -inf
        if not (L_mu_tau_min < L_mu_tau < L_mu_tau_max):
            return -inf
        if not (sigma_tau_min < sigma_tau < sigma_tau_max):
            return -inf
        if not (f_j_min < f_j < f_j_max):
            return -inf
        
        return 0
    
    return log_prior_custom

def initialize_walkers(n_walkers, seed = 40, max_fj = 1.0, fj_bounds = None):  
    np.random.seed(seed)  # Set seed for reproducibility

    k       = np.random.uniform(2, 4, n_walkers)      # Moderate power law index
    L_L0    = np.random.uniform(-1.5, 0, n_walkers)     # Higher luminosity
    L_mu_E  = np.random.uniform(3, 3.5, n_walkers)      # Reasonable peak energy
    sigma_E = np.random.uniform(0.2, 0.4, n_walkers)    # Moderate scatter
    L_mu_t  = np.random.uniform(-1, 0, n_walkers)       # Reasonable peak time
    sigma_t = np.random.uniform(0.5, 1, n_walkers)      # Moderate scatter in time  
    #f_j     = np.random.uniform(0.4, max_fj, n_walkers)    # Reasonable fraction
    fj_bounds_considered = (0.4, max_fj) if fj_bounds is None else fj_bounds
    f_j     = np.random.uniform(fj_bounds_considered[0], fj_bounds_considered[1], n_walkers)  # Reasonable fraction
    walkers = np.column_stack((k, L_L0, L_mu_E, sigma_E, L_mu_t, sigma_t, f_j))  # Combine into a 2D array

    return np.array(walkers)