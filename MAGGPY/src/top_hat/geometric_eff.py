import numpy as np
from scipy.stats import lognorm
from scipy.integrate import quad
from scipy.interpolate import interp1d


# =============================================================================
# Fixed Theta_c Model (Single top-hat jet)
# =============================================================================

def geometric_efficiency_fixed(theta_c_deg):
    """
    Beaming fraction for a single top-hat jet with fixed opening angle.
    
    f_b = 1 - cos(theta_c)
    
    Parameters
    ----------
    theta_c_deg : float or array
        Core angle in degrees.
        
    Returns
    -------
    float or array
        Beaming fraction (probability of detection).
    """
    theta_c_rad = np.deg2rad(theta_c_deg)
    return 1.0 - np.cos(theta_c_rad)


# =============================================================================
# Flat Theta_c Distribution (uniform between theta_min and theta_max)
# =============================================================================

def geometric_efficiency_flat(theta_c_max, theta_c_min=1):
    """
    Average geometric efficiency for a flat (uniform) theta_c distribution.
    
    Integrates f_b(theta) = 1 - cos(theta) over uniform distribution.
    
    Parameters
    ----------
    theta_c_max : float
        Maximum core angle in degrees (this is your MCMC parameter).
    theta_c_min : float
        Minimum core angle in degrees (default: 1°).
        
    Returns
    -------
    float
        Average beaming fraction.
    """
    theta_max_rad = np.deg2rad(theta_c_max)
    theta_min_rad = np.deg2rad(theta_c_min)
    sin_max = np.sin(theta_max_rad)
    sin_min = np.sin(theta_min_rad)
    return 1 - (sin_max - sin_min) / (theta_max_rad - theta_min_rad)

# =============================================================================
# Log-Normal Theta_c Distribution
# =============================================================================

def _calculate_geometric_efficiency_lognormal_raw(theta_c_med, sigma_theta_c=0.5, minimum_theta_c=1.0, maximum_theta_c=45.0):
    """
    Raw calculation of geometric efficiency for log-normal theta_c distribution.
    
    Use the interpolator version in MCMC for speed.
    
    Parameters
    ----------
    theta_c_med : float
        Median of the log-normal distribution in degrees.
    sigma_theta_c : float
        Width parameter (in log10 space, default: 0.5).
    """
    mu = np.log(theta_c_med)
    sigma = sigma_theta_c * np.log(10)
    
    shape = sigma
    scale = np.exp(mu)

    theta_c_min = minimum_theta_c
    theta_c_max = 45.0

    cdf_max = lognorm.cdf(theta_c_max, s=shape, scale=scale)
    cdf_min = lognorm.cdf(theta_c_min, s=shape, scale=scale)
    norm = cdf_max - cdf_min
    
    if norm <= 1e-9:
        return 0.0

    def integrand(theta_c_deg):
        theta_c_rad = np.deg2rad(theta_c_deg)
        detection_prob = 1.0 - np.cos(theta_c_rad)
        pdf = lognorm.pdf(theta_c_deg, s=shape, scale=scale)
        return detection_prob * pdf

    geometric_eff_raw, _ = quad(integrand, theta_c_min, theta_c_max, epsabs=1e-8, epsrel=1e-8)
    return geometric_eff_raw / norm


def create_geometric_efficiency_lognormal_interpolator(sigma_theta_c=0.5, n_points=200, minimum_theta_c=1.0, maximum_theta_c=25.0):
    """
    Create an interpolator for log-normal geometric efficiency.
    
    Call this ONCE before MCMC, then use the returned function.
    
    Parameters
    ----------
    sigma_theta_c : float
        Width of log-normal in log10 space (default: 0.5).
    n_points : int
        Number of grid points for interpolation.
        
    Returns
    -------
    callable
        Interpolator function: f(theta_c_med) -> geometric_efficiency
    """
    theta_c_med_grid = np.linspace(minimum_theta_c, maximum_theta_c, n_points)
    
    efficiencies = np.array([
        _calculate_geometric_efficiency_lognormal_raw(t, sigma_theta_c, minimum_theta_c = minimum_theta_c)
        for t in theta_c_med_grid
    ])
    
    return interp1d(
        theta_c_med_grid, efficiencies, kind='cubic',
        bounds_error=False, fill_value=(efficiencies[0], efficiencies[-1])
    )