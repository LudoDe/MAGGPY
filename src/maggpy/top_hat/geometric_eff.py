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

def geometric_efficiency_flat_midpoint(theta_c_mid, half_width=6):
    """
    Average geometric efficiency for a flat (uniform) theta_c distribution
    given the midpoint and a half-width.
    
    The distribution is [theta_c_mid - half_width, theta_c_mid + half_width].
    For half_width=6, this yields a ~12 degree total width.
    
    Parameters
    ----------
    theta_c_mid : float
        Midpoint of the core angle distribution in degrees.
    half_width : float
        Half-width of the flat distribution in degrees.
        
    Returns
    -------
    float
        Average beaming fraction.
    """
    theta_c_max = theta_c_mid + half_width
    #theta_c_min = max(1e-3, theta_c_mid - half_width) # Avoid zero or negative angles
    #vectorized version of max to handle arrays and single valued inputs
    theta_c_min = np.maximum(1e-3, theta_c_mid - half_width)

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

# =============================================================================
# functions to generate the angles when needed
# =============================================================================
def generate_theta_c_universal(theta_c_med, n_samples=1000):
    """
    Generate theta_c samples from a universal distribution 
    
    Inputs:
    - theta_c_med: median of the log-normal distribution (in degrees)
    - n_samples: number of samples to generate
    
    """
    return np.ones(n_samples) * theta_c_med

def generate_theta_c_flat(theta_c_max, theta_c_min=1.0, n_samples=1000):
    """
    Generate theta_c samples from a flat distribution between theta_c_min and theta_c_max.
    
    Inputs:
    - theta_c_max: maximum core angle (in degrees)
    - theta_c_min: minimum core angle (in degrees, default: 1.0)
    - n_samples: number of samples to generate
    """
    return np.random.uniform(theta_c_min, theta_c_max, n_samples)

def generate_theta_c_flat_midpoint(theta_c_mid, half_width=6, n_samples=1000):
    """
    Generate theta_c samples from a flat distribution defined by a midpoint and half-width.
    
    The distribution is [theta_c_mid - half_width, theta_c_mid + half_width].
    
    Inputs:
    - theta_c_mid: midpoint of the core angle distribution (in degrees)
    - half_width: half-width of the flat distribution (in degrees, default: 6)
    - n_samples: number of samples to generate
    """
    theta_c_min = max(1e-3, theta_c_mid - half_width) # Avoid zero or negative angles
    theta_c_max = theta_c_mid + half_width
    return np.random.uniform(theta_c_min, theta_c_max, n_samples)

def generate_theta_c_lognormal(theta_c_med, sigma_theta_c=0.5, minimum_theta_c=1.0, maximum_theta_c=45, n_samples=1000):
    """
    Generate theta_c samples from a log-normal distribution.
    
    Inputs:
    - theta_c_med: median of the log-normal distribution (in degrees)
    - sigma_theta_c: width of the log-normal in log10 space (default: 0.5)
    - minimum_theta_c: minimum core angle (in degrees, default: 1.0)
    - maximum_theta_c: maximum core angle (in degrees, default: 45.0)
    - n_samples: number of samples to generate
    """
    mu      = np.log(theta_c_med)
    sigma   = sigma_theta_c * np.log(10)
    
    shape   = sigma
    scale   = np.exp(mu)

    cdf_min = lognorm.cdf(minimum_theta_c, s=shape, scale=scale)
    cdf_max = lognorm.cdf(maximum_theta_c, s=shape, scale=scale)
    norm    = cdf_max - cdf_min
    if norm <= 1e-9:
        return np.ones(n_samples) * minimum_theta_c
    
    uniform_samples     = np.random.uniform(cdf_min, cdf_max, n_samples)
    samples             = lognorm.ppf(uniform_samples, s=shape, scale=scale)
    return samples