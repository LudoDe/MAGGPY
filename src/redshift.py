# filepath: 
import numpy as np
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM, Planck18
from scipy.interpolate import interp1d
from pathlib import Path
from typing import Callable, Tuple, Optional

# using older numpy version so trapezoid is sometimes trapz make a quick check
if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz

def get_mrd_redshift_distribution(
    datafiles: Path,
    population: str = "fiducial_Hrad",
    alpha: str = "A1.0",
    component: str = "BNSs",
    sigma: float = 0.1,
    filename: str = "MRD_spread_15Z_40_No_MandF2017_0.1_No_No_0.dat"
) -> Tuple[Callable, float, float, np.ndarray, np.ndarray]:
    """
    Load MRD (Merger Rate Density) data and compute redshift distribution.
    
    Parameters:
        datafiles: Path to datafiles directory
        population: Population model name (e.g., 'fiducial_Hrad')
        alpha: Alpha parameter (e.g., 'A1.0')
        component: Component type ('BNSs', 'BBHs', etc.)
        sigma: Sigma value for the MRD spread
        filename: MRD data filename
    
    Returns:
        P_z_interp: Interpolator for P(z) probability density
        total_rate: Total merger rate per year (integrated over all z)
        local_rate: Local merger rate R_0 at z=0 in Gpc^-3 yr^-1
        z_grid: Redshift grid
        P_z_density: P(z) density values
    """
    # Build path to MRD file
    mrd_path = (datafiles / "populations" / "MRD" / f"output_sigma{sigma}" / 
                population / alpha / component / filename)
    
    if not mrd_path.exists():
        raise FileNotFoundError(f"MRD file not found: {mrd_path}")
    
    # Load MRD data: z and R(z) in Gpc^-3 yr^-1
    z_mrd, mrd = np.loadtxt(mrd_path, unpack=True)
    
    # Get local rate R_0 at z=0 (interpolate to z=0)
    mrd_interp = interp1d(z_mrd, mrd, kind='linear', bounds_error=False, fill_value='extrapolate')
    local_rate = float(mrd_interp(0.0))  # R_0 in Gpc^-3 yr^-1
    
    # Compute cosmological quantities
    cosmology = FlatLambdaCDM(H0=Planck18.H0, Om0=Planck18.Om0)
    dVc_dz = cosmology.differential_comoving_volume(z_mrd).to(u.Gpc**3 / u.sr).value * 4 * np.pi
    
    # dN/dz = R(z) * dV_c/dz / (1+z)  [the (1+z) accounts for time dilation]
    dN_dz = mrd * dVc_dz / (1 + z_mrd)
    
    # Total rate: integrate dN/dz over z
    total_rate = float(np.trapezoid(dN_dz, z_mrd))
    
    # Probability density P(z) = dN/dz / total_rate
    P_z_density = dN_dz / total_rate
    P_z_interp = interp1d(z_mrd, P_z_density, kind='linear', bounds_error=False, fill_value=0.0)
    
    return P_z_interp, total_rate, local_rate, z_mrd, P_z_density

def sample_from_mrd(
    P_z_interp: Callable,
    z_grid: np.ndarray,
    P_z_density: np.ndarray,
    n_samples: int,
    rng: np.random.Generator = None
) -> np.ndarray:
    """
    Sample redshifts from the MRD probability distribution using inverse CDF.
    
    Parameters:
        P_z_interp: Interpolator for P(z)
        z_grid: Redshift grid
        P_z_density: P(z) density values
        n_samples: Number of samples to draw
        rng: Random number generator
    
    Returns:
        z_samples: Array of sampled redshifts
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Build CDF
    cdf = np.cumsum(P_z_density)
    cdf = cdf / cdf[-1]  # Normalize
    cdf_interp = interp1d(cdf, z_grid, kind='linear', bounds_error=False, 
                          fill_value=(z_grid[0], z_grid[-1]))
    
    # Inverse CDF sampling
    u = rng.uniform(0, 1, n_samples)
    z_samples = cdf_interp(u)
    
    return z_samples

def get_mrd_rate_interpolator(
    datafiles: Path,
    population: str = "fiducial_Hrad",
    alpha: str = "A1.0",
    component: str = "BNSs",
    sigma: float = 0.1,
    filename: str = "MRD_spread_15Z_40_No_MandF2017_0.1_No_No_0.dat"
) -> Tuple[Callable, float]:
    """
    Get an interpolator for R(z) - the merger rate density as function of redshift.
    
    Returns:
        R_z_interp: Interpolator for R(z) in Gpc^-3 yr^-1
        R_0: Local rate at z=0 in Gpc^-3 yr^-1
    """
    mrd_path = (datafiles / "populations" / "MRD" / f"output_sigma{sigma}" / 
                population / alpha / component / filename)
    
    z_mrd, mrd = np.loadtxt(mrd_path, unpack=True)
    R_z_interp = interp1d(z_mrd, mrd, kind='linear', bounds_error=False, fill_value='extrapolate')
    R_0 = float(R_z_interp(0.0))
    
    return R_z_interp, R_0


# Quick test when run directly
if __name__ == "__main__":
    datafiles = Path("datafiles")
    
    P_z, total_rate, local_rate, z_grid, P_z_density = get_mrd_redshift_distribution(datafiles)
    
    print(f"Local merger rate R_0: {local_rate:.1f} Gpc^-3 yr^-1")
    print(f"Total merger rate (all z): {total_rate:.1f} yr^-1")