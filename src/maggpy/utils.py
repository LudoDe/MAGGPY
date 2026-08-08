import time
import numpy            as np
import scipy.special    as sc
from pathlib            import Path
from functools          import wraps
from typing             import Any, Callable, Tuple
from scipy.stats        import gengamma
from astropy.cosmology  import FlatLambdaCDM, Planck18
from scipy.stats        import cramervonmises_2samp

DEFAULT_LIMITS = {
    "F_LIM"         : 4,         # 4 ph/cm^2/s in 64 ms
    "T90_LIM"       : 2,         # 1 s
    "EP_LIM_UPPER"  : 10_000,    # 10_000 keV
    "EP_LIM_LOWER"  : 50,        # 50 keV
}

def luminosity_gen(A, n, rng=None):
    """Generate luminosities from modified Schechter distribution (Salafia et al 2024)."""
    return gengamma.rvs(a=(A - 1)/A, c=-A, size=n, random_state=rng)

def compute_luminosity_distance(z, cosmology=FlatLambdaCDM(H0=Planck18.H0, Om0=Planck18.Om0)):
    """Compute luminosity distance in cm."""
    return cosmology.luminosity_distance(z).cgs.value

def poiss_log(k, mu):
    """Log Poisson probability (numerically stable)."""
    return -mu + k * np.log(mu) - sc.gammaln(k + 1)

def cdf_sample(data, n, rng):
    """Inverse transform sampling from empirical CDF."""
    x_sorted = np.sort(data)
    u = rng.uniform(0, 1, n)
    return np.interp(u, np.linspace(0, 1, len(data)), x_sorted)

def score_func_cvm(y_sim, y_obs, rng=np.random.default_rng(42)):
    """Cramér-von Mises score function."""
    y_sim_f         = np.asarray(y_sim, dtype=np.float64)
    y_obs_f         = np.asarray(y_obs, dtype=np.float64)
    y_resample  = cdf_sample(y_sim_f, len(y_obs_f), rng=rng)
    y_in        = np.log10(y_resample)
    y_out       = np.log10(y_obs_f)
    return np.log(cramervonmises_2samp(y_in, y_out).pvalue)

def timer_decorator(func: Callable) -> Callable:
    """
    Decorator to time function execution.

    Returns:
        A tuple containing the original function's result and the elapsed time.
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Tuple[Any, float]:
        start_time  = time.perf_counter()
        result      = func(*args, **kwargs)
        end_time    = time.perf_counter()
        return result, end_time - start_time
    return wrapper

def running_avg(arr: np.ndarray, window_size: int = 1600) -> np.ndarray:
    """
    Calculate the running average of an array using a sliding window.

    Parameters:
        arr (np.ndarray): Input array.
        window_size (int): Size of the averaging window.

    Returns:
        np.ndarray: Array of running averages computed only over non-NaN elements.
    """
    valid_values = arr[~np.isnan(arr)]
    avg = np.convolve(valid_values, np.ones(window_size) / window_size, mode='valid')
    return avg

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

    if output_files.exists():
       msg = f"Loading existing directory  : {output_files}"
    
    if not QUIET_FLAG:
        print(msg)

    output_files.mkdir(parents=True, exist_ok=True) # Create the directory if it doesn't exist

    return output_files