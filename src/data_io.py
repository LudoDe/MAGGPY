import  numpy       as np
import  pandas      as pd
from    scipy       import interpolate
from    typing      import Tuple, Callable, Dict
from    .montecarlo  import DEFAULT_LIMITS

def get_Rf_Re(filename: str) -> Tuple[Callable, Callable, np.ndarray]:
    """
    Load F_max data from a file and return interpolators for f_fmax and E_Emax.

    Parameters:
        filename (str): Path to the file containing F_max data.

    Returns:
        R_F: Interpolator for the normalized f_fmax.
        R_E: Interpolator for the normalized E_Emax.
        theta_v_arr_f: Array of theta values.
    """
    file_f_max  = np.loadtxt(filename).T
    theta_v_arr_f, f_fmax, E_Emax = file_f_max

    f_fmax = f_fmax/f_fmax[0] 
    E_Emax = E_Emax/E_Emax[0] 

    R_F = interpolate.interp1d(theta_v_arr_f, f_fmax, fill_value="extrapolate")
    R_E = interpolate.interp1d(theta_v_arr_f, E_Emax, fill_value="extrapolate")

    return R_F, R_E, theta_v_arr_f

def get_alpha_n_alpha_e(file_n: str, file_e: str) -> Tuple[Callable, Callable, np.ndarray, np.ndarray]:
    """
    Load the alpha_n and alpha_e strucure functions data (see tutorial) from files and return their interpolators.

    Parameters:
        file_n (str): File path for alpha_n data.
        file_e (str): File path for alpha_e data.

    Returns:
        alpha_n: Interpolator for alpha_n.
        alpha_e: Interpolator for alpha_e.
        theta_v_arr_n: Array of theta values for alpha_n (in radians).
        theta_v_arr_e: Array of theta values for alpha_e (in radians).
    """
    deg_to_rad 	    = np.pi/180

    # Process alpha_n
    file_alpha      = np.loadtxt(file_n).T
    theta_v_arr_n, alpha_n_values = file_alpha
    theta_v_arr_n   = theta_v_arr_n * deg_to_rad
    alpha_n         = interpolate.interp1d(theta_v_arr_n, alpha_n_values, fill_value="extrapolate")

    # Process alpha_e
    file_alpha_e    = np.loadtxt(file_e).T
    theta_v_arr_e, alpha_e_values = file_alpha_e
    theta_v_arr_e   = theta_v_arr_e * deg_to_rad 
    alpha_e         = interpolate.interp1d(theta_v_arr_e, alpha_e_values, fill_value="extrapolate") 

    return alpha_n, alpha_e, theta_v_arr_n, theta_v_arr_e

def get_observables_data(filename: str) -> Dict[str, np.ndarray]:
    """
    Load constraints data from the specified file and return a dictionary containing observables.

    Parameters:
        filename (str): Path to the constraints file.

    Returns:
        Dictionary with keys:
          'epeak', 'epeak_err', 'duration', 'duration_err',
          'pflux', 'pflux_err', 'fluence', 'fluence_err'.
    """
    file_constraints = np.loadtxt(filename).T
    pflux_data, duration_data, fluence_data, epeak_data = file_constraints 

    print(f"Loaded {len(epeak_data)} events from {filename}.")
    # the bounds for the data
    print(f"pflux: {np.min(pflux_data):.2e} - {np.max(pflux_data):.2e}")
    print(f"duration: {np.min(duration_data):.2e} - {np.max(duration_data):.2e}")
    print(f"fluence: {np.min(fluence_data):.2e} - {np.max(fluence_data):.2e}")
    print(f"epeak: {np.min(epeak_data):.2e} - {np.max(epeak_data):.2e}")


    return {
        "epeak"         : epeak_data,
        "duration"      : duration_data,
        "pflux"         : pflux_data,
        "fluence"       : fluence_data,
    }

def get_redshift_distribution(filename: str) -> np.ndarray:
    """
    Load redshift distribution from a file.

    Parameters:
        filename (str): Path to the redshift data file.

    Returns:
        Array of redshift values.
    """
    parameters  = ['mass_1', 'mass_2', 'redshift', 'cmu1', 'cmu2', 'dl']
    err_ET      = pd.read_csv(filename, names = parameters, delimiter=' ')
    z_arr       = err_ET['redshift'].to_numpy()
    return z_arr

def catalogue_prep(datafiles, limits = DEFAULT_LIMITS):
    
    #prep catalogue with limits
    print("Preparing catalogue with limits:", limits)
    
    catalogue_data = datafiles / "burst_catalog.dat"
    df = pd.read_csv(catalogue_data)

    f_64_lim        = limits["F_LIM"]
    t90_lim         = limits["T90_LIM"]
    ep_upper_lim    = limits["EP_LIM_UPPER"]
    ep_lower_lim    = limits["EP_LIM_LOWER"]

    trigger_condition = (df['FLUX_BATSE_64'] > f_64_lim) & (df['T90'] < t90_lim)
    shape_condition   = trigger_condition & (df['PFLX_COMP_EPEAK'] > ep_lower_lim) & (df['PFLX_COMP_EPEAK'] < ep_upper_lim)

    df_trig = df[trigger_condition] # trigger condition is less strict, as we want to include all events that triggered the GBM
    df_shape = df[shape_condition]  # shape condition is more strict, as GBM fit doesn't always converge for peak energy
    pflux, t90, fluence_bat, epeak, trigger_time = df_shape.T.to_numpy()
    
    trigger_time_trig   = df_trig['TRIGGER_TIME'].to_numpy()
    days_in_yr          = 365.25
    trigger_years       = (max(trigger_time_trig) - min(trigger_time_trig)) / days_in_yr
    triggered_events    = len(df_trig)
    yearly_rate         = triggered_events / trigger_years

    print(f"Triggered events: {triggered_events}, Trigger years: {trigger_years:.2f}, Yearly rate: {yearly_rate:.2f} events/year")

    return {
        "df_trig"           : df_trig,
        "df_shape"          : df_shape,
        "trigger_time"      : trigger_time,
        "trigger_years"     : trigger_years,
        "triggered_events"  : triggered_events,
        
        "pflux"             : pflux,
        "t90"               : t90,
        "fluence"           : fluence_bat,
        "epeak"             : epeak,
        "c_det"             : yearly_rate,
    }