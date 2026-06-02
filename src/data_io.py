import  numpy       as np
import  pandas      as pd
from    pathlib     import Path
from    scipy       import interpolate
from    typing      import Tuple, Callable, Optional
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

def load_structure_constants(
    datafiles: Path,
    structure_source: Optional[Path] = None,
) -> Tuple[Callable, Callable, Callable, Callable, np.ndarray, np.ndarray]:
    """
    Load structured-jet constants either from the legacy split files or from a
    unified CSV file containing precomputed structure values.

    Parameters:
        datafiles (Path): Base data directory.
        structure_source (Path, optional): Directory or CSV file containing the
            custom structure constants. If omitted, the legacy files are loaded.

    Returns:
        R_F, R_E, alpha_n, alpha_e: Interpolators for the structure constants.
        theta_rf, theta_alpha: Angle arrays used to build the interpolators.
    """
    if structure_source is None:
        R_F, R_E, theta_rf = get_Rf_Re(datafiles / "F_Fmax_3.4_s4.0.txt")
        alpha_n, alpha_e, theta_alpha_n, theta_alpha_e = get_alpha_n_alpha_e(
            datafiles / "alpha.txt",
            datafiles / "alpha_e.txt",
        )
        return R_F, R_E, alpha_n, alpha_e, theta_rf, theta_alpha_n

    structure_path = Path(structure_source)
    if structure_path.is_dir():
        preferred_csv = structure_path / "struct_results_100.csv"
        if preferred_csv.exists():
            structure_path = preferred_csv
        else:
            csv_files = sorted(structure_path.glob("struct_results_*.csv"))
            if not csv_files:
                raise FileNotFoundError(
                    f"No structure CSV files were found in {structure_path}"
                )
            structure_path = csv_files[0]

    if structure_path.suffix.lower() != ".csv":
        raise ValueError(
            f"Unsupported structure source {structure_path}. Provide a directory or a CSV file."
        )

    structure_df = pd.read_csv(structure_path)
    required_columns = {"theta_v", "R_E", "R_F", "alpha_E", "alpha_N"}
    missing_columns = required_columns.difference(structure_df.columns)
    if missing_columns:
        raise ValueError(
            f"Structure file {structure_path} is missing columns: {sorted(missing_columns)}"
        )

    theta_v = structure_df["theta_v"].to_numpy()
    R_F = interpolate.interp1d(theta_v, structure_df["R_F"].to_numpy(), fill_value="extrapolate")
    R_E = interpolate.interp1d(theta_v, structure_df["R_E"].to_numpy(), fill_value="extrapolate")
    alpha_n = interpolate.interp1d(theta_v, structure_df["alpha_N"].to_numpy(), fill_value="extrapolate")
    alpha_e = interpolate.interp1d(theta_v, structure_df["alpha_E"].to_numpy(), fill_value="extrapolate")

    return R_F, R_E, alpha_n, alpha_e, theta_v, theta_v

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