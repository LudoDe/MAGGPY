from .montecarlo import *

DEFAULT_LIMITS_GRINTA: Dict[str, Any] = {
    "T90_LIM"       : 2,         # 2 s
    "EP_LIM_UPPER"  : 10_000,    # 10_000 keV
    "EP_LIM_LOWER"  : 50,        # 50 keV
}

def peak_flux_lim(time_array, slope = -0.5, intercept = -0.50):
    # returns the peak flux limit in ph/cm^2/s given time_array in seconds
    return 10**(intercept) * time_array**(slope)

def make_observations_w_sensitivity(thetas, params : SimParams, interps : Interps, 
                      limits : Dict[str, Any] = DEFAULT_LIMITS_GRINTA, n_yrs : float = 4):
    """
    Accumulates GRB observables until the number of counts reaches min_count.

    Returns:
        t_det, f_det, Ep_det, Fp_det: Lists with aggregated observables.
        count_real_grbs: Total count of base valid GRBs.
        current_n: Total samples drawn.
        count: Total count of observed catalogue GRBs.
    """
    n_yrs = params.triggered_years if n_yrs is None else n_yrs
    
    fj                      = thetas[-1]
    GBM_eff                 = 0.6 # GBM duty cycle 85 %, 70% of the sky visible -> 0.85*0.7 ~ 0.6
    #GRINTA_FOV              = 8 #sr
    #GRINTA_FOV              /= 4 * np.pi # fraction of sky
    #GRINTA_DC               = 0.5 
    #GBM_eff                 = GRINTA_FOV * GRINTA_DC # keep same variable name for simplicity
    #print(GBM_eff)
    #GBM_eff = 1 # we renormalize later
    geometric_efficiency    = 1 - np.cos(params.theta_v_max)
    
    # Total BNS mergers per year in the universe (all-sky)
    total_bns_all_sky       = n_yrs * len(params.z_arr)
    
    # Total number of GRBs to simulate (already accounts for geometry and time)
    available_events        = total_bns_all_sky * geometric_efficiency * GBM_eff * fj
    n_events                = int(available_events)

    m_prop          = generate_macro_properties(thetas, params, interps, n_events)

    limit_ph       = peak_flux_lim(m_prop['t_peak_c_z'])
    
    cond_time = (m_prop['t_peak_c_z'] < limits["T90_LIM"])
    cond_flux = (m_prop['F_p_real'] > limit_ph)
    #cond_pflux_ph = (m_prop['F_p_real'] > 4)
    trigger_mask = cond_time & cond_flux #& cond_pflux_ph

    if np.sum(trigger_mask) == 0:
        return None  # Avoid empty arrays

    # Filter properties to triggered events only
    m_prop_triggered = {k: v[trigger_mask] for k, v in m_prop.items()}

    t_90_array, f_det_in    = compute_time_evolution(m_prop_triggered, interps)
    detection_mask = (
        (t_90_array             < limits["T90_LIM"]) 
    )

    triggered_events = np.sum(detection_mask) # Total triggered events before shape cuts

    if triggered_events == 0:
        return None
    
    # Shape analysis cuts (additional Ep cuts for distribution comparison)
    shape_mask = (
        (m_prop_triggered["E_p_obs"] > limits["EP_LIM_LOWER"]) &
        (m_prop_triggered["E_p_obs"] < limits["EP_LIM_UPPER"])
    )

    # Combine detection and shape masks for final sample
    final_mask = detection_mask & shape_mask

    if np.sum(final_mask) == 0:
        return None

    return {
        "t_det"                 : t_90_array[final_mask],
        #peak peak
        "t_peak_det"            : m_prop_triggered['t_peak_c_z'][final_mask],
        "f_det"                 : f_det_in[final_mask],
        "Ep_det"                : m_prop_triggered["E_p_obs"][final_mask],
        #"Ep_on_axis_det"        : m_prop_triggered["E_p_on_axis"][final_mask],
        #"Fp_phot_det"           : P_F_64ms_50_300_phot[final_mask],
        #"Fp_erg_det"            : P_F_64ms_50_300_erg[final_mask],
        "Fp_phot"               : m_prop_triggered['F_p_real'][final_mask],
        #"Fp_erg_det"            : m_prop_triggered['F_p_real_erg'][final_mask],
        "z_det"                 : m_prop_triggered["z"][final_mask],
        "theta_v_det"           : m_prop_triggered["theta_v"][final_mask],
        "triggered_events"      : triggered_events,
        "isotropic_energy_det"  : m_prop_triggered["isotropic_energy"][final_mask],
    }