from pathlib import Path
import emcee
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy import stats
import corner

from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from scipy.stats import gengamma, norm
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid as cumtrapz

import src.init
from src.nsbh.init import initialize_combined_simulation
from src.nsbh.montecarlo import GBM_EFF, FJ_NSBH_FIXED, N_MC_EVENTS
from src.top_hat.montecarlo import (
    apply_detection_cuts,
    check_and_resume_mcmc,
    compute_luminosity_distance,
    create_k_interpolator,
    poiss_log,
    run_mcmc,
    score_func_cvm,
    simplified_montecarlo,
)

DATAFILES   = Path("../../datafiles")
FJ_BNS_MAX  = 10.0

def _run_name(alpha: str) -> str:
    return f"midpoint_complete_alpha_{alpha}"

def _backend_path(alpha: str) -> Path:
    return src.init.create_run_dir(_run_name(alpha), output_files_default="Output_files") / "emcee.h5"

def _load_chain(alpha: str, burn_frac: float, thin: int):
    backend_path = _backend_path(alpha)
    if not backend_path.exists():
        return None, None, None

    backend = emcee.backends.HDFBackend(backend_path)
    flat = backend.get_chain(
        discard=int(backend.iteration * burn_frac),
        thin=thin,
        flat=True,
    )
    return backend, flat, backend_path

labels = [
    r"$A$",
    r"$\log_{10}(L_0)$",
    r"$\theta_c^{\mathrm{BNS}}$ [deg]",
    r"$f_j^{\mathrm{BNS}}$",
    r"$\theta_c^{\mathrm{NSBH}}$ [deg]",
]

def build_redshift_ppf(z_interp_func, z_min=1e-3, z_max=15.0, n_points=1000):
    """
    Creates a numerical Percent Point Function (PPF / Inverse CDF) 
    from a differential dP/dz interpolator function.
    """
    z_fine = np.linspace(z_min, z_max, n_points)
    dP_dz = z_interp_func(z_fine)
    
    # Compute Cumulative Distribution Function (CDF)
    cdf_fine = cumtrapz(dP_dz, z_fine, initial=0)
    cdf_fine /= cdf_fine[-1]  # Normalize to ensure it tops out exactly at 1.0
    
    # Invert the CDF: map Probability -> Redshift
    # bounds_error=False handles tiny floating point edges safely
    ppf_func = interp1d(cdf_fine, z_fine, kind='linear', bounds_error=False, fill_value=(z_min, z_max))
    return ppf_func


#luminosity distance calculations are really expensive # make an interpolator given we only need resolution z = 0-15
z_grid          = np.linspace(0, 15, 200)
distances_grid  = compute_luminosity_distance(z_grid)
d_L_interp      = interp1d(z_grid, distances_grid, kind='cubic', bounds_error=False, fill_value="extrapolate")

from scipy.stats import gamma
def luminosity_gen(A, q):
    # u -> [0, 1]
    shape       = (A - 1) / A
    y           = gamma.ppf(q, shape)
    return y ** (-1 / A)

def quantile_grid_uniform(n, q_min=1e-5, q_max=1.0 - 1e-5):
    edges = np.linspace(0.0, 1.0, n + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    centers = np.clip(centers, q_min, q_max)
    weights = np.diff(edges)
    return centers, weights

def quantile_grid_luminosity_tail(n, q_floor=1e-5):
    s_edges = np.logspace(np.log10(q_floor), 0.0, n + 1)
    q_edges = 1.0 - s_edges[::-1]

    q_edges[0] = 0.0
    q_edges[-1] = 1.0

    q_centers = 0.5 * (q_edges[:-1] + q_edges[1:])
    q_centers = np.clip(q_centers, q_floor, 1.0 - q_floor)

    weights = np.diff(q_edges)
    return q_centers, weights

def deterministic_grid_observables_pure(thetas, z_ppf_func, k_interpolator, total_rate, observation_time, n_z=50, n_L=40, n_E=40):
    """
    Pure 3D deterministic grid using uniform quantiles for Redshift, Luminosity, and Energy.
    Grid size is exactly (n_z * n_L * n_E).
    """
    A_index, L_L0, L_mu_E_10, sigma_E_10 = thetas[:4]
    l_10 = np.log(10)

    # 1. Create uniform quantile grids for all 3 dimensions
    #q_z = np.linspace(1e-5, 1.0 - 1e-5, n_z)
    #q_L = np.linspace(1e-5, 1.0 - 1e-5, n_L)
    #q_L = 1.0 - np.logspace(np.log10(1e-5), 0, n_L)[::-1] #concentrate more as q->1 to better sample the high-luminosity tail, which is more relevant for detection
    #q_E = np.linspace(1e-5, 1.0 - 1e-5, n_E)

    # 2. Compute the exact probability mass (dq) each cell represents
    #dq_z = np.diff(np.concatenate([[0], q_z])) # weight of each z-slice
    #dq_L = np.diff(np.concatenate([[0], q_L])) # weight of each L-slice
    #dq_E = np.diff(np.concatenate([[0], q_E])) # weight of each E-slice

    q_z, dq_z = quantile_grid_uniform(n_z)
    q_L, dq_L = quantile_grid_luminosity_tail(n_L)
    q_E, dq_E = quantile_grid_uniform(n_E)

    # 2. Map quantiles to physical values via PPFs
    z_grid          = z_ppf_func(q_z)
    distances_grid  = d_L_interp(z_grid) # Use interpolator for faster computation
    d_L_sq_grid     = distances_grid**2

    #a               = (A_index - 1) / A_index
    #c               = -A_index
    #L_unit          = gengamma.ppf(q_L, a=a, c=c)
    #use gamma ppf and transform 
    shape           = (A_index - 1) / A_index
    L_unit          = gamma.ppf(1-q_L, shape)** (-1 / A_index)
    
    L_obs_iso_grid  = L_unit * 10**(L_L0 + 49)

    E_p_rest_grid = np.exp(norm.ppf(q_E) * (sigma_E_10 * l_10) + (L_mu_E_10 * l_10))

    # 3. Broadcast to 3D Coordinate Shapes: (z, L, E)
    z_3d        = z_grid[:, np.newaxis, np.newaxis]
    d_L_sq_3d   = d_L_sq_grid[:, np.newaxis, np.newaxis]
    L_3d        = L_obs_iso_grid[np.newaxis, :, np.newaxis]
    E_rest_3d   = E_p_rest_grid[np.newaxis, np.newaxis, :]

    z_3d, d_L_sq_3d, L_3d, E_rest_3d = np.broadcast_arrays(z_3d, d_L_sq_3d, L_3d, E_rest_3d)

    # 4. Calculate identical base probability weights per cell
    # Total physical events scaling over the observation timeline
    prob_mass_3d = dq_z[:, np.newaxis, np.newaxis] * dq_L[np.newaxis, :, np.newaxis] * dq_E[np.newaxis, np.newaxis, :]
    total_expected_events = total_rate * observation_time
    #weight_per_cell = total_expected_events / (n_z * n_L * n_E)
    #weight_3d = np.full(z_3d.shape, weight_per_cell)
    weight_3d = prob_mass_3d * total_expected_events # not uniform weights, but properly scaled to the physical distribution and total expected events.

    # 5. Physical Observables & K-Correction
    E_p_obs_3d = E_rest_3d / (1 + z_3d)

    flat_log_E = np.log10(E_p_obs_3d).ravel()
    flat_z = z_3d.ravel()
    
    #flat_k = k_interpolator.ev(flat_log_E, flat_z)
    pts = np.column_stack((flat_log_E, flat_z))
    flat_k = k_interpolator(pts)
    
    k_corr_3d = flat_k.reshape(E_p_obs_3d.shape)

    p_flux_3d = L_3d / (4 * np.pi * d_L_sq_3d * k_corr_3d) * 6.242e8

    return {
        "p_flux"    : p_flux_3d.ravel(),
        "E_p_obs"   : E_p_obs_3d.ravel(),
        "z_arr"     : flat_z,
        "L_p_obs"   : L_3d.ravel(),
        "weights"   : weight_3d.ravel(),
    }

def compute_binned_cash_likelihood(model_pflux, model_epeak, model_weights, data_pflux, data_epeak, bin_edges_pflux, bin_edges_epeak):
    """
    Computes the 2D binned Cash (Poisson) likelihood combining BNS and NSBH populations.
    
    Parameters:
    -----------
    model_pflux, model_epeak : arrays
        The flat arrays of physical observables surviving the ANALYSIS threshold.
    model_weights : array
        The absolute expected counts (weights * epsilon * efficiency) matching the model arrays.
    data_pflux, data_epeak : arrays
        The real observed data arrays.
    bin_edges_pflux, bin_edges_epeak : arrays
        Pre-defined histogram bin walls.
    """
    # 1. Bin the actual observed data (K)
    k_matrix, _, _ = np.histogram2d(
        data_pflux, data_epeak, 
        bins=[bin_edges_pflux, bin_edges_epeak]
    )
    
    # 2. Bin the model predicted counts (Mu) by accumulating weights
    mu_matrix, _, _ = np.histogram2d(
        model_pflux, model_epeak, 
        bins=[bin_edges_pflux, bin_edges_epeak], 
        weights=model_weights
    )
    

    # Mask to compute safely
    safe_mu = np.where(mu_matrix > 0, mu_matrix, 1e-10)
    
    # Cash statistic formula element-wise: k * ln(mu) - mu
    #log_lik_matrix = k_matrix * np.log(safe_mu) - mu_matrix
    
    # Where mu is 0 and data is > 0, set to -inf
    #log_lik_matrix = np.where((mu_matrix <= 0) & (k_matrix > 0), -np.inf, log_lik_matrix)
    
    #return np.sum(log_lik_matrix)

    #shape_term = k_matrix * np.log(safe_mu)
    #shape_term = np.where(
    #    (mu_matrix <= 0) & (k_matrix > 0), -np.inf, shape_term
    #)

    #total_expected_filtered_events = np.sum(model_weights)

    #return np.sum(shape_term) - total_expected_filtered_events

    mu_total = np.sum(mu_matrix)
    mu_norm  = mu_matrix / mu_total  # pure shape

    k_total  = np.sum(k_matrix)
    shape_ll = np.sum(k_matrix * np.log(np.where(mu_norm > 0, mu_norm, 1e-300)))
    count_ll = k_total * np.log(mu_total) - mu_total  # global Poisson term

    return shape_ll + count_ll

def run_pop(
    alphas,
    geom_eff_func,
    datafiles=DATAFILES,
    n_walkers   : int   = 20,
    n_steps             = 30_000,
    fixed_params        = None
):
    """Run the complete MCMC pipeline with free f_j^{BNS} and spectral parameters."""

    #LOG_EPS_PRIOR, LOG_EPS_MIN, LOG_EPS_MAX = build_log_eps_prior(geom_eff_func)

    for alpha in alphas:
        demo_params = {
            "z_model": f"fiducial_delayed_{alpha}",
        }
        fixed_params_alpha = fixed_params[alpha]
        L_mu_E, sigma_E = fixed_params_alpha["L_mu_E"],  fixed_params_alpha["sigma_E"]

        bns_params, nsbh_data, _ = initialize_combined_simulation(
            datafiles=datafiles,
            params=demo_params,
            nsbh_population="fiducial_delayed_cut",
            nsbh_alpha=alpha,
        )

        k_interpolator          = create_k_interpolator()
        total_merger_rate_bns   = bns_params.total_merger_rate # yr-1
        total_merger_rate_nsbh  = nsbh_data.total_merger_rate # yr-1 
        
        print(f"\n\nAlpha {alpha}: Total BNS Merger Rate = {total_merger_rate_bns:.2f} yr^-1, Total NSBH Merger Rate = {total_merger_rate_nsbh:.2f} yr^-1")
        #ratio
        print(f"Alpha {alpha}: NSBH/BNS Merger Rate Ratio = {total_merger_rate_nsbh / total_merger_rate_bns:.3f}\n\n")

        backend_path    = _backend_path(alpha)
        # Create the numerical PPFs outside your likelihood loop for speed
        bns_z_ppf  = build_redshift_ppf(bns_params.P_z_interp)
        nsbh_z_ppf = build_redshift_ppf(nsbh_data.P_z_interp)

        n_bins_pflux = 5 
        n_bins_epeak = 5

        #bin_edges_pflux = np.linspace(np.min(bns_params.pflux_data), np.max(bns_params.pflux_data), n_bins_pflux + 1)
        #bin_edges_epeak = np.linspace(np.min(bns_params.epeak_data), np.max(bns_params.epeak_data), n_bins_epeak + 1)

        #defined bin edges by quantiles to ensure data is well distributed across bins, which is important for the Cash likelihood to be informative
        bin_edges_pflux = np.quantile(bns_params.pflux_data, np.linspace(0, 1, n_bins_pflux + 1))
        bin_edges_epeak = np.quantile(bns_params.epeak_data, np.linspace(0, 1, n_bins_epeak + 1))



        def log_likelihood(thetas, n_z=30, n_L=30, n_E=20):
            A_index, L_L0, theta_c_bns, fj, theta_c_nsbh = thetas
            grb_thetas = [A_index, L_L0, L_mu_E, sigma_E]
 
            epsilon_bns     = fj * geom_eff_func(theta_c_bns)
            epsilon_nsbh    = FJ_NSBH_FIXED * geom_eff_func(theta_c_nsbh)

            # Evaluate BNS Deterministic Grid (Size: n_z * n_L * n_E)
            bns_results = deterministic_grid_observables_pure(
                grb_thetas, bns_z_ppf, k_interpolator,
                total_rate=total_merger_rate_bns, 
                observation_time=bns_params.triggered_years,
                n_z=n_z, n_L=n_L, n_E=n_E
            )
            bns_trig, bns_analysis = apply_detection_cuts(bns_results["p_flux"], bns_results["E_p_obs"])

            # Evaluate NSBH Deterministic Grid (Size: n_z * n_L * n_E)
            nsbh_results = deterministic_grid_observables_pure(
                grb_thetas, nsbh_z_ppf, k_interpolator,
                total_rate=total_merger_rate_nsbh, 
                observation_time=bns_params.triggered_years,
                n_z=n_z, n_L=n_L, n_E=n_E
            )
            nsbh_trig, nsbh_analysis = apply_detection_cuts(nsbh_results["p_flux"], nsbh_results["E_p_obs"])

            bns_weights     = bns_results["weights"] * epsilon_bns * GBM_EFF
            nsbh_weights    = nsbh_results["weights"] * epsilon_nsbh * GBM_EFF
            # Concatenate thresholds
            pflux_det   = np.concatenate([bns_results["p_flux"][bns_analysis], nsbh_results["p_flux"][nsbh_analysis]])
            epeak_det   = np.concatenate([bns_results["E_p_obs"][bns_analysis], nsbh_results["E_p_obs"][nsbh_analysis]])
            weights_det = np.concatenate([bns_weights[bns_analysis], nsbh_weights[nsbh_analysis]])

            if len(pflux_det) <= 3: return -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf

            logL_total = compute_binned_cash_likelihood(
                pflux_det, epeak_det, weights_det,
                bns_params.pflux_data, bns_params.epeak_data,
                bin_edges_pflux, bin_edges_epeak
            )

            # Extra info tracking for the MCMC blobs
            predicted_bns   = np.sum(bns_weights[bns_trig])
            predicted_nsbh  = np.sum(nsbh_weights[nsbh_trig])
            
            # Dummy placeholders matching your original backend blob footprint structure 
            # (since shape and poisson are now merged directly into logL_total)
            return logL_total, logL_total, 0.0, 0.0, predicted_bns, predicted_nsbh
        
        def flat_prior(thetas):
            A_index, L_L0, theta_c_bns, fj, theta_c_nsbh = thetas
            if not (1.5 < A_index      < 5):   return -np.inf
            if not (-2  < L_L0         < 7):   return -np.inf
            if not (1   < theta_c_bns  < 25):  return -np.inf
            if not (0   < fj           < 10):  return -np.inf
            if not (1   < theta_c_nsbh < 50):  return -np.inf
            return 0.0

        def log_probability(thetas):
            #add prior values to all 
            lp_f = flat_prior(thetas)
            if not np.isfinite(lp_f):
                return -np.inf, 0.0, 0.0, 0.0, 0.0, 0.0
                        
            ll = log_likelihood(thetas=thetas)
            if not np.isfinite(ll[0]): return -np.inf, 0.0, 0.0, 0.0, 0.0, 0.0

            return lp_f + ll[0], ll[1], ll[2], ll[3], ll[4], ll[5]

        rng = np.random.default_rng(123)

        n_params = len(labels) 

        samples_for_start = rng.uniform(
            low=    [1.5, -2, 1, 0, 1],
            high=   [5  , 6, 25, 10, 50],
            size=   (n_walkers, n_params),
        )

        n_steps_in = n_steps
        # if n_steps is array with different steps for each alpha, use n_steps_in = n_steps_alpha[i]
        if isinstance(n_steps, list): n_steps_in = n_steps[alphas.index(alpha)]

        initial_pos, n_steps_rem, backend = check_and_resume_mcmc(
            filename=backend_path,
            n_steps=n_steps_in,
            starting_point=samples_for_start,
            n_walkers=n_walkers,
        )

        run_mcmc(
            log_probability_func=log_probability,
            initial_walkers=initial_pos,
            n_iterations=n_steps_rem,
            n_walkers=n_walkers,
            n_params=n_params,
            backend=backend,
            blobs_dtype=[
                ("l_pflux", float),
                ("l_epeak", float),
                ("l_poiss", float),
                ("mu_bns", float),
                ("mu_nsbh", float),
            ],
            moves=[
                (emcee.moves.DEMove(), 0.8),
                (emcee.moves.DESnookerMove(), 0.2),
            ], 
        )

def posterior_predictive_check(
    alpha,
    geom_eff_func=None,
    datafiles=DATAFILES,
    fixed_params=None,
    burn_frac=0.5,
    thin    =   50,
    n_draws =   100,
    seed    =   123,
):
    
    demo_params = {
        "z_model": f"fiducial_delayed_{alpha}",
    }
    fixed_params_alpha = fixed_params[alpha]
    L_mu_E, sigma_E = fixed_params_alpha["L_mu_E"],  fixed_params_alpha["sigma_E"]

    bns_params, nsbh_data, _ = initialize_combined_simulation(
        datafiles=datafiles,
        params=demo_params,
        nsbh_population="fiducial_delayed_cut",
        nsbh_alpha=alpha,
    )

    k_interpolator  = create_k_interpolator()
    total_merger_rate_bns = bns_params.total_merger_rate # yr-1
    total_merger_rate_nsbh = nsbh_data.total_merger_rate # yr-1
    bns_z_ppf  = build_redshift_ppf(bns_params.P_z_interp)
    nsbh_z_ppf = build_redshift_ppf(nsbh_data.P_z_interp)

    #log likelihood but without the likelihood, just the observables and weights, which are needed for the posterior predictive check. This is essentially a forward model that maps from the parameter space to the observable space, which we can then use to generate posterior predictive samples.
    def forward_model(theta):
        #A_index, L_L0, log_eps_bns, theta_c_nsbh = theta
        #epsilon_bns = 10.0**log_eps_bns

        A_index, L_L0, theta_c_bns, fj, theta_c_nsbh = theta
        epsilon_bns = fj * geom_eff_func(theta_c_bns)

        grb_thetas = [A_index, L_L0, L_mu_E, sigma_E]

        geom_eff_nsbh = geom_eff_func(theta_c_nsbh)
        epsilon_nsbh = geom_eff_nsbh * FJ_NSBH_FIXED

        bns_results = deterministic_grid_observables_pure(
            grb_thetas, bns_z_ppf, k_interpolator,
            total_rate=total_merger_rate_bns, 
            observation_time=bns_params.triggered_years,
            n_z=30, n_L=30, n_E=20
        )
        bns_weights = bns_results["weights"] * epsilon_bns * GBM_EFF
        bns_trig, bns_analysis = apply_detection_cuts(bns_results["p_flux"], bns_results["E_p_obs"])

        nsbh_results = deterministic_grid_observables_pure(
            grb_thetas, nsbh_z_ppf, k_interpolator,
            total_rate=total_merger_rate_nsbh, 
            observation_time=bns_params.triggered_years,
            n_z=30, n_L=30, n_E=20
        )
        nsbh_weights = nsbh_results["weights"] * epsilon_nsbh * GBM_EFF
        nsbh_trig, nsbh_analysis = apply_detection_cuts(nsbh_results["p_flux"], nsbh_results["E_p_obs"])

        pflux_det = np.concatenate([bns_results["p_flux"][bns_analysis], nsbh_results["p_flux"][nsbh_analysis]])
        epeak_det = np.concatenate([bns_results["E_p_obs"][bns_analysis], nsbh_results["E_p_obs"][nsbh_analysis]])
        weights_det = np.concatenate([bns_weights[bns_analysis], nsbh_weights[nsbh_analysis]])

        return pflux_det, epeak_det, weights_det, np.sum(bns_weights[nsbh_trig]), np.sum(nsbh_weights[nsbh_trig])

    backend, flat_chain, _ = _load_chain(alpha, burn_frac, thin)

    rng = np.random.default_rng(seed)
    draw_idx = rng.choice(len(flat_chain), size=n_draws, replace=False)
    posterior_draws = flat_chain[draw_idx]

    def weighted_cdf(x, w):
        idx = np.argsort(x)

        x_sorted = x[idx]
        w_sorted = w[idx]

        cdf = np.cumsum(w_sorted)
        cdf /= cdf[-1]

        return x_sorted, cdf
    
    ppc_rates = []

    cdf_pflux = []
    cdf_epeak = []

    for theta in posterior_draws:

        (
            pflux_det,
            epeak_det,
            weights_det,
            mu_bns,
            mu_nsbh,
        ) = forward_model(theta)

        mu_total = mu_bns + mu_nsbh
        ppc_rates.append(mu_total)

        x_pf, cdf_pf = weighted_cdf(pflux_det, weights_det)
        x_ep, cdf_ep = weighted_cdf(epeak_det, weights_det)

        cdf_pflux.append((x_pf, cdf_pf))
        cdf_epeak.append((x_ep, cdf_ep))


    obs_pflux = np.sort(bns_params.pflux_data)
    obs_cdf_pflux = np.arange(1, len(obs_pflux)+1)/len(obs_pflux)

    obs_epeak = np.sort(bns_params.epeak_data)
    obs_cdf_epeak = np.arange(1, len(obs_epeak)+1)/len(obs_epeak)

    figsize = (6, 6)

    fig, ax = plt.subplots(figsize=figsize)

    for x, cdf in cdf_pflux:
        ax.plot(x, cdf, alpha=0.08, color="tab:blue")

    ax.plot(
        obs_pflux,
        obs_cdf_pflux,
        color="k",
        lw=3,
        label="Observed",
    )

    ax.set_xscale("log")
    ax.set_xlim(min(bns_params.pflux_data)*0.8, max(bns_params.pflux_data)*1.2)
    ax.set_xlabel("Peak Flux")
    ax.set_ylabel("CDF")
    ax.legend()

    fig, ax = plt.subplots(figsize=figsize)

    for x, cdf in cdf_epeak:
        ax.plot(x, cdf, alpha=0.08, color="tab:orange")

    ax.plot(
        obs_epeak,
        obs_cdf_epeak,
        color="k",
        lw=3,
        label="Observed",
    )

    ax.set_xscale("log")
    ax.set_xlim(min(bns_params.epeak_data)*0.8, max(bns_params.epeak_data)*1.2)
    ax.set_xlabel(r"$E_{\rm peak}$")
    ax.set_ylabel("CDF")
    ax.legend()

    observed_total = len(bns_params.pflux_data)

    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(
        ppc_rates,
        bins=20,
        density=True,
    )

    ax.axvline(
        observed_total,
        color="r",
        lw=3,
        label=f"Observed ({observed_total:.0f})",
    )

    ax.set_xlabel("Expected detected GRBs")
    ax.set_ylabel("Posterior predictive density")
    ax.legend()
    plt.show()
