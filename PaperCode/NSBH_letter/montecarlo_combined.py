import numpy as np
from pathlib                    import Path
from typing                     import Callable, Sequence
from maggpy.nsbh.init           import initialize_combined_simulation
from maggpy.top_hat.montecarlo  import apply_detection_cuts, start_mcmc, montecarlo
from maggpy.utils               import poiss_log, score_func_cvm

PARAMETER_NAMES = (
    "A_index",
    "L_L0",
    "log10_kappa_nsbh",
    "L_mu_E",
    "sigma_E",
    "theta_c_bns",
    "theta_c_nsbh",
)

LABELS = (
    r"$A$",
    r"$\log_{10}(L_0)$",
    r"$\log_{10}(\kappa_{\rm NSBH})$",
    r"$\mu_{E,p}$",
    r"$\sigma_{E,p}$",
    r"$\theta_c^{\mathrm{BNS}}$ [deg]",
    r"$\theta_c^{\mathrm{NSBH}}$ [deg]",
)

def _bad_likelihood(): return (-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf)

FJ_NSBH_FIXED   = 1
N_MC_EVENTS     = 10_000
GBM_EFF         = 0.6

def run_pop(
    mrd_bns_path,
    mrd_nsbh_path,
    fj_bns,
    geom_eff_func,
    datafiles,
    backend_dir,
    fj_nsbh= FJ_NSBH_FIXED,
    n_walkers= 24,
    n_steps= 10_000,
    n_events= N_MC_EVENTS,
):

    bns_data, nsbh_data, observations = initialize_combined_simulation(
        datafiles       =   datafiles,
        mrd_bns_path    =   mrd_bns_path,
        mrd_nsbh_path   =   mrd_nsbh_path,
    )

    def flat_prior(thetas):
        a_index, l_l0, log10_kappa_nsbh, l_mu_e, sigma_e, theta_c_bns, theta_c_nsbh = thetas
        if not (1.5 < a_index   < 6): return -np.inf
        if not (-2  < l_l0      < 7): return -np.inf
        if not (-2  < log10_kappa_nsbh < 0): return -np.inf
        if not (0.1 < l_mu_e    < 7): return -np.inf
        if not (0   < sigma_e   < 2.5): return -np.inf
        if not (1   < theta_c_bns   < 25): return -np.inf
        if not (1   < theta_c_nsbh  < 25): return -np.inf
        return 0.0

    def log_likelihood(thetas):

        (
            a_index,
            l_l0,
            log10_kappa_nsbh,
            l_mu_e,
            sigma_e,
            theta_c_bns,
            theta_c_nsbh,
        ) = thetas
        bns_grb_thetas      = [a_index, l_l0, l_mu_e, sigma_e]
        nsbh_grb_thetas = [a_index, l_l0 + log10_kappa_nsbh, l_mu_e, sigma_e]

        geom_eff_bns    = geom_eff_func(theta_c_bns)
        geom_eff_nsbh   = geom_eff_func(theta_c_nsbh)
        if (
            not np.isfinite(geom_eff_bns)
            or not np.isfinite(geom_eff_nsbh)
            or geom_eff_bns < 0
            or geom_eff_nsbh < 0
        ):
            return _bad_likelihood()

        intrinsic_bns = (
            geom_eff_bns * fj_bns * bns_data.total_merger_rate * GBM_EFF
        )
        intrinsic_nsbh = (
            geom_eff_nsbh * fj_nsbh * nsbh_data.total_merger_rate * GBM_EFF
        )

        bns_results = montecarlo(
            bns_grb_thetas,
            n_events,
            bns_data,
        )

        nsbh_results = montecarlo(
            nsbh_grb_thetas,
            n_events,
            nsbh_data,
        )

        bns_trig, bns_analysis = apply_detection_cuts(
            bns_results["p_flux"],
            bns_results["E_p_obs"],
        )
        nsbh_trig, nsbh_analysis = apply_detection_cuts(
            nsbh_results["p_flux"],
            nsbh_results["E_p_obs"],
        )

        pflux_detected = np.concatenate(
            (
                bns_results["p_flux"][bns_analysis],
                nsbh_results["p_flux"][nsbh_analysis],
            )
        )
        epeak_detected = np.concatenate(
            (
                bns_results["E_p_obs"][bns_analysis],
                nsbh_results["E_p_obs"][nsbh_analysis],
            )
        )
        if pflux_detected.size <= 3 or epeak_detected.size <= 3: return _bad_likelihood()

        logl_pflux = score_func_cvm(
            pflux_detected,
            observations["pflux"]
        )
        logl_epeak = score_func_cvm(
            epeak_detected,
            observations["epeak"]
        )

        triggered_years = observations["trigger_years"]
        observed_yearly_rate = observations["c_det"]

        phys_eff_bns = np.mean(bns_trig)
        phys_eff_nsbh = np.mean(nsbh_trig)
        predicted_bns = (
            intrinsic_bns * triggered_years * phys_eff_bns
        )

        predicted_nsbh = (
            intrinsic_nsbh * triggered_years * phys_eff_nsbh
        )
        predicted_total = predicted_bns + predicted_nsbh
        observed_total = observed_yearly_rate * triggered_years

        if not np.isfinite(predicted_total) or predicted_total <= 0: return _bad_likelihood()

        logl_poisson = poiss_log(k=observed_total, mu=predicted_total)
        logl_total = logl_pflux + logl_epeak + logl_poisson
        if not np.isfinite(logl_total): return _bad_likelihood()

        return (
            logl_total,
            logl_pflux,
            logl_epeak,
            logl_poisson,
            predicted_bns,
            predicted_nsbh,
        )

    def log_probability(thetas):
        log_prior = flat_prior(thetas)
        if not np.isfinite(log_prior): return _bad_likelihood()

        likelihood = log_likelihood(thetas)
        if not np.isfinite(likelihood[0]): return _bad_likelihood()

        return (log_prior + likelihood[0], *likelihood[1:])

    def initialize_walkers(n_walkers):
        return np.column_stack([
            np.random.uniform(1.5, 6, n_walkers),           # A_index
            np.random.uniform(-2, 7, n_walkers),            # L_L0
            np.random.uniform(-2, 0, n_walkers),            # log10_kappa_nsbh
            np.random.uniform(0.1, 7, n_walkers),           # L_mu_E
            np.random.uniform(0.1, 2.5, n_walkers),         # sigma_E
            np.random.uniform(1, 25, n_walkers),            # theta_c_bns
            np.random.uniform(1, 25, n_walkers),            # theta_c_nsbh
        ])

    backend_dir.mkdir(parents=True, exist_ok=True) # Create the backend directory if it doesn't exist
    sampler = start_mcmc(
        log_probability_func    =   log_probability,
        initialize_walkers_func =   initialize_walkers,
        n_iterations            =   n_steps,
        n_walkers               =   n_walkers,
        backend_fn              =   backend_dir / "mcmc_run.h5",
        n_params                =   len(PARAMETER_NAMES),
        blobs                   =   [
            ("l_pflux", float),
            ("l_epeak", float),
            ("l_poiss", float),
            ("predicted_bns", float),
            ("predicted_nsbh", float),
        ],
    )

    return sampler