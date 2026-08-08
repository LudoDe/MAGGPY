import sys
from pathlib import Path

import emcee
import numpy as np

import src.init
from src.nsbh.init import initialize_combined_simulation
from src.nsbh.montecarlo import GBM_EFF, N_MC_EVENTS
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

DATAFILES = Path("../../datafiles")

def _backend_path(run_dir: str, alpha: str) -> Path:
    run_name = f"baseline_{run_dir}_alpha_{alpha}"
    return src.init.create_run_dir(run_name, output_files_default="Output_files") / "emcee.h5"

def run_baseline_populations(
    alphas,
    geom_eff_func,
    run_dir_name: str,
    datafiles=DATAFILES,
    n_params: int = 6,
    n_walkers: int = 24,
    n_steps: int = 20_000,
):
    """
    Run a normal (baseline) MCMC pipeline with no NSBH contributions.
    Finds spectral parameters and theta_c for a specific geometry.
    """
    for alpha in alphas:
        print("\n============================================================")
        print(f"Running Baseline MCMC pipeline for alpha = {alpha} in {run_dir_name}")
        print("============================================================")

        demo_params = {
            "alpha": -0.6,
            "beta_s": -2.5,
            "n": 2.0,
            "theta_c": 3.4,
            "theta_v_max": 10.0,
            "z_model": f"fiducial_delayed_{alpha}",
        }

        # We initialize combined simulation just to get the BNS data, but we only use BNS.
        bns_params, _, _ = initialize_combined_simulation(
            datafiles=datafiles,
            params=demo_params,
            size_test=2_000,
            nsbh_population="fiducial_delayed_cut",
            nsbh_alpha=alpha,
        )

        k_interpolator = create_k_interpolator()
        bns_distances = compute_luminosity_distance(bns_params.z_arr)

        backend_path = _backend_path(run_dir_name, alpha)

        def log_likelihood_baseline(thetas, n_events=N_MC_EVENTS):
            A_index, L_L0, L_mu_E, sigma_E, theta_c_bns, fj_bns = thetas
            grb_thetas = [A_index, L_L0, L_mu_E, sigma_E]

            geom_eff_bns = geom_eff_func(theta_c_bns)
            epsilon_bns = geom_eff_bns * fj_bns
            intrinsic_bns = epsilon_bns * len(bns_params.z_arr) * GBM_EFF

            bns_results = simplified_montecarlo(
                grb_thetas, n_events, bns_params, bns_distances, k_interpolator
            )
            bns_trig, bns_analysis = apply_detection_cuts(
                bns_results["p_flux"], bns_results["E_p_obs"]
            )

            pflux_det = bns_results["p_flux"][bns_analysis]
            epeak_det = bns_results["E_p_obs"][bns_analysis]
            if len(pflux_det) <= 3:
                return -np.inf, -np.inf, -np.inf, -np.inf

            logL_pflux = score_func_cvm(pflux_det, bns_params.pflux_data, bns_params.rng)
            logL_epeak = score_func_cvm(epeak_det, bns_params.epeak_data, bns_params.rng)

            phys_eff_bns = np.sum(bns_trig) / n_events
            predicted_bns = intrinsic_bns * bns_params.triggered_years * phys_eff_bns
            observed_total = bns_params.yearly_rate * bns_params.triggered_years

            if predicted_bns <= 0:
                return -np.inf, -np.inf, -np.inf, -np.inf

            logL_poisson = poiss_log(k=observed_total, mu=predicted_bns)
            logL_total = logL_pflux + logL_epeak + logL_poisson

            return logL_total, logL_pflux, logL_epeak, logL_poisson

        def log_prior_baseline(thetas):
            A_index, L_L0, L_mu_E, sigma_E, theta_c_bns, fj_bns = thetas
            if not (1.5 < A_index       < 12): return -np.inf
            if not (-2  < L_L0          < 7): return -np.inf
            if not (0.1 < L_mu_E        < 7): return -np.inf
            if not (0   < sigma_E       < 2.5): return -np.inf
            if not (1   < theta_c_bns   < 25): return -np.inf
            if not (0   < fj_bns        < 10): return -np.inf
            return 0.0

        def init_walkers_baseline(n_walkers, seed=123):
            rng = np.random.default_rng(seed)
            return np.column_stack([
                rng.uniform(2.0, 3.5, n_walkers),
                rng.uniform(2.0, 4.5, n_walkers),
                rng.uniform(1.5, 4.5, n_walkers),
                rng.uniform(0.2, 1.2, n_walkers),
                rng.uniform(3.0, 20.0, n_walkers),
                rng.uniform(1.0, 9.0, n_walkers),
            ])

        def log_probability_baseline(thetas):
            lp = log_prior_baseline(thetas)
            if not np.isfinite(lp):
                return -np.inf, 0.0, 0.0, 0.0

            ll = log_likelihood_baseline(thetas=thetas, n_events=N_MC_EVENTS)
            if not np.isfinite(ll[0]):
                return -np.inf, 0.0, 0.0, 0.0

            return lp + ll[0], ll[1], ll[2], ll[3]

        initial_pos, n_steps_rem, backend = check_and_resume_mcmc(
            filename=backend_path,
            n_steps=n_steps,
            initialize_walkers_func=init_walkers_baseline,
            n_walkers=n_walkers,
        )

        run_mcmc(
            log_probability_func=log_probability_baseline,
            initial_walkers=initial_pos,
            n_iterations=n_steps_rem,
            n_walkers=n_walkers,
            n_params=n_params,
            backend=backend,
            blobs_dtype=[
                ("l_pflux", float),
                ("l_epeak", float),
                ("l_poiss", float),
            ],
        )
