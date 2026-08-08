from pathlib import Path
import emcee
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.stats import gaussian_kde
import corner

import src.init
from maggpy.nsbh.init import initialize_combined_simulation
from maggpy.nsbh.montecarlo import GBM_EFF, FJ_NSBH_FIXED, N_MC_EVENTS
from maggpy.top_hat.montecarlo import (
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
FJ_BNS_MAX = 10.0

rect_args = {'xy': (2.17, 1.0), 'width': 33.8, 'height': 24.0, 'alpha': 0.1, 'color': 'tab:green'}

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

def run_populations(
    alphas,
    geom_eff_func,
    datafiles=DATAFILES,
    n_params: int = 7,
    n_walkers: int = 24,
    n_steps: int = 10000,
    burn_frac: float = 0.33,
    thin: int = 10,
):
    """Run the complete MCMC pipeline with free f_j^{BNS} and spectral parameters."""

    for alpha in alphas:
        print("\n============================================================")
        print(f"Running complete MCMC pipeline for alpha = {alpha}")
        print("============================================================")

        demo_params = {
            "alpha": -0.6,
            "beta_s": -2.5,
            "n": 2.0,
            "theta_c": 3.4,
            "theta_v_max": 10.0,
            "z_model": f"fiducial_delayed_{alpha}",
        }

        bns_params, nsbh_data, _ = initialize_combined_simulation(
            datafiles=datafiles,
            params=demo_params,
            size_test=2_000,
            nsbh_population="fiducial_delayed_cut",
            nsbh_alpha=alpha,
        )

        k_interpolator = create_k_interpolator()
        bns_distances = compute_luminosity_distance(bns_params.z_arr)
        nsbh_distances = nsbh_data.distances

        backend_path = _backend_path(alpha)

        def log_likelihood(thetas, n_events=N_MC_EVENTS):
            A_index, L_L0, L_mu_E, sigma_E, theta_c_bns, theta_c_nsbh, fj_bns = thetas
            grb_thetas = [A_index, L_L0, L_mu_E, sigma_E]

            geom_eff_bns = geom_eff_func(theta_c_bns)
            geom_eff_nsbh = geom_eff_func(theta_c_nsbh)

            epsilon_bns = geom_eff_bns * fj_bns
            epsilon_nsbh = geom_eff_nsbh * FJ_NSBH_FIXED

            intrinsic_bns = epsilon_bns * len(bns_params.z_arr) * GBM_EFF
            intrinsic_nsbh = epsilon_nsbh * len(nsbh_data.z_arr) * GBM_EFF

            bns_results = simplified_montecarlo(
                grb_thetas, n_events, bns_params, bns_distances, k_interpolator
            )
            bns_trig, bns_analysis = apply_detection_cuts(
                bns_results["p_flux"], bns_results["E_p_obs"]
            )

            nsbh_results = simplified_montecarlo(
                grb_thetas,
                n_events,
                nsbh_data,
                nsbh_distances,
                k_interpolator,
                rng=bns_params.rng,
            )
            nsbh_trig, nsbh_analysis = apply_detection_cuts(
                nsbh_results["p_flux"], nsbh_results["E_p_obs"]
            )

            pflux_det = np.concatenate(
                [bns_results["p_flux"][bns_analysis], nsbh_results["p_flux"][nsbh_analysis]]
            )
            epeak_det = np.concatenate(
                [bns_results["E_p_obs"][bns_analysis], nsbh_results["E_p_obs"][nsbh_analysis]]
            )

            if len(pflux_det) <= 3:
                return -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf

            logL_pflux = score_func_cvm(pflux_det, bns_params.pflux_data, bns_params.rng)
            logL_epeak = score_func_cvm(epeak_det, bns_params.epeak_data, bns_params.rng)

            phys_eff_bns = np.sum(bns_trig) / n_events
            phys_eff_nsbh = np.sum(nsbh_trig) / n_events

            predicted_bns = intrinsic_bns * bns_params.triggered_years * phys_eff_bns
            predicted_nsbh = intrinsic_nsbh * bns_params.triggered_years * phys_eff_nsbh
            predicted_total = predicted_bns + predicted_nsbh
            observed_total = bns_params.yearly_rate * bns_params.triggered_years

            if predicted_total <= 0:
                return -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf

            logL_poisson = poiss_log(k=observed_total, mu=predicted_total)
            logL_total = logL_pflux + logL_epeak + logL_poisson

            return logL_total, logL_pflux, logL_epeak, logL_poisson, predicted_bns, predicted_nsbh

        def log_prior_single_stage(thetas):
            A_index, L_L0, L_mu_E, sigma_E, theta_c_bns, theta_c_nsbh, fj_bns = thetas
            if not (1.5 < A_index < 12): return -np.inf
            if not (-2 < L_L0 < 7): return -np.inf
            if not (0.1 < L_mu_E < 7): return -np.inf
            if not (0 < sigma_E < 2.5): return -np.inf
            if not (1 < theta_c_bns < 50): return -np.inf
            if not (1 < theta_c_nsbh < 50): return -np.inf
            if not (0.01 < fj_bns <= FJ_BNS_MAX): return -np.inf
            return 0.0

        def init_walkers_single_stage(n_walkers, seed=123):
            rng = np.random.default_rng(seed)
            return np.column_stack([
                rng.uniform(2.0, 3.5, n_walkers), # A_index
                rng.uniform(2.0, 4.5, n_walkers), # L_L0
                rng.uniform(1.5, 4.5, n_walkers), # L_mu_E
                rng.uniform(0.2, 1.2, n_walkers), # sigma_E
                rng.uniform(3.0, 20.0, n_walkers), # theta_c_bns
                rng.uniform(2.0, 25.0, n_walkers), # theta_c_nsbh
                rng.uniform(0.1, min(5.0, FJ_BNS_MAX), n_walkers), # fj_bns
            ])

        def log_probability(thetas):
            lp = log_prior_single_stage(thetas)
            if not np.isfinite(lp):
                return -np.inf, 0.0, 0.0, 0.0, 0.0, 0.0

            ll = log_likelihood(thetas=thetas, n_events=N_MC_EVENTS)
            if not np.isfinite(ll[0]):
                return -np.inf, 0.0, 0.0, 0.0, 0.0, 0.0

            return lp + ll[0], ll[1], ll[2], ll[3], ll[4], ll[5]

        initial_pos, n_steps_rem, backend = check_and_resume_mcmc(
            filename=backend_path,
            n_steps=n_steps,
            initialize_walkers_func=init_walkers_single_stage,
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
        )

def _contour_thresholds(density_values, probs=(0.68, 0.95)):
    vals = np.sort(density_values)[::-1]
    csum = np.cumsum(vals)
    csum /= csum[-1]
    return [vals[np.searchsorted(csum, prob)] for prob in probs]

def plot_populations(alphas, burn_frac: float = 0.33, thin: int = 10):
    FS = 18
    x = np.linspace(1, 25, 80)
    y = np.linspace(1, 25, 80)
    X, Y = np.meshgrid(x, y)
    pos = np.vstack([X.ravel(), Y.ravel()])

    for alpha in alphas:
        backend, flat, backend_path = _load_chain(alpha, burn_frac, thin)
        if flat is None or flat.size == 0:
            print(f"Warning: Results missing or empty for alpha={alpha}")
            continue

        theta_bns = flat[:, 4]
        theta_nsbh = flat[:, 5]
        fj_bns = flat[:, 6]

        xy = np.vstack([theta_bns, theta_nsbh])
        kde = gaussian_kde(xy)
        Z = kde(pos).reshape(X.shape)
        density_samples = kde(xy)
        t68, t95 = _contour_thresholds(density_samples, probs=(0.68, 0.95))

        fig, ax = plt.subplots(figsize=(7.2, 6.0))
        sc = ax.scatter(
            theta_bns,
            theta_nsbh,
            c=fj_bns,
            cmap="plasma",
            alpha=0.18,
            s=10,
            vmin=0.0,
            vmax=FJ_BNS_MAX,
            linewidths=0,
        )
        ax.contour(
            X,
            Y,
            Z,
            levels=[t95, t68],
            colors=["tab:blue", "tab:blue"],
            linewidths=[1.2, 2.2],
        )

        rect = Rectangle(**rect_args)
        ax.add_patch(rect)
        ax.legend([rect], ["RE23 (90% C.I.)"], loc="upper right", fontsize=FS)
        ax.set_title(f"$\\alpha = {alpha}$", fontsize=FS)
        ax.set_xlabel(r"$\\theta_c^{\\mathrm{BNS}}$ [deg]", fontsize=FS)
        ax.set_ylabel(r"$\\theta_c^{\\mathrm{NSBH}}$ [deg]", fontsize=FS)
        ax.tick_params(axis="both", which="major", labelsize=FS - 2)
        ax.grid(True, alpha=0.2)
        ax.set_xlim(1, 25)
        ax.set_ylim(1, 25)
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label(r"$f_j^{\\mathrm{BNS}}$", fontsize=FS - 2)
        plt.savefig(
            f"complete_contours_midpoint_alpha_{alpha}.pdf",
            dpi=150,
            bbox_inches="tight",
        )
        plt.show()

def plot_corner(alphas, burn_frac: float = 0.33, thin: int = 10):
    labels = [
        r"$A$",
        r"$\\log_{10}(L_0)$",
        r"$\\mu_{E,p}$",
        r"$\\sigma_{E,p}$",
        r"$\\theta_c^{\\mathrm{BNS}}$ [deg]",
        r"$\\theta_c^{\\mathrm{NSBH}}$ [deg]",
        r"$f_j^{\\mathrm{BNS}}$",
    ]

    for alpha in alphas:
        backend, flat, backend_path = _load_chain(alpha, burn_frac, thin)
        if flat is None or flat.size == 0:
            print(f"Warning: Results missing or empty for alpha={alpha}")
            continue

        fig = corner.corner(
            flat,
            labels=labels,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 11},
            label_kwargs={"fontsize": 12},
        )
        fig.suptitle(f"$\\alpha = {alpha}$", fontsize=14)
        plt.savefig(
            f"complete_corner_midpoint_alpha_{alpha}.pdf",
            dpi=150,
            bbox_inches="tight",
        )
        plt.show()

def plotting_fractions(alpha_val, theta_threshold=10.0, geom_eff=None):
    pass # Empty placeholder for now, you can copy the fraction plots directly into the notebook if desired
