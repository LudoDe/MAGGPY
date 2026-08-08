"""Two-stage NSBH runs for the lognormal geometry with free f_j^{BNS}."""

from pathlib import Path

import emcee
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.stats import gaussian_kde, lognorm

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

# The notebook sets the working directory to this folder, so these relative
# paths resolve against the same project layout used by the fixed-f_j notebooks.
DATAFILES = Path("../../datafiles")
FJ_BNS_MAX = 10.0
DEFAULT_SIGMA_THETA_C = 0.5

def _run_name_stage1(alpha: str) -> str:
    return f"lognormal_fj_free_stage1_bns_only_alpha_{alpha}"

def _run_name_stage2(alpha: str) -> str:
    return f"lognormal_fj_free_stage2_bns_plus_nsbh_alpha_{alpha}"

def _backend_path(alpha: str, stage: int) -> Path:
    run_name = _run_name_stage1(alpha) if stage == 1 else _run_name_stage2(alpha)
    return src.init.create_run_dir(run_name, output_files_default="Output_files") / "emcee.h5"

def _load_stage2_chain(alpha: str, burn_frac: float, thin: int):
    backend_path = _backend_path(alpha, stage=2)
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
    n_params_1: int = 6,
    n_walkers_1: int = 24,
    n_steps_1: int = 5000,
    n_params_2: int = 3,
    n_walkers_2: int = 24,
    n_steps_2: int = 5000,
    burn_frac: float = 0.33,
    thin: int = 10,
):
    """Run the two-stage MCMC pipeline with free f_j^{BNS}.

    Stage 1 samples the spectral parameters, theta_c^{BNS}, and f_j^{BNS}.
    Stage 2 freezes the spectral parameters at the Stage-1 medians and samples
    theta_c^{BNS}, theta_c^{NSBH}, and f_j^{BNS}.
    """

    for alpha in alphas:
        print("\n============================================================")
        print(f"Running MCMC pipeline for alpha = {alpha}")
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

        # =========================================================
        # STAGE 1: BNS ONLY, free f_j^{BNS}
        # =========================================================
        backend_path_stage1 = _backend_path(alpha, stage=1)

        def log_likelihood_stage1_bns_only(thetas, n_events=N_MC_EVENTS):
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
                return -np.inf, -np.inf, -np.inf, -np.inf, -np.inf

            logL_pflux = score_func_cvm(pflux_det, bns_params.pflux_data, bns_params.rng)
            logL_epeak = score_func_cvm(epeak_det, bns_params.epeak_data, bns_params.rng)

            phys_eff_bns = np.sum(bns_trig) / n_events
            predicted_bns = intrinsic_bns * bns_params.triggered_years * phys_eff_bns
            observed_total = bns_params.yearly_rate * bns_params.triggered_years

            if predicted_bns <= 0:
                return -np.inf, -np.inf, -np.inf, -np.inf, -np.inf

            logL_poisson = poiss_log(k=observed_total, mu=predicted_bns)
            logL_total = logL_pflux + logL_epeak + logL_poisson

            return logL_total, logL_pflux, logL_epeak, logL_poisson, predicted_bns

        def log_prior_stage1(thetas):
            A_index, L_L0, L_mu_E, sigma_E, theta_c_bns, fj_bns = thetas
            if not (1.5 < A_index < 12):
                return -np.inf
            if not (-2 < L_L0 < 7):
                return -np.inf
            if not (0.1 < L_mu_E < 7):
                return -np.inf
            if not (0 < sigma_E < 2.5):
                return -np.inf
            if not (1 < theta_c_bns < 50):
                return -np.inf
            if not (0.01 < fj_bns <= FJ_BNS_MAX):
                return -np.inf
            return 0.0

        def init_walkers_stage1(n_walkers, seed=123):
            rng = np.random.default_rng(seed)
            return np.column_stack(
                [
                    rng.uniform(2.0, 3.5, n_walkers),
                    rng.uniform(2.0, 4.5, n_walkers),
                    rng.uniform(1.5, 4.5, n_walkers),
                    rng.uniform(0.2, 1.2, n_walkers),
                    rng.uniform(3.0, 20.0, n_walkers),
                    rng.uniform(0.1, min(5.0, FJ_BNS_MAX), n_walkers),
                ]
            )

        def log_probability_stage1(thetas):
            lp = log_prior_stage1(thetas)
            if not np.isfinite(lp):
                return -np.inf, 0.0, 0.0, 0.0, 0.0

            ll = log_likelihood_stage1_bns_only(thetas=thetas, n_events=N_MC_EVENTS)
            if not np.isfinite(ll[0]):
                return -np.inf, 0.0, 0.0, 0.0, 0.0

            return lp + ll[0], ll[1], ll[2], ll[3], ll[4]

        initial_pos_1, n_steps_rem_1, backend_1 = check_and_resume_mcmc(
            filename=backend_path_stage1,
            n_steps=n_steps_1,
            initialize_walkers_func=init_walkers_stage1,
            n_walkers=n_walkers_1,
        )

        run_mcmc(
            log_probability_func=log_probability_stage1,
            initial_walkers=initial_pos_1,
            n_iterations=n_steps_rem_1,
            n_walkers=n_walkers_1,
            n_params=n_params_1,
            backend=backend_1,
            blobs_dtype=[
                ("l_pflux", float),
                ("l_epeak", float),
                ("l_poiss", float),
                ("mu_bns", float),
            ],
        )

        flat_1 = backend_1.get_chain(
            discard=int(backend_1.iteration * burn_frac),
            thin=thin,
            flat=True,
        )
        if flat_1.size == 0:
            print(f"Warning: empty Stage 1 chain for alpha={alpha}")
            continue

        medians_1 = np.median(flat_1, axis=0)
        spectral_fixed = medians_1[:4].copy()
        theta_bns_stage1_med = medians_1[4]
        fj_bns_stage1_med = medians_1[5]

        print(f"Stage 1 theta_c^BNS median = {theta_bns_stage1_med:.3f} deg")
        print(f"Stage 1 f_j^BNS median     = {fj_bns_stage1_med:.3f}")

        # =========================================================
        # STAGE 2: BNS + NSBH, free f_j^{BNS}
        # =========================================================
        backend_path_stage2 = _backend_path(alpha, stage=2)

        def log_likelihood_stage2_fixed_spectral(thetas, spectral_fixed, n_events=N_MC_EVENTS):
            theta_c_bns, theta_c_nsbh, fj_bns = thetas
            A_index, L_L0, L_mu_E, sigma_E = spectral_fixed
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

        def log_prior_stage2(thetas):
            theta_c_bns, theta_c_nsbh, fj_bns = thetas
            if not (1 < theta_c_bns < 50):
                return -np.inf
            if not (1 < theta_c_nsbh < 50):
                return -np.inf
            if not (0.01 < fj_bns <= FJ_BNS_MAX):
                return -np.inf
            return 0.0

        def init_walkers_stage2(n_walkers, seed=321):
            rng = np.random.default_rng(seed)
            theta_bns_low = max(1.2, theta_bns_stage1_med * 0.6)
            theta_bns_high = min(45.0, theta_bns_stage1_med * 1.4)
            if not theta_bns_low < theta_bns_high:
                theta_bns_low, theta_bns_high = 3.0, 20.0

            fj_low = max(0.05, fj_bns_stage1_med * 0.6)
            fj_high = min(FJ_BNS_MAX, fj_bns_stage1_med * 1.4)
            if not fj_low < fj_high:
                fj_low, fj_high = 0.1, min(5.0, FJ_BNS_MAX)

            return np.column_stack(
                [
                    rng.uniform(theta_bns_low, theta_bns_high, n_walkers),
                    rng.uniform(2.0, 25.0, n_walkers),
                    rng.uniform(fj_low, fj_high, n_walkers),
                ]
            )

        def log_probability_stage2(thetas):
            lp = log_prior_stage2(thetas)
            if not np.isfinite(lp):
                return -np.inf, 0.0, 0.0, 0.0, 0.0, 0.0

            ll = log_likelihood_stage2_fixed_spectral(
                thetas=thetas,
                spectral_fixed=spectral_fixed,
                n_events=N_MC_EVENTS,
            )
            if not np.isfinite(ll[0]):
                return -np.inf, 0.0, 0.0, 0.0, 0.0, 0.0

            return lp + ll[0], ll[1], ll[2], ll[3], ll[4], ll[5]

        initial_pos_2, n_steps_rem_2, backend_2 = check_and_resume_mcmc(
            filename=backend_path_stage2,
            n_steps=n_steps_2,
            initialize_walkers_func=init_walkers_stage2,
            n_walkers=n_walkers_2,
        )

        run_mcmc(
            log_probability_func=log_probability_stage2,
            initial_walkers=initial_pos_2,
            n_iterations=n_steps_rem_2,
            n_walkers=n_walkers_2,
            n_params=n_params_2,
            backend=backend_2,
            blobs_dtype=[
                ("l_pflux", float),
                ("l_epeak", float),
                ("l_poiss", float),
                ("mu_bns", float),
                ("mu_nsbh", float),
            ],
        )

        print(f"Stage 2 complete for alpha={alpha}")


def _contour_thresholds(density_values, probs=(0.68, 0.95)):
    vals = np.sort(density_values)[::-1]
    csum = np.cumsum(vals)
    csum /= csum[-1]
    return [vals[np.searchsorted(csum, prob)] for prob in probs]


def plot_populations(alphas, burn_frac: float = 0.33, thin: int = 10):
    """Plot the 2D angle posterior with KDE contours and f_j^{BNS} coloring."""

    FS = 18
    x = np.linspace(1, 25, 80)
    y = np.linspace(1, 25, 80)
    X, Y = np.meshgrid(x, y)
    pos = np.vstack([X.ravel(), Y.ravel()])

    for alpha in alphas:
        backend, flat, backend_path = _load_stage2_chain(alpha, burn_frac, thin)
        if flat is None:
            print(f"Warning: Results missing for alpha={alpha}")
            continue

        if flat.size == 0:
            print(f"Warning: empty Stage 2 chain for alpha={alpha}")
            continue

        theta_bns = flat[:, 0]
        theta_nsbh = flat[:, 1]
        fj_bns = flat[:, 2]

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
            f"free_fj_stage2_contours_lognormal_alpha_{alpha}.pdf",
            dpi=150,
            bbox_inches="tight",
        )
        plt.show()


def plot_corner(alphas, burn_frac: float = 0.33, thin: int = 10):
    """Plot a corner-style summary for theta_c^{BNS}, theta_c^{NSBH}, and f_j^{BNS}."""

    labels = [
        r"$\\theta_c^{\\mathrm{BNS}}$ [deg]",
        r"$\\theta_c^{\\mathrm{NSBH}}$ [deg]",
        r"$f_j^{\\mathrm{BNS}}$",
    ]

    for alpha in alphas:
        backend, flat, backend_path = _load_stage2_chain(alpha, burn_frac, thin)
        if flat is None:
            print(f"Warning: Results missing for alpha={alpha}")
            continue

        if flat.size == 0:
            print(f"Warning: empty Stage 2 chain for alpha={alpha}")
            continue

        samples = [flat[:, 0], flat[:, 1], flat[:, 2]]

        fig, axes = plt.subplots(3, 3, figsize=(10.5, 10.0))
        for row in range(3):
            for col in range(3):
                ax = axes[row, col]
                if row < col:
                    ax.axis("off")
                    continue

                if row == col:
                    ax.hist(samples[row], bins=30, color="tab:blue", alpha=0.7, density=True)
                    ax.set_yticks([])
                else:
                    x = samples[col]
                    y = samples[row]
                    xy = np.vstack([x, y])
                    kde = gaussian_kde(xy)
                    x_grid = np.linspace(np.min(x), np.max(x), 80)
                    y_grid = np.linspace(np.min(y), np.max(y), 80)
                    X, Y = np.meshgrid(x_grid, y_grid)
                    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
                    density_samples = kde(xy)
                    t68, t95 = _contour_thresholds(density_samples, probs=(0.68, 0.95))
                    ax.contour(
                        X,
                        Y,
                        Z,
                        levels=[t95, t68],
                        colors=["tab:blue", "tab:blue"],
                        linewidths=[1.0, 2.0],
                    )
                    ax.scatter(x, y, s=4, color="tab:blue", alpha=0.08, linewidths=0)

                if row == 2:
                    ax.set_xlabel(labels[col])
                else:
                    ax.set_xticklabels([])

                if col == 0:
                    ax.set_ylabel(labels[row])
                else:
                    ax.set_yticklabels([])

        fig.suptitle(rf"Free $f_j^{{\mathrm{{BNS}}}}$ corner plot: $\alpha = {alpha}$", y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.985])
        plt.savefig(
            f"free_fj_corner_lognormal_alpha_{alpha}.pdf",
            dpi=150,
            bbox_inches="tight",
        )
        plt.show()


RE_lims = [6.1, -3.2, +9.3]
RE_median = RE_lims[0]
RE_low = RE_median + RE_lims[1]
RE_high = RE_median + RE_lims[2]
rect_args = {
    "xy": (RE_low, RE_low),
    "width": RE_high - RE_low,
    "height": RE_high - RE_low,
    "color": "k",
    "alpha": 0.2,
}


def plotting_fractions(alpha_val, theta_threshold, geom_eff, rouco_flag=True):
    """Plot tail fractions and NSBH contribution for the free-f_j posterior."""

    alphas = [alpha_val] if isinstance(alpha_val, str) else list(alpha_val)
    n_rows = len(alphas)

    fig1, axes1 = plt.subplots(
        n_rows,
        1,
        figsize=(7.0, max(4.5, 3.5 * n_rows)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    fig2, axes2 = plt.subplots(
        n_rows,
        1,
        figsize=(7.0, max(4.5, 3.5 * n_rows)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    sc1 = None
    sc2 = None

    for i, alpha_diag in enumerate(alphas):
        demo_params = {
            "alpha": -0.6,
            "beta_s": -2.5,
            "n": 2.0,
            "theta_c": 3.4,
            "theta_v_max": 10.0,
            "z_model": f"fiducial_delayed_{alpha_diag}",
        }
        bns_params, nsbh_data, _ = initialize_combined_simulation(
            datafiles=DATAFILES,
            params=demo_params,
            size_test=2_000,
            nsbh_population="fiducial_delayed_cut",
            nsbh_alpha=alpha_diag,
        )
        N_bns = len(bns_params.z_arr)
        N_nsbh = len(nsbh_data.z_arr)

        backend_path = _backend_path(alpha_diag, stage=2)
        if not backend_path.exists():
            print(f"Warning: Data missing for alpha={alpha_diag}")
            continue

        backend = emcee.backends.HDFBackend(backend_path)
        burn_in = int(backend.iteration * 0.33)
        flat_samples = backend.get_chain(discard=burn_in, thin=10, flat=True)

        if flat_samples.size == 0:
            print(f"Warning: Empty chain for alpha={alpha_diag}")
            continue

        theta_bns_2     = flat_samples[:, 0]
        theta_nsbh_2    = flat_samples[:, 1]
        fj_bns_2        = flat_samples[:, 2]

        eff_bns         = np.array([geom_eff(t) for t in theta_bns_2])
        eff_nsbh        = np.array([geom_eff(t) for t in theta_nsbh_2])

        total_bns_yr_samp   = eff_bns * fj_bns_2 * N_bns
        total_nsbh_yr_samp  = eff_nsbh * FJ_NSBH_FIXED * N_nsbh

        min_tc = 1.0
        max_tc = 45.0
        shape_param = 0.5 * np.log(10.0)
        mu_bns = theta_bns_2
        mu_nsbh = theta_nsbh_2

        cdf_max_bns_s = lognorm.cdf(max_tc, s=shape_param, scale=mu_bns)
        cdf_min_bns_s = lognorm.cdf(min_tc, s=shape_param, scale=mu_bns)
        cdf_thr_bns_s = lognorm.cdf(theta_threshold, s=shape_param, scale=mu_bns)
        norm_bns_s = cdf_max_bns_s - cdf_min_bns_s
        tail_bns_s = np.zeros_like(norm_bns_s)
        m_ok = norm_bns_s > 1e-12
        tail_bns_s[m_ok] = (cdf_max_bns_s[m_ok] - cdf_thr_bns_s[m_ok]) / norm_bns_s[m_ok]

        cdf_max_ns_s = lognorm.cdf(max_tc, s=shape_param, scale=mu_nsbh)
        cdf_min_ns_s = lognorm.cdf(min_tc, s=shape_param, scale=mu_nsbh)
        cdf_thr_ns_s = lognorm.cdf(theta_threshold, s=shape_param, scale=mu_nsbh)
        norm_ns_s = cdf_max_ns_s - cdf_min_ns_s
        tail_ns_s = np.zeros_like(norm_ns_s)
        m_ok_ns = norm_ns_s > 1e-12
        tail_ns_s[m_ok_ns] = (cdf_max_ns_s[m_ok_ns] - cdf_thr_ns_s[m_ok_ns]) / norm_ns_s[m_ok_ns]

        above_bns = total_bns_yr_samp * tail_bns_s
        above_nsbh = total_nsbh_yr_samp * tail_ns_s
        above_samp = above_bns + above_nsbh
        total_events_samp = total_bns_yr_samp + total_nsbh_yr_samp

        frac_samp_total = np.zeros_like(above_samp)
        nz = total_events_samp > 0
        frac_samp_total[nz] = above_samp[nz] / total_events_samp[nz]

        nsbh_contribution = np.zeros_like(total_nsbh_yr_samp)
        nsbh_contribution[nz] = total_nsbh_yr_samp[nz] / total_events_samp[nz]

        ax1 = axes1[i, 0]
        ax2 = axes2[i, 0]

        sc1 = ax1.scatter(
            theta_bns_2,
            theta_nsbh_2,
            c=frac_samp_total,
            cmap="coolwarm",
            alpha=0.7,
            s=25,
            vmin=0,
            vmax=1,
        )
        sc2 = ax2.scatter(
            theta_bns_2,
            theta_nsbh_2,
            c=nsbh_contribution,
            cmap="viridis",
            alpha=0.7,
            s=25,
            vmin=0,
            vmax=1,
        )

        ax1.set_xlim(1, 25)
        ax1.set_ylim(1, 25)
        ax2.set_xlim(1, 25)
        ax2.set_ylim(1, 25)

        for ax in (ax1, ax2):
            if rouco_flag: ax.add_patch(Rectangle(**rect_args))

        ax1.set_title(f"$\\alpha={alpha_diag}$")
        ax2.set_title(f"$\\alpha={alpha_diag}$")
        ax1.set_ylabel(r"$\theta_c^{\mathrm{NSBH}}$ (deg)")
        ax2.set_ylabel(r"$\theta_c^{\mathrm{NSBH}}$ (deg)")
        if i == n_rows - 1:
            ax1.set_xlabel(r"$\theta_c^{\mathrm{BNS}}$ (deg)")
            ax2.set_xlabel(r"$\theta_c^{\mathrm{BNS}}$ (deg)")
        #ax2.set_xlabel(r"$\theta_c^{\mathrm{BNS}}$ (deg)")

    if sc1 is not None and sc2 is not None:
        fig1.subplots_adjust(right=0.88, hspace=0.15)
        cb_ax1 = fig1.add_axes([0.9, 0.15, 0.02, 0.7])
        fig1.colorbar(sc1, cax=cb_ax1, label=f"Fraction $\\theta_c > {theta_threshold}^\\circ$")

        fig2.subplots_adjust(right=0.88, hspace=0.15)
        cb_ax2 = fig2.add_axes([0.9, 0.15, 0.02, 0.7])
        fig2.colorbar(sc2, cax=cb_ax2, label="NSBH / (BNS + NSBH)")

    if rouco_flag and sc1 is not None:
        for axes in (axes1, axes2):
            ax_tr = axes[0, 0]
            ax_tr.legend([ax_tr.patches[-1]], ["RE23 (90% C.I.)"], loc="upper right", fontsize=10)

    plt.show()
