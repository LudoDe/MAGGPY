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

#rect_args = {'xy': (2.17, 1.0), 'width': 33.8, 'height': 24.0, 'alpha': 0.1, 'color': 'tab:green'}
#plt.axhspan(
#            6.1-3.2, 6.1+9.3, color='gray', alpha=0.3, label='R23'
#        )
#rect args should give same rectangle as axhspan
min_R = 6.1 - 3.2
max_R = 6.1 + 9.3

rect_args = {
    "xy": (min_R, min_R),
    "width": max_R - min_R,
    "height": max_R - min_R,
    "alpha": 0.1,
    "color": "k",
}

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

from scipy import stats
def run_pop(
    baseline_paths, 
    alphas,
    geom_eff_func,
    datafiles=DATAFILES,
    n_params: int = 7,
    n_walkers: int = 24,
    n_steps: int = 10000,
):
    """Run the complete MCMC pipeline with free f_j^{BNS} and spectral parameters."""

    for alpha in alphas:
        test_backend = baseline_paths.get(alpha)
        if test_backend is not None:
            print(f"Found existing backend for alpha={alpha} at {test_backend}. ")

        #test_backend    = baseline_paths["A5.0"]
        test_backend    = baseline_paths.get(alpha)
        samples         = emcee.backends.HDFBackend(test_backend).get_chain(flat=True, discard=3000, thin = 10)
        kernel          = stats.gaussian_kde(samples.T, bw_method=0.2)
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

        k_interpolator  = create_k_interpolator()
        bns_distances   = compute_luminosity_distance(bns_params.z_arr)
        nsbh_distances  = nsbh_data.distances

        backend_path = _backend_path(alpha)

        def log_likelihood(thetas, n_events=N_MC_EVENTS):
            A_index, L_L0, L_mu_E, sigma_E, theta_c_bns, fj_bns, theta_c_nsbh = thetas
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

        def flat_prior(thetas):
            A_index, L_L0, L_mu_E, sigma_E, theta_c_bns, fj_bns, theta_c_nsbh = thetas
            if not (1.5 < A_index < 12)         : return -np.inf
            if not (-2 < L_L0 < 7)              : return -np.inf
            if not (0.1 < L_mu_E < 7)           : return -np.inf
            if not (0 < sigma_E < 2.5)          : return -np.inf 
            if not (1 < theta_c_bns < 50)       : return -np.inf
            if not (0 < fj_bns < FJ_BNS_MAX)    : return -np.inf
            #flat prior for theta_c_nsbh between 1, 50 degrees
            if not (1.0 <= theta_c_nsbh <= 50.0): return -np.inf
            return 0.0

        def log_probability(thetas):
            A_index, L_L0, L_mu_E, sigma_E, theta_c_bns, fj_bns, theta_c_nsbh = thetas

            #add prior values to all 
            lp_f = flat_prior(thetas)
            if not np.isfinite(lp_f):
                return -np.inf, 0.0, 0.0, 0.0, 0.0, 0.0
            
            lp = kernel.logpdf([A_index, L_L0, L_mu_E, sigma_E, theta_c_bns, fj_bns])[0]
            
            if not np.isfinite(lp): return -np.inf, 0.0, 0.0, 0.0, 0.0, 0.0

            ll = log_likelihood(thetas=thetas, n_events=N_MC_EVENTS)
            if not np.isfinite(ll[0]): return -np.inf, 0.0, 0.0, 0.0, 0.0, 0.0

            return lp + ll[0], ll[1], ll[2], ll[3], ll[4], ll[5]

        #samples_for_start = emcee.backends.HDFBackend(test_backend).get_chain()[-1] #(24, 6)
        #get best fitting_sample from previous chain and use as starting point for new chain with free theta_c_nsbh
        backend_start       = emcee.backends.HDFBackend(test_backend)
        samples_for_start   = backend_start.get_chain(discard=3000, thin=10) #(24, 6, N)
        log_prob            = backend_start.get_log_prob(discard=3000, thin=10) #(6, N)
        best_index          = np.argmax(log_prob, axis=0)[-1] # get index of best walker at last step
        best_sample         = samples_for_start[best_index] #(6, )


        #A_index, L0, mu_e, sigma_e, theta_c_bns, fj
        #need to randomize for theta_c_nsbh between theta_c_bns, fj
        rng = np.random.default_rng(123)
        #samples_for_start_augmented = np.hstack([
        #    samples_for_start[:, :5], # first 5 params
        #    rng.uniform(2.0, 25.0, (24, 1)), # theta_c_nsbh
        #    samples_for_start[:, -1].reshape(-1, 1) # fj_bns
        #]) # take last step of previous chain and augment with random theta_c_nsbh and same fj_bns (24, 7)

        theta_c_nsbh_start = rng.uniform(2.0, 25.0, (n_walkers, 1))
        samples_for_start_augmented = np.hstack([best_sample, theta_c_nsbh_start]) # (N_samples, 7)
        # rewrote it so nsbh is last parameter for easier indexing and to avoid issues with hstack and best_sample shape
    

        print(f"Running starting position {samples_for_start_augmented.shape}")
        
        initial_pos, n_steps_rem, backend = check_and_resume_mcmc(
            filename=backend_path,
            n_steps=n_steps,
            starting_point=samples_for_start_augmented,
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
        fj_bns = flat[:, 5]
        theta_nsbh = flat[:, 6]

        #xy              = np.vstack([theta_bns, theta_nsbh])
        #kde             = gaussian_kde(xy)
        #Z               = kde(pos).reshape(X.shape)
        #density_samples = kde(xy)
        #t68, t95        = _contour_thresholds(density_samples, probs=(0.68, 0.95))

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
        from corner import hist2d
        hist2d(
            theta_bns,
            theta_nsbh, 
            ax=ax,
            no_fill_contours=True,
            levels = [0.68, 0.95],
            plot_datapoints=False,
            smooth=1.0,
            plot_density=False,)
        

        rect = Rectangle(**rect_args)
        ax.add_patch(rect)
        ax.legend([rect], ["RE23 (90\% C.I.)"], loc="upper right", fontsize=FS)
        ax.set_title(f"$\\alpha = {alpha}$", fontsize=FS)
        ax.set_xlabel(r"$\theta_c^{\mathrm{BNS}}$ [deg]", fontsize=FS)
        ax.set_ylabel(r"$\theta_c^{\mathrm{NSBH}}$ [deg]", fontsize=FS)
        ax.tick_params(axis="both", which="major", labelsize=FS - 2)
        ax.grid(True, alpha=0.2)
        ax.set_xlim(1, 25)
        ax.set_ylim(1, 25)
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label(r"$f_j^{\mathrm{BNS}}$", fontsize=FS - 2)
        plt.savefig(
            f"complete_contours_midpoint_alpha_{alpha}.pdf",
            dpi=150,
            bbox_inches="tight",
        )
        plt.show()


def plot_populations_splot(alphas, burn_frac: float = 0.33, thin: int = 10, split=1):
    FS = 18
    for alpha in alphas:
        backend, flat, backend_path = _load_chain(alpha, burn_frac, thin)
        if flat is None or flat.size == 0:
            print(f"Warning: Results missing or empty for alpha={alpha}")
            continue

        theta_bns   = flat[:, 4]
        theta_nsbh  = flat[:, 6]
        fj_bns      = flat[:, 5]

        c = ["k", "red"]

        mask_split  = fj_bns < split
        print(sum(mask_split), sum(~mask_split))
        color_array = np.where(mask_split, c[0], c[1])

        fj_leq = fj_bns[mask_split]
        fj_geq = fj_bns[~mask_split]

        fig, ax = plt.subplots(figsize=(7.2, 6.0))
        from corner import hist2d
        scatter_args = {
            "s": 10,
            "alpha": 0.18,
            "linewidths": 0,
        }
        hist_args = {
            "no_fill_contours": True,
            "levels": [0.68, 0.95],
            "plot_datapoints": False,
            "smooth": 1.0,
            "plot_density": False,
        }
        ax.scatter(theta_bns[mask_split], theta_nsbh[mask_split], c=c[0], **scatter_args, label=f"$f_j^{{BNS}} < {split}$")
        ax.scatter(theta_bns[~mask_split], theta_nsbh[~mask_split], c=c[1], **scatter_args, label=f"$f_j^{{BNS}} \\geq {split}$")
        hist2d(theta_bns[mask_split], theta_nsbh[mask_split], ax=ax, color=c[0], **hist_args)
        hist2d(theta_bns[~mask_split], theta_nsbh[~mask_split], ax=ax, color=c[1], **hist_args)

        rect = Rectangle(**rect_args, label="RE23 (90\% C.I.)")
        ax.add_patch(rect)
        ax.legend(loc="upper right", fontsize=FS-4)
        ax.set_title(f"$\\alpha = {alpha}$", fontsize=FS)
        ax.set_xlabel(r"$\theta_c^{\mathrm{BNS}}$ [deg]", fontsize=FS)
        ax.set_ylabel(r"$\theta_c^{\mathrm{NSBH}}$ [deg]", fontsize=FS)
        ax.tick_params(axis="both", which="major", labelsize=FS - 4)
        ax.grid(True, alpha=0.2)
        ax.set_xlim(1, 25)
        ax.set_ylim(1, 50)
        #cb = fig.colorbar(sc, ax=ax)
        #cb.set_label(r"$f_j^{\mathrm{BNS}}$", fontsize=FS - 2)
        plt.savefig(
            f"complete_contours_midpoint_alpha_{alpha}.pdf",
            dpi=150,
            bbox_inches="tight",
        )
        plt.show()

from corner import hist2d
def plot_populations_grid(alphas, burn_frac: float = 0.33, thin: int = 10):
    fig, ax = plt.subplots(1, len(alphas), figsize=(7.2 * len(alphas), 6.0), sharey=True)
    for i, alpha in enumerate(alphas):
        backend, flat, backend_path = _load_chain(alpha, burn_frac, thin)
        if flat is None or flat.size == 0:
            print(f"Warning: Results missing or empty for alpha={alpha}")
            continue

        theta_bns = flat[:, 4]
        theta_nsbh = flat[:, 6]
        fj_bns = flat[:, 5]

        hist_args = {
            "no_fill_contours": True,
            "levels": [0.68, 0.95],
            "plot_datapoints": False,
            "smooth": 1.5,
            "plot_density": False,
        }
        fj_mask = fj_bns < 1.0
        hist2d(
            theta_bns[fj_mask],
            theta_nsbh[fj_mask], 
            ax=ax[i],
            color="k",
            **hist_args
        )
        hist2d(
            theta_bns[~fj_mask],
            theta_nsbh[~fj_mask], 
            ax=ax[i],
            color="red",
            **hist_args
        )

        rect = Rectangle(**rect_args)
        ax[i].add_patch(rect)
        labels = [f"$f_j^{{BNS}} < 1.0$", f"$f_j^{{BNS}} \\geq 1.0$", "RE23 (90\% C.I.)"]
        rect_patches = [Rectangle((0, 0), 1, 1, color=c, alpha=0.18) for c in ["k", "red"]]
        handles = rect_patches + [rect]
        if i == 0:ax[i].legend(handles, labels, loc="upper right", fontsize=18)
        ax[i].set_title(f"$\\alpha = {alpha}$", fontsize=18)
        ax[i].set_xlabel(r"$\theta_c^{\mathrm{BNS}}$ [deg]", fontsize=18)
        if i == 0:
            ax[i].set_ylabel(r"$\theta_c^{\mathrm{NSBH}}$ [deg]", fontsize=18)
        ax[i].tick_params(axis="both", which="major", labelsize=16)
        ax[i].grid(True, alpha=0.2)
        ax[i].set_xlim(1, 25)
        ax[i].set_ylim(1, 50)
    plt.savefig(
        f"complete_contours_midpoint_grid.pdf",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()


labels = [
    r"$A$",
    r"$\log_{10}(L_0)$",
    r"$\mu_{E,p}$",
    r"$\sigma_{E,p}$",
    r"$\theta_c^{\mathrm{BNS}}$ [deg]",
    r"$f_j^{\mathrm{BNS}}$",
    r"$\theta_c^{\mathrm{NSBH}}$ [deg]",
]
def plot_corner(alphas, burn_frac: float = 0.33, thin: int = 10):

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


def plot_corner_cut(alphas, burn_frac: float = 0.33, thin: int = 10, fj_cut = 1.0, log_fj=False):

    for alpha in alphas:
        backend, flat, backend_path = _load_chain(alpha, burn_frac, thin)
        if flat is None or flat.size == 0:
            print(f"Warning: Results missing or empty for alpha={alpha}")
            continue

        mask        = flat[:, 5] < fj_cut
        flat_leq    = flat[mask]
        flat_geq    = flat[~mask] 
        color_leq   = "k"
        color_geq   = "red"

        #adjust geq by weights
        weights = np.ones(len(flat_geq)) * (len(flat_leq) / len(flat_geq)) 

        label_in = labels
        if log_fj:
            flat_leq[:, 5] = np.log10(flat_leq[:, 5] + 1e-3) # add small value to avoid log(0)
            flat_geq[:, 5] = np.log10(flat_geq[:, 5] + 1e-3)
            label_in[5] = r"$\log_{10}(f_j^{\mathrm{BNS}})$"

        fig_leq = corner.corner(
            flat_leq,
            labels=label_in,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            color=color_leq,
            title_kwargs={"fontsize": 11},
            label_kwargs={"fontsize": 12},

        )
        fig_geq = corner.corner(
            flat_geq,
            labels=label_in,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 11},
            label_kwargs={"fontsize": 12},
            color=color_geq,
            fig=fig_leq,
            weights=weights
        )

        rect_labels = [f"$f_j^{{BNS}} < {fj_cut}$", f"$f_j^{{BNS}} \\geq {fj_cut}$"]
        rect_colors = [color_leq, color_geq]
        rect_patches = [Rectangle((0, 0), 1, 1, color=c, alpha=0.18) for c in rect_colors]
        handles = rect_patches
        fig_leq.legend(handles, rect_labels, loc="upper right", fontsize=12)

        fig_leq.suptitle(f"$\\alpha = {alpha}$", fontsize=14)
        plt.savefig(
            f"complete_corner_midpoint_alpha_{alpha}_comp.pdf",
            dpi=150,
            bbox_inches="tight",
        )
        plt.show()


def plotting_fractions(alpha_val, theta_threshold=10.0, geom_eff=None):
    pass # Empty placeholder for now, you can copy the fraction plots directly into the notebook if desired
