from pathlib import Path
import emcee
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy import stats
import corner

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

from scipy.stats import norm

from scipy.stats import gamma

#def luminosity_gen_crn(A, u):
#    return gengamma.ppf(
#        u,
#        a=(A - 1)/A,
#        c=-A,
#    )

# Optimized formulation
def luminosity_gen_crn(A, u):
    shape = (A - 1) / A
    y = gamma.ppf(u, shape)
    return y ** (-1 / A)

def make_mc_catalog(n_events, params, seed):
    rng = np.random.default_rng(seed)
    u_E = rng.random(n_events)
    # Precompute norm.ppf outside the likelihood loop
    z_norm = norm.ppf(u_E)

    return {
        "u_L": rng.random(n_events),
        "z_norm": z_norm,
        "id_z": rng.integers(
            0,
            len(params.z_arr),
            size=n_events,
        ),
    }

from scipy.interpolate import RectBivariateSpline
from scipy.stats import gengamma

def create_gengamma_ppf_interpolator(a_min=1.4, a_max=5.1, n_a=100, n_u=500):
    """Precomputes a 2D interpolator for the gengamma PPF."""
    a_grid = np.linspace(a_min, a_max, n_a)
    u_grid = np.linspace(1e-6, 1 - 1e-6, n_u)
    
    z_values = np.zeros((n_a, n_u))
    for i, a_val in enumerate(a_grid):
        z_values[i, :] = gengamma.ppf(
            u_grid,
            a=(a_val - 1) / a_val,
            c=-a_val,
        )
    
    return RectBivariateSpline(a_grid, u_grid, z_values)
gengamma_interp = create_gengamma_ppf_interpolator()

def luminosity_gen_crn_interp(A, u):
    # Pack A into a coordinate array matching the size of u
    a_coords = np.full_like(u, A)
    return gengamma_interp.ev(a_coords, u)

def simplified_montecarlo_crn(
    thetas,
    catalog,
    params_in,
    distances,
    k_interpolator,
):

    A_index, L_L0, L_mu_E_10, sigma_E_10 = thetas[:4]

    ln10 = np.log(10)

    # luminosities
    #L_draw = luminosity_gen_crn(
    #    A_index,
    #    catalog["u_L"],
    #)
    L_draw = luminosity_gen_crn_interp(A_index, catalog["u_L"])

    L_obs_iso = L_draw * 10**(L_L0 + 49)

    # lognormal from fixed uniforms
    #z_norm = norm.ppf(catalog["u_E"])
    z_norm = catalog["z_norm"]

    E_p_rest = np.exp(
        L_mu_E_10 * ln10
        + sigma_E_10 * ln10 * z_norm
    )

    # fixed redshift selection
    id_z = catalog["id_z"]

    z_arr = params_in.z_arr[id_z]

    d_L_sq = distances[id_z]**2

    E_p_obs = E_p_rest / (1 + z_arr)

    #k_corr = k_interpolator.ev(
    #    np.log10(E_p_obs),
    #    z_arr,
    #)

    #uses regular grid interpolator so the args change 
    pts = np.column_stack((np.log10(E_p_obs), z_arr))
    k_corr = k_interpolator(pts)

    p_flux = (
        L_obs_iso
        / (4 * np.pi * d_L_sq * k_corr)
        * 6.242e8
    )

    return {
        "p_flux": p_flux,
        "E_p_obs": E_p_obs,
        "z_arr": z_arr,
        "L_p_obs": L_obs_iso,
    }

def run_pop(
    alphas,
    geom_eff_func,
    datafiles=DATAFILES,
    n_walkers   : int = 20,
    n_steps = 30_000,
    fixed_params = None,
):
    """Run the complete MCMC pipeline with free f_j^{BNS} and spectral parameters."""

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

        k_interpolator  = create_k_interpolator()
        bns_distances   = compute_luminosity_distance(bns_params.z_arr)
        nsbh_distances  = nsbh_data.distances

        backend_path = _backend_path(alpha)

        total_merger_rate_bns   = bns_params.total_merger_rate
        total_merger_rate_nsbh  = nsbh_data.total_merger_rate
        years_data              = 1/2 # hyper parameter

        #n_events_bns            = int(total_merger_rate_bns * years_data)
        #n_events_nsbh           = int(total_merger_rate_nsbh * years_data)
        #print(f"Running MCMC for alpha={alpha} with {n_events_bns} BNS events and {n_events_nsbh} NSBH events.")

        n_events_bns = n_events_nsbh = N_MC_EVENTS

        bns_catalog = make_mc_catalog(
            n_events_bns,
            bns_params,
            seed=1234,
        )

        nsbh_catalog = make_mc_catalog(
            n_events_nsbh,
            nsbh_data,
            seed=5678,
        )

        def log_likelihood(thetas):
            #A_index, L_L0, L_mu_E, sigma_E, theta_c_bns, fj_bns, theta_c_nsbh = thetas
            A_index, L_L0, theta_c_bns, fj_bns, theta_c_nsbh = thetas
            grb_thetas = [A_index, L_L0, L_mu_E, sigma_E]

            geom_eff_bns    = geom_eff_func(theta_c_bns)
            geom_eff_nsbh   = geom_eff_func(theta_c_nsbh)

            epsilon_bns     = geom_eff_bns * fj_bns
            epsilon_nsbh    = geom_eff_nsbh * FJ_NSBH_FIXED

            intrinsic_bns_grb   = epsilon_bns * len(bns_params.z_arr) * GBM_EFF
            intrinsic_nsbh_grb  = epsilon_nsbh * len(nsbh_data.z_arr) * GBM_EFF

            bns_results = simplified_montecarlo_crn(
                grb_thetas, bns_catalog, bns_params, bns_distances, k_interpolator
            )
            bns_trig, bns_analysis = apply_detection_cuts(
                bns_results["p_flux"], bns_results["E_p_obs"]
            )

            nsbh_results = simplified_montecarlo_crn(
                grb_thetas,
                nsbh_catalog,
                nsbh_data,
                nsbh_distances,
                k_interpolator,
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

            phys_eff_bns = np.sum(bns_trig) / n_events_bns
            phys_eff_nsbh = np.sum(nsbh_trig) / n_events_nsbh

            predicted_bns   = intrinsic_bns_grb * bns_params.triggered_years * phys_eff_bns
            predicted_nsbh  = intrinsic_nsbh_grb * bns_params.triggered_years * phys_eff_nsbh
            predicted_total = predicted_bns + predicted_nsbh
            observed_total  = bns_params.yearly_rate * bns_params.triggered_years

            if predicted_total <= 0:
                return -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf

            logL_poisson = poiss_log(k=observed_total, mu=predicted_total)
            logL_total = logL_pflux + logL_epeak + logL_poisson

            return logL_total, logL_pflux, logL_epeak, logL_poisson, predicted_bns, predicted_nsbh

        def flat_prior(thetas):
            A_index, L_L0, theta_c_bns, fj_bns, theta_c_nsbh = thetas
            if not (1.5 < A_index       < 5)               : return -np.inf
            if not (-2  < L_L0          < 7)                : return -np.inf
            #if not (1   < L_mu_E        < 7)               : return -np.inf
            #if not (0   < sigma_E       < 2)               : return -np.inf 
            if not (1   < theta_c_bns   < 25)               : return -np.inf
            if not (0   < fj_bns        < FJ_BNS_MAX)       : return -np.inf
            if not (1   < theta_c_nsbh  < 50)               : return -np.inf
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

        #random positions for the walkers, within the prior bounds
        #samples_for_start = rng.uniform(
        #    low=[1.5, 2.5, 3, 0.15, 1, 0, 1],
        #    high=[12, 3.5, 4, 0.40, 50, FJ_BNS_MAX, 50],
        #    size=(n_walkers, n_params),
        #)
        n_params = len(labels)
        samples_for_start = rng.uniform(
            low=    [1.5, -2, 1,  0,            1],
            high=   [5  , 6,  25, FJ_BNS_MAX,   50],
            size=   (n_walkers, n_params),
        )

        n_steps_in = n_steps
        # if n_steps is array with different steps for each alpha, use n_steps_in = n_steps_alpha[i]
        if isinstance(n_steps, list):
            n_steps_in = n_steps[alphas.index(alpha)]

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
        )

def plot_populations(alphas, burn_frac: float = 0.33, thin: int = 10):
    FS = 18
    for alpha in alphas:
        _, flat, _ = _load_chain(alpha, burn_frac, thin)
        if flat is None or flat.size == 0:
            print(f"Warning: Results missing or empty for alpha={alpha}")
            continue

        theta_bns = flat[:, 4]
        fj_bns = flat[:, 5]
        theta_nsbh = flat[:, 6]

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
        _, flat, _ = _load_chain(alpha, burn_frac, thin)
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
        _, flat, _ = _load_chain(alpha, burn_frac, thin)
        if flat is None or flat.size == 0:
            print(f"Warning: Results missing or empty for alpha={alpha}")
            continue

        theta_bns   = flat[:, 4]
        theta_nsbh  = flat[:, 6]
        fj_bns      = flat[:, 5]

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
    #r"$\mu_{E,p}$",
    #r"$\sigma_{E,p}$",
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

bin_ranges = {
    0: (1.5, 5),         # A_index
    1: (-2, 6),        # L_L0
    #2: (0.1, 7),            # L_mu_E
    #3: (0, 2.5),      # sigma_E
    2: (1, 25),           # theta_c_bns
    3: (0, FJ_BNS_MAX),   # fj_bns
    4: (1, 50),           # theta_c_nsbh
}

def plot_corner_cut(alphas, burn_frac: float = 0.33, thin: int = 10, fj_cut = 1.0, log_fj=False):

    for alpha in alphas:
        _, flat, _ = _load_chain(alpha, burn_frac, thin)
        if flat is None or flat.size == 0:
            print(f"Warning: Results missing or empty for alpha={alpha}")
            continue

        mask        = flat[:, 3] < fj_cut #! careful index has changed due to removal of L_mu_E and sigma_E
        flat_leq    = flat[mask]
        flat_geq    = flat[~mask] 
        color_leq   = "k"
        color_geq   = "red"

        #adjust geq by weights
        weights = np.ones(len(flat_geq)) * (len(flat_leq) / len(flat_geq)) 

        label_in = labels
        if log_fj:
            flat_leq[:, 3] = np.log10(flat_leq[:, 3] + 1e-3) # add small value to avoid log(0)
            flat_geq[:, 3] = np.log10(flat_geq[:, 3] + 1e-3)
            label_in[3] = r"$\log_{10}(f_j^{\mathrm{BNS}})$"

        corner_args = {
            "labels"            : label_in,
            "quantiles"         : [0.16, 0.5, 0.84],
            "show_titles"       : True,
            "title_kwargs"      : {"fontsize": 11},
            "label_kwargs"      : {"fontsize": 12},
            "bins"              : 15,
            "smooth"            : 1.0,
            "range"             : [bin_ranges[i] for i in range(len(labels))],
            "plot_datapoints"   : False,
            "plot_density"      : False,
            "fill_contours"     : True,
            "levels"            : [0.68, 0.95],
        }

        weights_geq = np.ones(len(flat_geq)) * (len(flat_leq) / len(flat_geq))

        fig_leq = corner.corner(
            flat_leq,
            color=color_leq,
            **corner_args
        )
        fig_geq = corner.corner(
            flat_geq,
            color=color_geq,
            fig = fig_leq,
            weights=weights_geq,
            **corner_args
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