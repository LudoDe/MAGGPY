from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import corner
from astropy.cosmology import Planck18, FlatLambdaCDM
import src.init
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.interpolate import interp1d
from src.nsbh.init import initialize_combined_simulation
from src.nsbh.montecarlo import GBM_EFF, FJ_NSBH_FIXED, N_MC_EVENTS
from src.top_hat.montecarlo import (
    apply_detection_cuts,
    compute_luminosity_distance,
    create_k_interpolator,
    score_func_cvm,
)
from scipy.stats import cramervonmises_2samp, poisson

import src.init
from src.nsbh.init import initialize_combined_simulation
from src.top_hat.montecarlo import create_k_interpolator
from src.nsbh.montecarlo import GBM_EFF, FJ_NSBH_FIXED, N_MC_EVENTS

DATAFILES = Path("../../../datafiles")
FJ_BNS_MAX = 10.0

#FJ_BNS = 1

labels = [
    r"$\log_{10}(L_0)$",
    r"$\log_{10}(k_{\mathrm{NSBH}})$",
    r"$\theta_c^{\mathrm{BNS}}$ [deg]",
    r"$\theta_c^{\mathrm{NSBH}}$ [deg]",
]
bin_ranges = {
    0: (1, 5),             # L_L0
    1: (-2, 0),             # k_nsbh
    2: (1, 25),             # theta_c_bns
    3: (1, 25),             # theta_c_nsbh
}

def lognormal_numpy(mu, sigma, n, rng):
    l10 = np.log(10)
    return rng.lognormal(
        mean=mu*l10,
        sigma=sigma*l10,
        size=n
    )

def luminosity_gen(A, n, rng):
    shape = (A - 1)/A
    return rng.gamma(shape, size=n)**(-1/A)

def compute_luminosity_distance(z, cosmology=FlatLambdaCDM(H0=Planck18.H0, Om0=Planck18.Om0)):
    """Compute luminosity distance in cm."""
    return cosmology.luminosity_distance(z).cgs.value

# physical efficiency, pflux distribution only depends on L0, P_z_interp

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

def generate_base_population(
    data_obj,
    redshift_ppf,
    k_interpolator,
    L_mu_E,
    sigma_E,
    A_index,
    n_events
):
    rng = np.random.default_rng(123)
    z_samples = redshift_ppf(rng.uniform(size=n_events))
    d_L = compute_luminosity_distance(z_samples)
    
    # Base energies
    E_p_rest = lognormal_numpy(L_mu_E, sigma_E, n_events, rng=rng)
    E_p_obs = E_p_rest / (1 + z_samples)
    
    # Base luminosity scatter
    lum_scatter = luminosity_gen(A_index, n_events, rng=rng)
    
    # K-correction
    pts = np.column_stack((np.log10(E_p_obs), z_samples))
    k_corr = k_interpolator(pts)
    
    # Base p_flux calculated lacking only the 10**(L_L0 + 49) scaling factor
    base_flux = lum_scatter / (4 * np.pi * d_L**2 * k_corr) * 6.242e8
    
    return {
        "base_flux": base_flux,
        "E_p_obs": E_p_obs,
        "z": z_samples
    }

from scipy.interpolate import interp1d

def build_efficiency_interpolator(base_pop, pflux_min=4.0):
    L0_grid         = np.linspace(-2, 7, 100) # prior space
    efficiencies    = np.zeros_like(L0_grid)
    
    for i, L0 in enumerate(L0_grid):
        scaled_flux = base_pop["base_flux"] * 10**(L0 + 49)
        # Only require trigger mask for physical efficiency
        trigger_mask = scaled_flux > pflux_min
        efficiencies[i] = np.mean(trigger_mask)
        
    return interp1d(
        L0_grid, 
        efficiencies, 
        kind="cubic",
        bounds_error=False,
        fill_value=(0.0, 1.0)
    )

def prepare_interpolators(
    data_bns, data_nsbh, k_interpolator, L_mu_E, sigma_E, A_index, n_events
):
    ppf_bns     = build_redshift_ppf(data_bns.P_z_interp)
    ppf_nsbh    = build_redshift_ppf(data_nsbh.P_z_interp)

    base_bns = generate_base_population(
        data_bns, ppf_bns, k_interpolator, L_mu_E, sigma_E, A_index, n_events
    )
    base_nsbh = generate_base_population(
        data_nsbh, ppf_nsbh, k_interpolator, L_mu_E, sigma_E, A_index, n_events
    )
    
    eff_interp_bns = build_efficiency_interpolator(base_bns)
    eff_interp_nsbh = build_efficiency_interpolator(base_nsbh)
    
    return base_bns, base_nsbh, eff_interp_bns, eff_interp_nsbh

def forward_model_fast(
    thetas,
    bns_params,
    nsbh_data,
    base_bns,
    base_nsbh,
    eff_interp_bns,
    eff_interp_nsbh,
    geom_eff_func,
    FJ_BNS,
):
    L_L0, k_nsbh_log, theta_c_bns, theta_c_nsbh = thetas

    geom_eff_bns    = geom_eff_func(theta_c_bns)
    geom_eff_nsbh   = geom_eff_func(theta_c_nsbh)

    epsilon_bns     = geom_eff_bns  * FJ_BNS
    epsilon_nsbh    = geom_eff_nsbh * FJ_NSBH_FIXED

    intrinsic_bns   = epsilon_bns   * len(bns_params.z_arr) * GBM_EFF
    intrinsic_nsbh  = epsilon_nsbh  * len(nsbh_data.z_arr) * GBM_EFF

    # Fast detection samples using precomputed base arrays
    scaled_flux_bns = base_bns["base_flux"] * 10**(L_L0 + 49)
    bns_trig, bns_analysis = apply_detection_cuts(
        scaled_flux_bns, base_bns["E_p_obs"]
    )
    
    scaled_flux_nsbh = base_nsbh["base_flux"] * 10**(L_L0 + k_nsbh_log + 49)
    nsbh_trig, nsbh_analysis = apply_detection_cuts(
        scaled_flux_nsbh, base_nsbh["E_p_obs"]
    )

    pflux_det = np.concatenate([
        scaled_flux_bns[bns_analysis],
        scaled_flux_nsbh[nsbh_analysis],
    ])

    epeak_det = np.concatenate([
        base_bns["E_p_obs"][bns_analysis],
        base_nsbh["E_p_obs"][nsbh_analysis],
    ])

    # Interpolated efficiencies replace the manual sum
    phys_eff_bns = float(eff_interp_bns(L_L0))
    phys_eff_nsbh = float(eff_interp_nsbh(L_L0 + k_nsbh_log))

    mu_bns = intrinsic_bns * bns_params.triggered_years * phys_eff_bns
    mu_nsbh = intrinsic_nsbh * bns_params.triggered_years * phys_eff_nsbh
    mu_total = mu_bns + mu_nsbh

    return {
        "pflux": pflux_det,
        "epeak": epeak_det,
        "mu_bns": mu_bns,
        "mu_nsbh": mu_nsbh,
        "mu_total": mu_total,
        "rate_per_year": mu_total / bns_params.triggered_years,
    }

def setup_forward_ppc(
    alpha,
    geom_eff_func,
    datafiles=DATAFILES,
    fixed_params=None,
    n_events=N_MC_EVENTS * 3,
):
    demo_params = {
        "z_model": f"fiducial_delayed_{alpha}",
    }

    fixed_params_alpha = fixed_params[alpha]
    L_mu_E = fixed_params_alpha["L_mu_E"]
    sigma_E = fixed_params_alpha["sigma_E"]
    A_index = fixed_params_alpha["A_index"]

    bns_params, nsbh_data, _ = initialize_combined_simulation(
        datafiles=datafiles,
        params=demo_params,
        nsbh_population="fiducial_delayed_cut",
        nsbh_alpha=alpha,
    )

    k_interpolator = create_k_interpolator()

    base_bns, base_nsbh, eff_bns, eff_nsbh = prepare_interpolators(
        data_bns=bns_params,
        data_nsbh=nsbh_data,
        k_interpolator=k_interpolator,
        L_mu_E=L_mu_E,
        sigma_E=sigma_E,
        A_index=A_index,
        n_events=n_events,
    )

    def forward_model_ppc(thetas):
        return forward_model_fast(
            thetas=thetas,
            bns_params=bns_params,
            nsbh_data=nsbh_data,
            base_bns=base_bns,
            base_nsbh=base_nsbh,
            eff_interp_bns=eff_bns,
            eff_interp_nsbh=eff_nsbh,
            geom_eff_func=geom_eff_func,
        )

    observed = {
        "pflux": bns_params.pflux_data,
        "epeak": bns_params.epeak_data,
        "yearly_rate": bns_params.yearly_rate,
        "triggered_years": bns_params.triggered_years,
    }

    return forward_model_ppc, observed

def plot_ppc(
    theta_draws,
    forward_model_ppc,
    observed,
):
    from scipy.stats import gaussian_kde

    sims = [forward_model_ppc(theta) for theta in theta_draws]

    fig, axs = plt.subplots(
        1,
        3,
        figsize=(8, 3),
        constrained_layout=True,
    )

    for sim in sims:
        cut_pflux = sim["pflux"][sim["pflux"] > 4]
        x = np.sort(cut_pflux)
        y = np.arange(1, len(x) + 1) / len(x)

        axs[0].plot(
            x,
            y,
            alpha=0.1,
            color="C0",
        )

    x = np.sort(observed["pflux"])
    y = np.arange(1, len(x) + 1) / len(x)

    axs[0].plot(
        x,
        y,
        color="k",
        lw=2,
    )

    axs[0].set_xscale("log")
    axs[0].set_xlim(4, max(observed["pflux"]))
    axs[0].set_ylim(0, 1)
    axs[0].set_xlabel(r"$P_{\mathrm{flux}}$ [ph/cm$^2$/s]")

    for sim in sims:
        cut_epeak = sim["epeak"][
            (sim["epeak"] > 50) & (sim["epeak"] < max(observed["epeak"]))
        ]
        x = np.sort(cut_epeak)
        y = np.arange(1, len(x) + 1) / len(x)

        axs[1].plot(
            x,
            y,
            alpha=0.1,
            color="C1",
        )

    x = np.sort(observed["epeak"])
    y = np.arange(1, len(x) + 1) / len(x)

    axs[1].plot(
        x,
        y,
        color="k",
        lw=2,
    )

    axs[1].set_xscale("log")
    axs[1].set_xlim(50, max(observed["epeak"]))
    axs[1].set_ylim(0, 1)
    axs[1].set_xlabel(r"$E_{\mathrm{peak}}$ [keV]")

    rates = np.array([sim["rate_per_year"] for sim in sims])

    axs[2].hist(
        rates,
        bins=20,
        density=True,
    )

    axs[2].axvline(
        observed["yearly_rate"],
        color="k",
        lw=2,
    )

    axs[2].set_xlabel(r"Yearly Rate [yr$^{-1}$]")
    plt.show()

    fig2, axs2 = plt.subplots(figsize=(5, 5))

    x_all, y_all = [], []

    for sim in sims:
        mask = (
            (sim["pflux"] > 4)
            & (sim["epeak"] > 50)
            & (sim["epeak"] < max(observed["epeak"]))
        )

        x_all.append(np.log10(sim["pflux"][mask]))
        y_all.append(np.log10(sim["epeak"][mask]))

    x_all = np.concatenate(x_all)
    y_all = np.concatenate(y_all)

    kde = gaussian_kde(np.vstack([x_all, y_all]))

    xx, yy = np.mgrid[
        x_all.min() : x_all.max() : 100j,
        y_all.min() : y_all.max() : 100j,
    ]
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    zsort = np.sort(zz.ravel())[::-1]
    cdf = np.cumsum(zsort)
    cdf /= cdf[-1]

    level_1sigma = zsort[np.searchsorted(cdf, 0.68)]
    level_2sigma = zsort[np.searchsorted(cdf, 0.95)]

    axs2.contour(
        10**xx,
        10**yy,
        zz,
        levels=[level_2sigma, level_1sigma],
        colors="C2",
        linewidths=2,
    )

    axs2.scatter(
        observed["pflux"],
        observed["epeak"],
        color="k",
        label="Observed",
    )

    axs2.set_xscale("log")
    axs2.set_yscale("log")
    axs2.set_xlabel(r"$P_{\mathrm{flux}}$ [ph/cm$^2$/s]")
    axs2.set_ylabel(r"$E_{\mathrm{peak}}$ [keV]")
    axs2.set_xlim(4, max(observed["pflux"]))
    axs2.set_ylim(50, max(observed["epeak"]))
    axs2.legend()
    plt.show()

    fig3, axs3 = plt.subplots(figsize=(5, 5))

    for sim in sims:
        mask_pflux = sim["pflux"] > 4
        mask_epeak = (sim["epeak"] > 50) & (
            sim["epeak"] < max(observed["epeak"])
        )
        mask = mask_pflux & mask_epeak
        
        axs3.scatter(
            sim["pflux"][mask],
            sim["epeak"][mask],
            alpha=0.1,
            color="C2",
        )

    axs3.scatter(
        observed["pflux"],
        observed["epeak"],
        color="k",
        label="Observed",
    )

    axs3.set_xscale("log")
    axs3.set_yscale("log")
    axs3.set_xlabel(r"$P_{\mathrm{flux}}$ [ph/cm$^2$/s]")
    axs3.set_ylabel(r"$E_{\mathrm{peak}}$ [keV]")
    axs3.set_xlim(4, max(observed["pflux"]))
    axs3.set_ylim(50, max(observed["epeak"]))

    plt.legend()
    plt.show()

    return fig3, axs3


def score_func_cvm(y_sim, y_obs):
    """Evaluates score using exact values without resampling."""
    y_in = np.log10(y_sim)
    y_out = np.log10(y_obs)
    return np.log(cramervonmises_2samp(y_in, y_out).pvalue)

def run_pop_a(
    alphas,
    geom_eff_func,
    datafiles,
    FJ_BNS,
    n_l0    =   50,
    n_k     =   40,
    n_tbns  =   80,
    n_tnsbh =   80,
    fixed_params=None,
    nsbh_population="fiducial_delayed_cut",
    **kwargs,
):
    """
    Evaluates the factored likelihood across a 4D grid directly.
    Unused MCMC arguments resolve seamlessly via kwargs.
    """
    out_paths = []
    local_rate_dict = {}

    for alpha in alphas:
        demo_params         = {"z_model": f"fiducial_delayed_{alpha}"}
        fixed_params_alpha  = fixed_params[alpha]
        L_mu_E              = fixed_params_alpha["L_mu_E"]
        sigma_E             = fixed_params_alpha["sigma_E"]
        A_index             = fixed_params_alpha["A_index"]

        bns_params, nsbh_data, _ = initialize_combined_simulation(
            datafiles=datafiles,
            params=demo_params,
            nsbh_population=nsbh_population,
            nsbh_alpha=alpha,
        )

        local_rate_bns          = bns_params.local_rate
        local_rate_nsbh         = nsbh_data.local_rate
        local_rate_dict[alpha]  = (local_rate_bns, local_rate_nsbh)

        k_interpolator = create_k_interpolator()

        base_bns, base_nsbh, eff_bns, eff_nsbh = prepare_interpolators(
            data_bns=bns_params,
            data_nsbh=nsbh_data,
            k_interpolator=k_interpolator,
            L_mu_E=L_mu_E,
            sigma_E=sigma_E,
            A_index=A_index,
            n_events=N_MC_EVENTS * 2,
        )

        l0_grid     = np.linspace(-1, 5, n_l0)
        k_grid      = np.linspace(-2, 0, n_k)
        tbns_grid   = np.linspace(1, 25, n_tbns)
        tnsbh_grid  = np.linspace(1, 25, n_tnsbh)

        logL_cvm    = np.zeros((n_l0, n_k))

        for i, l0 in enumerate(l0_grid):
            for j, k_val in enumerate(k_grid):
                # A dummy expected geometry bypasses calculation dependencies
                # to extract raw observable arrays.
                sim = forward_model_fast(
                    thetas=[l0, k_val, 10.0, 25.0],
                    bns_params=bns_params,
                    nsbh_data=nsbh_data,
                    base_bns=base_bns,
                    base_nsbh=base_nsbh,
                    eff_interp_bns=eff_bns,
                    eff_interp_nsbh=eff_nsbh,
                    geom_eff_func=geom_eff_func,
                    FJ_BNS=FJ_BNS
                )

                if len(sim["pflux"]) <= 3:
                    logL_cvm[i, j] = -np.inf
                    continue

                cvm_pflux = score_func_cvm(sim["pflux"], bns_params.pflux_data)
                cvm_epeak = score_func_cvm(sim["epeak"], bns_params.epeak_data)
                logL_cvm[i, j] = cvm_pflux + cvm_epeak



        # Vectorize geometry operations utilizing array broadcasting
        l0_2d = l0_grid[:, None]
        k_2d = k_grid[None, :]

        phys_eff_bns_arr = eff_bns(l0_grid)[:, None, None, None]
        phys_eff_nsbh_arr = eff_nsbh(l0_2d + k_2d)[:, :, None, None]

        fj_bns_test = FJ_BNS
        fj_nsbh_test = FJ_NSBH_FIXED

        c_bns = (
            fj_bns_test
            * len(bns_params.z_arr)
            * GBM_EFF
            * bns_params.triggered_years
        )
        c_nsbh = (
            fj_nsbh_test
            * len(nsbh_data.z_arr)
            * GBM_EFF
            * bns_params.triggered_years
        )

        geom_bns_arr = geom_eff_func(tbns_grid)[None, None, :, None]
        geom_nsbh_arr = geom_eff_func(tnsbh_grid)[None, None, None, :]

        mu_bns_4d       = c_bns * geom_bns_arr * phys_eff_bns_arr
        mu_nsbh_4d      = c_nsbh * geom_nsbh_arr * phys_eff_nsbh_arr
        mu_total_4d     = mu_bns_4d + mu_nsbh_4d

        observed_total  = bns_params.yearly_rate * bns_params.triggered_years
        logL_poisson_4d = poisson.logpmf(observed_total, mu_total_4d)

        logL_total = logL_cvm[:, :, None, None] + logL_poisson_4d
        
        #append fj of bns too so that the file name is unique for each run
        #add the geometry efficiency function name to the file name for clarity
        run_name = f"grid_likelihood_alpha_{alpha}_{FJ_BNS}_{geom_eff_func.__name__}"
        # if the nsbh_population is not default include the _{ending} to the file name for clarity
        if nsbh_population != "fiducial_delayed_cut":
            run_name += f"_{nsbh_population.split('_')[-1]}"
        
        
        output_dir = src.init.create_run_dir(
            run_name, output_files_default="Output_files"
        )
        output_file = output_dir / "grid_eval.npz"

        np.savez_compressed(
            output_file,
            l0=l0_grid,
            k=k_grid,
            theta_bns=tbns_grid,
            theta_nsbh=tnsbh_grid,
            logL_cvm=logL_cvm,
            logL_poiss=logL_poisson_4d,
            logL_total=logL_total,
            mu_bns=mu_bns_4d,
            mu_nsbh=mu_nsbh_4d,
        )

        print(f"Grid tensor computed for alpha {alpha}.")
        print(f"Results archived at: {output_file}")
        out_paths.append(output_file)

    return out_paths, local_rate_dict

def draw_grid_samples(grid_path: Path, n_samples: int = 40_000):
    """
    Transforms the 4D evaluated grid tensor back into a continuous MCMC-style
    trace array through probability mass sampling and uniform jitter.
    """
    data = np.load(grid_path)
    l0_grid = data["l0"]
    k_grid = data["k"]
    tbns_grid = data["theta_bns"]
    tnsbh_grid = data["theta_nsbh"]
    logL = data["logL_total"]
    mu_bns = data["mu_bns"]
    mu_nsbh = data["mu_nsbh"]

    # Flatten the probability space and prevent overflow.
    flat_logL = logL.ravel()
    valid = np.isfinite(flat_logL)
    flat_logL_clean = flat_logL[valid]
    # maximum value of flat_logL_clean
    print(f"Maximum log-likelihood value: {np.max(flat_logL_clean)}")

    flat_logL_clean -= np.max(flat_logL_clean)
    prob_clean = np.exp(flat_logL_clean)
    prob_clean /= np.sum(prob_clean)


    # Draw discrete grid coordinates based on the likelihood weights.
    chosen_indices_clean = np.random.choice(
        len(flat_logL_clean), size=n_samples, p=prob_clean
    )
    
    # Map valid flattened indices back to original flattened indices.
    original_indices = np.where(valid)[0][chosen_indices_clean]
    coords = np.unravel_index(original_indices, logL.shape)

    samples = np.column_stack(
        [
            l0_grid[coords[0]],
            k_grid[coords[1]],
            tbns_grid[coords[2]],
            tnsbh_grid[coords[3]],
        ]
    )

    # Compute step sizes to smear discrete intervals into continuous ones.
    dl0 = l0_grid[1] - l0_grid[0]
    dk = k_grid[1] - k_grid[0]
    dtbns = tbns_grid[1] - tbns_grid[0]
    dtnsbh = tnsbh_grid[1] - tnsbh_grid[0]

    jitter = np.random.uniform(
        low=[-dl0 / 2, -dk / 2, -dtbns / 2, -dtnsbh / 2],
        high=[dl0 / 2, dk / 2, dtbns / 2, dtnsbh / 2],
        size=(n_samples, 4),
    )

    # Broadcast the rate arrays to match the full 4D grid shape
    mu_bns_full = np.broadcast_to(mu_bns, logL.shape)
    mu_nsbh_full = np.broadcast_to(mu_nsbh, logL.shape)

    # Pull the rates corresponding to the drawn coordinates
    samples_mu_bns = mu_bns_full[coords]
    samples_mu_nsbh = mu_nsbh_full[coords]

    return samples + jitter, samples_mu_bns, samples_mu_nsbh

def plot_corner_grid(alphas, FJ_BNS, labels=labels, bin_ranges=bin_ranges, nsbh_population=None, geom_eff_func=None):
    """
    Renders pure corner plots using upsampled grid representations.
    Omit the fj_bns bounding box logic since the parameter was fixed.
    """
    for alpha in alphas:
        infile = f"grid_likelihood_alpha_{alpha}_{FJ_BNS}_{geom_eff_func.__name__}"
        if nsbh_population is not None: 
            infile += f"_{nsbh_population.split('_')[-1]}"
        grid_file = (
            src.init.create_run_dir(
                infile,
                output_files_default="Output_files",
            )
            / "grid_eval.npz"
        )
        if not grid_file.exists():
            print(f"Bypassing alpha={alpha}; target grid file not located.")
            continue

        samples, samples_mu_bns, samples_mu_nsbh = draw_grid_samples(grid_file)

        corner_args = {
            "labels": labels,
            "quantiles": [0.16, 0.5, 0.84],
            "show_titles": True,
            "title_kwargs": {"fontsize": 11},
            "label_kwargs": {"fontsize": 12},
            "bins": 15,
            "smooth": 1,
            "smooth1d": 1,
            "range": [bin_ranges[i] for i in range(len(labels))],
            "plot_datapoints": False,
            "plot_density": True,
            #"fill_contours": True,
            "levels": [0.68, 0.95],
        }

        fig = corner.corner(samples, color="k", **corner_args)
        fig.suptitle(f"$\\alpha = {alpha}$ Grid Projected", fontsize=14)
        plt.savefig(
            f"complete_corner_grid_alpha_{alpha}.pdf",
            dpi=150,
            bbox_inches="tight",
        )
        plt.show()

def plot_corner_grid_multiples(alphas, nsbh_populations, FJ_BNS,local_rate_dicts=None, geom_eff_func=None, labels=labels, bin_ranges=bin_ranges):
    """
    Renders pure corner plots using upsampled grid representations.
    Omit the fj_bns bounding box logic since the parameter was fixed.
    """
    for alpha in alphas:
        infile_base = f"grid_likelihood_alpha_{alpha}_{FJ_BNS}_{geom_eff_func.__name__}"
        infiles     = []
        for nsbh_population in nsbh_populations:
            infile = infile_base + f"_{nsbh_population.split('_')[-1]}"
            infiles.append(infile)
        grid_files = [
            src.init.create_run_dir(
                infile,
                output_files_default="Output_files",
            )
            / "grid_eval.npz"
            for infile in infiles
        ]

        samples_0, samples_mu_bns_0, samples_mu_nsbh_0 = draw_grid_samples(grid_files[0])
        samples_1, samples_mu_bns_1, samples_mu_nsbh_1 = draw_grid_samples(grid_files[1])

        corner_args = {
            "labels": labels,
            "quantiles": [0.16, 0.5, 0.84],
            "show_titles": True,
            "title_kwargs": {"fontsize": 11},
            "label_kwargs": {"fontsize": 12},
            "bins"  : 20,
            "smooth": 1,
            "smooth1d": 1,
            "range": [bin_ranges[i] for i in range(len(labels))],
            "plot_datapoints": False,
            "plot_density": True,
            #"fill_contours": True,
            "levels": [0.68, 0.95],
        }

        fig = corner.corner(samples_0, color="k", **corner_args)
        fig_comp = corner.corner(samples_1, color="C1", fig=fig, **corner_args)
        # add labels with local rates if provided
        if local_rate_dicts is not None:
            bns_rate = local_rate_dicts[0].get(alpha, (None, None))[0] # always the same for both populations

            dict_0 = local_rate_dicts[0]
            dict_1 = local_rate_dicts[1]

            nsbh_local_rate_0 = dict_0.get(alpha, (None, None))[1]
            nsbh_local_rate_1 = dict_1.get(alpha, (None, None))[1]

            # nsbh handles 0 is sfho and 1 is dd2, Gpc^-3 yr^-1
            # For the bns there is no line as it is not plotted just need it for reference
            legend_handles = [
                plt.Line2D([0], [0], color="k", lw=2, label=f"SFHo: $R_{{\\rm NSBH}}={nsbh_local_rate_0:.2f}\,\mathrm{{Gpc}}^{{-3}}\,\mathrm{{yr}}^{{-1}}$"),
                plt.Line2D([0], [0], color="C1", lw=2, label=f"DD2: $R_{{\\rm NSBH}}={nsbh_local_rate_1:.2f}\,\mathrm{{Gpc}}^{{-3}}\,\mathrm{{yr}}^{{-1}}$")
            ]
            fig_comp.legend(handles=legend_handles, fontsize=16, 
                            title=f"$R_{{\\rm BNS}}={bns_rate:.2f}\,\mathrm{{Gpc}}^{{-3}}\,\mathrm{{yr}}^{{-1}}$", 
                            title_fontsize=16, loc="upper right")

        fig.suptitle(f"$\\alpha = {alpha}$", fontsize=14)
        plt.savefig(
            f"complete_corner_grid_alpha_{alpha}.pdf",
            dpi=150,
            bbox_inches="tight",
        )
        plt.show()

def posterior_rate_contribution_corner_plot(
    alphas,
    nsbh_populations,
    FJ_BNS,
    local_rate_dicts=None,
    geom_eff_func=None,
    bin_ranges=[[0, 25], [0, 25]],
):
    """Renders corner plots of the rate posteriors for both populations."""
    rate_labels = [
        r"$R_{\rm BNS} \ [\mathrm{yr}^{-1}]$",
        r"$R_{\rm NSBH} \ [\mathrm{yr}^{-1}]$",
    ]

    for alpha in alphas:
        infile_base = f"grid_likelihood_alpha_{alpha}_{FJ_BNS}_{geom_eff_func.__name__}"
        infiles = []
        for nsbh_population in nsbh_populations:
            infile = infile_base + f"_{nsbh_population.split('_')[-1]}"
            infiles.append(infile)
        grid_files = [
            src.init.create_run_dir(
                infile,
                output_files_default="Output_files",
            )
            / "grid_eval.npz"
            for infile in infiles
        ]

        _, samples_mu_bns_0, samples_mu_nsbh_0 = draw_grid_samples(grid_files[0])
        _, samples_mu_bns_1, samples_mu_nsbh_1 = draw_grid_samples(grid_files[1])

        bns_params, _, _ = initialize_combined_simulation(
            datafiles=DATAFILES,
            params={"z_model": f"fiducial_delayed_{alpha}"},
            nsbh_population=nsbh_populations[0],
            nsbh_alpha=alpha,
        )
        triggered_years = bns_params.triggered_years

        rates_0 = np.column_stack(
            [
                samples_mu_bns_0 / triggered_years,
                samples_mu_nsbh_0 / triggered_years,
            ]
        )

        rates_1 = np.column_stack(
            [
                samples_mu_bns_1 / triggered_years,
                samples_mu_nsbh_1 / triggered_years,
            ]
        )

        corner_args = {
            "labels": rate_labels,
            "quantiles": [0.16, 0.5, 0.84],
            "show_titles": True,
            "title_kwargs": {"fontsize": 11},
            "label_kwargs": {"fontsize": 12},
            "bins": 20,
            "smooth": 1,
            "smooth1d": 1,
            "range": bin_ranges,
            "plot_datapoints": False,
            "plot_density": True,
            "levels": [0.68, 0.95],
        }

        # Plot first population (SFHo)
        fig = corner.corner(rates_0, color="k", **corner_args)

        # Overlay second population (DD2)
        fig_comp = corner.corner(rates_1, color="C1", fig=fig, **corner_args)

        legend_handles = [
            plt.Line2D([0], [0], color="k", lw=2, label="SFHo"),
            plt.Line2D([0], [0], color="C1", lw=2, label="DD2"),
        ]
        fig_comp.legend(
            handles=legend_handles,
            fontsize=12,
            loc="upper right",
            bbox_to_anchor=(0.9, 0.9),
        )

        fig.suptitle(f"$\\alpha = {alpha}$ Rate Contributions", fontsize=14)
        plt.savefig(
            f"posterior_rate_corner_alpha_{alpha}.pdf",
            dpi=150,
            bbox_inches="tight",
        )
        plt.show()


def posterior_rate_contribution_corner_plot(
    alphas,
    nsbh_populations,
    FJ_BNS,
    local_rate_dicts=None,
    geom_eff_func=None,
    bin_ranges=[[0, 1], [0, 1]],
):
    """Renders corner plots of the rate posteriors for both populations."""
    rate_labels = [
        r"$R_{\rm BNS} / R_{\rm total} \ [\mathrm{yr}^{-1}]$",
        r"$R_{\rm NSBH} / R_{\rm total} \ [\mathrm{yr}^{-1}]$",
    ]

    for alpha in alphas:
        infile_base = f"grid_likelihood_alpha_{alpha}_{FJ_BNS}_{geom_eff_func.__name__}"
        infiles = []
        for nsbh_population in nsbh_populations:
            infile = infile_base + f"_{nsbh_population.split('_')[-1]}"
            infiles.append(infile)
        grid_files = [
            src.init.create_run_dir(
                infile,
                output_files_default="Output_files",
            )
            / "grid_eval.npz"
            for infile in infiles
        ]

        _, samples_mu_bns_0, samples_mu_nsbh_0 = draw_grid_samples(grid_files[0])
        _, samples_mu_bns_1, samples_mu_nsbh_1 = draw_grid_samples(grid_files[1])

        bns_params, _, _ = initialize_combined_simulation(
            datafiles=DATAFILES,
            params={"z_model": f"fiducial_delayed_{alpha}"},
            nsbh_population=nsbh_populations[0],
            nsbh_alpha=alpha,
        )
        triggered_years = bns_params.triggered_years

        rates_0 = np.column_stack(
            [
                samples_mu_bns_0 / triggered_years,
                samples_mu_nsbh_0 / triggered_years,
            ]
        )

        rates_1 = np.column_stack(
            [
                samples_mu_bns_1 / triggered_years,
                samples_mu_nsbh_1 / triggered_years,
            ]
        )

        corner_args = {
            "labels": rate_labels,
            "quantiles": [0.16, 0.5, 0.84],
            "show_titles": True,
            "title_kwargs": {"fontsize": 11},
            "label_kwargs": {"fontsize": 12},
            "bins": 20,
            "smooth": 1,
            "smooth1d": 1,
            "range": bin_ranges,
            "plot_datapoints": False,
            "plot_density": True,
            "levels": [0.68, 0.95],
        }

        # Plot first population (SFHo)
        total_rate_0 = np.sum(rates_0, axis=1)
        rates_0 /= total_rate_0[:, None]  # Normalize to get fractions
        total_rate_1 = np.sum(rates_1, axis=1)
        rates_1 /= total_rate_1[:, None]  # Normalize to get fractions

        fig = corner.corner(rates_0, color="k", **corner_args)

        # Overlay second population (DD2)
        fig_comp = corner.corner(rates_1, color="C1", fig=fig, **corner_args)

        legend_handles = [
            plt.Line2D([0], [0], color="k", lw=2, label="SFHo"),
            plt.Line2D([0], [0], color="C1", lw=2, label="DD2"),
        ]
        fig_comp.legend(
            handles=legend_handles,
            fontsize=12,
            loc="upper right",
            bbox_to_anchor=(0.9, 0.9),
        )

        fig.suptitle(f"$\\alpha = {alpha}$ Rate Contributions", fontsize=14)
        plt.savefig(
            f"posterior_rate_corner_alpha_{alpha}.pdf",
            dpi=150,
            bbox_inches="tight",
        )
        plt.show()