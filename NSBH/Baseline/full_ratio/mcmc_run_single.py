from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping, Sequence

import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np

import src.init
from maggpy.nsbh.init import initialize_combined_simulation
from maggpy.nsbh.montecarlo import FJ_NSBH_FIXED, GBM_EFF, N_MC_EVENTS
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
from src_local.shared_functions import (
    _backend_path, _geometry_name, 
    _bad_likelihood, _make_initial_walkers, BLOBS_DTYPE,
    flat_prior
)

DATAFILES   = Path("../..../datafiles")

# All available parameters
ALL_PARAMETER_NAMES = (
    "A_index",
    "L_L0",
    "log10_kappa_nsbh",
    "L_mu_E",
    "sigma_E",
    "theta_c_bns",
    "theta_c_nsbh",
)

# Spectral parameters that are fixed in the current analysis
FIXED_PARAMETER_NAMES = (
    "A_index", 
    "L_mu_E",
    "sigma_E",
)

# BNS only parameters, we fix fj_bns
BNS_ONLY_PARAMETER_NAMES = (
    "A_index",
    "L_L0",
    "L_mu_E",
    "sigma_E",
    "theta_c_bns",
)

# NSBH + BNS parameters, not including the fixed parameters
BNS_NSBH_PARAMETER_NAMES = (
    "L_L0",
    "log10_kappa_nsbh",
    "theta_c_bns",
    "theta_c_nsbh",
)

N_PARAMS_SINGLE = 5
LABELS = (
    r"$A$",
    r"$\log_{10}(L_0)$",
    r"$\log_{10}(\kappa_{\rm NSBH})$",
    r"$\mu_{E,p}$",
    r"$\sigma_{E,p}$",
    r"$\theta_c^{\mathrm{BNS}}$ [deg]",
    r"$\theta_c^{\mathrm{NSBH}}$ [deg]",
)

# Open bounds, matching the support used by the original complete runner.
PRIOR_BOUNDS = np.asarray(
    [
        (1.5, 6),       # A
        (-2.0, 7.0),    # L_L0
        (0.1, 7.0),     # L_mu_E
        (0.0, 2.5),     # sigma_E
        (1, 25),        # theta_c_bns
    ],
    dtype=float,
)
DEFAULT_INITIAL_CENTER_SINGLE  = np.asarray([2, 1, 1, 1, 5])
DEFAULT_INITIAL_SCALE_SINGLE   = np.asarray([0.30, 0.25, 0.15, 0.15, 0.08])

def run_pop(
    alphas          : Sequence[str],
    fj_bns          : float,
    geom_eff_func   : Callable[[float], float],
    datafiles       : Path = DATAFILES,
    population      : str = "delayed",
    nsbh_series     : str = "NSBH_DD2_uniform_chi_0_1",
    n_params        : int = N_PARAMS_SINGLE,
    n_walkers       : int = 24,
    n_steps         : int = 10_000,
    n_events        : int = N_MC_EVENTS,
    sample_size     : int | None = None,
    seed            : int = 123,
    initial_center  : Sequence[float] | None = DEFAULT_INITIAL_CENTER_SINGLE,
    initial_scale   : Sequence[float] | None = DEFAULT_INITIAL_SCALE_SINGLE,
) -> list[Path]:
    """Run the full joint likelihood for each alpha at fixed ``fj_bns``.

    Existing matching HDF backends are resumed by ``check_and_resume_mcmc``.
    The starting ensemble for a new backend is drawn around
    ``initial_center`` and is constrained only by :func:`flat_prior`.
    """
    if n_params != N_PARAMS_SINGLE: raise ValueError(f"This model samples exactly {N_PARAMS_SINGLE} parameters, not {n_params}")
    if not np.isfinite(fj_bns) or fj_bns <= 0: raise ValueError("fj_bns must be a positive finite value fixed for the entire run")
    if n_events <= 0 or n_steps < 0: raise ValueError("n_events must be positive and n_steps cannot be negative")

    output_paths: list[Path] = []

    for alpha_index, alpha in enumerate(alphas):
        population_sample_size = (
            n_events
            if sample_size is None
            else max(sample_size, n_events)
        )

        bns_data, _, observations = initialize_combined_simulation(
            datafiles=Path(datafiles),
            population=population,
            alpha=alpha,
            nsbh_series=nsbh_series,
            sample_size=population_sample_size,
            seed=seed + alpha_index,
        ) # first runs are BNS only then BNS + NSBH only 

        k_interpolator  = create_k_interpolator()
        bns_distances   = compute_luminosity_distance(bns_data.z_arr)
        #nsbh_distances  = nsbh_data.distances
        backend_label = f"{population}_{nsbh_series}"

        backend_path = _backend_path(
            alpha,
            fj_bns,
            geom_eff_func,
            backend_label,
        )

        def log_likelihood_single(thetas: Sequence[float]):
            bns_likelihood_rng = np.random.default_rng(seed)
            a_index, l_l0, l_mu_e, sigma_e, theta_c_bns = thetas
            bns_grb_thetas      = [a_index, l_l0, l_mu_e, sigma_e]

            geom_eff_bns    = geom_eff_func(theta_c_bns)
            if (
                not np.isfinite(geom_eff_bns)
                or geom_eff_bns < 0
            ):
                return _bad_likelihood()

            intrinsic_bns = (
                geom_eff_bns * fj_bns * bns_data.total_merger_rate * GBM_EFF
            )

            bns_results = simplified_montecarlo(
                bns_grb_thetas,
                n_events,
                bns_data,
                bns_distances,
                k_interpolator,
                rng = bns_likelihood_rng
            )

            bns_trig, bns_analysis = apply_detection_cuts(
                bns_results["p_flux"],
                bns_results["E_p_obs"],
            )

            pflux_detected = bns_results["p_flux"][bns_analysis]

            epeak_detected = bns_results["E_p_obs"][bns_analysis]
 
            if pflux_detected.size <= 3 or epeak_detected.size <= 3: return _bad_likelihood()

            logl_pflux = score_func_cvm(
                pflux_detected,
                observations["pflux"],
                bns_likelihood_rng
            )
            logl_epeak = score_func_cvm(
                epeak_detected,
                observations["epeak"],
                bns_likelihood_rng
            )

            triggered_years = observations["trigger_years"]
            observed_yearly_rate = observations["c_det"]

            phys_eff_bns = np.mean(bns_trig)
            predicted_bns = (
                intrinsic_bns * triggered_years * phys_eff_bns
            )

            predicted_total = predicted_bns
            observed_total  = observed_yearly_rate * triggered_years

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
                0, # no nsbh contribution
            )

        def log_probability(thetas: Sequence[float]):
            log_prior = flat_prior(thetas, n_params=n_params, bounds=PRIOR_BOUNDS)
            if not np.isfinite(log_prior):
                return (-np.inf, 0.0, 0.0, 0.0, 0.0, 0.0)

            likelihood = log_likelihood_single(thetas)
            if not np.isfinite(likelihood[0]):
                return (-np.inf, 0.0, 0.0, 0.0, 0.0, 0.0)
            return (log_prior + likelihood[0], *likelihood[1:])

        rng = np.random.default_rng(seed + alpha_index)
        starting_point = _make_initial_walkers(
            n_walkers=n_walkers,
            rng=rng,
            initial_center=initial_center,
            initial_scale=initial_scale,
        )

        initial_pos, n_steps_remaining, backend = check_and_resume_mcmc(
            filename=backend_path,
            n_steps=n_steps,
            starting_point=starting_point,
            n_walkers=n_walkers,
        )

        print(
            f"alpha={alpha}; "
            f"fj_bns={fj_bns:g}; "
            f"geometry={_geometry_name(geom_eff_func)}; "
            f"population={population}; "
            f"nsbh_series={nsbh_series}; "
            f"remaining steps={n_steps_remaining}"
        )
        if n_steps_remaining > 0:
            run_mcmc(
                log_probability_func=log_probability,
                initial_walkers=initial_pos,
                n_iterations=n_steps_remaining,
                n_walkers=n_walkers,
                n_params=N_PARAMS_SINGLE,
                backend=backend,
                blobs_dtype=BLOBS_DTYPE,
            )
        output_paths.append(backend_path)

    return output_paths

def run_pop_combined(
        
)

def _load_chain(
    alpha: str,
    fj_bns: float,
    geom_eff_func: Callable,
    nsbh_population: str,
    burn_frac: float,
    thin: int,
):
    if not 0 <= burn_frac < 1:
        raise ValueError("burn_frac must satisfy 0 <= burn_frac < 1")
    if thin < 1:
        raise ValueError("thin must be at least 1")

    backend_path = _backend_path(alpha, fj_bns, geom_eff_func, nsbh_population)
    if not backend_path.exists():
        return None, None, None, backend_path

    backend = emcee.backends.HDFBackend(backend_path, read_only=True)
    discard = int(backend.iteration * burn_frac)
    flat_chain = backend.get_chain(discard=discard, thin=thin, flat=True)
    flat_blobs = backend.get_blobs(discard=discard, thin=thin, flat=True)
    return backend, flat_chain, flat_blobs, backend_path

def plot_corner(
    alphas,
    fj_bns,
    geom_eff_func,
    population,
    nsbh_series,
    burn_frac=0.33,
    thin=10,
):
    backend_label = f"{population}_{nsbh_series}"

    figures = {}

    for alpha in alphas:
        _, flat, _, backend_path = _load_chain(
            alpha,
            fj_bns,
            geom_eff_func,
            backend_label,
            burn_frac,
            thin,
        )

        if flat is None or flat.size == 0:
            print(f"Warning: missing or empty chain: {backend_path}")
            continue

        common_args = {
            "bins": 30,
            "smooth": 1,
            "plot_datapoints": False,
            "plot_density": False,
            "fill_contours": False,
            "no_fill_contours": True,
            "levels": [0.68, 0.95],
            "smooth1d": 1,
        }

        contour_style = {
            "linestyles": ["--", "-"],
            "linewidths": [1.2, 1.5],
        }

        fig = corner.corner(
            flat,
            labels=LABELS,
            color="C0",
            **common_args,
            contour_kwargs=contour_style,
        )

        alpha_label = (
            str(alpha)[1:]
            if str(alpha).startswith("A")
            else str(alpha)
        )

        fig.suptitle(
            rf"QCBSE, $\alpha_{{\rm CE}}={alpha_label}$, "
            rf"$f_j={fj_bns:g}$" + "\n" + nsbh_series,
            fontsize=14,
        )

        figures[alpha] = fig

    return figures

def collect_rate_posteriors(
    alphas: Sequence[str],
    fj_list: Sequence[float],
    population: str,
    nsbh_series: Sequence[str],
    geom_eff_func: Callable,
    datafiles: Path = DATAFILES,
    burn_frac: float = 0.33,
    thin: int = 10,
) -> dict[tuple[float, str, str], dict[str, np.ndarray]]:
    """Collect BNS and NSBH rate posteriors from saved MCMC blobs."""

    alphas = tuple(alphas)
    nsbh_series = tuple(nsbh_series)

    if not alphas:
        raise ValueError("At least one alpha is required")
    if not nsbh_series:
        raise ValueError("At least one NSBH series is required")

    # The observing duration is independent of population synthesis.
    _, _, observations = initialize_combined_simulation(
        datafiles=Path(datafiles),
        population=population,
        alpha=alphas[0],
        nsbh_series=nsbh_series[0],
        sample_size=1,
        seed=0,
    )

    triggered_years = float(
        np.asarray(observations["trigger_years"]).squeeze()
    )

    results = {}

    for alpha in alphas:
        for fj_bns in fj_list:
            for series in nsbh_series:
                backend_label = f"{population}_{series}"

                _, _, blobs, backend_path = _load_chain(
                    alpha,
                    fj_bns,
                    geom_eff_func,
                    backend_label,
                    burn_frac,
                    thin,
                )

                if blobs is None or blobs.size == 0:
                    raise FileNotFoundError(
                        f"Missing or empty chain: {backend_path}"
                    )

                required_fields = {"mu_bns", "mu_nsbh"}

                if (
                    blobs.dtype.names is None
                    or not required_fields.issubset(blobs.dtype.names)
                ):
                    raise ValueError(
                        f"Backend does not contain rate blobs: {backend_path}"
                    )

                results[(float(fj_bns), alpha, series)] = {
                    "R_BNS": (
                        np.asarray(blobs["mu_bns"], dtype=float)
                        / triggered_years
                    ),
                    "R_NSBH": (
                        np.asarray(blobs["mu_nsbh"], dtype=float)
                        / triggered_years
                    ),
                }

    return results

def plot_posterior_grid(
    dictionary_of_results: Mapping[
        tuple[float, str, str], Mapping[str, np.ndarray]
    ],
    fj_list: Sequence[float],
    alpha_list: Sequence[str],
    nsbh_populations: Sequence[str],
    swap_axes: bool = False,
    bins: int = 40,
    levels: Sequence[float] = (0.68, 0.95),
    log_axes: bool = True,
    minimum_fraction: float = 5e-2,
    population_labels: Sequence[str] | None = None,
):
    """Plot posterior BNS/NSBH rate fractions in an alpha-by-fj grid."""
    if population_labels is None:
        population_labels = list(nsbh_populations)
    if len(population_labels) != len(nsbh_populations):
        raise ValueError("population_labels must match nsbh_populations in length")

    row_values = fj_list if swap_axes else alpha_list
    col_values = alpha_list if swap_axes else fj_list
    rows, cols = len(row_values), len(col_values)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(4 * cols, 4 * rows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    colors = ["k", "C1", "C2", "C3", "C4", "C5"]
    lower = minimum_fraction if log_axes else 0.0

    for row_index, row_value in enumerate(row_values):
        for col_index, col_value in enumerate(col_values):
            alpha = col_value if swap_axes else row_value
            fj_bns = row_value if swap_axes else col_value
            ax = axes[row_index, col_index]

            for pop_index, population in enumerate(nsbh_populations):
                entry = dictionary_of_results[(float(fj_bns), alpha, population)]
                rate_bns = np.asarray(entry["R_BNS"], dtype=float)
                rate_nsbh = np.asarray(entry["R_NSBH"], dtype=float)
                total = rate_bns + rate_nsbh
                valid = np.isfinite(total) & (total > 0)
                eta_bns = rate_bns[valid] / total[valid]
                eta_nsbh = rate_nsbh[valid] / total[valid]
                if log_axes:
                    eta_bns = np.clip(eta_bns, minimum_fraction, 1.0)
                    eta_nsbh = np.clip(eta_nsbh, minimum_fraction, 1.0)

                corner_args = {
                    "bins": bins,
                    "smooth": 1,
                    "plot_datapoints": False,
                    "plot_density": True,
                    "fill_contours": False,
                    "no_fill_contours": False,
                    "levels": list(levels),
                }
                contour_args = {
                    "linestyles": ["--", "-"],
                    "linewidths": [1.2, 1.5],
                }

                corner.hist2d(
                    eta_bns,
                    eta_nsbh,
                    ax=ax,
                    color=colors[pop_index % len(colors)],
                    axes_scale=["log", "log"] if log_axes else ["linear", "linear"],
                    **corner_args,                    
                    contour_kwargs=contour_args,
                )

            if log_axes:
                ax.set_xscale("log")
                ax.set_yscale("log")
            ax.set_xlim(lower, 1.0)
            ax.set_ylim(lower, 1.0)
            if row_index == rows - 1:
                ax.set_xlabel(r"$\eta_{\rm BNS}$")
            if col_index == 0:
                ax.set_ylabel(r"$\eta_{\rm NSBH}$")

            alpha_label = str(alpha)[1:] if str(alpha).startswith("A") else str(alpha)
            ax.text(
                0.05,
                0.05,
                rf"$\alpha_{{\rm CE}}={alpha_label}$ | $f_j={float(fj_bns):g}$",
                transform=ax.transAxes,
            )

    handles = [
        plt.Line2D(
            [0],
            [0],
            color=colors[index % len(colors)],
            lw=2,
            label=label,
        )
        for index, label in enumerate(population_labels)
    ]
    axes[0, min(2, cols - 1)].legend(handles=handles, loc="upper right")
    fig.tight_layout()
    return fig, axes