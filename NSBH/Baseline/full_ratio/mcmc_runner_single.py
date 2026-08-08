"""BNS-only MCMC followed by an explicit 4D BNS+NSBH likelihood grid.

The workflow in this module is deliberately asymmetric:

1. ``run_pop`` samples the five-parameter BNS-only posterior with ``emcee``.
2. ``extract_bns_best_fits`` selects one maximum-posterior BNS sample per
   population-synthesis value, optionally imposing ``A_index < 3``.
3. ``run_pop_combined`` fixes ``A_index``, ``L_mu_E``, and ``sigma_E`` at
   those values and evaluates the remaining four-dimensional BNS+NSBH
   likelihood directly on a regular grid.

The second stage is not an MCMC.  Its observable-shape term is calculated on
the two-dimensional ``(L_L0, log10_kappa_nsbh)`` grid, while the Poisson term
is broadcast analytically over both opening-angle axes.  This retains the fast
factorization used by ``mcmc_runner_a.py`` and gives a deterministic posterior
tensor at the requested resolution.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Callable, Mapping, Sequence

import emcee
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from scipy.special import gammaln
from scipy.stats import cramervonmises_2samp

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
    BLOBS_DTYPE,
    _backend_path as full_backend_path,
    _bad_likelihood,
    _geometry_name,
    flat_prior,
)


DATAFILES = Path("../../../datafiles")

ALL_PARAMETER_NAMES = (
    "A_index",
    "L_L0",
    "log10_kappa_nsbh",
    "L_mu_E",
    "sigma_E",
    "theta_c_bns",
    "theta_c_nsbh",
)

FIXED_PARAMETER_NAMES = (
    "A_index",
    "L_mu_E",
    "sigma_E",
)

BNS_ONLY_PARAMETER_NAMES = (
    "A_index",
    "L_L0",
    "L_mu_E",
    "sigma_E",
    "theta_c_bns",
)

BNS_NSBH_PARAMETER_NAMES = (
    "L_L0",
    "log10_kappa_nsbh",
    "theta_c_bns",
    "theta_c_nsbh",
)

N_PARAMS_SINGLE = len(BNS_ONLY_PARAMETER_NAMES)
N_PARAMS_COMBINED = len(BNS_NSBH_PARAMETER_NAMES)

BNS_ONLY_LABELS = (
    r"$A$",
    r"$\log_{10}(L_0)$",
    r"$\mu_{E,p}$",
    r"$\sigma_{E,p}$",
    r"$\theta_c^{\mathrm{BNS}}$ [deg]",
)

COMBINED_LABELS = (
    r"$\log_{10}(L_0)$",
    r"$\log_{10}(\kappa_{\rm NSBH})$",
    r"$\theta_c^{\mathrm{BNS}}$ [deg]",
    r"$\theta_c^{\mathrm{NSBH}}$ [deg]",
)

# Backwards-compatible name for callers that used LABELS for the BNS chain.
LABELS = BNS_ONLY_LABELS

# Open BNS-only bounds, matching the corresponding dimensions of the full run.
PRIOR_BOUNDS = np.asarray(
    [
        (1.5, 6.0),
        (-2.0, 7.0),
        (0.1, 7.0),
        (0.0, 2.5),
        (1.0, 25.0),
    ],
    dtype=float,
)

COMBINED_PRIOR_BOUNDS = np.asarray(
    [
        (-2.0, 7.0),
        (-2.0, 0.0),
        (1.0, 25.0),
        (1.0, 25.0),
    ],
    dtype=float,
)

DEFAULT_INITIAL_CENTER_SINGLE = np.asarray([2.0, 1.0, 1.0, 1.0, 5.0])
DEFAULT_INITIAL_SCALE_SINGLE = np.asarray([0.30, 0.25, 0.15, 0.15, 0.08])

FULL_SHARED_COLUMNS = (1, 2, 5, 6)


def _safe_name(value: object) -> str:
    """Return a filesystem-friendly representation of a run setting."""
    return "".join(
        character if character.isalnum() or character in "-." else "-"
        for character in str(value)
    )


def _fixed_shape_values(
    fixed_params_alpha: Mapping[str, float],
) -> dict[str, float]:
    """Validate the three BNS parameters fixed in the grid calculation."""
    missing = [
        name for name in FIXED_PARAMETER_NAMES if name not in fixed_params_alpha
    ]
    if missing:
        raise KeyError(
            "The BNS best-fit dictionary is missing: " + ", ".join(missing)
        )

    values = {
        name: float(fixed_params_alpha[name])
        for name in FIXED_PARAMETER_NAMES
    }
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("All fixed BNS parameters must be finite")

    relevant_bounds = {
        "A_index": PRIOR_BOUNDS[0],
        "L_mu_E": PRIOR_BOUNDS[2],
        "sigma_E": PRIOR_BOUNDS[3],
    }
    for name, (lower, upper) in relevant_bounds.items():
        if not lower < values[name] < upper:
            raise ValueError(
                f"{name}={values[name]} lies outside the open prior "
                f"({lower}, {upper})"
            )
    return values


def _fixed_shape_tag(fixed_params_alpha: Mapping[str, float]) -> str:
    """Fingerprint the values that define a combined-grid likelihood."""
    values = _fixed_shape_values(fixed_params_alpha)
    payload = "|".join(
        f"{name}={values[name]:.17g}" for name in FIXED_PARAMETER_NAMES
    )
    return sha256(payload.encode("utf-8")).hexdigest()[:12]


def _bns_run_name(
    alpha: str,
    fj_bns: float,
    geom_eff_func: Callable,
    population: str,
) -> str:
    return (
        f"bns_only_alpha_{_safe_name(alpha)}"
        f"_fj_{float(fj_bns):.6g}"
        f"_{_safe_name(_geometry_name(geom_eff_func))}"
        f"_{_safe_name(population)}"
    )


def bns_backend_path(
    alpha: str,
    fj_bns: float,
    geom_eff_func: Callable,
    population: str,
) -> Path:
    """Return the dedicated HDF path for a BNS-only chain."""
    return src.init.create_run_dir(
        _bns_run_name(alpha, fj_bns, geom_eff_func, population),
        output_files_default="Output_files",
    ) / "emcee.h5"


def _grid_run_name(
    alpha: str,
    fj_bns: float,
    geom_eff_func: Callable,
    population: str,
    nsbh_series: str,
    fixed_params_alpha: Mapping[str, float],
) -> str:
    return (
        f"analytical_joint_alpha_{_safe_name(alpha)}"
        f"_fj_{float(fj_bns):.6g}"
        f"_{_safe_name(_geometry_name(geom_eff_func))}"
        f"_{_safe_name(population)}"
        f"_{_safe_name(nsbh_series)}"
        f"_bnsfit_{_fixed_shape_tag(fixed_params_alpha)}"
    )


def combined_grid_path(
    alpha: str,
    fj_bns: float,
    geom_eff_func: Callable,
    population: str,
    nsbh_series: str,
    fixed_params_alpha: Mapping[str, float],
) -> Path:
    """Return the output path for one explicit four-dimensional grid."""
    return src.init.create_run_dir(
        _grid_run_name(
            alpha,
            fj_bns,
            geom_eff_func,
            population,
            nsbh_series,
            fixed_params_alpha,
        ),
        output_files_default="Output_files",
    ) / "grid_eval.npz"


def _make_initial_walkers_single(
    *,
    n_walkers: int,
    rng: np.random.Generator,
    initial_center: Sequence[float],
    initial_scale: Sequence[float],
) -> np.ndarray:
    """Draw a prior-valid five-dimensional walker ensemble."""
    if n_walkers < 2 * N_PARAMS_SINGLE:
        raise ValueError(
            f"emcee requires at least {2 * N_PARAMS_SINGLE} walkers for "
            f"{N_PARAMS_SINGLE} parameters"
        )

    center = np.asarray(initial_center, dtype=float)
    scale = np.asarray(initial_scale, dtype=float)
    if center.shape != (N_PARAMS_SINGLE,) or scale.shape != (N_PARAMS_SINGLE,):
        raise ValueError(
            f"initial_center and initial_scale must each have "
            f"{N_PARAMS_SINGLE} entries"
        )
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(scale)):
        raise ValueError("Initial centers and scales must be finite")
    if not np.all(scale > 0):
        raise ValueError("Every initial scale must be positive")
    if not np.isfinite(
        flat_prior(
            center,
            n_params=N_PARAMS_SINGLE,
            bounds=PRIOR_BOUNDS,
        )
    ):
        raise ValueError("initial_center must lie strictly inside the prior")

    walkers = center + rng.normal(
        size=(n_walkers, N_PARAMS_SINGLE)
    ) * scale
    invalid = np.asarray(
        [
            not np.isfinite(
                flat_prior(
                    walker,
                    n_params=N_PARAMS_SINGLE,
                    bounds=PRIOR_BOUNDS,
                )
            )
            for walker in walkers
        ]
    )
    attempts = 0
    while np.any(invalid):
        walkers[invalid] = center + rng.normal(
            size=(int(np.sum(invalid)), N_PARAMS_SINGLE)
        ) * scale
        invalid = np.asarray(
            [
                not np.isfinite(
                    flat_prior(
                        walker,
                        n_params=N_PARAMS_SINGLE,
                        bounds=PRIOR_BOUNDS,
                    )
                )
                for walker in walkers
            ]
        )
        attempts += 1
        if attempts >= 10_000:
            raise RuntimeError("Could not draw prior-valid initial walkers")
    return walkers


def _observation_scalars(
    observations: Mapping[str, object],
) -> tuple[float, float]:
    triggered_years = float(
        np.asarray(observations["trigger_years"], dtype=float).squeeze()
    )
    yearly_rate = float(
        np.asarray(observations["c_det"], dtype=float).squeeze()
    )
    if (
        not np.isfinite(triggered_years)
        or triggered_years <= 0
        or not np.isfinite(yearly_rate)
        or yearly_rate < 0
    ):
        raise ValueError("Invalid observing duration or observed yearly rate")
    return triggered_years, yearly_rate


def _scalar_geometry(
    geom_eff_func: Callable[[float], float],
    theta_c: float,
) -> float:
    value = np.asarray(geom_eff_func(theta_c), dtype=float)
    if value.size != 1:
        raise ValueError(
            "geom_eff_func must return one value for one opening angle"
        )
    return float(value.reshape(-1)[0])


def run_pop(
    alphas: Sequence[str],
    fj_bns: float,
    geom_eff_func: Callable[[float], float],
    datafiles: Path = DATAFILES,
    population: str = "delayed",
    nsbh_series: str = "NSBH_DD2_uniform_chi_0_1",
    n_params: int = N_PARAMS_SINGLE,
    n_walkers: int = 24,
    n_steps: int = 10_000,
    n_events: int = N_MC_EVENTS,
    sample_size: int | None = None,
    seed: int = 123,
    initial_center: Sequence[float] | None = None,
    initial_scale: Sequence[float] | None = None,
) -> list[Path]:
    """Run or resume one five-parameter BNS-only chain per alpha.

    ``nsbh_series`` is needed by ``initialize_combined_simulation`` but the
    returned NSBH population is discarded and never enters this likelihood.
    """
    if n_params != N_PARAMS_SINGLE:
        raise ValueError(
            f"This model samples exactly {N_PARAMS_SINGLE} parameters, "
            f"not {n_params}"
        )
    if not alphas:
        raise ValueError("At least one alpha value is required")
    if not np.isfinite(fj_bns) or fj_bns <= 0:
        raise ValueError("fj_bns must be positive and finite")
    if n_events <= 0 or n_steps < 0:
        raise ValueError(
            "n_events must be positive and n_steps cannot be negative"
        )
    if sample_size is not None and sample_size <= 0:
        raise ValueError("sample_size must be positive when supplied")

    center = (
        DEFAULT_INITIAL_CENTER_SINGLE
        if initial_center is None
        else np.asarray(initial_center, dtype=float)
    )
    scale = (
        DEFAULT_INITIAL_SCALE_SINGLE
        if initial_scale is None
        else np.asarray(initial_scale, dtype=float)
    )
    population_sample_size = (
        n_events if sample_size is None else max(sample_size, n_events)
    )
    output_paths: list[Path] = []

    for alpha_index, alpha in enumerate(alphas):
        bns_data, _, observations = initialize_combined_simulation(
            datafiles=Path(datafiles),
            population=population,
            alpha=alpha,
            nsbh_series=nsbh_series,
            sample_size=population_sample_size,
            seed=seed + alpha_index,
        )
        k_interpolator = create_k_interpolator()
        bns_distances = compute_luminosity_distance(bns_data.z_arr)
        backend_path = bns_backend_path(
            alpha,
            fj_bns,
            geom_eff_func,
            population,
        )

        def log_likelihood_single(thetas: Sequence[float]):
            likelihood_rng = np.random.default_rng(seed)
            a_index, l_l0, l_mu_e, sigma_e, theta_c_bns = thetas

            try:
                geom_eff_bns = _scalar_geometry(
                    geom_eff_func,
                    theta_c_bns,
                )
            except (TypeError, ValueError, FloatingPointError):
                return _bad_likelihood()
            if not np.isfinite(geom_eff_bns) or geom_eff_bns < 0:
                return _bad_likelihood()

            bns_results = simplified_montecarlo(
                [a_index, l_l0, l_mu_e, sigma_e],
                n_events,
                bns_data,
                bns_distances,
                k_interpolator,
                rng=likelihood_rng,
            )
            bns_triggered, bns_analysis = apply_detection_cuts(
                bns_results["p_flux"],
                bns_results["E_p_obs"],
            )
            pflux_detected = bns_results["p_flux"][bns_analysis]
            epeak_detected = bns_results["E_p_obs"][bns_analysis]
            if pflux_detected.size <= 3 or epeak_detected.size <= 3:
                return _bad_likelihood()

            logl_pflux = score_func_cvm(
                pflux_detected,
                observations["pflux"],
                likelihood_rng,
            )
            logl_epeak = score_func_cvm(
                epeak_detected,
                observations["epeak"],
                likelihood_rng,
            )
            triggered_years, observed_yearly_rate = _observation_scalars(
                observations
            )
            intrinsic_bns = (
                geom_eff_bns
                * fj_bns
                * bns_data.total_merger_rate
                * GBM_EFF
            )
            predicted_bns = (
                intrinsic_bns
                * triggered_years
                * float(np.mean(bns_triggered))
            )
            observed_total = observed_yearly_rate * triggered_years
            if not np.isfinite(predicted_bns) or predicted_bns <= 0:
                return _bad_likelihood()

            logl_poisson = poiss_log(k=observed_total, mu=predicted_bns)
            logl_total = logl_pflux + logl_epeak + logl_poisson
            if not np.isfinite(logl_total):
                return _bad_likelihood()

            return (
                float(logl_total),
                float(logl_pflux),
                float(logl_epeak),
                float(logl_poisson),
                float(predicted_bns),
                0.0,
            )

        def log_probability(thetas: Sequence[float]):
            log_prior = flat_prior(
                thetas,
                n_params=n_params,
                bounds=PRIOR_BOUNDS,
            )
            if not np.isfinite(log_prior):
                return (-np.inf, 0.0, 0.0, 0.0, 0.0, 0.0)

            likelihood = log_likelihood_single(thetas)
            if not np.isfinite(likelihood[0]):
                return (-np.inf, 0.0, 0.0, 0.0, 0.0, 0.0)
            return (float(log_prior + likelihood[0]), *likelihood[1:])

        starting_point = _make_initial_walkers_single(
            n_walkers=n_walkers,
            rng=np.random.default_rng(seed + alpha_index),
            initial_center=center,
            initial_scale=scale,
        )
        initial_pos, n_steps_remaining, backend = check_and_resume_mcmc(
            filename=backend_path,
            n_steps=n_steps,
            starting_point=starting_point,
            n_walkers=n_walkers,
        )

        print(
            f"stage=BNS-only; alpha={alpha}; fj_bns={fj_bns:g}; "
            f"geometry={_geometry_name(geom_eff_func)}; "
            f"population={population}; remaining steps={n_steps_remaining}"
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


# Descriptive alias; run_pop is retained to match the supplied draft.
run_bns_only = run_pop


def _resolve_discard(
    backend: emcee.backends.HDFBackend,
    *,
    discard: int | None,
    burn_frac: float,
) -> int:
    if backend.iteration <= 0:
        raise ValueError("The backend contains no MCMC iterations")
    if discard is None:
        if not 0 <= burn_frac < 1:
            raise ValueError("burn_frac must satisfy 0 <= burn_frac < 1")
        resolved = int(backend.iteration * burn_frac)
    else:
        resolved = int(discard)
        if resolved < 0:
            raise ValueError("discard cannot be negative")
    if resolved >= backend.iteration:
        raise ValueError(
            f"discard={resolved} removes all {backend.iteration} iterations"
        )
    return resolved


def extract_bns_best_fits(
    backend_paths: Mapping[str, str | Path],
    *,
    discard: int | None = None,
    burn_frac: float = 0.33,
    thin: int = 10,
    max_a_index: float | None = 3.0,
    verbose: bool = True,
) -> dict[str, dict[str, float]]:
    """Select the maximum-posterior BNS sample for each alpha.

    The returned dictionaries contain all five BNS parameters and the selected
    log probability.  ``run_pop_combined`` reads only ``A_index``, ``L_mu_E``,
    and ``sigma_E``.
    """
    if thin < 1:
        raise ValueError("thin must be at least 1")
    if max_a_index is not None and not np.isfinite(max_a_index):
        raise ValueError("max_a_index must be finite or None")

    best_fits: dict[str, dict[str, float]] = {}
    for alpha, backend_value in backend_paths.items():
        path = Path(backend_value)
        if not path.is_file():
            raise FileNotFoundError(f"Missing BNS-only backend: {path}")

        backend = emcee.backends.HDFBackend(path, read_only=True)
        resolved_discard = _resolve_discard(
            backend,
            discard=discard,
            burn_frac=burn_frac,
        )
        samples = backend.get_chain(
            flat=True,
            discard=resolved_discard,
            thin=thin,
        )
        log_prob = backend.get_log_prob(
            flat=True,
            discard=resolved_discard,
            thin=thin,
        )
        if samples.ndim != 2 or samples.shape[1] != N_PARAMS_SINGLE:
            raise ValueError(
                f"Expected a {N_PARAMS_SINGLE}-parameter chain at {path}; "
                f"found {samples.shape}"
            )
        if samples.shape[0] == 0 or samples.shape[0] != log_prob.shape[0]:
            raise ValueError(f"Empty or inconsistent BNS chain: {path}")

        valid = np.isfinite(log_prob) & np.all(np.isfinite(samples), axis=1)
        if max_a_index is not None:
            valid &= samples[:, 0] < max_a_index
        valid_indices = np.flatnonzero(valid)
        if valid_indices.size == 0:
            constraint = (
                ""
                if max_a_index is None
                else f" satisfying A_index < {max_a_index:g}"
            )
            raise ValueError(
                f"No finite post-burn-in samples{constraint} in {path}"
            )

        best_index = valid_indices[
            int(np.argmax(log_prob[valid_indices]))
        ]
        selected = {
            name: float(samples[best_index, index])
            for index, name in enumerate(BNS_ONLY_PARAMETER_NAMES)
        }
        selected["log_probability"] = float(log_prob[best_index])
        best_fits[alpha] = selected

        if verbose:
            constraint = (
                ""
                if max_a_index is None
                else f", A_index < {max_a_index:g}"
            )
            print(
                f"alpha={alpha}{constraint}: "
                f"log_prob={selected['log_probability']:.3f}, "
                f"A={selected['A_index']:.3f}, "
                f"L0={selected['L_L0']:.3f}, "
                f"mu_E={selected['L_mu_E']:.3f}, "
                f"sigma_E={selected['sigma_E']:.3f}, "
                f"theta_BNS={selected['theta_c_bns']:.3f}"
            )
    return best_fits


def lognormal_numpy(
    mu: float,
    sigma: float,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw a base-10 parameterized lognormal population."""
    log_ten = np.log(10.0)
    return rng.lognormal(
        mean=mu * log_ten,
        sigma=sigma * log_ten,
        size=n,
    )


def luminosity_gen(
    a_index: float,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw the dimensionless luminosity scatter used by the MC model."""
    shape = (a_index - 1.0) / a_index
    return rng.gamma(shape, size=n) ** (-1.0 / a_index)


def build_redshift_ppf(
    z_interp_func: Callable[[np.ndarray], np.ndarray],
    *,
    z_min: float = 1e-3,
    z_max: float = 15.0,
    n_points: int = 2_000,
) -> Callable[[np.ndarray], np.ndarray]:
    """Construct a numerical inverse CDF from a differential redshift model."""
    if not 0 < z_min < z_max:
        raise ValueError("Require 0 < z_min < z_max")
    if n_points < 3:
        raise ValueError("n_points must be at least 3")

    z_fine = np.linspace(z_min, z_max, n_points)
    density = np.asarray(z_interp_func(z_fine), dtype=float)
    if density.shape != z_fine.shape:
        density = np.broadcast_to(density, z_fine.shape).astype(float)
    density = np.where(np.isfinite(density), density, 0.0)
    density = np.clip(density, 0.0, None)

    cdf = cumulative_trapezoid(density, z_fine, initial=0.0)
    if not np.isfinite(cdf[-1]) or cdf[-1] <= 0:
        raise ValueError("The redshift model has no positive finite mass")
    cdf /= cdf[-1]

    unique_cdf, unique_indices = np.unique(cdf, return_index=True)
    unique_z = z_fine[unique_indices]
    if unique_cdf[0] > 0:
        unique_cdf = np.insert(unique_cdf, 0, 0.0)
        unique_z = np.insert(unique_z, 0, z_min)
    if unique_cdf[-1] < 1:
        unique_cdf = np.append(unique_cdf, 1.0)
        unique_z = np.append(unique_z, z_max)

    return interp1d(
        unique_cdf,
        unique_z,
        kind="linear",
        bounds_error=False,
        fill_value=(z_min, z_max),
        assume_sorted=True,
    )


def generate_base_population(
    data_obj: object,
    redshift_ppf: Callable[[np.ndarray], np.ndarray],
    k_interpolator: Callable[[np.ndarray], np.ndarray],
    l_mu_e: float,
    sigma_e: float,
    a_index: float,
    n_events: int,
    *,
    seed: int,
) -> dict[str, np.ndarray]:
    """Generate fixed random draws whose luminosity scale can be varied cheaply."""
    rng = np.random.default_rng(seed)
    z_samples = np.asarray(
        redshift_ppf(rng.uniform(size=n_events)),
        dtype=float,
    )
    distances = compute_luminosity_distance(z_samples)
    epeak_rest = lognormal_numpy(l_mu_e, sigma_e, n_events, rng)
    epeak_observed = epeak_rest / (1.0 + z_samples)
    luminosity_scatter = luminosity_gen(a_index, n_events, rng)
    points = np.column_stack((np.log10(epeak_observed), z_samples))
    k_correction = np.asarray(k_interpolator(points), dtype=float)

    valid = (
        np.isfinite(distances)
        & (distances > 0)
        & np.isfinite(k_correction)
        & (k_correction > 0)
        & np.isfinite(luminosity_scatter)
        & (luminosity_scatter > 0)
    )
    base_flux = np.zeros(n_events, dtype=float)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        base_flux[valid] = (
            luminosity_scatter[valid]
            / (
                4.0
                * np.pi
                * distances[valid] ** 2
                * k_correction[valid]
            )
            * 6.242e8
        )
    base_flux[~np.isfinite(base_flux)] = 0.0

    return {
        "base_flux": base_flux,
        "E_p_obs": np.asarray(epeak_observed, dtype=float),
        "z": z_samples,
    }


def prepare_base_populations(
    bns_data: object,
    nsbh_data: object,
    k_interpolator: Callable,
    l_mu_e: float,
    sigma_e: float,
    a_index: float,
    n_events: int,
    *,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Create the deterministic BNS and NSBH base populations for a grid."""
    bns_ppf = build_redshift_ppf(bns_data.P_z_interp)
    nsbh_ppf = build_redshift_ppf(nsbh_data.P_z_interp)
    base_bns = generate_base_population(
        bns_data,
        bns_ppf,
        k_interpolator,
        l_mu_e,
        sigma_e,
        a_index,
        n_events,
        seed=seed,
    )
    base_nsbh = generate_base_population(
        nsbh_data,
        nsbh_ppf,
        k_interpolator,
        l_mu_e,
        sigma_e,
        a_index,
        n_events,
        seed=seed + 1,
    )
    return base_bns, base_nsbh


def _grid_centers(
    bounds: Sequence[float],
    n_points: int,
) -> tuple[np.ndarray, float]:
    """Return equal-width cell centers and their common width."""
    lower, upper = map(float, bounds)
    if not lower < upper:
        raise ValueError("Every grid lower bound must be below its upper bound")
    if n_points < 2:
        raise ValueError("Every grid axis requires at least two cells")
    edges = np.linspace(lower, upper, n_points + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, float(edges[1] - edges[0])


def _detection_summary(
    base_population: Mapping[str, np.ndarray],
    log_luminosity: float,
) -> dict[str, object]:
    """Return detected samples and trigger efficiency at one luminosity scale."""
    base_flux = np.asarray(base_population["base_flux"], dtype=float)
    epeak = np.asarray(base_population["E_p_obs"], dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        scaled_flux = base_flux * 10.0 ** (float(log_luminosity) + 49.0)
    triggered, analysis = apply_detection_cuts(scaled_flux, epeak)
    return {
        "pflux": np.asarray(scaled_flux[analysis], dtype=float),
        "epeak": np.asarray(epeak[analysis], dtype=float),
        "efficiency": float(
            np.mean(np.asarray(triggered, dtype=float))
        ),
    }


def score_func_cvm_exact(
    simulated: Sequence[float],
    observed: Sequence[float],
) -> float:
    """Return the exact two-sample CvM log p-value without resampling."""
    simulated_array = np.asarray(simulated, dtype=float)
    observed_array = np.asarray(observed, dtype=float)
    simulated_array = simulated_array[
        np.isfinite(simulated_array) & (simulated_array > 0)
    ]
    observed_array = observed_array[
        np.isfinite(observed_array) & (observed_array > 0)
    ]
    if simulated_array.size <= 3 or observed_array.size <= 3:
        return -np.inf

    p_value = float(
        cramervonmises_2samp(
            np.log10(simulated_array),
            np.log10(observed_array),
        ).pvalue
    )
    if not np.isfinite(p_value) or p_value <= 0:
        return -np.inf
    return float(np.log(p_value))


def _evaluate_geometry_grid(
    geom_eff_func: Callable[[float], float],
    theta_grid: np.ndarray,
) -> np.ndarray:
    """Evaluate scalar- or array-aware geometry functions over one axis."""
    try:
        values = np.asarray(geom_eff_func(theta_grid), dtype=float)
        if values.shape != theta_grid.shape:
            values = np.asarray(
                [geom_eff_func(theta) for theta in theta_grid],
                dtype=float,
            )
    except (TypeError, ValueError):
        values = np.asarray(
            [geom_eff_func(theta) for theta in theta_grid],
            dtype=float,
        )
    if (
        values.shape != theta_grid.shape
        or np.any(~np.isfinite(values))
        or np.any(values < 0)
    ):
        raise ValueError(
            "geom_eff_func returned invalid values on the angle grid"
        )
    return values


def _poisson_logpmf(k: float, mu: np.ndarray) -> np.ndarray:
    """Vectorized Poisson log likelihood, also well-defined for float k."""
    result = np.full(np.shape(mu), -np.inf, dtype=float)
    valid = np.isfinite(mu) & (mu > 0)
    result[valid] = (
        k * np.log(mu[valid]) - mu[valid] - gammaln(k + 1.0)
    )
    return result


def _resolve_fj_bns(
    *,
    FJ_BNS: float | None,
    fj_bns: float | None,
) -> float:
    if FJ_BNS is None and fj_bns is None:
        raise ValueError("Supply FJ_BNS or fj_bns")
    if FJ_BNS is not None and fj_bns is not None:
        if not np.isclose(FJ_BNS, fj_bns):
            raise ValueError("FJ_BNS and fj_bns disagree")
    value = float(FJ_BNS if FJ_BNS is not None else fj_bns)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("The BNS jet fraction must be positive and finite")
    return value


def _reported_rate(data: object) -> float:
    """Return a population's local rate when available."""
    value = getattr(data, "local_rate", None)
    if value is None:
        value = getattr(data, "total_merger_rate")
    return float(np.asarray(value, dtype=float).squeeze())


def run_pop_combined(
    alphas: Sequence[str],
    geom_eff_func: Callable[[float], float],
    datafiles: Path = DATAFILES,
    fixed_params: Mapping[str, Mapping[str, float]] | None = None,
    nsbh_series: str = "NSBH_DD2_uniform_chi_0_1",
    population: str = "delayed",
    FJ_BNS: float | None = None,
    *,
    fj_bns: float | None = None,
    n_l0: int = 50,
    n_k: int = 40,
    n_tbns: int = 80,
    n_tnsbh: int = 80,
    l0_bounds: Sequence[float] = (-2.0, 7.0),
    k_bounds: Sequence[float] = (-2.0, 0.0),
    theta_bns_bounds: Sequence[float] = (1.0, 25.0),
    theta_nsbh_bounds: Sequence[float] = (1.0, 25.0),
    n_events: int = N_MC_EVENTS * 2,
    sample_size: int | None = None,
    seed: int = 123,
) -> tuple[list[Path], dict[str, tuple[float, float]]]:
    """Evaluate the fixed-shape BNS+NSBH likelihood on a 4D grid.

    The four axes are ``L_L0``, ``log10_kappa_nsbh``, ``theta_c_bns``, and
    ``theta_c_nsbh``.  Grid values are cell centers, so probability-mass
    sampling with half-cell jitter remains within the requested bounds.
    """
    resolved_fj_bns = _resolve_fj_bns(FJ_BNS=FJ_BNS, fj_bns=fj_bns)
    if not alphas:
        raise ValueError("At least one alpha value is required")
    if fixed_params is None:
        raise ValueError("fixed_params is required")
    missing_alphas = [alpha for alpha in alphas if alpha not in fixed_params]
    if missing_alphas:
        raise KeyError(
            "No BNS best fit supplied for: " + ", ".join(missing_alphas)
        )
    if n_events <= 0:
        raise ValueError("n_events must be positive")
    if sample_size is not None and sample_size <= 0:
        raise ValueError("sample_size must be positive when supplied")

    l0_grid, dl0 = _grid_centers(l0_bounds, n_l0)
    k_grid, dk = _grid_centers(k_bounds, n_k)
    theta_bns_grid, dtheta_bns = _grid_centers(
        theta_bns_bounds,
        n_tbns,
    )
    theta_nsbh_grid, dtheta_nsbh = _grid_centers(
        theta_nsbh_bounds,
        n_tnsbh,
    )
    population_sample_size = (
        n_events if sample_size is None else max(sample_size, n_events)
    )

    output_paths: list[Path] = []
    local_rate_dict: dict[str, tuple[float, float]] = {}

    for alpha_index, alpha in enumerate(alphas):
        fixed = _fixed_shape_values(fixed_params[alpha])
        a_index = fixed["A_index"]
        l_mu_e = fixed["L_mu_E"]
        sigma_e = fixed["sigma_E"]

        bns_data, nsbh_data, observations = initialize_combined_simulation(
            datafiles=Path(datafiles),
            population=population,
            alpha=alpha,
            nsbh_series=nsbh_series,
            sample_size=population_sample_size,
            seed=seed + alpha_index,
        )
        local_rate_dict[alpha] = (
            _reported_rate(bns_data),
            _reported_rate(nsbh_data),
        )

        k_interpolator = create_k_interpolator()
        base_bns, base_nsbh = prepare_base_populations(
            bns_data,
            nsbh_data,
            k_interpolator,
            l_mu_e,
            sigma_e,
            a_index,
            n_events,
            seed=seed + 2 * alpha_index,
        )

        observed_pflux = np.asarray(observations["pflux"], dtype=float)
        observed_epeak = np.asarray(observations["epeak"], dtype=float)
        logl_cvm = np.full((n_l0, n_k), -np.inf, dtype=float)
        phys_eff_bns = np.empty(n_l0, dtype=float)
        phys_eff_nsbh = np.empty((n_l0, n_k), dtype=float)

        print(
            f"stage=analytical-grid; alpha={alpha}; "
            f"observable cells={n_l0 * n_k}; "
            f"angle cells={n_tbns * n_tnsbh}"
        )
        for i in range(n_l0):
            bns_summary = _detection_summary(base_bns, l0_grid[i])
            phys_eff_bns[i] = float(bns_summary["efficiency"])
            for j in range(n_k):
                nsbh_summary = _detection_summary(
                    base_nsbh,
                    l0_grid[i] + k_grid[j],
                )
                phys_eff_nsbh[i, j] = float(
                    nsbh_summary["efficiency"]
                )
                pflux_detected = np.concatenate(
                    (
                        np.asarray(bns_summary["pflux"], dtype=float),
                        np.asarray(nsbh_summary["pflux"], dtype=float),
                    )
                )
                epeak_detected = np.concatenate(
                    (
                        np.asarray(bns_summary["epeak"], dtype=float),
                        np.asarray(nsbh_summary["epeak"], dtype=float),
                    )
                )
                if pflux_detected.size <= 3 or epeak_detected.size <= 3:
                    continue
                logl_cvm[i, j] = (
                    score_func_cvm_exact(pflux_detected, observed_pflux)
                    + score_func_cvm_exact(epeak_detected, observed_epeak)
                )

        geometry_bns = _evaluate_geometry_grid(
            geom_eff_func,
            theta_bns_grid,
        )
        geometry_nsbh = _evaluate_geometry_grid(
            geom_eff_func,
            theta_nsbh_grid,
        )
        triggered_years, observed_yearly_rate = _observation_scalars(
            observations
        )
        observed_total = observed_yearly_rate * triggered_years

        bns_count_scale = (
            resolved_fj_bns
            * bns_data.total_merger_rate
            * GBM_EFF
            * triggered_years
        )
        nsbh_count_scale = (
            FJ_NSBH_FIXED
            * nsbh_data.total_merger_rate
            * GBM_EFF
            * triggered_years
        )
        mu_bns = (
            bns_count_scale
            * phys_eff_bns[:, None, None, None]
            * geometry_bns[None, None, :, None]
        )
        mu_nsbh = (
            nsbh_count_scale
            * phys_eff_nsbh[:, :, None, None]
            * geometry_nsbh[None, None, None, :]
        )
        mu_total = mu_bns + mu_nsbh
        logl_poisson = _poisson_logpmf(observed_total, mu_total)
        logl_total = logl_cvm[:, :, None, None] + logl_poisson

        output_path = combined_grid_path(
            alpha,
            resolved_fj_bns,
            geom_eff_func,
            population,
            nsbh_series,
            fixed_params[alpha],
        )
        np.savez_compressed(
            output_path,
            l0=l0_grid,
            k=k_grid,
            theta_bns=theta_bns_grid,
            theta_nsbh=theta_nsbh_grid,
            cell_widths=np.asarray(
                [dl0, dk, dtheta_bns, dtheta_nsbh],
                dtype=float,
            ),
            l0_bounds=np.asarray(l0_bounds, dtype=float),
            k_bounds=np.asarray(k_bounds, dtype=float),
            theta_bns_bounds=np.asarray(
                theta_bns_bounds,
                dtype=float,
            ),
            theta_nsbh_bounds=np.asarray(
                theta_nsbh_bounds,
                dtype=float,
            ),
            logL_cvm=logl_cvm,
            logL_poiss=logl_poisson,
            logL_total=logl_total,
            mu_bns=mu_bns,
            mu_nsbh=mu_nsbh,
            fixed_A_index=np.asarray(a_index),
            fixed_L_mu_E=np.asarray(l_mu_e),
            fixed_sigma_E=np.asarray(sigma_e),
            fj_bns=np.asarray(resolved_fj_bns),
            triggered_years=np.asarray(triggered_years),
            observed_total=np.asarray(observed_total),
            n_events=np.asarray(n_events),
            seed=np.asarray(seed),
        )
        finite_cells = int(np.sum(np.isfinite(logl_total)))
        print(
            f"Grid completed for alpha={alpha}; "
            f"finite 4D cells={finite_cells:,}; saved to {output_path}"
        )
        output_paths.append(output_path)

    return output_paths, local_rate_dict


# Clear descriptive alias for new callers.
run_analytical_joint = run_pop_combined


def draw_grid_samples(
    grid_path: str | Path,
    n_samples: int = 40_000,
    *,
    seed: int = 123,
    jitter: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw probability-mass samples from a saved four-dimensional grid."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    path = Path(grid_path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing grid: {path}")

    with np.load(path) as data:
        l0_grid = np.asarray(data["l0"], dtype=float)
        k_grid = np.asarray(data["k"], dtype=float)
        theta_bns_grid = np.asarray(data["theta_bns"], dtype=float)
        theta_nsbh_grid = np.asarray(data["theta_nsbh"], dtype=float)
        log_likelihood = np.asarray(data["logL_total"], dtype=float)
        mu_bns = np.asarray(data["mu_bns"], dtype=float)
        mu_nsbh = np.asarray(data["mu_nsbh"], dtype=float)
        if "cell_widths" in data.files:
            cell_widths = np.asarray(data["cell_widths"], dtype=float)
        else:
            axes = (l0_grid, k_grid, theta_bns_grid, theta_nsbh_grid)
            cell_widths = np.asarray(
                [np.median(np.diff(axis)) for axis in axes],
                dtype=float,
            )

    expected_shape = (
        l0_grid.size,
        k_grid.size,
        theta_bns_grid.size,
        theta_nsbh_grid.size,
    )
    if log_likelihood.shape != expected_shape:
        raise ValueError(
            f"Grid likelihood shape {log_likelihood.shape} does not match "
            f"axes {expected_shape}"
        )

    flat_log_likelihood = log_likelihood.ravel()
    valid = np.isfinite(flat_log_likelihood)
    if not np.any(valid):
        raise ValueError(f"The grid has no finite likelihood cells: {path}")

    valid_log_likelihood = flat_log_likelihood[valid]
    weights = np.exp(valid_log_likelihood - np.max(valid_log_likelihood))
    normalization = np.sum(weights)
    if not np.isfinite(normalization) or normalization <= 0:
        raise ValueError(f"Could not normalize grid likelihood: {path}")
    weights /= normalization

    rng = np.random.default_rng(seed)
    selected_valid = rng.choice(
        valid_log_likelihood.size,
        size=n_samples,
        replace=True,
        p=weights,
    )
    original_flat_indices = np.flatnonzero(valid)[selected_valid]
    coordinates = np.unravel_index(
        original_flat_indices,
        log_likelihood.shape,
    )
    samples = np.column_stack(
        (
            l0_grid[coordinates[0]],
            k_grid[coordinates[1]],
            theta_bns_grid[coordinates[2]],
            theta_nsbh_grid[coordinates[3]],
        )
    )
    if jitter:
        samples += rng.uniform(
            low=-0.5 * cell_widths,
            high=0.5 * cell_widths,
            size=samples.shape,
        )

    mu_bns_full = np.broadcast_to(mu_bns, log_likelihood.shape)
    mu_nsbh_full = np.broadcast_to(mu_nsbh, log_likelihood.shape)
    return (
        samples,
        np.asarray(mu_bns_full[coordinates], dtype=float),
        np.asarray(mu_nsbh_full[coordinates], dtype=float),
    )


def _load_bns_chain(
    backend_path: str | Path,
    *,
    discard: int | None,
    burn_frac: float,
    thin: int,
) -> np.ndarray:
    if thin < 1:
        raise ValueError("thin must be at least 1")
    path = Path(backend_path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing backend: {path}")
    backend = emcee.backends.HDFBackend(path, read_only=True)
    resolved_discard = _resolve_discard(
        backend,
        discard=discard,
        burn_frac=burn_frac,
    )
    samples = backend.get_chain(
        discard=resolved_discard,
        thin=thin,
        flat=True,
    )
    if samples.size == 0:
        raise ValueError(f"No post-burn-in samples in {path}")
    return np.asarray(samples, dtype=float)


def plot_bns_corners(
    backend_paths: Mapping[str, str | Path],
    *,
    discard: int | None = None,
    burn_frac: float = 0.33,
    thin: int = 10,
) -> dict[str, plt.Figure]:
    """Plot the five-parameter BNS-only chains."""
    import corner

    figures: dict[str, plt.Figure] = {}
    for alpha, backend_path in backend_paths.items():
        samples = _load_bns_chain(
            backend_path,
            discard=discard,
            burn_frac=burn_frac,
            thin=thin,
        )
        if samples.shape[1] != N_PARAMS_SINGLE:
            raise ValueError(
                f"Expected {N_PARAMS_SINGLE} BNS columns in {backend_path}"
            )
        fig = corner.corner(
            samples,
            labels=BNS_ONLY_LABELS,
            range=[tuple(bounds) for bounds in PRIOR_BOUNDS],
            quantiles=[0.05, 0.5, 0.95],
            show_titles=True,
            bins=30,
            smooth=1,
            smooth1d=1,
            plot_datapoints=False,
            plot_density=True,
            levels=[0.68, 0.95],
            color="C0",
        )
        fig.suptitle(rf"BNS only, $\alpha_{{\rm CE}}={alpha}$", fontsize=14)
        figures[alpha] = fig
    return figures


def plot_corner(
    alphas: Sequence[str],
    fj_bns: float,
    geom_eff_func: Callable,
    population: str,
    nsbh_series: str | None = None,
    burn_frac: float = 0.33,
    thin: int = 10,
):
    """Compatibility wrapper around :func:`plot_bns_corners`."""
    del nsbh_series
    paths = {
        alpha: bns_backend_path(alpha, fj_bns, geom_eff_func, population)
        for alpha in alphas
    }
    return plot_bns_corners(
        paths,
        burn_frac=burn_frac,
        thin=thin,
    )


def plot_grid_corner(
    grid_path: str | Path,
    *,
    n_samples: int = 40_000,
    seed: int = 123,
    color: str = "C1",
    title: str | None = None,
) -> plt.Figure:
    """Plot a corner representation of one explicit grid posterior."""
    import corner

    samples, _, _ = draw_grid_samples(
        grid_path,
        n_samples=n_samples,
        seed=seed,
    )
    fig = corner.corner(
        samples,
        labels=COMBINED_LABELS,
        range=[tuple(bounds) for bounds in COMBINED_PRIOR_BOUNDS],
        quantiles=[0.05, 0.5, 0.95],
        show_titles=True,
        bins=30,
        smooth=1,
        smooth1d=1,
        plot_datapoints=False,
        plot_density=True,
        levels=[0.68, 0.95],
        color=color,
    )
    if title is not None:
        fig.suptitle(title, fontsize=14)
    return fig


def plot_corner_grid_multiples(
    alphas: Sequence[str],
    grid_paths_by_series: Mapping[str, Mapping[str, str | Path]],
    *,
    population_labels: Mapping[str, str] | None = None,
    local_rate_dicts: Mapping[
        str,
        Mapping[str, tuple[float, float]],
    ]
    | None = None,
    n_samples: int = 40_000,
    seed: int = 123,
) -> dict[str, plt.Figure]:
    """Overlay explicit-grid posteriors for several NSBH populations."""
    import corner

    if not grid_paths_by_series:
        raise ValueError("At least one NSBH series is required")
    colors = ["k", "C1", "C2", "C3", "C4", "C5"]
    figures: dict[str, plt.Figure] = {}

    for alpha_index, alpha in enumerate(alphas):
        fig = None
        handles = []
        bns_rate = None
        for series_index, (series, alpha_paths) in enumerate(
            grid_paths_by_series.items()
        ):
            if alpha not in alpha_paths:
                raise KeyError(f"No grid path for alpha={alpha}, series={series}")
            color = colors[series_index % len(colors)]
            samples, _, _ = draw_grid_samples(
                alpha_paths[alpha],
                n_samples=n_samples,
                seed=seed + alpha_index + 100 * series_index,
            )
            fig = corner.corner(
                samples,
                labels=COMBINED_LABELS,
                range=[tuple(bounds) for bounds in COMBINED_PRIOR_BOUNDS],
                bins=25,
                smooth=1,
                smooth1d=1,
                plot_datapoints=False,
                plot_density=True,
                levels=[0.68, 0.95],
                color=color,
                fig=fig,
            )
            label = (
                series
                if population_labels is None
                else population_labels.get(series, series)
            )
            if local_rate_dicts is not None and series in local_rate_dicts:
                rates = local_rate_dicts[series][alpha]
                bns_rate = rates[0]
                label += rf": $R_{{\rm NSBH}}={rates[1]:.2f}$"
            handles.append(
                plt.Line2D([0], [0], color=color, lw=2, label=label)
            )

        if fig is None:
            continue
        legend_title = (
            None
            if bns_rate is None
            else rf"$R_{{\rm BNS}}={bns_rate:.2f}$"
        )
        fig.legend(
            handles=handles,
            title=legend_title,
            loc="upper right",
            frameon=False,
        )
        fig.suptitle(
            rf"Analytical grid, $\alpha_{{\rm CE}}={alpha}$",
            fontsize=14,
        )
        figures[alpha] = fig
    return figures


def collect_rate_posteriors(
    grid_paths_by_series: Mapping[str, Mapping[str, str | Path]],
    *,
    fj_bns: float,
    n_samples: int = 40_000,
    seed: int = 123,
) -> dict[tuple[float, str, str], dict[str, np.ndarray]]:
    """Draw BNS and NSBH yearly-rate posteriors from saved grid tensors."""
    results: dict[
        tuple[float, str, str],
        dict[str, np.ndarray],
    ] = {}
    for series_index, (series, alpha_paths) in enumerate(
        grid_paths_by_series.items()
    ):
        for alpha_index, (alpha, grid_path) in enumerate(alpha_paths.items()):
            _, mu_bns, mu_nsbh = draw_grid_samples(
                grid_path,
                n_samples=n_samples,
                seed=seed + alpha_index + 100 * series_index,
            )
            with np.load(grid_path) as data:
                triggered_years = float(
                    np.asarray(data["triggered_years"]).squeeze()
                )
            results[(float(fj_bns), alpha, series)] = {
                "R_BNS": mu_bns / triggered_years,
                "R_NSBH": mu_nsbh / triggered_years,
            }
    return results


def plot_posterior_grid(
    dictionary_of_results: Mapping[
        tuple[float, str, str],
        Mapping[str, np.ndarray],
    ],
    fj_list: Sequence[float],
    alpha_list: Sequence[str],
    nsbh_populations: Sequence[str],
    swap_axes: bool = False,
    bins: int = 40,
    levels: Sequence[float] = (0.68, 0.95),
    log_axes: bool = True,
    minimum_fraction: float = 1e-2,
    minimum_plot_fraction: float = 5e-2,
    population_labels: Sequence[str] | None = None,
):
    """Plot BNS/NSBH rate fractions in an alpha-by-jet-fraction grid."""
    import corner

    if population_labels is None:
        population_labels = list(nsbh_populations)
    if len(population_labels) != len(nsbh_populations):
        raise ValueError(
            "population_labels must match nsbh_populations in length"
        )

    row_values = fj_list if swap_axes else alpha_list
    col_values = alpha_list if swap_axes else fj_list
    rows, columns = len(row_values), len(col_values)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(4 * columns, 4 * rows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    colors = ["k", "C1", "C2", "C3", "C4", "C5"]
    lower = minimum_fraction if log_axes else 0.0
    lower = minimum_plot_fraction

    for row_index, row_value in enumerate(row_values):
        for column_index, column_value in enumerate(col_values):
            alpha = column_value if swap_axes else row_value
            fj_bns = row_value if swap_axes else column_value
            axis = axes[row_index, column_index]

            for population_index, nsbh_population in enumerate(
                nsbh_populations
            ):
                entry = dictionary_of_results[
                    (float(fj_bns), alpha, nsbh_population)
                ]
                rate_bns = np.asarray(entry["R_BNS"], dtype=float)
                rate_nsbh = np.asarray(entry["R_NSBH"], dtype=float)
                total = rate_bns + rate_nsbh
                valid = np.isfinite(total) & (total > 0)
                eta_bns = rate_bns[valid] / total[valid]
                eta_nsbh = rate_nsbh[valid] / total[valid]
                if log_axes:
                    eta_bns = np.clip(
                        eta_bns,
                        minimum_fraction,
                        1.0,
                    )
                    eta_nsbh = np.clip(
                        eta_nsbh,
                        minimum_fraction,
                        1.0,
                    )
                corner.hist2d(
                    eta_bns,
                    eta_nsbh,
                    ax=axis,
                    color=colors[population_index % len(colors)],
                    bins=bins,
                    smooth=1,
                    plot_datapoints=False,
                    plot_density=True,
                    fill_contours=False,
                    levels=list(levels),
                    axes_scale=(
                        ["log", "log"]
                        if log_axes
                        else ["linear", "linear"]
                    ),
                    contour_kwargs={
                        "linestyles": ["--", "-"],
                        "linewidths": [1.2, 1.5],
                    },
                )

            if log_axes:
                axis.set_xscale("log")
                axis.set_yscale("log")
            axis.set_xlim(lower, 1.0)
            axis.set_ylim(lower, 1.0)
            if row_index == rows - 1:
                axis.set_xlabel(r"$\eta_{\rm BNS}$")
            if column_index == 0:
                axis.set_ylabel(r"$\eta_{\rm NSBH}$")
            alpha_label = (
                str(alpha)[1:]
                if str(alpha).startswith("A")
                else str(alpha)
            )
            axis.text(
                0.05,
                0.05,
                rf"$\alpha_{{\rm CE}}={alpha_label}$ | "
                rf"$f_j={float(fj_bns):g}$",
                transform=axis.transAxes,
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
    axes[0, min(2, columns - 1)].legend(
        handles=handles,
        loc="upper right",
    )
    fig.tight_layout()
    return fig, axes


def _load_full_shared_chain(
    backend_path: str | Path,
    *,
    burn_frac: float,
    thin: int,
) -> np.ndarray:
    if not 0 <= burn_frac < 1:
        raise ValueError("burn_frac must satisfy 0 <= burn_frac < 1")
    if thin < 1:
        raise ValueError("thin must be at least 1")
    path = Path(backend_path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing full-run backend: {path}")
    backend = emcee.backends.HDFBackend(path, read_only=True)
    discard = int(backend.iteration * burn_frac)
    full = backend.get_chain(discard=discard, thin=thin, flat=True)
    if full.shape[1] <= max(FULL_SHARED_COLUMNS):
        raise ValueError(
            "The full chain does not contain the expected seven parameters"
        )
    return np.asarray(full[:, FULL_SHARED_COLUMNS], dtype=float)


def summarize_grid_vs_full(
    grid_path: str | Path,
    full_backend: str | Path,
    *,
    n_samples: int = 40_000,
    seed: int = 123,
    burn_frac: float = 0.33,
    thin: int = 10,
    quantiles: Sequence[float] = (5.0, 50.0, 95.0),
) -> dict[str, dict[str, np.ndarray]]:
    """Return shared-parameter quantiles for grid and fully joint results."""
    grid_samples, _, _ = draw_grid_samples(
        grid_path,
        n_samples=n_samples,
        seed=seed,
    )
    full_samples = _load_full_shared_chain(
        full_backend,
        burn_frac=burn_frac,
        thin=thin,
    )
    q = np.asarray(quantiles, dtype=float)
    if q.ndim != 1 or q.size == 0 or np.any((q < 0) | (q > 100)):
        raise ValueError("quantiles must be a nonempty sequence in [0, 100]")
    return {
        name: {
            "analytical_grid": np.percentile(grid_samples[:, index], q),
            "fully_joint_mcmc": np.percentile(full_samples[:, index], q),
        }
        for index, name in enumerate(BNS_NSBH_PARAMETER_NAMES)
    }


def plot_grid_vs_full(
    grid_path: str | Path,
    full_backend: str | Path,
    *,
    n_samples: int = 40_000,
    seed: int = 123,
    burn_frac: float = 0.33,
    thin: int = 10,
    grid_label: str = "BNS shape fixed: analytical grid",
    full_label: str = "Fully joint MCMC",
) -> plt.Figure:
    """Overlay the explicit-grid posterior and full MCMC shared marginals."""
    import corner

    grid_samples, _, _ = draw_grid_samples(
        grid_path,
        n_samples=n_samples,
        seed=seed,
    )
    full_samples = _load_full_shared_chain(
        full_backend,
        burn_frac=burn_frac,
        thin=thin,
    )
    ranges = [tuple(bounds) for bounds in COMBINED_PRIOR_BOUNDS]
    fig = corner.corner(
        full_samples,
        labels=COMBINED_LABELS,
        range=ranges,
        bins=30,
        smooth=1,
        smooth1d=1,
        plot_datapoints=False,
        plot_density=True,
        levels=[0.68, 0.95],
        color="C0",
    )
    corner.corner(
        grid_samples,
        labels=COMBINED_LABELS,
        range=ranges,
        bins=30,
        smooth=1,
        smooth1d=1,
        plot_datapoints=False,
        plot_density=True,
        levels=[0.68, 0.95],
        color="C1",
        fig=fig,
    )
    fig.legend(
        handles=[
            plt.Line2D([0], [0], color="C0", lw=2, label=full_label),
            plt.Line2D([0], [0], color="C1", lw=2, label=grid_label),
        ],
        loc="upper right",
        frameon=False,
    )
    return fig


__all__ = [
    "ALL_PARAMETER_NAMES",
    "BNS_NSBH_PARAMETER_NAMES",
    "BNS_ONLY_LABELS",
    "BNS_ONLY_PARAMETER_NAMES",
    "COMBINED_LABELS",
    "COMBINED_PRIOR_BOUNDS",
    "DATAFILES",
    "DEFAULT_INITIAL_CENTER_SINGLE",
    "DEFAULT_INITIAL_SCALE_SINGLE",
    "FIXED_PARAMETER_NAMES",
    "FULL_SHARED_COLUMNS",
    "LABELS",
    "N_PARAMS_COMBINED",
    "N_PARAMS_SINGLE",
    "PRIOR_BOUNDS",
    "bns_backend_path",
    "build_redshift_ppf",
    "collect_rate_posteriors",
    "combined_grid_path",
    "draw_grid_samples",
    "extract_bns_best_fits",
    "full_backend_path",
    "generate_base_population",
    "lognormal_numpy",
    "luminosity_gen",
    "plot_bns_corners",
    "plot_corner",
    "plot_corner_grid_multiples",
    "plot_grid_corner",
    "plot_grid_vs_full",
    "plot_posterior_grid",
    "prepare_base_populations",
    "run_analytical_joint",
    "run_bns_only",
    "run_pop",
    "run_pop_combined",
    "score_func_cvm_exact",
    "summarize_grid_vs_full",
]
