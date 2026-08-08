import src.init
import numpy as np

from typing import Callable, Mapping, Sequence
from pathlib import Path

N_PARAMS                = 7
DEFAULT_INITIAL_CENTER  = np.asarray([2, 0.5, -1.0, 2.5, 0.5, 10.0, 10.0])
DEFAULT_INITIAL_SCALE   = np.asarray([0.30, 0.25, 0.15, 0.15, 0.08, 1.50, 1.50])
BLOBS_DTYPE = [
    ("l_pflux", float),
    ("l_epeak", float),
    ("l_poiss", float),
    ("mu_bns", float),
    ("mu_nsbh", float),
]
PRIOR_BOUNDS = np.asarray(
    [
        (1.5, 6),
        (-2.0, 7.0),
        (-2.0, 0.0),
        (0.1, 7.0),
        (0.0, 2.5),
        (1, 25),
        (1, 25),
    ],
    dtype=float,
)

def _safe_name(value: object) -> str:
    """Return a filesystem-friendly representation of a run setting."""
    return "".join(char if char.isalnum() or char in "-." else "-" for char in str(value))

def _geometry_name(geom_eff_func: Callable) -> str:
    return _safe_name(getattr(geom_eff_func, "__name__", geom_eff_func.__class__.__name__))

def _run_name(
    alpha: str,
    fj_bns: float,
    geom_eff_func: Callable,
    nsbh_population: str = "fiducial_delayed_cut",
) -> str:
    return (
        f"full_likelihood_alpha_{_safe_name(alpha)}"
        f"_fj_{float(fj_bns):.6g}"
        f"_{_geometry_name(geom_eff_func)}"
        f"_{_safe_name(nsbh_population)}"
    )

def _backend_path(
    alpha: str,
    fj_bns: float,
    geom_eff_func: Callable,
    nsbh_population: str = "fiducial_delayed_cut",
) -> Path:
    run_name = _run_name(alpha, fj_bns, geom_eff_func, nsbh_population)
    return src.init.create_run_dir(
        run_name,
        output_files_default="Output_files",
    ) / "emcee.h5"

def flat_prior(thetas: Sequence[float], n_params: int = N_PARAMS, bounds: np.ndarray = PRIOR_BOUNDS) -> float:
    """Independent flat prior over the six-parameter support."""
    theta = np.asarray(thetas, dtype=float)
    if theta.shape != (n_params,) or not np.all(np.isfinite(theta)): return -np.inf
    lower = bounds[:, 0] 
    upper = bounds[:, 1] 
    if np.all((theta > lower) & (theta < upper)): return 0.0
    return -np.inf

def _bad_likelihood() -> tuple[float, float, float, float, float, float]:
    return (-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf)

def _make_initial_walkers(
    n_walkers: int,
    rng: np.random.Generator,
    initial_center: Sequence[float] | None = None,
    initial_scale: Sequence[float] | None = None,
) -> np.ndarray:
    """Create prior-valid walkers without consulting any earlier posterior."""

    n_params = len(DEFAULT_INITIAL_CENTER) # has to match anyways

    center = np.asarray(
        DEFAULT_INITIAL_CENTER if initial_center is None else initial_center,
        dtype=float,
    )
    scale = np.asarray(
        DEFAULT_INITIAL_SCALE if initial_scale is None else initial_scale,
        dtype=float,
    )

    if not np.all(scale > 0): raise ValueError("Every initial_scale entry must be positive")
    if not np.isfinite(flat_prior(center)): raise ValueError("initial_center must lie strictly inside the flat-prior bounds")

    walkers = center + rng.normal(size=(n_walkers, n_params)) * scale
    invalid = np.asarray([not np.isfinite(flat_prior(row)) for row in walkers])
    while np.any(invalid):
        walkers[invalid] = center + rng.normal(size=(np.sum(invalid), n_params)) * scale
        invalid = np.asarray([not np.isfinite(flat_prior(row)) for row in walkers])
    return walkers

