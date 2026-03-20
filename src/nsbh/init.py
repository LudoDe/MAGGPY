"""
src/nsbh/init.py
================
Initialisation for the combined BNS + NSBH top-hat model (lognormal θ_c).

Returns the standard BNS ``SimParams`` plus an ``NSBHData`` container
that carries the NSBH merger-rate-density information and pre-computed
luminosity distances needed at each MCMC step.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from ..montecarlo import SimParams, Interps
from ..spectral_models import DEFAULT_SPECTRAL_PARAMS
from ..init import initialize_simulation as _init_bns
from ..redshift import get_mrd_redshift_distribution, sample_from_mrd
from ..top_hat.montecarlo import compute_luminosity_distance

SEED = 42


# ---------------------------------------------------------------------------
# NSBH data container
# ---------------------------------------------------------------------------
@dataclass
class NSBHData:
    """
    Holds NSBH (BHNSs) population data for the top-hat combined model.

    Fields
    ------
    z_arr             : 1-year sample of NSBH merger redshifts
    distances         : luminosity distances in cm (one per z_arr entry)

    # MRD information
    P_z_interp, total_merger_rate, local_rate, z_grid, P_z_density
    """
    z_arr               : np.ndarray
    distances           : np.ndarray

    # MRD-derived quantities
    P_z_interp          : Callable
    total_merger_rate   : float
    local_rate          : float
    z_grid              : np.ndarray
    P_z_density         : np.ndarray


# ---------------------------------------------------------------------------
# Helper: build NSBHData from an MRD file
# ---------------------------------------------------------------------------
def _load_nsbh_population(
    datafiles   : Path,
    population  : str,
    alpha       : str,
    sigma       : float = 0.1,
) -> NSBHData:
    """
    Load the BHNSs merger-rate density for *one* population model and
    return an ``NSBHData`` container.
    """
    # 1. MRD distribution
    P_z_interp, total_rate, local_rate, z_grid, P_z_density = (
        get_mrd_redshift_distribution(
            datafiles   = datafiles,
            population  = population,
            alpha       = alpha,
            component   = "BHNSs",
            sigma       = sigma,
        )
    )

    # 2. Draw 1-year redshift sample via inverse-CDF
    rng = np.random.default_rng(SEED)
    n_samples = int(total_rate)  #? Number of events in 1 year = total merger rate
    z_arr = sample_from_mrd(P_z_interp, z_grid, P_z_density, n_samples, rng=rng)

    # 3. Pre-compute luminosity distances (cm)
    distances = compute_luminosity_distance(z_arr)

    return NSBHData(
        z_arr             = z_arr,
        distances         = distances,
        P_z_interp        = P_z_interp,
        total_merger_rate = total_rate,
        local_rate        = local_rate,
        z_grid            = z_grid,
        P_z_density       = P_z_density,
    )


# ---------------------------------------------------------------------------
# Main initialisation entry-point
# ---------------------------------------------------------------------------
def initialize_combined_simulation(
    datafiles       : Path           = Path("datafiles"),
    params          : Dict[str, Any] = DEFAULT_SPECTRAL_PARAMS,
    size_test       : int            = 2_000,
    nsbh_population : Optional[str]  = None,
    nsbh_alpha      : Optional[str]  = None,
    sigma           : float          = 0.1,
) -> Tuple[SimParams, NSBHData, Dict[str, np.ndarray]]:
    """
    Initialise the combined BNS + NSBH top-hat simulation.

    Parameters
    ----------
    datafiles : Path
        Root data directory.
    params : dict
        Spectral / geometry parameters (passed to the BNS init).
    size_test : int
        Number of BNS viewing-angle samples.
    nsbh_population : str or None
        Population model name for BHNSs (e.g. ``"fiducial_Hrad"``).
        If *None*, inferred from ``params["z_model"]``.
    nsbh_alpha : str or None
        Alpha parameter for BHNSs (e.g. ``"A1.0"``).
        If *None*, inferred from ``params["z_model"]``.
    sigma : float
        MRD sigma parameter (default 0.1).

    Returns
    -------
    bns_params : SimParams
        Standard BNS simulation parameters (from top-hat init).
    nsbh_data  : NSBHData
        NSBH population container.
    data_dict  : dict
        Observed catalogue data.
    """
    import re

    # ── 1. Standard BNS initialisation (top-hat pipeline) ───────────────────
    bns_params, interps, data_dict = _init_bns(
        datafiles=datafiles, params=params, size_test=size_test
    )

    # ── 2. Resolve NSBH population / alpha from z_model if not given ────────
    if nsbh_population is None or nsbh_alpha is None:
        z_model = params.get("z_model", None)
        if z_model is not None:
            alpha_match = re.search(r"_(A\d+\.?\d*)$", z_model)
            if alpha_match:
                if nsbh_alpha is None:
                    nsbh_alpha = alpha_match.group(1)
                if nsbh_population is None:
                    nsbh_population = z_model[: alpha_match.start()]

    # Fall back to defaults
    if nsbh_population is None:
        print("Warning: NSBH population not specified, defaulting to 'fiducial_Hrad'")
        nsbh_population = "fiducial_Hrad"
    if nsbh_alpha is None:
        print("Warning: NSBH alpha not specified, defaulting to 'A1.0'")
        nsbh_alpha = "A1.0"

    # ── 3. Load NSBH population ─────────────────────────────────────────────
    nsbh_data = _load_nsbh_population(
        datafiles   = datafiles,
        population  = nsbh_population,
        alpha       = nsbh_alpha,
        sigma       = sigma,
    )

    print(f"NSBH population: {nsbh_population} / {nsbh_alpha}")
    print(f"  BHNSs local rate R_0 = {nsbh_data.local_rate:.1f} Gpc^-3 yr^-1")
    print(f"  BHNSs total rate     = {nsbh_data.total_merger_rate:.0f} yr^-1")
    print(f"  1-year z sample size = {len(nsbh_data.z_arr)}")

    return bns_params, nsbh_data, data_dict
