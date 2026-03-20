"""
structured_jet/init.py
======================
Initialisation wrapper for the structured-jet pipeline.

Wraps ``src.init.initialize_simulation`` and patches the SimParams with the
clamped R_F / R_E arrays produced by ``get_Rf_Re_clamped`` (§6.1).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from ..init import initialize_simulation as _initialize_simulation_parent
from ..montecarlo import SimParams, Interps
from ..spectral_models import DEFAULT_SPECTRAL_PARAMS
from .data_io import get_Rf_Re_clamped


def initialize_simulation(
        datafiles : Path           = Path("datafiles"),
        params    : Dict[str, Any] = DEFAULT_SPECTRAL_PARAMS,
        size_test : int            = 2_000,
) -> Tuple[SimParams, Interps, Dict[str, np.ndarray]]:
    """
    Initialise the structured-jet Monte Carlo simulation.

    Delegates to ``src.init.initialize_simulation`` then replaces the
    R_F / R_E arrays in the returned ``SimParams`` with clamped versions
    (§6.1).

    Parameters
    ----------
    datafiles : Path
        Directory containing data files (same convention as parent).
    params : dict
        Spectral and geometry parameters (same convention as parent).
    size_test : int
        Number of viewing-angle samples to draw.

    Returns
    -------
    sim_params : SimParams
        Simulation parameters with clamped R_F and R_E arrays.
    interps : Interps
        Pre-computed spectral and temporal interpolators.
    data_dict : dict
        Catalogue observables.
    """
    # Run the full parent initialisation
    sim_params, interps, data_dict = _initialize_simulation_parent(
        datafiles=datafiles, params=params, size_test=size_test
    )

    # §6.1 – Patch R_F and R_E with clamped interpolators
    jet_table = datafiles / "F_Fmax_3.4_s4.0.txt"
    R_F_clamped, R_E_clamped, _ = get_Rf_Re_clamped(jet_table)

    # sim_params.theta_v holds the viewing-angle samples used at init time
    sim_params.R_F = R_F_clamped(sim_params.theta_v)
    sim_params.R_E = R_E_clamped(sim_params.theta_v)

    return sim_params, interps, data_dict
