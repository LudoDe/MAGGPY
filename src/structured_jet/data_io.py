"""
structured_jet/data_io.py
=========================
Data I/O helpers for the structured-jet model.

Improvements vs. src/data_io.py:

  §6.1 – get_Rf_Re_clamped: R_F and R_E interpolators now use clamped boundary
          values for out-of-table viewing angles instead of linear extrapolation.
          Linear extrapolation can produce negative (unphysical) values for very
          large viewing angles.  Clamping to the last tabulated value is the
          physically motivated choice: beyond the table edge the jet structure is
          effectively off-axis and no additional information is available.
"""

from __future__ import annotations

import numpy as np
from scipy import interpolate
from typing import Callable, Tuple
from pathlib import Path

# Re-export unchanged helpers from the parent package
from ..data_io import (
    get_Rf_Re,          # original (extrapolate="extrapolate") – kept for reference
    get_alpha_n_alpha_e,
    get_observables_data,
    get_redshift_distribution,
    catalogue_prep,
)


def get_Rf_Re_clamped(filename: str | Path) -> Tuple[Callable, Callable, np.ndarray]:
    """
    Load the jet-structure scaling functions R_F(θ_v) and R_E(θ_v) and return
    clamped interpolators.

    §6.1 – Unlike the parent get_Rf_Re, the interpolators produced here use
    ``fill_value=(f[0], f[-1])`` (boundary clamping) with ``bounds_error=False``
    so that viewing angles outside the table are silently clamped to the nearest
    tabulated value.  

    Parameters
    ----------
    filename : str or Path
        Path to the jet-structure table (e.g. ``F_Fmax_3.4_s4.0.txt``).
        File must have three columns: theta_v, F/F_max, E/E_max.

    Returns
    -------
    R_F : callable
        Interpolator for the normalised flux scaling F(θ_v) / F(0).
    R_E : callable
        Interpolator for the normalised peak-energy scaling E(θ_v) / E(0).
    theta_v_arr : np.ndarray
        Viewing-angle grid from the file (radians).
    """
    data = np.loadtxt(filename).T
    theta_v_arr, f_fmax, E_Emax = data

    # Normalise to on-axis value
    f_fmax  = f_fmax / f_fmax[0]
    E_Emax  = E_Emax / E_Emax[0]

    # §6.1 – clamped interpolators
    R_F = interpolate.interp1d(
        theta_v_arr, f_fmax,
        kind="linear",
        bounds_error=False,
        fill_value=(f_fmax[0], f_fmax[-1]),   # clamp to boundary
    )
    R_E = interpolate.interp1d(
        theta_v_arr, E_Emax,
        kind="linear",
        bounds_error=False,
        fill_value=(E_Emax[0], E_Emax[-1]),   # clamp to boundary
    )

    return R_F, R_E, theta_v_arr
