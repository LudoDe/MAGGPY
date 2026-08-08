"""
init.py - Initialization Module

This module contains all the functions needed to initialize the Monte Carlo simulation. 
"""
from __future__ import annotations
from dataclasses import dataclass

import time
import numpy            as np
import pandas           as pd
from pathlib            import Path
from scipy.interpolate  import interp1d
from astropy            import units as u
from typing             import Mapping, Callable
from astropy.cosmology  import Planck18
from .montecarlo        import create_k_interpolator
from ..utils            import compute_luminosity_distance
from ..redshift         import sample_from_mrd
from ..data_io          import catalogue_prep


SEED = 42  # Seed for reproducibility

@dataclass
class PopulationData:
    """MRD and Monte Carlo data shared by BNS and NSBH populations."""
    """Exact identification of one exported MRD curve."""
    mrd_path            : Path
    # Fixed Monte Carlo sample
    z_arr               : np.ndarray
    distances           : np.ndarray

    # Underlying MRD
    z_grid      : np.ndarray
    mrd_density : np.ndarray

    # Observer-frame redshift distribution
    P_z_density         : np.ndarray
    total_merger_rate   : float
    local_rate          : float

    # Just to have everything in the same place
    k_interpolator     : Callable

def load_population(
    mrd_path    : Path,
    rng         : np.random.Generator,
) -> PopulationData:
    """Load one exported MRD curve and construct its observer-frame P(z)."""

    if not mrd_path.is_file():
        raise FileNotFoundError(
            f"MRD file not found for : {mrd_path}"
        )

    frame       = pd.read_csv(mrd_path)
    # first col is redshift, second col is MRD in Gpc^-3 yr^-1
    z_grid      = frame.iloc[:, 0].to_numpy(float)
    mrd_density = frame.iloc[:, 1].to_numpy(float)

    if np.any(np.diff(z_grid) <= 0): raise ValueError(f"MRD redshifts are not strictly increasing: {mrd_path}")

    # Full-sky differential comoving volume in Gpc^3 per unit redshift.
    dVc_dz = (
        Planck18
        .differential_comoving_volume(z_grid)
        .to_value(u.Gpc**3 / u.sr)
        * 4.0
        * np.pi
    )

    # Observer-frame rate:
    # dN/dz = R(z) [dVc/dz] / (1 + z)
    dN_dz = mrd_density * dVc_dz / (1.0 + z_grid)

    total_rate = float(np.trapezoid(dN_dz, z_grid))

    if not np.isfinite(total_rate) or total_rate <= 0:
        raise ValueError(
            f"Integrated merger rate is invalid for {mrd_path}: {total_rate}"
        )

    P_z_density = dN_dz / total_rate

    z_arr = sample_from_mrd(
        z_grid,
        P_z_density,
        int(total_rate), 
        rng=rng,
    )

    distances       = compute_luminosity_distance(z_arr)
    k_interpolator  = create_k_interpolator()
    return PopulationData(
        mrd_path=mrd_path,
        z_arr=z_arr,
        distances=distances,
        z_grid=z_grid,
        mrd_density=mrd_density,
        P_z_density=P_z_density,
        total_merger_rate=total_rate,
        local_rate=float(mrd_density[0]),
        k_interpolator=k_interpolator,
    )

def initialize_bns_simulation(
    datafiles   : Path,
    mrd_path    : Path,
    seed        : int = 42,
) -> tuple[PopulationData, Mapping[str, np.ndarray]]:
    """Load one BNS MRD and the observed GBM catalogue.

    Unlike ``initialize_combined_simulation``, this function never resolves or
    reads an NSBH MRD file.
    """
    datafiles   = Path(datafiles)

    bns_data = load_population(
        mrd_path=mrd_path,
        rng=np.random.default_rng(seed),
    )
    observations = catalogue_prep(datafiles=datafiles)

    print(
        f"BNS: R(z={bns_data.z_grid[0]:.3g})="
        f"{bns_data.local_rate:.2f} Gpc^-3 yr^-1; "
        f"all-sky rate={bns_data.total_merger_rate:.3e} yr^-1; "
        f"MC sample={len(bns_data.z_arr)}"
    )
    return bns_data, observations