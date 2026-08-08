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
from typing import Callable, Dict

import astropy.units as u
import pandas as pd

from astropy.cosmology import Planck18
from scipy.interpolate import interp1d

import numpy as np
from ..redshift             import sample_from_mrd
from ..top_hat.montecarlo   import compute_luminosity_distance
from ..data_io              import catalogue_prep

SEED = 42


# ---------------------------------------------------------------------------
# NSBH data container
# ---------------------------------------------------------------------------
@dataclass
class MRDSelection:
    """Exact identification of one exported MRD curve."""
    population: str
    alpha: str
    series: str

    @property
    def filename(self) -> str:
        alpha_token = self.alpha.replace(".", "_")
        return f"{self.population}_{alpha_token}_{self.series}.csv"

@dataclass
class PopulationData:
    """MRD and Monte Carlo data shared by BNS and NSBH populations."""
    """Exact identification of one exported MRD curve."""
    selection: MRDSelection
    source_file: Path
    # Fixed Monte Carlo sample
    z_arr               : np.ndarray
    distances           : np.ndarray

    # Underlying MRD
    z_grid: np.ndarray
    mrd_density: np.ndarray

    # Observer-frame redshift distribution
    P_z_interp: Callable
    P_z_density: np.ndarray
    total_merger_rate: float

    # Lowest-redshift MRD point
    local_rate: float
    local_redshift: float

def load_population(
    mrd_directory: Path,
    selection: MRDSelection,
    n_samples: int,
    rng: np.random.Generator,
) -> PopulationData:
    """Load one exported MRD curve and construct its observer-frame P(z)."""

    mrd_path = Path(mrd_directory) / selection.filename

    if not mrd_path.is_file():
        raise FileNotFoundError(
            f"MRD file not found for {selection}: {mrd_path}"
        )

    frame       = pd.read_csv(mrd_path)
    # first col is redshift, second col is MRD in Gpc^-3 yr^-1
    z_grid      = frame["redshift"].to_numpy(float)
    mrd_density = frame["MRD_Gpc-3_yr-1"].to_numpy(float)

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

    P_z_interp = interp1d(
        z_grid,
        P_z_density,
        kind="linear",
        bounds_error=False,
        fill_value=0.0,
        assume_sorted=True,
    )

    z_arr = sample_from_mrd(
        P_z_interp,
        z_grid,
        P_z_density,
        n_samples,
        rng=rng,
    )

    distances = compute_luminosity_distance(z_arr)

    return PopulationData(
        selection=selection,
        source_file=mrd_path,
        z_arr=z_arr,
        distances=distances,
        z_grid=z_grid,
        mrd_density=mrd_density,
        P_z_interp=P_z_interp,
        P_z_density=P_z_density,
        total_merger_rate=total_rate,
        local_rate=float(mrd_density[0]),
        local_redshift=float(z_grid[0]),
    )

def initialize_combined_simulation(
    datafiles       : Path,
    population      : str = "delayed",
    alpha           : str = "A1.0",
    nsbh_series     : str = "NSBH_DD2_uniform_chi_0_1",
    sample_size     : int = 2_000,
    seed            : int = SEED,
) -> tuple[PopulationData, PopulationData, Dict[str, np.ndarray]]:
    """Initialize matching BNS and NSBH top-hat populations."""

    mrd_directory   = datafiles / "MRD_outputs"

    # Independent, reproducible random streams.
    bns_seed, nsbh_seed = np.random.SeedSequence(seed).spawn(2)
    bns_rng     = np.random.default_rng(bns_seed)
    nsbh_rng    = np.random.default_rng(nsbh_seed)

    bns_selection = MRDSelection(
        population=population,
        alpha=alpha,
        series="BNS",
    )

    nsbh_selection = MRDSelection(
        population=population,
        alpha=alpha,
        series=nsbh_series,
    )

    bns_data = load_population(
        mrd_directory=mrd_directory,
        selection=bns_selection,
        n_samples=sample_size,
        rng=bns_rng,
    )

    nsbh_data = load_population(
        mrd_directory=mrd_directory,
        selection=nsbh_selection,
        n_samples=sample_size,
        rng=nsbh_rng,
    )

    # Observational GBM catalogue. This is not population-specific.
    observations = catalogue_prep(datafiles=datafiles)

    for data in (bns_data, nsbh_data):
        print(
            f"{data.selection.series}: "
            f"R(z={data.local_redshift:.3g})="
            f"{data.local_rate:.2f} Gpc^-3 yr^-1; "
            f"all-sky rate={data.total_merger_rate:.3e} yr^-1; "
            f"MC sample={len(data.z_arr)}"
        )

    return bns_data, nsbh_data, observations