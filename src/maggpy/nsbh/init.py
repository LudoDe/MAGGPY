from __future__ import annotations

from pathlib import Path
from typing import Dict
import numpy as np
from ..data_io              import catalogue_prep
from ..top_hat.init         import PopulationData, load_population

SEED = 42

def initialize_combined_simulation(
    datafiles       : Path,
    mrd_bns_path    : Path,
    mrd_nsbh_path   : Path,
    seed            : int = SEED,
) -> tuple[PopulationData, PopulationData, Dict[str, np.ndarray]]:
    """Initialize matching BNS and NSBH top-hat populations."""

    # Independent, reproducible random streams.
    bns_seed, nsbh_seed = np.random.SeedSequence(seed).spawn(2)
    bns_rng     = np.random.default_rng(bns_seed)
    nsbh_rng    = np.random.default_rng(nsbh_seed)

    bns_data = load_population(
        mrd_path=mrd_bns_path,
        rng=bns_rng,
    )

    nsbh_data = load_population(
        mrd_path=mrd_nsbh_path,
        rng=nsbh_rng,
    )

    # Observational GBM catalogue. This is not population-specific.
    observations = catalogue_prep(datafiles=datafiles)

    return bns_data, nsbh_data, observations