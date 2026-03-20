"""
src/structured_jet
==================
Improved pipeline for the power-law structured-jet model.

Applies all fixes from code_review_suggestions.md that are
tractable without adding external dependencies:

  Critical
  --------
  §1.1 – RNG reset removed from log_likelihood_ (now deprecated).
  §1.2 – Per-call local RNG seeded from hash(thetas) for CvM scoring.

  Detector model
  --------------
  §2.1 – GBM_EFF documented as a module constant with citation.
  §2.2 – Hard flux cut kept (catalogue was built with this exact cut).
  §2.4 – Double T90 cut fixed: t_peak pre-filter uses 3× safety factor.

  Luminosity function
  -------------------
  §3.1 – L_MIN_FACTOR cut applied after Schechter draw.

  Likelihood robustness
  ---------------------
  §4.3 – Assertion: n_events_fixed adequate for expected rate.
  §4.4 – log_likelihood_ deprecated; log_likelihood_fj is the canonical call.

  Numerics / software
  -------------------
  §6.1 – R_F / R_E interpolators clamped at table boundaries.
  §6.3 – Dead S_obs variable eliminated.
  §6.4 – generate_grb_population forwards R_F_theta_det.

Public API
----------
  initialize_simulation   – sets up SimParams with clamped R_F/R_E
  log_likelihood_fj       – canonical likelihood (fj via Poisson term, CvM shape)
  log_likelihood_fj_pca   – alternative likelihood (PCA-whitened binned shape term)
  precompute_pca_transform – pre-compute PCA basis from observed data
  make_observations       – forward simulation with fixed-sample size
  generate_grb_population – PPC helper
  score_func              – CvM shape score with local-RNG fix
  GBM_EFF, L_MIN_FACTOR, N_YEARS_DEFAULT – module constants
"""

from .montecarlo import (
    log_likelihood_fj,
    log_likelihood_fj_pca,
    log_likelihood_fj_yonetoku,
    precompute_pca_transform,
    make_observations,
    make_observations_yonetoku,
    generate_macro_properties,
    generate_macro_properties_yonetoku,
    generate_grb_population,
    generate_grb_population_yonetoku,
    score_func,
    GBM_EFF,
    L_MIN_FACTOR,
    N_YEARS_DEFAULT,
)

from .init import initialize_simulation

from .data_io import (
    get_Rf_Re_clamped,
    # pass-through from parent
    get_alpha_n_alpha_e,
    get_observables_data,
    catalogue_prep,
)

# Re-export unchanged symbols from the parent montecarlo for convenience
from ..montecarlo import (
    SimParams,
    Interps,
    DEFAULT_LIMITS,
    N_SIMS,
    poiss_log,
)

from ..prior_factory import (
    create_log_prior,
    initialize_walkers,
    DEFAULT_PRIOR_BOUNDS,
    create_log_prior_yonetoku,
    initialize_walkers_yonetoku,
    DEFAULT_PRIOR_BOUNDS_YONETOKU,
)

__all__ = [
    # core simulation
    "initialize_simulation",
    "make_observations",
    "make_observations_yonetoku",
    "generate_macro_properties",
    "generate_macro_properties_yonetoku",
    "generate_grb_population",
    "generate_grb_population_yonetoku",
    # likelihood
    "log_likelihood_fj",
    "log_likelihood_fj_yonetoku",
    "log_likelihood_deprecated",
    # shape scoring
    "score_func",
    # data helpers
    "get_Rf_Re_clamped",
    "get_alpha_n_alpha_e",
    "get_observables_data",
    "catalogue_prep",
    # data structures
    "SimParams",
    "Interps",
    "DEFAULT_LIMITS",
    "N_SIMS",
    "poiss_log",
    # priors
    "create_log_prior",
    "create_log_prior_yonetoku",
    "initialize_walkers_yonetoku",
    "DEFAULT_PRIOR_BOUNDS_YONETOKU",
    "initialize_walkers",
    "DEFAULT_PRIOR_BOUNDS",
    # constants
    "GBM_EFF",
    "L_MIN_FACTOR",
    "N_YEARS_DEFAULT",
]
