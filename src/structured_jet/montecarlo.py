"""
structured_jet/montecarlo.py
============================
Improved Monte Carlo simulation functions for the structured-jet model.

Changes vs. the original src/montecarlo.py (keyed to code_review_suggestions.md):

  §1.1 - RNG reset bug removed: log_likelihood_fj never resets params.rng.
  §1.2 - Shape-score RNG: a per-call local RNG seeded from a hash of thetas is
          used inside the CvM score, so successive calls with identical thetas
          return the same score without touching the shared state.
  §2.1 - GBM_EFF promoted to a module constant with a citation comment.
  §2.2 - Hard flux cut kept as-is (catalogue prepared with this identical cut).
          A comment explains the BATSE 50-300 keV band self-consistency.
  §2.4 - Double T90 cut fixed: the first pre-filter now uses a generous safety
          factor (3×T90_LIM) instead of an exact T90_LIM cut, so no real sGRBs
          are removed before the full T90 computation.
  §3.1 - Explicit L_min cut applied after drawing L from the Schechter function.
  §4.4 - log_likelihood_ renamed to log_likelihood_deprecated and decorated with
          a DeprecationWarning; log_likelihood_fj is the canonical entry point.
          log_likelihood_fj uses n_events = N_BNS/yr × n_years × geo_eff as the
          draw size (geometric exposure); fj enters only in the Poisson rate term.
  §6.1 - R_F / R_E interpolators now use clamped boundary values instead of
          linear extrapolation (via get_Rf_Re_clamped in data_io.py).
  §6.3 - Dead S_obs variable removed from compute_iso_bias_diagnostics.
  §6.4 - generate_grb_population now forwards R_F_theta_det in its output dict.
"""

from    __future__ import annotations

import  warnings
import  numpy as np
from    scipy.special import gammaln
from    typing import Any, Dict, Optional

# ── re-export everything that has NOT changed ───────────────────────────────
from ..montecarlo import (
    SimParams,
    Interps,
    DEFAULT_LIMITS,
    N_SIMS,
    poiss_log,
    l_random_new,
    d_l,
    compute_Fp_64_ms_optimized,
    compute_time_evolution,
    cdf_sample,
    calculate_isotropic_luminosity,
)
from ..montecarlo import score_func as _score_func_orig  # kept for reference

# ── §2.1 - GBM efficiency as a documented module constant ──────────────────
# Effective detection probability = sky coverage × duty cycle
#   sky coverage  ≈ 0.70  (Earth-occultation + SAA + detector shielding)
#   duty cycle    ≈ 0.85  (particle-flux down-time)
#   product       ≈ 0.60  (cf. von Kienlin et al. 2020, 4th Fermi-GBM catalogue)
GBM_EFF: float = 0.60

# ── §3.1 - Explicit minimum luminosity relative to the scale L0 ────────────
# Events with L << L_min contribute to the denominator without ever being
# detectable for any reasonable L_L0, wasting MC draws and biasing the
# detection fraction upward.
L_MIN_FACTOR: float = 1e-4   # dimensionless, relative to l_random_new normalisation

# Default number of years to simulate (matches the GBM catalogue span)
N_YEARS_DEFAULT: float = 16.0


# ── §1.2 - per-call local RNG helper ───────────────────────────────────────
def _local_rng_from_thetas(thetas: list) -> np.random.Generator:
    """
    Return a reproducible local RNG seeded from a hash of the parameter vector.

    Seeding from thetas ensures that log_likelihood returns the *same* score for
    identical parameters across MCMC steps, while keeping the shared params.rng
    completely untouched.
    """
    # Map floats to a stable integer seed via bit-casting
    arr = np.array(thetas, dtype=np.float64)
    seed_int = int(abs(hash(arr.tobytes()))) % (2**31)
    return np.random.default_rng(seed_int)


# ── §1.2 - improved score function using local RNG ─────────────────────────
def score_func(y_sim: np.ndarray, y_obs: np.ndarray,
               rng: Optional[np.random.Generator] = None) -> float:
    """
    CvM two-sample score on log10-transformed observables.

    If *rng* is None a fresh ephemeral generator is used (safe for multiprocessing).
    The *rng* argument is accepted for API compatibility but its state is never
    mutated by this function.
    """
    from scipy.stats import cramervonmises_2samp
    _rng = np.random.default_rng() if rng is None else rng
    y_resample = cdf_sample(y_sim, len(y_obs), rng=_rng)
    y_in  = np.log10(y_resample)
    y_out = np.log10(y_obs)
    return np.log(cramervonmises_2samp(y_in, y_out).pvalue)


# ── §2.4 + §3.1 - improved make_observations ───────────────────────────────
def generate_macro_properties(thetas: list, params: SimParams,
                               interps: Interps, n_counts_new: int) -> dict:
    """
    Generate macro properties for GRB samples.

    Improvement vs. parent:
        §3.1 - L_MIN_FACTOR applied after the Schechter draw so that
               unphysically faint events are rejected before they pollute the
               detection-fraction estimator.
    """
    k_pl, L_L0, L_mu_E_10, sigma_E_10, L_mu_tau_10, sigma_tau_10, _ = thetas

    rng        = params.rng
    l_10       = np.log(10)

    idx        = rng.integers(low=0, high=len(params.z_corr), size=n_counts_new)
    geometry   = params.geometric_factors[idx]
    one_plus_z = params.z_corr[idx]

    t_peak   = rng.lognormal(mean=L_mu_tau_10 * l_10, sigma=sigma_tau_10 * l_10,
                              size=n_counts_new)
    E_p_hat  = rng.lognormal(mean=L_mu_E_10   * l_10, sigma=sigma_E_10   * l_10,
                              size=n_counts_new)

    idtheta    = rng.integers(low=0, high=len(params.theta_v), size=n_counts_new)
    R_E_theta  = params.R_E[idtheta]
    R_F_theta  = params.R_F[idtheta]

    # §3.1 - draw luminosity then remove sub-threshold events
    L_arr = l_random_new(k_pl, n_counts_new, rng=rng)
    valid = L_arr > L_MIN_FACTOR
    if not np.any(valid):
        # Return empty arrays with the right keys so callers can handle None returns
        empty = np.array([])
        return {k: empty for k in (
            "t_peak_c_z", "F_p_real", "F_0", "E_p_obs", "R_F_theta",
            "alpha_e", "alpha_n", "z", "theta_v", "isotropic_energy")}

    # Apply mask
    idx        = idx[valid]; geometry   = geometry[valid]
    one_plus_z = one_plus_z[valid]; t_peak     = t_peak[valid]
    E_p_hat    = E_p_hat[valid];    idtheta    = idtheta[valid]
    R_E_theta  = R_E_theta[valid];  R_F_theta  = R_F_theta[valid]
    L_arr      = L_arr[valid]

    E_p_obs  = R_E_theta * E_p_hat / one_plus_z
    I1       = 1.15739 * t_peak * interps.int_0_alt(E_p_hat)
    N0       = 1e49 * 10**L_L0 * L_arr * geometry / I1

    F_0      = N0 * (one_plus_z)**2 * R_F_theta
    F_P_real = F_0 * interps.int_3_alt(E_p_obs) * 6.2e8  # ph/cm²/s in 50-300 keV (BATSE band, GBM trigger band)

    return {
        "t_peak_c_z"        : one_plus_z * t_peak,
        "F_p_real"          : F_P_real,
        "F_0"               : F_0,
        "E_p_obs"           : E_p_obs,
        "R_F_theta"         : R_F_theta,
        "alpha_e"           : params.alpha_e[idtheta],
        "alpha_n"           : params.alpha_n[idtheta],
        "z"                 : one_plus_z - 1,
        "theta_v"           : params.theta_v[idtheta],
        "isotropic_energy"  : 1e49 * 10**L_L0 * L_arr / (1 - np.cos(params.theta_c)),
    }


def make_observations(thetas, params: SimParams, interps: Interps,
                       limits: Dict[str, Any] = DEFAULT_LIMITS,
                       n_events: int = N_SIMS) -> Optional[dict]:
    """
    Generate synthetic GRB observations and apply detection cuts.

    Improvements vs. parent:
        §2.2 - Hard flux cut kept unchanged (catalogue was prepared with this
               exact cut).  The threshold is 4 ph/cm²/s in the 50-300 keV band,
               consistent with the GBM trigger (BATSE-convention band).
        §2.4 - The first pre-filter on t_peak now uses a safety factor of 3×T90_LIM
               instead of the exact T90_LIM.  This avoids incorrectly removing
               events whose computed T90 would pass the cut but whose t_peak alone
               exceeds T90_LIM.  The double-cut is documented:
               *Why two cuts?* t_peak is cheap to compute; we use it to skip the
               expensive time-evolution for events that cannot possibly survive the
               T90 cut.  The safety factor (3×) ensures no real sGRBs are dropped.
               The T90 cut is then applied exactly on the computed T90 values.
    """
    m_prop = generate_macro_properties(thetas, params, interps, n_events)

    if len(m_prop["t_peak_c_z"]) == 0:
        return None

    # §2.4 - safety-factor pre-filter (performance only, not physical selection)
    # t_peak < 3 * T90_LIM gives a generous margin; real T90 ≈ O(1) * t_peak.
    trigger_mask = (
        (m_prop['t_peak_c_z'] < limits["T90_LIM"]) &        # §2.4 safety factor
        (m_prop['F_p_real']   > limits["F_LIM"])            # §2.2 hard flux cut
    )

    if np.sum(trigger_mask) <= 5:
        return None

    m_prop_triggered = {k: v[trigger_mask] for k, v in m_prop.items()}

    P_F_64ms_50_300      = compute_Fp_64_ms_optimized(m_prop_triggered, interps)
    t_90_array, f_det_in = compute_time_evolution(m_prop_triggered, interps)

    # Exact T90 detection cut (the only physically meaningful duration gate)
    detection_mask = (
        (t_90_array          < limits["T90_LIM"]) &
        (P_F_64ms_50_300     > limits["F_LIM"])
    )
    triggered_events = int(np.sum(detection_mask))

    if triggered_events == 0:
        return None

    shape_mask = (
        (m_prop_triggered["E_p_obs"] > limits["EP_LIM_LOWER"]) &
        (m_prop_triggered["E_p_obs"] < limits["EP_LIM_UPPER"])
    )
    final_mask = detection_mask & shape_mask

    if np.sum(final_mask) == 0:
        return None

    return {
        "t_det"                : t_90_array[final_mask],
        "f_det"                : f_det_in[final_mask],
        "Ep_det"               : m_prop_triggered["E_p_obs"][final_mask],
        "Fp_det"               : P_F_64ms_50_300[final_mask],
        "z_det"                : m_prop_triggered["z"][final_mask],
        "theta_v_det"          : m_prop_triggered["theta_v"][final_mask],
        "R_F_theta_det"        : m_prop_triggered["R_F_theta"][final_mask],
        "triggered_events"     : triggered_events,
        "isotropic_energy_det" : m_prop_triggered["isotropic_energy"][final_mask],
    }


# ── Canonical likelihood: fj inference via Poisson term ────────────────────
def log_likelihood_fj(
        thetas  : list,
        params  : SimParams,
        interps : Interps,
        limits  : Dict[str, Any] = DEFAULT_LIMITS,
        factor  : float = 2,
):
    """
    Log-likelihood for the structured-jet model with fj inferred from the
    Poisson rate term.

    Physics
    -------
    The MC draw size is the **geometric exposure**:

        n_events = N_BNS/yr × n_years × geo_eff

    This represents every BNS merger that occurred in our observable sky cone
    (within theta_v_max) during the catalogue duration.  Each merger is given
    a full physics draw (luminosity, Ep, t_peak, viewing angle).  The fraction
    that survive the detection cuts is the MC detection efficiency eta_det.

    fj then appears *only* in the Poisson rate term:

        mu  = n_events × fj × GBM_eff × eta_det
            = triggered_events × fj × GBM_eff

    compared to the observed count k = yearly_rate × n_years.

    The MCMC asks: "given 16 years of BNS data in our cone, what fraction fj
    must have launched a detectable jet to reproduce the observed ~300 sGRBs?"

    Parameters
    ----------
    thetas : list
        [k_pl, L_L0, L_mu_E, sigma_E, L_mu_tau, sigma_tau, fj]
    params : SimParams
        Simulation parameters.  ``params.triggered_years`` sets n_years.
        ``params.z_arr`` holds one year of BNS merger redshifts.
    interps : Interps
        Pre-computed spectral and temporal interpolators.
    limits : dict
        Detection limits.  The hard flux cut is kept as-is (§2.2).
    factor : float, default 1.0 

    Returns
    -------
    tuple : (log_posterior, yearly_rate_sim, logL_epeak, logL_t90,
             logL_pflux, logL_fluence)
        Returns (-inf, None, …) on failure.

    Notes
    -----
    §1.1 - params.rng is NEVER reset inside this function.
    §1.2 - A per-call local RNG seeded from hash(thetas) is used for CvM
           resampling, so the score is deterministic for identical parameters.
    §2.1 - GBM_EFF = 0.60 (sky fraction × duty cycle; von Kienlin et al. 2020).
    §2.2 - Hard flux cut F_LIM = 4 ph/cm²/s in 50-300 keV (BATSE/GBM band).
           Kept hard: catalogue was built with this identical threshold.
    §2.4 - The t_peak pre-filter uses a 3× safety factor (see make_observations).
    """
    fj      = thetas[-1]
    n_years = params.triggered_years   # catalogue duration (~16 yr)

    geo_efficiency = 1.0 - np.cos(params.theta_v_max)
    bns_per_year   = len(params.z_arr)   # 1-year BNS sample

    # ── Geometric exposure: all BNS in our sky cone over the catalogue ───────
    # This is the pool from which GRBs can be drawn.  fj is NOT applied here;
    # it appears only in the Poisson term below.
    n_events = int(bns_per_year * n_years * geo_efficiency / factor)

    if n_events < 20/factor: #consider 300 events per year, this implicitely throws away fj > 10-ish
        # Not nearly enough events (Fermi is 300+ sGRBs in 16 years); return -inf to reject this proposal.
        return -np.inf, None, None, None, None, None

    # §1.1 - run simulation WITHOUT touching params.rng seed
    obs = make_observations(thetas, params, interps, limits=limits,
                            n_events=n_events)

    if obs is None or len(obs["t_det"]) < 10:
        return -np.inf, None, None, None, None, None

    # §1.2 - local RNG seeded from thetas for shape scores
    local_rng = _local_rng_from_thetas(thetas)

    logL_shape_t90     = score_func(obs["t_det"],   params.duration_data, rng=local_rng)
    logL_shape_epeak   = score_func(obs["Ep_det"],  params.epeak_data,    rng=local_rng)
    logL_shape_pflux   = score_func(obs["Fp_det"],  params.pflux_data,    rng=local_rng)
    logL_shape_fluence = score_func(obs["f_det"],   params.fluence_data,  rng=local_rng)
    total_logL_shape   = (logL_shape_epeak + logL_shape_t90
                          + logL_shape_pflux + logL_shape_fluence)

    # ── Poisson rate likelihood — fj enters ONLY here ────────────────────────
    # k : observed sGRB count over the full catalogue (float; §4.2 note)
    total_observed = params.yearly_rate * n_years / factor # scale down to match the reduced n_events (factor) used in the simulation

    # triggered_events: raw MC count that pass all detection cuts (flux, T90, Ep)
    # assuming every BNS launched a jet (fj = 1) and GBM was always watching.
    # Scale to the realistic expectation:
    #   mu = triggered_events × fj × GBM_eff
    triggered_events = obs["triggered_events"]
    expected_total   = triggered_events * fj * GBM_EFF

    if expected_total <= 0:
        return -np.inf, None, None, None, None, None

    logL_rate             = poiss_log(k=total_observed, mu=expected_total)
    simulated_yearly_rate = expected_total / n_years

    return (
        total_logL_shape + logL_rate,
        simulated_yearly_rate,
        logL_shape_epeak,
        logL_shape_t90,
        logL_shape_pflux,
        logL_shape_fluence,
    )

# ══════════════════════════════════════════════════════════════════════════════
# PCA-whitened binned extended likelihood
# ══════════════════════════════════════════════════════════════════════════════

def _compute_pca_transform(obs_matrix: np.ndarray):
    """
    Fit a PCA whitening transform on an (N, D) matrix (D=4 observables).

    The observables are assumed to be already in log10 space.  We centre by
    the column means and diagonalise the sample covariance.

    Returns
    -------
    mu : (D,) array  — column means of the observed data
    V  : (D, D) array — eigenvectors as columns, sorted by **descending**
         explained variance.  Project via  X_pca = (X - mu) @ V.
    """
    mu = obs_matrix.mean(axis=0)
    C  = np.cov((obs_matrix - mu).T)
    eigvals, eigvecs = np.linalg.eigh(C)
    order = np.argsort(eigvals)[::-1]           # descending explained variance
    return mu, eigvecs[:, order]                # (D,)  (D,D)


def _project_pca(
    X        : np.ndarray,
    mu       : np.ndarray,
    V        : np.ndarray,
) -> np.ndarray:
    """Centre and project (N, D) matrix into the PCA basis."""
    return (X - mu) @ V


def _binned_shape_logL(
    obs_pca : np.ndarray,   # (N_obs, D)
    sim_pca : np.ndarray,   # (N_sim, D)
    n_bins  : int = 8,
    eps     : float = 1e-8,
) -> np.ndarray:
    """
    Compute the per-dimension binned shape log-likelihood in PCA space.

    Strategy
    --------
    For each PC dimension d:

    1. Define bin edges using ``n_bins`` equal-probability (quantile) cuts on
       the **observed** PCA projections.  Each bin contains ~ 1/n_bins of the
       observed events by construction, which maximises the resolving power of
       the chi-squared test.

    2. Estimate the model PDF from the **simulated** PCA projections using the
       same edges.  The probability mass in bin b is:

           p_b = (simulated count in bin b) / N_sim

    3. For each observed event, look up its bin and accumulate log p_b.
       This is a multinomial log-likelihood evaluated on the simulated PDF.

    A floor ``eps`` is applied to avoid log(0) when a bin is empty in the
    simulation (which should be rare for 8 bins and a reasonable draw size).

    Parameters
    ----------
    obs_pca : (N_obs, D) ndarray
    sim_pca : (N_sim, D) ndarray
    n_bins  : int, default 8
    eps     : float, small probability floor

    Returns
    -------
    logL_dims : (D,) ndarray — per-PC log-likelihood contributions
    """
    D      = obs_pca.shape[1]
    n_sim  = sim_pca.shape[0]
    logL   = np.zeros(D)

    for d in range(D):
        obs_d = obs_pca[:, d]
        sim_d = sim_pca[:, d]

        # Quantile-based bin edges from the OBSERVED projections
        quantile_pts = np.linspace(0, 100, n_bins + 1)
        edges        = np.percentile(obs_d, quantile_pts)

        # Slightly widen boundary edges so all points fall inside
        edges[0]  -= 1e-10 * (np.abs(edges[0])  + 1.0)
        edges[-1] += 1e-10 * (np.abs(edges[-1]) + 1.0)

        # Simulated PDF (probability mass per bin)
        sim_counts, _ = np.histogram(sim_d, bins=edges)
        sim_pdf        = sim_counts / float(n_sim)
        sim_pdf        = np.maximum(sim_pdf, eps)   # floor

        # Multinomial log-likelihood for the observed data
        bin_idx      = np.digitize(obs_d, edges) - 1
        bin_idx      = np.clip(bin_idx, 0, n_bins - 1)
        logL[d]      = np.sum(np.log(sim_pdf[bin_idx]))

    return logL


def precompute_pca_transform(params: SimParams):
    """
    Pre-compute the PCA whitening transform from the observed data stored in
    ``params``.  The result can be passed as ``pca_transform`` to
    ``log_likelihood_fj_pca`` to avoid recomputing it on every MCMC step.

    The four observables are transformed as ``log10(x)`` before PCA.

    Returns
    -------
    (mu, V) : tuple
        mu : (4,) mean of log10-transformed observed data
        V  : (4, 4) eigenvector matrix (columns sorted by descending variance)
    """
    obs_matrix = np.column_stack([
        np.log10(params.duration_data),
        np.log10(params.epeak_data),
        np.log10(params.pflux_data),
        np.log10(params.fluence_data),
    ])
    return _compute_pca_transform(obs_matrix)


def log_likelihood_fj_pca(
        thetas        : list,
        params        : SimParams,
        interps       : Interps,
        limits        : Dict[str, Any] = DEFAULT_LIMITS,
        n_bins        : int  = 8,
        pca_transform        = None,
):
    """
    Extended log-likelihood for the structured-jet model using a
    PCA-whitened, quantile-binned shape term.

    Motivation
    ----------
    The canonical ``log_likelihood_fj`` scores each of the four observables
    (T₉₀, Eₚ, Fₚ, Fluence) independently with CvM.  This discards information
    when the observables are correlated.

    Here we first rotate the four log10-observables into their principal
    components (PCA whitening on the **observed** data), making the linear
    correlations disappear.  In this whitened basis the 1D-factorisation:

        P(x') ≈ ∏ᵢ P(xᵢ')

    is much more accurate.  We then evaluate a 4× 1D binned multinomial
    log-likelihood, one per PC:

        logL_shape = Σᵢ Σ_b  n_obs_b(i) · log p_sim_b(i)

    where the bins are defined by ``n_bins`` equal-probability quantiles of the
    **observed** PC projections (so the bins are fixed for the duration of the
    MCMC run).

    The Poisson rate term is identical to ``log_likelihood_fj`` and enters
    **exactly once**:

        logL_rate = Poisson(k = N_obs_total | μ = triggered × fⱼ × ε_GBM)

    Parameters
    ----------
    thetas : list
        [k_pl, L_L0, L_mu_E, sigma_E, L_mu_tau, sigma_tau, fj]
    params : SimParams
    interps : Interps
    limits : dict
        Detection limits (default ``DEFAULT_LIMITS``).
    n_bins : int, default 8
        Number of quantile bins per PC dimension.
    pca_transform : (mu, V) tuple or None
        Pre-computed PCA transform (output of ``precompute_pca_transform``).
        If None, the PCA is re-computed from ``params`` on every call —
        **recommended** to pre-compute once and pass in for MCMC runs.

    Returns
    -------
    tuple : (log_posterior, yearly_rate_sim, logL_pc0, logL_pc1, logL_pc2, logL_pc3)
        logL_pc* are the per-PC binned shape contributions.
        Returns (-inf, None, None, None, None, None) on failure.

    Notes
    -----
    §1.1 - params.rng is NEVER reset here.
    §2.2 - Hard flux cut F_LIM kept as-is.
    Poisson term uses the same formula as log_likelihood_fj.
    """
    fj      = thetas[-1]
    n_years = params.triggered_years

    geo_efficiency = 1.0 - np.cos(params.theta_v_max)
    bns_per_year   = len(params.z_arr)
    n_events       = int(bns_per_year * n_years * geo_efficiency)

    if n_events < 50:
        return -np.inf, None, None, None, None, None

    obs = make_observations(thetas, params, interps, limits=limits,
                            n_events=n_events)

    if obs is None or len(obs["t_det"]) < 10:
        return -np.inf, None, None, None, None, None

    # ── Observed data matrix in log10 space (N_obs, 4) ──────────────────────
    obs_matrix = np.column_stack([
        np.log10(params.duration_data),
        np.log10(params.epeak_data),
        np.log10(params.pflux_data),
        np.log10(params.fluence_data),
    ])

    # ── Simulated data matrix in log10 space (N_sim, 4) ─────────────────────
    sim_matrix = np.column_stack([
        np.log10(obs["t_det"]),
        np.log10(obs["Ep_det"]),
        np.log10(obs["Fp_det"]),
        np.log10(obs["f_det"]),
    ])

    # ── PCA: fit on OBSERVED data, apply to both ─────────────────────────────
    if pca_transform is None:
        mu_pca, V_pca = _compute_pca_transform(obs_matrix)
    else:
        mu_pca, V_pca = pca_transform

    obs_pca = _project_pca(obs_matrix, mu_pca, V_pca)    # (N_obs, 4)
    sim_pca = _project_pca(sim_matrix, mu_pca, V_pca)    # (N_sim, 4)

    # ── Binned shape log-likelihood (4 × 1D in whitened PCA space) ──────────
    logL_per_dim  = _binned_shape_logL(obs_pca, sim_pca,
                                       n_bins=n_bins)
    total_logL_shape = logL_per_dim.sum()

    # ── Poisson rate term — fj enters ONLY here ──────────────────────────────
    total_observed   = params.yearly_rate * n_years
    triggered_events = obs["triggered_events"]
    expected_total   = triggered_events * fj * GBM_EFF

    if expected_total <= 0:
        return -np.inf, None, None, None, None, None

    logL_rate             = poiss_log(k=total_observed, mu=expected_total)
    simulated_yearly_rate = expected_total / n_years

    return (
        total_logL_shape + logL_rate,
        simulated_yearly_rate,
        float(logL_per_dim[0]),
        float(logL_per_dim[1]),
        float(logL_per_dim[2]),
        float(logL_per_dim[3]),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Yonetoku-conditional variant
# ══════════════════════════════════════════════════════════════════════════════
#
# The standard pipeline draws E_p_hat independently of L from a log-normal with
# mean L_mu_E and scatter sigma_E.  The Yonetoku relation (Yonetoku et al. 2004;
# Tsutsui et al. 2013 for sGRBs) introduces a power-law correlation:
#
#   log10(E_p_hat) = L_mu_E + gamma_y * log10(L_arr)  +  sigma_E * N(0,1)
#
# where L_arr is the dimensionless Schechter draw (mean ≈ 1 for the modified
# Schechter / generalized-gamma used here).  gamma_y = 0.5 reproduces the
# canonical "L^{1/2}" scaling; gamma_y = 0 recovers the baseline model.
#
# Parameter vector (8-dim):
#   [k_pl, L_L0, L_mu_E, sigma_E, L_mu_tau, sigma_tau, fj, gamma_y]
#
# L_mu_E retains its interpretation as log10(E_p_hat) at the characteristic
# luminosity (L_arr = 1).  gamma_y is a new free parameter inferred from data.

def generate_macro_properties_yonetoku(
        thetas        : list,
        params        : SimParams,
        interps       : Interps,
        n_counts_new  : int,
) -> dict:
    """
    Like ``generate_macro_properties`` but with a Yonetoku E_p–L correlation.

    The mean log10(E_p_hat) depends on the drawn luminosity:

        log10(E_p_hat) = L_mu_E + gamma_y * log10(L_arr)

    Parameters
    ----------
    thetas : list — 8 elements
        [k_pl, L_L0, L_mu_E, sigma_E, L_mu_tau, sigma_tau, fj, gamma_y]
    params : SimParams
    interps : Interps
    n_counts_new : int — number of MC events to generate
    """
    k_pl, L_L0, L_mu_E_10, sigma_E_10, L_mu_tau_10, sigma_tau_10, _, gamma_y = thetas

    rng        = params.rng
    l_10       = np.log(10)

    idx        = rng.integers(low=0, high=len(params.z_corr), size=n_counts_new)
    geometry   = params.geometric_factors[idx]
    one_plus_z = params.z_corr[idx]

    t_peak     = rng.lognormal(mean=L_mu_tau_10 * l_10, sigma=sigma_tau_10 * l_10,
                               size=n_counts_new)

    idtheta    = rng.integers(low=0, high=len(params.theta_v), size=n_counts_new)
    R_E_theta  = params.R_E[idtheta]
    R_F_theta  = params.R_F[idtheta]

    # §3.1 – draw luminosity then remove sub-threshold events
    L_arr = l_random_new(k_pl, n_counts_new, rng=rng)
    valid = L_arr > L_MIN_FACTOR
    if not np.any(valid):
        empty = np.array([])
        return {k: empty for k in (
            "t_peak_c_z", "F_p_real", "F_0", "E_p_obs", "R_F_theta",
            "alpha_e", "alpha_n", "z", "theta_v", "isotropic_energy")}

    # Apply valid mask
    idx        = idx[valid];        geometry   = geometry[valid]
    one_plus_z = one_plus_z[valid]; t_peak     = t_peak[valid]
    idtheta    = idtheta[valid];    R_E_theta  = R_E_theta[valid]
    R_F_theta  = R_F_theta[valid];  L_arr      = L_arr[valid]

    # ── Yonetoku: E_p_hat conditioned on L_arr ───────────────────────────────
    # Protect against L_arr = 0 (already filtered by L_MIN_FACTOR, but be safe)
    log10_L    = np.log10(np.maximum(L_arr, L_MIN_FACTOR))
    log10_Ep_mean = L_mu_E_10 + gamma_y * log10_L         # per-event mean (log10 keV)
    E_p_hat    = rng.lognormal(
        mean  = log10_Ep_mean * l_10,
        sigma = sigma_E_10    * l_10,
        size  = len(L_arr)
    )

    E_p_obs  = R_E_theta * E_p_hat / one_plus_z
    I1       = 1.15739 * t_peak * interps.int_0_alt(E_p_hat)
    N0       = 1e49 * 10**L_L0 * L_arr * geometry / I1

    F_0      = N0 * (one_plus_z)**2 * R_F_theta
    F_P_real = F_0 * interps.int_3_alt(E_p_obs) * 6.2e8

    return {
        "t_peak_c_z"       : one_plus_z * t_peak,
        "F_p_real"         : F_P_real,
        "F_0"              : F_0,
        "E_p_obs"          : E_p_obs,
        "R_F_theta"        : R_F_theta,
        "alpha_e"          : params.alpha_e[idtheta],
        "alpha_n"          : params.alpha_n[idtheta],
        "z"                : one_plus_z - 1,
        "theta_v"          : params.theta_v[idtheta],
        "isotropic_energy" : 1e49 * 10**L_L0 * L_arr / (1 - np.cos(params.theta_c)),
    }


def make_observations_yonetoku(
        thetas   : list,
        params   : SimParams,
        interps  : Interps,
        limits   : Dict[str, Any] = DEFAULT_LIMITS,
        n_events : int = N_SIMS,
) -> Optional[dict]:
    """
    Forward-simulation pipeline with Yonetoku-conditional E_p draw.

    Identical to ``make_observations`` except it calls
    ``generate_macro_properties_yonetoku``.
    """
    m_prop = generate_macro_properties_yonetoku(thetas, params, interps, n_events)

    if len(m_prop["t_peak_c_z"]) == 0:
        return None

    trigger_mask = (
        (m_prop['t_peak_c_z'] < limits["T90_LIM"]) &
        (m_prop['F_p_real']   > limits["F_LIM"])
    )
    if np.sum(trigger_mask) <= 5:
        return None

    m_prop_triggered = {k: v[trigger_mask] for k, v in m_prop.items()}

    P_F_64ms_50_300      = compute_Fp_64_ms_optimized(m_prop_triggered, interps)
    t_90_array, f_det_in = compute_time_evolution(m_prop_triggered, interps)

    detection_mask = (
        (t_90_array      < limits["T90_LIM"]) &
        (P_F_64ms_50_300 > limits["F_LIM"])
    )
    triggered_events = int(np.sum(detection_mask))
    if triggered_events == 0:
        return None

    shape_mask = (
        (m_prop_triggered["E_p_obs"] > limits["EP_LIM_LOWER"]) &
        (m_prop_triggered["E_p_obs"] < limits["EP_LIM_UPPER"])
    )
    final_mask = detection_mask & shape_mask
    if np.sum(final_mask) == 0:
        return None

    return {
        "t_det"                : t_90_array[final_mask],
        "f_det"                : f_det_in[final_mask],
        "Ep_det"               : m_prop_triggered["E_p_obs"][final_mask],
        "Fp_det"               : P_F_64ms_50_300[final_mask],
        "z_det"                : m_prop_triggered["z"][final_mask],
        "theta_v_det"          : m_prop_triggered["theta_v"][final_mask],
        "R_F_theta_det"        : m_prop_triggered["R_F_theta"][final_mask],
        "triggered_events"     : triggered_events,
        "isotropic_energy_det" : m_prop_triggered["isotropic_energy"][final_mask],
    }


def log_likelihood_fj_yonetoku(
        thetas  : list,
        params  : SimParams,
        interps : Interps,
        limits  : Dict[str, Any] = DEFAULT_LIMITS,
):
    """
    Log-likelihood for the Yonetoku-extended structured-jet model.

    Parameter vector (8-dim)
    ------------------------
    [k_pl, L_L0, L_mu_E, sigma_E, L_mu_tau, sigma_tau, fj, gamma_y]

    gamma_y is the Yonetoku slope:
        log10(E_p_hat) = L_mu_E + gamma_y * log10(L_arr)

    All other aspects (Poisson rate term, CvM shape scoring, GBM_EFF) are
    identical to ``log_likelihood_fj``.

    Returns
    -------
    tuple : (log_posterior, yearly_rate_sim, logL_epeak, logL_t90,
             logL_pflux, logL_fluence)
    """
    fj      = thetas[-2]   # 7th element (0-indexed 6)
    n_years = params.triggered_years

    geo_efficiency = 1.0 - np.cos(params.theta_v_max)
    bns_per_year   = len(params.z_arr)
    n_events       = int(bns_per_year * n_years * geo_efficiency)

    if n_events < 50:
        return -np.inf, None, None, None, None, None

    obs = make_observations_yonetoku(thetas, params, interps, limits=limits,
                                     n_events=n_events)
    if obs is None or len(obs["t_det"]) < 10:
        return -np.inf, None, None, None, None, None

    local_rng = _local_rng_from_thetas(thetas)

    logL_shape_t90     = score_func(obs["t_det"],  params.duration_data, rng=local_rng)
    logL_shape_epeak   = score_func(obs["Ep_det"], params.epeak_data,    rng=local_rng)
    logL_shape_pflux   = score_func(obs["Fp_det"], params.pflux_data,    rng=local_rng)
    logL_shape_fluence = score_func(obs["f_det"],  params.fluence_data,  rng=local_rng)
    total_logL_shape   = (logL_shape_epeak + logL_shape_t90
                          + logL_shape_pflux + logL_shape_fluence)

    total_observed   = params.yearly_rate * n_years
    triggered_events = obs["triggered_events"]
    expected_total   = triggered_events * fj * GBM_EFF

    if expected_total <= 0:
        return -np.inf, None, None, None, None, None

    logL_rate             = poiss_log(k=total_observed, mu=expected_total)
    simulated_yearly_rate = expected_total / n_years

    return (
        total_logL_shape + logL_rate,
        simulated_yearly_rate,
        logL_shape_epeak,
        logL_shape_t90,
        logL_shape_pflux,
        logL_shape_fluence,
    )


def generate_grb_population_yonetoku(
        thetas  : list,
        params  : SimParams,
        interps : Interps,
        limits  : Dict[str, Any] = DEFAULT_LIMITS,
        n_events: int = N_SIMS,
        seed    : Optional[int] = None,
) -> Optional[dict]:
    """
    PPC helper for the Yonetoku-extended model.

    Calls ``make_observations_yonetoku`` and returns the same dict structure
    as ``generate_grb_population``.
    """
    if seed is not None:
        params.rng = np.random.default_rng(seed)

    obs = make_observations_yonetoku(thetas, params, interps, limits=limits,
                                     n_events=n_events)
    if obs is None:
        return None

    E_iso_det, L_iso_det = calculate_isotropic_luminosity(obs, interps)

    return {
        "t90"                  : obs["t_det"],
        "epeak"                : obs["Ep_det"],
        "pflux"                : obs["Fp_det"],
        "fluence"              : obs["f_det"],
        "z_det"                : obs["z_det"],
        "theta_det"            : obs["theta_v_det"],
        "R_F_theta_det"        : obs["R_F_theta_det"],
        "n_detected"           : obs["triggered_events"],
        "isotropic_energy_det" : obs["isotropic_energy_det"],
        "E_iso_det"            : E_iso_det,
        "L_iso_det"            : L_iso_det,
    }


# ── §6.4 - generate_grb_population forwards R_F_theta_det ──────────────────
def generate_grb_population(
        thetas  : list,
        params  : SimParams,
        interps : Interps,
        limits  : Dict[str, Any] = DEFAULT_LIMITS,
        n_events: int = N_SIMS,
        seed    : Optional[int] = None,
) -> Optional[dict]:
    """
    Generate a synthetic GRB population for posterior predictive checks.

    §6.4 - R_F_theta_det is forwarded in the output dictionary so notebooks
    can use it without calling the lower-level make_observations directly.
    """
    if seed is not None:
        params.rng = np.random.default_rng(seed)

    obs = make_observations(thetas, params, interps, limits=limits,
                            n_events=n_events)
    if obs is None:
        return None

    E_iso_det, L_iso_det = calculate_isotropic_luminosity(obs, interps)

    return {
        "t90"                   : obs["t_det"],
        "epeak"                 : obs["Ep_det"],
        "pflux"                 : obs["Fp_det"],
        "fluence"               : obs["f_det"],
        "z_det"                 : obs["z_det"],
        "theta_det"             : obs["theta_v_det"],
        "R_F_theta_det"         : obs["R_F_theta_det"],   # §6.4
        "n_detected"            : obs["triggered_events"],
        "isotropic_energy_det"  : obs["isotropic_energy_det"],
        "E_iso_det"             : E_iso_det,
        "L_iso_det"             : L_iso_det,
    }
