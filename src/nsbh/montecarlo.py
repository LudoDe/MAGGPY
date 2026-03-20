"""
src/nsbh/montecarlo.py
======================
Combined BNS + NSBH Monte Carlo likelihood for the top-hat model
with lognormal θ_c geometric efficiency.

The observed sGRB catalogue is modelled as:

    detected sGRBs  =  (BNS-origin)  +  (NSBH-origin)

Both components share the same intrinsic GRB physics parameters:
    [A_index, L_L0, L_mu_E, sigma_E]

Population-specific parameters:
    theta_c_med_bns     Median θ_c for BNS (degrees, MCMC)
    theta_c_med_nsbh    Median θ_c for NSBH (degrees, MCMC)
    fj_bns              BNS jet-launching fraction (MCMC)
    fj_nsbh             NSBH jet-launching fraction (FIXED = 0.5)

Parameter vector (7-dim):
    [A_index, L_L0, L_mu_E, sigma_E, theta_c_med_bns, theta_c_med_nsbh, fj_bns]
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, Optional

from ..montecarlo import SimParams
from ..top_hat.montecarlo import (
    simplified_montecarlo,
    apply_detection_cuts,
    score_func_cvm,
    poiss_log,
    compute_luminosity_distance,
    create_k_interpolator,
    create_geometric_efficiency_lognormal_interpolator,
)
from .init import NSBHData


# ── Constants ───────────────────────────────────────────────────────────────
GBM_EFF        = 0.6       # GBM sky fraction
FJ_NSBH_FIXED  = 1       # Fixed NSBH jet-launching fraction (prefactor)
N_MC_EVENTS    = 10_000     # MC draw size per population

# ══════════════════════════════════════════════════════════════════════════════
# Combined log-likelihood (top-hat, lognormal θ_c)
# ══════════════════════════════════════════════════════════════════════════════
def log_likelihood_combined_tophat_ln(
    thetas,
    bns_params,
    nsbh_data,
    bns_distances,
    nsbh_distances,
    k_interpolator,
    geom_eff_interp,
    n_events: int = N_MC_EVENTS,
):
    """
    Joint BNS + NSBH log-likelihood for the top-hat model (lognormal θ_c).

    Parameter vector (7-dim)
    ------------------------
    thetas = [A_index, L_L0, L_mu_E, sigma_E,
              theta_c_med_bns, theta_c_med_nsbh, fj_bns]

    Returns
    -------
    tuple : (logL_total, logL_pflux, logL_epeak, logL_poisson, physics_eff)
    """
    A_index, L_L0, L_mu_E, sigma_E, theta_c_med_bns, theta_c_med_nsbh, fj_bns = thetas

    grb_thetas = [A_index, L_L0, L_mu_E, sigma_E]
    triggered_years = bns_params.triggered_years

    # ── Geometric efficiencies ──────────────────────────────────────────────
    geom_eff_bns  = geom_eff_interp(theta_c_med_bns)
    geom_eff_nsbh = geom_eff_interp(theta_c_med_nsbh)

    epsilon_bns  = geom_eff_bns  * fj_bns
    epsilon_nsbh = geom_eff_nsbh * FJ_NSBH_FIXED

    # ── Expected intrinsic rates (events/year × GBM efficiency) ─────────────
    intrinsic_bns  = epsilon_bns  * len(bns_params.z_arr) * GBM_EFF
    intrinsic_nsbh = epsilon_nsbh * len(nsbh_data.z_arr)  * GBM_EFF

    n_years = triggered_years

    # ── BNS Monte Carlo ────────────────────────────────────────────────────
    bns_results = simplified_montecarlo(
        grb_thetas, n_events, bns_params, bns_distances, k_interpolator,
    )
    bns_trig, bns_analysis = apply_detection_cuts(
        bns_results["p_flux"], bns_results["E_p_obs"],
    )

    # ── NSBH Monte Carlo ───────────────────────────────────────────────────
    nsbh_results = simplified_montecarlo(
        grb_thetas, n_events, nsbh_data, nsbh_distances, k_interpolator,
        rng=bns_params.rng,
    )
    nsbh_trig, nsbh_analysis = apply_detection_cuts(
        nsbh_results["p_flux"], nsbh_results["E_p_obs"],
    )

    # ── Physics efficiencies (triggered / drawn) ───────────────────────────
    n_bns_trig  = np.sum(bns_trig)
    n_nsbh_trig = np.sum(nsbh_trig)

    phys_eff_bns  = n_bns_trig  / n_events 
    phys_eff_nsbh = n_nsbh_trig / n_events 

    # ── Merge analysis-passing detections ──────────────────────────────────
    pflux_det = np.concatenate([
        bns_results["p_flux"][bns_analysis],
        nsbh_results["p_flux"][nsbh_analysis],
    ])
    epeak_det = np.concatenate([
        bns_results["E_p_obs"][bns_analysis],
        nsbh_results["E_p_obs"][nsbh_analysis],
    ])

    if len(pflux_det) <= 3:
        return -np.inf, -np.inf, -np.inf, -np.inf, 0.0

    # ── Shape likelihood (CvM on merged sample vs catalogue) ───────────────
    logL_pflux = score_func_cvm(pflux_det, bns_params.pflux_data, bns_params.rng)
    logL_epeak = score_func_cvm(epeak_det, bns_params.epeak_data, bns_params.rng)

    # ── Poisson rate likelihood ─────────────────────────────────────────────
    predicted_bns  = intrinsic_bns  * n_years * phys_eff_bns
    predicted_nsbh = intrinsic_nsbh * n_years * phys_eff_nsbh
    predicted_total = predicted_bns + predicted_nsbh

    observed_total = bns_params.yearly_rate * triggered_years

    if predicted_total <= 0:
        return -np.inf, -np.inf, -np.inf, -np.inf, 0.0

    logL_poisson = poiss_log(k=observed_total, mu=predicted_total)

    phys_eff_combined = (phys_eff_bns + phys_eff_nsbh) / 2

    return (
        logL_pflux + logL_epeak + logL_poisson,
        logL_pflux,
        logL_epeak,
        logL_poisson,
        phys_eff_combined,
    )


# ══════════════════════════════════════════════════════════════════════════════
# emcee wrapper
# ══════════════════════════════════════════════════════════════════════════════
def create_log_probability_combined(
    log_prior_func,
    bns_params,
    nsbh_data,
    k_interpolator,
    geom_eff_interp,
    n_events: int = N_MC_EVENTS,
):
    """
    Return a callable ``log_probability(thetas)`` for emcee.

    The returned function signature matches emcee's requirements:
        log_probability(thetas) → (log_prob, *blobs)
    """
    bns_distances  = compute_luminosity_distance(bns_params.z_arr)
    nsbh_distances = nsbh_data.distances

    def log_probability(thetas):
        lp = log_prior_func(thetas)
        if not np.isfinite(lp):
            return -np.inf, 0.0, 0.0, 0.0, 0.0

        ll = log_likelihood_combined_tophat_ln(
            thetas, bns_params, nsbh_data,
            bns_distances, nsbh_distances,
            k_interpolator, geom_eff_interp,
            n_events=n_events,
        )
        if not np.isfinite(ll[0]):
            return -np.inf, 0.0, 0.0, 0.0, 0.0

        return lp + ll[0], ll[1], ll[2], ll[3], ll[4]

    return log_probability


# ══════════════════════════════════════════════════════════════════════════════
# PPC population generator
# ══════════════════════════════════════════════════════════════════════════════
def generate_combined_population_tophat(
    thetas,
    bns_params,
    nsbh_data,
    k_interpolator,
    geom_eff_interp,
    n_events: int = N_MC_EVENTS,
    seed: Optional[int] = None,
) -> Optional[dict]:
    """
    Generate a combined BNS + NSBH synthetic GRB population for PPCs.

    Parameter vector (7-dim):
        [A_index, L_L0, L_mu_E, sigma_E,
         theta_c_med_bns, theta_c_med_nsbh, fj_bns]

    Returns a dict with observable arrays plus ``origin`` (0=BNS, 1=NSBH).
    """
    A_index, L_L0, L_mu_E, sigma_E, theta_c_med_bns, theta_c_med_nsbh, fj_bns = thetas
    grb_thetas = [A_index, L_L0, L_mu_E, sigma_E]

    if seed is not None:
        bns_params.rng = np.random.default_rng(seed)

    bns_distances  = compute_luminosity_distance(bns_params.z_arr)
    nsbh_distances = nsbh_data.distances

    # ── BNS ─────────────────────────────────────────────────────────────────
    bns_results = simplified_montecarlo(
        grb_thetas, n_events, bns_params, bns_distances, k_interpolator,
    )
    _, bns_analysis = apply_detection_cuts(
        bns_results["p_flux"], bns_results["E_p_obs"],
    )

    # ── NSBH ────────────────────────────────────────────────────────────────
    nsbh_results = simplified_montecarlo(
        grb_thetas, n_events, nsbh_data, nsbh_distances, k_interpolator,
        rng=bns_params.rng,
    )
    _, nsbh_analysis = apply_detection_cuts(
        nsbh_results["p_flux"], nsbh_results["E_p_obs"],
    )

    n_bns  = int(np.sum(bns_analysis))
    n_nsbh = int(np.sum(nsbh_analysis))

    if n_bns + n_nsbh == 0:
        return None

    pflux = np.concatenate([
        bns_results["p_flux"][bns_analysis],
        nsbh_results["p_flux"][nsbh_analysis],
    ])
    epeak = np.concatenate([
        bns_results["E_p_obs"][bns_analysis],
        nsbh_results["E_p_obs"][nsbh_analysis],
    ])
    z_det = np.concatenate([
        bns_results["z_arr"][bns_analysis],
        nsbh_results["z_arr"][nsbh_analysis],
    ])
    origin = np.concatenate([
        np.zeros(n_bns, dtype=int),
        np.ones(n_nsbh, dtype=int),
    ])

    return {
        "pflux"      : pflux,
        "epeak"      : epeak,
        "z_det"      : z_det,
        "origin"     : origin,
        "n_detected" : n_bns + n_nsbh,
        "n_bns_det"  : n_bns,
        "n_nsbh_det" : n_nsbh,
    }
