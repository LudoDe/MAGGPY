"""
src.nsbh — Combined BNS + NSBH (BHNS) top-hat model (lognormal θ_c).

This sub-package extends the top-hat pipeline to include a
neutron-star–black-hole (NSBH / BHNSs) population alongside the
standard BNS population.  The key idea:

    detected sGRBs = BNS-origin GRBs + NSBH-origin GRBs

Both populations share the same intrinsic GRB physics (luminosity
function, spectral model) but differ in:

    1. Redshift distribution  (BNS vs BHNSs merger rate density)
    2. Geometric efficiency   (theta_c_med_bns vs theta_c_med_nsbh,
                               both via lognormal θ_c averaging)
    3. Jet fraction           (fj_bns free,  fj_nsbh = 0.5 fixed)

Parameter vector (7-dim):
    [A_index, L_L0, L_mu_E, sigma_E, theta_c_med_bns, theta_c_med_nsbh, fj_bns]
"""
