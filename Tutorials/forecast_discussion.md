# Interpreting the end-to-end forecast

## From a gamma-ray trigger to a useful multimessenger event

A joint detection is the beginning of a follow-up sequence. A wide-field prompt
monitor provides a time and a sky region; the GW network adds a distance and
inclination constraint; X-ray imaging can establish the position needed for
deep optical and radio observations. The scientific return therefore depends
on localization and response as well as sensitivity. The chapter tables report
both the annual joint rate and its fraction of the parent prompt sample, then
the smaller subsets passing GW sky-area and X-ray selections. This makes the
cost of each observational requirement visible.

The population here is generated with the retained MAGGPY chain and its fitted
jet fraction. GWFish is evaluated on the union of the prompt-detected events.
Consequently these tables address **prompt-triggered** follow-up. They cannot
measure the additional off-axis population discoverable through GW-triggered
HXI/SXI observations when no prompt instrument triggered. That requires a GW
calculation on the parent jet catalogue, with its own pointing/tiling selection.
The two channels should not be added without removing their overlap.

The prompt sample favours the bright jet core. Larger viewing angles generally
make the forward shock fainter early and shift its peak later, so an imager can
remain valuable even after the prompt signal has become difficult to detect.
The viewing-angle distributions and angle-coloured light curves show these
selection effects directly. Redshift and sky-area distributions expose a second
bottleneck: a GW detection can lie beyond the distance at which its location is
useful for prompt electromagnetic follow-up. These are the considerations behind
the distribution and response-delay comparisons in
[Ronchini et al. 2022](https://doi.org/10.1051/0004-6361/202243705).

## Comparison with GRINTA and THESEUS forecasts

The September 2026 GRINTA paper quotes approximately 65 TED sGRBs per year,
about 2 joint events per year with 3A-sharp, and about 50 with ET-2L. The exported
`publication_rate_comparison.csv` places the newly computed values beside these
benchmarks without rescaling the simulated population to force agreement.
Changes in sensitivities, duty, fitted population or detector assumptions should
be checked before interpreting a difference as a disagreement in physics.

The earlier [Rodi & Natalucci 2025](https://arxiv.org/abs/2504.04913) design
snapshot quotes approximately 90 sGRBs/yr and 60 or 85 joint events/yr with ET
or ET+CE. These numbers are useful history, but should not be combined with the
later forecast as if all were calculated with the same payload and population.
The [2025 Indico presentation](https://indico.ict.inaf.it/event/3361/contributions/25108/attachments/11440/23536/5%20Natalucci_GRINTA25-NAPOLI-29GEN.pdf)
explicitly attributes joint-rate estimates to the GSSI work.

The [THESEUS M5 Yellow Book](https://sci.esa.int/documents/34375/36249/Theseus_YB_final.pdf)
distinguishes hard-band triggering, soft-X-ray imaging and precise follow-up.
A lower native SXI flux threshold cannot be ranked directly against HXI's
photon-flux threshold: both the band and the source spectrum differ. The code
integrates the same source through each band and uses its native flux unit.
Nor should a published SXI discovery yield be compared directly with this
prompt-selected sample. Its fraction is conditional on a different parent
population and observational strategy. Crystal Eye is a prompt comparison in
this workflow, not an additional X-ray imager.

## Why the X-ray light curves change

The previous forward-shock calculation supplied the fitted radiated prompt
energy directly as the kinetic energy, used a 5-degree top-hat jet and drew
magnetic fractions between 1e-4 and 1e-2. Those choices remain available as
the `legacy` microphysics preset with `jet_structure='tophat'`.

The literature comparison instead uses the explicitly named `FS_SGRB` or
`FS_GW17` presets. For the former, density is uniform in 0.003–0.015 cm^-3 and
epsilon_B is uniform in 0.01–0.1. The latter fixes density at 2.5e-4 cm^-3 and
epsilon_B at 1.3e-4. Both use p=2.2, epsilon_e=0.1 and Gamma0=500, with
Erad/Ekin drawn uniformly from 0.01–0.1. This ratio is named explicitly because
the efficiency notation in the prose of Ronchini Sec. 2.5 is ambiguous. The code
uses Ekin = Erad/(Erad/Ekin), not an unlabelled multiplicative brightening.
The new notebook selects a 3.4-degree power-law core. This is VegasAfterglow's
power-law profile and is not asserted to be identical to the paper's profile.

[Fong et al. 2015](https://arxiv.org/abs/1509.02922) provide an observational
context for short-GRB energies, densities and microphysics, rather than a
universal parameter set. The paper's broad-band modelling demonstrates why
uncertainty in these quantities must be examined before interpreting an X-ray
envelope. A GRB170817A-like low-density prescription is a useful comparison,
but should not silently stand in for every classical short burst.

## Comparing the bright end without forcing a match

`xray_light_curves.csv` contains a deterministic angle-stratified display sample
from the actual forecast and reuses its source parameters. Its flux panels use
observer time and instrument bands. A second plot reports source time and
nu Lnu at **rest-frame 5 keV**. The transformation uses t_obs=(1+z)t_src,
nu_obs=nu_rest/(1+z), and nu_rest Lnu_rest=4 pi DL^2 nu_obs Fnu_obs. A band flux
in erg cm^-2 s^-1 must not be relabelled as this monochromatic luminosity.
The display sample is not a draw from the parameter posterior and its maximum
is not a population confidence limit.

`validate_afterglow_models.py` additionally compares VegasAfterglow 2.0.4 with
[afterglowpy / Ryan et al. 2020](https://arxiv.org/abs/1909.11691) for matched
top-hat forward shocks. It converts afterglowpy's mJy to CGS and exports solver
ratios, rather than assuming agreement. A separate controlled figure changes
the microphysics/energy prescription while holding geometry fixed. These are
model checks, not ten-year catalogue results. Solver dynamics and the early
coasting phase can differ even when input parameters match.

The committed controlled check finds a median VegasAfterglow/afterglowpy ratio
of approximately 4.26 on-axis between 10^3 and 10^5 source-frame seconds. It does
**not** establish solver normalization agreement. The very faint early off-axis
regime has even larger relative differences; full curves and the stated comparison
selection are retained in [the validation data](validation/README.md). Rest-frame
redshift invariance is checked separately. The notebook keeps the existing 2.0.4
VegasAfterglow pin for reproducibility; its
[upstream changelog](https://github.com/YihanWangAstro/VegasAfterglow/blob/main/CHANGELOG.md)
documents later radiation/dynamics changes, so changing versions requires a fresh
comparison rather than an assumed correction factor.

The upper light curves in Ronchini Fig. 1 include **high-latitude prompt emission
and forward shock**; the new diagnostics show forward shock alone. Early steep
decays and plateaux, energy injection, absorption and the exact jet profile can
therefore prevent an early-time match. Likewise the Swift/XRT curves extrapolated
to 5–30 keV in [GRINTA 2026 Fig. 1](https://arxiv.org/abs/2609.02802v2) are an
empirically selected on-axis sample with an XRT detection within 300 s. Their
approximately 50% HXI detectability is not the fraction of every TED or GW event
with an afterglow. The upper envelope should be compared in the same band,
time frame and selection; it should not be made to agree by an extra flux factor.

## Statistics and practical limitations

Ten simulated years improve count resolution and reduce Monte Carlo noise at
fixed model parameters. They do not constrain population or payload systematics,
and do not imply a ten-year mission. The joint fractions, finite-sample intervals,
localization cuts and imager response delays should accompany the rate table in
the thesis. Detector duty and optimistic arranged follow-up are explicit inputs.
Sensitivity changes and the two microphysics presets should be reported as
separate scenarios, rather than pooled into an unexplained error bar.

The repository does not contain the user's retained `emcee.h5` or the saved
event catalogues. The revised notebook therefore clears the old four-year
outputs and records provenance when a new run is made. No four-year result is
relabelled as a ten-year forecast. A full chapter run still requires that chain,
the fitted MRD/jet structure, GWFish/LALSuite and the existing detector PSDs.
