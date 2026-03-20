# Bias in Recovered Isotropic Quantities

## Overview

The simulation pipeline computes isotropic-equivalent quantities in two distinct ways:

1. **True** — computed directly from the intrinsic model parameters during event generation (`generate_macro_properties`).
2. **Recovered** — computed from the simulated observables (fluence, $T_{90}$, redshift) via `calculate_isotropic_luminosity`, mimicking what an observer would do with real data.

Even for noise-free simulations, these two values are **not expected to agree in general**. This document enumerates and quantifies the sources of this discrepancy.

---

## Definitions

### True isotropic energy

Set at generation time in `generate_macro_properties`:

$$
E_{\rm iso,true} = \frac{10^{49} \cdot 10^{L_{L_0}} \cdot L}{1 - \cos\theta_c}
\quad [\text{erg}]
$$

where $L$ is the dimensionless luminosity drawn from the modified Schechter distribution, $10^{49} \cdot 10^{L_{L_0}}$ is the energy scale, and $1-\cos\theta_c$ is the fractional solid angle of the jet core. This quantity represents the energy the burst **would have** if its jet energy were emitted isotropically into $4\pi$ sr.

### Recovered isotropic energy

Computed from observables in `calculate_isotropic_luminosity`:

$$
E_{\rm iso,rec} = \frac{4\pi D_L^2}{1+z} \, S_{\rm obs} \, k_{\rm corr}
$$

where $S_{\rm obs}$ is the observed fluence in the detector band (50–300 keV, observer frame), $D_L$ is the luminosity distance, and the k-correction is

$$
k_{\rm corr} = \frac{\displaystyle\int_{50}^{300} E\, N(E, E_{p,\rm rest})\,dE}{\displaystyle\int_{10}^{1000} E\, N(E, E_{p,\rm obs})\,dE}
$$

with $N(E, E_p)$ the broken-power-law spectral model and $E_{p,\rm rest} = E_{p,\rm obs}(1+z)$.

The recovered isotropic luminosity is then

$$
L_{\rm iso,rec} = \frac{E_{\rm iso,rec}}{T_{90,\rm rest}} = \frac{E_{\rm iso,rec}(1+z)}{T_{90,\rm obs}}
$$

---

## Sources of Discrepancy

### 1. Jet structure and viewing angle (dominant effect)

This is the **primary and most fundamental** source of discrepancy.

The intrinsic flux normalization includes a viewing-angle-dependent factor $R_F(\theta_v)$:

$$
F_0 = \frac{10^{49} \cdot 10^{L_{L_0}} \cdot L}{4\pi D_L^2 (1-\cos\theta_c) \, I_1} \cdot (1+z)^2 \cdot R_F(\theta_v)
$$

where $I_1 \propto t_{\rm peak} \int_1^{10^4} E\, N(E, E_{p,\rm rest})\,dE$ is the spectral-temporal normalization integral and $R_F(\theta_v)$ is the ratio of the observed flux at viewing angle $\theta_v$ to the on-axis flux.

When the recovered estimator is applied to the simulated fluence, the result contains $R_F(\theta_v)$ as a multiplicative factor:

$$
E_{\rm iso,rec} \propto E_{\rm iso,true} \cdot R_F(\theta_v) \cdot \Bigl[\text{band correction}\Bigr]
$$

**Consequences:**
- For **on-axis** GRBs ($\theta_v \approx 0$): $R_F \approx 1$, so the two quantities agree (up to band effects).
- For **off-axis** GRBs ($\theta_v > \theta_c$): $R_F \ll 1$, so $E_{\rm iso,rec} \ll E_{\rm iso,true}$.

This is not a numerical artifact. It reflects a physical truth: a real observer who measures the fluence of an off-axis GRB and applies the standard isotropic estimator **will underestimate its intrinsic energy**, because they observe only the dimmed, off-axis emission. An observer cannot correct for this without independent knowledge of the jet structure and $\theta_v$.

---

### 2. Spectral energy band mismatch (k-correction approximation)

The true $E_{\rm iso,true}$ is normalized using the **near-bolometric** integral $I_1 = \int_1^{10^4 \, \rm keV} E\, N(E,E_p)\,dE$, which captures essentially the full spectral power.

The recovered quantity uses the fluence in the **50–300 keV observer-frame band**, corrected by $k_{\rm corr}$ to the **50–300 keV rest-frame band** (not the full bolometric range). The k-correction numerator and denominator use slightly different bands (50–300 keV rest vs 10–1000 keV observer), but neither reaches the bolometric range of 1 keV to 10 MeV.

As a result, the recovered quantity systematically falls short of the true bolometric energy, by a factor equal to the fraction of the bolometric emission captured in the 50–300 keV rest-frame band:

$$
\frac{E_{\rm iso,rec}}{E_{\rm iso,true}} \sim R_F(\theta_v) \cdot \frac{\int_{50}^{300} E\, N(E, E_{p,\rm rest})\,dE}{\int_1^{10^4} E\, N(E, E_{p,\rm rest})\,dE}
$$

This ratio is $<1$ for all typical GRB spectra and depends on $E_{p,\rm rest}$:

- When $E_{p,\rm rest}$ is well within the range [50, 300] keV, the ratio is close to 1.
- When $E_{p,\rm rest}$ is outside this range (soft or very hard spectra), significant spectral power falls outside the band and the ratio is much less than 1.

---

### 3. K-correction band inconsistency

The k-correction as implemented is:

$$
k_{\rm corr} = \frac{\int_{50}^{300} E\, N(E, E_{p,\rm rest})\,dE}{\int_{10}^{1000} E\, N(E, E_{p,\rm obs})\,dE}
$$

while the observed fluence $S_{\rm obs}$ is computed in the 50–300 keV observer-frame band. A self-consistent k-correction would use the **same band in the denominator** as the one in which fluence is measured. The mismatch between the denominator band (10–1000 keV) and the fluence band (50–300 keV) introduces an error of order

$$
\frac{\int_{50}^{300} E\, N(E, E_{p,\rm obs})\,dE}{\int_{10}^{1000} E\, N(E, E_{p,\rm obs})\,dE}
$$

which is again $E_p$-dependent. This can either over- or under-correct depending on where $E_{p,\rm obs}$ falls.

---

### 4. Duration estimator: $T_{90}$ vs $t_{\rm peak}$

The recovered luminosity is

$$
L_{\rm iso,rec} = \frac{E_{\rm iso,rec}}{T_{90,\rm rest}}
$$

The intrinsic duration scale in the model is $t_{\rm peak}$ (the pulse rise time). $T_{90}$ is a specific observational quantity — the interval containing 90 % of the fluence — computed in the 50–300 keV band with a fixed 16 ms binning, and it scales roughly as $T_{90} \approx \mathcal{O}(1)\times t_{\rm peak}$ but with a proportionality constant that depends on the spectral index $\alpha_n$ and the peak energy $E_{p,\rm obs}$.

Because $T_{90}$ is measured in a limited band, it is also spectral-index-biased: bursts with rapidly evolving spectra will have their measured $T_{90}$ shortened. This introduces scatter in $L_{\rm iso,rec}$ that is absent by construction in $E_{\rm iso,true}$.

---

### 5. Selection effects (population-level bias)

Even if the per-event estimator were unbiased, the **detected sample** is flux-limited and angle-limited. The selection functions $R_F(\theta_v) > F_{\rm lim}$ and $T_{90} < T_{\rm lim}$ preferentially select:

- **On-axis** or **mildly off-axis** events.
- **Short, bright** events.

This means the distribution of $E_{\rm iso,rec}$ over detected events is biased high, while the distribution of $E_{\rm iso,true}$ over all generated events would be lower. This is a **classical Malmquist bias** for an isotropic-luminosity-limited survey.

---

## Summary Table

| Source | Direction | Magnitude | Depends on |
|--------|-----------|-----------|------------|
| Viewing angle $R_F(\theta_v)$ | Underestimate at $\theta_v > 0$ | Can be orders of magnitude | $\theta_v$, jet structure model |
| Band-limited k-correction (not bolometric) | Underestimate | Factor $\sim 0.3$–$0.9$ | $E_{p,\rm rest}$ |
| K-correction band inconsistency | Either sign | Factor $\sim 0.8$–$1.0$ | $E_{p,\rm obs}$ |
| $T_{90}$ vs $t_{\rm peak}$ | Scatter | Factor $\sim$ few | $\alpha_n$, $E_{p,\rm obs}$ |
| Malmquist / selection bias (population) | Upward bias in sample mean | Significant | Survey depth, jet model |

---

## Implication for the MCMC

In the likelihood, the simulation uses `isotropic_energy_det` (the **true** quantity) passed directly from the macro-properties array for the rate computation, rather than the recovered $E_{\rm iso,rec}$ from `calculate_isotropic_luminosity`. This is self-consistent.

`calculate_isotropic_luminosity` / `make_observations_with_iso` are intended for **posterior predictive checks**: generating and plotting simulated observational counterparts of $E_{\rm iso}$ and $L_{\rm iso}$ as an observer would compute them, to be compared with catalogue values that are derived the same way. In this context, the biases described above are **features, not bugs** — they must be present in both the data and the simulation for a fair comparison.

---

*Reference: Poolakkil et al. 2021 (arXiv:2103.13528) for the k-correction formula.*
