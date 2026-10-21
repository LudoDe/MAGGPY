# Reference assumptions for Tutorial 6

The ten years are **Monte Carlo exposure**, not a forecast of any mission's
funded lifetime. Every rate is the selected count divided by that exposure.
The fitted jet fraction and viewing-angle solid angle enter catalogue generation;
the absolute EM observing efficiency enters the prompt mask once. Do not apply
these factors again to the final rates.

## Instrument audit

| Instrument | Native reference used | Tutorial convention | Evidence and qualification |
| --- | --- | --- | --- |
| GRINTA / TED | 0.3 ph cm^-2 s^-1, 50–300 keV, 1 s, 5 sigma | Effective sky efficiency = 0.5 × 8/(4 pi) | Retained **author forecast assumptions**. The 2025 Indico slide 8 and 2026 paper Table 2 specify approximately 8 sr and a threshold below 0.5 in this band, but do not establish all these reference-time, significance and duty factors. |
| Crystal Eye | 0.304 ph cm^-2 s^-1, 50–300 keV, 5 sigma | 2 s and effective sky efficiency 0.3 remain adopted assumptions | Barbato, ASAPP 2025, slide 11. The 0.766 threshold belongs to **10–1000 keV**. The lower 0.180 value is 3 sigma in 50–300 keV. The distinct 2.19 ph cm^-2 s^-1 efficiency midpoint is not a substitute for a fixed-significance sensitivity. |
| THESEUS / XGIS | 3e-8 erg cm^-2 s^-1, 30–150 keV, 1 s, 3 sigma | Rescaled to 5e-8 at 5 sigma; 2/(4 pi) × 0.65 effective coverage | M5 Yellow Book, printed p. 48, Table 3-1. The 0.65 duty/visibility factor is adopted; 2 sr is the published FoV. |
| GRINTA / HXI | 1.2e-3 ph cm^-2 s^-1, 5–30 keV, 10^4 s, 3 sigma | Multiply by 5/3; background-limited exposure scaling | Natalucci et al. 2026 Table 2; also Indico 2025 slide 8. The integration band is narrower than HXI's full 5–200 keV range. |
| THESEUS / SXI | 1e-10 and 1.8e-11 erg cm^-2 s^-1, 0.3–5 keV, 100 and 1500 s, 3 sigma | Multiply both by 5/3 | Yellow Book Table 3-1. Requirements assume photon index 1.8 and absorbing column 5e20 cm^-2. |

All significance rescaling is an explicit Gaussian, background-limited
approximation, not a trigger-response simulation. HXI exposure scaling outside
its reference exposure is also approximate. The SXI calculation integrates the
synchrotron spectrum but does not fold a response and absorption model through
the instrument. Its detectability is therefore spectrum dependent.

The imager follow-up fraction of one is **conditional on obtaining a pointing**.
It does not mean that HXI or SXI observes the entire sky, nor that every wide-area
TED localization can be covered completely. The 0, 60, 100, 300 and 3600 s delays
span idealized and delayed response cases; zero is an ideal limit. A specific
slew angle, ground-alert delay, tiling strategy and exposure budget would be
needed for a fully operational forecast.
The implementation measures each delay from the modelled prompt peak
(`ready_s = t_peak_s + delay_s`), not directly from merger time.

## GW networks and annual rate tables

The notebook uses 0.85 independent duty per detector identifier. A single
operational state is reused wherever the detector occurs in several networks.
SNR and Fisher localization use the **same active subset**. With duty set to one,
the original all-operational calculation is recovered. A scalar duty factor
applied to completed network rates would discard viable partial networks.
The triangular ET is treated as one jointly operational observatory, not three
independently sampled Michelsons. Detector outages are idealized and uncorrelated.

The ET-2L + CE40 comparison is now available alongside ET-2L, triangular ET and
3A-sharp. The CE40 site and PSD are the repository's existing configuration;
its availability is a scenario, not an assertion of an approved construction plan.
The former `LIO_sharp` entry had longitude -77 degrees, which places it in the
wrong hemisphere. Its coordinates now use the Aundha site at 19 degrees
36 minutes 50 seconds N, 77 degrees 01 minute 54 seconds E from
[Nuttall's April 2026 Indico presentation, slide 8](https://indico.global/event/16271/contributions/148740/attachments/68774/133197/Edinburgh_0426.pdf).
The arm azimuth remains the repository's adopted 126-degree scenario; it has
not been re-certified against final engineering drawings. Old localization
numbers must be rerun after the site correction.

| Output | Numerator | Denominator |
| --- | --- | --- |
| Joint / yr | Prompt and GW detections | Simulated years |
| Joint fraction of prompt | Prompt and GW detections | Prompt detections of that instrument |
| Localized fraction of joint | Joint events within the selected positive, finite GW sky area | Prompt and GW detections |
| Afterglow fraction of prompt | Prompt and imager detections at the stated delay | Prompt detections of that instrument |
| Afterglow fraction of joint prompt | Prompt, GW and imager detections | Prompt and GW detections |
| Localized fraction of joint afterglow | Localized prompt, GW and imager detections | Prompt, GW and imager detections |

CSV fractions lie in [0,1]; an empty denominator is NaN. The 68.27% Wilson
intervals characterize counting uncertainty conditional on this model. They are
not posterior population uncertainty. Ten years give 0.1 yr^-1 count resolution;
small localized samples can still have large fractional uncertainty. A zero
count is not proof of a zero astrophysical rate.

## Source references

1. [THESEUS M5 Yellow Book, ESA/SCI(2020)2](https://sci.esa.int/documents/34375/36249/Theseus_YB_final.pdf), Table 3-1, printed p. 48 (PDF p. 50). The [mission site](https://theseus.astro-ge.ch/) still identifies this document as the **M5** Yellow Book; it is not a newly verified M7 design baseline.
2. [Natalucci et al. 2026, GRINTA: a new Probe of the Dynamic Universe](https://arxiv.org/abs/2609.02802v2), Sec. 2.1 and Table 2. Includes De Santis and Ronchini among the authors.
3. [Rodi & Natalucci 2025, The GRINTA hard X-ray mission](https://arxiv.org/abs/2504.04913), an earlier design/forecast snapshot.
4. [Natalucci, Indico 2025, Probing the dynamic universe with GRINTA](https://indico.ict.inaf.it/event/3361/contributions/25108/attachments/11440/23536/5%20Natalucci_GRINTA25-NAPOLI-29GEN.pdf), slide 8 for payload sensitivity and slide 24 for the GSSI forecast attribution.
5. [Barbato, ASAPP 2025, Crystal Eye: monitoring the X and gamma ray sky](https://indico.cern.ch/event/1463191/contributions/6434158/attachments/3067124/5425559/CrystalEyeASAPP.pdf), slide 11. These are preliminary performance estimates.
6. [Barbato, ASAPP 2023](https://indico.cern.ch/event/1208314/contributions/5342877/attachments/2670338/4629757/Barbato_ASAPP2023.pdf), earlier Crystal Eye design context.
