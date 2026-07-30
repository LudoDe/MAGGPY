<p align="center">
  <img src="LOGO.png" alt="MAGGPY logo" width="400"/>
</p>

<h1 align="center">MAGGPY</h1>
<p align="center"><b>Multimessenger Astronomy for GRBs and Gravitational waves in PYthon</b></p>

---

## Overview

**MAGGPY** is a MCMC framework for simulating and fitting **short gamma-ray burst (sGRB) populations** against Fermi/GBM catalogue data. It forward-models GRB jet emission, both **top-hat** and **structured jet** profiles, to infer population-level parameters such as the jet fraction $f_j$, opening-angle distributions, and luminosity functions. It also supports joint **GW + EM detection predictions** using [GWFish](https://github.com/janosch314/GWFish).

### Key capabilities

- Forward Monte Carlo simulation of sGRB observables (peak flux, T90, fluence, peak energy)
- MCMC inference with [`emcee`](https://emcee.readthedocs.io/) using a Cramér–von Mises goodness-of-fit likelihood
- Top-hat and structured jet angular profiles
- Merger rate density models from population synthesis (multiple $\alpha_{\rm CE}$ values, multiple channels)
- GW detection efficiency and sky-localisation forecasts via GWFish (Einstein Telescope, Cosmic Explorer, LIGO)
- Posterior predictive checks with CDF comparisons

---

## Installation

I really reccomend you create a new environment as the multiple libraries that are being juggled in this code don't play well with too old or too new versions of python
```bash
conda create -n acme_env python=3.10 -y
conda activate acme_env
```

You can then easily install all relevant libraries or by running `Tutorials_ACME/setup.ipynb`.

```bash
git clone https://github.com/LudoDe/MAGGPY.git
cd MAGGPY
pip install -r requirements.txt
```

### Key dependencies

| Package | Role |
|---|---|
| `emcee` | Affine-invariant MCMC sampler |
| `astropy` | Cosmology (Planck18), units |
| `GWFish` | GW Fisher-matrix detector simulation |
| `astro-gdt-fermi` | Fermi Gamma-ray Data Tools |
| `corner` | Posterior corner plots |
| `h5py` | HDF5 chain storage |
| `healpy` | HEALPix sky maps |

---

## Tutorials (ACME)

| # | Notebook | Description |
|---|---|---|
| 0 | `tutorial0_data_preparation` | Load & filter the Fermi/GBM catalogue; prepare MRD redshift samples |
| 1 | `tutorial1_tophat` | Top-hat jet model: prior setup, short MCMC, CDF visualisation, $f_j$ posterior |
| 2 | `tutorial2_structured_jet` | Structured jet (7 params): MCMC, convergence diagnostics, posterior predictive checks |
| 3 | `tutorial3_gwfish_joint_detections` | GW detection efficiency with GWFish (ET + CE networks) |
| 4 | `tutorial4_gwfish_skyloc` | Sky localisation & Fisher-matrix parameter estimation |

---

## Citation

If you use this code, please cite:

```bibtex
@ARTICLE{2026A&A...710A.388D,
       author = {{De Santis}, A.~L. and {Ronchini}, S. and {Santoliquido}, F. and {Branchesi}, M.},
        title = "{Constraining binary neutron star population synthesis models using short gamma-ray burst data}",
      journal = {\aap},
     keywords = {astroparticle physics, gravitational waves, gamma-ray burst: general, High Energy Astrophysical Phenomena, Astrophysics of Galaxies},
         year = 2026,
        month = jul,
       volume = {710},
          eid = {A388},
        pages = {A388},
          doi = {10.1051/0004-6361/202659597},
archivePrefix = {arXiv},
       eprint = {2602.13391},
 primaryClass = {astro-ph.HE},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2026A&A...710A.388D},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}


```

---

## License

This work is licensed under a [Creative Commons Attribution 4.0 International
License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
