<p align="center">
  <img src="LOGO.png" alt="MAGGPY logo" width="400">
</p>

<h1 align="center">MAGGPY</h1>

<p align="center">
  <b>Multimessenger Astronomy for GRBs and Gravitational Waves in Python</b>
</p>

## About

MAGGPY is a Python package I developed for simulating populations of short gamma-ray bursts and comparing them with observations from the Fermi/GBM catalogue.

The package uses forward Monte Carlo simulations and MCMC inference to investigate how assumptions about GRB jets, luminosity functions and binary neutron star merger rates affect the observed population. Both top-hat and structured jet models are included.

MAGGPY was originally developed for the analysis presented in [De Santis et al. (2026)](https://doi.org/10.1051/0004-6361/202659597).

## What MAGGPY does

* Simulates short GRB populations and their observable properties
* Fits simulated populations to Fermi/GBM catalogue data
* Supports top-hat and structured jet models
* Includes different binary neutron star merger-rate models
* Runs MCMC inference with `emcee`
* Produces posterior and population-comparison plots
* Supports joint gravitational-wave and electromagnetic predictions through GWFish

MAGGPY is research software and is still under active development.

## Installation

MAGGPY currently supports Python 3.10 and 3.11. I recommend installing it in a clean environment.

Using Conda:

```bash
conda create -n maggpy python=3.10
conda activate maggpy
python -m pip install maggpy
```

Or using Python's built-in virtual environments:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install maggpy
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

### Optional dependencies

Plotting support:

```bash
python -m pip install "maggpy[plot]"
```

Gravitational-wave calculations:

```bash
python -m pip install "maggpy[gw]"
```

Fermi catalogue tools:

```bash
python -m pip install "maggpy[fermi]"
```

To install everything used by the tutorial notebooks:

```bash
python -m pip install "maggpy[plot,gw,fermi,notebooks]"
```

## Getting started

Check that MAGGPY is installed correctly:

```python
import maggpy

print(maggpy.__version__)
```

The example notebooks in [`Tutorials`](Tutorials) cover:

1. Preparing Fermi/GBM catalogue data
2. Running the top-hat jet model
3. Running the structured jet model
4. Predicting joint gravitational-wave and gamma-ray detections
5. Estimating gravitational-wave sky localisation

The tutorials are intended to be read in order, but they can also be used as examples for setting up an independent analysis.

## Installing for development

To work on the source code:

```bash
git clone https://github.com/LudoDe/MAGGPY.git
cd MAGGPY
python -m pip install -e ".[plot,gw,fermi,notebooks,dev]"
```

Run the tests with:

```bash
pytest
```

## Citation

If you use MAGGPY in your work, please cite:

> A. L. De Santis, S. Ronchini, F. Santoliquido and M. Branchesi,
> “Constraining binary neutron star population synthesis models using short gamma-ray burst data,”
> *Astronomy & Astrophysics*, 710, A388 (2026).
> https://doi.org/10.1051/0004-6361/202659597

Full citation metadata is available in [`CITATION.cff`](CITATION.cff). GitHub also provides a **Cite this repository** button on the repository page.

## Questions and problems

If you find a bug, have trouble reproducing a result, or have a question about the package, please [open an issue](https://github.com/LudoDe/MAGGPY/issues).

## License

MAGGPY is distributed under the [BSD 3-Clause License](LICENSE).
