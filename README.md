<p align="center">
  <img src="LOGO.png" alt="MAGGPY logo" width="400">
</p>

<h1 align="center">MAGGPY</h1>

<p align="center">
  <b>Multimessenger Astronomy for GRBs and Gravitational Waves in Python</b>
</p>

## 🍉 About 

MAGGPY is a Python package I developed for simulating populations of short gamma-ray bursts and comparing them with observations from the Fermi/GBM catalogue.

The package uses forward Monte Carlo simulations and MCMC inference to investigate how assumptions about GRB jets, luminosity functions and binary neutron star merger rates affect the observed population. Both top-hat and structured jet models are included.

MAGGPY was originally developed for the analysis presented in [De Santis et al. (2026)](https://doi.org/10.1051/0004-6361/202659597).

## 🍉 What MAGGPY does

* Simulates short GRB populations and their observable properties
* Fits simulated populations to Fermi/GBM catalogue data
* Supports top-hat and structured jet models
* Includes different binary neutron star merger-rate models
* Runs MCMC inference with `emcee`
* Produces posterior and population-comparison plots
* Supports joint gravitational-wave and electromagnetic predictions through GWFish

MAGGPY is research software and is still under active development.

## 🍉 Installation

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

### 🍉 Optional dependencies

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

## 🍉 Getting started

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
6. [Structured-jet chain to detection forecasts](Tutorials/tutorial6_end_to_end_forecasts.ipynb): generate a full catalogue from the Tutorial 2 chain, apply the tutorial peak-flux cuts, then obtain joint GW and X-ray afterglow rates for GRINTA/TED, Crystal Eye and THESEUS

The tutorials are intended to be read in order, but they can also be used as examples for setting up an independent analysis.

Tutorial 6 starts from the converged structured-jet `emcee.h5` chain saved by Tutorial 2 and defaults to ten Monte Carlo years. Set the matching chain, merger-rate and structure files, then run the notebook. Install the forecast helpers in an editable checkout with `python -m pip install -e ".[plot,gw,notebooks,forecast]"`. Keep the `Tutorials/forecast_*.py` helpers alongside it. GWFish runs on the prompt-detected union, with consistent detector duty states for SNR and localization; the afterglow angle cut is an explicit setting (30° by default). Outputs in `Output_files/tutorial6_forecasts_10yr/` include annual joint rates, conditional fractions and counting intervals, angle-coloured X-ray curves, LaTeX tables and run provenance.

The [instrument audit](Tutorials/forecast_assumptions.md) distinguishes published Yellow Book/GRINTA/Crystal Eye specifications from adopted sensitivity, coverage and response assumptions. The [discussion](Tutorials/forecast_discussion.md) explains literature comparisons and the forward-shock/HLE distinction. Replot existing CSVs without rerunning the simulation using `python Tutorials/replot_forecasts.py /path/to/outputs --years 10`. Independent solver checks require no fitted chain: `python Tutorials/validate_afterglow_models.py /path/to/validation`. Ten simulated years are not an assumed mission lifetime, and the repository does not include the retained chain needed to execute the full forecast.

## 🍉 Installing for development

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

## 🍉 Citation

If you use MAGGPY in your work, please cite:

> A. L. De Santis, S. Ronchini, F. Santoliquido and M. Branchesi,
> “Constraining binary neutron star population synthesis models using short gamma-ray burst data,”
> *Astronomy & Astrophysics*, 710, A388 (2026).
> https://doi.org/10.1051/0004-6361/202659597

Full citation metadata is available in [`CITATION.cff`](CITATION.cff). GitHub also provides a **Cite this repository** button on the repository page.

## 🍉 Questions and problems

If you find a bug, have trouble reproducing a result, or have a question about the package, please [open an issue](https://github.com/LudoDe/MAGGPY/issues).

## 🍉 License

MAGGPY is distributed under the [BSD 3-Clause License](LICENSE).
