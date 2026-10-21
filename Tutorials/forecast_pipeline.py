"""Numerical functions for Tutorial 6: catalogue, prompt, GW and afterglows."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import quad_vec

KEV_TO_ERG = 1.602176634e-9
H_CGS = 6.62607015e-27

# in case trapz doesn't exist rename
np.trapz = np.trapz if hasattr(np, 'trapz') else np.trapezoid

@dataclass
class PromptInstrument:
    name: str
    band_keV: tuple
    reference_time_s: float
    flux_limit: float
    flux_unit: str
    observing_efficiency: float
    position_arcmin: float = None
    position_fraction: float = None
    reference_sigma: float = 5.


def reference_prompt(sigma=5.):
    """Explicit common significance; see forecast_assumptions.md for provenance.

    TED's 0.3 at 1 s and the EM efficiencies remain adopted forecast inputs.
    Crystal Eye's published 50--300 keV 5-sigma value is 0.304, not the
    0.766 value for 10--1000 keV. Its 2-s reference time remains an assumption.
    The XGIS Yellow Book requirement is 3e-8 at 1 s and 3 sigma.
    Linear significance rescaling is a background-limited approximation.
    """
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('sigma must be positive and finite')
    return (
        PromptInstrument('GRINTA_TED', (50., 300.), 1., 0.3*sigma/5, 'ph',
                         0.5*8/(4*np.pi), reference_sigma=sigma),
        PromptInstrument('Crystal_Eye', (50., 300.), 2., 0.304*sigma/5, 'ph',
                         0.3, reference_sigma=sigma),
        PromptInstrument('THESEUS_XGIS', (30., 150.), 1., 3e-8*sigma/3, 'erg',
                         2/(4*np.pi)*0.65, 15., 0.9, sigma),
    )


DEFAULT_PROMPT = reference_prompt()


def make_full_catalogue(theta, simulation, interps, years, batch_size=10_000):
    """Generate all jet sources for T all-sky years, using MAGGPY's model.

    f_j comes from theta[-1]. Small batches keep the time-evolution arrays
    manageable; instrument sky coverage is applied later.
    """
    from maggpy.structured_jet.montecarlo import (
        generate_macro_properties_catalogue, compute_time_evolution,
        compute_Fp_64_ms_optimized,
    )
    int_30_150 = energy_band_interpolator()
    geometry = 1 - np.cos(simulation.theta_v_max)
    n_sources = int(years * len(simulation.z_arr) * geometry * theta[-1])
    print(f'Generating {n_sources:,} sources for {years:g} years; f_j = {theta[-1]:.3g}')
    parts = []
    for start in range(0, n_sources, batch_size):
        n = min(batch_size, n_sources - start)
        source = generate_macro_properties_catalogue(theta, simulation, interps, n)
        source['F_p_real_erg'] = source['F_0'] * int_30_150(source['E_p_obs'])
        t90, fluence = compute_time_evolution(source, interps)
        parts.append(pd.DataFrame({
            'z': source['z'],
            'theta_v_rad': source['theta_v'],
            'theta_v_deg': np.rad2deg(source['theta_v']),
            'Ep_keV': source['E_p_obs'],
            'Ep_onaxis_rest_keV': source['E_p_hat'],
            'F_p_real': source['F_p_real'],                 # ph cm^-2 s^-1, 50–300 keV
            'F_p_real_erg': source['F_p_real_erg'],         # erg cm^-2 s^-1, 30–150 keV
            'peak64_ph_50_300': compute_Fp_64_ms_optimized(source, interps),
            'fluence_50_300_erg_cm2': fluence,
            't_peak_s': source['t_peak_c_z'],
            'T90_s': t90,
            'E_iso_onaxis_erg': source['isotropic_energy'],
            'E_iso_structure_erg': source['isotropic_energy_w_structure'],
            'alpha_e': source['alpha_e'],
            'alpha_n': source['alpha_n'],
        }))
    catalogue = pd.concat(parts, ignore_index=True)
    catalogue.insert(0, 'event_id', np.arange(len(catalogue)))
    return catalogue


def energy_band_interpolator():
    """The Tutorial 2 energy-flux factor for XGIS, evaluated once per catalogue."""
    from maggpy.spectral_models import broken_power_law
    ep = np.logspace(-3, 5, 400)
    integral, _ = quad_vec(lambda energy: energy * broken_power_law(energy, ep), 30., 150.)
    return lambda peak_energy: np.exp(np.interp(np.log(peak_energy), np.log(ep), np.log(integral)))


def peak_flux_limit(peak_time_s, instrument):
    return instrument.flux_limit * np.sqrt(instrument.reference_time_s / np.asarray(peak_time_s))


def select_prompt(catalogue, instruments=DEFAULT_PROMPT, seed=42, max_peak_time_s=2.):
    """Apply the tutorial's peak-time and peak-flux cuts, then instrument coverage."""
    out = catalogue.copy()
    peak_time = out.t_peak_s.to_numpy()
    cond_time = np.ones(len(out), bool) if max_peak_time_s is None else peak_time < max_peak_time_s
    for j, instrument in enumerate(instruments):
        column = 'F_p_real_erg' if instrument.flux_unit == 'erg' else 'F_p_real'
        flux = out[column].to_numpy()
        limit = peak_flux_limit(peak_time, instrument)
        bright = cond_time & (flux > limit)
        available = np.random.default_rng(seed + 10 + j).random(len(out)) < instrument.observing_efficiency
        out[f'flux_limit_{instrument.name}'] = limit
        out[f'flux_pass_{instrument.name}'] = bright
        out[f'prompt_{instrument.name}'] = bright & available
    return out


def rate_fields(count, years):
    if not np.isfinite(years) or years <= 0:
        raise ValueError('years must be positive and finite')
    return {'count': int(count), 'rate_per_yr': count / years}


def fraction_fields(count, total, prefix):
    """Conditional fraction and 68.27% Wilson interval, including zero counts.

    Undefined denominators are NaN. These intervals describe counting noise,
    not the uncertainty in the population model or instrument response.
    """
    if count < 0 or total < count:
        raise ValueError('fraction requires 0 <= count <= total')
    if not total:
        value = low = high = np.nan
    else:
        value = count / total
        centre = (value + 0.5/total)/(1+1/total)
        half = np.sqrt(value*(1-value)/total + 0.25/total**2)/(1+1/total)
        low, high = max(0., centre-half), min(1., centre+half)
    return {prefix: value, prefix+'_lo68': low, prefix+'_hi68': high}


def prompt_summary(catalogue, years, instruments=DEFAULT_PROMPT):
    rows = []
    for i in instruments:
        detected = catalogue[f'prompt_{i.name}']
        rows.append({
            'instrument': i.name, 'band_keV': f'{i.band_keV[0]:g}-{i.band_keV[1]:g}',
            'reference_time_s': i.reference_time_s, 'reference_flux_limit': i.flux_limit,
            'flux_unit': i.flux_unit, 'observing_efficiency': i.observing_efficiency,
            'flux_qualified_count': int(catalogue[f'flux_pass_{i.name}'].sum()),
            **rate_fields(detected.sum(), years),
            'median_z': catalogue.loc[detected, 'z'].median(),
            'median_viewing_angle_deg': catalogue.loc[detected, 'theta_v_deg'].median(),
        })
    return pd.DataFrame(rows)


def binary_parameters(frame, seed=42, mass_mean=1.33, mass_sigma=0.09):
    from astropy.cosmology import Planck18
    rng = np.random.default_rng(seed + 100)
    n = len(frame)
    if {"mass_1_source", "mass_2_source"}.issubset(frame.columns):
        masses = frame[["mass_1_source", "mass_2_source"]].to_numpy(float)
    else:
        masses = rng.normal(mass_mean, mass_sigma, (n, 2))
        bad = (masses < 1) | (masses > 2.5)
        while bad.any():
            masses[bad] = rng.normal(mass_mean, mass_sigma, bad.sum())
            bad = (masses < 1) | (masses > 2.5)
        masses = np.sort(masses, axis=1)[:, ::-1]
    # The catalogue folds the two jets together. Restore either orbital hemisphere.
    theta = np.where(rng.random(n) < 0.5, frame.theta_v_rad, np.pi - frame.theta_v_rad)
    data = {
        "mass_1_source": masses[:, 0], "mass_2_source": masses[:, 1],
        "redshift": frame.z.to_numpy(), "luminosity_distance": Planck18.luminosity_distance(frame.z.to_numpy()).value,
        "theta_jn": theta, "ra": rng.uniform(0, 2 * np.pi, n),
        "dec": np.arcsin(rng.uniform(-1, 1, n)), "psi": rng.uniform(0, np.pi, n),
        "phase": rng.uniform(0, 2 * np.pi, n), "geocent_time": rng.uniform(1577491218, 1609027217, n),
        "a_1": np.zeros(n), "a_2": np.zeros(n),
    }
    for name in ["theta_jn", "ra", "dec", "psi", "phase", "geocent_time", "a_1", "a_2", "lambda_1", "lambda_2"]:
        if name in frame:
            data[name] = frame[name].to_numpy(float)
    result = pd.DataFrame(data, index=frame.index)
    return result


def resolve_gw_config(template, psd_dir, output_path, detector_names):
    """Use the tutorial PSDs with paths relative to the repository."""
    import yaml
    config = yaml.safe_load(Path(template).read_text())
    selected = {name: dict(config[name]) for name in detector_names}
    for entry in selected.values():
        entry['psd_data'] = str((Path(psd_dir) / entry['psd_data']).resolve())
    Path(output_path).write_text(yaml.safe_dump(selected, sort_keys=False))
    return Path(output_path)


def _network(detectors, config, threshold):
    from astropy.utils import iers
    from GWFish.modules.detection import Network
    iers.conf.auto_download = False   # The simulated epochs are in 2019–2020.
    return Network(detector_ids=list(detectors), config=Path(config), detection_SNR=(0, threshold))


def _snr_chunk(parameters, detectors, config, waveform):
    from GWFish.modules.utilities import get_snr
    snr = get_snr(parameters, network=_network(detectors, config, 0), waveform_model=waveform)
    return snr[list(detectors)].to_numpy()


def _fisher_chunk(parameters, detectors, config, waveform, threshold):
    from GWFish.modules.fishermatrix import compute_network_errors, sky_localization_percentile_factor
    fisher_parameters = ['mass_1_source', 'mass_2_source', 'luminosity_distance',
                         'theta_jn', 'ra', 'dec', 'psi', 'phase', 'geocent_time']
    _, snr, errors, sky = compute_network_errors(
        network=_network(detectors, config, threshold), parameter_values=parameters,
        fisher_parameters=fisher_parameters, waveform_model=waveform, use_duty_cycle=False,
    )
    area = np.asarray(sky) * sky_localization_percentile_factor(90)
    valid = (snr > threshold) & np.isfinite(area) & (area > 0)
    valid &= np.isfinite(errors[:, 4:6]).all(axis=1) & (errors[:, 4:6] > 0).all(axis=1)
    return np.where(valid, area, np.nan)


def run_gw(parameters, networks, config, waveform='IMRPhenomD_NRTidalv2', threshold=8.,
           n_jobs=1, chunk_size=256, detector_duty_cycle=1., seed=42):
    """Compute SNRs and Fisher areas for the same active detector subsets.

    One Bernoulli operational state per physical detector/source is shared by
    every network. Never multiply a completed network rate by duty**N: that
    loses partially operational networks and misstates their localizations.
    A duty cycle of one preserves the original all-operational calculation.
    """
    from joblib import Parallel, delayed
    detectors = sorted({detector for network in networks.values() for detector in network})
    if not 0 <= detector_duty_cycle <= 1:
        raise ValueError('detector_duty_cycle must be in [0, 1]')
    if len(parameters) == 0:
        results = pd.DataFrame(index=parameters.index)
        for name in networks:
            results[f'snr_{name}'] = pd.Series(index=parameters.index, dtype=float)
            results[f'gw_{name}'] = pd.Series(index=parameters.index, dtype=bool)
            results[f'area90_{name}_deg2'] = pd.Series(index=parameters.index, dtype=float)
        return results
    chunks = [parameters.iloc[a:a + chunk_size].reset_index(drop=True)
              for a in range(0, len(parameters), chunk_size)]
    print(f'GWFish: {len(parameters):,} sources, {len(detectors)} detectors')
    snr = np.concatenate(Parallel(n_jobs=n_jobs)(
        delayed(_snr_chunk)(chunk, detectors, config, waveform) for chunk in chunks))
    active = np.random.default_rng(seed + 150).random(snr.shape) < detector_duty_cycle
    snr = np.where(active, snr, 0.)
    results = pd.DataFrame(index=parameters.index)
    for j, detector in enumerate(detectors):
        results[f'active_{detector}'] = active[:, j]
    for name, network in networks.items():
        columns = [detectors.index(detector) for detector in network]
        network_snr = np.sqrt(np.sum(snr[:, columns] ** 2, axis=1))
        detected = network_snr > threshold
        positions = np.flatnonzero(detected)
        area = np.full(len(parameters), np.nan)
        print(f'  {name}: {len(positions):,} GW detections')
        if len(positions):
            # Localize using exactly the detectors contributing to this SNR.
            patterns = active[positions][:, columns]
            for pattern in np.unique(patterns, axis=0):
                subset = [d for d, live in zip(network, pattern) if live]
                if not subset:
                    continue
                selected = positions[np.all(patterns == pattern, axis=1)]
                batches = [selected[a:a + chunk_size] for a in range(0, len(selected), chunk_size)]
                values = Parallel(n_jobs=n_jobs)(
                    delayed(_fisher_chunk)(parameters.iloc[p].reset_index(drop=True), subset,
                                          config, waveform, threshold) for p in batches)
                area[selected] = np.concatenate(values)
        results[f'snr_{name}'] = network_snr
        results[f'gw_{name}'] = detected
        results[f'area90_{name}_deg2'] = area
    return results


def joint_summary(frame, gw_results, exposure_years, instruments=DEFAULT_PROMPT, localization_cuts=(1, 10, 100, 1000)):
    rows = []
    networks = [name[3:] for name in gw_results.columns if name.startswith("gw_")]
    for network in networks:
        area = gw_results[f"area90_{network}_deg2"].to_numpy()
        gw = gw_results[f"gw_{network}"].to_numpy(bool)
        for instrument in instruments:
            prompt = frame[f"prompt_{instrument.name}"].to_numpy(bool)
            joint = prompt & gw
            row = {"instrument": instrument.name, "network": network, **rate_fields(joint.sum(), exposure_years),
                   "prompt_count": int(prompt.sum()), "prompt_per_yr": prompt.sum() / exposure_years,
                   "gw_detection_fraction": joint.sum() / prompt.sum() if prompt.any() else np.nan,
                   "median_z": frame.loc[joint, "z"].median(),
                   "z_90th_percentile": frame.loc[joint, "z"].quantile(0.9),
                   "median_viewing_angle_deg": frame.loc[joint, "theta_v_deg"].median(),
                   "joint_without_valid_localization": int((joint & (~np.isfinite(area) | (area <= 0))).sum()),
                   "median_area90_deg2": np.median(area[joint & np.isfinite(area) & (area > 0)]) if np.any(joint & np.isfinite(area) & (area > 0)) else np.nan}
            row['joint_per_yr'] = row['rate_per_yr']
            row.update(fraction_fields(int(joint.sum()), int(prompt.sum()), 'joint_fraction_of_prompt'))
            for cut in localization_cuts:
                count = int((joint & np.isfinite(area) & (area > 0) & (area <= cut)).sum())
                row[f"localized_le_{cut:g}_deg2_count"] = count
                row[f"localized_le_{cut:g}_deg2_per_yr"] = count / exposure_years
                row.update(fraction_fields(count, int(joint.sum()), f'localized_le_{cut:g}_deg2_fraction_of_joint'))
            if instrument.position_arcmin is not None:
                row["em_position_arcmin_benchmark"] = instrument.position_arcmin
                row["em_position_fraction_requirement"] = instrument.position_fraction
            rows.append(row)
    return pd.DataFrame(rows)


@dataclass
class AfterglowInstrument:
    name: str
    band_keV: tuple
    flux_unit: str
    exposures_s: tuple
    limits: tuple
    followup_fraction: float = 1.
    position_arcmin: float = None
    reference_sigma: float = 5.


HXI_EXPOSURES = (10., 100., 1000., 10000., 100000.)
def reference_afterglow(sigma=5.):
    """HXI and SXI published 3-sigma limits rescaled to a common significance."""
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('sigma must be positive and finite')
    return (
        AfterglowInstrument('GRINTA_HXI', (5., 30.), 'ph', HXI_EXPOSURES,
                            tuple(1.2e-3*sigma/3*np.sqrt(1e4/dt) for dt in HXI_EXPOSURES),
                            reference_sigma=sigma),
        AfterglowInstrument('THESEUS_SXI', (0.3, 5.), 'erg', (100., 1500.),
                            (1e-10*sigma/3, 1.8e-11*sigma/3), position_arcmin=2., reference_sigma=sigma),
    )


DEFAULT_AFTERGLOW = reference_afterglow()


def afterglow_parameters(frame, seed=42, preset='legacy'):
    from astropy.cosmology import Planck18
    rng = np.random.default_rng(seed + 200)
    n = len(frame)
    result = pd.DataFrame({"n_ism_cm3": rng.uniform(2.5e-4, 1.5e-2, n),
                         "eps_B": 10 ** rng.uniform(-4, -2, n),
                         "Gamma0": 10 ** rng.normal(2.3, 0.3, n),
                         "D_L_cm": Planck18.luminosity_distance(frame.z.to_numpy()).to_value("cm")}, index=frame.index)
    if preset == 'legacy':
        result['Erad_over_Ekin'] = 1.
    elif preset in ('FS_SGRB', 'FS_GW17'):
        # Explicit ratio avoids the ambiguous eta convention in the paper.
        result['Erad_over_Ekin'] = rng.uniform(0.01, 0.1, n)
        result['Gamma0'] = 500.
        if preset == 'FS_SGRB':
            result['n_ism_cm3'] = rng.uniform(3e-3, 15e-3, n)
            result['eps_B'] = rng.uniform(0.01, 0.1, n)
        else:
            result['n_ism_cm3'], result['eps_B'] = 2.5e-4, 1.3e-4
    else:
        raise ValueError('preset must be legacy, FS_SGRB or FS_GW17')
    result['E_kin_iso_erg'] = frame.E_iso_onaxis_erg.to_numpy()/result.Erad_over_Ekin
    result['preset'] = preset
    return result


def afterglow_model(row, micro, theta_core_deg=5., resolutions=(0.05, 0.2, 5),
                    jet_structure='tophat', jet_slope=4.):
    """One forward-shock model shared by detection and light-curve diagnostics.

    E_iso_onaxis_erg is radiated prompt energy; VegasAfterglow needs kinetic
    energy. Legacy Series without the new energy field retain their old value.
    The power-law option is VegasAfterglow's profile, not an exact reproduction
    of the Ronchini HLE+FS structured-jet calculation.
    """
    from VegasAfterglow import ISM, TophatJet, Observer, Radiation, Model
    common = dict(theta_c=np.deg2rad(theta_core_deg),
                  E_iso=float(micro.get('E_kin_iso_erg', row.E_iso_onaxis_erg)),
                  Gamma0=float(micro.Gamma0), spreading=False)
    if jet_structure == 'tophat':
        jet = TophatJet(**common)
    elif jet_structure == 'powerlaw':
        from VegasAfterglow import PowerLawJet
        jet = PowerLawJet(**common, k_e=jet_slope, k_g=jet_slope)
    else:
        raise ValueError('jet_structure must be tophat or powerlaw')
    return Model(jet=jet, medium=ISM(n_ism=float(micro.n_ism_cm3)),
                 observer=Observer(lumi_dist=float(micro.D_L_cm), z=float(row.z), theta_obs=float(row.theta_v_rad)),
                 fwd_rad=Radiation(eps_e=0.1, eps_B=float(micro.eps_B), p=2.2, xi_e=1.0), resolutions=resolutions)


def band_light_curve(model, time_s, band_keV, flux_unit='erg', n_frequency=32):
    """Observer-frame band flux: erg cm^-2 s^-1 or photons cm^-2 s^-1."""
    if flux_unit not in ('erg', 'ph'):
        raise ValueError('flux_unit must be erg or ph')
    frequencies = np.geomspace(*band_keV, n_frequency)*KEV_TO_ERG/H_CGS
    density = np.asarray(model.flux_density_grid(np.asarray(time_s), frequencies).total)
    integrand = density if flux_unit == 'erg' else density/(H_CGS*frequencies[:, None])
    return np.trapz(integrand, frequencies, axis=0)


def _light_curve_event(row, micro, instruments, time_source_s, theta_core_deg,
                       resolutions, jet_structure, jet_slope):
    model = afterglow_model(row, micro, theta_core_deg, resolutions, jet_structure, jet_slope)
    time_observer = time_source_s*(1+row.z)
    nu_observer = 5.*KEV_TO_ERG/H_CGS/(1+row.z)
    fnu = np.asarray(model.flux_density_grid(time_observer, np.array([nu_observer])).total)[0]
    # nu_rest Lnu_rest = 4 pi DL^2 nu_obs Fnu_obs (no extra redshift factor).
    luminosity = 4*np.pi*micro.D_L_cm**2*nu_observer*fnu
    frames = []
    for instrument in instruments:
        flux = band_light_curve(model, time_observer, instrument.band_keV, instrument.flux_unit)
        frames.append(pd.DataFrame({
            'event_id': row.event_id, 'imager': instrument.name,
            'theta_v_deg': row.theta_v_deg, 'redshift': row.z,
            'time_observer_s': time_observer, 'time_source_s': time_source_s,
            'band_flux': flux, 'flux_unit': instrument.flux_unit,
            'nuLnu_5kev_erg_s': luminosity, 'E_kin_iso_erg': micro.E_kin_iso_erg,
            'preset': micro.get('preset', 'legacy'), 'jet_structure': jet_structure,
        }))
    return pd.concat(frames, ignore_index=True)


def sample_xray_light_curves(catalogue, microphysics, instruments=DEFAULT_AFTERGLOW,
                             max_curves=60, seed=42, n_time=160, n_jobs=1,
                             theta_core_deg=3.4, jet_structure='powerlaw', jet_slope=4.,
                             resolutions=(0.05, 0.2, 5)):
    """Deterministic sample stratified by angle; not a posterior confidence band.

    All saved curves use the same event microphysics as the detection forecast.
    A native-band flux and rest-frame 5-keV nu Lnu are exported separately.
    """
    from joblib import Parallel, delayed
    if not catalogue.index.equals(microphysics.index):
        raise ValueError('Catalogue and microphysics must have identical row indices')
    if len(catalogue) == 0:
        return pd.DataFrame(columns=LIGHT_CURVE_COLUMNS)
    if max_curves < 1:
        raise ValueError('max_curves must be positive')
    rng = np.random.default_rng(seed+250)
    # Divide the angle-ordered sample into strata before drawing within strata.
    strata = np.array_split(np.argsort(catalogue.theta_v_deg.to_numpy()), min(12, max_curves, len(catalogue)))
    chosen = []
    budgets = np.full(len(strata), max_curves//len(strata))
    budgets[:max_curves%len(strata)] += 1
    for positions, budget in zip(strata, budgets):
        chosen.extend(rng.choice(positions, min(len(positions), budget), replace=False))
    time = np.geomspace(1., 1e6, n_time)
    frames = Parallel(n_jobs=n_jobs)(delayed(_light_curve_event)(
        catalogue.iloc[p], microphysics.iloc[p], instruments, time, theta_core_deg,
        resolutions, jet_structure, jet_slope) for p in chosen)
    return pd.concat(frames, ignore_index=True)


AF_COLUMNS = ['event_id', 'imager', 'delay_s', 'ready_s', 'peak_time_s',
              'best_start_s', 'best_exposure_s', 'average_flux', 'flux_limit', 'margin', 'detected']

LIGHT_CURVE_COLUMNS = ['event_id', 'imager', 'theta_v_deg', 'redshift', 'time_observer_s',
                      'time_source_s', 'band_flux', 'flux_unit', 'nuLnu_5kev_erg_s',
                      'E_kin_iso_erg', 'preset', 'jet_structure']


def _afterglow_event(row, micro, imagers, delays, time, theta_core_deg, resolutions, exposure_samples,
                     jet_structure='tophat', jet_slope=4.):
    model = afterglow_model(row, micro, theta_core_deg, resolutions, jet_structure, jet_slope)
    records = []
    for imager in imagers:
        frequencies = np.geomspace(*imager.band_keV, 32) * KEV_TO_ERG / H_CGS
        density = np.asarray(model.flux_density_grid(time, frequencies).total)
        integrand = density if imager.flux_unit == 'erg' else density / (H_CGS * frequencies[:, None])
        curve = np.trapz(integrand, frequencies, axis=0)
        for delay in delays:
            ready = row.t_peak_s + delay
            after_alert = time >= ready
            if not after_alert.any():
                raise ValueError('The light-curve time grid ends before the response time')
            peak_time = time[after_alert][np.argmax(curve[after_alert])]
            trials = []
            for dt, limit in zip(imager.exposures_s, imager.limits):
                start = max(peak_time - dt / 2, ready, time[0])
                averaged = np.asarray(model.flux_density_exposures(
                    np.full(len(frequencies), start), frequencies, np.full(len(frequencies), dt),
                    num_points=exposure_samples).total)
                integrand = averaged if imager.flux_unit == 'erg' else averaged / (H_CGS * frequencies)
                flux = float(np.trapz(integrand, frequencies))
                trials.append((flux / limit, start, dt, flux, limit))
            margin, start, dt, flux, limit = max(trials)
            records.append({'event_id': row.event_id, 'imager': imager.name, 'delay_s': delay,
                            'ready_s': ready, 'peak_time_s': peak_time, 'best_start_s': start,
                            'best_exposure_s': dt, 'average_flux': flux, 'flux_limit': limit,
                            'margin': margin, 'detected': bool(margin > 1)})
    return pd.DataFrame(records, columns=AF_COLUMNS)


def run_afterglow(catalogue, microphysics, instruments=DEFAULT_AFTERGLOW,
                  delays=(0., 100., 300., 3600.), theta_core_deg=5., n_jobs=1,
                  n_time=256, resolutions=(0.05, 0.2, 5), exposure_samples=32,
                  jet_structure='tophat', jet_slope=4.):
    """Use the supplied prompt sample with its explicitly chosen angle cut."""
    from joblib import Parallel, delayed
    print(f'VegasAfterglow: {len(catalogue):,} sources')
    if len(catalogue) == 0:
        return pd.DataFrame(columns=AF_COLUMNS)
    if not catalogue.index.equals(microphysics.index):
        raise ValueError('Catalogue and microphysics must have identical row indices')
    time = np.geomspace(1., 1e7, n_time)
    records = Parallel(n_jobs=n_jobs)(
        delayed(_afterglow_event)(catalogue.iloc[p], microphysics.iloc[p], instruments,
                                 delays, time, theta_core_deg, resolutions, exposure_samples,
                                 jet_structure, jet_slope)
        for p in range(len(catalogue)))
    return pd.concat(records, ignore_index=True)


def afterglow_summary(frame, gw_results, records, years, instruments=DEFAULT_AFTERGLOW,
                     prompt_instruments=DEFAULT_PROMPT, delays=(0., 100., 300., 3600.),
                     localization_cut=100., seed=42):
    """Count afterglows of prompt events and their GW-detected, localized subsets."""
    rows = []
    networks = [name[3:] for name in gw_results if name.startswith('gw_')]
    for j, imager in enumerate(instruments):
        available = np.random.default_rng(seed + 300 + j).random(len(frame)) < imager.followup_fraction
        selected = records[records.imager == imager.name]
        for delay in delays:
            data = selected[selected.delay_s == delay]
            found = set(data.loc[data.detected.astype(bool), 'event_id'])
            afterglow = frame.event_id.isin(found).to_numpy() & available
            for prompt in prompt_instruments:
                gamma = frame[f'prompt_{prompt.name}'].to_numpy(bool)
                for network in networks:
                    gw = gw_results[f'gw_{network}'].to_numpy(bool)
                    area = gw_results[f'area90_{network}_deg2'].to_numpy()
                    joint = gamma & gw & afterglow
                    localized = joint & np.isfinite(area) & (area > 0) & (area <= localization_cut)
                    row = {'prompt_instrument': prompt.name, 'imager': imager.name,
                                 'network': network, 'delay_s': delay,
                                 'localization_cut_deg2': localization_cut,
                                 'followup_fraction': imager.followup_fraction,
                                 'afterglow_position_arcmin': imager.position_arcmin,
                                 'prompt_afterglow_count': int((gamma & afterglow).sum()),
                                 'prompt_afterglow_per_yr': (gamma & afterglow).sum() / years,
                                 'prompt_gw_afterglow_count': int(joint.sum()),
                                 'prompt_gw_afterglow_per_yr': joint.sum() / years,
                                 'localized_prompt_gw_afterglow_count': int(localized.sum()),
                                 'localized_prompt_gw_afterglow_per_yr': localized.sum() / years,
                                 'prompt_count': int(gamma.sum()), 'joint_prompt_count': int((gamma & gw).sum())}
                    row.update(fraction_fields(int((gamma & afterglow).sum()), int(gamma.sum()), 'afterglow_fraction_of_prompt'))
                    row.update(fraction_fields(int(joint.sum()), int((gamma & gw).sum()), 'afterglow_fraction_of_joint_prompt'))
                    row.update(fraction_fields(int(localized.sum()), int(joint.sum()), 'localized_fraction_of_joint_afterglow'))
                    rows.append(row)
    return pd.DataFrame(rows)
