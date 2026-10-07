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


DEFAULT_PROMPT = (
    PromptInstrument('GRINTA_TED', (50., 300.), 1., 0.32 * 3 / 5, 'ph', 0.5 * 8 / (4 * np.pi)),
    PromptInstrument('Crystal_Eye', (50., 300.), 2., 0.180, 'ph', 0.3),
    PromptInstrument('THESEUS_XGIS', (30., 150.), 1., 3e-8, 'erg', 0.16 * 0.65, 15., 0.9),
)


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
    return {'count': int(count), 'rate_per_yr': count / years}


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
           n_jobs=1, chunk_size=256):
    """Compute SNRs for the supplied prompt sample and Fisher areas above threshold."""
    from joblib import Parallel, delayed
    detectors = sorted({detector for network in networks.values() for detector in network})
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
    results = pd.DataFrame(index=parameters.index)
    for name, network in networks.items():
        columns = [detectors.index(detector) for detector in network]
        network_snr = np.sqrt(np.sum(snr[:, columns] ** 2, axis=1))
        detected = network_snr > threshold
        positions = np.flatnonzero(detected)
        area = np.full(len(parameters), np.nan)
        print(f'  {name}: {len(positions):,} GW detections')
        if len(positions):
            batches = [positions[a:a + chunk_size] for a in range(0, len(positions), chunk_size)]
            values = Parallel(n_jobs=n_jobs)(
                delayed(_fisher_chunk)(parameters.iloc[p].reset_index(drop=True), network,
                                      config, waveform, threshold) for p in batches)
            area[positions] = np.concatenate(values)
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
            for cut in localization_cuts:
                count = int((joint & np.isfinite(area) & (area > 0) & (area <= cut)).sum())
                row[f"localized_le_{cut:g}_deg2_count"] = count
                row[f"localized_le_{cut:g}_deg2_per_yr"] = count / exposure_years
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


HXI_EXPOSURES = (10., 100., 1000., 10000., 100000.)
DEFAULT_AFTERGLOW = (
    AfterglowInstrument('GRINTA_HXI', (5., 30.), 'ph', HXI_EXPOSURES,
                        tuple(1.2e-3 * np.sqrt(1e4 / dt) for dt in HXI_EXPOSURES)),
    AfterglowInstrument('THESEUS_SXI', (0.3, 5.), 'erg', (100., 1500.), (1e-10, 1.8e-11),
                        position_arcmin=2.),
)


def afterglow_parameters(frame, seed=42):
    from astropy.cosmology import Planck18
    rng = np.random.default_rng(seed + 200)
    n = len(frame)
    return pd.DataFrame({"n_ism_cm3": rng.uniform(2.5e-4, 1.5e-2, n),
                         "eps_B": 10 ** rng.uniform(-4, -2, n),
                         "Gamma0": 10 ** rng.normal(2.3, 0.3, n),
                         "D_L_cm": Planck18.luminosity_distance(frame.z.to_numpy()).to_value("cm")}, index=frame.index)


AF_COLUMNS = ['event_id', 'imager', 'delay_s', 'ready_s', 'peak_time_s',
              'best_start_s', 'best_exposure_s', 'average_flux', 'flux_limit', 'margin', 'detected']


def _afterglow_event(row, micro, imagers, delays, time, theta_core_deg, resolutions, exposure_samples):
    from VegasAfterglow import ISM, TophatJet, Observer, Radiation, Model
    model = Model(jet=TophatJet(theta_c=np.deg2rad(theta_core_deg), E_iso=float(row.E_iso_onaxis_erg),
                                Gamma0=float(micro.Gamma0), spreading=False),
                  medium=ISM(n_ism=float(micro.n_ism_cm3)),
                  observer=Observer(lumi_dist=float(micro.D_L_cm), z=float(row.z), theta_obs=float(row.theta_v_rad)),
                  fwd_rad=Radiation(eps_e=0.1, eps_B=float(micro.eps_B), p=2.2, xi_e=1.0), resolutions=resolutions)
    records = []
    for imager in imagers:
        frequencies = np.geomspace(*imager.band_keV, 32) * KEV_TO_ERG / H_CGS
        density = np.asarray(model.flux_density_grid(time, frequencies).total)
        integrand = density if imager.flux_unit == 'erg' else density / (H_CGS * frequencies[:, None])
        curve = np.trapz(integrand, frequencies, axis=0)
        for delay in delays:
            ready = row.t_peak_s + delay
            after_alert = time >= ready
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
                  n_time=256, resolutions=(0.05, 0.2, 5), exposure_samples=32):
    """Use the supplied prompt catalogue, already restricted to viewing angles ≤10°."""
    from joblib import Parallel, delayed
    print(f'VegasAfterglow: {len(catalogue):,} sources')
    if len(catalogue) == 0:
        return pd.DataFrame(columns=AF_COLUMNS)
    time = np.geomspace(1., 1e7, n_time)
    records = Parallel(n_jobs=n_jobs)(
        delayed(_afterglow_event)(catalogue.iloc[p], microphysics.iloc[p], instruments,
                                 delays, time, theta_core_deg, resolutions, exposure_samples)
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
                    rows.append({'prompt_instrument': prompt.name, 'imager': imager.name,
                                 'network': network, 'delay_s': delay,
                                 'localization_cut_deg2': localization_cut,
                                 'followup_fraction': imager.followup_fraction,
                                 'afterglow_position_arcmin': imager.position_arcmin,
                                 'prompt_afterglow_count': int((gamma & afterglow).sum()),
                                 'prompt_afterglow_per_yr': (gamma & afterglow).sum() / years,
                                 'prompt_gw_afterglow_count': int(joint.sum()),
                                 'prompt_gw_afterglow_per_yr': joint.sum() / years,
                                 'localized_prompt_gw_afterglow_count': int(localized.sum()),
                                 'localized_prompt_gw_afterglow_per_yr': localized.sum() / years})
    return pd.DataFrame(rows)
