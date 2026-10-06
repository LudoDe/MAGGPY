"""Numerical functions for Tutorial 6: catalogue, prompt, GW and afterglows."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid

KEV_TO_ERG = 1.602176634e-9
H_CGS = 6.62607015e-27


@dataclass
class PromptInstrument:
    name: str
    band_keV: tuple
    window_s: float
    flux_limit: float
    flux_unit: str
    observing_efficiency: float
    position_arcmin: float = None
    position_fraction: float = None


DEFAULT_PROMPT = (
    PromptInstrument('GRINTA_TED', (50., 300.), 1., 0.35, 'ph', 0.6),
    PromptInstrument('Crystal_Eye', (50., 300.), 2., 0.455, 'ph', 0.3),
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
    geometry = 1 - np.cos(simulation.theta_v_max)
    n_sources = int(years * len(simulation.z_arr) * geometry * theta[-1])
    print(f'Generating {n_sources:,} sources for {years:g} years; f_j = {theta[-1]:.3g}')
    parts = []
    for start in range(0, n_sources, batch_size):
        n = min(batch_size, n_sources - start)
        source = generate_macro_properties_catalogue(theta, simulation, interps, n)
        t90, fluence = compute_time_evolution(source, interps)
        parts.append(pd.DataFrame({
            'z': source['z'],
            'theta_v_rad': source['theta_v'],
            'theta_v_deg': np.rad2deg(source['theta_v']),
            'Ep_keV': source['E_p_obs'],
            'Ep_onaxis_rest_keV': source['E_p_hat'],
            'peak_ph_50_300': source['F_p_real'],
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


def sbpl_spectrum(energy_keV, ep_keV, alpha=-0.67, beta_s=-2.59, n=2.0):
    """MAGGPY's smoothly broken power law; E and Ep are observer-frame keV."""
    eps = (-(2 + alpha) / (2 + beta_s)) ** (1 / (n * (alpha - beta_s)))
    y = np.asarray(energy_keV) / (np.asarray(ep_keV) / eps)
    # logaddexp evaluates exactly the same shape without overflow in the tails.
    return np.exp(np.log(2) / n - np.logaddexp(-alpha * n * np.log(y), -beta_s * n * np.log(y)) / n)


class BandIntegrals:
    """Observer-frame photon and energy integrals for the prompt bands."""
    def __init__(self, ep_bounds, instruments=DEFAULT_PROMPT, n_ep=1600, n_energy=512):
        self.log_ep = np.linspace(*np.log(ep_bounds), n_ep)
        self.tables = {}
        bands = {((50., 300.), 'ph')} | {(tuple(i.band_keV), i.flux_unit) for i in instruments}
        for band, unit in bands:
            energy = np.geomspace(*band, n_energy)
            shape = sbpl_spectrum(energy[:, None], np.exp(self.log_ep)[None, :])
            if unit == 'erg':
                shape = shape * energy[:, None] * KEV_TO_ERG
            self.tables[(band, unit)] = np.log(np.trapz(shape, energy, axis=0))

    def __call__(self, ep, band, unit):
        return np.exp(np.interp(np.log(ep), self.log_ep, self.tables[(tuple(band), unit)]))


def _prompt_flux(time, peak_time, ep, normalization, alpha_e, alpha_n, integrals, instrument):
    ratio = np.maximum(np.asarray(time) / peak_time, 0.0)
    tail = np.maximum(ratio, 1.0)
    ep_time = ep * tail ** (-alpha_e)
    amplitude = np.where(ratio < 1, ratio, tail ** (-alpha_n))
    return normalization * amplitude * integrals(ep_time, instrument.band_keV, instrument.flux_unit)


def fixed_window_flux(frame, instrument, integrals, quadrature_order=48):
    """Maximum flux averaged over the instrument's fixed observer-frame window.

    The tutorial profile rises linearly to t_peak and decays afterwards with
    evolving Ep. Its best averaging window starts between zero and t_peak.
    Locate it from F(start + dt) = F(start). Integrate the linear rise
    analytically and the tail in log time, resolving even very short pulses.
    """
    peak = frame.t_peak_s.to_numpy(float)
    ep = frame.Ep_keV.to_numpy(float)
    alpha_e = frame.alpha_e.to_numpy(float)
    alpha_n = frame.alpha_n.to_numpy(float)
    norm = frame.peak_ph_50_300.to_numpy(float) / integrals(ep, (50.0, 300.0), "ph")
    dt = instrument.window_s
    left = np.zeros(len(frame))
    right = peak.copy()
    for _ in range(44):
        mid = (left + right) / 2
        difference = (_prompt_flux(mid + dt, peak, ep, norm, alpha_e, alpha_n, integrals, instrument)
                      - _prompt_flux(mid, peak, ep, norm, alpha_e, alpha_n, integrals, instrument))
        left = np.where(difference > 0, mid, left)
        right = np.where(difference > 0, right, mid)
    start = (left + right) / 2
    end = start + dt
    split = np.minimum(np.maximum(peak, start), end)
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    peak_flux = norm * integrals(ep, instrument.band_keV, instrument.flux_unit)
    total = peak_flux * (split ** 2 - start ** 2) / (2 * peak)
    log_a, log_b = np.log(split), np.log(end)
    time = np.exp((log_a + log_b)[None, :] / 2 + nodes[:, None] * (log_b - log_a)[None, :] / 2)
    flux = _prompt_flux(time, peak, ep, norm, alpha_e, alpha_n, integrals, instrument)
    total += (log_b - log_a) / 2 * np.sum(weights[:, None] * flux * time, axis=0)
    return total / dt, end


def select_prompt(catalogue, instruments=DEFAULT_PROMPT, seed=42, max_t90_s=None, chunk_size=2048):
    """Apply each fixed-window flux cut and its sky coverage once."""
    out = catalogue.copy()
    max_window = max(i.window_s for i in instruments)
    low_ep = np.min(out.Ep_keV * (1 + max_window / out.t_peak_s) ** (-out.alpha_e))
    integrals = BandIntegrals((max(low_ep * 0.5, 1e-100), out.Ep_keV.max() * 2), instruments)
    source_cut = np.ones(len(out), bool) if max_t90_s is None else out.T90_s.to_numpy() < max_t90_s
    for j, instrument in enumerate(instruments):
        flux = np.empty(len(out))
        trigger = np.empty(len(out))
        for first in range(0, len(out), chunk_size):
            last = first + chunk_size
            flux[first:last], trigger[first:last] = fixed_window_flux(out.iloc[first:last], instrument, integrals)
        bright = source_cut & (flux >= instrument.flux_limit)
        available = np.random.default_rng(seed + 10 + j).random(len(out)) < instrument.observing_efficiency
        name = instrument.name
        out[f'flux_{name}'] = flux
        out[f'trigger_time_{name}_s'] = trigger
        out[f'flux_pass_{name}'] = bright
        out[f'prompt_{name}'] = bright & available
    return out


def rate_fields(count, years):
    return {'count': int(count), 'rate_per_yr': count / years}


def prompt_summary(catalogue, years, instruments=DEFAULT_PROMPT):
    rows = []
    for i in instruments:
        detected = catalogue[f'prompt_{i.name}']
        rows.append({
            'instrument': i.name, 'band_keV': f'{i.band_keV[0]:g}-{i.band_keV[1]:g}',
            'window_s': i.window_s, 'flux_limit': i.flux_limit, 'flux_unit': i.flux_unit,
            'observing_efficiency': i.observing_efficiency,
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
    valid = (snr >= threshold) & np.isfinite(area) & (area > 0)
    valid &= np.isfinite(errors[:, 4:6]).all(axis=1) & (errors[:, 4:6] > 0).all(axis=1)
    return np.where(valid, area, np.nan)


def run_gw(parameters, networks, config, waveform='IMRPhenomD_NRTidalv2', threshold=8.,
           n_jobs=1, chunk_size=256):
    """Compute SNRs, then 90% Fisher areas for the GW-detected sources."""
    from joblib import Parallel, delayed
    detectors = sorted({detector for network in networks.values() for detector in network})
    chunks = [parameters.iloc[a:a + chunk_size].reset_index(drop=True)
              for a in range(0, len(parameters), chunk_size)]
    print(f'GWFish: {len(parameters):,} sources, {len(detectors)} detectors')
    snr = np.concatenate(Parallel(n_jobs=n_jobs)(
        delayed(_snr_chunk)(chunk, detectors, config, waveform) for chunk in chunks))
    results = pd.DataFrame(index=parameters.index)
    for name, network in networks.items():
        columns = [detectors.index(detector) for detector in network]
        network_snr = np.sqrt(np.sum(snr[:, columns] ** 2, axis=1))
        detected = network_snr >= threshold
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
            joint = frame[f"prompt_{instrument.name}"].to_numpy(bool) & gw
            row = {"instrument": instrument.name, "network": network, **rate_fields(joint.sum(), exposure_years),
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


def best_exposure(time, flux, duration, ready_time):
    """Exact best exposure for the piecewise-linear sampled light curve."""
    time, flux = np.asarray(time, float), np.asarray(flux, float)
    start_min = max(float(ready_time), time[0])
    start_max = time[-1] - duration
    boundaries = np.unique(np.r_[start_min, start_max, time, time - duration])
    starts = boundaries[(boundaries >= start_min) & (boundaries <= start_max)]
    gradient = np.interp(starts + duration, time, flux) - np.interp(starts, time, flux)
    changing = gradient[:-1] * gradient[1:] < 0
    roots = starts[:-1][changing] - gradient[:-1][changing] * np.diff(starts)[changing] / np.diff(gradient)[changing]
    starts = np.r_[starts, roots]
    integral = cumulative_trapezoid(flux, time, initial=0)
    slopes = np.diff(flux) / np.diff(time)
    def primitive(t):
        index = np.clip(np.searchsorted(time, t, side="right") - 1, 0, len(time) - 2)
        delta = t - time[index]
        return integral[index] + flux[index] * delta + 0.5 * slopes[index] * delta ** 2
    means = (primitive(starts + duration) - primitive(starts)) / duration
    best = np.argmax(means)
    return float(starts[best]), float(means[best])


def afterglow_parameters(frame, seed=42):
    from astropy.cosmology import Planck18
    rng = np.random.default_rng(seed + 200)
    n = len(frame)
    return pd.DataFrame({"n_ism_cm3": rng.uniform(2.5e-4, 1.5e-2, n),
                         "eps_B": 10 ** rng.uniform(-4, -2, n),
                         "Gamma0": 10 ** rng.normal(2.3, 0.3, n),
                         "D_L_cm": Planck18.luminosity_distance(frame.z.to_numpy()).to_value("cm")}, index=frame.index)


AF_COLUMNS = ['event_id', 'imager', 'route', 'prompt_instrument', 'delay_s', 'ready_s',
              'best_start_s', 'best_exposure_s', 'average_flux', 'flux_limit', 'margin', 'detected']


def _afterglow_event(row, micro, imagers, prompt_instruments, delays, time, theta_core_deg, resolutions, exposure_samples):
    from VegasAfterglow import ISM, TophatJet, Observer, Radiation, Model
    model = Model(jet=TophatJet(theta_c=np.deg2rad(theta_core_deg), E_iso=float(row.E_iso_onaxis_erg),
                                Gamma0=float(micro.Gamma0), spreading=False),
                  medium=ISM(n_ism=float(micro.n_ism_cm3)),
                  observer=Observer(lumi_dist=float(micro.D_L_cm), z=float(row.z), theta_obs=float(row.theta_v_rad)),
                  fwd_rad=Radiation(eps_e=0.1, eps_B=float(micro.eps_B), p=2.2, xi_e=1.0), resolutions=resolutions)
    records = []
    routes = [("gw", "", 0.0)] + [("prompt", i.name, float(row[f"trigger_time_{i.name}_s"]))
                                  for i in prompt_instruments if bool(row[f"prompt_{i.name}"])]
    for imager in imagers:
        frequencies = np.geomspace(*imager.band_keV, 32) * KEV_TO_ERG / H_CGS
        density = np.asarray(model.flux_density_grid(time, frequencies).total)
        integrand = density if imager.flux_unit == "erg" else density / (H_CGS * frequencies[:, None])
        curve = np.trapz(integrand, frequencies, axis=0)
        for route, parent, alert_time in routes:
            for delay in delays:
                ready = alert_time + delay
                trials = []
                for dt, limit in zip(imager.exposures_s, imager.limits):
                    start, _ = best_exposure(time, curve, dt, ready)
                    averaged = np.asarray(model.flux_density_exposures(
                        np.full(len(frequencies), start), frequencies, np.full(len(frequencies), dt),
                        num_points=exposure_samples).total)
                    integrand = averaged if imager.flux_unit == "erg" else averaged / (H_CGS * frequencies)
                    flux = float(np.trapz(integrand, frequencies))
                    trials.append((flux / limit, start, dt, flux, limit))
                margin, start, dt, flux, limit = max(trials)
                records.append({"event_id": row.event_id, "imager": imager.name, "route": route,
                                "prompt_instrument": parent, "delay_s": delay, "ready_s": ready,
                                "best_start_s": start, "best_exposure_s": dt, "average_flux": flux,
                                "flux_limit": limit, "margin": margin, "detected": bool(margin >= 1)})
    return pd.DataFrame(records, columns=AF_COLUMNS)


def run_afterglow(catalogue, gw_results, microphysics, instruments=DEFAULT_AFTERGLOW,
                  prompt_instruments=DEFAULT_PROMPT, delays=(0., 100., 300., 3600.),
                  localization_cut=100., theta_core_deg=5., n_jobs=1,
                  n_time=256, resolutions=(0.05, 0.2, 5), exposure_samples=32):
    """Model prompt follow-up and additional localized GW-alert targets."""
    from joblib import Parallel, delayed
    possible = np.zeros(len(catalogue), bool)
    for instrument in prompt_instruments:
        possible |= catalogue[f'prompt_{instrument.name}'].to_numpy()
    for name in [column[3:] for column in gw_results if column.startswith('gw_')]:
        area = gw_results[f'area90_{name}_deg2'].to_numpy()
        possible |= gw_results[f'gw_{name}'].to_numpy() & (area > 0) & (area <= localization_cut)
    positions = np.flatnonzero(possible)
    print(f'VegasAfterglow: {len(positions):,} sources')
    if not len(positions):
        return pd.DataFrame(columns=AF_COLUMNS)
    time = np.geomspace(1., 1e7, n_time)
    records = Parallel(n_jobs=n_jobs)(
        delayed(_afterglow_event)(catalogue.iloc[p], microphysics.iloc[p], instruments,
                                 prompt_instruments, delays, time, theta_core_deg,
                                 resolutions, exposure_samples) for p in positions)
    return pd.concat(records, ignore_index=True)


def afterglow_summary(frame, gw_results, records, exposure_years, instruments=DEFAULT_AFTERGLOW,
                     prompt_instruments=DEFAULT_PROMPT, delays=(0.0, 100.0, 300.0, 3600.0),
                     localization_cut=100.0, seed=42):
    """Exclusive added afterglows and unique counterpart unions, per scenario."""
    rows = []
    networks = [x[3:] for x in gw_results if x.startswith("gw_")]
    for j, imager in enumerate(instruments):
        available = np.random.default_rng(seed + 300 + j).random(len(frame)) < imager.followup_fraction
        selected = records[records.imager == imager.name]
        for delay in delays:
            data = selected[selected.delay_s == delay]
            gw_found = set(data.loc[(data.route == "gw") & data.detected.astype(bool), "event_id"])
            gw_flux = frame.event_id.isin(gw_found).to_numpy() & available
            for prompt in prompt_instruments:
                gamma = frame[f"prompt_{prompt.name}"].to_numpy(bool)
                prompt_found = set(data.loc[(data.route == "prompt") & (data.prompt_instrument == prompt.name)
                                            & data.detected.astype(bool), "event_id"])
                prompt_flux = frame.event_id.isin(prompt_found).to_numpy() & available
                for network in networks:
                    gw = gw_results[f"gw_{network}"].to_numpy(bool)
                    area = gw_results[f"area90_{network}_deg2"].to_numpy()
                    localized = gw & np.isfinite(area) & (area > 0) & (area <= localization_cut)
                    prompt_joint = gamma & gw
                    prompt_afterglow = prompt_joint & prompt_flux
                    gw_afterglow = localized & gw_flux
                    extra = gw_afterglow & ~gamma
                    union = prompt_joint | extra
                    rows.append({"prompt_instrument": prompt.name, "imager": imager.name, "network": network,
                                 "delay_s": delay, "gw_localization_cut_deg2": localization_cut,
                                 "followup_fraction": imager.followup_fraction, "afterglow_position_arcmin": imager.position_arcmin,
                                 "prompt_joint_count": int(prompt_joint.sum()),
                                 "prompt_afterglow_joint_count": int(prompt_afterglow.sum()),
                                 "prompt_afterglow_joint_per_yr": prompt_afterglow.sum() / exposure_years,
                                 "gw_triggered_afterglow_count": int(gw_afterglow.sum()),
                                 "gw_triggered_afterglow_per_yr": gw_afterglow.sum() / exposure_years,
                                 "extra_without_prompt_count": int(extra.sum()),
                                 "extra_without_prompt_per_yr": extra.sum() / exposure_years,
                                 "unique_prompt_or_added_afterglow_count": int(union.sum()),
                                 "unique_prompt_or_added_afterglow_per_yr": union.sum() / exposure_years})
    return pd.DataFrame(rows)
