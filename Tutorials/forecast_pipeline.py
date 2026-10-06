"""Catalogue-based prompt, GWFish, and VegasAfterglow forecasts for Tutorial 6.

Sources: MAGGPY Tutorials 2.1, 3, 4; GWFish public API; the supplied
AfterGlowCode; THESEUS Yellow Book (2021), Table 3-1. Rates are counts / years.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.stats import beta

KEV_TO_ERG = 1.602176634e-9
H_CGS = 6.62607015e-27


@dataclass(frozen=True)
class PromptInstrument:
    name: str
    band_keV: tuple[float, float]
    window_s: float
    flux_limit: float
    flux_unit: str
    observing_efficiency: float
    position_arcmin: float | None = None
    position_fraction: float | None = None


DEFAULT_PROMPT = (
    PromptInstrument("GRINTA_TED", (50.0, 300.0), 1.0, 0.35, "ph", 0.6),
    PromptInstrument("Crystal_Eye", (50.0, 300.0), 2.0, 0.455, "ph", 0.3),
    PromptInstrument("THESEUS_XGIS", (30.0, 150.0), 1.0, 3e-8, "erg", 0.16 * 0.65, 15.0, 0.9),
)


def _rng(seed, tag):
    digest = sha256(str(tag).encode()).digest()
    return np.random.default_rng(np.random.SeedSequence([int(seed), int.from_bytes(digest[:4], "little")]))


def catalogue_exposure(years, catalogue_fj=0.7, target_fj=0.7, reference_efficiency=0.6):
    """Successful-jet exposure represented by the full tutorial export.

    generate_catalogue uses N = T * rate * geometry * 0.6 * fj. Thus
    the exported, unselected population represents T*0.6*fj_cat/fj_target
    all-sky years, within its sampled viewing-angle range. Apply each
    instrument's absolute observing efficiency once to this population.
    """
    if not np.isfinite(years) or years <= 0:
        raise ValueError("Catalogue duration must be positive and match the generation run.")
    for name, value in [("catalogue_fj", catalogue_fj), ("target_fj", target_fj), ("reference_efficiency", reference_efficiency)]:
        if not np.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"{name} must lie in (0, 1].")
    return float(years * reference_efficiency * catalogue_fj / target_fj)


def canonical_catalogue(raw):
    """Accept both raw generate_catalogue keys and Tutorial 2.1's renamed keys."""
    aliases = {
        "z": ("z", "redshift"),
        "theta_v_rad": ("theta", "viewing_angle", "theta_v_rad"),
        "Ep_keV": ("Ep_detected", "Ep_observed"),
        "peak_ph_50_300": ("pflux", "Pflux_observed"),
        "t_peak_s": ("t_peak", "T_peak_observed"),
        "T90_s": ("t90", "T90"),
        "E_iso_onaxis_erg": ("E_iso", "E_iso_rest_on_axis"),
    }
    frame = raw.copy().reset_index(drop=True)
    if frame.empty:
        raise ValueError("The catalogue is empty.")
    missing = []
    for target, names in aliases.items():
        if target in frame:
            continue
        found = next((name for name in names if name in frame), None)
        if found is None:
            missing.append(target)
        else:
            frame[target] = pd.to_numeric(frame[found], errors="raise")
    if missing:
        raise ValueError(
            "Use the full, unselected generate_catalogue export from Tutorial 2.1 "
            "(the 'Making a Full catalogue' section), saved before a later detected-only export. "
            f"Missing source fields: {missing}. A triggered-only CSV cannot supply the missing bursts."
        )
    for name in aliases:
        values = frame[name].to_numpy(float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Nonfinite values in {name}; correct the catalogue before forecasting.")
    for name in ["z", "Ep_keV", "peak_ph_50_300", "t_peak_s", "T90_s", "E_iso_onaxis_erg"]:
        if (frame[name] <= 0).any():
            raise ValueError(f"{name} must be positive.")
    if ((frame.theta_v_rad < 0) | (frame.theta_v_rad > np.pi / 2)).any():
        raise ValueError("The full catalogue viewing angles must be in radians, in [0, pi/2].")
    if "event_id" not in frame:
        frame["event_id"] = np.arange(len(frame), dtype=np.int64)
    if frame.event_id.isna().any() or not frame.event_id.is_unique:
        raise ValueError("event_id must be unique and nonmissing.")
    frame["theta_v_deg"] = np.rad2deg(frame.theta_v_rad)
    return frame


def attach_temporal_slopes(frame, structure_csv=None, default_structure_dir=None):
    """Use the same angle-dependent slopes as the catalogue's structure run."""
    out = frame.copy()
    if {"alpha_e", "alpha_n"}.issubset(out.columns):
        pass
    elif structure_csv is not None:
        structure = pd.read_csv(structure_csv).sort_values("theta_v")
        # load_custom_structure_constants uses theta_v in radians.
        if out.theta_v_rad.max() > structure.theta_v.max() + 1e-10:
            raise ValueError("The supplied structure table does not cover the catalogue viewing angles.")
        out["alpha_e"] = np.interp(out.theta_v_rad, structure.theta_v, structure.alpha_E)
        out["alpha_n"] = np.interp(out.theta_v_rad, structure.theta_v, structure.alpha_N)
    elif default_structure_dir is not None:
        for target, name in [("alpha_e", "alpha_e.txt"), ("alpha_n", "alpha.txt")]:
            values = np.loadtxt(Path(default_structure_dir) / name)
            if out.theta_v_deg.max() > values[:, 0].max() + 1e-10:
                raise ValueError("The default structure constants do not cover this viewing-angle range.")
            out[target] = np.interp(out.theta_v_deg, values[:, 0], values[:, 1])
    else:
        raise ValueError("Supply the catalogue's structure table, or its alpha_e/alpha_n columns.")
    for name in ["alpha_e", "alpha_n"]:
        if not np.all(np.isfinite(out[name])) or (out[name] < 0).any():
            raise ValueError(f"{name} must be finite and nonnegative for the tutorial's temporal profile.")
    return out


def sbpl_spectrum(energy_keV, ep_keV, alpha=-0.67, beta_s=-2.59, n=2.0):
    """MAGGPY's smoothly broken power law; E and Ep are observer-frame keV."""
    eps = (-(2 + alpha) / (2 + beta_s)) ** (1 / (n * (alpha - beta_s)))
    y = np.asarray(energy_keV) / (np.asarray(ep_keV) / eps)
    # logaddexp evaluates exactly the same shape without overflow in the tails.
    return np.exp(np.log(2) / n - np.logaddexp(-alpha * n * np.log(y), -beta_s * n * np.log(y)) / n)


class BandIntegrals:
    """Tabulate observer-frame photon and energy integrals in the required bands."""
    def __init__(self, ep_bounds, instruments=DEFAULT_PROMPT, n_ep=1600, n_energy=512, spectral_parameters=None):
        lo, hi = ep_bounds
        self.log_ep = np.linspace(np.log(lo), np.log(hi), n_ep)
        self.parameters = spectral_parameters or {}
        self.tables = {}
        bands = {((50.0, 300.0), "ph")} | {(tuple(i.band_keV), i.flux_unit) for i in instruments}
        for band, unit in bands:
            energy = np.geomspace(*band, n_energy)
            shape = sbpl_spectrum(energy[:, None], np.exp(self.log_ep)[None, :], **self.parameters)
            if unit == "erg":
                shape = shape * energy[:, None] * KEV_TO_ERG
            elif unit != "ph":
                raise ValueError("flux_unit must be 'ph' or 'erg'.")
            self.tables[(band, unit)] = np.log(np.trapz(shape, energy, axis=0))

    def __call__(self, ep, band, unit):
        values = np.asarray(ep, float)
        log_values = np.log(values)
        if np.any(log_values < self.log_ep[0] - 1e-8) or np.any(log_values > self.log_ep[-1] + 1e-8):
            raise ValueError("Ep falls outside the flux-integral grid; extend ep_bounds.")
        return np.exp(np.interp(log_values, self.log_ep, self.tables[(tuple(band), unit)]))


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


def select_prompt(frame, instruments=DEFAULT_PROMPT, seed=42, max_t90_s=None, chunk_size=2048, spectral_parameters=None):
    out = frame.copy()
    if len(out) == 0:
        raise ValueError("The catalogue is empty.")
    max_window = max(i.window_s for i in instruments)
    low_ep = np.min(out.Ep_keV.to_numpy() * (1 + max_window / out.t_peak_s.to_numpy()) ** (-out.alpha_e.to_numpy()))
    integrals = BandIntegrals((max(low_ep * 0.5, 1e-100), out.Ep_keV.max() * 2), instruments, spectral_parameters=spectral_parameters)
    source_cut = np.ones(len(out), dtype=bool) if max_t90_s is None else out.T90_s.to_numpy() < max_t90_s
    for instrument in instruments:
        if not 0 <= instrument.observing_efficiency <= 1 or instrument.flux_limit <= 0 or instrument.window_s <= 0:
            raise ValueError(f"Invalid instrument configuration: {instrument}")
        flux = np.empty(len(out)); trigger = np.empty(len(out))
        for first in range(0, len(out), chunk_size):
            last = first + chunk_size
            flux[first:last], trigger[first:last] = fixed_window_flux(out.iloc[first:last], instrument, integrals)
        name = instrument.name
        bright = source_cut & (flux >= instrument.flux_limit)
        available = _rng(seed, "prompt:" + name).random(len(out)) < instrument.observing_efficiency
        out[f"flux_{name}"] = flux
        out[f"trigger_time_{name}_s"] = trigger
        out[f"flux_pass_{name}"] = bright
        out[f"observable_{name}"] = available
        out[f"prompt_{name}"] = bright & available
    return out


def rate_fields(count, exposure_years, n_sources=None):
    """Count and annual rate, with an optional exact binomial MC interval.

    The tutorial generates a fixed number of iid source rows. A Clopper-Pearson
    interval for the selected fraction is therefore appropriate for its finite
    Monte Carlo uncertainty; it does not describe population-fit uncertainty.
    """
    n = int(count)
    if not np.isfinite(exposure_years) or exposure_years <= 0:
        raise ValueError("The represented exposure must be positive.")
    result = {"count": n, "rate_per_yr": n / exposure_years}
    if n_sources is not None:
        total = int(n_sources)
        if total <= 0 or not 0 <= n <= total:
            raise ValueError("Counts must lie between zero and the number of source rows.")
        low = 0.0 if n == 0 else beta.ppf(0.05, n, total - n + 1)
        high = 1.0 if n == total else beta.ppf(0.95, n + 1, total - n)
        result.update(mc_low90_per_yr=low * total / exposure_years,
                      mc_high90_per_yr=high * total / exposure_years)
    return result


def prompt_summary(frame, exposure_years, instruments=DEFAULT_PROMPT):
    rows = []
    for i in instruments:
        detected = frame[f"prompt_{i.name}"].to_numpy(bool)
        rows.append({"instrument": i.name, "band_keV": f"{i.band_keV[0]:g}-{i.band_keV[1]:g}",
                     "window_s": i.window_s, "flux_limit": i.flux_limit, "unit": i.flux_unit,
                     "observing_efficiency": i.observing_efficiency,
                     "flux_qualified_count": int(frame[f"flux_pass_{i.name}"].sum()),
                     **rate_fields(detected.sum(), exposure_years, len(frame)),
                     "median_z": frame.loc[detected, "z"].median(),
                     "median_viewing_angle_deg": frame.loc[detected, "theta_v_deg"].median(),
                     "prompt_position_arcmin_benchmark": i.position_arcmin,
                     "prompt_position_fraction_requirement": i.position_fraction})
    return pd.DataFrame(rows)


def binary_parameters(frame, seed=42, mass_mean=1.33, mass_sigma=0.09):
    from astropy.cosmology import Planck18
    rng = _rng(seed, "binary")
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
    if not np.isfinite(result.to_numpy()).all() or (result.luminosity_distance <= 0).any() or (masses <= 0).any():
        raise ValueError("Invalid binary parameters.")
    if not np.allclose(np.arccos(np.abs(np.cos(result.theta_jn))), frame.theta_v_rad, atol=1e-8):
        raise ValueError("Existing GW inclinations disagree with the catalogue viewing angles.")
    return result


def resolve_gw_config(template, psd_dir, output_path, detector_names):
    import yaml
    cfg = yaml.safe_load(Path(template).read_text())
    resolved = {}
    for name in detector_names:
        if name not in cfg:
            raise ValueError(f"Detector {name} is absent from {template}.")
        entry = dict(cfg[name])
        path = Path(entry["psd_data"])
        if not path.is_absolute():
            path = Path(psd_dir) / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing PSD for {name}: {path}")
        entry["psd_data"] = str(path)
        resolved[name] = entry
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(yaml.safe_dump(resolved, sort_keys=False))
    return Path(output_path)


def _cache_key(frame, settings, packages=()):
    h = sha256(Path(__file__).read_bytes())
    h.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    h.update(json.dumps(settings, sort_keys=True, default=str).encode())
    for package in packages:
        try:
            h.update(f"{package}:{version(package)}".encode())
        except PackageNotFoundError:
            pass
    return h.hexdigest()[:24]


def _save_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _require_aligned(frame, *others):
    if any(len(other) != len(frame) or not other.index.equals(frame.index) for other in others):
        raise ValueError("Catalogue, GW results, and afterglow parameters must share their row order and index.")


@lru_cache(maxsize=32)
def _network(detectors, config, snr_threshold):
    from astropy.utils import iers
    # The tutorial's simulated 2019–2020 epochs are covered by the bundled table.
    # This also keeps parallel workers from requesting IERS downloads per source.
    iers.conf.auto_download = False
    from GWFish.modules.detection import Network
    return Network(detector_ids=list(detectors), config=Path(config), detection_SNR=(0, snr_threshold))


def _snr_chunk(params, names, config, waveform, path):
    if Path(path).is_file():
        with np.load(path, allow_pickle=False) as stored:
            return stored["snr"]
    from GWFish.modules.utilities import get_snr
    result = get_snr(params, network=_network(tuple(names), config, 0.0), waveform_model=waveform)
    if isinstance(result, pd.DataFrame):
        snr = result[list(names)].to_numpy(float)
    else:
        snr = np.asarray(result, float)[:, :len(names)]
    if snr.shape != (len(params), len(names)) or not np.isfinite(snr).all():
        raise RuntimeError("GWFish returned invalid detector SNRs.")
    _save_npz(path, snr=snr)
    return snr


def _fisher_chunk(params, availability, detectors, config, waveform, threshold, path):
    if Path(path).is_file():
        with np.load(path, allow_pickle=False) as stored:
            return stored["area90_deg2"]
    from GWFish.modules.fishermatrix import compute_network_errors, sky_localization_percentile_factor
    parameters = ["mass_1_source", "mass_2_source", "luminosity_distance", "theta_jn", "ra", "dec", "psi", "phase", "geocent_time"]
    area = np.full(len(params), np.nan)
    patterns, inverse = np.unique(availability, axis=0, return_inverse=True)
    for j, pattern in enumerate(patterns):
        rows = np.flatnonzero(inverse == j)
        active = tuple(name for name, up in zip(detectors, pattern) if up)
        if not active:
            continue
        _, snr, errors, sky = compute_network_errors(
            network=_network(active, config, threshold), parameter_values=params.iloc[rows].reset_index(drop=True),
            fisher_parameters=parameters.copy(), waveform_model=waveform, use_duty_cycle=False,
        )
        if sky is not None:
            values = np.asarray(sky) * sky_localization_percentile_factor(90)
            valid = (np.asarray(snr) >= threshold) & np.isfinite(values) & (values > 0)
            valid &= np.isfinite(errors[:, 4:6]).all(axis=1) & (errors[:, 4:6] > 0).all(axis=1)
            area[rows[valid]] = values[valid]
    _save_npz(path, area90_deg2=area)
    return area


def run_gw(params, networks, config, cache_dir, seed=42, waveform="IMRPhenomD_NRTidalv2", threshold=8.0,
           n_jobs=1, chunk_size=256, use_duty_cycle=False):
    """SNR for all sources; Fisher localization only for GW-detected sources.

    A detector's uptime is drawn once and shared across network comparisons.
    Fisher calculations use that same active detector subset, without a second
    uptime draw. Setting use_duty_cycle=False reproduces the tutorial comparison.
    """
    import yaml
    from joblib import Parallel, delayed
    if len(params) == 0:
        raise ValueError("No binary parameters were supplied.")
    if not networks or any(not names or len(names) != len(set(names)) for names in networks.values()):
        raise ValueError("Each network must contain a nonempty list of distinct detector names.")
    if threshold <= 0 or chunk_size < 1:
        raise ValueError("SNR threshold and chunk size must be positive.")
    names = sorted({name for network in networks.values() for name in network})
    cfg = yaml.safe_load(Path(config).read_text())
    psd_hashes = {name: sha256(Path(cfg[name]["psd_data"]).read_bytes()).hexdigest() for name in names}
    key = _cache_key(params, {"config": Path(config).read_text(), "psds": psd_hashes,
                             "waveform": waveform, "networks": networks, "names": names, "threshold": threshold,
                             "seed": seed, "duty": use_duty_cycle}, ("GWFish", "lalsuite"))
    cache = Path(cache_dir) / key; cache.mkdir(parents=True, exist_ok=True)
    # A content-specific path also prevents stale in-memory detector instances.
    frozen_config = cache / "detectors.yaml"
    frozen_config.write_text(Path(config).read_text())
    slices = [(a, min(a + chunk_size, len(params))) for a in range(0, len(params), chunk_size)]
    print(f"GWFish: SNR for {len(params)} sources, {len(names)} detectors; cache {key}")
    chunks = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_snr_chunk)(params.iloc[a:b].reset_index(drop=True), names, str(frozen_config), waveform, cache / f"snr_{a}_{b}.npz")
        for a, b in slices
    )
    detector_snr = np.concatenate(chunks)
    up = np.ones(detector_snr.shape, dtype=bool)
    if use_duty_cycle:
        for j, name in enumerate(names):
            duty = float(cfg[name]["duty_factor"])
            if not 0 <= duty <= 1:
                raise ValueError(f"Invalid duty factor for {name}.")
            up[:, j] = _rng(seed, "gw_duty:" + name).random(len(params)) < duty
    result = pd.DataFrame(index=params.index)
    for j, name in enumerate(names):
        result[f"snr_{name}"] = detector_snr[:, j]
        result[f"up_{name}"] = up[:, j]
    for name, detectors in networks.items():
        columns = [names.index(detector) for detector in detectors]
        net_snr = np.sqrt(np.sum((detector_snr[:, columns] * up[:, columns]) ** 2, axis=1))
        detected = net_snr >= threshold
        area = np.full(len(params), np.nan)
        positions = np.flatnonzero(detected)
        print(f"  {name}: {len(positions)} detections; Fisher localization for this subset")
        pieces = [positions[a:a + chunk_size] for a in range(0, len(positions), chunk_size)]
        values = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(_fisher_chunk)(params.iloc[p].reset_index(drop=True), up[p][:, columns], detectors,
                                  str(frozen_config), waveform, threshold, cache / f"fisher_{name}_{int(p[0])}_{int(p[-1])}.npz")
            for p in pieces
        )
        if pieces:
            area[positions] = np.concatenate(values)
        result[f"snr_{name}"] = net_snr
        result[f"gw_{name}"] = detected
        result[f"area90_{name}_deg2"] = area
    return result


def joint_summary(frame, gw_results, exposure_years, instruments=DEFAULT_PROMPT, localization_cuts=(1, 10, 100, 1000)):
    _require_aligned(frame, gw_results)
    rows = []
    networks = [name[3:] for name in gw_results.columns if name.startswith("gw_")]
    for network in networks:
        area = gw_results[f"area90_{network}_deg2"].to_numpy()
        gw = gw_results[f"gw_{network}"].to_numpy(bool)
        for instrument in instruments:
            joint = frame[f"prompt_{instrument.name}"].to_numpy(bool) & gw
            row = {"instrument": instrument.name, "network": network, **rate_fields(joint.sum(), exposure_years, len(frame)),
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


@dataclass(frozen=True)
class AfterglowInstrument:
    name: str
    band_keV: tuple[float, float]
    flux_unit: str
    exposures_s: tuple[float, ...]
    limits: tuple[float, ...]
    followup_fraction: float = 1.0
    position_arcmin: float | None = None


HXI_EXPOSURES = (10.0, 100.0, 1000.0, 10000.0, 100000.0)
DEFAULT_AFTERGLOW = (
    AfterglowInstrument("GRINTA_HXI", (5.0, 30.0), "ph", HXI_EXPOSURES,
                        tuple(1.2e-3 * np.sqrt(1e4 / dt) for dt in HXI_EXPOSURES)),
    AfterglowInstrument("THESEUS_SXI", (0.3, 5.0), "erg", (100.0, 1500.0), (1e-10, 1.8e-11), position_arcmin=2.0),
)


def best_exposure(time, flux, duration, ready_time):
    """Exact best exposure for the piecewise-linear sampled light curve."""
    time, flux = np.asarray(time, float), np.asarray(flux, float)
    if len(time) < 2 or np.any(np.diff(time) <= 0) or not np.isfinite(flux).all() or (flux < 0).any():
        raise ValueError("A light curve needs increasing times and finite, nonnegative fluxes.")
    start_min = max(float(ready_time), time[0])
    start_max = time[-1] - duration
    if duration <= 0 or start_min > start_max:
        raise ValueError("The time grid must cover the full exposure after the response delay.")
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
    rng = _rng(seed, "afterglow_microphysics")
    n = len(frame)
    return pd.DataFrame({"n_ism_cm3": rng.uniform(2.5e-4, 1.5e-2, n),
                         "eps_B": 10 ** rng.uniform(-4, -2, n),
                         "Gamma0": 10 ** rng.normal(2.3, 0.3, n),
                         "D_L_cm": Planck18.luminosity_distance(frame.z.to_numpy()).to_value("cm")}, index=frame.index)


AF_COLUMNS = ["event_id", "imager", "route", "prompt_instrument", "delay_s", "ready_s", "best_start_s",
              "best_exposure_s", "average_flux", "flux_limit", "margin", "detected"]


def _afterglow_event(row, micro, imagers, prompt_instruments, delays, time, theta_core_deg, cache_path, resolutions, exposure_samples):
    if Path(cache_path).is_file():
        dtype = {"event_id": str} if isinstance(row.event_id, str) else None
        return pd.read_csv(cache_path, keep_default_na=False, dtype=dtype)
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
        if density.shape != (len(frequencies), len(time)):
            raise RuntimeError("Unexpected VegasAfterglow flux grid shape.")
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
                    if not np.isfinite(flux) or flux < 0:
                        raise RuntimeError("Invalid VegasAfterglow band flux.")
                    trials.append((flux / limit, start, dt, flux, limit))
                margin, start, dt, flux, limit = max(trials)
                records.append({"event_id": row.event_id, "imager": imager.name, "route": route,
                                "prompt_instrument": parent, "delay_s": delay, "ready_s": ready,
                                "best_start_s": start, "best_exposure_s": dt, "average_flux": flux,
                                "flux_limit": limit, "margin": margin, "detected": bool(margin >= 1)})
    result = pd.DataFrame(records, columns=AF_COLUMNS)
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    temp = Path(cache_path).with_suffix(".tmp")
    result.to_csv(temp, index=False); temp.replace(cache_path)
    return result


def run_afterglow(frame, gw_results, micro, cache_dir, instruments=DEFAULT_AFTERGLOW, prompt_instruments=DEFAULT_PROMPT,
                  delays=(0.0, 100.0, 300.0, 3600.0), localization_cut=100.0, theta_core_deg=5.0,
                  n_jobs=1, n_time=256, resolutions=(0.05, 0.2, 5), exposure_samples=32):
    """Afterglows for prompt triggers and possible localized GW follow-up targets."""
    from joblib import Parallel, delayed
    _require_aligned(frame, gw_results, micro)
    for i in instruments:
        if i.flux_unit not in ("ph", "erg") or len(i.exposures_s) != len(i.limits) or not i.exposures_s:
            raise ValueError(f"Invalid afterglow imager: {i.name}")
        if min(i.exposures_s) <= 0 or min(i.limits) <= 0 or not 0 <= i.followup_fraction <= 1:
            raise ValueError(f"Invalid afterglow limits or follow-up fraction: {i.name}")
    if not delays or not np.isfinite(delays).all() or min(delays) < 0:
        raise ValueError("Response delays must be nonnegative.")
    possible = np.zeros(len(frame), bool)
    for i in prompt_instruments:
        possible |= frame[f"prompt_{i.name}"].to_numpy(bool)
    for key in [x[3:] for x in gw_results if x.startswith("gw_")]:
        area = gw_results[f"area90_{key}_deg2"].to_numpy()
        possible |= gw_results[f"gw_{key}"].to_numpy(bool) & np.isfinite(area) & (area > 0) & (area <= localization_cut)
    indices = np.flatnonzero(possible)
    if not len(indices):
        return pd.DataFrame(columns=AF_COLUMNS)
    columns = ["event_id", "z", "theta_v_rad", "E_iso_onaxis_erg"]
    columns += [f"prompt_{i.name}" for i in prompt_instruments] + [f"trigger_time_{i.name}_s" for i in prompt_instruments]
    settings = {"imagers": [asdict(i) for i in instruments], "delays": delays, "theta_core_deg": theta_core_deg,
                "n_time": n_time, "resolutions": resolutions, "exposure_samples": exposure_samples}
    fingerprint = pd.concat([frame[columns], micro], axis=1)
    key = _cache_key(fingerprint, settings, ("VegasAfterglow",))
    cache = Path(cache_dir) / key
    time = np.geomspace(1.0, 1e7, n_time)
    print(f"VegasAfterglow: {len(indices)} sources; cache {key}")
    records = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_afterglow_event)(frame.iloc[p], micro.iloc[p], instruments, prompt_instruments, delays, time,
                                 theta_core_deg, cache / f"event_{p}.csv", resolutions, exposure_samples)
        for p in indices
    )
    return pd.concat(records, ignore_index=True)


def afterglow_summary(frame, gw_results, records, exposure_years, instruments=DEFAULT_AFTERGLOW,
                     prompt_instruments=DEFAULT_PROMPT, delays=(0.0, 100.0, 300.0, 3600.0),
                     localization_cut=100.0, seed=42):
    """Exclusive added afterglows and unique counterpart unions, per scenario."""
    _require_aligned(frame, gw_results)
    rows = []
    networks = [x[3:] for x in gw_results if x.startswith("gw_")]
    for imager in instruments:
        available = _rng(seed, "followup:" + imager.name).random(len(frame)) < imager.followup_fraction
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


def save_run_metadata(path, catalogue_path, exposure_years, settings):
    payload = {"catalogue_path": str(Path(catalogue_path).resolve()),
               "catalogue_sha256": sha256(Path(catalogue_path).read_bytes()).hexdigest(),
               "represented_years": exposure_years, "settings": settings, "packages": {}}
    for package in ["numpy", "scipy", "pandas", "astropy", "GWFish", "lalsuite", "VegasAfterglow", "joblib"]:
        try:
            payload["packages"][package] = version(package)
        except PackageNotFoundError:
            pass
    payload["helper_sha256"] = sha256(Path(__file__).read_bytes()).hexdigest()
    Path(path).write_text(json.dumps(payload, indent=2, default=str))
    return payload
