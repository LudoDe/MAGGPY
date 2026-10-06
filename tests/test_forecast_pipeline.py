"""Numerical and catalogue-accounting checks for the end-to-end tutorial."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Tutorials"))
import forecast_pipeline as fp


def catalogue():
    return pd.DataFrame({
        "z": [0.01, 0.1, 1.0], "theta": [0.02, 0.1, 0.3],
        "Ep_detected": [100.0, 500.0, 1500.0], "pflux": [100.0, 2.0, 0.01],
        "t_peak": [0.05, 0.5, 2.0], "t90": [0.2, 1.0, 5.0],
        "E_iso": [1e51, 1e52, 1e53], "alpha_e": [1.0, 1.2, 0.8],
        "alpha_n": [1.5, 2.0, 2.5],
    })


def test_full_catalogue_aliases_and_radian_angles():
    raw = catalogue()
    renamed = raw.rename(columns={"z": "redshift", "theta": "viewing_angle", "Ep_detected": "Ep_observed",
                                  "pflux": "Pflux_observed", "t_peak": "T_peak_observed", "t90": "T90",
                                  "E_iso": "E_iso_rest_on_axis"})
    a, b = fp.canonical_catalogue(raw), fp.canonical_catalogue(renamed)
    columns = ["z", "theta_v_rad", "Ep_keV", "peak_ph_50_300", "t_peak_s", "T90_s", "E_iso_onaxis_erg"]
    np.testing.assert_array_equal(a[columns], b[columns])
    np.testing.assert_allclose(a.theta_v_deg, np.rad2deg(raw.theta))
    with pytest.raises(ValueError, match="radians"):
        fp.canonical_catalogue(raw.assign(theta=[2, 5, 10]))
    with pytest.raises(ValueError, match="full, unselected"):
        fp.canonical_catalogue(raw.drop(columns="E_iso"))


def test_exposure_counts_fj_and_reference_efficiency_once():
    years = fp.catalogue_exposure(10, catalogue_fj=0.7, target_fj=0.7, reference_efficiency=0.6)
    assert years == pytest.approx(6)
    # For 6000 stored jets, GRINTA's 0.6 gives the same rate as flux counts / 10 yr.
    assert (6000 * 0.6) / years == pytest.approx(6000 / 10)
    assert (6000 * 0.3) / years == pytest.approx(0.5 * 6000 / 10)
    assert fp.catalogue_exposure(10, 0.7, 0.35, 0.6) == pytest.approx(12)
    with pytest.raises(ValueError):
        fp.catalogue_exposure(0)


@pytest.mark.parametrize("ep", [1.0, 100.0, 8000.0])
def test_sbpl_band_conversion_against_adaptive_energy_integral(ep):
    tables = fp.BandIntegrals((0.01, 1e5))
    for instrument in fp.DEFAULT_PROMPT:
        def integrand(energy):
            shape = fp.sbpl_spectrum(energy, ep)
            return shape if instrument.flux_unit == "ph" else shape * energy * fp.KEV_TO_ERG
        exact = quad(integrand, *instrument.band_keV, epsabs=1e-25, epsrel=1e-10)[0]
        assert tables(ep, instrument.band_keV, instrument.flux_unit) == pytest.approx(exact, rel=6e-5)


@pytest.mark.parametrize("peak", [1e-5, 0.02, 0.5, 10.0])
def test_fixed_window_handles_short_pulses_and_energy_flux(peak):
    raw = catalogue().iloc[:1].assign(t_peak=peak)
    frame = fp.canonical_catalogue(raw)
    tables = fp.BandIntegrals((1e-8, 1e4))
    for instrument in (fp.DEFAULT_PROMPT[0], fp.DEFAULT_PROMPT[2]):
        flux, end = fp.fixed_window_flux(frame, instrument, tables)
        norm = frame.peak_ph_50_300.iloc[0] / tables(100.0, (50, 300), "ph")
        def instantaneous(time):
            return float(fp._prompt_flux(time, peak, 100.0, norm, 1.0, 1.5, tables, instrument))
        start = brentq(lambda s: instantaneous(s + instrument.window_s) - instantaneous(s), 0, peak, xtol=1e-16)
        rise = instantaneous(peak) * (peak ** 2 - start ** 2) / (2 * peak)
        # Integrating log time is an independent adaptive reference for the tail.
        tail = quad(lambda u: instantaneous(np.exp(u)) * np.exp(u), np.log(peak),
                    np.log(start + instrument.window_s), epsabs=1e-24, epsrel=3e-6, limit=150)[0]
        exact = (rise + tail) / instrument.window_s
        assert flux[0] == pytest.approx(exact, rel=1e-4)
        assert end[0] == pytest.approx(start + instrument.window_s, rel=1e-9)
        assert flux[0] <= instantaneous(peak)


def test_prompt_flux_cuts_do_not_use_empirical_detection_weights():
    frame = fp.canonical_catalogue(catalogue()).assign(detection_probability=0.0)
    one = fp.PromptInstrument("one", (50, 300), 1, 1e-12, "ph", 1)
    two = fp.PromptInstrument("two", (50, 300), 2, 1e-12, "ph", 1)
    result = fp.select_prompt(frame, [one, two])
    assert result.prompt_one.all() and result.prompt_two.all()
    assert (result.flux_two <= result.flux_one * (1 + 1e-10)).all()
    short_only = fp.select_prompt(frame, [one], max_t90_s=2)
    assert short_only.prompt_one.tolist() == [True, True, False]
    pd.testing.assert_frame_equal(result, fp.select_prompt(frame, [one, two]))


def test_best_afterglow_exposure_uses_integrated_flux_and_delay():
    time = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    flux = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
    start, average = fp.best_exposure(time, flux, duration=2, ready_time=1)
    assert start == pytest.approx(2)
    assert average == pytest.approx(1.5)
    start, average = fp.best_exposure(time, flux, duration=2, ready_time=3)
    assert start == pytest.approx(3)
    assert average == pytest.approx(1)
    with pytest.raises(ValueError, match="full exposure"):
        fp.best_exposure(time, flux, duration=2, ready_time=4)


def test_binary_masses_distance_and_bipolar_inclination():
    pytest.importorskip("astropy")
    frame = fp.canonical_catalogue(catalogue())
    binaries = fp.binary_parameters(frame)
    np.testing.assert_allclose(np.arccos(abs(np.cos(binaries.theta_jn))), frame.theta_v_rad, atol=1e-12)
    assert (binaries.mass_1_source >= binaries.mass_2_source).all()
    assert (binaries.luminosity_distance > 0).all()
    pd.testing.assert_frame_equal(binaries, fp.binary_parameters(frame))


def test_joint_localization_rejects_zero_and_nonfinite_areas():
    frame = fp.canonical_catalogue(catalogue()).assign(prompt_test=True)
    gw = pd.DataFrame({"gw_ET": [True, True, True], "area90_ET_deg2": [0.0, np.nan, 2.0]})
    instrument = fp.PromptInstrument("test", (50, 300), 1, 0.35, "ph", 0.6)
    row = fp.joint_summary(frame, gw, 10, [instrument]).iloc[0]
    assert row["count"] == 3
    assert row["joint_without_valid_localization"] == 2
    assert row["localized_le_10_deg2_count"] == 1
    with pytest.raises(ValueError, match="row order"):
        fp.joint_summary(frame, gw.iloc[::-1], 10, [instrument])


def test_afterglow_extra_counts_are_exclusive_and_union_is_unique(tmp_path):
    frame = fp.canonical_catalogue(catalogue()).assign(prompt_test=[True, False, False])
    gw = pd.DataFrame({"gw_ET": [True, True, True], "area90_ET_deg2": [2.0, 20.0, np.nan]})
    prompt = fp.PromptInstrument("test", (50, 300), 1, 0.35, "ph", 0.6)
    imager = fp.AfterglowInstrument("imager", (0.3, 5), "erg", (100,), (1e-10,))
    records = pd.DataFrame({"event_id": [0, 1, 2, 0], "imager": "imager", "delay_s": 100.0,
                            "route": ["gw", "gw", "gw", "prompt"],
                            "prompt_instrument": ["", "", "", "test"],
                            "detected": [True, True, True, False]})
    # Cached CSVs must preserve False as a boolean rather than a truthy string.
    path = tmp_path / "af.csv"
    records.to_csv(path, index=False)
    cached = pd.read_csv(path)
    assert not cached.detected.iloc[-1]
    row = fp.afterglow_summary(frame, gw, cached, 10, [imager], [prompt], [100.0]).iloc[0]
    assert row.prompt_joint_count == 1
    assert row.prompt_afterglow_joint_count == 0
    assert row.gw_triggered_afterglow_count == 2
    assert row.extra_without_prompt_count == 1
    assert row.unique_prompt_or_added_afterglow_count == 2
    assert row.unique_prompt_or_added_afterglow_per_yr == pytest.approx(0.2)


def test_gw_config_resolves_only_requested_psds(tmp_path):
    pytest.importorskip("yaml")
    import yaml
    (tmp_path / "good.txt").write_text("10 1e-44\n20 1e-44\n")
    template = tmp_path / "original.yaml"
    template.write_text(yaml.safe_dump({"good": {"psd_data": "good.txt"}, "unused": {"psd_data": "missing.txt"}}))
    output = fp.resolve_gw_config(template, tmp_path, tmp_path / "resolved.yaml", ["good"])
    config = yaml.safe_load(output.read_text())
    assert list(config) == ["good"]
    assert config["good"]["psd_data"] == str((tmp_path / "good.txt").resolve())


def test_empty_count_has_nonzero_mc_upper_limit():
    result = fp.rate_fields(0, 10, n_sources=1000)
    assert result["rate_per_yr"] == 0
    assert result["mc_low90_per_yr"] == 0
    assert result["mc_high90_per_yr"] == pytest.approx((1 - 0.05 ** (1 / 1000)) * 1000 / 10)


def test_cached_afterglow_keeps_string_identifiers_and_false_flags(tmp_path):
    path = tmp_path / "cached.csv"
    pd.DataFrame({"event_id": ["001"], "prompt_instrument": [""], "detected": [False]}).to_csv(path, index=False)
    result = fp._afterglow_event(pd.Series({"event_id": "001"}), micro=None, imagers=(), prompt_instruments=(),
                                 delays=(), time=None, theta_core_deg=5, cache_path=path,
                                 resolutions=(), exposure_samples=32)
    assert result.event_id.iloc[0] == "001"
    assert result.prompt_instrument.iloc[0] == ""
    assert not result.detected.iloc[0]
