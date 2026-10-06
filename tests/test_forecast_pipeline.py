"""Scientific checks for catalogue generation, fluxes and counterpart counts."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Tutorials'))
import forecast_pipeline as fp


def catalogue():
    return pd.DataFrame({
        'event_id': [0, 1, 2], 'z': [0.01, 0.1, 1.0],
        'theta_v_rad': [0.02, 0.1, 0.3],
        'theta_v_deg': np.rad2deg([0.02, 0.1, 0.3]),
        'Ep_keV': [100., 500., 1500.], 'peak_ph_50_300': [100., 2., 0.01],
        't_peak_s': [0.05, 0.5, 2.], 'T90_s': [0.2, 1., 5.],
        'E_iso_onaxis_erg': [1e51, 1e52, 1e53],
        'alpha_e': [1., 1.2, 0.8], 'alpha_n': [1.5, 2., 2.5],
    })


def test_catalogue_uses_chain_fj_and_all_sky_duration():
    from maggpy.structured_jet.montecarlo import SimParams, Interps
    from maggpy.structured_jet.init import create_integral_interpolators
    n_angles = 32
    simulation = SimParams(
        theta_c=np.deg2rad(3.4), theta_v_max=np.deg2rad(30),
        z_arr=np.linspace(0.01, 0.03, 1000), theta_v=np.linspace(0, np.deg2rad(30), n_angles),
        epeak_data=np.array([]), duration_data=np.array([]), pflux_data=np.array([]),
        fluence_data=np.array([]), yearly_rate=1, triggered_years=1,
        rng=np.random.default_rng(42), R_E=np.ones(n_angles), R_F=np.ones(n_angles),
        alpha_e=np.ones(n_angles), alpha_n=np.full(n_angles, 1.5))
    interps = Interps(*create_integral_interpolators()[:5])
    theta = np.array([2.5, 0., 2.7, 0.3, -0.5, 0.2, 1.4])
    years = 0.2
    expected = int(years * len(simulation.z_arr) * (1 - np.cos(simulation.theta_v_max)) * theta[-1])
    frame = fp.make_full_catalogue(theta, simulation, interps, years, batch_size=7)
    assert len(frame) == expected
    np.testing.assert_array_equal(frame.event_id, np.arange(expected))
    assert np.isfinite(frame.to_numpy()).all()
    assert (frame.E_iso_onaxis_erg > 0).all() and (frame.T90_s > 0).all()
    np.testing.assert_allclose(frame.alpha_e, 1)
    np.testing.assert_allclose(frame.alpha_n, 1.5)


@pytest.mark.parametrize('ep', [1., 100., 8000.])
def test_sbpl_band_conversion_against_energy_integral(ep):
    tables = fp.BandIntegrals((0.01, 1e5))
    for instrument in fp.DEFAULT_PROMPT:
        def integrand(energy):
            shape = fp.sbpl_spectrum(energy, ep)
            return shape if instrument.flux_unit == 'ph' else shape * energy * fp.KEV_TO_ERG
        exact = quad(integrand, *instrument.band_keV, epsabs=1e-25, epsrel=1e-10)[0]
        assert tables(ep, instrument.band_keV, instrument.flux_unit) == pytest.approx(exact, rel=6e-5)


@pytest.mark.parametrize('peak', [1e-5, 0.02, 0.5, 10.])
def test_fixed_window_handles_short_pulses_and_energy_flux(peak):
    frame = catalogue().iloc[:1].assign(t_peak_s=peak)
    tables = fp.BandIntegrals((1e-8, 1e4))
    for instrument in (fp.DEFAULT_PROMPT[0], fp.DEFAULT_PROMPT[2]):
        flux, end = fp.fixed_window_flux(frame, instrument, tables)
        norm = frame.peak_ph_50_300.iloc[0] / tables(100., (50, 300), 'ph')
        def instantaneous(time):
            return float(fp._prompt_flux(time, peak, 100., norm, 1., 1.5, tables, instrument))
        start = brentq(lambda s: instantaneous(s + instrument.window_s) - instantaneous(s), 0, peak, xtol=1e-16)
        rise = instantaneous(peak) * (peak ** 2 - start ** 2) / (2 * peak)
        tail = quad(lambda u: instantaneous(np.exp(u)) * np.exp(u), np.log(peak),
                    np.log(start + instrument.window_s), epsabs=1e-24, epsrel=3e-6, limit=150)[0]
        assert flux[0] == pytest.approx((rise + tail) / instrument.window_s, rel=1e-4)
        assert end[0] == pytest.approx(start + instrument.window_s, rel=1e-9)
        assert flux[0] <= instantaneous(peak)


def test_prompt_flux_cuts_and_optional_duration_cut():
    frame = catalogue()
    one = fp.PromptInstrument('one', (50, 300), 1, 1e-12, 'ph', 1)
    two = fp.PromptInstrument('two', (50, 300), 2, 1e-12, 'ph', 1)
    result = fp.select_prompt(frame, [one, two])
    assert result.prompt_one.all() and result.prompt_two.all()
    assert (result.flux_two <= result.flux_one * (1 + 1e-10)).all()
    short_only = fp.select_prompt(frame, [one], max_t90_s=2)
    assert short_only.prompt_one.tolist() == [True, True, False]


def test_best_afterglow_exposure_uses_integrated_flux_and_delay():
    time = np.array([1., 2., 3., 4., 5.])
    flux = np.array([0., 1., 2., 1., 0.])
    start, average = fp.best_exposure(time, flux, duration=2, ready_time=1)
    assert start == pytest.approx(2)
    assert average == pytest.approx(1.5)
    start, average = fp.best_exposure(time, flux, duration=2, ready_time=3)
    assert start == pytest.approx(3)
    assert average == pytest.approx(1)


def test_binary_masses_distance_and_bipolar_inclination():
    frame = catalogue()
    binaries = fp.binary_parameters(frame)
    np.testing.assert_allclose(np.arccos(abs(np.cos(binaries.theta_jn))), frame.theta_v_rad, atol=1e-12)
    assert (binaries.mass_1_source >= binaries.mass_2_source).all()
    assert (binaries.luminosity_distance > 0).all()


def test_joint_localization_counts_valid_areas():
    frame = catalogue().assign(prompt_test=True)
    gw = pd.DataFrame({'gw_ET': [True, True, True], 'area90_ET_deg2': [0., np.nan, 2.]})
    instrument = fp.PromptInstrument('test', (50, 300), 1, 0.35, 'ph', 0.6)
    row = fp.joint_summary(frame, gw, 10, [instrument]).iloc[0]
    assert row['count'] == 3 and row.rate_per_yr == pytest.approx(0.3)
    assert row.joint_without_valid_localization == 2
    assert row.localized_le_10_deg2_count == 1


def test_afterglow_extra_counts_are_exclusive_and_union_is_unique():
    frame = catalogue().assign(prompt_test=[True, False, False])
    gw = pd.DataFrame({'gw_ET': [True, True, True], 'area90_ET_deg2': [2., 20., np.nan]})
    prompt = fp.PromptInstrument('test', (50, 300), 1, 0.35, 'ph', 0.6)
    imager = fp.AfterglowInstrument('imager', (0.3, 5), 'erg', (100,), (1e-10,))
    records = pd.DataFrame({'event_id': [0, 1, 2, 0], 'imager': 'imager', 'delay_s': 100.,
                            'route': ['gw', 'gw', 'gw', 'prompt'],
                            'prompt_instrument': ['', '', '', 'test'],
                            'detected': [True, True, True, False]})
    row = fp.afterglow_summary(frame, gw, records, 10, [imager], [prompt], [100.]).iloc[0]
    assert row.prompt_joint_count == 1
    assert row.prompt_afterglow_joint_count == 0
    assert row.gw_triggered_afterglow_count == 2
    assert row.extra_without_prompt_count == 1
    assert row.unique_prompt_or_added_afterglow_count == 2
    assert row.unique_prompt_or_added_afterglow_per_yr == pytest.approx(0.2)
