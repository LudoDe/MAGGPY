"""Checks of catalogue normalization, peak-flux selection and joint counts."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Tutorials'))
import forecast_pipeline as fp


def catalogue():
    return pd.DataFrame({
        'event_id': [10, 30, 90], 'z': [0.01, 0.1, 1.],
        'theta_v_rad': [0.02, 0.1, 0.3], 'theta_v_deg': np.rad2deg([0.02, 0.1, 0.3]),
        'Ep_keV': [100., 500., 1500.], 'F_p_real': [100., 2., 0.01],
        'F_p_real_erg': [1e-5, 1e-7, 1e-10],
        't_peak_s': [0.05, 0.5, 2.], 'T90_s': [5., 10., 0.2],
        'E_iso_onaxis_erg': [1e51, 1e52, 1e53],
        'alpha_e': [1., 1.2, 0.8], 'alpha_n': [1.5, 2., 2.5],
    }, index=[3, 7, 11])


def test_catalogue_uses_chain_fj_all_sky_duration_and_native_flux_units():
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
    ratio = fp.energy_band_interpolator()(frame.Ep_keV) / interps.int_3_alt(frame.Ep_keV) / 6.2e8
    np.testing.assert_allclose(frame.F_p_real_erg / frame.F_p_real, ratio)


@pytest.mark.parametrize('ep', [1., 100., 8000.])
def test_xgis_energy_factor_matches_spectral_integral(ep):
    from maggpy.spectral_models import broken_power_law
    exact = quad(lambda energy: energy * broken_power_law(energy, ep), 30, 150,
                 epsabs=1e-25, epsrel=1e-10)[0]
    assert fp.energy_band_interpolator()(ep) == pytest.approx(exact, rel=5e-4)


def test_peak_time_cut_and_reference_time_scaling():
    frame = catalogue()
    ted = fp.PromptInstrument('TED', (50, 300), 1., 0.35, 'ph', 1.)
    ce = fp.PromptInstrument('CE', (50, 300), 2., 0.455, 'ph', 1.)
    xgis = fp.PromptInstrument('XGIS', (30, 150), 1., 3e-8, 'erg', 1.)
    result = fp.select_prompt(frame, [ted, ce, xgis])
    for instrument in (ted, ce, xgis):
        np.testing.assert_allclose(result[f'flux_limit_{instrument.name}'],
                                   instrument.flux_limit * np.sqrt(instrument.reference_time_s / frame.t_peak_s))
        assert result[f'prompt_{instrument.name}'].tolist() == [True, True, False]
    # T90 differs deliberately: the requested duration proxy is t_peak.
    assert result.T90_s.iloc[0] > 2 and result.prompt_TED.iloc[0]
    all_bright = frame.assign(F_p_real=100.)
    assert fp.select_prompt(all_bright, [ted], max_peak_time_s=None).prompt_TED.all()


def test_absolute_instrument_coverage_is_applied_once():
    frame = pd.concat([catalogue().iloc[:1]] * 10000, ignore_index=True)
    instruments = [fp.PromptInstrument('a', (50, 300), 1, 0.35, 'ph', 0.6),
                   fp.PromptInstrument('b', (50, 300), 1, 0.35, 'ph', 0.3)]
    result = fp.select_prompt(frame, instruments)
    assert result.prompt_a.mean() == pytest.approx(0.6, abs=0.015)
    assert result.prompt_b.mean() == pytest.approx(0.3, abs=0.015)


def test_binary_parameters_preserve_source_indices_and_bipolar_inclination():
    frame = catalogue().iloc[[0, 2]]
    binaries = fp.binary_parameters(frame)
    np.testing.assert_array_equal(binaries.index, frame.index)
    np.testing.assert_allclose(np.arccos(abs(np.cos(binaries.theta_jn))), frame.theta_v_rad, atol=1e-12)
    assert (binaries.mass_1_source >= binaries.mass_2_source).all()
    assert (binaries.luminosity_distance > 0).all()


def test_joint_denominator_and_localization_subsets():
    frame = catalogue().assign(prompt_test=[True, True, False])
    instrument = fp.PromptInstrument('test', (50, 300), 1, 0.35, 'ph', 0.6)
    gw = pd.DataFrame({'gw_ET': [True, False, True],
                       'area90_ET_deg2': [20., np.nan, 2.]}, index=frame.index)
    row = fp.joint_summary(frame, gw, 10, [instrument]).iloc[0]
    assert row.prompt_count == 2 and row['count'] == 1
    assert row.gw_detection_fraction == pytest.approx(0.5)
    assert row.rate_per_yr == pytest.approx(0.1)
    assert row.localized_le_10_deg2_count == 0 and row.localized_le_100_deg2_count == 1
    gw.loc[3, 'area90_ET_deg2'] = 0
    row = fp.joint_summary(frame, gw, 10, [instrument]).iloc[0]
    assert row.joint_without_valid_localization == 1 and row.localized_le_100_deg2_count == 0


@pytest.mark.parametrize('unit', ['erg', 'ph'])
def test_afterglow_exposure_follows_available_peak_and_preserves_units(monkeypatch, unit):
    def density(mean, frequencies):
        factor = np.ones_like(frequencies) if unit == 'erg' else fp.H_CGS * frequencies
        return mean * factor / (frequencies[-1] - frequencies[0])
    class Model:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
        def flux_density_grid(self, time, frequencies):
            curve = np.interp(time, [1, 2, 3, 4, 5], [0, 1, 2, 1, 0])
            return SimpleNamespace(total=density(1., frequencies)[:, None] * curve)
        def flux_density_exposures(self, starts, frequencies, durations, num_points):
            start, dt = starts[0], durations[0]
            mean = quad(lambda t: np.interp(t, [1, 2, 3, 4, 5], [0, 1, 2, 1, 0]),
                        start, start + dt, points=[4, 5])[0] / dt
            return SimpleNamespace(total=density(mean, frequencies))
    constructors = {name: (lambda **kwargs: kwargs) for name in ['ISM', 'TophatJet', 'Observer', 'Radiation']}
    monkeypatch.setitem(sys.modules, 'VegasAfterglow', SimpleNamespace(Model=Model, **constructors))
    row = catalogue().iloc[0].copy()
    row.t_peak_s = 0.25
    micro = pd.Series({'Gamma0': 200, 'n_ism_cm3': 0.001, 'D_L_cm': 1e27, 'eps_B': 0.001})
    imager = fp.AfterglowInstrument('test', (0.3, 5), unit, (2.,), (0.1,))
    result = fp._afterglow_event(row, micro, [imager], [3.], np.arange(1., 6.), 5., (0.05, 0.2, 5), 32).iloc[0]
    assert result.ready_s == pytest.approx(3.25) and result.peak_time_s == 4
    assert result.best_start_s == pytest.approx(3.25)
    assert result.average_flux == pytest.approx(0.765625)
    assert result.detected


def test_afterglow_counts_follow_each_prompt_mask_and_gw_localization():
    frame = catalogue().assign(prompt_a=[True, True, False], prompt_b=[False, False, True])
    gw = pd.DataFrame({'gw_ET': [True, False, True], 'area90_ET_deg2': [20., 5., np.nan]}, index=frame.index)
    prompts = [fp.PromptInstrument(name, (50, 300), 1, 0.35, 'ph', 1) for name in ('a', 'b')]
    imager = fp.AfterglowInstrument('SXI', (0.3, 5), 'erg', (100,), (1e-10,))
    records = pd.DataFrame({'event_id': [10, 90], 'imager': 'SXI', 'delay_s': 100., 'detected': True})
    table = fp.afterglow_summary(frame, gw, records, 10, [imager], prompts, [100.]).set_index('prompt_instrument')
    assert (table.prompt_afterglow_count == 1).all()
    assert (table.prompt_gw_afterglow_count == 1).all()
    assert table.loc['a', 'localized_prompt_gw_afterglow_count'] == 1
    assert table.loc['b', 'localized_prompt_gw_afterglow_count'] == 0
    np.testing.assert_allclose(table.prompt_gw_afterglow_per_yr, 0.1)
