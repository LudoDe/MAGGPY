"""Scientific invariants introduced by the ten-year forecast workflow."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'Tutorials'))
import forecast_pipeline as fp
import forecast_reporting as reporting


def test_reference_sensitivities_share_significance_and_keep_the_correct_band():
    ted, ce, xgis = fp.reference_prompt(5.)
    assert ce.band_keV == (50., 300.) and ce.flux_limit == pytest.approx(.304)
    assert ce.reference_time_s == 2.  # Adopted timing assumption, documented separately.
    assert xgis.flux_limit == pytest.approx(5e-8)
    assert xgis.observing_efficiency == pytest.approx(2/(4*np.pi)*.65)
    hxi, sxi = fp.reference_afterglow(5.)
    assert hxi.limits[-2] == pytest.approx(1.2e-3*5/3)
    np.testing.assert_allclose(sxi.limits, np.array([1e-10, 1.8e-11])*5/3)
    assert {i.reference_sigma for i in (ted, ce, xgis, hxi, sxi)} == {5.}


def test_fraction_denominators_and_zero_count_intervals_are_explicit():
    result = fp.fraction_fields(0, 10, 'joint_fraction_of_prompt')
    assert result['joint_fraction_of_prompt'] == 0
    assert result['joint_fraction_of_prompt_hi68'] > 0
    assert np.isnan(fp.fraction_fields(0, 0, 'fraction')['fraction'])
    with pytest.raises(ValueError):
        fp.fraction_fields(11, 10, 'fraction')
    assert fp.rate_fields(3, 10)['rate_per_yr'] == pytest.approx(.3)
    for years in (0, -1, np.nan):
        with pytest.raises(ValueError):
            fp.rate_fields(1, years)


def test_duty_states_are_shared_between_networks_and_used_for_localization(monkeypatch):
    n = 200
    parameters = pd.DataFrame({'row_id': np.arange(n)})
    networks = {'AB': ['a', 'b'], 'ABC': ['a', 'b', 'c']}
    calls = []
    def snr(frame, detectors, config, waveform):
        return np.full((len(frame), len(detectors)), 12.)
    def fisher(frame, detectors, config, waveform, threshold):
        calls.append((frame.row_id.to_numpy(), list(detectors)))
        return np.full(len(frame), 100./len(detectors))
    monkeypatch.setattr(fp, '_snr_chunk', snr)
    monkeypatch.setattr(fp, '_fisher_chunk', fisher)
    result = fp.run_gw(parameters, networks, None, n_jobs=1, detector_duty_cycle=.5)
    for network, detectors in networks.items():
        live = result[[f'active_{d}' for d in detectors]].to_numpy()
        np.testing.assert_allclose(result[f'snr_{network}'], 12*np.sqrt(live.sum(axis=1)))
        np.testing.assert_array_equal(result[f'gw_{network}'], live.any(axis=1))
        expected = np.full(n, np.nan)
        np.divide(100., live.sum(axis=1), out=expected, where=live.any(axis=1))
        np.testing.assert_allclose(result[f'area90_{network}_deg2'], expected)
    for row_ids, detectors in calls:
        assert result.loc[row_ids, [f'active_{d}' for d in detectors]].to_numpy().all()
    offline = fp.run_gw(parameters, networks, None, n_jobs=1, detector_duty_cycle=0.)
    assert not offline[['gw_AB', 'gw_ABC']].to_numpy().any()


def test_literature_presets_convert_prompt_radiation_to_kinetic_energy():
    pytest.importorskip('astropy')
    frame = pd.DataFrame({'z': [.1, .5, 1.], 'E_iso_onaxis_erg': [1e50, 1e51, 1e52]}, index=[3, 7, 11])
    sgrb = fp.afterglow_parameters(frame, preset='FS_SGRB')
    assert sgrb.index.equals(frame.index)
    assert sgrb.n_ism_cm3.between(.003, .015).all()
    assert sgrb.eps_B.between(.01, .1).all()
    assert sgrb.Erad_over_Ekin.between(.01, .1).all()
    np.testing.assert_allclose(sgrb.E_kin_iso_erg*sgrb.Erad_over_Ekin, frame.E_iso_onaxis_erg)
    assert (sgrb.Gamma0 == 500).all()
    gw17 = fp.afterglow_parameters(frame, preset='FS_GW17')
    np.testing.assert_allclose(gw17.n_ism_cm3, 2.5e-4)
    np.testing.assert_allclose(gw17.eps_B, 1.3e-4)
    legacy = fp.afterglow_parameters(frame, preset='legacy')
    np.testing.assert_allclose(legacy.E_kin_iso_erg, frame.E_iso_onaxis_erg)


def test_rest_frame_luminosity_has_no_extra_redshift_factor(monkeypatch):
    z, distance, source_lnu = .8, 1e28, 1e28
    time_source = np.array([10., 100.])
    class Model:
        def flux_density_grid(self, time_observer, frequencies):
            np.testing.assert_allclose(time_observer, time_source*(1+z))
            np.testing.assert_allclose(frequencies, [5*fp.KEV_TO_ERG/fp.H_CGS/(1+z)])
            return SimpleNamespace(total=np.full((1, len(time_source)),
                                   (1+z)*source_lnu/(4*np.pi*distance**2)))
    monkeypatch.setattr(fp, 'afterglow_model', lambda *args, **kwargs: Model())
    monkeypatch.setattr(fp, 'band_light_curve', lambda *args: np.ones(len(time_source)))
    row = pd.Series(dict(event_id=1, z=z, theta_v_rad=.1, theta_v_deg=5.7))
    micro = pd.Series(dict(D_L_cm=distance, E_kin_iso_erg=1e52))
    imager = fp.AfterglowInstrument('SXI', (.3, 5), 'erg', (100.,), (1e-10,))
    frame = fp._light_curve_event(row, micro, [imager], time_source, 3.4, (.1, .2, 5), 'powerlaw', 4.)
    np.testing.assert_allclose(frame.nuLnu_5kev_erg_s, 5*fp.KEV_TO_ERG/fp.H_CGS*source_lnu)


def test_publication_rates_are_context_not_normalization_targets():
    prompt = pd.DataFrame({'instrument': ['GRINTA_TED'], 'rate_per_yr': [60.]})
    joint = pd.DataFrame({'instrument': ['GRINTA_TED']*2, 'network': ['Asharp', 'ET_2L'],
                          'joint_per_yr': [1.5, 44.]})
    comparison = reporting.publication_rate_comparison(prompt, joint)
    np.testing.assert_allclose(comparison.reference_per_yr, [65., 2., 50.])
    np.testing.assert_allclose(comparison.forecast_per_yr, [60., 1.5, 44.])


def test_ligo_india_is_at_the_published_eastern_site():
    import yaml
    site = yaml.safe_load((Path(__file__).resolve().parents[1]/'Tutorials/configs/coba.yaml').read_text())['LIO_sharp']
    # Evaluate the trusted repository coordinate expressions in the GWFish convention.
    latitude = np.rad2deg(eval(site['lat'], {'__builtins__': {}}, {'np': np}))
    longitude = np.rad2deg(eval(site['lon'], {'__builtins__': {}}, {'np': np}))
    # Nuttall, Indico April 2026, slide 8: 19d36m50s N, 77d01m54s E.
    assert latitude == pytest.approx(19.6138888889)
    assert longitude == pytest.approx(77.0316666667)


def test_run_metadata_records_input_digests(tmp_path):
    import hashlib
    import json
    mrd = tmp_path/'mrd.csv'
    mrd.write_text('z,rate\n0,1\n')
    metadata = reporting.save_run_metadata(tmp_path, {'exposure_years': 10., 'mrd_path': str(mrd)})
    saved = json.loads(metadata.read_text())
    assert saved['exposure_years'] == 10.
    assert saved['input_sha256']['mrd_path'] == hashlib.sha256(mrd.read_bytes()).hexdigest()


def test_saved_curves_must_match_event_catalogue(tmp_path):
    pytest.importorskip('matplotlib')
    from forecast_plotting import _saved_light_curves
    events = pd.DataFrame({'event_id': [1], 'z': [.5], 'theta_v_deg': [3.]})
    curves = pd.DataFrame({'event_id': [1], 'redshift': [.5], 'theta_v_deg': [3.],
                           'time_source_s': [100.], 'time_observer_s': [150.]})
    curves.to_csv(tmp_path/'xray_light_curves.csv', index=False)
    loaded, _ = _saved_light_curves(tmp_path, events)
    assert len(loaded) == 1
    for column, wrong in [('event_id', 999), ('redshift', .9), ('time_observer_s', 100.)]:
        bad = curves.copy()
        bad[column] = wrong
        bad.to_csv(tmp_path/'xray_light_curves.csv', index=False)
        with pytest.raises(ValueError):
            _saved_light_curves(tmp_path, events)
