"""Reproducible physical checks, independent of a fitted forecast chain.

Compare CGS VegasAfterglow fluxes to afterglowpy's mJy output for identical
top-hat forward shocks; show the effect of literature microphysics separately.
These controlled curves are not a ten-year population forecast or an HLE model.
"""
import argparse
from pathlib import Path
from importlib.metadata import version
import json
import platform

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import forecast_pipeline as fp
from forecast_plotting import apply_forecast_style, save_forecast_figure


def solver_comparison(time_source_s=None, angles=(0., 3.4, 7.5, 15.)):
    import afterglowpy as ag
    from astropy.cosmology import Planck18
    time = np.geomspace(1e2, 1e6, 120) if time_source_s is None else np.asarray(time_source_s)
    z = 0.5
    distance = Planck18.luminosity_distance(z).to_value('cm')
    nu_obs = 5*fp.KEV_TO_ERG/fp.H_CGS/(1+z)
    records = []
    for angle in angles:
        row = pd.Series(dict(event_id=0, z=z, theta_v_rad=np.deg2rad(angle), theta_v_deg=angle,
                             E_iso_onaxis_erg=1e51))
        micro = pd.Series(dict(n_ism_cm3=9e-3, eps_B=0.055, Gamma0=500.,
                               D_L_cm=distance, E_kin_iso_erg=2e52))
        model = fp.afterglow_model(row, micro, theta_core_deg=3.4,
                                   resolutions=(0.1, 0.35, 10), jet_structure='tophat')
        vegas = np.asarray(model.flux_density_grid(time*(1+z), np.array([nu_obs])).total)[0]
        other_mjy = ag.fluxDensity(time*(1+z), np.full(len(time), nu_obs),
                                  jetType=ag.jet.TopHat, specType=ag.jet.SimpleSpec,
                                  thetaObs=np.deg2rad(angle), E0=2e52, thetaCore=np.deg2rad(3.4),
                                  n0=9e-3, p=2.2, epsilon_e=0.1, epsilon_B=0.055,
                                  xi_N=1., d_L=distance, z=z, g0=500., spread=False, counterjet=False)
        other = other_mjy*1e-26  # 1 mJy = 1e-26 erg cm^-2 s^-1 Hz^-1.
        records.append(pd.DataFrame(dict(time_source_s=time, theta_v_deg=angle,
                                         vegas_nuLnu=4*np.pi*distance**2*nu_obs*vegas,
                                         afterglowpy_nuLnu=4*np.pi*distance**2*nu_obs*other,
                                         vegas_to_afterglowpy=vegas/other)))
    return pd.concat(records, ignore_index=True)


def microphysics_comparison(angles=(0., 3.4, 7.5, 15.)):
    from astropy.cosmology import Planck18
    time = np.geomspace(1., 1e6, 160)
    z = 0.5
    distance = Planck18.luminosity_distance(z).to_value('cm')
    nu_obs = 5*fp.KEV_TO_ERG/fp.H_CGS/(1+z)
    records = []
    # Published intervals are represented by their midpoints, not random fits.
    cases = [('legacy example', 7.625e-3, 1e-3, 1e51, 10**2.3),
             ('FS-GW17', 2.5e-4, 1.3e-4, 2e52, 500.),
             ('FS-SGRB', 9e-3, 0.055, 2e52, 500.)]
    for preset, density, eps_b, energy, gamma in cases:
        for angle in angles:
            row = pd.Series(dict(z=z, theta_v_rad=np.deg2rad(angle), E_iso_onaxis_erg=1e51))
            micro = pd.Series(dict(n_ism_cm3=density, eps_B=eps_b, Gamma0=gamma,
                                   D_L_cm=distance, E_kin_iso_erg=energy))
            model = fp.afterglow_model(row, micro, theta_core_deg=3.4,
                                       resolutions=(0.1, 0.35, 10), jet_structure='powerlaw')
            fnu = np.asarray(model.flux_density_grid(time*(1+z), np.array([nu_obs])).total)[0]
            records.append(pd.DataFrame(dict(time_source_s=time, theta_v_deg=angle, preset=preset,
                                             nuLnu_5kev_erg_s=4*np.pi*distance**2*nu_obs*fnu)))
    return pd.concat(records, ignore_index=True)


def redshift_unit_check():
    """Hold physical source and distance fixed and vary only the redshift mapping."""
    time = np.array([1e3, 1e4, 1e5])
    distance = 1e28
    frames = []
    for z in (0., .5, 1.):
        row = pd.Series(dict(z=z, theta_v_rad=0., E_iso_onaxis_erg=1e51))
        micro = pd.Series(dict(n_ism_cm3=9e-3, eps_B=.055, Gamma0=500.,
                               D_L_cm=distance, E_kin_iso_erg=2e52))
        model = fp.afterglow_model(row, micro, theta_core_deg=3.4,
                                   resolutions=(.1, .35, 10), jet_structure='tophat')
        nu = 5*fp.KEV_TO_ERG/fp.H_CGS/(1+z)
        fnu = np.asarray(model.flux_density_grid(time*(1+z), np.array([nu])).total)[0]
        frames.append(pd.DataFrame(dict(redshift=z, time_source_s=time,
                                         nuLnu_5kev_erg_s=4*np.pi*distance**2*nu*fnu)))
    result = pd.concat(frames, ignore_index=True)
    reference = result.loc[result.redshift == 0, 'nuLnu_5kev_erg_s'].to_numpy()
    for _, group in result.groupby('redshift'):
        np.testing.assert_allclose(group.nuLnu_5kev_erg_s, reference, rtol=1e-8)
    return result


def render_checks(output_dir):
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    apply_forecast_style()
    solver = solver_comparison()
    micro = microphysics_comparison()
    redshift_unit_check().to_csv(directory/'afterglow_redshift_unit_check.csv', index=False)
    (directory/'validation_metadata.json').write_text(json.dumps({
        'kind': 'controlled model checks; not a population forecast',
        'python': platform.python_version(),
        'versions': {name: version(name) for name in ['VegasAfterglow', 'afterglowpy', 'numpy', 'astropy']},
        'time_frame': 'source', 'luminosity': 'nu Lnu at rest 5 keV [erg/s]',
        'solver_comparison': {'E_kin_iso_erg': 2e52, 'n_ism_cm3': .009, 'eps_B': .055,
                              'eps_e': .1, 'p': 2.2, 'theta_core_deg': 3.4,
                              'Gamma0': 500., 'spreading': False, 'redshift': .5},
        'ratio_time_window_s': [1e3, 1e5], 'ratio_luminosity_floor_erg_s': 1e38,
    }, indent=2)+'\n')
    solver.to_csv(directory/'afterglow_solver_comparison.csv', index=False)
    micro.to_csv(directory/'afterglow_microphysics_comparison.csv', index=False)
    norm, cmap = Normalize(0., 15.), plt.get_cmap('viridis')
    fig, ax = plt.subplots(figsize=(11., 7.2), layout='constrained')
    for angle, group in solver.groupby('theta_v_deg', sort=False):
        ax.loglog(group.time_source_s, group.vegas_nuLnu, color=cmap(norm(angle)), lw=2.5)
        ax.loglog(group.time_source_s, group.afterglowpy_nuLnu, color=cmap(norm(angle)), lw=2., ls='--')
    ax.text(.97, .96, 'Matched top-hat inputs\n'+r'$E_{\rm K,iso}=2\times10^{52}$ erg',
            transform=ax.transAxes, va='top', ha='right',
            bbox=dict(facecolor='white', edgecolor='none', alpha=.95, pad=2))
    ax.set_xlabel(r'Source time $t_{\rm src}$ [s]')
    ax.set_ylabel(r'$\nu L_\nu(5\,\mathrm{keV})\ [\mathrm{erg\,s}^{-1}]$')
    ax.set_ylim(1e38, 1e51)
    ax.legend(handles=[Line2D([], [], color='0.15', ls=ls, label=label)
                        for label, ls in [('VegasAfterglow', '-'), ('afterglowpy', '--')]], loc='lower left')
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=r'$\theta_{\rm v}$ [deg]')
    save_forecast_figure(fig, directory, 'afterglow_solver_comparison')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11., 7.2), layout='constrained')
    styles = {'legacy example': ':', 'FS-GW17': '--', 'FS-SGRB': '-'}
    for (preset, angle), group in micro.groupby(['preset', 'theta_v_deg'], sort=False):
        ax.loglog(group.time_source_s, group.nuLnu_5kev_erg_s,
                  color=cmap(norm(angle)), ls=styles[preset], lw=2.)
    ax.text(.97, .96, 'Controlled power-law jets\nForward shock only', transform=ax.transAxes,
            va='top', ha='right', bbox=dict(facecolor='white', edgecolor='none', alpha=.95, pad=2))
    ax.set_xlabel(r'Source time $t_{\rm src}$ [s]')
    ax.set_ylabel(r'$\nu L_\nu(5\,\mathrm{keV})\ [\mathrm{erg\,s}^{-1}]$')
    ax.set_ylim(1e38, 1e51)
    ax.legend(handles=[Line2D([], [], color='0.15', ls=ls, label=label)
                        for label, ls in styles.items()], loc='upper left', bbox_to_anchor=(.01, .98))
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=r'$\theta_{\rm v}$ [deg]')
    save_forecast_figure(fig, directory, 'afterglow_microphysics_comparison')
    plt.close(fig)
    # Ratios are diagnostics, not a forced bright-end normalization.
    selected = solver.loc[solver.time_source_s.between(1e3, 1e5) &
                          (solver.vegas_nuLnu >= 1e38) & (solver.afterglowpy_nuLnu >= 1e38)]
    summary = selected.groupby('theta_v_deg').vegas_to_afterglowpy.agg(['count', 'median', 'min', 'max'])
    summary = summary.reindex(solver.theta_v_deg.unique())
    summary['count'] = summary['count'].fillna(0).astype(int)
    summary.to_csv(directory/'afterglow_solver_ratios.csv')
    table_lines = ['| Viewing angle [deg] | Points | Median ratio | Minimum | Maximum |',
                   '| --- | --- | --- | --- | --- |']
    for angle, row in summary.iterrows():
        values = ['--' if pd.isna(row[key]) else f'{row[key]:.4g}' for key in ('median', 'min', 'max')]
        table_lines.append(f'| {angle:g} | {int(row["count"])} | '+ ' | '.join(values)+' |')
    table_markdown = '\n'.join(table_lines)
    (directory/'README.md').write_text(
        '# Controlled afterglow checks\n\n'
        'These curves use fixed published-style parameters, not a fitted population. '
        'Both figures show source-frame nu Lnu at rest 5 keV. '
        'Data below the displayed 1e38 erg/s limit remain in the full CSVs.\n\n'
        'Solver ratios use 1e3 <= t_source <= 1e5 s and require both luminosities '
        'to be at least 1e38 erg/s. An empty row means there are no points meeting '
        'that comparison selection, not solver agreement.\n\n'
        + table_markdown+'\n\n'
        'The on-axis normalization differs by several-fold; agreement is not established. '
        'Finite-Lorentz-factor dynamics and radiation prescriptions need further comparison '
        'before treating the bright end as calibrated. No empirical normalization was applied. '
        'Rest-frame luminosity invariance under the redshift mapping was checked at z=0, 0.5 and 1.\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir')
    args = parser.parse_args()
    print(render_checks(args.output_dir).to_string())
