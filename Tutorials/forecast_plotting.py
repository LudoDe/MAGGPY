"""Consistent thesis figures for MAGGPY Tutorial 6.

Colours identify the prompt instrument; line styles identify the GW network.
All rates retain the original count / catalogue-duration normalization.
Only finite, positive Fisher sky areas enter localization selections.
"""
from pathlib import Path
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, LogLocator, FixedLocator


FONT_SIZE = 20
INSTRUMENT_COLOURS = {
    'GRINTA_TED': '#1f77b4',
    'Crystal_Eye': '#2ca02c',
    'THESEUS_XGIS': '#ff7f0e',
}
INSTRUMENT_LABELS = {
    'GRINTA_TED': 'GRINTA / TED',
    'Crystal_Eye': 'Crystal Eye',
    'THESEUS_XGIS': 'THESEUS / XGIS',
    'GRINTA_HXI': 'GRINTA / HXI',
    'THESEUS_SXI': 'THESEUS / SXI',
}
NETWORK_STYLES = {
    'Asharp': (0, (1.2, 2.2)),
    'ET_2L': '-',
    'ET_triangle': (0, (7, 3)),
    'ET_2L_CE': (0, (7, 2, 1.3, 2)),
}
NETWORK_LABELS = {
    'Asharp': r'$3A^{\sharp}$',
    'ET_2L': 'ET-2L',
    'ET_triangle': r'ET-$\Delta$',
    'ET_2L_CE': 'ET-2L + CE',
}
INSTRUMENT_MARKERS = {'GRINTA_TED': 'o', 'Crystal_Eye': 's', 'THESEUS_XGIS': '^'}
STAGE_STYLES = {'prompt': (0, (7, 3)), 'joint': '-',
                'localized': (0, (5, 2, 1.2, 2))}


def apply_forecast_style(font_size=FONT_SIZE):
    """MAGGPY serif typography, inward ticks, no grid or external titles."""
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
        'mathtext.fontset': 'dejavuserif', 'text.usetex': False,
        'font.size': font_size, 'axes.labelsize': font_size + 2,
        'xtick.labelsize': font_size - 2, 'ytick.labelsize': font_size - 2,
        'legend.fontsize': font_size - 2, 'axes.linewidth': 1.2,
        'axes.grid': False, 'axes.formatter.use_mathtext': True,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True,
        'xtick.major.size': 6, 'ytick.major.size': 6,
        'xtick.minor.size': 3, 'ytick.minor.size': 3,
        'xtick.major.width': 1.2, 'ytick.major.width': 1.2,
        'xtick.minor.width': 1.0, 'ytick.minor.width': 1.0,
        'lines.linewidth': 2.5, 'lines.markersize': 7,
        'legend.frameon': False, 'legend.handlelength': 3.0,
        'legend.borderaxespad': 0.5, 'legend.labelspacing': 0.35,
        'figure.dpi': 120, 'savefig.dpi': 300,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white', 'pdf.fonttype': 42, 'ps.fonttype': 42,
    })


def _names(items):
    return [item.name if hasattr(item, 'name') else str(item) for item in items]


def _label(name):
    return INSTRUMENT_LABELS.get(name, NETWORK_LABELS.get(name, name.replace('_', ' ')))


def _colour(name):
    if name not in INSTRUMENT_COLOURS:
        raise ValueError(f'Choose a fixed colour for instrument {name!r} in INSTRUMENT_COLOURS.')
    return INSTRUMENT_COLOURS[name]


def _network_style(name):
    if name not in NETWORK_STYLES:
        raise ValueError(f'Choose a line style for network {name!r} in NETWORK_STYLES.')
    return NETWORK_STYLES[name]


def _years(years):
    if not np.isfinite(years) or years <= 0:
        raise ValueError('The catalogue duration must be positive and finite.')
    return float(years)


def _bool_values(series):
    """Read booleans safely from both a live catalogue and a CSV."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).to_numpy(dtype=bool)
    converted = series.map({True: True, False: False, 1: True, 0: False,
                            'True': True, 'False': False, 'true': True, 'false': False})
    if (converted.isna() & series.notna()).any():
        raise ValueError(f'Unrecognised boolean values in {series.name!r}.')
    return converted.fillna(False).to_numpy(dtype=bool)


def _selections(events, gw_results, instrument, network, cut):
    if not events.index.equals(gw_results.index):
        raise ValueError('Prompt and GW catalogues must have matching row indices.')
    prompt = _bool_values(events[f'prompt_{instrument}'])
    joint = prompt & _bool_values(gw_results[f'gw_{network}'])
    area = gw_results[f'area90_{network}_deg2'].to_numpy(dtype=float)
    localized = joint & np.isfinite(area) & (area > 0) & (area <= cut)
    return prompt, joint, localized


def _format_ax(ax):
    ax.grid(False, which='both')
    ax.tick_params(which='both', direction='in', top=True, right=True)
    if ax.get_xscale() == 'linear' and not isinstance(ax.xaxis.get_major_locator(), FixedLocator):
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    if ax.get_yscale() == 'linear':
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
    ax.set_title('')


def _panel(ax, text, letter=None):
    prefix = f'({letter}) ' if letter else ''
    return ax.text(0.04, 0.96, prefix + text, transform=ax.transAxes,
                   va='top', ha='left', fontsize=plt.rcParams['font.size'],
                   bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.9, 'pad': 1.5},
                   zorder=20)


def _instrument_handles(names):
    return [Line2D([], [], color=_colour(n), lw=3, label=_label(n)) for n in names]


def _network_handles(networks):
    return [Line2D([], [], color='0.15', lw=2.5, linestyle=_network_style(n),
                   label=_label(n)) for n in networks]


def _stage_labels(snr_threshold, cut):
    return ['Prompt', rf'Prompt + GW ($\rho>{snr_threshold:g}$)',
            rf'+ $\Delta\Omega_{{90}}\leq {cut:g}\,\mathrm{{deg}}^2$']


def save_forecast_figure(fig, output_dir, stem):
    """Export the same layout to vector PDF and 300-dpi PNG."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for ax in fig.axes:
        _format_ax(ax)
    fig.savefig(directory / f'{stem}.pdf', bbox_inches='tight', pad_inches=0.05)
    fig.savefig(directory / f'{stem}.png', bbox_inches='tight', pad_inches=0.05)
    return fig


def plot_prompt_distributions(events, instruments, years, theta_v_max=30.):
    years = _years(years)
    names = _names(instruments)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), layout='constrained')
    z = events.z.to_numpy(dtype=float)
    zmax = max(0.1, np.max(z[np.isfinite(z)], initial=0))
    bins = {'z': np.linspace(0, zmax, 25),
            'theta_v_deg': np.linspace(0, theta_v_max, 25)}
    for name in names:
        selected = _bool_values(events[f'prompt_{name}'])
        for ax, column in zip(axes, ('z', 'theta_v_deg')):
            counts, edges = np.histogram(events.loc[selected, column], bins=bins[column])
            ax.stairs(counts / years, edges, color=_colour(name), label=_label(name))
    for i, ax in enumerate(axes):
        ax.set_ylim(bottom=0)
        ax.margins(y=0.3)
        ax.set_ylabel(r'$\dot N\ [\mathrm{yr}^{-1}\,\mathrm{bin}^{-1}]$')
        _panel(ax, 'Prompt detections', chr(97+i))
    axes[0].set_xlabel(r'Redshift $z$')
    axes[1].set_xlabel(r'Viewing angle $\theta_{\rm v}$ [deg]')
    axes[1].legend(loc='upper right', bbox_to_anchor=(0.99, 0.84))
    return fig


def plot_joint_detection_stages(table, instruments, networks, cut=100., snr_threshold=8.):
    names, networks = _names(instruments), list(networks)
    fig, axes = plt.subplots(1, len(names), figsize=(6.1*len(names), 5.8),
                             sharey=True, squeeze=False, layout='constrained')
    labels = _stage_labels(snr_threshold, cut)
    face_alphas, hatches = (0.12, 0.45, 0.9), ('//', '', '..')
    maximum = float(table.prompt_per_yr.max()) if len(table) else 0.
    ymax = max(maximum*1.42, 1.)
    for i, (ax, name) in enumerate(zip(axes[0], names)):
        rows = table.loc[table.instrument == name].set_index('network').loc[networks]
        stages = [rows.prompt_per_yr, rows.rate_per_yr,
                  rows[f'localized_le_{cut:g}_deg2_per_yr']]
        x = np.arange(len(networks))
        for offset, rates, alpha, hatch in zip((-0.26, 0, 0.26), stages, face_alphas, hatches):
            from matplotlib.colors import to_rgba
            bars = ax.bar(x+offset, rates, width=0.24,
                          facecolor=to_rgba(_colour(name), alpha), edgecolor=_colour(name),
                          hatch=hatch, linewidth=1.3)
            ax.bar_label(bars, labels=[f'{v:g}' for v in rates],
                         padding=4, fontsize=plt.rcParams['font.size']-2, rotation=90)
        ax.set_xticks(x, [_label(n) for n in networks])
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.set_ylim(0, ymax)
        _panel(ax, _label(name), chr(97+i))
    axes[0, 0].set_ylabel(r'Detection rate $[\mathrm{yr}^{-1}]$')
    handles = [Patch(facecolor=(0.3, 0.3, 0.3, a), edgecolor='0.2', hatch=h, label=l)
               for a, h, l in zip(face_alphas, hatches, labels)]
    axes[0, -1].legend(handles=handles, loc='upper right', bbox_to_anchor=(0.99, 0.82))
    return fig


def plot_redshift_localization(events, gw_results, instruments, networks, years,
                               cut=100., snr_threshold=8.):
    """Staged annual histograms inspired by Ronchini et al. Figs. 2, 3 and 5.

    These are prompt-selected events. They do not represent an all-BNS GW sample.
    """
    years = _years(years)
    names, networks = _names(instruments), list(networks)
    z = events.z.to_numpy(dtype=float)
    zmax = max(0.1, np.max(z[np.isfinite(z)], initial=0) + 0.1)
    edges = np.linspace(0, zmax, 30)
    fig, axes = plt.subplots(len(networks), len(names),
                             figsize=(6.1*len(names), 4.5*len(networks)),
                             sharex=True, sharey='row', squeeze=False, layout='constrained')
    row_peaks = np.zeros(len(networks))
    for row, network in enumerate(networks):
        for col, name in enumerate(names):
            ax = axes[row, col]
            masks = _selections(events, gw_results, name, network, cut)
            for stage, mask in zip(STAGE_STYLES, masks):
                counts, _ = np.histogram(z[mask], bins=edges)
                row_peaks[row] = max(row_peaks[row], counts.max(initial=0)/years)
                ax.stairs(counts / years, edges, color=_colour(name),
                          linestyle=STAGE_STYLES[stage], linewidth=2.5)
            _panel(ax, _label(name)+'\n'+_label(network), chr(97+row*len(names)+col))
            if row == len(networks)-1:
                ax.set_xlabel(r'Redshift $z$')
        axes[row, 0].set_ylabel(r'$\dot N\ [\mathrm{yr}^{-1}\,\mathrm{bin}^{-1}]$')
        for ax in axes[row]:
            ax.set_ylim(0, max(row_peaks[row]*1.45, 1./years))
    handles = [Line2D([], [], color='0.15', lw=2.5, linestyle=ls, label=label)
               for ls, label in zip(STAGE_STYLES.values(), _stage_labels(snr_threshold, cut))]
    axes[0, -1].legend(handles=handles, loc='upper right', bbox_to_anchor=(0.99, 0.73))
    return fig


def cumulative_localization(area, years):
    """Exact empirical cumulative rate; no smoothing, binning, or extrapolation."""
    years = _years(years)
    values = np.asarray(area, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    thresholds, counts = np.unique(values, return_counts=True)
    return thresholds, np.cumsum(counts)/years


def plot_localization_cumulative(events, gw_results, instruments, networks, years):
    names, networks = _names(instruments), list(networks)
    fig, ax = plt.subplots(figsize=(11.5, 7.1), layout='constrained')
    curves = []
    for name in names:
        for network in networks:
            _, joint, _ = _selections(events, gw_results, name, network, np.inf)
            thresholds, rates = cumulative_localization(
                gw_results.loc[joint, f'area90_{network}_deg2'], years)
            curves.append((name, network, thresholds, rates))
    nonempty = [x for _, _, x, _ in curves if len(x)]
    if nonempty:
        xmin = min(x[0] for x in nonempty)/1.4
        xmax = max(x[-1] for x in nonempty)*1.4
        for name, network, x, y in curves:
            if len(x):
                # Extend the zero baseline and measured plateau to common bounds.
                ax.step(np.r_[xmin, x, xmax], np.r_[0., y, y[-1]], where='post',
                        color=_colour(name), linestyle=_network_style(network), lw=2.6)
        ax.set_xlim(xmin, xmax)
    else:
        ax.set_xlim(1., 1e4)
        ax.text(0.5, 0.3, 'No valid joint localizations', transform=ax.transAxes, ha='center')
    ax.set_xscale('log')
    ax.set_ylim(bottom=0)
    ax.margins(y=0.08)
    ax.set_xlabel(r'90% GW sky area $\Delta\Omega_{90}$ [deg$^2$]')
    ax.set_ylabel(r'$\dot N(\leq\Delta\Omega_{90})\ [\mathrm{yr}^{-1}]$')
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=8))
    instrument_legend = ax.legend(handles=_instrument_handles(names), loc='upper left',
                                  bbox_to_anchor=(0.02, 0.98))
    ax.add_artist(instrument_legend)
    ax.legend(handles=_network_handles(networks), loc='upper left',
              bbox_to_anchor=(0.02, 0.70))
    ax.text(0.97, 0.06, 'Prompt + GW', transform=ax.transAxes, ha='right', va='bottom')
    return fig


def plot_joint_viewing_angles(events, gw_results, instruments, networks, years):
    """Network comparison of the angular selection, following Fig. 4's idea.

    Annual rates for the fitted catalogue, without posterior uncertainty bands.
    """
    years = _years(years)
    names, networks = _names(instruments), list(networks)
    angles = events.theta_v_deg.to_numpy(dtype=float)
    valid = angles[np.isfinite(angles) & (angles >= 0)]
    xmax = max(1., np.ceil(valid.max(initial=0)+0.1))
    edges = np.linspace(0, xmax, 25)
    fig, axes = plt.subplots(1, len(names), figsize=(6.1*len(names), 5.6),
                             sharex=True, sharey=True, squeeze=False, layout='constrained')
    peak = 0.
    for i, (ax, name) in enumerate(zip(axes[0], names)):
        for network in networks:
            _, joint, _ = _selections(events, gw_results, name, network, np.inf)
            counts, _ = np.histogram(angles[joint], bins=edges)
            peak = max(peak, counts.max(initial=0)/years)
            ax.stairs(counts/years, edges, color=_colour(name),
                      linestyle=_network_style(network), lw=2.5)
        _panel(ax, _label(name)+'\nPrompt + GW', chr(97+i))
        ax.set_xlabel(r'Viewing angle $\theta_{\rm v}$ [deg]')
    for ax in axes[0]:
        ax.set_ylim(0, max(peak*1.35, 1./years))
    axes[0, 0].set_ylabel(r'$\dot N\ [\mathrm{yr}^{-1}\,\mathrm{bin}^{-1}]$')
    axes[0, -1].legend(handles=_network_handles(networks), loc='upper right',
                       bbox_to_anchor=(0.99, 0.80))
    return fig


def plot_afterglow_rates(table, instruments, networks, imagers=None):
    """Separate networks to avoid nine overlapping colour-cycled curves."""
    names, networks = _names(instruments), list(networks)
    imagers = list(pd.unique(table.imager)) if imagers is None else _names(imagers)
    fig, axes = plt.subplots(len(networks), len(imagers),
                             figsize=(7.2*len(imagers), 4.5*len(networks)),
                             sharex=True, sharey='row', squeeze=False, layout='constrained')
    for row, network in enumerate(networks):
        peak = 0.
        for col, imager in enumerate(imagers):
            ax = axes[row, col]
            panel_peak = 0.
            for name in names:
                group = table.loc[(table.imager == imager) & (table.network == network) &
                                  (table.prompt_instrument == name)].sort_values('delay_s')
                if group.empty:
                    continue
                rates = group.prompt_gw_afterglow_per_yr.to_numpy(dtype=float)
                peak = max(peak, rates.max(initial=0))
                panel_peak = max(panel_peak, rates.max(initial=0))
                ax.plot(group.delay_s, rates, color=_colour(name), linestyle=_network_style(network),
                        marker=INSTRUMENT_MARKERS.get(name, 'o'), markerfacecolor='white',
                        markeredgewidth=1.6, lw=2.5)
            ax.set_xscale('symlog', linthresh=10)
            _panel(ax, _label(imager)+'\n'+_label(network), chr(97+row*len(imagers)+col))
            if panel_peak == 0:
                ax.text(0.5, 0.35, 'No joint afterglows', transform=ax.transAxes,
                        ha='center', va='center', fontsize=plt.rcParams['font.size']-2)
            if row == len(networks)-1:
                ax.set_xlabel(r'Response delay $t_{\rm resp}$ [s]')
        axes[row, 0].set_ylabel(r'$\dot N_{\gamma+\mathrm{GW}+X}\ [\mathrm{yr}^{-1}]$')
        for ax in axes[row]:
            ax.set_ylim(-max(peak, 0.1)*0.06, max(peak*1.45, 0.1))
    axes[0, -1].legend(handles=_instrument_handles(names), loc='upper right',
                       bbox_to_anchor=(0.99, 0.74))
    return fig


def _saved_light_curves(directory, events):
    """Reject stale curve files before writing any replacement figures."""
    path = directory/'xray_light_curves.csv'
    if not path.exists():
        return None, None
    curves = pd.read_csv(path)
    if not len(curves):
        return curves, None
    if not curves.event_id.isin(events.event_id).all():
        raise ValueError('The light curves contain event IDs from a different forecast.')
    reference = events.set_index('event_id').loc[curves.event_id]
    if not np.allclose(reference.z, curves.redshift) or not np.allclose(
            reference.theta_v_deg, curves.theta_v_deg):
        raise ValueError('The light curves and event catalogue are from different forecasts.')
    if not np.allclose(curves.time_observer_s, curves.time_source_s*(1+curves.redshift)):
        raise ValueError('The light-curve observer/source time mapping is inconsistent.')
    metadata_path = directory/'forecast_metadata.json'
    imagers = None
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        for column, key in [('preset', 'afterglow_preset'), ('jet_structure', 'afterglow_jet')]:
            if key in metadata and not curves[column].eq(metadata[key]).all():
                raise ValueError('The light-curve model does not match the saved run metadata.')
        if metadata.get('afterglow_imagers'):
            imagers = [SimpleNamespace(**item) for item in metadata['afterglow_imagers']]
    return curves, imagers


def replot_saved_forecasts(output_dir, years, instruments=None, networks=None,
                           theta_v_max=30., cut=100., snr_threshold=8.):
    """Redraw exported CSVs without importing MAGGPY, GWFish or VegasAfterglow.

    `years` must be the physical duration used to generate these CSVs.
    Files are rewritten only after all inputs have been loaded and checked.
    """
    years = _years(years)
    directory = Path(output_dir)
    events = pd.read_csv(directory/'catalogue_prompt.csv')
    combined = pd.read_csv(directory/'events_prompt_gw.csv')
    joint_table = pd.read_csv(directory/'joint_prompt_rates.csv')
    afterglow_table = pd.read_csv(directory/'afterglow_rates.csv')
    names = list(pd.unique(joint_table.instrument)) if instruments is None else _names(instruments)
    networks = list(pd.unique(joint_table.network)) if networks is None else list(networks)
    if joint_table[['instrument', 'network']].duplicated().any():
        raise ValueError('The joint table has duplicate instrument/network rows.')
    if not np.allclose(joint_table['count']/years, joint_table.rate_per_yr):
        raise ValueError('years does not match the annual rates in joint_prompt_rates.csv.')
    if not np.allclose(joint_table.prompt_count/years, joint_table.prompt_per_yr):
        raise ValueError('years does not match the prompt rates in joint_prompt_rates.csv.')
    if afterglow_table[['prompt_instrument', 'network', 'imager', 'delay_s']].duplicated().any():
        raise ValueError('The afterglow table has duplicate scenario rows.')
    if not np.allclose(afterglow_table.prompt_gw_afterglow_count/years,
                       afterglow_table.prompt_gw_afterglow_per_yr):
        raise ValueError('years does not match the annual rates in afterglow_rates.csv.')
    required_cut = f'localized_le_{cut:g}_deg2_per_yr'
    if required_cut not in joint_table:
        raise ValueError(f'No exported rates for localization cut {cut:g} deg^2.')
    if events.event_id.duplicated().any() or combined.event_id.duplicated().any():
        raise ValueError('Each event catalogue must have unique event_id values.')
    if not combined.event_id.isin(events.event_id).all():
        raise ValueError('The GW catalogue contains event IDs absent from the prompt catalogue.')
    reference = events.set_index('event_id').loc[combined.event_id]
    if not np.allclose(reference.z, combined.z) or not np.allclose(
            reference.theta_v_deg, combined.theta_v_deg):
        raise ValueError('The prompt and GW catalogues are from different runs.')
    for name in names:
        _colour(name)
        if not np.array_equal(_bool_values(reference[f'prompt_{name}']),
                              _bool_values(combined[f'prompt_{name}'])):
            raise ValueError('The prompt flags in the two catalogues differ.')
        for network in networks:
            _network_style(network)
            snr_column = f'snr_{network}'
            if snr_column in combined and not np.array_equal(
                    combined[snr_column].to_numpy(dtype=float) > snr_threshold,
                    _bool_values(combined[f'gw_{network}'])):
                raise ValueError('snr_threshold does not match the stored GW detection flags.')
            prompt, joint, localized = _selections(combined, combined, name, network, cut)
            rows = joint_table.loc[(joint_table.instrument == name) & (joint_table.network == network)]
            if len(rows) != 1:
                raise ValueError(f'Missing joint summary for {name} / {network}.')
            row = rows.iloc[0]
            if not np.allclose([prompt.sum()/years, joint.sum()/years, localized.sum()/years],
                               [row.prompt_per_yr, row.rate_per_yr, row[required_cut]]):
                raise ValueError('The joint rates and event catalogue are from different runs.')
    curves, curve_imagers = _saved_light_curves(directory, combined)
    apply_forecast_style()
    makers = [
        ('prompt_distributions', lambda: plot_prompt_distributions(events, names, years, theta_v_max)),
        ('joint_detection_stages', lambda: plot_joint_detection_stages(joint_table, names, networks, cut, snr_threshold)),
        ('redshift_localization_cuts', lambda: plot_redshift_localization(combined, combined, names, networks, years, cut, snr_threshold)),
        ('joint_localization_cumulative', lambda: plot_localization_cumulative(combined, combined, names, networks, years)),
        ('joint_viewing_angle_distributions', lambda: plot_joint_viewing_angles(combined, combined, names, networks, years)),
        ('afterglow_rates', lambda: plot_afterglow_rates(afterglow_table, names, networks)),
    ]
    paths = []
    for stem, make in makers:
        fig = make()
        save_forecast_figure(fig, directory, stem)
        plt.close(fig)
        paths.extend([directory/f'{stem}.pdf', directory/f'{stem}.png'])
    if curves is not None and len(curves):
        for stem, fig in [('xray_light_curves', plot_xray_light_curves(curves, curve_imagers)),
                          ('xray_luminosity_5kev', plot_xray_luminosities(curves))]:
            save_forecast_figure(fig, directory, stem)
            plt.close(fig)
            paths.extend([directory/f'{stem}.pdf', directory/f'{stem}.png'])
    return paths


def _angle_colours(curves):
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    maximum = max(1., curves.theta_v_deg.max())
    norm = Normalize(0, maximum)
    cmap = plt.get_cmap('viridis')
    return cmap, norm, ScalarMappable(norm=norm, cmap=cmap)


def plot_xray_light_curves(curves, instruments=None):
    """Observer-frame band fluxes, coloured by viewing angle (continuous scale).

    The colourbar labels the deliberate exception to the instrument palette.
    Horizontal lines are exposure sensitivities, not a grid or light curves.
    """
    names = list(pd.unique(curves.imager)) if instruments is None else _names(instruments)
    names = [name for name in names if name != 'Swift_XRT']
    fig, axes = plt.subplots(1, len(names), figsize=(7.3*len(names), 6.),
                             squeeze=False, layout='constrained')
    cmap, norm, scalar = _angle_colours(curves)
    for i, (ax, name) in enumerate(zip(axes[0], names)):
        selected = curves.loc[curves.imager == name]
        for event_id, group in selected.groupby('event_id', sort=False):
            group = group.sort_values('time_observer_s')
            ax.loglog(group.time_observer_s, group.band_flux,
                      color=cmap(norm(group.theta_v_deg.iloc[0])), alpha=0.65, lw=1.3)
        if instruments is not None:
            imager = next(item for item in instruments if item.name == name)
            for j, (dt, limit) in enumerate(zip(imager.exposures_s, imager.limits)):
                if dt not in (100., 1000., 1500.):
                    continue
                ax.axhline(limit, color='0.2', linestyle=['--', ':', '-.'][j%3], lw=1.5,
                           label=f'{dt:g} s exposure')
            ax.legend(loc='lower left')
        ax.set_xlabel(r'Observer time $t_{\rm obs}$ [s]')
        unit = selected.flux_unit.iloc[0]
        ylabel = (r'$F_X\ [\mathrm{erg\,cm}^{-2}\,\mathrm{s}^{-1}]$' if unit == 'erg'
                  else r'$\Phi_X\ [\mathrm{ph\,cm}^{-2}\,\mathrm{s}^{-1}]$')
        ax.set_ylabel(ylabel)
        ax.text(.97, .96, f'({chr(97+i)}) '+_label(name)+'\nForward shock only',
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(facecolor='white', edgecolor='none', alpha=.9, pad=1.5))
    fig.colorbar(scalar, ax=list(axes[0]), label=r'Viewing angle $\theta_{\rm v}$ [deg]', pad=0.02)
    return fig


def plot_xray_luminosities(curves):
    """Source-frame nu L_nu at rest 5 keV: the ordinate used in Ronchini Fig. 1.

    This is forward shock only; no synthetic HLE or empirical bright-end scaling.
    """
    fig, ax = plt.subplots(figsize=(10.5, 7.), layout='constrained')
    cmap, norm, scalar = _angle_colours(curves)
    first_imager = curves.imager.iloc[0]
    for _, group in curves.loc[curves.imager == first_imager].groupby('event_id', sort=False):
        group = group.sort_values('time_source_s')
        ax.loglog(group.time_source_s, group.nuLnu_5kev_erg_s,
                  color=cmap(norm(group.theta_v_deg.iloc[0])), alpha=0.65, lw=1.3)
    ax.set_xlabel(r'Source time $t_{\rm src}$ [s]')
    ax.set_ylabel(r'$\nu L_\nu(5\,\mathrm{keV})\ [\mathrm{erg\,s}^{-1}]$')
    ax.text(.97, .96, 'Forward shock only', transform=ax.transAxes, ha='right', va='top',
            bbox=dict(facecolor='white', edgecolor='none', alpha=.9, pad=1.5))
    fig.colorbar(scalar, ax=ax, label=r'Viewing angle $\theta_{\rm v}$ [deg]', pad=0.02)
    return fig
