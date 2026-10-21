"""Compact chapter tables with explicit rate and fraction denominators."""
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError
import hashlib
import json
import platform

import numpy as np
import pandas as pd


def chapter_tables(joint, afterglow, cut=100.):
    """Keep numeric fractions in [0,1]; NaN denotes an empty denominator."""
    localized = f'localized_le_{cut:g}_deg2'
    prompt = joint[['instrument', 'network', 'prompt_count', 'prompt_per_yr', 'count',
                    'joint_per_yr', 'joint_fraction_of_prompt',
                    'joint_fraction_of_prompt_lo68', 'joint_fraction_of_prompt_hi68',
                    localized+'_count', localized+'_per_yr', localized+'_fraction_of_joint']].copy()
    followup = afterglow[['prompt_instrument', 'network', 'imager', 'delay_s', 'prompt_count',
                         'joint_prompt_count', 'prompt_afterglow_count', 'prompt_afterglow_per_yr',
                         'afterglow_fraction_of_prompt', 'prompt_gw_afterglow_count',
                         'prompt_gw_afterglow_per_yr', 'afterglow_fraction_of_joint_prompt',
                         'afterglow_fraction_of_joint_prompt_lo68', 'afterglow_fraction_of_joint_prompt_hi68',
                         'localized_prompt_gw_afterglow_per_yr', 'localized_fraction_of_joint_afterglow']].copy()
    return prompt, followup


def export_chapter_tables(joint, afterglow, output_dir, cut=100.):
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    prompt, followup = chapter_tables(joint, afterglow, cut)
    prompt.to_csv(directory/'chapter_joint_rates_fractions.csv', index=False)
    followup.to_csv(directory/'chapter_afterglow_rates_fractions.csv', index=False)
    # A small publication table; full precision and Wilson limits remain in CSV.
    compact = prompt[['instrument', 'network', 'prompt_per_yr', 'joint_per_yr',
                      'joint_fraction_of_prompt', f'localized_le_{cut:g}_deg2_per_yr',
                      f'localized_le_{cut:g}_deg2_fraction_of_joint']]
    instrument_labels = {'GRINTA_TED': 'GRINTA/TED', 'Crystal_Eye': 'Crystal Eye',
                         'THESEUS_XGIS': 'THESEUS/XGIS', 'GRINTA_HXI': 'GRINTA/HXI',
                         'THESEUS_SXI': 'THESEUS/SXI'}
    network_labels = {'Asharp': r'$3A^\sharp$', 'ET_2L': 'ET-2L',
                      'ET_triangle': r'ET-$\Delta$', 'ET_2L_CE': 'ET-2L+CE'}
    compact = compact.replace(instrument_labels).replace(network_labels)
    compact.columns = ['Trigger', 'GW network', r'Prompt [yr$^{-1}$]', r'Joint [yr$^{-1}$]',
                       r'$f_{\mathrm{GW}|\gamma}$', r'Localized [yr$^{-1}$]',
                       r'$f_{\Omega|\gamma,\mathrm{GW}}$']
    compact.to_latex(directory/'chapter_joint_rates_fractions.tex', index=False, escape=False,
                     float_format='%.3g', na_rep='--')
    compact_afterglow = followup[['prompt_instrument', 'network', 'imager', 'delay_s',
              'prompt_gw_afterglow_per_yr', 'afterglow_fraction_of_joint_prompt',
              'localized_prompt_gw_afterglow_per_yr']].replace(instrument_labels).replace(network_labels)
    compact_afterglow.columns = ['Trigger', 'GW network', 'Imager', 'Delay [s]',
                                 r'Joint X-ray [yr$^{-1}$]', r'$f_{X|\gamma,\mathrm{GW}}$',
                                 r'Loc. X-ray [yr$^{-1}$]']
    compact_afterglow.to_latex(
                  directory/'chapter_afterglow_rates_fractions.tex', index=False,
                  float_format='%.3g', na_rep='--', longtable=True, escape=False)
    return prompt, followup


def publication_rate_comparison(prompt, joint):
    """Contextual benchmarks, not equality tests or a fitted normalization.

    2026 GRINTA Sec. 2.1: about 65 TED sGRBs/yr, about 2 joint/yr for
    3A-sharp, about 50 for ET-2L. Geometry/selection/model differences remain.
    """
    rows = []
    ted = prompt.loc[prompt.instrument == 'GRINTA_TED']
    if len(ted) == 1:
        rows.append({'selection': 'TED prompt', 'reference_per_yr': 65.,
                     'forecast_per_yr': float(ted.rate_per_yr.iloc[0])})
    for network, expected in [('Asharp', 2.), ('ET_2L', 50.)]:
        found = joint.loc[(joint.instrument == 'GRINTA_TED') & (joint.network == network)]
        if len(found) == 1:
            rows.append({'selection': 'TED + '+network, 'reference_per_yr': expected,
                         'forecast_per_yr': float(found.joint_per_yr.iloc[0])})
    result = pd.DataFrame(rows)
    if len(result):
        result['forecast_to_reference'] = result.forecast_per_yr/result.reference_per_yr
        result['reference'] = 'Natalucci et al. 2026, arXiv:2609.02802v2, Sec. 2.1'
    return result


def save_run_metadata(output_dir, settings, chain_path=None):
    """Record the actual exposure and model choices beside every catalogue."""
    data = dict(settings)
    data['python_version'] = platform.python_version()
    data['packages'] = {}
    for name in ('numpy', 'pandas', 'astropy', 'emcee', 'h5py', 'VegasAfterglow', 'GWFish'):
        try:
            data['packages'][name] = version(name)
        except PackageNotFoundError:
            data['packages'][name] = None
    if chain_path is not None:
        path = Path(chain_path)
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b''):
                digest.update(chunk)
        data['chain_path'] = str(path)
        data['chain_sha256'] = digest.hexdigest()
    data['input_sha256'] = {}
    for key in ('mrd_path', 'structure_csv', 'gw_config_template'):
        name = data.get(key)
        if name is not None and Path(name).is_file():
            data['input_sha256'][key] = hashlib.sha256(Path(name).read_bytes()).hexdigest()
    path = Path(output_dir)/'forecast_metadata.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x))+'\n')
    return path
