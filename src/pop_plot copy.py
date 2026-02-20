"""
pop_plot.py
==========================
Centralized module for GW population analysis, data preprocessing, and visualization.

Usage:
    from src.population_plots import PopulationAnalysis
    
    analysis = PopulationAnalysis(output_folder="Output_files/ProductionPop_fj10_20k")
    analysis.run_preprocessing()
    analysis.plot_fj_vs_rates_comparison()
"""

import re
import json
import emcee
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
from scipy.special import gammaln
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from IPython.display import clear_output
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.ticker import LogLocator, ScalarFormatter

#def summarize_samples(samples: np.ndarray) -> np.ndarray:
"""
Returns array of shape (n_params, 3) -> [median, lower_err, upper_err]
"""
#    p5, p50, p95 = np.percentile(samples, [5, 50, 95], axis=0)
#    return np.column_stack([p50, p50 - p5, p95 - p50])


#def load_json_dict(file_path: Path) -> dict:
"""Load JSON dictionary from file."""
#    with open(file_path, "r") as f:
#        return json.load(f)


#def clean_model_name(sm: str) -> str:
"""Extract clean model name from sample name."""
#    mod = re.sub(r"_A\d+\.\d+", "", sm)
#    return re.sub(r"^samples_", "", mod)


def draw_vertical_violin(ax, data, x_pos, y_base, color, width=0.1, min_val=0, max_val=10):
    """Draws a vertical violin plot for a given distribution."""
    if len(data) < 5: return
    
    try:
        kde = gaussian_kde(data)
    except: return # Handle singular matrices

    grid = np.linspace(min_val, max_val, 100)
    density = kde(grid)
    if np.max(density) > 0:
        density_norm = density / np.max(density) * width
    else:
        density_norm = np.zeros_like(density)
    
    # Map grid to Y coordinates
    y_positions = y_base + np.interp(grid, [min_val, max_val], [0, 0.9])
    
    ax.fill_betweenx(y_positions, x_pos - density_norm, x_pos + density_norm,
                     facecolor=color, edgecolor="black", lw=0.5, alpha=0.7, zorder=2)
    
    # Draw Median
    median = np.median(data)
    median_y = y_base + np.interp(median, [min_val, max_val], [0, 0.9])
    idx = np.searchsorted(grid, median)
    if idx < len(density_norm):
        width_at_med = density_norm[idx] * 0.9
        ax.plot([x_pos - width_at_med, x_pos + width_at_med], [median_y, median_y], 
                color="k", lw=2.3, zorder=3)

def load_json_dict(file_path: Path) -> dict:
    with open(file_path, "r") as f: return json.load(f)

# =============================================================================
# DEFAULT CONFIGURATIONS
# =============================================================================

DEFAULT_ALPHA_KEYS = ["A0.5", "A1.0", "A3.0", "A5.0"]

DEFAULT_ALPHA_COLORS = {
    "A5.0": "sandybrown",
    "A3.0": "palevioletred",
    "A1.0": "blueviolet",
    "A0.5": "darkblue",
}

DEFAULT_ALPHA_LABELS = {
    "A5.0": r"$\alpha_{\rm CE} = 5.0$",
    "A3.0": r"$\alpha_{\rm CE} = 3.0$",
    "A1.0": r"$\alpha_{\rm CE} = 1.0$",
    "A0.5": r"$\alpha_{\rm CE} = 0.5$",
}

PARAM_LABELS = [
    "k", r"$\log_{10}(\frac{E^*}{10^{49}erg})$", r"$\log_{10}(\frac{\mu_E}{keV})$",
    r"$\sigma_{E}$", r"$\log_{10}(\frac{\mu_{\tau}}{s})$", r"$\sigma_{\tau}$", r"$f_j$"
]


# =============================================================================
# CONFIGURATION DATACLASS
# =============================================================================

@dataclass
class PopulationConfig:
    """Configuration for population analysis."""
    # Paths
    datafiles: Path = field(default_factory=lambda: Path("datafiles"))
    output_folder: Path = field(default_factory=lambda: Path("Output_files/ProductionPop_fj10_20k"))
    images_folder: Path = field(default_factory=lambda: Path("tutorial_images/populations"))
    
    # Analysis parameters
    discard: int = 10_000
    thin: int = 15
    yearly_rate: float = 18.61
    total_years: float = 16.6585
    f_max: float = 10
    k_params: int = 7
    
    # Alpha keys
    alpha_keys: List[str] = field(default_factory=lambda: DEFAULT_ALPHA_KEYS.copy())
    
    # Styling
    alpha_colors: Dict[str, str] = field(default_factory=lambda: DEFAULT_ALPHA_COLORS.copy())
    alpha_labels: Dict[str, str] = field(default_factory=lambda: DEFAULT_ALPHA_LABELS.copy())
    base_fontsize: int = 14
    
    def __post_init__(self):
        self.datafiles = Path(self.datafiles)
        self.output_folder = Path(self.output_folder)
        self.images_folder = Path(self.images_folder)
        self.population_folder = self.datafiles / "populations" / "samples"


# =============================================================================
# DATA PREPROCESSING
# =============================================================================

class DataPreprocessor:
    """Handles comprehensive data preprocessing from MCMC chains."""
    
    def __init__(self, config: PopulationConfig):
        self.config = config
        
    def preprocess(
        self, 
        samp_names: List[str] = None, 
        local_rates_dict: dict = None,
        quiet: bool = False
    ) -> Tuple[dict, dict, dict, dict, dict]:
        """
        Comprehensive data preprocessing that extracts all necessary data in a single pass.
        
        Returns:
            tuple: (processed_data, alpha_fj_dict, fj_samples_by_alpha, alpha_dict, alpha_params_dict)
        """
        if samp_names is None:
            samples = list(self.config.population_folder.glob("samples_*.dat"))
            samp_names = [s.name.split("samples_")[1].split("_BNSs")[0] for s in samples]
        
        # Initialize all data structures
        processed_data = {alpha: {} for alpha in self.config.alpha_keys}
        alpha_fj_dict = {alpha: {} for alpha in self.config.alpha_keys}
        fj_samples_by_alpha = {alpha: [] for alpha in self.config.alpha_keys}
        alpha_dict = {alpha: {} for alpha in self.config.alpha_keys}
        alpha_params_dict = {alpha: {} for alpha in self.config.alpha_keys}
        
        total_samples = len(samp_names)
        if not quiet:
            print("Starting comprehensive data preprocessing...")
        
        for i, sm in enumerate(samp_names):
            if not quiet:
                clear_output(wait=True)
                progress = (i + 1) / total_samples * 100
                progress_bar = f"[{'=' * int(progress // 2)}{' ' * (50 - int(progress // 2))}]"
                print(f"Processing: {progress_bar} {progress:.1f}%")
                print(f"Sample {i+1}/{total_samples}: {sm}")
            
            # Find matching folder
            matching_list = list(self.config.output_folder.glob(f"{sm}*"))
            if not matching_list:
                print(f"Warning: No matching folder found for sample {sm}. Skipping.")
                continue
            matching = matching_list[0]
            
            # Process each alpha
            for alpha in self.config.alpha_keys:
                if alpha not in matching.name:
                    continue
                
                stats_file = matching / "emcee.h5"
                alpha_dict[alpha][sm] = stats_file
                
                if not stats_file.is_file():
                    continue
                
                try:
                    self._process_sample(
                        sm, alpha, stats_file, local_rates_dict,
                        processed_data, alpha_fj_dict, fj_samples_by_alpha, alpha_params_dict
                    )
                except Exception as e:
                    if not quiet:
                        print(f"Error processing {sm} for {alpha}: {e}")
                    continue
        
        if not quiet:
            print(f"Processing: [{'=' * 50}] 100.0%")
            print(f"Completed preprocessing of {total_samples} samples!")
        
        return processed_data, alpha_fj_dict, fj_samples_by_alpha, alpha_dict, alpha_params_dict
    
    def _process_sample(
        self, sm: str, alpha: str, stats_file: Path, local_rates_dict: dict,
        processed_data: dict, alpha_fj_dict: dict, fj_samples_by_alpha: dict, alpha_params_dict: dict
    ):
        """Process a single sample file."""
        backend = emcee.backends.HDFBackend(stats_file.as_posix(), read_only=True)
        samples = backend.get_chain(discard=self.config.discard, thin=self.config.thin, flat=True)
        # return a warning if samples is [0, n_params] as discard is too high
        if samples.shape[0] == 0:
            print(f"Warning: No samples left after discarding for {sm} at {alpha}. Skipping.")
            return
        blobs = backend.get_blobs(discard=self.config.discard, thin=self.config.thin, flat=True)
        
        # Apply local rate filtering if needed
        if local_rates_dict is not None:
            model_key = clean_model_name(sm)
            if alpha in local_rates_dict and model_key in local_rates_dict[alpha]:
                rate = local_rates_dict[alpha][model_key]
                if rate < 1:
                    mask = samples[:, -1] >= 1
                    samples = samples[mask]
                    blobs = blobs[mask]
                    if len(samples) == 0:
                        return
        
        samples_res = summarize_samples(samples)
        fj_summary = samples_res[-1]
        
        # Store in alpha_fj_dict
        alpha_fj_dict[alpha][sm] = fj_summary
        
        # Store raw samples
        fj_posterior = samples[:, -1]
        fj_samples_by_alpha[alpha].append(fj_posterior)
        
        # Store parameter summaries
        for idx, label in enumerate(PARAM_LABELS[:-1]):  # Exclude f_j
            alpha_params_dict[alpha].setdefault(label, []).append(samples_res[idx])
        
        if blobs.size <= 0:
            return
        
        n_rate_yr, p_epeak, p_t90, p_pflux, p_fluence = blobs.T
        
        log_likelihood_posterior = (
            p_epeak + p_t90 + p_pflux + p_fluence +
            np.array([poiss_log(n, self.config.yearly_rate) for n in n_rate_yr])
        )
        
        total_events = n_rate_yr * self.config.total_years
        mask_fj_physical = fj_posterior < 1
        total_events_physical = total_events[mask_fj_physical] if mask_fj_physical.any() else np.array([])
        
        model_key = clean_model_name(sm)
        processed_data[alpha][model_key] = {
            "fj_summary": fj_summary,
            "fj_posterior": fj_posterior,
            "mean_log_likelihood": np.mean(log_likelihood_posterior),
            "mean_ks_log_p": {
                "E_peak": np.mean(p_epeak),
                "T_90": np.mean(p_t90),
                "F_peak": np.mean(p_pflux),
                "Fluence": np.mean(p_fluence),
            },
            "total_events_posterior": total_events,
            "total_events_physical": total_events_physical,
            "mean_rate": np.median(n_rate_yr),
            "simulated_rate_posterior": n_rate_yr,
        }


# =============================================================================
# MODEL ANALYZER CLASS
# =============================================================================

class ModelAnalyzer:
    """Centralized class for model comparison analysis that caches shared computations."""
    
    def __init__(
        self, 
        processed_data: dict,
        evs_yr: dict,
        evs_GPC3_yr_local: dict,
        model_dic: dict,
        alpha_label: dict = None,
        alpha_c: dict = None,
        alpha_keys: List[str] = None,
        alpha_fj_dict: dict = None,
        fj_samples_by_alpha: dict = None,
        images_folder: Path = None,
        alpha_params_dict: dict = None,
        k: int = 7,
        base_fontsize: int = 14
    ):
        self.processed_data = processed_data
        self.evs_yr = evs_yr
        self.evs_GPC3_yr_local = evs_GPC3_yr_local
        self.model_dic = model_dic
        self.alpha_label = alpha_label or DEFAULT_ALPHA_LABELS
        self.alpha_c = alpha_c or DEFAULT_ALPHA_COLORS
        self.alpha_keys = alpha_keys or DEFAULT_ALPHA_KEYS
        self.alpha_fj_dict = alpha_fj_dict
        self.fj_samples_by_alpha = fj_samples_by_alpha
        self.images_folder = Path(images_folder) if images_folder else None
        self.alpha_params_dict = alpha_params_dict
        self.k = k
        self.base_fontsize = base_fontsize
        self._cache = {}
        
    def _get_cached(self, key: str, compute_func):
        if key not in self._cache:
            self._cache[key] = compute_func()
        return self._cache[key]
    
    # =========================================================================
    # CACHED PROPERTIES
    # =========================================================================
    
    @property
    def all_data(self) -> List[dict]:
        return self._get_cached('all_data', lambda: [
            {"fj": v["fj_summary"][0], "ll": v["mean_log_likelihood"], "alpha": alpha,
             "model": model, "rate": self.evs_GPC3_yr_local.get(alpha, {}).get(model, np.nan),
             "total_rate": self.evs_yr.get(alpha, {}).get(model, np.nan)}
            for alpha, models in self.processed_data.items() for model, v in models.items()])
    
    @property
    def all_models(self) -> List[str]:
        return self._get_cached('all_models', lambda: sorted(set(d["model"] for d in self.all_data)))
    
    @property
    def model_labels(self) -> List[str]:
        return self._get_cached('model_labels', lambda: [self.model_dic.get(m, m) for m in self.all_models])
    
    @property
    def model_to_idx(self) -> Dict[str, int]:
        return self._get_cached('model_to_idx', lambda: {m: i for i, m in enumerate(self.all_models)})
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def save_plot(self, filename: str, dpi: int = 600, additional_formats: List[str] = None):
        if not self.images_folder:
            return
        self.images_folder.mkdir(parents=True, exist_ok=True)
        print(f"Saving plot to {self.images_folder / filename}")
        plt.savefig(self.images_folder / f"{filename}.png", dpi=dpi, bbox_inches='tight')
        for fmt in (additional_formats or []):
            plt.savefig(self.images_folder / f"{filename}.{fmt}", bbox_inches='tight')
    
    def _format_axes(self, ax, xlabel=None, ylabel=None, xscale='log', yscale=None):
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=self.base_fontsize)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=self.base_fontsize)
        if xscale:
            ax.set_xscale(xscale)
        if yscale:
            ax.set_yscale(yscale)
        ax.tick_params(axis="both", labelsize=self.base_fontsize - 2)
    
    def _add_gwtc4_band(self, ax, orientation='v', label='GWTC-4'):# (90\\% C.I.)'):
        if orientation == 'v':
            ax.axvspan(7.6, 250, alpha=0.3, color='gray', label=label, zorder=1)
        else:
            ax.axhspan(7.6, 250, alpha=0.3, color='gray', label=label, zorder=1)
    
    def _scatter_by_alpha(self, ax, data: List[dict], x_key: str, y_key: str, 
                          sizes=None, label: bool = True):
        for alpha in self.alpha_keys:
            alpha_data = [d for d in data if d["alpha"] == alpha]
            if not alpha_data:
                continue
            s = sizes if sizes else 120
            if callable(sizes):
                s = [sizes(d) for d in alpha_data]
            ax.scatter([d[x_key] for d in alpha_data], [d[y_key] for d in alpha_data],
                       c=[self.alpha_c[alpha]], alpha=0.8, edgecolors='black', linewidth=0.8, s=s,
                       label=self.alpha_label[alpha] if label else "")
    
    def _draw_violin(self, ax, data, x_pos, y_base, color, violin_width=0.1, min_val=0, max_val=10):
        if len(data) < 5:
            return
        try:
            kde = gaussian_kde(data)
        except np.linalg.LinAlgError:
            return
        
        grid = np.linspace(min_val, max_val, 100)
        density = kde(grid)
        if np.max(density) > 0:
            density_norm = density / np.max(density) * violin_width
        else:
            density_norm = np.zeros_like(density)
        
        y_positions = y_base + np.interp(grid, [min_val, max_val], [0, 0.91])
        ax.fill_betweenx(y_positions, x_pos - density_norm, x_pos + density_norm,
                         facecolor=color, edgecolor="black", lw=0.5, alpha=0.7, zorder=2)
        
        median = np.median(data)
        median_y = y_base + np.interp(median, [min_val, max_val], [0, 0.9])
        idx = np.searchsorted(grid, median)
        if idx < len(density_norm):
            width_at_med = density_norm[idx] * 0.9
            ax.plot([x_pos - width_at_med, x_pos + width_at_med], [median_y, median_y], 
                    color="k", lw=2.3, zorder=3)
    
    # =========================================================================
    # PLOTTING METHODS
    # =========================================================================
    
    def plot_local_rates_by_model(self, figsize=(14, 6)):
        if not self.all_data:
            return print("No data available.")
        
        models = self.all_models
        fig, ax = plt.subplots(figsize=figsize)
        
        self._add_gwtc4_band(ax, orientation='h')
        
        width = 0.6
        offsets = np.linspace(-width / 2, width / 2, len(self.alpha_keys))
        
        for i, alpha in enumerate(self.alpha_keys):
            alpha_data = [d for d in self.all_data if d['alpha'] == alpha]
            rate_map = {d['model']: d['rate'] for d in alpha_data if not np.isnan(d['rate'])}
            
            x, y = [], []
            for m_idx, model in enumerate(models):
                if model in rate_map:
                    x.append(m_idx + offsets[i])
                    y.append(rate_map[model])
            
            if x:
                ax.scatter(x, y, c=self.alpha_c[alpha], label=self.alpha_label[alpha],
                           s=120, edgecolors='black', linewidth=0.8, alpha=0.9, zorder=10)
        
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([self.model_dic.get(m, m) for m in models],
                           rotation=45, ha='right', fontsize=self.base_fontsize - 2)
        ax.set_xlim(-0.5, len(models) - 0.5)
        ax.set_yscale('log')
        ax.set_ylabel(r'Local BNS Rate $\mathcal{R}_{\text{BNS}}(0)$ [Gpc$^{-3}$yr$^{-1}$]',
                      fontsize=self.base_fontsize)
        ax.set_xlabel("Model", fontsize=self.base_fontsize, weight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=self.base_fontsize - 2, loc='lower right', framealpha=0.9)
        
        plt.tight_layout()
        self.save_plot("local_rates_by_model", additional_formats=['pdf'])
        plt.show()
        return fig, ax
    
    def plot_fj_vs_rates_comparison(self, figsize=(14, 6), f_max=1):
        if not self.all_data:
            return print("No data available for plotting.")
        
        local_data = [d for d in self.all_data if not np.isnan(d["rate"])]
        total_data = [d for d in self.all_data if not np.isnan(d["total_rate"])]
        if not local_data and not total_data:
            return print("No valid rate data for plotting.")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)
        
        if local_data:
            self._scatter_by_alpha(ax1, local_data, 'rate', 'fj')
        self._add_gwtc4_band(ax1)
        ax1.scatter(365, 0.26, marker="*", s=200, color="black", label="R22", zorder=10)
        ax1.scatter(107, 0.7, marker="^", s=150, color="black", label="L25", zorder=11)
        self._format_axes(ax1, r'Local BNS Rate $\mathcal{R}_{\text{BNS}}(0)$ [Gpc$^{-3}$yr$^{-1}$]',
                          r'Median Jet fraction $f_j$')
        ax1.legend(fontsize=self.base_fontsize - 2, loc='upper right', framealpha=0.9, ncol=2)
        
        if total_data:
            self._scatter_by_alpha(ax2, total_data, 'total_rate', 'fj', label=False)
        self._format_axes(ax2, r'Total BNS Rate $\Lambda$ [yr$^{-1}$]')
        ax1.set_ylim(0, f_max)
        
        plt.tight_layout()
        self.save_plot("fj_vs_rates_comparison", additional_formats=['pdf'])
        plt.show()
        return fig, (ax1, ax2)
    
    def plot_fj_fraction_vs_rate(self, figsize=(14, 6), fj_threshold=1.0):
        if not self.all_data:
            return print("No data available for plotting.")
        
        def get_valid_data(rate_key):
            return [{"rate": d[rate_key], "alpha": d["alpha"],
                     "percentile": np.mean(self.processed_data.get(d["alpha"], {}).get(d["model"], {}).get("fj_posterior", []) <= fj_threshold)}
                    for d in self.all_data
                    if not np.isnan(d[rate_key]) and len(self.processed_data.get(d["alpha"], {}).get(d["model"], {}).get("fj_posterior", [])) > 0]
        
        local_data, total_data = get_valid_data("rate"), get_valid_data("total_rate")
        if not local_data and not total_data:
            return print("No valid data with f_j posteriors for plotting.")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)
        #ylabel = f'Quantile of $f_j$ posterior at $f_j = {fj_threshold}$'
        ylabel = f"P($f_j \\leq {fj_threshold}$)"
        for ax, data, xlabel, add_band in [
            (ax1, local_data, r'Local BNS Rate $\mathcal{R}_{\text{BNS}}(0)$ [Gpc$^{-3}$yr$^{-1}$]', True),
            (ax2, total_data, r'Total BNS Rate $\Lambda$ [yr$^{-1}$]', False)
        ]:
            if data:
                self._scatter_by_alpha(ax, data, 'rate', 'percentile', label=(ax == ax1))
            if add_band:
                self._add_gwtc4_band(ax)
            self._format_axes(ax, xlabel, ylabel if ax == ax1 else None)
        
        ax1.set_ylim(-0.05, 1.05)
        ax1.legend(loc='upper left', fontsize=self.base_fontsize - 2, framealpha=0.9)
        plt.tight_layout()
        self.save_plot(f"fj_percentile_vs_rate_threshold_{fj_threshold}", additional_formats=['pdf'])
        plt.show()
        return fig, (ax1, ax2)

    def plot_fj_violins_(self, filename="fj_violin_plot", figsize=(12, 16), min_fj=0, max_fj=10):
            all_models = self.all_models
            alpha_keys = self.alpha_keys
            
            mid = len(all_models) // 2
            fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=False, sharey=True)
            
            # 1. Setup Logarithmic Color Mapping
            rates = [d['rate'] for d in self.all_data if not np.isnan(d['rate']) and d['rate'] > 0]
            
            # Safety fallback if no rates found
            vmin = min(rates) 
            vmax = max(rates) 
            
            # Use LogNorm instead of Normalize
            norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
            cmap = cm.viridis
            
            # Create ScalarMappable for the colorbar
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            
            model_groups = [all_models[:mid], all_models[mid:]]

            for ax, models_subset in zip(axes, model_groups):
                model_to_idx = {m: i for i, m in enumerate(models_subset)}
                
                for alpha in alpha_keys:
                    y_base = alpha_keys.index(alpha) * 1.2
                    
                    # Draw reference lines (keep subtle)
                    grid_vals = [0, 5, 10]
                    for val in grid_vals:
                        if val <= max_fj:
                            y_grid = y_base + np.interp(val, [min_fj, max_fj], [0, 0.9])
                            ax.axhline(y=y_grid, color='gray', ls='-', lw=0.3, alpha=0.3, zorder=0)
                            ax.text(len(models_subset) - 0.5, y_grid, f"- $f_j={val}$",
                                    fontsize=self.base_fontsize - 10, va="center", ha="left")
                    
                    for model_name in models_subset:
                        model_data = self.processed_data.get(alpha, {}).get(model_name)
                        if not model_data:
                            continue
                        
                        rate = self.evs_GPC3_yr_local.get(alpha, {}).get(model_name, np.nan)
                        if np.isnan(rate) or rate <= 0:
                            continue
                        
                        fj_samples = model_data.get("fj_posterior")
                        if fj_samples is None or len(fj_samples) == 0:
                            continue
                        
                        valid = fj_samples[(fj_samples >= min_fj) & (fj_samples <= max_fj)]
                        if len(valid) == 0:
                            continue
                    
                        # Map rate to color using the LogNorm
                        rate_color = cmap(norm(rate))

                        self._draw_violin(ax, valid, model_to_idx[model_name], y_base, rate_color,
                                        violin_width=0.25, min_val=min_fj, max_val=max_fj)
                        
                        med = np.median(valid)
                        p05, p95 = np.percentile(valid, [5, 95])
                        
                        # Annotate statistics
                        ax.text(model_to_idx[model_name], y_base - 0.23,
                                f"${med:.2f}_{{-{med - p05:.2f}}}^{{+{p95 - med:.2f}}}$",
                                fontsize=self.base_fontsize - 10, ha='center', va='bottom')
                
                ax.set_xticks(np.arange(len(models_subset)))
                ax.set_xticklabels([self.model_dic.get(m, m) for m in models_subset],
                                rotation=45, ha="right", fontsize=self.base_fontsize - 2)
                ax.set_xlim(-0.5, len(models_subset) - 0.5)
                # Ensure ticks point in
                ax.tick_params(direction='in', top=True, right=True)
            
            for ax in axes:
                ax.set_yticks([i * 1.2 + 0.455 for i in range(len(alpha_keys))])
                ax.set_yticklabels([self.alpha_label.get(a, a) for a in alpha_keys], fontsize=self.base_fontsize)

            axes[0].set_ylim(-0.3, len(alpha_keys) * 1.15)
            
            # 2. Configure Colorbar with Log Scale but Linear Labels (1, 10, 100)
            cbar = fig.colorbar(sm, ax=axes, aspect=40, pad=0.075)
            cbar.set_label(r'Local BNS Rate $\mathcal{R}_{\rm BNS}(0)$ [Gpc$^{-3}$ yr$^{-1}$]', fontsize=self.base_fontsize)
            
            # Force ticks to be 1, 10, 100, 1000...
            cbar.locator = LogLocator(base=10, subs=(1.0,), numticks=10)
            cbar.formatter = ScalarFormatter() # "100" instead of "10^2"
            cbar.update_ticks()
            cbar.ax.tick_params(labelsize=self.base_fontsize - 2)
            
            self.save_plot(filename, additional_formats=['pdf'])
            plt.show()
            return fig, axes

    def plot_sgrb_rate_posteriors(self, filename="sgrb_rate_posteriors.pdf", max_models=None):
        """
        Plot posterior distributions of R_sGRB = R0 * f_j compared to literature constraints.
        """
        GWTC4_low, GWTC4_high = 7.6, 250
        S23_central, S23_lower, S23_upper = 180, 145, 660
        RE23_central, RE23_lower, RE23_upper = 1786, 1507, 6346
        #[ ]á ñ = - +R 361 217, 4367true,mock Gpc−3 yr−1
        RE23_central, RE23_lower, RE23_upper = 361, 217, 4367
        fig, ax = plt.subplots(figsize=(10, 7))
        all_posteriors = []
        
        for alpha in self.alpha_keys:
            if alpha not in self.processed_data:
                continue
            
            for model_key, data in self.processed_data[alpha].items():
                if alpha not in self.evs_GPC3_yr_local:
                    continue
                if model_key not in self.evs_GPC3_yr_local[alpha]:
                    continue
                
                R0 = self.evs_GPC3_yr_local[alpha][model_key]
                if np.isnan(R0):
                    continue
                
                fj_samples = data["fj_posterior"]
                median_fj = np.median(fj_samples)
                if median_fj > 1.0:
                    continue
                
                R_sgrb_samples = R0 * fj_samples
                all_posteriors.append({
                    "samples": R_sgrb_samples,
                    "model": model_key,
                    "alpha": alpha,
                    "R0": R0
                })
        
        if not all_posteriors:
            print("No physical models found!")
            return
        
        if max_models and len(all_posteriors) > max_models:
            all_posteriors = sorted(all_posteriors, key=lambda x: x["R0"])
            step = len(all_posteriors) // max_models
            all_posteriors = all_posteriors[::step][:max_models]
        
        r0_values = [p["R0"] for p in all_posteriors]
        #norm = mcolors.LogNorm(vmin=min(r0_values), vmax=max(r0_values) * 2)
        # use a norm that works better with linear colormap
        norm = mcolors.Normalize(vmin=min(r0_values), vmax=max(r0_values)*1.3)
        
        cmap = cm.viridis
        
        x_min, x_max = 1, 10_000
        x_grid_log = np.linspace(np.log10(x_min), np.log10(x_max), 500)
        x_grid = 10 ** x_grid_log
        
        for post_data in all_posteriors:
            samples = post_data["samples"]
            r0 = post_data["R0"]
            color = cmap(norm(r0))
            
            try:
                log_samples = np.log10(samples)
                kde = gaussian_kde(log_samples, bw_method=0.2)
                pdf = kde(x_grid_log)
                pdf = pdf #/ pdf.max()
                ax.plot(x_grid, pdf, color=color, alpha=0.7, linewidth=2)
            except:
                continue
        
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        # move cbar tighter to the left
        cbar = plt.colorbar(sm, ax=ax, aspect=40, pad=0.02)
        label_size = 24
        tick_size = 20

        cbar.set_label(r'Local BNS Rate $\mathcal{R}_{\text{BNS}}(0)$ [Gpc$^{-3}$yr$^{-1}$]', fontsize=label_size)
        cbar.ax.tick_params(labelsize=tick_size)
        
        #ax.axvspan(GWTC4_low, GWTC4_high, alpha=0.1, color='blue', label='GWTC-4', zorder=0)
        #ax.axvline(GWTC4_low, color='blue', linestyle='--', linewidth=1.5, alpha=0.4)
        #ax.axvline(GWTC4_high, color='blue', linestyle='--', linewidth=1.5, alpha=0.4)
        # use gray shaded region instead as in other plots
        ax.axvspan(GWTC4_low, GWTC4_high, alpha=0.3, color='gray', label='GWTC-4', zorder=0)    

        s23_left = S23_central - S23_lower
        s23_right = S23_central + S23_upper
        #ax.annotate('', xy=(s23_left * 0.9, 1.01), xytext=(s23_right * 1.1, 1.01),
        #            arrowprops=dict(arrowstyle='<->', color='forestgreen', lw=2.5, alpha=0.9))
        #ax.plot([s23_left, s23_right], [1.01, 1.01],
        #        color='forestgreen', linewidth=2.5, alpha=0.9, label='S23 Flux-Limited')
        
        re23_left = RE23_central - RE23_lower
        re23_right = RE23_central + RE23_upper
        #ax.annotate('', xy=(re23_left * 0.9, 0.8), xytext=(re23_right * 1.1, 0.8),
        #            arrowprops=dict(arrowstyle='<->', color='darkorange', lw=2.5, alpha=0.9))
        #ax.plot([re23_left, re23_right], [0.8, 0.8],
        #        color='darkorange', linewidth=2.5, alpha=0.9, label='RE23 Mock')
        
        #legend_elements = [
        #    Line2D([0], [0], color='forestgreen', linewidth=2, alpha=0.9, label='S23 Flux-Limited'),
        #    Line2D([0], [0], color='darkorange', linewidth=2, alpha=0.9, label='RE23 Mock'),
            #Line2D([0], [0], color='gray', linewidth=2, alpha=0.3, label='GWTC-4'),
        #]
        #make legend element for gray GWTC-4 band
        legend_elements = [
            Patch(facecolor='gray', alpha=0.3, edgecolor='gray',
                  label='GWTC-4'),
            #Line2D([0], [0], color='forestgreen', linewidth=2, alpha=0.9, label='S23 Flux-Limited'),
            #Line2D([0], [0], color='darkorange', linewidth=2, alpha=0.9, label='RE23 Mock'),
        ]
        
        ax.set_xlabel(r'$R_{\rm sGRB} = \mathcal{R}_{\text{BNS}}(0) \times f_j$ [Gpc$^{-3}$ yr$^{-1}$]', fontsize=label_size)
        ax.set_ylabel('PDF', fontsize=label_size)
        ax.set_xscale('log')
        ax.set_xlim(5, 2000)
        ax.set_ylim(0, None)
        ax.tick_params(labelsize=tick_size)
        ax.legend(handles=legend_elements, fontsize=tick_size, loc='upper right',
                  framealpha=0.95, edgecolor='gray')
        
        plt.tight_layout()
        self.save_plot(filename.replace('.pdf', ''), additional_formats=['pdf'])
        plt.show()
    
    def plot_comparison_with_bounded(self, analyzer_bounded: 'ModelAnalyzer', figsize=(14, 6), f_max=10):
        """Compare unbounded vs bounded f_j constraints."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)
        
        def plot_on_ax(ax, data, x_key, marker):
            for alpha in self.alpha_keys:
                alpha_data = [d for d in data if d["alpha"] == alpha]
                if not alpha_data:
                    continue
                ax.scatter([d[x_key] for d in alpha_data], [d['fj'] for d in alpha_data],
                           c=[self.alpha_c[alpha]], marker=marker, s=120, alpha=0.8,
                           edgecolors='black', linewidth=0.8)
        
        # Local Rate Plot
        local_data_unb = [d for d in self.all_data if not np.isnan(d["rate"])]
        plot_on_ax(ax1, local_data_unb, 'rate', '^')
        
        local_data_bnd = [d for d in analyzer_bounded.all_data if not np.isnan(d["rate"])]
        plot_on_ax(ax1, local_data_bnd, 'rate', 'v')
        
        self._add_gwtc4_band(ax1)
        ax1.scatter(365, 0.26, marker="*", s=200, color="black", label="R22", zorder=10)
        ax1.scatter(107, 0.7, marker="^", s=150, color="black", label="L25", zorder=-1)
        self._format_axes(ax1, r'Local BNS Rate $\mathcal{R}_{\text{BNS}}(0)$ [Gpc$^{-3}$yr$^{-1}$]',
                          r'Median Jet fraction $f_j$')
        
        # Total Rate Plot
        total_data_unb = [d for d in self.all_data if not np.isnan(d["total_rate"])]
        plot_on_ax(ax2, total_data_unb, 'total_rate', '^')
        
        total_data_bnd = [d for d in analyzer_bounded.all_data if not np.isnan(d["total_rate"])]
        plot_on_ax(ax2, total_data_bnd, 'total_rate', 'v')
        
        self._format_axes(ax2, r'Total GRB Rate [yr$^{-1}$]')
        ax1.set_ylim(0, f_max)
        
        legend_elements = [
            Line2D([0], [0], marker='^', color='w', label='Unbounded',
                   markerfacecolor='gray', markersize=10, markeredgecolor='k'),
            Line2D([0], [0], marker='v', color='w', label='Bounded ($f_j < 1$)',
                   markerfacecolor='gray', markersize=10, markeredgecolor='k')
        ]
        for alpha in self.alpha_keys:
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w', label=self.alpha_label[alpha],
                       markerfacecolor=self.alpha_c[alpha], markersize=10, markeredgecolor='k')
            )
        
        ax1.legend(handles=legend_elements, fontsize=self.base_fontsize - 4, loc='upper left')
        plt.tight_layout()
        self.save_plot("comparison_bounded_unbounded", additional_formats=['pdf'])
        plt.show()
        return fig, (ax1, ax2)

    def plot_comparison_total_rate_side_by_side(self, analyzer_bounded: 'ModelAnalyzer', figsize=(14, 6), f_max=10):
        """
        Compare bounded vs unbounded f_j constraints with Total Rate plots side-by-side.
        Left: Bounded, Right: Unbounded. No GWTC-4 bands.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)
        
        def plot_on_ax(ax, data):
            for alpha in self.alpha_keys:
                alpha_data = [d for d in data if d["alpha"] == alpha and not np.isnan(d["total_rate"])]
                if not alpha_data:
                    continue
                ax.scatter([d['total_rate'] for d in alpha_data], [d['fj'] for d in alpha_data],
                           c=[self.alpha_c[alpha]], s=120, alpha=0.8,
                           edgecolors='black', linewidth=0.8)
        
        # Plot bounded analyzer on the left
        plot_on_ax(ax1, analyzer_bounded.all_data)
        self._format_axes(ax1, r'Total BNS Rate $\Lambda$ [yr$^{-1}$]', r'Median Jet fraction $f_j$')
        plot_on_ax(ax2, self.all_data)
        self._format_axes(ax2, r'Total BNS Rate $\Lambda$ [yr$^{-1}$]')
        ax2.text(0.5, 0.96, "Unbounded", transform=ax2.transAxes, fontsize=self.base_fontsize,
                 verticalalignment='top', horizontalalignment='center')
        ax1.text(0.5, 0.96, "Bounded ($f_j \leq 1$)", transform=ax1.transAxes, fontsize=self.base_fontsize,
                 verticalalignment='top', horizontalalignment='center')

        ax1.set_ylim(0, f_max)
        
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label=self.alpha_label[alpha],
                   markerfacecolor=self.alpha_c[alpha], markersize=10, markeredgecolor='k')
            for alpha in self.alpha_keys if any(d['alpha'] == alpha for d in self.all_data)
        ]
        
        ax2.legend(handles=legend_elements, fontsize=self.base_fontsize - 4, loc='upper right')
        plt.tight_layout()
        self.save_plot("comparison_total_rate_side_by_side", additional_formats=['pdf'])
        plt.show()

        return fig, (ax1, ax2)

    def get_physical_models(self, fj_threshold: float = 1.0, fraction_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Get list of physical models where median f_j ≤ threshold.
        
        Parameters:
            fj_threshold: Maximum median f_j to be considered physical (default: 1.0)
            fraction_threshold: Minimum fraction of posterior below threshold (default: 0.5)
        
        Returns:
            List of dicts with model info: name, shorthand label, alpha, median_fj, R0, fraction_physical
        """
        physical_models = []
        
        for alpha in self.alpha_keys:
            if alpha not in self.processed_data:
                continue
            
            for model_key, data in self.processed_data[alpha].items():
                fj_samples = data.get("fj_posterior")
                if fj_samples is None or len(fj_samples) == 0:
                    continue
                
                median_fj = np.median(fj_samples)
                fraction_physical = np.mean(fj_samples <= fj_threshold)
                
                # Get R0 if available
                R0 = np.nan
                if alpha in self.evs_GPC3_yr_local and model_key in self.evs_GPC3_yr_local[alpha]:
                    R0 = self.evs_GPC3_yr_local[alpha][model_key]
                
                if fraction_physical >= fraction_threshold:
                    physical_models.append({
                        "name": model_key,
                        "label": self.model_dic.get(model_key, model_key),  # Shorthand label
                        "alpha": alpha,
                        "alpha_label": self.alpha_label.get(alpha, alpha),
                        "median_fj": median_fj,
                        "R0": R0,
                        "fraction_physical": fraction_physical,
                        "fj_5": np.percentile(fj_samples, 5),
                        "fj_95": np.percentile(fj_samples, 95),
                    })
        
        # Sort by R0
        physical_models = sorted(physical_models, key=lambda x: x["R0"] if not np.isnan(x["R0"]) else float('inf'))
        
        return physical_models
    
    def print_physical_models(self, fj_threshold: float = 1.0):
        """Print a formatted table of physical models."""
        models = self.get_physical_models(fj_threshold=fj_threshold)
        
        if not models:
            print(f"No models with median f_j ≤ {fj_threshold}")
            return
        
        print(f"\n{'='*100}")
        print(f"PHYSICAL MODELS (median f_j ≤ {fj_threshold})")
        print(f"{'='*100}")
        print(f"{'#':<4} {'Label':<25} {'Alpha':<20} {'R0 [Gpc⁻³yr⁻¹]':<18} {'f_j (median)':<15} {'f_j (5-95%)':<20}")
        print(f"{'-'*100}")
        
        for i, m in enumerate(models, 1):
            r0_str = f"{m['R0']:.1f}" if not np.isnan(m['R0']) else "N/A"
            fj_range = f"[{m['fj_5']:.2f}, {m['fj_95']:.2f}]"
            print(f"{i:<4} {m['label']:<25} {m['alpha_label']:<20} {r0_str:<18} {m['median_fj']:<15.3f} {fj_range:<20}")
        
        print(f"{'='*100}")
        print(f"Total: {len(models)} physical models\n")
        
        return models

    def plot_combined_analysis(self, figsize=(16, 12), f_max=1, fj_threshold=1.0):
            """
            Generates a 2x2 plot: Median f_j vs Rates and Quantile vs Rates.
            A&A Optimized: No grids, clean markers, shared axes logic.
            """
            if not self.all_data:
                return print("No data available for plotting.")

            # --- 1. Data Preparation ---
            def get_quantile(d):
                posterior = self.processed_data.get(d["alpha"], {}).get(d["model"], {}).get("fj_posterior", [])
                if len(posterior) == 0: return np.nan
                return np.mean(posterior <= fj_threshold)

            plot_data = []
            for d in self.all_data:
                if np.isnan(d.get("rate", np.nan)) and np.isnan(d.get("total_rate", np.nan)):
                    continue
                plot_data.append({
                    "model": d["model"],
                    "short_name": self.model_dic.get(d["model"], d["model"]), 
                    "alpha": d["alpha"],
                    "rate": d.get("rate", np.nan),
                    "total_rate": d.get("total_rate", np.nan),
                    "fj": d.get("fj", np.nan),
                    "quantile": get_quantile(d)
                })

            if not plot_data: return

            unique_models = sorted(list(set(d["model"] for d in plot_data)))
            markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', 'h', 'H', '*', 'X', 'P', '8', 'd', '.']
            model_marker_map = {model: markers[i % len(markers)] for i, model in enumerate(unique_models)}

            fig, axes = plt.subplots(2, 2, figsize=figsize, sharey='row') 
            (ax_med_loc, ax_med_tot), (ax_quant_loc, ax_quant_tot) = axes

            labels = {
                "local_x": r'$\mathcal{R}_{\text{BNS}}(0)$ [Gpc$^{-3}$yr$^{-1}$]',
                "total_x": r'$\Lambda$ [yr$^{-1}$]',
                "median_y": r'Median $f_j$',
                "quant_y": f"P($f_j \\leq {fj_threshold}$)"
            }

            legend_model_handles = {}
            legend_alpha_handles = {}

            for d in plot_data:
                c = self.alpha_c.get(d["alpha"], "black") 
                m = model_marker_map[d["model"]]
                lw = 2.0 if m == '+' else 1
                s_size = 180
                
                kw = dict(c=[c], marker=m, s=s_size, alpha=0.9, edgecolors='k', linewidth=lw, zorder=5)

                if not np.isnan(d["rate"]) and not np.isnan(d["fj"]):
                    ax_med_loc.scatter(d["rate"], d["fj"], **kw)
                if not np.isnan(d["total_rate"]) and not np.isnan(d["fj"]):
                    ax_med_tot.scatter(d["total_rate"], d["fj"], **kw)
                if not np.isnan(d["rate"]) and not np.isnan(d["quantile"]):
                    ax_quant_loc.scatter(d["rate"], d["quantile"], **kw)
                if not np.isnan(d["total_rate"]) and not np.isnan(d["quantile"]):
                    ax_quant_tot.scatter(d["total_rate"], d["quantile"], **kw)

                if d["model"] not in legend_model_handles:
                    legend_model_handles[d["model"]] = mlines.Line2D([], [], color='k', marker=m, ls='None', ms=8, label=d["short_name"])
                if d["alpha"] not in legend_alpha_handles:
                    legend_alpha_handles[d["alpha"]] = mlines.Line2D([], [], color=c, marker='o', ls='None', ms=8, label=self.alpha_label.get(d["alpha"], d["alpha"]))

            self._add_gwtc4_band(ax_med_loc)
            self._add_gwtc4_band(ax_quant_loc)

            ax_med_loc.scatter(365, 0.26, marker="*", s=250, color="black", label="R22", zorder=10)
            ax_med_loc.scatter(107, 0.7, marker="^", s=200, color="black", label="L25", zorder=11)

            for ax in axes.flatten():
                ax.set_xscale('log')
                ax.tick_params(direction='in', top=True, right=True, which='both', labelsize=self.base_fontsize - 2)

            ax_med_loc.set_ylim(0, f_max)
            ax_quant_loc.set_ylim(-0.05, 1.05)

            ax_quant_loc.set_xlabel(labels["local_x"], fontsize=self.base_fontsize)
            ax_quant_tot.set_xlabel(labels["total_x"], fontsize=self.base_fontsize)
            ax_med_loc.set_ylabel(labels["median_y"], fontsize=self.base_fontsize)
            ax_quant_loc.set_ylabel(labels["quant_y"], fontsize=self.base_fontsize)

            # Legend Construction make the patch a bit smaller
            gwtc_handle = mpatches.Patch(color='gray', alpha=0.3, label='GWTC-4')
            r22_handle = mlines.Line2D([], [], color='black', marker='*', linestyle='None', markersize=10, label="R22")
            l25_handle = mlines.Line2D([], [], color='black', marker='^', linestyle='None', markersize=10, label="L25")
            
            ref_handles = [gwtc_handle, r22_handle, l25_handle]
            alpha_handles = [legend_alpha_handles[k] for k in sorted(legend_alpha_handles.keys())]
            model_handles = [legend_model_handles[k] for k in sorted(legend_model_handles.keys())]

            all_handles = ref_handles + alpha_handles + model_handles

            # Position legend at top, no frame
            fig.legend(handles=all_handles, loc='upper center', bbox_to_anchor=(0.52, 1.09), 
                    ncol=11, fontsize=self.base_fontsize - 4, frameon=True, columnspacing=0.2, handletextpad=0.11)

            plt.tight_layout()
            self.save_plot("combined_population_analysis", additional_formats=['pdf'])            
            plt.show()
            return fig, axes

    def plot_fj_violins(self, figsize=(20, 8), filename: str = "fj_violin_plots"):
        """
        Plot violin plots of epsilon distributions.
        - Single panel: All 16 models side-by-side on X-axis.
        - Vertical Violins: Epsilon values on Y-axis.
        - Rows: Different Alpha_CE values stacked vertically.
        """

        all_models = self.all_models
        alpha_keys = self.alpha_keys

        # Wide figure for side-by-side models
        fig, ax = plt.subplots(1, 1, figsize=figsize) 
                        
        # Color mapping logic
        rates = [d['rate'] for d in self.all_data if not np.isnan(d['rate']) and d['rate'] > 0]
        vmin = min(rates) 
        vmax = max(rates) 
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
        cmap = cm.viridis
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        model_to_idx = {m: i for i, m in enumerate(all_models)}
        
        # Vertical spacing between Alpha rows
        y_step = 1.15 

        min_fj = 0
        max_fj = 10
        models_subset = all_models  # Limit to first 16 models for clarity

        for alpha in alpha_keys:
            y_base = alpha_keys.index(alpha) * y_step
            
            # Draw reference lines (keep subtle)
            grid_vals = [0, 5, 10]
            for val in grid_vals:
                y_grid = y_base + np.interp(val, [min_fj, max_fj], [0, 0.9])
                ax.axhline(y=y_grid, color='gray', ls='-', lw=0.3, alpha=0.3, zorder=0)
                ax.text(len(models_subset) - 0.5, y_grid, f"- $f_j={val}$",
                        fontsize=self.base_fontsize - 10, va="center", ha="left")
                        
            for model_name in all_models:
                model_data = self.processed_data.get(alpha, {}).get(model_name)
                rate = self.evs_GPC3_yr_local.get(alpha, {}).get(model_name, np.nan)
                rate_color = cmap(norm(rate))
                fj_samples = model_data.get("fj_posterior")
                valid = fj_samples[(fj_samples >= min_fj) & (fj_samples <= max_fj)]
                self._draw_violin(ax, valid, model_to_idx[model_name], y_base, rate_color,
                                  violin_width=0.25, min_val=min_fj, max_val=max_fj)
                
                # Median Text
                med = np.median(valid)
                p05, p95 = np.percentile(valid, [5, 95])
                ax.text(model_to_idx[model_name], y_base - 0.05,
                        f"${med:.2f}_{{-{med-p05:.2f}}}^{{+{p95-med:.2f}}}$",
                        fontsize=self.base_fontsize - 10, ha='center', va='top')

        # X-Axis Settings
        ax.set_xticks(np.arange(len(all_models)))
        ax.set_xticklabels([self.model_dic.get(m, m) for m in all_models], rotation=45, ha="right", fontsize=self.base_fontsize-2)
        ax.set_xlim(-0.51, len(all_models) - 0.49)
        
        # Y-Axis Settings (Alpha labels)
        ax.set_yticks([i * y_step + 0.45 for i in range(len(alpha_keys))])
        ax.set_ylabel(f"$\\alpha_{{\mathrm{{CE}}}}$", fontsize=self.base_fontsize)
        ax.set_yticklabels([f"${a[1:]}$" for a in alpha_keys], fontsize=self.base_fontsize)
        ax.set_ylim(-0.3, len(alpha_keys) * y_step - 0.1)

        # Cbar
        cbar = fig.colorbar(sm, ax=ax, aspect=40, pad=0.049)
        cbar.set_label(r'Local BNS Rate $\mathcal{R}_{\rm BNS}(0)$ [Gpc$^{-3}$ yr$^{-1}$]', fontsize=self.base_fontsize)
        cbar.locator = LogLocator(base=10, subs=(1.0,), numticks=4)
        cbar.formatter = ScalarFormatter()
        cbar.ax.tick_params(labelsize=self.base_fontsize - 2)
        cbar.update_ticks()

        self.save_plot(filename, additional_formats=['pdf'])
        plt.show()
        return fig, ax

class PopulationAnalysis:
    """
    High-level interface for population analysis.
    """
    
    def __init__(
        self,
        output_folder: str = "Output_files/ProductionPop_fj10_20k",
        images_folder: str = "tutorial_images/populations",
        datafiles: str = "datafiles",
        **config_kwargs
    ):
        self.config = PopulationConfig(
            output_folder=Path(output_folder),
            images_folder=Path(images_folder),
            datafiles=Path(datafiles),
            **config_kwargs
        )
        
        # Ensure directories exist
        self.config.images_folder.mkdir(parents=True, exist_ok=True)
        
        # Load external data
        self.evs_yr = load_json_dict(self.config.datafiles / "evs_yr.json")
        self.evs_GPC3_yr_local = load_json_dict(self.config.datafiles / "evs_GPC3_yr_local.json")
        self.model_dic = load_json_dict(self.config.datafiles / "model_dic.json")
        
        # Initialize components
        self.preprocessor = DataPreprocessor(self.config)
        self.analyzer: Optional[ModelAnalyzer] = None
        
        # Data storage
        self.processed_data = None
        self.alpha_fj_dict = None
        self.fj_samples_by_alpha = None
        self.alpha_dict = None
        self.alpha_params_dict = None
    
    def run_preprocessing(self, samp_names: List[str] = None, filter_by_rate: bool = True, quiet: bool = False):
        """Run data preprocessing and create analyzer."""
        local_rates = self.evs_GPC3_yr_local if filter_by_rate else None
        
        (self.processed_data, self.alpha_fj_dict, self.fj_samples_by_alpha,
         self.alpha_dict, self.alpha_params_dict) = self.preprocessor.preprocess(
            samp_names=samp_names,
            local_rates_dict=local_rates,
            quiet=quiet
        )
        
        self.analyzer = ModelAnalyzer(
            processed_data=self.processed_data,
            evs_yr=self.evs_yr,
            evs_GPC3_yr_local=self.evs_GPC3_yr_local,
            model_dic=self.model_dic,
            alpha_label=self.config.alpha_labels,
            alpha_c=self.config.alpha_colors,
            alpha_keys=self.config.alpha_keys,
            alpha_fj_dict=self.alpha_fj_dict,
            fj_samples_by_alpha=self.fj_samples_by_alpha,
            images_folder=self.config.images_folder,
            alpha_params_dict=self.alpha_params_dict,
            k=self.config.k_params,
            base_fontsize=self.config.base_fontsize
        )
        
        return self.analyzer
    
    def get_sample_names(self) -> List[str]:
        """Get list of sample names from population folder."""
        samples = list(self.config.population_folder.glob("samples_*.dat"))
        return [s.name.split("samples_")[1].split("_BNSs")[0] for s in samples]
    


import json
import emcee
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.stats import gaussian_kde
from matplotlib.ticker import LogLocator, ScalarFormatter
from typing import Dict, List, Optional, Any


class PopulationAnalysis:
    """
    Streamlined population analysis and plotting class.
    """
    DEFAULT_ALPHA_KEYS = ["A0.5", "A1.0", "A3.0", "A5.0"]
    DEFAULT_COLORS = {"A5.0": "sandybrown", "A3.0": "palevioletred", 
                      "A1.0": "blueviolet", "A0.5": "darkblue"}
    DEFAULT_LABELS = {"A5.0": r"$\alpha_{\rm CE} = 5.0$", "A3.0": r"$\alpha_{\rm CE} = 3.0$",
                      "A1.0": r"$\alpha_{\rm CE} = 1.0$", "A0.5": r"$\alpha_{\rm CE} = 0.5$"}

    def __init__(self, output_folder, images_folder="images", datafiles="datafiles", 
                 discard=10000, thin=15, base_fontsize=22):
        self.output_folder = Path(output_folder)
        self.images_folder = Path(images_folder)
        self.datafiles_dir = Path(datafiles)
        self.discard = discard
        self.thin = thin
        self.base_fontsize = base_fontsize
        self.images_folder.mkdir(parents=True, exist_ok=True)
        
        # Load Metadata
        self.evs_yr = load_json_dict(self.datafiles_dir / "evs_yr.json")
        self.evs_GPC3_yr_local = load_json_dict(self.datafiles_dir / "evs_GPC3_yr_local.json")
        self.model_dic = load_json_dict(self.datafiles_dir / "model_dic.json")
        
        self.data: List[Dict[str, Any]] = []
        
        # Backward compatibility for "comparison_with_bounded" script usage
        self.analyzer = self 

    def run_preprocessing(self, quiet=False):
        """Loads f_j samples and rates for all found models."""
        self.data = []
        
        # Identify samples from output folder content
        found_folders = list(self.output_folder.glob("*_A*"))
        
        if not quiet: print(f"Processing {len(found_folders)} models...")

        for folder in found_folders:
            folder_name = folder.name
            try:
                # Extract identifiers (Assuming naming "ModelName_A#.#...")
                alpha = next((k for k in self.DEFAULT_ALPHA_KEYS if k in folder_name), None)
                if not alpha: continue
                
                # Basic string parsing to get model name
                # Adjust this split logic if your naming convention varies
                model_name = folder_name.split(f"_{alpha}")[0].replace("samples_", "")
                
                # Load Backend
                h5_path = folder / "emcee.h5"
                if not h5_path.exists(): continue
                
                reader = emcee.backends.HDFBackend(str(h5_path), read_only=True)
                chain = reader.get_chain(discard=self.discard, thin=self.thin, flat=True)
                
                if chain.shape[0] == 0: continue
                
                # Get Rates
                rate_loc = self.evs_GPC3_yr_local.get(alpha, {}).get(model_name, np.nan)
                rate_tot = self.evs_yr.get(alpha, {}).get(model_name, np.nan)

                if rate_loc < 1:
                    mask    = chain[:, -1] >= 1
                    chain   = chain[mask]

                # Store Data
                self.data.append({
                    "model": model_name,
                    "label": self.model_dic.get(model_name, model_name),
                    "alpha": alpha,
                    "rate": rate_loc,
                    "total_rate": rate_tot,
                    "fj_samples": chain[:, -1] # Assumes f_j is last parameter
                })
                
            except Exception as e:
                if not quiet: print(f"Skipping {folder_name}: {e}")

        if not quiet: print(f"Loaded {len(self.data)} models successfully.")

    # Property Helpers
    @property
    def unique_models(self):
        return sorted(list(set(d["model"] for d in self.data)))
    
    @property
    def unique_alphas(self):
        return sorted(list(set(d["alpha"] for d in self.data)))
        
    def _save_plot(self, filename):
        plt.savefig(self.images_folder / f"{filename}.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.images_folder / f"{filename}.pdf", bbox_inches='tight')
        print(f"Saved {filename}")

    def _add_gwtc4(self, ax, orientation='v'):
        if orientation == 'v':
            ax.axvspan(7.6, 250, alpha=0.3, color='gray', label='GWTC-4', zorder=1)
        else:
            ax.axhspan(7.6, 250, alpha=0.3, color='gray', label='GWTC-4', zorder=1)

    def plot_combined_analysis(self, figsize=(16, 12), f_max=5, fj_threshold=1.0):
        if not self.data: return print("No data.")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize, sharey='row')
        (ax_med_loc, ax_med_tot), (ax_quant_loc, ax_quant_tot) = axes
        
        markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', 'h', 'H', '*', 'X']
        model_markers = {m: markers[i % len(markers)] for i, m in enumerate(self.unique_models)}
        
        # Legend Handles
        alpha_handles = {}
        model_handles = {}

        for d in self.data:
            c = self.DEFAULT_COLORS.get(d["alpha"], "k")
            m = model_markers[d["model"]]
            
            med_fj = np.median(d["fj_samples"])
            quantile = np.mean(d["fj_samples"] <= fj_threshold)
            
            kw = dict(c=[c], marker=m, s=180, alpha=0.9, edgecolors='k', zorder=5)
            
            # Top Row: Median fj
            if not np.isnan(d["rate"]):     ax_med_loc.scatter(d["rate"], med_fj, **kw)
            if not np.isnan(d["total_rate"]): ax_med_tot.scatter(d["total_rate"], med_fj, **kw)
            
            # Bottom Row: Quantile
            if not np.isnan(d["rate"]):     ax_quant_loc.scatter(d["rate"], quantile, **kw)
            if not np.isnan(d["total_rate"]): ax_quant_tot.scatter(d["total_rate"], quantile, **kw)
            
            # Store handles
            if d["alpha"] not in alpha_handles:
                alpha_handles[d["alpha"]] = mlines.Line2D([], [], color=c, marker='o', ls='None', ms=10, label=self.DEFAULT_LABELS[d["alpha"]])
            if d["model"] not in model_handles:
                model_handles[d["model"]] = mlines.Line2D([], [], color='k', marker=m, ls='None', ms=10, label=d["label"])

        # Formatting
        self._add_gwtc4(ax_med_loc)
        self._add_gwtc4(ax_quant_loc)
        
        # R22 / L25 references
        # pick two non fileld markers
        #marker_r22 = "$R22$"
        #marker_l25 = "$L25$"
        #ax_med_loc.scatter(365, 0.26, marker=marker_r22, s=550, color="black", label="R22", zorder=10)
        #ax_med_loc.scatter(107, 0.7, marker=marker_l25, s=550, color="black", label="L25", zorder=11)
        # two small horizontal arrows to indicate the points ie. r22 text -> 

        # R22 / L25 references — labels left, arrows left→right pointing to the points
        pt_r22 = (365, 0.26)
        pt_l25 = (107, 0.7)

        ax_med_loc.annotate("R22", xy=pt_r22, xytext=(pt_r22[0] - 140, pt_r22[1]),
                            va="center", ha="right", fontsize=self.base_fontsize-4, zorder=11,
                            arrowprops=dict(arrowstyle="->", lw=1.5, color="k", shrinkA=0, shrinkB=0))

        ax_med_loc.annotate("L25", xy=pt_l25, xytext=(pt_l25[0] - 50, pt_l25[1]),
                            va="center", ha="right", fontsize=self.base_fontsize-4, zorder=11,
                            arrowprops=dict(arrowstyle="->", lw=1.5, color="k", shrinkA=0, shrinkB=0))

        for ax in axes.flatten():   
            ax.tick_params(direction='in', top=True, right=True, which='both', labelsize=self.base_fontsize - 4)
            ax.set_xscale('log')
            
        ax_med_loc.set_ylim(0, f_max)
        ax_quant_loc.set_ylim(-0.05, 1.05)
        
        ax_med_loc.set_ylabel(r'Median $f_j$', fontsize=self.base_fontsize)
        ax_quant_loc.set_ylabel(f"P($f_j \\leq {fj_threshold}$)", fontsize=self.base_fontsize)
        ax_quant_loc.set_xlabel(r'$\mathcal{R}_{\text{BNS}}(0)$ [Gpc$^{-3}$yr$^{-1}$]', fontsize=self.base_fontsize)
        ax_quant_tot.set_xlabel(r'$\Lambda$ [yr$^{-1}$]', fontsize=self.base_fontsize)

        gwtc_handle = mpatches.Patch(color='gray', alpha=0.3, label='GWTC-4')
        #r22_handle = mlines.Line2D([], [], color='black', marker=marker_r22, linestyle='None', markersize=10, label="R22")
        #l25_handle = mlines.Line2D([], [], color='black', marker=marker_l25, linestyle='None', markersize=10, label="L25")
        # use arrow R22/L25 to indicate legend of those references
        r22_handle = mlines.Line2D([], [], color='black', marker='$\\rightarrow$', linestyle='None', markersize=10, label="R22/L25")
        all_handles = [gwtc_handle, r22_handle] + \
                      list(alpha_handles.values()) + list(model_handles.values())


        fig.legend(handles=all_handles, loc='upper center', bbox_to_anchor=(0.52, 1.065), 
                    ncol=11, fontsize=self.base_fontsize - 4, frameon=True, columnspacing=0.2, handletextpad=0.11)

        plt.tight_layout()
        self._save_plot("combined_population_analysis")
        plt.show()

    def plot_fj_violins(self, filename="fj_violin_plots"):
        if not self.data: return
        
        fig, ax = plt.subplots(figsize=(20, 8))
        
        # Color mapping based on rate
        rates = [d['rate'] for d in self.data if not np.isnan(d['rate']) and d['rate'] > 0]
        norm = mcolors.LogNorm(vmin=min(rates), vmax=max(rates))
        cmap = cm.viridis
        
        models = self.unique_models
        alphas = self.DEFAULT_ALPHA_KEYS
        
        y_step = 1.15
        min_fj, max_fj = 0, 10
        
        for i, alpha in enumerate(alphas):
            y_base = i * y_step
            
            # Grid lines
            for val in [0, 5, 10]:
                y = y_base + np.interp(val, [min_fj, max_fj], [0, 0.9])
                ax.axhline(y, color='gray', alpha=0.3, lw=0.5)
                ax.text(len(models) - 0.55, y-0.01, f"- $f_j={val}$", fontsize=self.base_fontsize-10, va='center', ha='left')
                
            for j, model in enumerate(models):
                entry = next((d for d in self.data if d["model"] == model and d["alpha"] == alpha), None)
                if not entry or np.isnan(entry["rate"]): continue
                
                valid_samps = entry["fj_samples"][(entry["fj_samples"] >= min_fj) & (entry["fj_samples"] <= max_fj)]
                
                draw_vertical_violin(ax, valid_samps, j, y_base, cmap(norm(entry["rate"])), 
                                     width=0.25, min_val=min_fj, max_val=max_fj)
                
                # Stats text
                med = np.median(valid_samps)
                p05, p95 = np.percentile(valid_samps, [5, 95])
                ax.text(j, y_base - 0.05, f"${med:.2f}_{{-{med-p05:.2f}}}^{{+{p95-med:.2f}}}$", 
                        fontsize=self.base_fontsize-10, ha='center', va='top')

        # Axis styling
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([self.model_dic.get(m, m) for m in models], rotation=45, ha='right', fontsize=self.base_fontsize-2)
        ax.set_yticks([i * y_step + 0.45 for i in range(len(alphas))])
        ax.set_yticklabels([self.DEFAULT_LABELS.get(a, a) for a in alphas], fontsize=self.base_fontsize)
        #inside ticks for all
        ax.tick_params(direction='in', top=True, right=True, which='both', labelsize=self.base_fontsize - 2)
        ax.set_ylim(-0.32, len(alphas) * y_step - 0.15)
        ax.set_xlim(-0.51, len(models) - 0.49)

        # Colorbar
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        
        # Cbar
        cbar = fig.colorbar(sm, ax=ax, aspect=40, pad=0.049)
        cbar.set_label(r'Local BNS Rate $\mathcal{R}_{\rm BNS}(0)$ [Gpc$^{-3}$ yr$^{-1}$]', fontsize=self.base_fontsize)
        cbar.locator = LogLocator(base=10, subs=(1.0,), numticks=4)
        cbar.formatter = ScalarFormatter()
        cbar.ax.tick_params(labelsize=self.base_fontsize - 2)
        cbar.update_ticks()

        self._save_plot(filename)
        plt.show()


    def plot_sgrb_rate_posteriors(self, filename="sgrb_rate_posteriors"):
        fig, ax = plt.subplots(figsize=(10, 7))
        
        x_grid = np.logspace(0, 4, 500)
        max_pdf = 0
        
        cmap        = cm.viridis
        r0_values   = [d["rate"] for d in self.data if not np.isnan(d["rate"])]
        norm        = mcolors.Normalize(vmin=80, vmax=max(r0_values)*1.3)

        for d in self.data:
            if np.isnan(d["rate"]): continue

            fj_samples = d["fj_samples"]
            if np.median(fj_samples) >= 1: continue
            
            sgrb_rate = d["rate"] * fj_samples
            color = cmap(norm(d["rate"]))
            try:
                kde         = gaussian_kde(np.log10(sgrb_rate), bw_method=0.2)
                pdf         = kde(np.log10(x_grid))
                ax.plot(x_grid, pdf, color=color, alpha=0.7, lw=2.5) # Simplified coloring
                max_pdf = max(max_pdf, pdf.max())
            except: continue

        self._add_gwtc4(ax)
        ax.set_xscale('log')
        ax.set_xlabel(r'$R_{\rm sGRB} = \mathcal{R}_{\text{BNS}}(0) \times f_j$ [Gpc$^{-3}$ yr$^{-1}$]', fontsize=self.base_fontsize)
        ax.set_ylabel('PDF', fontsize=self.base_fontsize)
        ax.tick_params(direction='in', top=False, right=False, which='both', labelsize=self.base_fontsize-4)
        ax.set_ylim(0, None)
        ax.set_xlim(5, 2000)
        legend_elements = [
            Patch(facecolor='gray', alpha=0.3, edgecolor='gray',
                  label='GWTC-4'),
        ]
        ax.legend(handles=legend_elements, fontsize=self.base_fontsize-4, loc='upper right',
                  framealpha=0.95, edgecolor='gray')
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        # move cbar tighter to the left
        cbar = plt.colorbar(sm, ax=ax, aspect=40, pad=0.02)
        cbar.set_label(r'Local BNS Rate $\mathcal{R}_{\text{BNS}}(0)$ [Gpc$^{-3}$yr$^{-1}$]', fontsize=self.base_fontsize)
        cbar.ax.tick_params(labelsize=self.base_fontsize-4)
        self._save_plot(filename)
        plt.show()

    def plot_comparison_with_bounded(self, bounded_analyzer, f_max=5):
        """Bounded analyzer is passed as an argument (another PopulationAnalysis instance)."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        
        def plot_dataset(ax, dataset, marker, label_prefix):
            for d in dataset:
                if np.isnan(d["rate"]): continue
                c = self.DEFAULT_COLORS.get(d["alpha"], "k")
                med = np.median(d["fj_samples"])
                ax.scatter(d["rate"] if ax == ax1 else d["total_rate"], med, 
                           c=[c], marker=marker, s=120, edgecolors='k', alpha=0.8)

        # Plot Current (Unbounded)
        plot_dataset(ax1, self.data, '^', "Unbounded")
        plot_dataset(ax2, self.data, '^', "Unbounded")
        
        # Plot Bounded (Passed Argument)
        # Note: bounded_analyzer might be the class instance itself depending on how it's called
        b_data = bounded_analyzer.data if hasattr(bounded_analyzer, 'data') else bounded_analyzer.analyzer.data
        plot_dataset(ax1, b_data, 'v', "Bounded")
        plot_dataset(ax2, b_data, 'v', "Bounded")

        self._add_gwtc4(ax1)
        ax1.set_ylim(0, f_max)
        ax1.set_xlabel(r'Local Rate [Gpc$^{-3}$yr$^{-1}$]', fontsize=self.base_fontsize)
        ax2.set_xlabel(r'Total Rate [yr$^{-1}$]', fontsize=self.base_fontsize)
        ax1.set_ylabel(r'Median $f_j$', fontsize=self.base_fontsize)
        
        # Legend construction
        legend_elements = [
            mlines.Line2D([], [], color='k', marker='^', ls='None', ms=10, label='Unbounded'),
            mlines.Line2D([], [], color='k', marker='v', ls='None', ms=10, label='Bounded ($f_j \leq 1$)')
        ]
        ax1.legend(handles=legend_elements, loc='upper left', fontsize=self.base_fontsize-4)
        
        self._save_plot("comparison_bounded_unbounded")
        plt.show()