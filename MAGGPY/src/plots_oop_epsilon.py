# plot_oop_epsilon.py

import json
import emcee
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import gaussian_kde, binned_statistic, lognorm
from scipy.optimize import brentq
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from typing import List, Dict, Any, Callable, Optional
import matplotlib.colors as mcolors
from matplotlib import cm
from matplotlib.ticker import LogLocator, ScalarFormatter
from .init import create_run_dir

FONTSIZE = 22

def draw_vertical_violin(ax, log_data, x_pos, y_base, color, hatch=None, 
                        width=0.1, min_log=-3, max_log=0):
    """Standalone function to draw a single vertical violin."""
    if len(log_data) < 5: return
    
    kde             = gaussian_kde(log_data)
    log_range       = np.linspace(min_log, max_log, 100)
    density         = kde(log_range)
    density_norm    = density / np.max(density) * width
    
    # Map to visual coordinates
    y_positions = y_base + np.interp(log_range, [min_log, max_log], [0, 0.9])
    
    ax.fill_betweenx(y_positions, x_pos - density_norm, x_pos + density_norm,
                    facecolor=color, hatch=hatch, edgecolor="black", 
                    lw=0.5, alpha=0.7, zorder=2)
    
    # Draw Median
    median_val = np.median(log_data)
    median_y = y_base + np.interp(median_val, [min_log, max_log], [0, 0.9])
    width_at_med = density_norm[np.searchsorted(log_range, median_val)] * 0.9
    ax.plot([x_pos - width_at_med, x_pos + width_at_med], [median_y, median_y], 
            color="k", lw=2.3, zorder=3)

class BaseModelPlotter:
    """
    Base class with shared functionality for plotting MCMC model comparison results.
    """
    # Pre-define styles to avoid re-declaring in every method
    COLORS = ["#3333a1", "#a155e7", "#e28ca8", "#f5b57f"]
    MARKERS = ["o", "s", "^", "D"]

    def __init__(
        self,
        samp_names: List[str],
        base_dir: str,
        data_files_dir: str,
        output_dir: str,
        discard: int,
        thin: int,
        k_params: int,
        fontsize: int = FONTSIZE,
    ):
        self.fontsize       = fontsize
        self.output_dir     = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_files_dir = Path(data_files_dir)  

        self.local_rates    = self._load_json(self.data_files_dir / "evs_GPC3_yr_local.json")
        self.model_dic      = self._load_json(self.data_files_dir / "model_dic.json")
        self.total_rates    = self._load_json(self.data_files_dir / "evs_yr.json")  # Add this line
        
        self.model_data = self._load_and_process_data(
            samp_names, base_dir, self.local_rates, discard, thin
        )

        # Create a lookup for quick access
        self.data_lookup = {
            (d["name"], d["alpha"]): d for d in self.model_data
        }

        # Cache unique alphas for plotting groups
        self.alpha_keys = sorted(list(set(d["alpha"] for d in self.model_data)))
        
        # Setup Colormap constraints
        self.cmap   = plt.cm.viridis
        rates       = [d["rate"] for d in self.model_data if not np.isnan(d["rate"])] # just a list of self.local_rates, for colormap reasons
        self.norm = mcolors.LogNorm(vmin=min(rates) if rates else 0.1, vmax=max(rates) if rates else 1000)
    
    def _load_json(self, file_path: Path) -> Dict:
        with open(file_path, "r") as f:
            return json.load(f)
    
    def _save_and_show(self, fig: plt.Figure, filename: str, **savefig_kwargs):
        if self.output_dir:
            output_path = self.output_dir / filename
            savefig_kwargs.setdefault('bbox_inches', 'tight')
            fig.savefig(output_path, **savefig_kwargs)
            print(f"Saved plot to {output_path}")
        plt.show()
        plt.close(fig)
    
    def _load_and_process_data(
        self, samp_names, base_dir, local_rates, discard, thin
    ) -> List[Dict[str, Any]]:
        processed_data = []
        print("Loading and processing model data...")

        base_path_obj = Path(base_dir)
        for z_name in samp_names:
            try:
                alpha = z_name[-4:]
                model_name = z_name[:-5]
                local_rate = local_rates[alpha][model_name]
                total_rate = self.total_rates[alpha][model_name]

                run_path = base_path_obj / z_name
                outdir = create_run_dir(run_path, use_timestamp=False, QUIET_FLAG=True)
                
                backend = emcee.backends.HDFBackend(outdir / "emcee.h5", read_only=True)
                                                    
                all_samples             = backend.get_chain(discard=discard, thin=thin, flat=True)
                blobs                   = backend.get_blobs(discard=discard, thin=thin, flat=True)
                detection_efficiency    = blobs["l_eff"] 
                
                model_data_entry = {
                    "full_name"             : z_name,
                    "name"                  : model_name,
                    "alpha"                 : alpha,
                    "rate"                  : local_rate,
                    "total_rate"            : total_rate, 
                    "all_samples"           : all_samples,
                    "detection_efficiency"  : detection_efficiency,  
                } 

                self._extract_model_specific_samples(model_data_entry)
                
                processed_data.append(model_data_entry)
                
            except (FileNotFoundError, KeyError, OSError) as e:
                print(f"Could not process {z_name}. Error: {e}")
                continue

        print(f"Successfully processed {len(processed_data)} models.")
        return processed_data
    
    def _extract_model_specific_samples(self, data: Dict[str, Any]): pass
    
    def _iter_alpha_groups(self):
        """Helper generator to iterate over models grouped by alpha."""
        for i, alpha in enumerate(self.alpha_keys):
            style = {
                'color': self.COLORS[i % len(self.COLORS)],
                'marker': self.MARKERS[i % len(self.MARKERS)],
                'label': f'$\\alpha_{{CE}} = {alpha[1:]}$'
            }
            # Filter models for this alpha
            models = [d for d in self.model_data if d["alpha"] == alpha and not np.isnan(d["rate"])]
            yield alpha, models, style

    @staticmethod
    def calculate_theta_star_from_beaming(theta_samples, fj_samples, beaming_func, boundary_limit=9.0, n_bins=50):
        """
        Calculates theta* (where f_j = 1).
        """
        min_theta, max_theta = np.min(theta_samples), np.max(theta_samples)        
        bins = np.logspace(np.log10(min_theta), np.log10(max_theta), n_bins + 1)
        
        # Calculate 95th Percentile in each bin
        p95_per_bin, bin_edges, bin_indices = binned_statistic(
            theta_samples, fj_samples, 
            statistic=lambda y: np.percentile(y, 95), 
            bins=bins
        )
        
        valid_bin_indices = np.where(p95_per_bin < boundary_limit)[0] + 1
        
        if len(valid_bin_indices) == 0:
            print("No valid samples found within boundary limit.")
            return np.nan, np.nan, np.zeros(len(theta_samples), dtype=bool)

        # Calculate Constant
        valid_samples_mask = np.isin(bin_indices, valid_bin_indices)

        theta_valid = theta_samples[valid_samples_mask]
        fj_valid    = fj_samples[valid_samples_mask]
 
        beaming_vals = beaming_func(theta_valid)
        C_est        = np.median(fj_valid * beaming_vals)
        
        def objective(t):
            return beaming_func(t) - C_est
            
        # Determine search range respecting function bounds
        lower, upper = 1.1, 45
        if hasattr(beaming_func, 'x'): 
            lower = max(lower, beaming_func.x[0])
            upper = min(upper, beaming_func.x[-1])

        theta_star = np.nan
        val_lower = objective(lower)
        val_upper = objective(upper)
        
        if np.isnan(val_lower) or np.isnan(val_upper):
            return np.nan, C_est, valid_samples_mask
        
        if val_lower * val_upper < 0:
            theta_star = brentq(objective, lower, upper)

        return theta_star, C_est, valid_samples_mask

    def plot_theta_star_robust_vs_rate(self, filename: str = "theta_star_robust_vs_rate.pdf",
                                       boundary_limit: float = 9.0, n_bins: int = 50, label_flag: int = 0,
                                       beaming_func: callable = None, legend_flag: bool = False, xlabel: bool = False, ax=None):
        """
        Plot the robustly estimated theta* vs local BNS rate.
        
        Parameters
        ----------
        beaming_func : callable, optional
            If provided, uses calculate_theta_star_from_beaming with this function.
            If None, shouldn't be.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
            external_ax = False
        else:
            fig = ax.get_figure()
            external_ax = True
        
        theta_stars = {} # Store results here
                
        for alpha, models, style in self._iter_alpha_groups():
            rates, thetas = [], []

            for data in models:
                if "theta_c_samples" not in data or "fj_samples" not in data:
                    continue
                    
                theta_star, _, _ = self.calculate_theta_star_from_beaming(
                    data["theta_c_samples"], data["fj_samples"], 
                    beaming_func=beaming_func,
                    boundary_limit=boundary_limit, 
                    n_bins=n_bins
                )

                if label_flag == 1:
                    theta_star = (theta_star + 1) / 2

                rates.append(data["rate"])
                thetas.append(theta_star)
                theta_stars[data["full_name"]] = theta_star
            
            ax.scatter(rates, thetas, s=180, alpha=0.8, edgecolors='k', linewidth=0.5, **style)
        
        ax.axvspan(7.6, 250, alpha=0.2, color='gray', label='GWTC-4', zorder=0)
        ax.axhline(y=6.1, linestyle='--', alpha=0.7, label=r'RE23', color='blue')
        ax.fill_betweenx(y=[6.1-3.2, 6.1+9.3], x1=0.07, x2=1000, color='blue', alpha=0.1)
        
        ax.set_xscale('log')
        ax.set_xlim(0.07, 1000)
        ax.set_ylim(0, 25)

        if legend_flag: ax.legend(loc='best', fontsize=self.fontsize-2, framealpha=0.9)        
        # xticks and yticks fontsize
        ax.tick_params(axis='both', which='major', labelsize=self.fontsize-2)
        ax.tick_params(axis='both', which='both', direction='in')

        if xlabel == True:
            ax.set_xlabel(r'Local BNS Rate $\mathcal{R}_{BNS}(0)$ [Gpc$^{-3}$yr$^{-1}$]', fontsize=self.fontsize)
        labels = {
            0: r'Top-Hat $\theta^*$ [deg]',
            1: r'Flat $\theta^*$ [deg]',
            2: r'Log-Normal $\theta^*$ [deg]'
        }
        ax.set_ylabel(labels.get(label_flag, r'$\theta^*$'), fontsize=self.fontsize)

        fig.tight_layout()
        if not external_ax:
            self._save_and_show(fig, filename, dpi=600)
            
        return fig, ax, theta_stars

    # NEW 
    def _plot_grid_base(self, filename: str, draw_cell_callback: Callable, 
                        grid_lines: List[float], min_log: float, max_log: float,
                        label_generator: Optional[Callable] = None, 
                        legend_handles: List = None):
        """
        Generic plotting grid-based violin plots.
        
        Args:
            draw_cell_callback: func(ax, data, x_pos, y_base, rate_color) -> None
            label_generator: func(y_base, log_range, y_interp_func) -> None (draws labels on right)
        """
        #all_models = sorted(list(set(d["name"] for d in self.model_data)))
        all_models = self.model_dic.keys()
        fig, ax = plt.subplots(1, 1, figsize=(20, 8))
        
        # Colorbar Setup
        sm = cm.ScalarMappable(cmap=self.cmap, norm=self.norm)
        sm.set_array([])
        
        model_to_idx = {m: i for i, m in enumerate(all_models)}
        y_step = 1.15
        
        for i_alpha, alpha in enumerate(self.alpha_keys):
            y_base = i_alpha * y_step
            
            # Draw Horizontal Grid Lines
            for val in grid_lines:
                y_grid = y_base + np.interp(np.log10(val), [min_log, max_log], [0, 0.9])
                ax.axhline(y=y_grid, color='gray', ls='--', lw=0.5, alpha=0.3, zorder=1)

            # Draw Right-Side Labels
            if label_generator:
                log_range = np.linspace(min_log, max_log, 100)
                y_pos_func = lambda v: y_base + np.interp(v, [min_log, max_log], [0, 0.9])
                label_generator(ax, len(all_models), y_pos_func)

            # Draw Data
            for model_name in all_models:
                data = self.data_lookup.get((model_name, alpha))
                if not data or np.isnan(data["rate"]): continue
                
                draw_cell_callback(ax, data, model_to_idx[model_name], y_base, 
                                   self.cmap(self.norm(data["rate"])))

        # Final Formatting
        ax.set_xticks(np.arange(len(all_models)))
        ax.set_xticklabels([self.model_dic.get(m, m) for m in all_models], 
                           rotation=45, ha="right", fontsize=self.fontsize-2)
        ax.set_xlim(-0.51, len(all_models) - 0.49)
        
        ax.set_yticks([i * y_step + 0.45 for i in range(len(self.alpha_keys))])
        ax.set_ylabel(r"$\alpha_{\mathrm{CE}}$", fontsize=self.fontsize)
        ax.set_yticklabels([f"${a[1:]}$" for a in self.alpha_keys], fontsize=self.fontsize)
        ax.set_ylim(-0.3, len(self.alpha_keys) * y_step - 0.1)

        if legend_handles:
             ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 1.102), 
                       ncol=len(legend_handles), fontsize=self.fontsize - 5)

        cbar = fig.colorbar(sm, ax=ax, aspect=40, pad=0.06)
        cbar.set_label(r'Local BNS Rate $\mathcal{R}_{\rm BNS}(0)$ [Gpc$^{-3}$ yr$^{-1}$]', fontsize=self.fontsize)
        cbar.locator = LogLocator(base=10, subs=(1.0,), numticks=4)
        cbar.formatter = ScalarFormatter()
        #cbar.ax.tick_params(labelsize=self.base_fontsize - 2)
        cbar.ax.tick_params(labelsize=self.fontsize-2)
        
        self._save_and_show(fig, filename, dpi=300)

class FjModelPlotter(BaseModelPlotter):
    """
    Plotter for models where f_j and theta_c are free parameters (6 parameters).
    Directly uses f_j posteriors.
    """
    
    def __init__(self, *args, theta_c_idx: int = 4, fj_idx: int = 5, **kwargs):
        self.theta_c_idx = theta_c_idx
        self.fj_idx = fj_idx
        # Simply call super, no loop here!
        super().__init__(*args, **kwargs)

    def _extract_model_specific_samples(self, data: Dict[str, Any]):
        # This is called automatically during initialization for each model
        data["theta_c_samples"] = data["all_samples"][:, self.theta_c_idx]
        data["fj_samples"]      = data["all_samples"][:, self.fj_idx]

class LogNormalModelPlotter(BaseModelPlotter):
    """
    Plotter for models where theta_c follows a log-normal distribution (6 parameters).
    theta_c_med_10 is stored as log10(theta_c_median).
    """
    
    def __init__(self, *args, theta_c_idx: int = 4, fj_idx: int = 5, **kwargs):
        self.theta_c_idx = theta_c_idx
        self.fj_idx = fj_idx
        super().__init__(*args, **kwargs)

    def _extract_model_specific_samples(self, data: Dict[str, Any]):
        data["theta_c_samples"] = data["all_samples"][:, self.theta_c_idx]
        data["fj_samples"] = data["all_samples"][:, self.fj_idx]

    def plot_lognormal_tail_fraction_vs_rate(self, cut_deg: float = 15.0, 
                                             sigma_log10: float = 0.5,
                                             filename: str = "lognormal_tail_fraction_vs_rate.pdf", 
                                             beaming_func: callable = None,
                                             xlabel: bool = False, 
                                             ax = None):
        """
        Plot the fraction of the Log-Normal distribution above `cut_deg` vs local BNS rate.
        Uses the inferred median theta_c (theta*) for each model.
        """
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
            external_ax = False
        else:
            fig = ax.get_figure()
            external_ax = True
        
        # Log-Normal parameters
        s = sigma_log10 * np.log(10)
        t_min, t_max = 1.0, 45.0
        
        model_fractions = {}

        for alpha, models, style in self._iter_alpha_groups():
            rates, fractions = [], []

            for data in models:
                if "theta_c_samples" not in data: continue

                theta_star, _, _ = self.calculate_theta_star_from_beaming(
                    data["theta_c_samples"], data["fj_samples"], beaming_func,
                    boundary_limit=9.0, n_bins=50
                )

                if np.isnan(theta_star): continue
                
                # Calculate Tail Fraction
                scale = theta_star
                
                # Normalization
                cdf_max = lognorm.cdf(t_max, s=s, scale=scale)
                cdf_min = lognorm.cdf(t_min, s=s, scale=scale)
                cdf_total = cdf_max - cdf_min
                
                if cdf_total <= 0: continue
                
                # Mass above cut
                if cut_deg >= t_max:
                    frac = 0.0
                else:
                    cdf_cut = lognorm.cdf(max(cut_deg, t_min), s=s, scale=scale)
                    frac = (cdf_max - cdf_cut) / cdf_total
                
                rates.append(data["rate"])
                fractions.append(frac)
                model_fractions[data["full_name"]] = frac

            ax.scatter(rates, fractions, s=180, alpha=0.8, edgecolors='k', linewidth=0.5, **style)
        

        ax.axvspan(7.6, 250, alpha=0.2, color='gray', label='GWTC-4', zorder=0)
        
        ax.set_xscale('log')
        ax.set_xlim(0.07, 1000)
        ax.set_ylim(0, 0.6)
        
        if xlabel:
            ax.set_xlabel(r'Local BNS Rate $\mathcal{R}_{BNS}(0)$ [Gpc$^{-3}$yr$^{-1}$]', fontsize=self.fontsize)
        ax.set_ylabel(fr'Log-Normal $P(\theta_c > {cut_deg:.0f}^\circ)$', fontsize=self.fontsize)
        ax.tick_params(axis='both', which='major', labelsize=self.fontsize-2)
        ax.tick_params(axis='both', which='both', direction='in')
        if not external_ax:
            self._save_and_show(fig, filename, dpi=300)
        
        return fig, ax, model_fractions

class ThetaCModelPlotter(BaseModelPlotter):
    """
    Plotter for THETA_C model: directly samples theta_c (degrees) and f_j.
    Similar to FjModelPlotter but theta_c is NOT the max of a distribution.
    
    Parameters are: [A_index, L_L0, L_mu_E, sigma_E, theta_c_deg, f_j]
    """

    def __init__(self, *args, theta_c_idx: int = 4, fj_idx: int = 5, **kwargs):
        self.theta_c_idx    = theta_c_idx
        self.fj_idx         = fj_idx
        super().__init__(*args, **kwargs)

    def _extract_model_specific_samples(self, data: Dict[str, Any]):
        data["theta_c_samples"] = data["all_samples"][:, self.theta_c_idx]
        data["fj_samples"]      = data["all_samples"][:, self.fj_idx]
        # Pre-calculate epsilon to save time in loops epsilon = f_j * (1 - cos(theta_c))
        theta_rad = np.deg2rad(data["theta_c_samples"])
        data["epsilon_samples"] = data["fj_samples"] * (1 - np.cos(theta_rad))


    def plot_violins_epsilon(self, filename="epsilon_violin_plot.pdf"):
        min_log, max_log = -4, 0
        
        def cell_drawer(ax, data, x, y_base, color):
            valid = data["epsilon_samples"]
            valid = valid[(valid > 1e-5) & (valid <= 1)]
            if len(valid) == 0: return
            
            draw_vertical_violin(ax, np.log10(valid), x, y_base, color, 
                               width=0.4, min_log=min_log, max_log=max_log)
            
            # Add text stats
            log_v = np.log10(valid)
            med = np.median(log_v)
            p05, p95 = np.percentile(log_v, [5, 95])
            ax.text(x, y_base - 0.165, f"${med:.2f}_{{-{med-p05:.2f}}}^{{+{p95-med:.2f}}}$",
                    fontsize=self.fontsize-12, ha='center', va='bottom')

        def label_gen(ax, x_end, y_map):
            for val, lab in zip([1e-4, 1], ["- $\\epsilon=10^{-4}$", "- $\\epsilon=1$"]):
                ax.text(x_end-0.5, y_map(np.log10(val)), lab, fontsize=15.5, va="center")

        self._plot_grid_base(filename, cell_drawer, [1e-4, 1e-3, 1e-2, 1e-1, 1], 
                             min_log, max_log, label_generator=label_gen)

    def plot_violins_fj_at_fixed_angles(self, theta_c_values=[5, 10, 20], filename="fj_fixed.pdf"):
        min_log, max_log = -1, 1
        beam_denoms = [(1 - np.cos(np.deg2rad(th))) for th in theta_c_values]
        offsets = [(i - (len(theta_c_values) - 1) / 2) * 0.23 for i in range(len(theta_c_values))]
        patterns = ["///", "---", "xxx"]
        
        def cell_drawer(ax, data, x, y_base, color):
            for j, denom in enumerate(beam_denoms):
                f_j = data["epsilon_samples"] / denom
                valid = f_j[(f_j > 1e-3) & (f_j <= 10.0)]
                if len(valid) <= 0: continue
                draw_vertical_violin(ax, np.log10(valid), x + offsets[j], y_base, 
                                    color, hatch=patterns[j], width=0.135,
                                    min_log=min_log, max_log=max_log)

        def label_gen(ax, x_end, y_map):
            for val, lab in zip([1e-1, 1, 10], ["- $f_j=0.1$", "- $f_j=1$", "- $f_j=10$"]):
                ax.text(x_end-0.5, y_map(np.log10(val)), lab, fontsize=15.5, va="center")

        legend_h = [Patch(facecolor="gray", hatch=p, label=f"$\\theta_c={th}^\\circ$") 
                   for p, th in zip(patterns, theta_c_values)]

        self._plot_grid_base(filename, cell_drawer, [1e-1, 1, 10], min_log, max_log, 
                             label_generator=label_gen, legend_handles=legend_h)

    def plot_fj_fraction_vs_rate(self, theta_c_values: List[float] = [5, 10, 20], 
                                fj_threshold: float = 1.0, 
                                filename: str = "fj_fraction_vs_rate.pdf"):
        """
        Plot the fraction of implied f_j samples <= threshold vs local BNS rate 
        for different theta_c assumptions.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # --- 1. Setup Styles & Constants ---
        colors = [self.cmap(i) for i in np.linspace(0.1, 0.9, len(theta_c_values))]
        markers = ["o", "s", "^", "D", "v", "<", ">"]
        
        beam_factors = [1 - np.cos(np.deg2rad(th)) for th in theta_c_values]

        for i, (theta_c, beam_factor) in enumerate(zip(theta_c_values, beam_factors)):
            color = colors[i]
            adjusted_threshold = fj_threshold * beam_factor
            
            for j, alpha in enumerate(self.alpha_keys):
                marker          = markers[j % len(markers)]
                group_rates     = []
                group_fractions = []

                for data in self.model_data:
                    if data["alpha"] != alpha or np.isnan(data["rate"]): continue

                    epsilon = data.get("epsilon_samples")
                    frac    = np.mean(epsilon <= adjusted_threshold)
                    
                    group_rates.append(data["rate"])
                    group_fractions.append(frac)
                

                ax.scatter(group_rates, group_fractions, color=color, marker=marker, 
                            s=180, alpha=0.8, edgecolors='k', linewidth=1)

        legend_elements = []
        
        # Color Legend (Theta_c)
        for i, theta_c in enumerate(theta_c_values):
            legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                        label=f"$\\theta_c = {theta_c}^\\circ$",
                                        markerfacecolor=colors[i], markersize=10))
            
        # Marker Legend (Alpha)
        for i, alpha in enumerate(self.alpha_keys):
            m = markers[i % len(markers)]
            legend_elements.append(Line2D([0], [0], marker=m, color='w', 
                                        label=f"$\\alpha_{{CE}} = {alpha[1:]}$",
                                        markerfacecolor='k', markersize=8))

        ax.axvspan(7.6, 250, alpha=0.15, color='gray', label='GWTC-4')
        legend_elements.append(Patch(facecolor='gray', alpha=0.15, label='GWTC-4'))

        ax.set_xscale('log')
        ax.set_xlim(0.1, 1200)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel(r'$\mathcal{R}_{BNS}(0)$ [Gpc$^{-3}$yr$^{-1}$]', fontsize=self.fontsize)
        ax.set_ylabel(f"P($f_j \\leq {fj_threshold}$)", fontsize=self.fontsize)
        ax.legend(handles=legend_elements, loc='best', fontsize=self.fontsize-4, ncol=1, handletextpad=0.11)

        ax.tick_params(labelsize=self.fontsize - 2, direction='in', which='both')
        self._save_and_show(fig, filename, dpi=600)