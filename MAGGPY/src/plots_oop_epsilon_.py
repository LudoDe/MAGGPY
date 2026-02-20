import json
import emcee
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import gaussian_kde, binned_statistic
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d
from typing import List, Dict, Any, Optional, Tuple
from .init import create_run_dir
import matplotlib.colors as mcolors
from matplotlib import cm
from matplotlib.ticker import LogLocator, ScalarFormatter

FONTSIZE = 14

class BaseModelPlotter:
    """
    Base class with shared functionality for plotting MCMC model comparison results.
    """
    
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
        self.fontsize = fontsize
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_files_dir = data_files_dir  # Store this BEFORE calling _load_and_process_data

        data_path = Path(data_files_dir)
        self.local_rates = self._load_json(data_path / "evs_GPC3_yr_local.json")
        self.model_dic = self._load_json(data_path / "model_dic.json")
        self.total_rates = self._load_json(data_path / "evs_yr.json")  # Add this line
        
        self.model_data = self._load_and_process_data(
            samp_names, base_dir, self.local_rates, discard, thin
        )
        
        self._setup_colormap()
    
    def _load_json(self, file_path: Path) -> Dict:
        with open(file_path, "r") as f:
            return json.load(f)
    
    def _save_and_show(self, fig: plt.Figure, filename: str, **savefig_kwargs):
        if self.output_dir:
            output_path = self.output_dir / filename
            fig.savefig(output_path, bbox_inches='tight', **savefig_kwargs)
            print(f"Saved plot to {output_path}")
        plt.show()
        plt.close(fig)
    
    def _load_and_process_data(
        self, samp_names, base_dir, local_rates, discard, thin
    ) -> List[Dict[str, Any]]:
        processed_data = []
        print("Loading and processing model data...")

        pop_samples_dir = Path(self.data_files_dir) / "populations" / "samples"


        for z_name in samp_names:
            try:
                alpha = z_name[-4:]
                model_name = z_name[:-5]
                local_rate = local_rates[alpha][model_name]
                total_rate = self.total_rates[alpha][model_name]

                run_path = Path(base_dir) / z_name
                outdir = create_run_dir(run_path, use_timestamp=False, QUIET_FLAG=True)
                
                backend = emcee.backends.HDFBackend(
                    outdir / "emcee.h5", read_only=True
                )
                
                all_samples = backend.get_chain(discard=discard, thin=thin, flat=True)
                blobs = backend.get_blobs(discard=discard, thin=thin, flat=True)
                detection_efficiency = blobs["l_eff"] 
                avg_likelihood = np.mean(backend.get_log_prob(discard=discard, thin=thin, flat=True))
                
                # Load n_z_sources from population file
                n_z_sources = None
                try:
                    pop_file = pop_samples_dir / f"samples_{z_name}_BNSs.dat"
                    if pop_file.exists():
                        n_z_sources = len(np.loadtxt(pop_file))
                    else:
                        # Try alternative naming patterns
                        matches = list(pop_samples_dir.glob(f"samples*{z_name}*.dat"))
                        if matches:
                            n_z_sources = len(np.loadtxt(matches[0]))
                except Exception as e:
                    print(f"  Warning: Could not load n_z_sources for {z_name}: {e}")
                
                model_data_entry = {
                    "full_name": z_name,
                    "name": model_name,
                    "alpha": alpha,
                    "rate": local_rate,
                    "total_rate": total_rate,  # Add this
                    "all_samples": all_samples,
                    "detection_efficiency": detection_efficiency,  # Add this
                    "avg_likelihood": avg_likelihood,
                    "n_z_sources": n_z_sources,  # Add this

                } 

                self._extract_model_specific_samples(model_data_entry)
                
                processed_data.append(model_data_entry)
                
            except (FileNotFoundError, KeyError, OSError) as e:
                print(f"Could not process {z_name}. Error: {e}")
                continue
        print(f"Successfully processed {len(processed_data)} models.")
        return processed_data
    
    def _extract_model_specific_samples(self, data: Dict[str, Any]):
        """
        Hook for subclasses to extract specific parameters (theta_c, fj) from all_samples.
        Base implementation does nothing.
        """
        pass

    def _setup_colormap(self):
        if not self.model_data: return
        valid_rates = [d["rate"] for d in self.model_data if not np.isnan(d["rate"])]
        if not valid_rates:
            self.norm = plt.Normalize(0, 1)
        else:
            log_rates = np.log10(valid_rates)
            self.norm = plt.Normalize(np.min(log_rates), np.max(log_rates))
        self.cmap = plt.cm.viridis
    
    def _draw_violin(self, ax, log_data, x_pos, y_base, color, hatch=None, violin_width=0.1, min_log=-3, max_log=0):
        if len(log_data) < 5: return
        kde = gaussian_kde(log_data)
        log_range = np.linspace(min_log, max_log, 100)
        density = kde(log_range)
        density_norm = density / np.max(density) * violin_width
        y_positions = y_base + np.interp(log_range, [min_log, max_log], [0, 0.91])
        
        ax.fill_betweenx(y_positions, x_pos - density_norm, x_pos + density_norm,
                        facecolor=color, hatch=hatch, edgecolor="black", lw=0.5, alpha=0.7, zorder=2)
        
        median_y = y_base + np.interp(np.median(log_data), [min_log, max_log], [0, 0.91])
        width_at_med = density_norm[np.searchsorted(log_range, np.median(log_data))] * 0.9
        ax.plot([x_pos - width_at_med, x_pos + width_at_med], [median_y, median_y], color="k", lw=2.3, zorder=3)

    @staticmethod
    def calculate_theta_star_from_beaming(theta_samples, fj_samples, beaming_func, boundary_limit=9.0, n_bins=50):
        """
        Calculates theta* (where f_j = 1) using the provided beaming function.
        Assumes fj * beaming_func(theta) = constant.
        """
        from scipy.optimize import brentq

        min_theta, max_theta = np.min(theta_samples), np.max(theta_samples)
        if min_theta <= 0: min_theta = 1e-3 
        
        bins = np.logspace(np.log10(min_theta), np.log10(max_theta), n_bins + 1)
        
        # Calculate 95th Percentile in each bin
        p95_per_bin, bin_edges, bin_indices = binned_statistic(
            theta_samples, fj_samples, 
            statistic=lambda y: np.percentile(y, 95), 
            bins=bins
        )
        
        # Identify Valid Bins (where p95 < limit)
        valid_bin_flags = p95_per_bin < boundary_limit
        sample_bin_indices = bin_indices - 1
        
        # Create mask for valid samples
        valid_samples_mask = np.zeros(len(theta_samples), dtype=bool)
        for i, is_valid in enumerate(valid_bin_flags):
            if is_valid:
                valid_samples_mask |= (sample_bin_indices == i)
                
        if np.sum(valid_samples_mask) == 0:
            print("No valid samples found within boundary limit.")
            return np.nan, np.nan, valid_samples_mask

        theta_valid = theta_samples[valid_samples_mask]
        fj_valid = fj_samples[valid_samples_mask]
        
        # Calculate the constant C = fj * beaming_func(theta)
        beaming_vals    = beaming_func(theta_valid)
        C_samples       = fj_valid * beaming_vals
        C_est           = np.median(C_samples)
        
        # Solve for theta* where fj = 1 => beaming_func(theta*) = C_est
        def objective(t):
            return beaming_func(t) - C_est
            
        # Determine search range
        lower = 1.1 
        upper = 80
        
        # If beaming_func is an interpolator, respect its bounds
        if hasattr(beaming_func, 'x'): 
            lower = max(lower, beaming_func.x[0])
            upper = min(upper, beaming_func.x[-1])

        try:
            val_lower = objective(lower)
            val_upper = objective(upper)
            
            if np.isnan(val_lower) or np.isnan(val_upper):
                 print("Beaming function returned NaN at bounds.")
                 theta_star = np.nan
            elif val_lower * val_upper < 0:
                theta_star = brentq(objective, lower, upper)
            else:
                print("No sign change found in the beaming function within the search range.")
                theta_star = np.nan
        except Exception:
            theta_star = np.nan
            
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
        
        alpha_keys = sorted(list(set(d["alpha"] for d in self.model_data)))
        colors = ["#3333a1", "#a155e7", "#e28ca8", "#f5b57f"]
        markers = ["o", "s", "^", "D"]
        theta_stars = {} # Store results here

        for i, alpha in enumerate(alpha_keys):
            rates, thetas = [], []

            for data in self.model_data:
                if data["alpha"] != alpha or np.isnan(data["rate"]): continue

                # Ensure samples exist
                if "theta_c_samples" not in data or "fj_samples" not in data:
                    print(f"Missing samples for model {data['full_name']}")
                    continue
                    
                t_samps = data["theta_c_samples"]
                f_samps = data["fj_samples"]
                
                # Use beaming function if provided, otherwise fall back to old method
                theta_star, _, _ = type(self).calculate_theta_star_from_beaming(
                    t_samps, f_samps, 
                    beaming_func=beaming_func,
                    boundary_limit=boundary_limit, 
                    n_bins=n_bins
                )

                if not np.isnan(theta_star):
                    if label_flag == 1:
                        theta_star = (theta_star + 1) / 2
                    rates.append(data["rate"])
                    thetas.append(theta_star)
                    theta_stars[data["full_name"]] = theta_star

                else:
                    print(f"Could not compute theta* for model {data['full_name']} with local rate {data['rate']}")
            
            if rates:
                print(len(rates), len(thetas))
                ax.scatter(rates, thetas, c=colors[i % len(colors)], 
                          marker=markers[i % len(markers)], s=180, alpha=0.8, 
                          edgecolors='k', linewidth=0.5,
                          label=f'$\\alpha_{{CE}} = {alpha[1:]}$')
        
        ax.axvspan(7.6, 250, alpha=0.2, color='gray', label='GWTC-4', zorder=0)
        ax.axhline(y=6.1, linestyle='--', alpha=0.7, label=r'RE23', color='blue')
        ax.fill_betweenx(y=[6.1-3.2, 6.1+9.3], x1=0.07, x2=1000, color='blue', alpha=0.1)
        
        ax.set_xscale('log')
        ax.set_xlim(0.07, 1000)
        ax.set_ylim(0, 25)
        if legend_flag == True:
            ax.legend(loc='best', fontsize=self.fontsize-2, framealpha=0.9)        
        # xticks and yticks fontsize
        ax.tick_params(axis='both', which='major', labelsize=self.fontsize-2)
        ax.tick_params(axis='both', which='both', direction='in')

        if xlabel == True:
            ax.set_xlabel(r'Local BNS Rate $\mathcal{R}_{BNS}(0)$ [Gpc$^{-3}$yr$^{-1}$]', fontsize=self.fontsize)

        if label_flag == 0:
            ax.set_ylabel(r'Top-Hat $\theta^*$ [deg]', fontsize=self.fontsize)
        elif label_flag == 1:
            ax.set_ylabel(r'Flat $\theta^*$ [deg]', fontsize=self.fontsize)
        elif label_flag == 2:
            ax.set_ylabel(r'Log-Normal $\theta^*$ [deg]', fontsize=self.fontsize)

        #tight layout
        fig.tight_layout()
        if not external_ax:
            self._save_and_show(fig, filename, dpi=600)
            
        return fig, ax, theta_stars

    def plot_fj_fraction_vs_rate(self, fj_threshold: float = 1.0, 
                                  filename: str = "fj_fraction_vs_rate.pdf",
                                  figsize: Tuple[float, float] = (10, 8)):
        """
        Plot the fraction of f_j posterior samples <= threshold vs local BNS rate.
        
        Parameters:
        -----------
        fj_threshold : float
            Threshold value for f_j (default: 1.0)
        filename : str
            Output filename
        figsize : tuple
            Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        alpha_keys = sorted(list(set(d["alpha"] for d in self.model_data)))
        colors = ["#3333a1", "#a155e7", "#e28ca8", "#f5b57f"]
        markers = ["o", "s", "^", "D"]
        
        for i, alpha in enumerate(alpha_keys):
            rates, fractions = [], []
            
            for data in self.model_data:
                if data["alpha"] != alpha or np.isnan(data["rate"]): 
                    continue
                
                fj_samples = data["fj_samples"]
                fraction = np.mean(fj_samples <= fj_threshold)
                
                rates.append(data["rate"])
                fractions.append(fraction)
            
            if rates:
                ax.scatter(rates, fractions, c=colors[i % len(colors)], 
                          marker=markers[i % len(markers)], s=180, alpha=0.8, 
                          edgecolors='k', linewidth=0.5,
                          label=f'$\\alpha_{{CE}} = {alpha[1:]}$')
        
        # GWTC-4 band
        ax.axvspan(7.6, 250, alpha=0.2, color='gray', label='GWTC-4 (90% C.I.)', zorder=0)
        
        ax.set_xscale('log')
        ax.set_xlim(0.1, 1000)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel(r'Local BNS Rate $\mathcal{R}_{BNS}(0)$ [Gpc$^{-3}$yr$^{-1}$]', fontsize=self.fontsize)
        ax.set_ylabel(f'Quantile of $f_j$ posterior at $f_j = {fj_threshold}$', fontsize=self.fontsize)
        ax.legend(loc='upper left', fontsize=self.fontsize-3, framealpha=0.9)
        self._save_and_show(fig, filename, dpi=300)
        return fig, ax

    def plot_violins_fj(self, filename: str = "fj_violin_plot.pdf", max_fj: float = 10.0, val_vals: List[float] = None):
        """Plot violin plots of f_j distributions."""
        if val_vals is None:
            val_vals = [0.1, 1, 10]
        
        all_models = sorted(list(set(d["name"] for d in self.model_data)))
        alpha_keys = sorted(list(set(d["alpha"] for d in self.model_data)))
        model_groups = [all_models[:len(all_models)//2], all_models[len(all_models)//2:]]
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 16), sharex=False, sharey=True)
        
        for ax, models_subset in zip(axes, model_groups):
            model_to_idx = {m: i for i, m in enumerate(models_subset)}
            for alpha in alpha_keys:
                y_base = alpha_keys.index(alpha) * 1.2
                min_log, max_log = -1, 1
                
                labels = [f"- $f_j={v}$" if v < 1 else f"- $f_j={int(v)}$" for v in val_vals]
                side_val_lab = zip(val_vals, labels)

                for val in val_vals:
                    y_grid = y_base + np.interp(np.log10(val), [min_log, max_log], [0, 0.91])
                    ax.axhline(y=y_grid, color='gray', ls='--', lw=0.5, alpha=0.5, zorder=1)
                
                for model_name in models_subset:
                    data = next((d for d in self.model_data if d["name"] == model_name and d["alpha"] == alpha), None)
                    if not data or np.isnan(data["rate"]): continue
                    
                    rate_color = self.cmap(self.norm(np.log10(data["rate"])))
                    
                    if model_name == models_subset[0]:
                        log_range = np.linspace(min_log, max_log, 100)
                        y_pos = y_base + np.interp(log_range, [min_log, max_log], [0, 0.91])
                        side_val_lab_iter = zip(val_vals, labels)
                        for val, lab in side_val_lab_iter:
                            ly = y_pos[np.searchsorted(log_range, np.log10(val))]
                            ax.text(len(models_subset)-0.5, ly, lab, fontsize=self.fontsize-4, va="center")
                    
                    valid = data["fj_samples"][(data["fj_samples"] > 1e-3) & (data["fj_samples"] <= max_fj)]
                    if len(valid) > 0:
                        self._draw_violin(ax, np.log10(valid), model_to_idx[model_name], y_base, rate_color,
                                        min_log=min_log, max_log=max_log, violin_width=0.25)
                        p05, med, p95 = np.percentile(valid, [5, 50, 95])
                        ax.text(model_to_idx[model_name], y_base - 0.245,
                                f"${med:.2f}_{{-{med-p05:.2f}}}^{{+{p95-med:.2f}}}$",
                                fontsize=self.fontsize-3, ha='center', va='bottom')
            
            ax.set_xticks(np.arange(len(models_subset)))
            ax.set_xticklabels([self.model_dic.get(m, m) for m in models_subset], rotation=45, ha="right", fontsize=self.fontsize)
            ax.set_xlim(-0.5, len(models_subset) - 0.5)
        
        axes[0].set_yticks([i * 1.2 + 0.455 for i in range(len(alpha_keys))])
        axes[0].set_yticklabels([f"$\\alpha_{{CE}} = {a[1:]}$" for a in alpha_keys], fontsize=self.fontsize)
        axes[0].set_ylim(-0.3, len(alpha_keys) * 1.15)
        
        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=self.cmap, norm=self.norm), ax=axes, aspect=40, pad=0.085)
        cbar.set_label(r"$\log_{10}(\mathcal{R}_{BNS}(0))$ [Gpc$^{-3}$yr$^{-1}$]", fontsize=self.fontsize-2)
        self._save_and_show(fig, filename, dpi=300)

    def plot_sgrb_rate_posteriors(self, filename: str = "sgrb_rate_posteriors.pdf", 
                                   max_models: Optional[int] = None,
                                   theta_c_fixed: Optional[float] = None,
                                   beaming_func: Optional[callable] = None,
                                   physical_only: bool = False):
        """
        Plot posterior distributions of R_sGRB = R0 * f_j compared to literature constraints.
        
        For geometric models (theta_c, f_j), you can either:
        1. Use the sampled theta_c and f_j directly (theta_c_fixed=None)
        2. Fix theta_c to a specific value and derive f_j from epsilon (theta_c_fixed=value)
        
        Parameters
        ----------
        filename : str
            Output filename
        max_models : int, optional
            Maximum number of models to plot (subsampled by R0)
        theta_c_fixed : float, optional
            If provided, fix theta_c to this value (in degrees) and compute f_j from epsilon.
            If None, uses the median f_j from the posterior directly.
        beaming_func : callable, optional
            Custom beaming function. If None, uses (1 - cos(theta_c)).
        physical_only : bool
            If True, only include samples where f_j <= 1
        """
        # Literature constraints
        GWTC4_low, GWTC4_high = 7.6, 250
        S23_central, S23_lower, S23_upper = 180, 145, 660
        RE23_central, RE23_lower, RE23_upper = 1786, 1507, 6346
        
        fig, ax = plt.subplots(figsize=(12, 8))
        all_posteriors = []
        
        for data in self.model_data:
            R0 = data["rate"]
            if np.isnan(R0):
                continue
            
            # Get theta_c and f_j samples
            if "theta_c_samples" not in data or "fj_samples" not in data:
                continue
                
            theta_c_samples = data["theta_c_samples"]
            fj_samples = data["fj_samples"]
            
            # Compute epsilon = f_j * beaming_factor
            if beaming_func is not None:
                beaming_factor = beaming_func(theta_c_samples)
            else:
                beaming_factor = 1 - np.cos(np.deg2rad(theta_c_samples))
            
            epsilon_samples = fj_samples * beaming_factor
            
            # Determine which f_j to use for R_sGRB
            if theta_c_fixed is not None:
                # Fix theta_c and derive f_j from epsilon
                if beaming_func is not None:
                    fixed_beaming = beaming_func(theta_c_fixed)
                else:
                    fixed_beaming = 1 - np.cos(np.deg2rad(theta_c_fixed))
                fj_for_rate = epsilon_samples / fixed_beaming
            else:
                # Use the sampled f_j directly
                fj_for_rate = fj_samples
            
            # Filter for physical models if requested
            if physical_only:
                mask = fj_for_rate <= 1.0
                if np.sum(mask) < 100:  # Need enough samples
                    continue
                fj_for_rate = fj_for_rate[mask]
            
            # Check median f_j
            median_fj = np.median(fj_for_rate)
            if physical_only and median_fj > 1.0:
                continue

            if median_fj > 1:
                continue
            
            # Compute R_sGRB = R0 * f_j
            R_sgrb_samples  = R0 * fj_for_rate
            
            all_posteriors.append({
                "samples": R_sgrb_samples,
                "model": data["name"],
                "alpha": data["alpha"],
                "R0": R0,
                "full_name": data["full_name"]
            })
        
        if not all_posteriors:
            print("No physical models found!")
            return fig, ax
        
        # Subsample if needed
        if max_models and len(all_posteriors) > max_models:
            all_posteriors = sorted(all_posteriors, key=lambda x: x["R0"])
            step = len(all_posteriors) // max_models
            all_posteriors = all_posteriors[::step][:max_models]
        
        # Setup colormap based on R0
        r0_values = [p["R0"] for p in all_posteriors]
        norm = mcolors.LogNorm(vmin=min(r0_values), vmax=max(r0_values) * 2)
        cmap = cm.plasma
        
        # Grid for KDE
        x_min, x_max = 1, 10_000
        x_grid_log = np.linspace(np.log10(x_min), np.log10(x_max), 500)
        x_grid = 10 ** x_grid_log
        
        # Plot each posterior
        for post_data in all_posteriors:
            samples = post_data["samples"]
            r0 = post_data["R0"]
            color = cmap(norm(r0))
            
            try:
                # Filter valid samples for KDE
                valid_samples = samples[(samples > 0) & np.isfinite(samples)]
                if len(valid_samples) < 50:
                    continue
                    
                log_samples = np.log10(valid_samples)
                kde = gaussian_kde(log_samples, bw_method=0.2)
                pdf = kde(x_grid_log)
                pdf = pdf / pdf.max()
                ax.plot(x_grid, pdf, color=color, alpha=0.6, linewidth=1.2)
            except Exception as e:
                print(f"KDE failed for {post_data['full_name']}: {e}")
                continue
        
        # Colorbar
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label(r'Local BNS Rate $R_0$ [Gpc$^{-3}$ yr$^{-1}$]', fontsize=self.fontsize)
        cbar.ax.tick_params(labelsize=self.fontsize - 2)
        
        # GWTC-4 band
        ax.axvspan(GWTC4_low, GWTC4_high, alpha=0.1, color='blue', label='GWTC-4 BNS Rate', zorder=0)
        ax.axvline(GWTC4_low, color='blue', linestyle='--', linewidth=1.5, alpha=0.4)
        ax.axvline(GWTC4_high, color='blue', linestyle='--', linewidth=1.5, alpha=0.4)
        
        # S23 constraint
        s23_left = S23_central - S23_lower
        s23_right = S23_central + S23_upper
        ax.annotate('', xy=(s23_left * 0.9, 1.01), xytext=(s23_right * 1.1, 1.01),
                    arrowprops=dict(arrowstyle='<->', color='forestgreen', lw=2.5, alpha=0.9))
        ax.plot([s23_left, s23_right], [1.01, 1.01],
                color='forestgreen', linewidth=2.5, alpha=0.9, label='S23 Flux-Limited')
        
        # RE23 constraint
        re23_left = RE23_central - RE23_lower
        re23_right = RE23_central + RE23_upper
        ax.annotate('', xy=(re23_left * 0.9, 0.8), xytext=(re23_right * 1.1, 0.8),
                    arrowprops=dict(arrowstyle='<->', color='darkorange', lw=2.5, alpha=0.9))
        ax.plot([re23_left, re23_right], [0.8, 0.8],
                color='darkorange', linewidth=2.5, alpha=0.9, label='RE23 True')
        
        # Legend
        legend_elements = [
            Line2D([0], [0], color='blue', linewidth=2, alpha=0.3, label='GWTC-4 BNS Rate'),
            Line2D([0], [0], color='forestgreen', linewidth=2, alpha=0.9, label='S23 Flux-Limited'),
            Line2D([0], [0], color='darkorange', linewidth=2, alpha=0.9, label='RE23 True'),
        ]
        
        # Title based on theta_c_fixed
        if theta_c_fixed is not None:
            title = f'sGRB Rate Posteriors (fixed $\\theta_c = {theta_c_fixed}^\\circ$)'
        else:
            title = 'sGRB Rate Posteriors (sampled $\\theta_c$)'
        ax.set_title(title, fontsize=self.fontsize)
        
        ax.set_xlabel(r'$R_{\rm sGRB} = R_0 \times f_j$ [Gpc$^{-3}$ yr$^{-1}$]', fontsize=self.fontsize)
        ax.set_ylabel('Normalized Probability Density', fontsize=self.fontsize)
        ax.set_xscale('log')
        ax.set_xlim(1, 12000)
        ax.set_ylim(0, 1.3)
        ax.tick_params(labelsize=self.fontsize - 2)
        ax.legend(handles=legend_elements, fontsize=self.fontsize - 2, loc='upper right',
                  framealpha=0.95, edgecolor='gray')
        ax.grid(True, alpha=0.3, axis='x', linestyle='--', which='both')
        
        plt.tight_layout()
        self._save_and_show(fig, filename, dpi=300)
        return fig, ax
    

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
    
    def __init__(self, *args, theta_c_idx: int = 4, fj_idx: int = 5, log_theta: bool = True, **kwargs):
        self.theta_c_idx = theta_c_idx
        self.fj_idx = fj_idx
        self.log_theta = log_theta
        super().__init__(*args, **kwargs)

    def _extract_model_specific_samples(self, data: Dict[str, Any]):
        if self.log_theta:
            data["theta_c_samples"] = 10 ** data["all_samples"][:, self.theta_c_idx]
        else:
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
        from scipy.stats import lognorm
        #fig, ax = plt.subplots(figsize=(10, 8))
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
            external_ax = False
        else:
            fig = ax.get_figure()
            external_ax = True

        alpha_keys = sorted(list(set(d["alpha"] for d in self.model_data)))
        colors = ["#3333a1", "#a155e7", "#e28ca8", "#f5b57f"]
        markers = ["o", "s", "^", "D"]
        
        # Log-Normal parameters
        s = sigma_log10 * np.log(10)
        t_min, t_max = 1.0, 45.0
        
        model_fractions = {}

        for i, alpha in enumerate(alpha_keys):
            rates, fractions = [], []
            
            for data in self.model_data:
                if data["alpha"] != alpha or np.isnan(data["rate"]): continue
                
                model_name = data["full_name"]

                # Get theta* (robust estimate)
                if "theta_c_samples" not in data or "fj_samples" not in data:
                    continue
                    
                t_samps = data["theta_c_samples"]
                f_samps = data["fj_samples"]

                theta_star, _, _ = self.calculate_theta_star_from_beaming(
                    t_samps, f_samps, beaming_func,
                    boundary_limit=9.0, n_bins=50
                )

                if np.isnan(theta_star): continue
                
                # Calculate Tail Fraction
                scale = theta_star
                
                # Normalization (Mass inside [1, 90])
                cdf_total = lognorm.cdf(t_max, s=s, scale=scale) - lognorm.cdf(t_min, s=s, scale=scale)
                
                if cdf_total <= 0: continue
                
                # Mass above cut
                if cut_deg >= t_max:
                    frac = 0.0
                else:
                    cdf_above = lognorm.cdf(t_max, s=s, scale=scale) - lognorm.cdf(max(cut_deg, t_min), s=s, scale=scale)
                    frac = cdf_above / cdf_total
                
                rates.append(data["rate"])
                fractions.append(frac)
                model_fractions[model_name] = frac
            
            if rates:
                ax.scatter(rates, fractions, c=colors[i % len(colors)], 
                          marker=markers[i % len(markers)], s=180, alpha=0.8, 
                          edgecolors='k', linewidth=0.5,
                          label=f'$\\alpha_{{CE}} = {alpha[1:]}$')
        
        ax.axvspan(7.6, 250, alpha=0.2, color='gray', label='GWTC-4', zorder=0)
        
        ax.set_xscale('log')
        ax.set_xlim(0.07, 1000)
        ax.set_ylim(0, 0.6)
        
        if xlabel:
            ax.set_xlabel(r'Local BNS Rate $\mathcal{R}_{BNS}(0)$ [Gpc$^{-3}$yr$^{-1}$]', fontsize=self.fontsize)
        #ax.set_ylabel(fr'Fraction of $\theta_c > {cut_deg}^\circ$', fontsize=self.fontsize)
        # log normal $P(\theta_c > {cut_deg}^\circ)$
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
    
    def __init__(self, samp_names: List[str], base_dir: str, data_files_dir: str,
                 output_dir: str, discard: int = 1000, thin: int = 20, 
                 k_params: int = 6, theta_c_idx: int = 4, fj_idx: int = 5):
        super().__init__(samp_names, base_dir, data_files_dir, output_dir, 
                         discard, thin, k_params)
        self.theta_c_idx = theta_c_idx
        self.fj_idx = fj_idx
        
        # Extract theta_c and f_j samples
        for data in self.model_data:
            # theta_c is directly sampled in degrees
            data["theta_c_samples"] = data["all_samples"][:, theta_c_idx]
            data["fj_samples"]      = data["all_samples"][:, fj_idx]

    def __init__(self, *args, theta_c_idx: int = 4, fj_idx: int = 5, **kwargs):
        self.theta_c_idx = theta_c_idx
        self.fj_idx = fj_idx
        super().__init__(*args, **kwargs)

    def _extract_model_specific_samples(self, data: Dict[str, Any]):
        data["theta_c_samples"] = data["all_samples"][:, self.theta_c_idx]
        data["fj_samples"]      = data["all_samples"][:, self.fj_idx]


    def plot_violins_fj_at_fixed_angles_vert(self, theta_c_values: List[float] = [5, 10, 20], 
                                        filename: str = "fj_violin_plot_fixed_angles.pdf", 
                                        max_fj: float = 10.0):
        """
        Plot violin plots of f_j distributions assuming fixed opening angles.
        This allows comparison with Epsilon models by converting the inferred epsilon 
        (derived from theta_c and f_j samples) back to f_j at fixed angles.
        """
        all_models = sorted(list(set(d["name"] for d in self.model_data)))
        alpha_keys = sorted(list(set(d["alpha"] for d in self.model_data)))
        model_groups = [all_models[:len(all_models)//2], all_models[len(all_models)//2:]]
        hatch_patterns = ["///", "---", "xxx"]
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 16), sharex=False, sharey=True)
        
        # Color mapping (log scale on R0), same approach as plot_fj_violins
        rates = [d["rate"] for d in self.model_data if not np.isnan(d["rate"]) and d["rate"] > 0]
        vmin = min(rates)
        vmax = max(rates)
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
        cmap = cm.viridis
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        for ax, models_subset in zip(axes, model_groups):
            model_to_idx = {m: i for i, m in enumerate(models_subset)}
            for alpha in alpha_keys:
                y_base              = alpha_keys.index(alpha) * 1.2

                min_log, max_log    = -1, 1
                label_vals          = zip([1e-1, 1, 10], ["- $f_j=10^{-1}$", "- $f_j=1$",  "- $f_j=10$"])
                
                # Grid lines
                grid_lines          = [1e-1, 1, 10]
                for val in grid_lines:
                    y_grid = y_base + np.interp(np.log10(val), [min_log, max_log], [0, 0.91])
                    ax.axhline(y=y_grid, color='gray', ls='--', lw=0.5, alpha=0.5, zorder=1)
                    ax.text(len(models_subset) - 0.5, y_grid, f"- $f_j={val}$",
                                    fontsize=self.fontsize - 10, va="center", ha="left")
                
                for model_name in models_subset:
                    data = next((d for d in self.model_data if d["name"] == model_name and d["alpha"] == alpha), None)
                    if not data or np.isnan(data["rate"]): continue
                    
                    rate = data["rate"]
                    rate_color = cmap(norm(rate))
                    
                    # Calculate epsilon from samples
                    theta_c_samples = data["theta_c_samples"]
                    fj_samples = data["fj_samples"]
                    epsilon_samples = fj_samples * (1 - np.cos(np.deg2rad(theta_c_samples)))
                    
                    for i, theta_c in enumerate(theta_c_values):
                        # Calculate f_j for the fixed angle
                        f_j = epsilon_samples / (1 - np.cos(np.deg2rad(theta_c)))
                        valid = f_j[(f_j > 1e-3) & (f_j <= max_fj)]
                        
                        if len(valid) > 0:
                            self._draw_violin(ax, np.log10(valid), model_to_idx[model_name] + (i-1)*0.18,
                                            y_base, rate_color, hatch=hatch_patterns[i], min_log=min_log, max_log=max_log)
            
            ax.set_xticks(np.arange(len(models_subset)))
            ax.set_xticklabels([self.model_dic.get(m, m) for m in models_subset], rotation=45, ha="right", fontsize=self.fontsize-2)
            ax.set_xlim(-0.5, len(models_subset) - 0.5)
        
        axes[0].set_yticks([i * 1.2 + 0.455 for i in range(len(alpha_keys))])
        axes[0].set_yticklabels([f"$\\alpha_{{CE}} = {a[1:]}$" for a in alpha_keys], fontsize=self.fontsize)
        axes[1].set_yticklabels([f"$\\alpha_{{CE}} = {a[1:]}$" for a in alpha_keys], fontsize=self.fontsize)
        axes[0].set_ylim(-0.3, len(alpha_keys) * 1.15)
        
        legend_elements = [Patch(facecolor="gray", edgecolor="black", hatch=h, label=f"$\\theta_c={th}^\\circ$")
                         for h, th in zip(hatch_patterns, theta_c_values)]
        axes[0].legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, 1.11), ncol=3, fontsize=self.fontsize-5)
        
        cbar = fig.colorbar(sm, ax=axes, aspect=40, pad=0.075)
        cbar.set_label(r'Local BNS Rate $\mathcal{R}_{\rm BNS}(0)$ [Gpc$^{-3}$ yr$^{-1}$]', fontsize=self.fontsize)
        cbar.locator = LogLocator(base=10, subs=(1.0,), numticks=10)
        cbar.formatter = ScalarFormatter() # "100" instead of "10^2"
        cbar.update_ticks()
        cbar.ax.tick_params(labelsize=self.fontsize - 2)
        self._save_and_show(fig, filename, dpi=300)

    def plot_violins_epsilon_vertical(self, filename: str = "epsilon_violin_plot.pdf"):
        """
        Plot violin plots of epsilon distributions.
        Epsilon is derived from theta_c and f_j samples: epsilon = f_j * (1 - cos(theta_c))
        """
        all_models = sorted(list(set(d["name"] for d in self.model_data)))
        alpha_keys = sorted(list(set(d["alpha"] for d in self.model_data)))
        model_groups = [all_models[:len(all_models)//2], all_models[len(all_models)//2:]]
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 16), sharex=False, sharey=True)
                
        # Color mapping (log scale on R0), same approach as plot_fj_violins
        rates = [d["rate"] for d in self.model_data if not np.isnan(d["rate"]) and d["rate"] > 0]
        vmin = min(rates)
        vmax = max(rates)
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
        cmap = cm.viridis
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        for ax, models_subset in zip(axes, model_groups):
            model_to_idx = {m: i for i, m in enumerate(models_subset)}
            for alpha in alpha_keys:
                y_base = alpha_keys.index(alpha) * 1.2
                min_log, max_log = -4, 0
                
                # Grid lines
                for val in [1e-4, 1e-3, 1e-2, 1e-1, 1]:
                    y_grid = y_base + np.interp(np.log10(val), [min_log, max_log], [0, 0.91])
                    ax.axhline(y=y_grid, color='gray', ls='--', lw=0.5, alpha=0.5, zorder=1)
                
                for model_name in models_subset:
                    data = next((d for d in self.model_data if d["name"] == model_name and d["alpha"] == alpha), None)
                    if not data or np.isnan(data["rate"]): continue
                    
                    rate_color = cmap(norm(data["rate"]))
                    
                    # Labels
                    if model_name == models_subset[0]:
                        log_range = np.linspace(min_log, max_log, 100)
                        y_pos = y_base + np.interp(log_range, [min_log, max_log], [0, 0.91])
                        for val, lab in zip([1e-4, 1], ["- $\\epsilon=10^{-4}$", "- $\\epsilon=1$"]):
                            ly = y_pos[np.searchsorted(log_range, np.log10(val))]
                            ax.text(len(models_subset)-0.5, ly, lab, fontsize=self.fontsize-10, va="center", ha="left")
                    
                    # Calculate epsilon
                    t_samps = data["theta_c_samples"]
                    f_samps = data["fj_samples"]
                    epsilon_samples = f_samps * (1 - np.cos(np.deg2rad(t_samps)))
                    
                    valid = epsilon_samples[(epsilon_samples > 1e-5) & (epsilon_samples <= 1)]
                    if len(valid) > 0:
                        self._draw_violin(ax, np.log10(valid), model_to_idx[model_name], y_base, rate_color,
                                        min_log=min_log, max_log=max_log, violin_width=0.25)
                        med = np.median(np.log10(valid))
                        p05, p95 = np.percentile(np.log10(valid), [5, 95])
                        ax.text(model_to_idx[model_name], y_base - 0.245,
                                f"${med:.2f}_{{-{med-p05:.2f}}}^{{+{p95-med:.2f}}}$",
                                fontsize=self.fontsize-10, ha='center', va='bottom')
            
            ax.set_xticks(np.arange(len(models_subset)))
            ax.set_xticklabels([self.model_dic.get(m, m) for m in models_subset], rotation=45, ha="right", fontsize=self.fontsize-2)
            ax.set_xlim(-0.5, len(models_subset) - 0.5)
        
        axes[0].set_yticks([i * 1.2 + 0.455 for i in range(len(alpha_keys))])
        axes[0].set_yticklabels([f"$\\alpha_{{CE}} = {a[1:]}$" for a in alpha_keys], fontsize=self.fontsize)
        axes[1].set_yticklabels([f"$\\alpha_{{CE}} = {a[1:]}$" for a in alpha_keys], fontsize=self.fontsize)
        axes[0].set_ylim(-0.3, len(alpha_keys) * 1.15)
        
        #cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=self.cmap, norm=self.norm), ax=axes, aspect=40, pad=0.085)
        #cbar.set_label(r"$\log_{10}(\mathcal{R}_{BNS}(0))$ [Gpc$^{-3}$yr$^{-1}$]", fontsize=self.fontsize-2)
        cbar = fig.colorbar(sm, ax=axes, aspect=40, pad=0.08)
        cbar.set_label(r'Local BNS Rate $\mathcal{R}_{\rm BNS}(0)$ [Gpc$^{-3}$ yr$^{-1}$]', fontsize=self.fontsize)
        cbar.locator = LogLocator(base=10, subs=(1.0,), numticks=10)
        cbar.formatter = ScalarFormatter() # "100" instead of "10^2"
        cbar.update_ticks()
        
        self._save_and_show(fig, filename, dpi=300)

    def plot_violins_epsilon(self, filename: str = "epsilon_violin_plot.pdf"):
        """
        Plot violin plots of epsilon distributions.
        - Single panel: All 16 models side-by-side on X-axis.
        - Vertical Violins: Epsilon values on Y-axis.
        - Rows: Different Alpha_CE values stacked vertically.
        """
        all_models = sorted(list(set(d["name"] for d in self.model_data)))
        alpha_keys = sorted(list(set(d["alpha"] for d in self.model_data)))
        
        # Wide figure for side-by-side models
        fig, ax = plt.subplots(1, 1, figsize=(20, 8)) 
                        
        # Color mapping logic
        rates = [d["rate"] for d in self.model_data if not np.isnan(d["rate"]) and d["rate"] > 0]
        vmin = min(rates) if rates else 1e-1
        vmax = max(rates) if rates else 100
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
        cmap = cm.viridis
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        model_to_idx = {m: i for i, m in enumerate(all_models)}
        
        # Vertical spacing between Alpha rows
        y_step = 1.15 

        for alpha in alpha_keys:
            y_base = alpha_keys.index(alpha) * y_step
            min_log, max_log = -4, 0
            
            # Draw horizontal grid lines for this Alpha level
            # We map log10(val) to the visual Y space relative to y_base
            for val in [1e-4, 1e-3, 1e-2, 1e-1, 1]:
                y_grid = y_base + np.interp(np.log10(val), [min_log, max_log], [0, 0.9])
                ax.axhline(y=y_grid, color='gray', ls='--', lw=0.5, alpha=0.3, zorder=1)
                        
            for model_name in all_models:
                data = next((d for d in self.model_data if d["name"] == model_name and d["alpha"] == alpha), None)
                if not data or np.isnan(data["rate"]): continue
                
                rate_color = cmap(norm(data["rate"]))
                
                # Labels: Only draw the log scale labels (10^-4, 1) on the far right or left
                # Here we put them on the far right (after the last model)
                if model_name == all_models[-1]:
                    log_range = np.linspace(min_log, max_log, 100)
                    y_pos = y_base + np.interp(log_range, [min_log, max_log], [0, 0.9])
                    for val, lab in zip([1e-4, 1], ["- $\\epsilon=10^{-4}$", "- $\\epsilon=1$"]):
                        ly = y_pos[np.searchsorted(log_range, np.log10(val))]
                        ax.text(len(all_models)-0.5, ly, lab, fontsize=self.fontsize-10, va="center", ha="left")
                
                # Calculate epsilon
                t_samps = data["theta_c_samples"]
                f_samps = data["fj_samples"]
                epsilon_samples = f_samps * (1 - np.cos(np.deg2rad(t_samps)))
                
                valid = epsilon_samples[(epsilon_samples > 1e-5) & (epsilon_samples <= 1)]
                
                if len(valid) > 0:
                    # violin_width=0.4 makes them wider, reducing the gap between models
                    self._draw_violin(ax, np.log10(valid), 
                                      x_pos=model_to_idx[model_name], 
                                      y_base=y_base, 
                                      color=rate_color,
                                      min_log=min_log, max_log=max_log, 
                                      violin_width=0.4) 
                    
                    # Median Text
                    med = np.median(np.log10(valid))
                    p05, p95 = np.percentile(np.log10(valid), [5, 95])
                    
                    # Place text below the violin baseline
                    #ax.text(model_to_idx[model_name], y_base - 0.15,
                    #        f"${med:.2f}$",
                    #        fontsize=self.fontsize-10, ha='center', va='top')
                    # med + - 
                    ax.text(model_to_idx[model_name], y_base - 0.165,
                            f"${med:.2f}_{{-{med-p05:.2f}}}^{{+{p95-med:.2f}}}$",
                            fontsize=self.fontsize-10, ha='center', va='bottom')

        # X-Axis Settings
        ax.set_xticks(np.arange(len(all_models)))
        ax.set_xticklabels([self.model_dic.get(m, m) for m in all_models], rotation=45, ha="right", fontsize=self.fontsize-2)
        ax.set_xlim(-0.51, len(all_models) - 0.49)
        
        # Y-Axis Settings (Alpha labels)
        ax.set_yticks([i * y_step + 0.45 for i in range(len(alpha_keys))])
        # vertical ylabel alpha ce
        ax.set_ylabel(f"$\\alpha_{{\mathrm{{CE}}}}$", fontsize=self.fontsize)
        
        ax.set_yticklabels([f"${a[1:]}$" for a in alpha_keys], fontsize=self.fontsize)
        ax.set_ylim(-0.3, len(alpha_keys) * y_step - 0.1)

        # Colorbar
        cbar = fig.colorbar(sm, ax=ax, aspect=40, pad=0.049)
        cbar.set_label(r'Local BNS Rate $\mathcal{R}_{\rm BNS}(0)$ [Gpc$^{-3}$ yr$^{-1}$]', fontsize=self.fontsize)
        #cbar ticks size should be same minus 2
        cbar.locator = LogLocator(base=10, subs=(1.0,), numticks=4)
        cbar.formatter = ScalarFormatter()
        #cbar.set_ticks(cbar.get_ticks(), labelsize=self.fontsize - 2)
        # make ticks bigger
        cbar.ax.tick_params(labelsize=self.fontsize - 2)
        cbar.update_ticks()
        
        self._save_and_show(fig, filename, dpi=300)

    def _draw_violin(self, ax, log_data, x_pos, y_base, color, hatch=None, violin_width=0.1, min_log=-3, max_log=0):
        # Helper function (Vertical orientation)
        if len(log_data) < 5: return
        kde = gaussian_kde(log_data)
        log_range = np.linspace(min_log, max_log, 100)
        density = kde(log_range)
        # Normalize density width
        density_norm = density / np.max(density) * violin_width
        
        # Map log values to visual Y coordinates
        y_positions = y_base + np.interp(log_range, [min_log, max_log], [0, 0.9])
        
        ax.fill_betweenx(y_positions, x_pos - density_norm, x_pos + density_norm,
                        facecolor=color, hatch=hatch, edgecolor="black", lw=0.5, alpha=0.7, zorder=2)
        
        median_y = y_base + np.interp(np.median(log_data), [min_log, max_log], [0, 0.9])
        width_at_med = density_norm[np.searchsorted(log_range, np.median(log_data))] * 0.9
        ax.plot([x_pos - width_at_med, x_pos + width_at_med], [median_y, median_y], color="k", lw=2.3, zorder=3)

    def plot_fj_fraction_vs_rate(self, theta_c_values: List[float] = [5, 10, 20], 
                                fj_threshold: float = 1.0, 
                                filename: str = "fj_fraction_vs_rate.pdf"):
        """
        Plot the fraction of implied f_j samples <= threshold vs local BNS rate 
        for different theta_c assumptions.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Setup colors for theta_c
        cmap = plt.cm.viridis
        colors = [cmap(i) for i in np.linspace(0.1, 0.9, len(theta_c_values))]
        
        # Setup markers for alphas
        alpha_keys = sorted(list(set(d["alpha"] for d in self.model_data)))
        markers = ["o", "s", "^", "D", "v", "<", ">"]
        
        # Dummy handles for legend
        legend_elements = []
        
        # 1. Theta_c legend entries (Colors)
        for i, theta_c in enumerate(theta_c_values):
            legend_elements.append(Line2D([0], [0], marker='o', color='w', label=f"$\\theta_c = {theta_c}^\\circ$",
                                        markerfacecolor=colors[i], markersize=10))
            
            # Plot data
            for data in self.model_data:
                if np.isnan(data["rate"]): continue
                
                # Calculate fj samples
                if "epsilon_samples" in data:
                    epsilon_samples = data["epsilon_samples"]
                else:
                    epsilon_samples = data["fj_samples"] * (1 - np.cos(np.deg2rad(data["theta_c_samples"])))
                
                beam_factor = 1 - np.cos(np.deg2rad(theta_c))
                fj_samples = epsilon_samples / beam_factor
                
                fraction = np.mean(fj_samples <= fj_threshold)
                
                # Find marker for this alpha
                m_idx = alpha_keys.index(data["alpha"]) % len(markers)
                marker = markers[m_idx]
                
                ax.scatter(data["rate"], fraction, color=colors[i], marker=marker, s=180, alpha=0.8, edgecolors='k', linewidth=1)

        # 2. Alpha legend entries (Markers) - Black markers
        for i, alpha in enumerate(alpha_keys):
            legend_elements.append(Line2D([0], [0], marker=markers[i%len(markers)], color='w', label=f"$\\alpha_{{CE}} = {alpha[1:]}$",
                                        markerfacecolor='k', markersize=8))

        # Add GWTC-4 band
        ax.axvspan(7.6, 250, alpha=0.15, color='gray', label='GWTC-4')
        legend_elements.append(Patch(facecolor='gray', alpha=0.15, label='GWTC-4'))

        ax.set_xscale('log')
        ax.set_xlim(0.1, 1200)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel(r'Local BNS Rate $\mathcal{R}_{BNS}(0)$ [Gpc$^{-3}$yr$^{-1}$]', fontsize=self.fontsize)
        #ax.set_ylabel(f'Quantile of $f_j$ posterior at $f_j = {fj_threshold}$', fontsize=self.fontsize)
        ax.set_ylabel(f"P($f_j \\leq {fj_threshold}$)", fontsize=self.fontsize)
        ax.legend(handles=legend_elements, loc='best', fontsize=self.fontsize-4, ncol=1, handletextpad=0.11)
        
        # make the ticks inside
        ax.tick_params(labelsize=self.fontsize - 2, direction='in', which='both')
        #ax.grid(True, alpha=0.3)
        
        
        self._save_and_show(fig, filename, dpi=600)

    def plot_violins_fj_at_fixed_angles(self, theta_c_values: List[float] = [5, 10, 20],
                                        filename: str = "fj_violin_plot_fixed_angles.pdf",
                                        max_fj: float = 10.0):
        """
        Plot violin plots of f_j distributions assuming fixed opening angles.
        Single wide panel: All models side-by-side on X-axis, rows per alpha.
        For each model, show implied f_j at the fixed `theta_c_values`.
        """
        all_models = sorted(list(set(d["name"] for d in self.model_data)))
        alpha_keys = sorted(list(set(d["alpha"] for d in self.model_data)))

        # Wide figure for side-by-side models
        fig, ax = plt.subplots(1, 1, figsize=(20, 8))

        # Color mapping logic (log scale on R0)
        rates = [d["rate"] for d in self.model_data if not np.isnan(d["rate"]) and d["rate"] > 0]
        vmin = min(rates) 
        vmax = max(rates) 
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
        cmap = cm.viridis
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        model_to_idx = {m: i for i, m in enumerate(all_models)}

        # Vertical spacing between Alpha rows
        y_step = 1.15

        # Offsets to separate multiple theta_c violins per model (centered)
        n_t = len(theta_c_values)
        #offsets = [(i - (n_t - 1) / 2) * 0.18 for i in range(n_t)]
        # make the offsets a bit more
        offsets = [(i - (n_t - 1) / 2) * 0.23 for i in range(n_t)]
        hatch_patterns = ["///", "---", "xxx"][:n_t]

        for alpha in alpha_keys:
            y_base = alpha_keys.index(alpha) * y_step
            min_log, max_log = -1, 1  # log10(f_j) plotting range

            # Draw horizontal grid lines for reference f_j values
            for val in [1e-1, 1, 10]:
                y_grid = y_base + np.interp(np.log10(val), [min_log, max_log], [0, 0.9])
                ax.axhline(y=y_grid, color='gray', ls='--', lw=0.5, alpha=0.3, zorder=1)

            for model_name in all_models:
                data = next((d for d in self.model_data if d["name"] == model_name and d["alpha"] == alpha), None)
                if not data or np.isnan(data["rate"]):
                    continue

                rate_color = cmap(norm(data["rate"]))

                # Labels: only put f_j labels at far right after last model for each alpha row
                if model_name == all_models[-1]:
                    log_range = np.linspace(min_log, max_log, 100)
                    y_pos = y_base + np.interp(log_range, [min_log, max_log], [0, 0.9])
                    for val, lab in zip([1e-1, 1, 10], ["- $f_j=0.1$", "- $f_j=1$", "- $f_j=10$"]):
                        ly = y_pos[np.searchsorted(log_range, np.log10(val))]
                        ax.text(len(all_models) - 0.5, ly, lab, fontsize=self.fontsize - 10, va="center", ha="left")

                # Compute epsilon from sampled theta_c and f_j
                t_samps = data["theta_c_samples"]
                f_samps = data["fj_samples"]
                epsilon_samples = f_samps * (1 - np.cos(np.deg2rad(t_samps)))

                # For each fixed theta_c, compute implied f_j and draw violin
                for j, theta_c in enumerate(theta_c_values):
                    denom = (1 - np.cos(np.deg2rad(theta_c)))
                    # avoid division by zero or extremely small denom
                    if denom <= 0:
                        continue
                    f_j_fixed = epsilon_samples / denom
                    valid = f_j_fixed[(f_j_fixed > 1e-3) & (f_j_fixed <= max_fj)]
                    if len(valid) == 0:
                        continue

                    x_pos = model_to_idx[model_name] + offsets[j]
                    self._draw_violin(ax, np.log10(valid),
                                    x_pos=x_pos,
                                    y_base=y_base,
                                    color=rate_color,
                                    hatch=hatch_patterns[j] if j < len(hatch_patterns) else None,
                                    violin_width=0.135,
                                    min_log=min_log, max_log=max_log)

        # X-Axis Settings
        ax.set_xticks(np.arange(len(all_models)))
        ax.set_xticklabels([self.model_dic.get(m, m) for m in all_models], rotation=45, ha="right", fontsize=self.fontsize - 2)
        ax.set_xlim(-0.51, len(all_models) - 0.49)

        # Y-Axis Settings (Alpha labels)
        ax.set_yticks([i * y_step + 0.45 for i in range(len(alpha_keys))])
        ax.set_ylabel(f"$\\alpha_{{\\mathrm{{CE}}}}$", fontsize=self.fontsize)
        ax.set_yticklabels([f"${a[1:]}$" for a in alpha_keys], fontsize=self.fontsize)
        ax.set_ylim(-0.1, len(alpha_keys) * y_step - 0.19)

        # Legend for theta_c hatch patterns
        legend_elements = [Patch(facecolor="gray", edgecolor="black", hatch=h, label=f"$\\theta_c={th}^\\circ$")
                        for h, th in zip(hatch_patterns, theta_c_values)]
        ax.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, 1.102), ncol=len(legend_elements), fontsize=self.fontsize - 5)

        # Colorbar
        cbar = fig.colorbar(sm, ax=ax, aspect=40, pad=0.049)
        cbar.set_label(r'Local BNS Rate $\mathcal{R}_{\rm BNS}(0)$ [Gpc$^{-3}$ yr$^{-1}$]', fontsize=self.fontsize)
        cbar.locator = LogLocator(base=10, subs=(1.0,), numticks=4)
        cbar.formatter = ScalarFormatter()
        cbar.ax.tick_params(labelsize=self.fontsize - 2)
        cbar.update_ticks()

        self._save_and_show(fig, filename, dpi=300)