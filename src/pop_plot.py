"""
pop_plot.py
==========================
Centralized module for GW population analysis, data preprocessing, and visualization.

Usage:
    from src.population_plots import PopulationAnalysis
    
    pop = PopulationAnalysis(output_folder="path/to/output", images_folder="path/to/images")
    pop.run_preprocessing()
    pop.plot_combined_analysis()
    pop.plot_fj_violins()
    pop.plot_sgrb_rate_posteriors()
    pop.get_physical_models_list()
"""

import json
import emcee
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.stats import gaussian_kde
from matplotlib.ticker import LogLocator, ScalarFormatter
from typing import Dict, List, Any


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

ordered_model_dict = {
  "fiducial_Hrad": "F",
  "fiducial_kick265": "K265",
  "fiducial_kick150": "K150",
  "fiducial_qcbse": "QCBSE",
  "fiducial_rad": "QCBB",
  "fiducial_fmtbse": "RBSE",
  "fiducial_klencki": "LK",
  "fiducial_l01": "LC",
  "fiducial_xuli": "LX",
  "fiducial_HGoptimistic": "OPT",
  "fiducial_pisnfarmer19": "F19",
  "fiducial_Hrad_5M": "F5M",
  "fiducial_delayed": "SND",
  "fiducial_qhe": "QHE",
  "fiducial_notides": "NT",
  "fiducial_notides_pericirc": "NTC"
}

models = list(ordered_model_dict.values())

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
        self.evs_yr             = load_json_dict(self.datafiles_dir / "evs_yr.json")
        self.evs_GPC3_yr_local  = load_json_dict(self.datafiles_dir / "evs_GPC3_yr_local.json")
        self.model_dic          = load_json_dict(self.datafiles_dir / "model_dic.json")
        
        self.data: List[Dict[str, Any]] = []

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

    def plot_combined_analysis(self, figsize=(16, 12), f_max=5, fj_threshold=1):
        if not self.data: return print("No data.")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize, sharey='row')
        (ax_med_loc, ax_med_tot), (ax_quant_loc, ax_quant_tot) = axes
        
        markers = mlines.Line2D.filled_markers
        
        models = self.model_dic.keys()
        model_markers = {m: markers[i % len(markers)] for i, m in enumerate(models)}#self.unique_models)}
        
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

        # R22 / L25 references — labels left, arrows left->right pointing to the points
        pt_r22 = (365, 0.26)
        pt_l25 = (107, 0.7)

        # use x for r22 and l25 and write R22/L25 below no arrows
        # color that is evident but not too much
        color_l22_25 = 'darkred'

        lw = 3
        ax_med_loc.scatter(pt_r22[0], pt_r22[1], color=color_l22_25, marker='x', s=180, zorder=11, linewidth=lw)
        ax_med_loc.scatter(pt_l25[0], pt_l25[1], color=color_l22_25, marker='x', s=180, zorder=11, linewidth=lw)
        ax_med_loc.text(pt_l25[0], pt_l25[1] - 0.11, "L25", va="top", ha="center", fontsize=self.base_fontsize-3, zorder=11)
        ax_med_loc.text(pt_r22[0], pt_r22[1] - 0.11, "R22", va="top", ha="center", fontsize=self.base_fontsize-3, zorder=11)

        for ax in axes.flatten():   
            ax.tick_params(direction='in', top=True, right=True, which='both', labelsize=self.base_fontsize - 4)
            ax.set_xscale('log')
            
        ax_med_loc.set_ylim(-0.06, f_max)
        ax_quant_loc.set_ylim(-0.05, 1.05)
        
        ax_med_loc.set_ylabel(r'Median $f_j$', fontsize=self.base_fontsize)
        ax_quant_loc.set_ylabel(f"P($f_j \\leq {fj_threshold}$)", fontsize=self.base_fontsize)
        ax_quant_loc.set_xlabel(r'$\mathcal{R}_{\text{BNS}}(0)$ [Gpc$^{-3}$yr$^{-1}$]', fontsize=self.base_fontsize)
        ax_quant_tot.set_xlabel(r'$\Lambda$ [yr$^{-1}$]', fontsize=self.base_fontsize)

        gwtc_handle = mpatches.Patch(color='gray', alpha=0.3, label='GWTC-4')
        r22_handle = mlines.Line2D([], [], color=color_l22_25, marker='x', linestyle='None', 
                                   markersize=10, markeredgewidth=lw, label="R22/L25")

        all_handles = [gwtc_handle, r22_handle] + \
                      list(alpha_handles.values()) + list(model_handles.values())
        # Sort handles by the predefined order
        sorted_alpha_handles = [alpha_handles[a] for a in self.DEFAULT_ALPHA_KEYS if a in alpha_handles]
        sorted_model_handles = [model_handles[m] for m in models if m in model_handles]

        all_handles = [gwtc_handle, r22_handle] + sorted_alpha_handles + sorted_model_handles

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
        
        # from self.model_dic which is just the same as ordered_model_dict get the model names (keys)
        models = self.model_dic.keys()

        #models = self.unique_models
        
        #print(self.unique_models, models)
        # juet get the keys from sorted model dict like "fiducial_Hrad"
        #models = ordered_model_dict.values()
        #print(self.unique_models, models)

        alphas = self.DEFAULT_ALPHA_KEYS
        
        y_step = 1.15
        min_fj, max_fj = 0, 10
        
        for i, alpha in enumerate(alphas):
            y_base = i * y_step
            
            # Grid lines
            for val in [0, 5, 10]:
                y = y_base + np.interp(val, [min_fj, max_fj], [0, 0.9])
                ax.axhline(y, color='gray', alpha=0.3, lw=0.5)
                ax.text(len(models) - 0.55, y-0.01, f"- $f_j={val}$", fontsize=self.base_fontsize-8.5, va='center', ha='left')
                
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
                        fontsize=self.base_fontsize-8.5, ha='center', va='top')

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
            mpatches.Patch(facecolor='gray', alpha=0.3, edgecolor='gray',
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
        
        def plot_dataset(ax, dataset, marker):
            for d in dataset:
                if np.isnan(d["rate"]): continue
                c = self.DEFAULT_COLORS.get(d["alpha"], "k")
                med = np.median(d["fj_samples"])
                ax.scatter(d["total_rate"], med, c=[c], marker=marker, s=120, edgecolors='k', alpha=0.8)
                
        # plot bounded on the left
        data_bounded = bounded_analyzer.data
        plot_dataset(ax1, data_bounded, 'o')
        # plot unbounded on the right
        plot_dataset(ax2, self.data, 'o')
        # add text
        ax1.text(0.5, 0.96, "Bounded ($f_j \leq 1$)", transform=ax1.transAxes, fontsize=self.base_fontsize-2, ha='center', va='top')
        ax2.text(0.5, 0.96, "Unbounded", transform=ax2.transAxes, fontsize=self.base_fontsize-2, ha='center', va='top')

        ax1.set_ylim(0, f_max)
        
        ax1.set_xlabel(r"$\Lambda$ [yr$^{-1}$]", fontsize=self.base_fontsize)
        ax1.set_ylabel(r'Median jet fraction $f_j$', fontsize=self.base_fontsize)
        ax2.set_xlabel(r"$\Lambda$ [yr$^{-1}$]", fontsize=self.base_fontsize)

        ax1.set_xscale('log')
        ax2.set_xscale('log')
        ax1.tick_params(direction='in', top=True, right=True, which='both', labelsize=self.base_fontsize - 4)
        ax2.tick_params(direction='in', top=True, right=True, which='both', labelsize=self.base_fontsize - 4)

        legend_elements = []
        for alpha in self.DEFAULT_ALPHA_KEYS:
            legend_elements.append(
                mlines.Line2D([0], [0], marker='o', color='w', label=self.DEFAULT_LABELS[alpha],
                                markerfacecolor=self.DEFAULT_COLORS[alpha], markersize=10, markeredgecolor='k')
            )

        ax2.legend(handles=legend_elements, fontsize=self.base_fontsize - 4, loc='upper right')

        fig.tight_layout()
        self._save_plot("comparison_bounded_unbounded")
        plt.show()

    def get_physical_models_list(self):
        """Prints a formatted list of physical models with their statistics."""
        models = []
        for d in self.data:
            fj_samples = d["fj_samples"]
            median_fj = np.median(fj_samples)
            fj_5, fj_95 = np.percentile(fj_samples, [5, 95])
            r0 = d["rate"] if median_fj <= 1 else np.nan
            
            models.append({
                "model": d["model"],
                "label": d["label"],
                "alpha": d["alpha"],
                "alpha_label": self.DEFAULT_LABELS.get(d["alpha"], d["alpha"]),
                "R0": r0,
                "median_fj": median_fj,
                "fj_5": fj_5,
                "fj_95": fj_95
            })
        
        # Sort by alpha then model name
        models.sort(key=lambda x: (x["alpha"], x["model"]))
        
        # Print Header
        print(f"{'='*100}")
        print(f"{'Idx':<4} {'Model':<25} {'Alpha_CE':<20} {'R0 [Gpc^-3 yr^-1]':<18} {'Median f_j':<15} {'f_j [5%, 95%]':<20}")
        print(f"{'-'*100}")
        
        for i, m in enumerate(models, start=1):
            r0_str = f"{m['R0']:.2f}" if not np.isnan(m['R0']) else "N/A"
            f5 = m['fj_5']
            f95 = m['fj_95']
            print(f"{i:<4} {m['label']:<25} {m['alpha_label']:<20} {r0_str:<18} {m['median_fj']:.3f}<15 {f'[{f5:.3f}, {f95:.3f}]':<20}")

    def physical_models_table(self):
        # plot ordered models x alpha (ie. 1 + 4 columns)
        model_order = list(ordered_model_dict.keys())
        alphas      = self.DEFAULT_ALPHA_KEYS
        table_pd    = pd.DataFrame(index=model_order, columns=alphas)
        for d in self.data:
            model_name = d["model"]
            alpha      = d["alpha"]
            if model_name in model_order and alpha in alphas:
                fj_samples = d["fj_samples"]
                median_fj  = np.median(fj_samples)
                fj_5, fj_95 = np.percentile(fj_samples, [5, 95])
                flag = median_fj < 1
                table_pd.at[model_name, alpha] = flag# [{fj_5:.3f}, {fj_95:.3f}]"
        return table_pd
        
