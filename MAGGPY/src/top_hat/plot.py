"""
Compact plotting utilities for Top Hat GRB model MCMC results.
"""

import emcee
import corner
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class TopHatLabels:
    """Label configurations for different Top Hat models."""
    EPSILON = {
        "params": [r"$k$", r"$\log_{10}(L_0/10^{49}\mathrm{erg/s})$", 
                   r"$\log_{10}(\mu_E/\mathrm{keV})$", r"$\sigma_E$", r"$\log(\epsilon)$"],
        "ranges": [(1.5, 6), (-2, 6), (1, 6.5), (0, 1.8), (-5, -1)],
        "log_last": True,  # Whether to log-transform the last parameter
    }
    FLAT_THETA = {
        "params": [r"$k$", r"$\log_{10}(L_0)$", r"$\log_{10}(\mu_E)$", 
                   r"$\sigma_E$", r"$\theta_{c,\max}$ (deg)", r"$f_j$"],
        "ranges": [(1.5, 6), (-2, 7), (0.1, 7), (0, 2.5), (1, 25), (0, 10)],
        "log_last": False,
    }
    LOGNORMAL_THETA = {
        "params": [r"$k$", r"$\log_{10}(L_0)$", r"$\log_{10}(\mu_E)$", 
                   r"$\sigma_E$", r"$\log_{10}(\theta_{c,\mathrm{med}})$", r"$f_j$"],
        "ranges": [(1.5, 6), (-2, 7), (0.1, 7), (0, 2.5), (np.log10(5), np.log10(25)), (0, 10)],
        "log_last": False,
    }
    LOGNORMAL_THETA_FLAT = {
        "params": [r"$k$", r"$\log_{10}(L_0)$", r"$\log_{10}(\mu_E)$", 
                   r"$\sigma_E$", r"$\theta_{c,\mathrm{med}}$", r"$f_j$"],
        "ranges": [(1.5, 6), (-2, 7), (0.1, 7), (0, 2.5), (1, 25), (0, 10)],
        "log_last": False,
    }
    THETA_C = {
        "params": [r"$k$", r"$\log_{10}(L_0)$", r"$\log_{10}(\mu_E)$", 
                   r"$\sigma_E$", r"$\theta_c$ (deg)", r"$f_j$"],
        "ranges": [(1.5, 6), (-2, 7), (0.1, 7), (0, 2.5), (0, 25), (0, 10)],
        "log_last": False,
    }

class TopHatPlotter:
    """Visualization tools for Top Hat model MCMC results."""
    
    COLORS = {'sim': '#1f77b4', 'obs': '#ff7f0e', 'fill': '#2ca02c'}
    
    def __init__(self, backend: emcee.backends.HDFBackend, output_dir: Path,
                 model_type: str = "epsilon", burn_in: int = 1000, thin: int = 15):
        self.backend = backend
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.burn_in = burn_in
        self.thin = thin
        self.model_type = model_type.upper()
        
        config          = getattr(TopHatLabels, self.model_type, TopHatLabels.EPSILON)
        self.labels     = config["params"]
        self.ranges     = config["ranges"]
        self.log_last   = config.get("log_last", False)
        
    def get_samples(self, flat: bool = True) -> np.ndarray:
        """Get processed chain samples."""
        return self.backend.get_chain(discard=self.burn_in, thin=self.thin, flat=flat)
    
    def get_blobs(self) -> Dict[str, np.ndarray]:
        """Get blob data (likelihood components)."""
        blobs = self.backend.get_blobs(discard=self.burn_in, flat=True)
        return {name: blobs[name] for name in blobs.dtype.names}
    
    def _calculate_rate(self, theta, n_detected: int, n_events: int, 
                        params_in, geometric_eff_func=None) -> float:
        """Calculate predicted rate based on model type."""
        gbm_eff = 0.6
        
        if self.model_type == "EPSILON":
            epsilon = theta[-1]
            grbs_per_year = epsilon * len(params_in.z_arr) * gbm_eff
            years_sim = n_events / grbs_per_year
            return n_detected / years_sim
        else:
            # FLAT_THETA, LOGNORMAL_THETA, or THETA_C
            theta_c_param, fj = theta[4], theta[5]
            
            if geometric_eff_func is None:
                # Default geometric efficiency calculation
                if self.model_type == "FLAT_THETA":
                    geometric_eff = 1 - np.cos(np.deg2rad(theta_c_param))
                elif self.model_type == "LOGNORMAL_THETA":
                    # theta_c_param is log10(theta_c_med)
                    theta_c_med = 10**theta_c_param
                    geometric_eff = 1 - np.cos(np.deg2rad(theta_c_med))
                elif self.model_type == "THETA_C":
                    # theta_c_param is theta_c in degrees (direct sampling)
                    geometric_eff = 1 - np.cos(np.deg2rad(theta_c_param))
            else:
                geometric_eff = geometric_eff_func(theta_c_param)
            
            physics_eff = n_detected / n_events
            total_eff = geometric_eff * physics_eff
            intrinsic_rate = fj * len(params_in.z_arr)
            return intrinsic_rate * total_eff * gbm_eff
    
    def plot_corner(self, filename: str = "corner_plot.pdf", **kwargs) -> plt.Figure:
        """Create corner plot of posterior distributions."""
        samples = self.get_samples().copy()
        print(samples.shape)
        # Log-transform last parameter if needed (epsilon model)
        if self.log_last:
            samples[:, -1] = np.log10(samples[:, -1])
        
        default_kwargs = dict(
            labels=self.labels, range=self.ranges, quantiles=[0.16, 0.5, 0.84],
            show_titles=False, title_kwargs={"fontsize": 14, "loc": "left"},
            label_kwargs={"fontsize": 16}, levels=[0.68, 0.95],
            plot_datapoints=False, 
            plot_density=False,      # Disable the density colormap
            fill_contours=True,      # Filled smooth contours only,
            bins=30,                # More bins = finer grid before smoothing
            smooth=1.5,              # Moderate smoothing
            smooth1d=1.0,
        )
        default_kwargs.update(kwargs)

        #Set1 first color
        col = [mcolors.to_hex(c) for c in plt.cm.Set1(np.linspace(0, 1, max(9, 1)))]
        default_kwargs['color'] = col[0]

        fig = corner.corner(samples, **default_kwargs)
        for ax in fig.get_axes():
            ax.grid(False)
            ax.tick_params(labelsize=20)
            ax.xaxis.label.set_fontsize(35)
            ax.yaxis.label.set_fontsize(35)
        
        #Removes the 1D histograms on the upper triangle
        #ndim = samples.shape[1]
        #axes = np.array(fig.axes).reshape((ndim, ndim))
        #for a in axes[np.triu_indices(ndim)]:
        #    a.remove()

        fig.savefig(self.output_dir / filename, dpi=600, bbox_inches='tight')
        
        # if kargs has close = False, do not close the figure (for interactive use)
        if kwargs.get("close", True):
            plt.close(fig)
        return fig
    
    def plot_likelihood_evolution(self, window: int = 200, 
                                   filename: str = "likelihood_evolution.pdf", **kargs) -> plt.Figure:
        """Plot evolution of likelihood components during MCMC."""
        blobs = self.get_blobs()
        n_walkers = self.backend.shape[0]
        
        def running_avg(arr):
            return np.convolve(arr, np.ones(window)/window, mode='valid')
        
        components = ['l_pflux', 'l_epeak', 'l_poiss', 'l_eff']
        titles = ['Peak Flux Score', 'Peak Energy Score', 'Poisson Score', 'Detection Efficiency']
        colors = plt.cm.tab10(np.linspace(0, 0.4, 4))
        
        fig, axs = plt.subplots(1, 4, figsize=(14, 3.5))
        
        for ax, comp, title, color in zip(axs, components, titles, colors):
            if comp in blobs:
                data = running_avg(blobs[comp])
                steps = np.arange(len(data)) / n_walkers
                ax.plot(steps, data, color=color, alpha=0.7, lw=1.5)
                ax.set_ylabel(title, fontsize=12)
                ax.set_xlabel('Step / Walker', fontsize=11)
        
        plt.tight_layout()
        fig.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        if kargs.get("close", True):
            plt.close(fig)
        return fig
    
    def plot_cdf_comparison(self, mc_func, params_in, distances, k_interpolator,
                            n_samples: int = 200, n_events: int = 10000,
                            geometric_eff_func=None,
                            filename: str = "cdf_comparison.pdf", **kargs) -> plt.Figure:
        """Compare simulated vs observed CDFs for pflux and epeak."""
        samples = self.get_samples()[-n_samples:]
        
        p_flx_all, e_pk_all, rates = [], [], []
        
        for theta in samples:
            results = mc_func(theta, n_events, params_in, distances, k_interpolator)
            
            trigger_mask = results["p_flux"] > 4
            analysis_mask = trigger_mask & (results["E_p_obs"] > 50) & (results["E_p_obs"] < 10000)
            
            p_flx_all.append(results["p_flux"][analysis_mask])
            e_pk_all.append(results["E_p_obs"][analysis_mask])
            
            # Calculate rate based on model type
            n_triggered = np.sum(trigger_mask)
            rate = self._calculate_rate(theta, n_triggered, n_events, params_in, geometric_eff_func)
            rates.append(rate)
        
        # Compute CDFs
        def ecdf(data):
            x = np.sort(data)
            return x, np.arange(1, len(x) + 1) / len(x)
        
        p_flx_sim, cdf_pflux_sim = ecdf(np.concatenate(p_flx_all))
        e_pk_sim, cdf_epeak_sim = ecdf(np.concatenate(e_pk_all))
        p_flx_obs, cdf_pflux_obs = ecdf(params_in.pflux_data)
        e_pk_obs, cdf_epeak_obs = ecdf(params_in.epeak_data)
        
        # Plot
        fig, axs = plt.subplots(1, 3, figsize=(14, 4))
        
        # Peak Flux
        axs[0].plot(p_flx_sim, cdf_pflux_sim, color=self.COLORS['sim'], lw=2, label='Simulated')
        axs[0].plot(p_flx_obs, cdf_pflux_obs, color=self.COLORS['obs'], lw=2, label='Observed')
        axs[0].set(xlabel=r'Peak Flux (ph/cm$^2$/s)', ylabel='CDF', xscale='log', 
                   xlim=(4, params_in.pflux_data.max()), ylim=(0, 1.01))
        axs[0].legend(fontsize=11)
        
        # Peak Energy
        axs[1].plot(e_pk_sim, cdf_epeak_sim, color=self.COLORS['sim'], lw=2)
        axs[1].plot(e_pk_obs, cdf_epeak_obs, color=self.COLORS['obs'], lw=2)
        axs[1].set(xlabel=r'Peak Energy (keV)', xscale='log',
                   xlim=(50, params_in.epeak_data.max()), ylim=(0, 1.01))
        
        # Rate Distribution
        axs[2].hist(rates, bins=15, alpha=0.7, color=self.COLORS['sim'], density=True)
        axs[2].axvline(params_in.yearly_rate, color=self.COLORS['obs'], ls='--', lw=2,
                       label=f'Expected: {params_in.yearly_rate:.1f}/yr')
        axs[2].set(xlabel='Rate (GRBs/year)', ylabel='Density')
        axs[2].text(0.05, 0.95, f'$\\mu$={np.mean(rates):.1f}\n$\\sigma$={np.std(rates):.1f}',
                    transform=axs[2].transAxes, va='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        axs[2].legend(fontsize=10)
        
        plt.tight_layout()
        fig.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        if kargs.get("close", True):
            plt.close(fig)
        return fig
    
    def plot_autocorrelation(self, c: float = 5.0, 
                              filename: str = "autocorrelation.pdf", close = True) -> plt.Figure:
        """Plot autocorrelation time convergence."""
        chain = self.backend.get_chain()
        n_steps, _, n_params = chain.shape
        n_vals = np.exp(np.linspace(np.log(100), np.log(n_steps), 10)).astype(int)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = plt.cm.viridis(np.linspace(0, 0.9, n_params))
        
        for i in range(n_params):
            tau = [np.mean(emcee.autocorr.integrated_time(chain[:n, :, i], c=c, tol=0, quiet=True)) 
                   for n in n_vals]
            ax.loglog(n_vals, tau, 'o-', color=colors[i], label=self.labels[i], alpha=0.8)
        
        ax.plot(n_vals, n_vals / 50, '--k', lw=1.5, label=r'$\tau = N/50$')
        ax.set(xlabel='Number of Samples', ylabel=r'$\hat{\tau}$', xlim=(100, n_steps))
        ax.legend(fontsize=9, ncol=2)
        
        fig.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        if close : plt.close(fig)
        return fig
    
    def summary_table(self) -> str:
        """Generate LaTeX summary table of parameter estimates."""
        samples = self.get_samples().copy()
        
        # Log-transform for display if needed
        if self.log_last:
            samples[:, -1] = np.log10(samples[:, -1])
        
        rows = []
        for i, label in enumerate(self.labels):
            lo, med, hi = np.percentile(samples[:, i], [16, 50, 84])
            rows.append(f"{label} & ${med:.3f}_{{-{med-lo:.3f}}}^{{+{hi-med:.3f}}}$ \\\\")
        
        return "\\begin{tabular}{lc}\n\\hline\nParameter & Value \\\\\n\\hline\n" + \
               "\n".join(rows) + "\n\\hline\n\\end{tabular}"
    

    def plot_corner_last_two(self, filename: str = "corner_plot_last_two.pdf", **kwargs) -> plt.Figure:
        """Create corner plot of only the last two parameters."""
        samples = self.get_samples().copy()
        
        # Log-transform last parameter if needed (epsilon model)
        if self.log_last:
            samples[:, -1] = np.log10(samples[:, -1])
            
        # Select only the last two parameters
        samples = samples[:, -2:]
        labels = self.labels[-2:]
        ranges = self.ranges[-2:]
        
        default_kwargs = dict(
            labels=labels, range=ranges, quantiles=[0.16, 0.5, 0.84],
            show_titles=False,  # No titles as requested
            label_kwargs={"fontsize": 16}, levels=[0.68, 0.95],
            plot_datapoints=False, smooth=1.0, bins=25,         
            plot_density=False,      # Disable the density colormap
            fill_contours=True,      # Filled smooth contours only,
        )
        default_kwargs.update(kwargs)

        #Set1 first color
        col = [mcolors.to_hex(c) for c in plt.cm.Set1(np.linspace(0, 1, max(9, 1)))]
        default_kwargs['color'] = col[0]

        fig = corner.corner(samples, **default_kwargs)
        for ax in fig.get_axes():
            ax.tick_params(labelsize=16)
            ax.xaxis.label.set_fontsize(18)
            ax.yaxis.label.set_fontsize(18)
            ax.tick_params(axis='both', pad=2)  # Default is typically 4-5
            # Force X Label Position (fraction relative to axes: 0=left, 1=right, <0=below)
            if ax.get_xlabel():
                ax.xaxis.set_label_coords(0.5, -0.178)
            
            # Force Y Label Position (fraction relative to axes: 0=bottom, 1=top, <0=left)
            if ax.get_ylabel():
                ax.yaxis.set_label_coords(-0.178, 0.5)
        
        fig.savefig(self.output_dir / filename, dpi=600, bbox_inches='tight')
        plt.close(fig)
        return fig


    def generate_all_plots(self, mc_func=None, params_in=None, 
                           distances=None, k_interpolator=None,
                           geometric_eff_func=None):
        """Generate all standard diagnostic plots."""
        print("Generating corner plot...")
        self.plot_corner()
        
        print("Generating likelihood evolution...")
        self.plot_likelihood_evolution()
        
        print("Generating autocorrelation plot...")
        self.plot_autocorrelation()
        
        print("Generating corner plot for last two parameters...")
        self.plot_corner_last_two()

        if all(x is not None for x in [mc_func, params_in, distances, k_interpolator]):
            print("Generating CDF comparison...")
            self.plot_cdf_comparison(mc_func, params_in, distances, k_interpolator,
                                     geometric_eff_func=geometric_eff_func)
        
        print(f"All plots saved to: {self.output_dir}")
        return self.summary_table()