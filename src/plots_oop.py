import emcee
import corner
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from IPython.display import display, Math
from typing import Dict, List, Optional, Tuple

class MCMCPlotter:
    """Encapsulates MCMC simulation analysis and plotting."""

    DEFAULT_LABELS = ["k", r"$\log_{10}(\frac{E^*}{10^{49}erg})$", r"$\log_{10}(\frac{\mu_E}{keV})$", 
                      r"$\sigma_{E}$", r"$\log_{10}(\frac{\mu_{\tau}}{s})$", r"$\sigma_{\tau}$", r"$f_j$"]
        #remove the units for now
    DEFAULT_LABELS = ["k", r"$\log_{10}(E^*)$", r"$\log_{10}(\mu_E)$", 
                      r"$\sigma_{E}$", r"$\log_{10}(\mu_{\tau})$", r"$\sigma_{\tau}$", r"$f_j$"]
    DEFAULT_RANGES = np.array([[1.5, 6], [-3, 2], [2, 5], [0.05, 1], [-2, 2.5], [0.05, 1.8], [0, 1]])
    LATEX_LABELS = [r"k", r"\log_{10}(E^*)", r"\log_{10}(\mu_E)", r"\sigma_{E}", 
                    r"\log_{10}(\mu_\tau)", r"\sigma_\tau", r"f_j"]

    def __init__(self, samplers: List[emcee.EnsembleSampler], names: Optional[List[str]] = None,
                 labels: Optional[List[str]] = None, ranges: Optional[List[Tuple[float, float]]] = None,
                 truths: Optional[List[float]] = None, output_dir: Optional[str] = None,
                 burn_in: int = 0, thin: int = 1, **kwargs):
        self.samplers = samplers if isinstance(samplers, list) else [samplers]
        self.n_chains = len(self.samplers)
        self.chains = [self._process_chain(s, burn_in, thin, kwargs.get('invert', False)) for s in self.samplers]
        self.names = names or [f"Chain {i+1}" for i in range(self.n_chains)]
        self.output_dir = Path(output_dir) if output_dir else None
        self.truths = truths
        self.n_params = self.chains[0].shape[1]
        self.labels = (labels or self.DEFAULT_LABELS)[:self.n_params]
        self.ranges = (ranges if ranges is not None else self.DEFAULT_RANGES)[:self.n_params]

    def _process_chain(self, sampler, burn_in, thin, invert):
        chain = sampler.get_chain(discard=burn_in, flat=True, thin=thin)
        if invert:
            chain[:, 0] = 1.0 / chain[:, 0]
        return chain

    def _save_and_show(self, fig: plt.Figure, filename: str, **kwargs):
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(self.output_dir / filename, **kwargs)
        plt.show()
        plt.close(fig)

    @staticmethod
    def _log_binned_avg(arr: np.ndarray, n_bins: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            return np.array([]), np.array([])
        edges = np.unique(np.logspace(0, np.log10(len(valid)), n_bins + 1).astype(int))
        centers, avgs = [], []
        for i in range(len(edges) - 1):
            s, e = int(edges[i]), min(int(edges[i+1]), len(valid))
            if s < e:
                centers.append(np.sqrt(s * e))
                avgs.append(np.mean(valid[s:e]))
        return np.array(centers), np.array(avgs)

    def plot_convergence(self, n_counts: float, sampler_index: int = 0, n_bins: int = 50, 
                         filename: str = "convergence_plot.pdf"):
        sampler = self.samplers[sampler_index]
        
        blobs = sampler.get_blobs(flat=True)
        #n_rate_yr, *probs = blobs.T
        # Handle both structured arrays (with blobs_dtype) and regular arrays
        if blobs.dtype.names is not None:
            # Structured array - extract by field name
            field_names = blobs.dtype.names
            n_rate_yr = blobs[field_names[0]]  # First field is rate
            probs = [blobs[name] for name in field_names[1:]]  # Rest are likelihood terms
        else:
            # Regular array - use transpose
            n_rate_yr, *probs = blobs.T
        
        n_walkers = sampler.shape[0]
        raw_x = np.arange(len(n_rate_yr)) / n_walkers

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(raw_x, n_rate_yr, 'k', alpha=0.05, lw=0.5, ls='--')
        x_bins, rate_avg = self._log_binned_avg(n_rate_yr, n_bins)
        ax.plot(x_bins / n_walkers, rate_avg, 'k', lw=2.5, label="GRB rate")
        ax.hlines(n_counts, 0, max(raw_x), 'r', 'dashed', label='Target rate')
        ax.fill_between([0, max(raw_x)], n_counts - n_counts**0.5, n_counts + n_counts**0.5, color='r', alpha=0.2)
        ax.set(ylabel=r'$R_{GRB} [yr^{-1}]$', xlabel=f'Iterations / {n_walkers} Walkers', 
               xscale='log', xlim=(10 / n_walkers, x_bins[-1] / n_walkers))

        ax2 = ax.twinx()
        likelihood = np.sum(probs, axis=0)
        x_like, like_avg = self._log_binned_avg(likelihood, n_bins)
        ax2.plot(raw_x, likelihood, 'r', alpha=0.05, lw=0.5)
        ax2.plot(x_like / n_walkers, like_avg, 'r', lw=2, label="Likelihood")
        ax2.set(ylabel=r'$\log(\mathcal{L}(\vec{d} | \vec{\theta}))$', ylim=(np.quantile(likelihood, 0.001), 0))

        lines = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
        labels = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
        ax.legend(lines, labels, loc='upper left', framealpha=1, fontsize=12).set_zorder(10)
        fig.tight_layout()
        self._save_and_show(fig, filename, dpi=300, bbox_inches='tight')

    def plot_diagnostics(self, sampler_index: int = 0, filename: str = "mcmc_diagnostics.pdf", **kwargs):
        samples = self.samplers[sampler_index].get_chain()
        n_steps = samples.shape[0]
        fig, axs = plt.subplots(self.n_params, 2, figsize=(12, 1.1 * self.n_params), sharex='col', squeeze=False)

        for i in range(self.n_params):
            axs[i, 0].plot(samples[:, :, i], alpha=0.3, color=kwargs.get("color_exploration", "k"))
            axs[i, 0].set(ylabel=self.labels[i], xlim=(0, n_steps))
            autocorr = emcee.autocorr.function_1d(np.mean(samples[:, :, i], axis=1))
            axs[i, 1].plot(autocorr, "k", alpha=0.8)
            axs[i, 1].axhline(0, color='r', lw=1, ls='--')
            axs[i, 1].set_xlim(0, len(autocorr))

        axs[-1, 0].set_xlabel("Step number")
        axs[-1, 1].set_xlabel("Lag (Steps)")
        axs[0, 0].set_title("Parameter Exploration", fontsize=14)
        axs[0, 1].set_title("Autocorrelation Function", fontsize=14)
        fig.tight_layout()
        self._save_and_show(fig, filename, dpi=300, bbox_inches='tight')

    DEFAULT_QUANTILES = [16, 50, 84]
    def create_parameter_table(self, quantiles=DEFAULT_QUANTILES) -> str:
        cols = "c" * (self.n_chains + (1 if self.truths else 0))
        header = rf"\begin{{array}}{{l{cols}}}" + "\n" + r"\hline" + "\n" + r"\text{Parameter} & "
        header += " & ".join(rf"\text{{{n}}}" for n in self.names)
        header += (r" & \text{Truth}" if self.truths else "") + r" \\\hline" + "\n"

        rows = []
        for i in range(self.n_params):
            results = []
            for samples in self.chains:
                lo, med, hi = np.percentile(samples[:, i], quantiles)
                results.append(f"{med:.2f}_{{-{med-lo:.2f}}}^{{+{hi-med:.2f}}}")
            row = f"{self.LATEX_LABELS[i]} & " + " & ".join(results)
            row += f" & ${self.truths[i]:.2f}$" if self.truths else ""
            rows.append(row + r" \\")
        return header + "\n".join(rows) + "\n" + r"\hline\end{array}"


    def show_parameter_table(self, quantiles=DEFAULT_QUANTILES) -> str:
        table = self.create_parameter_table(quantiles)
        display(Math(table))
        return table

    def create_corner_plot(self, filename: str = "corner_plot_comparison.pdf", **corner_kwargs):
        cols = corner_kwargs.pop('colors', [
            mcolors.to_hex(c) for c in plt.cm.Set1(
                np.linspace(0, 1, max(9, self.n_chains))
                )
            ])
        
        if invert_flag := corner_kwargs.pop('invert_flag', None):
            for i, flag in enumerate(invert_flag):
                if flag:
                    self.chains[i][:, 0] = 1.0 / self.chains[i][:, 0]
                    if self.truths:
                        self.truths[0] = 1.0 / self.truths[0]

        if log_flag := corner_kwargs.pop('log_flag', None):
            for i, flag in enumerate(log_flag):
                if flag:
                    # log scale last parameter
                    self.chains[i][:, -1] = np.log10(self.chains[i][:, -1])
                    if self.truths:
                        self.truths[-1] = np.log10(self.truths[-1])

        min_samples = min(c.shape[0] for c in self.chains)
        chains = [c[:min_samples] for c in self.chains]

        default_kwargs = dict(
            bins=30,                # More bins = finer grid before smoothing
            smooth=1.5,              # Moderate smoothing
            smooth1d=1.0,
            range=self.ranges, 
            labels=self.labels, 
            truths=self.truths,
            plot_datapoints=False, 
            show_titles=False, 
            title_kwargs={"fontsize": 24, "loc": "left"},
            label_kwargs={"fontsize": 26}, 
            levels=[0.68, 0.95], 
            plot_density=False,      # Disable the density colormap
            fill_contours=True,      # Filled smooth contours only,
        )
        default_kwargs.update(corner_kwargs)
        fig = corner.corner(chains[0], color=cols[0], **default_kwargs)
        for i, samples in enumerate(chains[1:], 1):
            corner.corner(samples, fig=fig, color=cols[i], **default_kwargs)

        for ax in fig.get_axes():
            ax.grid(False)
            ax.tick_params(labelsize=20)
            ax.xaxis.label.set_fontsize(37)
            ax.yaxis.label.set_fontsize(37)

        #ndim = self.n_params
        #axes = np.array(fig.axes).reshape((ndim, ndim))
        #for a in axes[np.triu_indices(ndim)]:
        #    a.remove()
        # make xlabels and ylabels larger

        if self.n_chains > 1:
            handles = [plt.Line2D([0], [0], color=cols[i], lw=4) for i in range(self.n_chains)]
            fig.get_axes()[0].legend(handles, self.names, loc='upper right', fontsize=10, framealpha=0.9)

        self._save_and_show(fig, filename, dpi=600, bbox_inches='tight')
        return fig

    def plot_autocorrelation_times(self, sampler_index: int = 0, c: float = 5.0, 
                                   filename: str = "autocorrelation_times.pdf"):
        chain = self.samplers[sampler_index].get_chain()
        n_steps, _, n_params = chain.shape
        n_vals = np.exp(np.linspace(np.log(100), np.log(n_steps), 10)).astype(int)

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, n_params))
        for i in range(n_params):
            tau = [np.mean(emcee.autocorr.integrated_time(chain[:n, :, i], c=c, tol=0, quiet=True)) for n in n_vals]
            ax.loglog(n_vals, tau, "o-", label=self.labels[i], color=colors[i], alpha=0.8)
        ax.plot(n_vals, n_vals / 50.0, "--k", label=r"$\tau = N/50$")
        ax.set(xlabel="Number of Samples, $N$", ylabel=r"$\hat{\tau}$", xlim=(100, n_steps))
        ax.legend(framealpha=0.9, loc="lower right")
        self._save_and_show(fig, filename)


def _empirical_cdf_with_dkw(data: np.ndarray, alpha: float = 0.1) -> Tuple[np.ndarray, ...]:
    data_sorted = np.sort(data)
    n = len(data_sorted)
    cdf = np.arange(1, n + 1) / n
    eps = np.sqrt(np.log(2.0 / alpha) / (2.0 * n))
    return data_sorted, cdf, np.maximum(0, cdf - eps), np.minimum(1, cdf + eps)


def plot_cdf_comparison(data_dict: Dict[str, np.ndarray], simulated_dicts: Optional[List[Dict]] = None,
                        output_dir: Optional[Path] = None, **kwargs):
    """Plots and compares empirical CDFs of observed vs. simulated data."""
    sim_dicts = simulated_dicts or []
    colors = kwargs.get('colors', ['k'] + plt.cm.viridis(np.linspace(0, 1, len(sim_dicts))).tolist())
    labels = kwargs.get('labels', ['Observed', 'Simulated'])
    keys = ['epeak', 't90', 'fluence', 'pflux']
    x_labels = {'epeak': 'Peak Energy (keV)', 't90': '$T_{90}$ (s)',
                'fluence': 'Fluence (erg/cm²)', 'pflux': 'Peak Flux (ph/cm²/s)'}

    fig, axs = plt.subplots(2, 2, figsize=kwargs.get('figsize', (10, 8)), sharey='row')
    
    for i, key in enumerate(keys):
        if key not in data_dict:
            continue
        ax = axs.flat[i]
        obs_data = data_dict[key]

        for j, sim_dict in enumerate(sim_dicts):
            if key not in sim_dict:
                continue
            all_sim = np.concatenate(sim_dict[key])
            x_common = np.sort(all_sim)
            ecdf_matrix = np.array([np.searchsorted(np.sort(s), x_common, side='right') / len(s) 
                                    for s in sim_dict[key]])
            lo, med, hi = np.percentile(ecdf_matrix, [5, 50, 95], axis=0)
            ax.fill_between(x_common, lo, hi, color=colors[j+1], alpha=0.3, label='$90\%$ CI')
            ax.plot(x_common, med, color=colors[j+1], ls='--', lw=2, label='Median')

        sorted_obs, cdf, *_ = _empirical_cdf_with_dkw(obs_data)
        ax.plot(sorted_obs, cdf, color=colors[0], lw=2, label=labels[0])
        
        if i == 0 and sim_dicts:
            ax.legend(loc='best', fontsize=18)
        ax.set(xlabel=x_labels.get(key, key), ylim=(0, 1), xscale=kwargs.get('xscale', 'log'))
        if i % 2 == 0:
            ax.set_ylabel('CDF', fontsize=26)
        if kwargs.get('limit_xaxis', True) and obs_data.size:
            ax.set_xlim(obs_data.min(), obs_data.max())
        ax.xaxis.label.set_fontsize(26)
        ax.tick_params(labelsize=18)

    plt.tight_layout()
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(output_dir) / 'CDF_comparison.pdf', dpi=600, bbox_inches='tight')
    print(f"Saving CDF comparison plot to {output_dir / 'CDF_comparison.pdf' if output_dir else 'current directory'}")
    plt.show()
    plt.close(fig)
    return fig, axs