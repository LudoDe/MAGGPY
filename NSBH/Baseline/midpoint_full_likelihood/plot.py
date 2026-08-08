import corner
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from mcmc_runner import _load_chain, labels, FJ_BNS_MAX

def plot_corner(alphas, burn_frac: float = 0.33, thin: int = 10):

    for alpha in alphas:
        backend, flat, backend_path = _load_chain(alpha, burn_frac, thin)
        if flat is None or flat.size == 0:
            print(f"Warning: Results missing or empty for alpha={alpha}")
            continue

        fig = corner.corner(
            flat,
            labels=labels,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 11},
            label_kwargs={"fontsize": 12},
        )
        fig.suptitle(f"$\\alpha = {alpha}$", fontsize=14)
        plt.savefig(
            f"complete_corner_midpoint_alpha_{alpha}.pdf",
            dpi=150,
            bbox_inches="tight",
        )
        plt.show()

bin_ranges = {
    0: (1.5, 5),            # A_index
    1: (-2, 6),             # L_L0
    2: (1, 25),             # theta_c_bns
    3: (0, FJ_BNS_MAX),     # fj_bns
    4: (1, 50),             # theta_c_nsbh
}

def plot_corner_cut(alphas, burn_frac: float = 0.33, thin: int = 10, fj_cut = 1.0):

    for alpha in alphas:
        _, flat, _ = _load_chain(alpha, burn_frac, thin)
        if flat is None or flat.size == 0:
            print(f"Warning: Results missing or empty for alpha={alpha}")
            continue

        fj_samples  = flat[:, 3]  # fixes the undefined name in plot_corner_cut
        mask        = fj_samples < fj_cut #! careful index has changed due to removal of L_mu_E and sigma_E
        flat_leq    = flat[mask]
        flat_geq    = flat[~mask] 
        color_leq   = "k"
        color_geq   = "red"

        corner_args = {
            "labels"            : labels,
            "quantiles"         : [0.16, 0.5, 0.84],
            "show_titles"       : True,
            "title_kwargs"      : {"fontsize": 11},
            "label_kwargs"      : {"fontsize": 12},
            "bins"              : 15,
            "smooth"            : 1.0,
            "range"             : [bin_ranges[i] for i in range(len(labels))],
            "plot_datapoints"   : False,
            "plot_density"      : False,
            "fill_contours"     : True,
            "levels"            : [0.68, 0.95],
        }

        weights_geq = np.ones(len(flat_geq)) * (len(flat_leq) / len(flat_geq))

        fig_leq = corner.corner(
            flat_leq,
            color=color_leq,
            **corner_args
        )
        corner.corner(
            flat_geq,
            color=color_geq,
            fig = fig_leq,
            weights=weights_geq,
            **corner_args
        )

        rect_labels = [f"$f_j^{{BNS}} < {fj_cut}$", f"$f_j^{{BNS}} \\geq {fj_cut}$"]
        rect_colors = [color_leq, color_geq]
        rect_patches = [Rectangle((0, 0), 1, 1, color=c, alpha=0.18) for c in rect_colors]
        handles = rect_patches
        fig_leq.legend(handles, rect_labels, loc="upper right", fontsize=12)

        fig_leq.suptitle(f"$\\alpha = {alpha}$", fontsize=14)
        plt.savefig(
            f"complete_corner_midpoint_alpha_{alpha}_comp.pdf",
            dpi=150,
            bbox_inches="tight",
        )
        plt.show()

