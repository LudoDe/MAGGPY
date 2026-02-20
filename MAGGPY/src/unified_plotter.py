import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.stats import gaussian_kde

from .pop_plot import PopulationAnalysis
from .plots_oop_epsilon import ThetaCModelPlotter, FjModelPlotter, LogNormalModelPlotter

# Try importing the geometric efficiency generator, or define a fallback
try:
    from .top_hat.geometric_eff import create_geometric_efficiency_lognormal_interpolator
except ImportError:
    print("Warning: Could not import create_geometric_efficiency_lognormal_interpolator. Using placeholder.")
    def create_geometric_efficiency_lognormal_interpolator(sigma_theta_c=0.5, minimum_theta_c=0.01):
        # Placeholder identity for testing if file missing
        return lambda x: x 

# =============================================================================
# CONFIGURATION
# =============================================================================

# Style Configuration (A&A Guidelines)
FONTSIZE = 22
plt.rcParams.update({
    'font.size': FONTSIZE,
    'axes.labelsize': FONTSIZE,
    'xtick.labelsize': FONTSIZE - 2,
    'ytick.labelsize': FONTSIZE - 2,
    'legend.fontsize': FONTSIZE - 4,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'axes.grid': False,   # No grid for A&A
    'figure.figsize': (12, 10),
    'font.family': 'serif', # A&A usually uses serif
    'mathtext.fontset': 'dejavuserif'
})

# Directory Configuration
DATA_FILES_DIR = "datafiles" # Location of evs_yr.json, etc.

DIRS = {
    "Structured": Path("Output_files/ProductionPop_fj10_20k_init_same"),
    "Fixed":      Path("Output_files/ProductionPop_Theta_c_10k"),
    "Flat":       Path("Output_files/ProductionPop_Flat_10k_new_new"),
    "LogNormal":  Path("Output_files/ProductionPop_LN_Flat"),
}

OUTPUT_PATH = Path("Output_files/Unified_Plots")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# Sample Names (Helper to get list of models)
# We assume the structured folder has the representative list of samples
def get_sample_names():
    # You can change this to a hardcoded list if you prefer
    # samp_names = ["model_A", "model_B", ...]
    # Here we try to grab them from the folder structure dynamically
    pop_folder = Path(DATA_FILES_DIR) / "populations" / "samples"
    samples = list(pop_folder.glob("samples_*.dat"))
    return [s.name.split("samples_")[1].split("_BNSs")[0] for s in samples]

SAMP_NAMES = get_sample_names()

# =============================================================================
# HELPER FUNCTIONS (BEAMING)
# =============================================================================

def beaming_theta_c(theta_c):
    """Beaming factor for standard Theta C models"""
    return 1 - np.cos(np.deg2rad(theta_c))

def beaming_efficiency(theta_max, theta_min=1):
    """Beaming efficiency for Flat distributions"""
    sin_max = np.sin(np.deg2rad(theta_max))
    sin_min = np.sin(np.deg2rad(theta_min))
    theta_max_rad = np.deg2rad(theta_max)
    theta_min_rad = np.deg2rad(theta_min)
    return 1 - (sin_max - sin_min) / (theta_max_rad - theta_min_rad)

# =============================================================================
# PLOTTING ROUTINES
# =============================================================================

def run_general_analysis():
    print("\n--- 1. Running General Population Analysis (Structured) ---")
    
    analysis = PopulationAnalysis(
        output_folder=DIRS["Structured"],
        images_folder=OUTPUT_PATH,
        datafiles=DATA_FILES_DIR,
        base_fontsize=FONTSIZE
    )
    
    # Run Preprocessing
    analysis.run_preprocessing(samp_names=SAMP_NAMES, quiet=True)
    analyzer = analysis.analyzer
    
    # 1.1 Combined Analysis (2x2 Grid: Rates vs Fj)
    print("Generating Combined Analysis Plot...")
    fig1, _ = analyzer.plot_combined_analysis(f_max=5)
    
    # 1.2 Violin Plots
    print("Generating Violin Plots...")
    fig2, _ = analyzer.plot_fj_violins(filename="fj_violin_plot_structured")
    
    # 1.3 sGRB Rate Posteriors
    print("Generating sGRB Rate Posteriors...")
    # Note: Using the method from PopulationAnalysis/ModelAnalyzer
    analyzer.plot_sgrb_rate_posteriors(filename="sgrb_rate_posteriors_structured.pdf")

def run_posterior_comparison(fiducial_name="fiducial_Hrad_A1.0"):
    print(f"\n--- 2. Running Fj Posterior Comparison (Fiducial: {fiducial_name}) ---")
    
    models_config = {
        "Structured":         {"path": DIRS["Structured"], "fj_idx": 6, "discard": 3000, "thin": 15},
        "Fixed $\\theta_c$":  {"path": DIRS["Fixed"],      "fj_idx": 5, "discard": 1000, "thin": 20},
        "Flat $\\theta_c$":   {"path": DIRS["Flat"],       "fj_idx": 5, "discard": 5000, "thin": 20},
        "Log-Normal $\\theta_c$": {"path": DIRS["LogNormal"], "fj_idx": 5, "discard": 3000, "thin": 15},
    }

    import emcee
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Viridis colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(models_config) + 1))
    
    for i, (label, config) in enumerate(models_config.items()):
        h5_path = config["path"] / fiducial_name / "emcee.h5"
        
        if not h5_path.exists():
            print(f"Skipping {label}: {h5_path} not found.")
            continue
            
        try:
            backend = emcee.backends.HDFBackend(str(h5_path), read_only=True)
            # -1 usually captures the last parameter, which is fj in these configs
            # explicitly using the index provided in config if logic varies, 
            # but usually fj is the last or second to last param. 
            # Based on user logic:
            samples = backend.get_chain(discard=config["discard"], thin=config["thin"], flat=True)
            fj_samples = samples[:, -1] # Assuming fj is always last
            
            # Filter
            fj_samples = fj_samples[fj_samples > 0]
            
            if len(fj_samples) < 100: continue
            
            # KDE
            log_fj = np.log10(fj_samples)
            kde = gaussian_kde(log_fj, bw_method=0.3)
            x_grid = np.logspace(-3, 1, 500)
            pdf = kde(np.log10(x_grid))
            
            ax.plot(x_grid, pdf, label=label, color=colors[i], lw=3)
            ax.fill_between(x_grid, pdf, color=colors[i], alpha=0.2)
            
        except Exception as e:
            print(f"Error processing {label}: {e}")

    ax.axvline(1, color='gray', linestyle='--', alpha=0.8, lw=2)
    
    ax.set_xscale('log')
    ax.set_xlim(1e-3, 10)
    ax.set_ylim(0, None)
    ax.set_xlabel(r'$f_j$', fontsize=FONTSIZE)
    ax.set_ylabel('PDF', fontsize=FONTSIZE)
    ax.legend(loc='upper left', framealpha=0.9)
    
    out_file = OUTPUT_PATH / "fj_posteriors_comparison.pdf"
    plt.tight_layout()
    plt.savefig(out_file)
    print(f"Saved {out_file}")
    plt.show()

def run_theta_star_grid_analysis():
    print("\n--- 3. Running Theta Star 2x2 Grid Analysis ---")
    
    # 3.1 Initialize Specific Plotters
    print("Initializing Fixed Theta_c Plotter...")
    plotter_theta_c = ThetaCModelPlotter(
        samp_names=SAMP_NAMES,
        base_dir=DIRS["Fixed"].name,
        data_files_dir=DATA_FILES_DIR,
        output_dir=str(OUTPUT_PATH),
        discard=1000, thin=20, k_params=6, theta_c_idx=4, fj_idx=5,
        fontsize=FONTSIZE
    )

    print("Initializing Flat Theta Plotter...")
    plotter_flat = FjModelPlotter(
        samp_names=SAMP_NAMES,
        base_dir=DIRS["Flat"].name,
        data_files_dir=DATA_FILES_DIR,
        output_dir=str(OUTPUT_PATH),
        discard=5000, thin=20, k_params=6, theta_c_idx=4, fj_idx=5,
        fontsize=FONTSIZE
    )

    print("Initializing Log-Normal Plotter...")
    plotter_ln = LogNormalModelPlotter(
        samp_names=SAMP_NAMES,
        base_dir=DIRS["LogNormal"].name,
        data_files_dir=DATA_FILES_DIR,
        output_dir=str(OUTPUT_PATH),
        discard=3000, thin=15, k_params=6, theta_c_idx=4, fj_idx=5,
        log_theta=False, # Assuming samples are already in degrees not log10 based on prompt context
        fontsize=FONTSIZE
    )

    # 3.2 Create Lognormal Interpolator
    geom_eff_ln = create_geometric_efficiency_lognormal_interpolator(sigma_theta_c=0.5)

    # 3.3 Generate 2x2 Plot
    fig, axes = plt.subplots(2, 2, figsize=(22, 18))
    ((ax1, ax2), (ax3, ax4)) = axes

    # Panel 1: Fixed Theta C
    plotter_theta_c.plot_theta_star_robust_vs_rate(
        beaming_func=beaming_theta_c,
        legend_flag=True,
        ax=ax1
    )
    # Note: Label is set inside the plotter based on logic, but we can override title if needed
    ax1.text(0.05, 0.93, "Fixed $\\theta_c$", transform=ax1.transAxes, fontsize=FONTSIZE)

    # Panel 2: Flat Theta
    plotter_flat.plot_theta_star_robust_vs_rate(
        label_flag=1, # Sets y-label to Flat Theta*
        beaming_func=beaming_efficiency,
        ax=ax2
    )
    ax2.text(0.05, 0.93, "Flat $\\theta_c$", transform=ax2.transAxes, fontsize=FONTSIZE)

    # Panel 3: Log-Normal Theta
    plotter_ln.plot_theta_star_robust_vs_rate(
        label_flag=2, # Sets y-label to Log-Normal Theta*
        beaming_func=geom_eff_ln,
        xlabel=True,
        ax=ax3
    )
    ax3.text(0.05, 0.93, "Log-Normal $\\theta_c$", transform=ax3.transAxes, fontsize=FONTSIZE)

    # Panel 4: Log-Normal Tail Fraction
    plotter_ln.plot_lognormal_tail_fraction_vs_rate(
        beaming_func=geom_eff_ln,
        xlabel=True,
        ax=ax4
    )
    ax4.text(0.05, 0.93, "Log-Normal Tail Fraction", transform=ax4.transAxes, fontsize=FONTSIZE)

    plt.tight_layout()
    out_file = OUTPUT_PATH / "theta_star_combined_grid.pdf"
    plt.savefig(out_file)
    print(f"Saved {out_file}")
    plt.show()

    # 3.4 Additional Individual Plots requested
    print("Generating Individual Helper Plots...")
    plotter_theta_c.plot_violins_fj_at_fixed_angles(filename="violins_fj_fixed_angles.pdf")
    plotter_theta_c.plot_violins_epsilon(filename="violins_epsilon.pdf")
    plotter_theta_c.plot_fj_fraction_vs_rate(filename="fj_fraction_vs_rate_thetac.pdf")


# function for just initializing the unified plotter module with all the plotter/analyzer classes
def initialize_unified_plotter():
    analysis = PopulationAnalysis(
        output_folder=DIRS["Structured"],
        images_folder=OUTPUT_PATH,
        datafiles=DATA_FILES_DIR,
        base_fontsize=FONTSIZE
    )
    analysis.run_preprocessing(samp_names=SAMP_NAMES, quiet=True)
    analyzer = analysis.analyzer

    # 3.1 Initialize Specific Plotters
    print("Initializing Fixed Theta_c Plotter...")
    plotter_theta_c = ThetaCModelPlotter(
        samp_names=SAMP_NAMES,
        base_dir=DIRS["Fixed"].name,
        data_files_dir=DATA_FILES_DIR,
        output_dir=str(OUTPUT_PATH),
        discard=1000, thin=20, k_params=6, theta_c_idx=4, fj_idx=5,
        fontsize=FONTSIZE
    )

    print("Initializing Flat Theta Plotter...")
    plotter_flat = FjModelPlotter(
        samp_names=SAMP_NAMES,
        base_dir=DIRS["Flat"].name,
        data_files_dir=DATA_FILES_DIR,
        output_dir=str(OUTPUT_PATH),
        discard=5000, thin=20, k_params=6, theta_c_idx=4, fj_idx=5,
        fontsize=FONTSIZE
    )

    print("Initializing Log-Normal Plotter...")
    plotter_ln = LogNormalModelPlotter(
        samp_names=SAMP_NAMES,
        base_dir=DIRS["LogNormal"].name,
        data_files_dir=DATA_FILES_DIR,
        output_dir=str(OUTPUT_PATH),
        discard=3000, thin=15, k_params=6, theta_c_idx=4, fj_idx=5,
        log_theta=False, # Assuming samples are already in degrees not log10 based on prompt context
        fontsize=FONTSIZE
    )

    return analyzer, plotter_theta_c, plotter_flat, plotter_ln

# function to plot everything but the single model comparison plot
def plot_all_unified(analyzer, plotter_theta_c, plotter_flat, plotter_ln):
    # 1.1 Combined Analysis (2x2 Grid: Rates vs Fj)
    print("Generating Combined Analysis Plot...")
    fig1, _ = analyzer.plot_combined_analysis(f_max=5)
    
    # 1.2 Violin Plots
    print("Generating Violin Plots...")
    fig2, _ = analyzer.plot_fj_violins(filename="fj_violin_plot_structured")
    
    # 1.3 sGRB Rate Posteriors
    print("Generating sGRB Rate Posteriors...")
    # Note: Using the method from PopulationAnalysis/ModelAnalyzer
    analyzer.plot_sgrb_rate_posteriors(filename="sgrb_rate_posteriors_structured.pdf")

    # 3.2 Create Lognormal Interpolator
    geom_eff_ln = create_geometric_efficiency_lognormal_interpolator(sigma_theta_c=0.5)

    # 3.3 Generate 2x2 Plot
    fig, axes = plt.subplots(2, 2, figsize=(22, 18))
    ((ax1, ax2), (ax3, ax4)) = axes

    # Panel 1: Fixed Theta C
    plotter_theta_c.plot_theta_star_robust_vs_rate(
        beaming_func=beaming_theta_c,
        legend_flag=True,
        ax=ax1
    )
    # Note: Label is set inside the plotter based on logic, but we can override title if needed
    ax1.text(0.05, 0.93, "Fixed $\\theta_c$", transform=ax1.transAxes, fontsize=FONTSIZE)

    # Panel 2: Flat Theta
    plotter_flat.plot_theta_star_robust_vs_rate(
        label_flag=1, # Sets y-label to Flat Theta*
        beaming_func=beaming_efficiency,
        ax=ax2
    )
    ax2.text(0.05, 0.93, "Flat $\\theta_c$", transform=ax2.transAxes, fontsize=FONTSIZE)

    # Panel 3: Log-Normal Theta
    plotter_ln.plot_theta_star_robust_vs_rate(
        label_flag=2, # Sets y-label to Log-Normal Theta*
        beaming_func=geom_eff_ln,
        xlabel=True,
        ax=ax3
    )
    ax3.text(0.05, 0.93, "Log-Normal $\\theta_c$", transform=ax3.transAxes, fontsize=FONTSIZE)

    # Panel 4: Log-Normal Tail Fraction
    plotter_ln.plot_lognormal_tail_fraction_vs_rate(
        beaming_func=geom_eff_ln,
        xlabel=True,
        ax=ax4
    )
    ax4.text(0.05, 0.93, "Log-Normal Tail Fraction", transform=ax4.transAxes, fontsize=FONTSIZE)

    plt.tight_layout()
    out_file = OUTPUT_PATH / "theta_star_combined_grid.pdf"
    plt.savefig(out_file)
    print(f"Saved {out_file}")
    plt.show()

    # 3.4 Additional Individual Plots requested
    print("Generating Individual Helper Plots...")
    plotter_theta_c.plot_violins_fj_at_fixed_angles(filename="violins_fj_fixed_angles.pdf")
    plotter_theta_c.plot_violins_epsilon(filename="violins_epsilon.pdf")
    plotter_theta_c.plot_fj_fraction_vs_rate(filename="fj_fraction_vs_rate_thetac.pdf")