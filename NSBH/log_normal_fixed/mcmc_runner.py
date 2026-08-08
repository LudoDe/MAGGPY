import numpy as np
import matplotlib.pyplot as plt
import emcee
from pathlib import Path
from scipy.stats import gaussian_kde
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import sys
# Assume sys.path has been configured in the notebook to find `src`
import src.init
from maggpy.nsbh.init import initialize_combined_simulation
from maggpy.top_hat.montecarlo import (
    create_k_interpolator,
    compute_luminosity_distance,
    simplified_montecarlo,
    apply_detection_cuts,
    score_func_cvm,
    poiss_log,
    run_mcmc,
    check_and_resume_mcmc,
)
from maggpy.nsbh.montecarlo import GBM_EFF, FJ_NSBH_FIXED, N_MC_EVENTS

def run_populations(alphas, fj_bns_values, geom_eff_func, datafiles, 
                    n_params_1=5, n_walkers_1=24, n_steps_1=5000,
                    n_params_2=2, n_walkers_2=24, n_steps_2=5000,
                    burn_frac=0.33, thin=10):
    """
    Run the two-stage MCMC for purely BNS and BNS+NSBH for a given list of alphas
    and fj_bns fractions.
    """
    for alpha in alphas:
        print(f"\n============================================================")
        print(f"Running MCMC pipeline for alpha = {alpha}")
        print(f"============================================================")
        
        demo_params = {
            'alpha': -0.6,
            'beta_s': -2.5,
            'n': 2.0,
            'theta_c': 3.4,
            'theta_v_max': 10.0,
            'z_model': f'fiducial_delayed_{alpha}',
        }

        bns_params, nsbh_data, data_dict = initialize_combined_simulation(
            datafiles=datafiles,
            params=demo_params,
            size_test=2_000,
            nsbh_population='fiducial_delayed_cut',
            nsbh_alpha=alpha,
        )

        k_interpolator = create_k_interpolator()
        bns_distances = compute_luminosity_distance(bns_params.z_arr)
        nsbh_distances = nsbh_data.distances

        for fj_bns in fj_bns_values:
            print(f'\n--- Running for FJ_BNS = {fj_bns} ---')

            # =========================================================
            # STAGE 1: BNS ONLY
            # =========================================================
            run_name_stage1 = f'lognormal_stage1_bns_only_alpha_{alpha}_fj{fj_bns:.1f}'
            out_dir_stage1 = src.init.create_run_dir(run_name_stage1, output_files_default='Output_files')
            backend_path_stage1 = out_dir_stage1 / 'emcee.h5'

            def log_likelihood_stage1_bns_only(thetas, n_events=N_MC_EVENTS, fj=fj_bns):
                A_index, L_L0, L_mu_E, sigma_E, theta_c_bns = thetas
                grb_thetas = [A_index, L_L0, L_mu_E, sigma_E]

                geom_eff_bns = geom_eff_func(theta_c_bns)
                epsilon_bns = geom_eff_bns * fj
                intrinsic_bns = epsilon_bns * len(bns_params.z_arr) * GBM_EFF

                bns_results = simplified_montecarlo(
                    grb_thetas, n_events, bns_params, bns_distances, k_interpolator
                )
                bns_trig, bns_analysis = apply_detection_cuts(
                    bns_results['p_flux'], bns_results['E_p_obs']
                )

                pflux_det = bns_results['p_flux'][bns_analysis]
                epeak_det = bns_results['E_p_obs'][bns_analysis]
                if len(pflux_det) <= 3:
                    return -np.inf, -np.inf, -np.inf, -np.inf, -np.inf

                logL_pflux = score_func_cvm(pflux_det, bns_params.pflux_data, bns_params.rng)
                logL_epeak = score_func_cvm(epeak_det, bns_params.epeak_data, bns_params.rng)

                phys_eff_bns = np.sum(bns_trig) / n_events
                predicted_bns = intrinsic_bns * bns_params.triggered_years * phys_eff_bns

                observed_total = bns_params.yearly_rate * bns_params.triggered_years
                if predicted_bns <= 0:
                    return -np.inf, -np.inf, -np.inf, -np.inf, -np.inf

                logL_poisson = poiss_log(k=observed_total, mu=predicted_bns)
                logL_total = logL_pflux + logL_epeak + logL_poisson

                return logL_total, logL_pflux, logL_epeak, logL_poisson, predicted_bns

            def log_prior_stage1(thetas):
                A_index, L_L0, L_mu_E, sigma_E, theta_c_bns = thetas
                if not (1.5 < A_index < 12): return -np.inf
                if not (-2 < L_L0 < 7): return -np.inf
                if not (0.1 < L_mu_E < 7): return -np.inf
                if not (0 < sigma_E < 2.5): return -np.inf
                if not (1 < theta_c_bns < 50): return -np.inf
                return 0.0

            def init_walkers_stage1(n_walkers, seed=123):
                rng = np.random.default_rng(seed)
                return np.column_stack([
                    rng.uniform(2.0, 3.5, n_walkers),
                    rng.uniform(2.0, 4.5, n_walkers),
                    rng.uniform(1.5, 4.5, n_walkers),
                    rng.uniform(0.2, 1.2, n_walkers),
                    rng.uniform(3.0, 20.0, n_walkers),
                ])

            def log_probability_stage1(thetas):
                lp = log_prior_stage1(thetas)
                if not np.isfinite(lp):
                    return -np.inf, 0.0, 0.0, 0.0, 0.0
                ll = log_likelihood_stage1_bns_only(thetas=thetas, n_events=N_MC_EVENTS, fj=fj_bns)
                if not np.isfinite(ll[0]):
                    return -np.inf, 0.0, 0.0, 0.0, 0.0
                return lp + ll[0], ll[1], ll[2], ll[3], ll[4]

            initial_pos_1, n_steps_rem_1, backend_1 = check_and_resume_mcmc(
                filename=backend_path_stage1, n_steps=n_steps_1,
                initialize_walkers_func=init_walkers_stage1, n_walkers=n_walkers_1,
            )

            run_mcmc(
                log_probability_func=log_probability_stage1,
                initial_walkers=initial_pos_1, n_iterations=n_steps_rem_1,
                n_walkers=n_walkers_1, n_params=n_params_1, backend=backend_1,
                blobs_dtype=[('l_pflux', float), ('l_epeak', float), ('l_poiss', float), ('mu_bns', float)],
            )

            flat_1 = backend_1.get_chain(discard=int(backend_1.iteration * burn_frac), thin=thin, flat=True)
            medians_1 = np.median(flat_1, axis=0)
            spectral_fixed = medians_1[:4].copy()
            theta_bns_stage1_med = medians_1[4]

            # =========================================================
            # STAGE 2: BNS + NSBH
            # =========================================================
            run_name_stage2 = f'lognormal_stage2_bns_plus_nsbh_alpha_{alpha}_fj{fj_bns:.1f}'
            out_dir_stage2 = src.init.create_run_dir(run_name_stage2, output_files_default='Output_files')
            backend_path_stage2 = out_dir_stage2 / 'emcee.h5'

            def log_likelihood_stage2_fixed_spectral(thetas, spectral_fixed, n_events=N_MC_EVENTS, fj=fj_bns):
                theta_c_bns, theta_c_nsbh = thetas
                A_index, L_L0, L_mu_E, sigma_E = spectral_fixed
                grb_thetas = [A_index, L_L0, L_mu_E, sigma_E]

                geom_eff_bns = geom_eff_func(theta_c_bns)
                geom_eff_nsbh = geom_eff_func(theta_c_nsbh)

                epsilon_bns = geom_eff_bns * fj
                epsilon_nsbh = geom_eff_nsbh * FJ_NSBH_FIXED

                intrinsic_bns = epsilon_bns * len(bns_params.z_arr) * GBM_EFF
                intrinsic_nsbh = epsilon_nsbh * len(nsbh_data.z_arr) * GBM_EFF

                bns_results = simplified_montecarlo(
                    grb_thetas, n_events, bns_params, bns_distances, k_interpolator
                )
                bns_trig, bns_analysis = apply_detection_cuts(
                    bns_results['p_flux'], bns_results['E_p_obs']
                )

                nsbh_results = simplified_montecarlo(
                    grb_thetas, n_events, nsbh_data, nsbh_distances, k_interpolator, rng=bns_params.rng
                )
                nsbh_trig, nsbh_analysis = apply_detection_cuts(
                    nsbh_results['p_flux'], nsbh_results['E_p_obs']
                )

                pflux_det = np.concatenate([bns_results['p_flux'][bns_analysis], nsbh_results['p_flux'][nsbh_analysis]])
                epeak_det = np.concatenate([bns_results['E_p_obs'][bns_analysis], nsbh_results['E_p_obs'][nsbh_analysis]])

                if len(pflux_det) <= 3:
                    return -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf

                logL_pflux = score_func_cvm(pflux_det, bns_params.pflux_data, bns_params.rng)
                logL_epeak = score_func_cvm(epeak_det, bns_params.epeak_data, bns_params.rng)

                phys_eff_bns = np.sum(bns_trig) / n_events
                phys_eff_nsbh = np.sum(nsbh_trig) / n_events

                predicted_bns = intrinsic_bns * bns_params.triggered_years * phys_eff_bns
                predicted_nsbh = intrinsic_nsbh * bns_params.triggered_years * phys_eff_nsbh
                predicted_total = predicted_bns + predicted_nsbh
                observed_total = bns_params.yearly_rate * bns_params.triggered_years

                if predicted_total <= 0:
                    return -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf

                logL_poisson = poiss_log(k=observed_total, mu=predicted_total)
                logL_total = logL_pflux + logL_epeak + logL_poisson

                return logL_total, logL_pflux, logL_epeak, logL_poisson, predicted_bns, predicted_nsbh

            def log_prior_stage2(thetas):
                theta_c_bns, theta_c_nsbh = thetas
                if not (1 < theta_c_bns < 50): return -np.inf
                if not (1 < theta_c_nsbh < 50): return -np.inf
                return 0.0

            def init_walkers_stage2(n_walkers, seed=321):
                rng = np.random.default_rng(seed)
                return np.column_stack([
                    rng.uniform(max(1.2, theta_bns_stage1_med * 0.6), min(45.0, theta_bns_stage1_med * 1.4), n_walkers),
                    rng.uniform(2.0, 25.0, n_walkers),
                ])

            def log_probability_stage2(thetas):
                lp = log_prior_stage2(thetas)
                if not np.isfinite(lp):
                    return -np.inf, 0.0, 0.0, 0.0, 0.0, 0.0

                ll = log_likelihood_stage2_fixed_spectral(
                    thetas=thetas, spectral_fixed=spectral_fixed, n_events=N_MC_EVENTS, fj=fj_bns
                )
                if not np.isfinite(ll[0]):
                    return -np.inf, 0.0, 0.0, 0.0, 0.0, 0.0

                return lp + ll[0], ll[1], ll[2], ll[3], ll[4], ll[5]

            initial_pos_2, n_steps_rem_2, backend_2 = check_and_resume_mcmc(
                filename=backend_path_stage2, n_steps=n_steps_2,
                initialize_walkers_func=init_walkers_stage2, n_walkers=n_walkers_2,
            )

            run_mcmc(
                log_probability_func=log_probability_stage2,
                initial_walkers=initial_pos_2, n_iterations=n_steps_rem_2,
                n_walkers=n_walkers_2, n_params=n_params_2, backend=backend_2,
                blobs_dtype=[('l_pflux', float), ('l_epeak', float), ('l_poiss', float), ('mu_bns', float), ('mu_nsbh', float)],
            )


def plot_populations(alphas, fj_bns_values, burn_frac=0.33, thin=10):
    """
    Plot the Stage 2 posteriors for the given alphas and fj_bns values.
    Reads results directly from the emcee backends.
    """
    FS = 18
    def contour_thresholds(density_values, probs=(0.68, 0.95)):
        vals = np.sort(density_values)[::-1]
        csum = np.cumsum(vals)
        csum /= csum[-1]
        return [vals[np.searchsorted(csum, p)] for p in probs]

    colors = {0.1: 'tab:green', 0.5: 'tab:orange', 1.0: 'tab:blue'}
    x = np.linspace(1, 25, 70)
    y = np.linspace(1, 25, 70)
    X, Y = np.meshgrid(x, y)
    pos = np.vstack([X.ravel(), Y.ravel()])

    for alpha in alphas:
        fig, ax = plt.subplots(figsize=(7.2, 6.0))
        label_collection = [] 
        
        for fj_bns in fj_bns_values:
            backend_path_stage2 = src.init.create_run_dir(
                f'lognormal_stage2_bns_plus_nsbh_alpha_{alpha}_fj{fj_bns:.1f}', 
                output_files_default='Output_files'
            ) / 'emcee.h5'
            
            if not backend_path_stage2.exists():
                print(f"Warning: Results missing for alpha={alpha}, fj_bns={fj_bns}")
                continue
                
            backend_2 = emcee.backends.HDFBackend(backend_path_stage2)
            try:
                flat_2 = backend_2.get_chain(discard=int(backend_2.iteration * burn_frac), thin=thin, flat=True)
                theta_bns = flat_2[:, 0]
                theta_nsbh = flat_2[:, 1]
            except Exception as e:
                print(f"Failed to read chain for {backend_path_stage2}: {e}")
                continue

            xy = np.vstack([theta_bns, theta_nsbh])
            kde = gaussian_kde(xy)
            Z = kde(pos).reshape(X.shape)

            sample_d = kde(xy)
            t68, t95 = contour_thresholds(sample_d, probs=(0.68, 0.95))

            ax.contour(X, Y, Z, levels=[t95, t68], colors=[colors[fj_bns], colors[fj_bns]], 
                       linewidths=[1.2, 2.2], alpha=0.9)
            
            label = f'$f_j^{{\mathrm{{BNS}}}} = {fj_bns}$'
            line_fj = Line2D([0], [0], color=colors[fj_bns], lw=2.2, label=label)
            label_collection.append(line_fj)
        
        rect = Rectangle(**rect_args)
        ax.add_patch(rect)
        label_collection.append(Rectangle((0, 0), 1, 1, color='k', alpha=0.2, label='RE23 (90\% C.I.)'))
        ax.legend(handles=label_collection, loc='upper right', fontsize=FS)
        ax.set_title(f'$\\alpha = {alpha}$', fontsize=FS)
        ax.set_xlabel(r'$\theta_c^{\mathrm{BNS}}$ [deg]', fontsize=FS)
        ax.set_ylabel(r'$\theta_c^{\mathrm{NSBH}}$ [deg]', fontsize=FS)
        ax.tick_params(axis='both', which='major', labelsize=FS-2)
        ax.grid(True, alpha=0.2)
        ax.set_xlim(1, 25)
        ax.set_ylim(1, 25)
        
        plt.savefig(f'contour_plots_fj_comparison_lognormal_alpha_{alpha}.pdf', dpi=150, bbox_inches='tight')
        plt.show()




RE_lims = [6.1, -3.2, +9.3]
RE_median = RE_lims[0]
RE_low = RE_median + RE_lims[1]
RE_high = RE_median + RE_lims[2]
rect_args = {
    'xy': (RE_low, RE_low),
    'width': RE_high - RE_low,
    'height': RE_high - RE_low,
    'color': 'k',
    'alpha': 0.2
}
#rect = Rectangle((RE_low, RE_low), RE_high-RE_low, RE_high-RE_low, color='k', alpha=0.2)
# =========================================================================
# Diagnostic Plots: Fraction > 10deg & NSBH Relative Contribution
# (Adapted for the FIXED universal geometry case)
# =========================================================================

from scipy.stats import lognorm

def plotting_fractions(
        alpha_val, fj_bns_val, theta_threshold, geom_eff, rouco_flag=True
):
    alphas = [alpha_val] if isinstance(alpha_val, str) else list(alpha_val)
    fjs = [fj_bns_val] if isinstance(fj_bns_val, (float, int)) else list(fj_bns_val)
    
    n_rows = len(alphas)
    n_cols = len(fjs)
    
    fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(max(6, 3.5*n_cols), max(4.5, 3.5*n_rows)), sharex=True, sharey=True, squeeze=False)
    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(max(6, 3.5*n_cols), max(4.5, 3.5*n_rows)), sharex=True, sharey=True, squeeze=False)

    sc1, sc2 = None, None

    for i, alpha_diag in enumerate(alphas):
        # Initialize combined simulation to get the population sizes N_bns and N_nsbh
        # (this is fast and avoids needing them strictly in memory)
        demo_params = {
            'alpha': -0.6, 'beta_s': -2.5, 'n': 2.0, 'theta_c': 3.4,
            'theta_v_max': 10.0, 'z_model': f'fiducial_delayed_{alpha_diag}'
        }
        bns_params, nsbh_data, _ = initialize_combined_simulation(
            datafiles       =   Path('../../datafiles'),
            params          =   demo_params,
            size_test       =   2_000,
            nsbh_population =   'fiducial_delayed_cut',
            nsbh_alpha      =   alpha_diag
        )
        N_bns   = len(bns_params.z_arr)
        N_nsbh  = len(nsbh_data.z_arr)

        for j, fj_bns_diag in enumerate(fjs):
            # Read the MCMC chains for this specific configuration
            backend_path = src.init.create_run_dir(
                f'lognormal_stage2_bns_plus_nsbh_alpha_{alpha_diag}_fj{fj_bns_diag:.1f}', 
                output_files_default='Output_files'
            ) / 'emcee.h5'

            if not backend_path.exists():
                print(f"Warning: Data missing for alpha={alpha_diag}, fj_bns={fj_bns_diag}")
                continue

            backend = emcee.backends.HDFBackend(backend_path)
            burn_in = int(backend.iteration * 0.33)
            flat_samples = backend.get_chain(discard=burn_in, thin=10, flat=True)

            theta_bns_2 = flat_samples[:, 0]
            theta_nsbh_2 = flat_samples[:, 1]

            # -------------------------------------------------------------------------
            # Calculations
            # -------------------------------------------------------------------------

            eff_bns = np.array([geom_eff(t) for t in theta_bns_2])
            eff_nsbh = np.array([geom_eff(t) for t in theta_nsbh_2])

            total_bns_yr_samp = eff_bns * fj_bns_diag * N_bns
            total_nsbh_yr_samp = eff_nsbh * FJ_NSBH_FIXED * N_nsbh

            min_tc      = 1.0
            max_tc      = 45.0
            shape_param = 0.5 * np.log(10.0) # SIGMA_THETA_C * np.log(10)
            mu_bns      = np.log(np.exp(theta_bns_2))
            mu_nsbh     = np.log(np.exp(theta_nsbh_2))

            # BNS tail fractions per-sample
            cdf_max_bns_s       = lognorm.cdf(max_tc, s=shape_param, scale=mu_bns)
            cdf_min_bns_s       = lognorm.cdf(min_tc, s=shape_param, scale=mu_bns)
            cdf_thr_bns_s        = lognorm.cdf(theta_threshold, s=shape_param, scale=mu_bns)
            norm_bns_s          = cdf_max_bns_s - cdf_min_bns_s
            tail_bns_s          = np.zeros_like(norm_bns_s)
            m_ok                = norm_bns_s > 1e-12
            tail_bns_s[m_ok]    = (cdf_max_bns_s[m_ok] - cdf_thr_bns_s[m_ok]) / norm_bns_s[m_ok]

            # NSBH tail fractions per-sample
            cdf_max_ns_s        = lognorm.cdf(max_tc, s=shape_param, scale=mu_nsbh)
            cdf_min_ns_s        = lognorm.cdf(min_tc, s=shape_param, scale=mu_nsbh)
            cdf_thr_ns_s        = lognorm.cdf(theta_threshold, s=shape_param, scale=mu_nsbh)
            norm_ns_s           = cdf_max_ns_s - cdf_min_ns_s
            tail_ns_s           = np.zeros_like(norm_ns_s)
            m_ok_ns             = norm_ns_s > 1e-12
            tail_ns_s[m_ok_ns]  = (cdf_max_ns_s[m_ok_ns] - cdf_thr_ns_s[m_ok_ns]) / norm_ns_s[m_ok_ns]

            # Calculate the number of events above the threshold
            above_bns           = total_bns_yr_samp * tail_bns_s
            above_nsbh          = total_nsbh_yr_samp * tail_ns_s

            above_samp          = above_bns + above_nsbh
            total_events_samp   = total_bns_yr_samp + total_nsbh_yr_samp

            # Fraction of total (BNS + NSBH) observable events > threshold
            frac_samp_total = np.zeros_like(above_samp)
            nz = total_events_samp > 0
            frac_samp_total[nz] = above_samp[nz] / total_events_samp[nz]

            # Relative contribution of NSBH (out of total GRBs)
            nsbh_contribution = np.zeros_like(total_nsbh_yr_samp)
            nsbh_contribution[nz] = total_nsbh_yr_samp[nz] / total_events_samp[nz]

            ax1 = axes1[i, j]
            ax2 = axes2[i, j]

            # -------------------------------------------------------------------------
            # Populating plots 1 and 2
            # -------------------------------------------------------------------------
            sc1 = ax1.scatter(theta_bns_2, theta_nsbh_2, c=frac_samp_total, cmap='coolwarm', alpha=0.7, s=25, vmin=0, vmax=1)
            sc2 = ax2.scatter(theta_bns_2, theta_nsbh_2, c=nsbh_contribution, cmap='viridis', alpha=0.7, s=25, vmin=0, vmax=1)

            #limity y to 25 deg for better visualization (since RE23 is around 9.3 deg)
            ax1.set_xlim(1, 25)
            ax1.set_ylim(1, 25)
            ax2.set_xlim(1, 25)
            ax2.set_ylim(1, 25)

            for ax in (ax1, ax2):
                ax.grid(True, alpha=0.2)
                if rouco_flag:
                    # copy rectangle args to avoid reusing same object reference
                    rect = Rectangle(**rect_args)
                    ax.add_patch(rect)
                
                # Layout logic for grid vs single
                if n_rows > 1 or n_cols > 1:
                    if i == 0: ax.set_title(f'$f_j^{{\mathrm{{BNS}}}}={fj_bns_diag}$')
                    if j == 0: ax.set_ylabel(f'$\\alpha={alpha_diag}$\n$\\theta_c^{{\mathrm{{NSBH}}}}$ (deg)')
                    if i == n_rows - 1: ax.set_xlabel(r'$\theta_c^{\mathrm{BNS}}$ (deg)')
                else:
                    ax.set_title(f'($\\alpha={alpha_diag}$, $f_j^{{\mathrm{{BNS}}}}={fj_bns_diag}$)')
                    ax.set_xlabel(r'$\theta_c^{\mathrm{BNS}}$ (deg, lognormal)')
                    ax.set_ylabel(r'$\theta_c^{\mathrm{NSBH}}$ (deg, lognormal)')

    if sc1 is not None and sc2 is not None:
        if n_rows > 1 or n_cols > 1:
            # Colorbars at the general right side for grid format
            fig1.subplots_adjust(right=0.88, hspace=0.1, wspace=0.1)
            cb_ax1 = fig1.add_axes([0.9, 0.15, 0.02, 0.7])
            fig1.colorbar(sc1, cax=cb_ax1, label=f'Fraction $\\theta_c > {theta_threshold}^\circ$')
            fig2.subplots_adjust(right=0.88, hspace=0.1, wspace=0.1)
            cb_ax2 = fig2.add_axes([0.9, 0.15, 0.02, 0.7])
            fig2.colorbar(sc2, cax=cb_ax2, label='NSBH / (BNS + NSBH)')
        else:
            fig1.colorbar(sc1, ax=axes1[0,0], label=f'Fraction $\\theta_c > {theta_threshold}^\circ$')
            fig2.colorbar(sc2, ax=axes2[0,0], label='NSBH / (BNS + NSBH)')
            fig1.tight_layout()
            fig2.tight_layout()

    # Show legend just on upper right plot
    if rouco_flag and sc1 is not None:
        for axes in (axes1, axes2):
            ax_tr = axes[0, -1] # Top-right axis
            ax_tr.legend([ax_tr.patches[-1]], ['RE23 (90\% C.I.)'], loc='upper right', fontsize=10)

    plt.show()