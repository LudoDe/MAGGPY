import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, '../../')
from maggpy.top_hat.geometric_eff import (
    geometric_efficiency_fixed,
    create_geometric_efficiency_lognormal_interpolator,
    geometric_efficiency_flat_midpoint
)

from mcmc_baseline import run_baseline_populations

def main():
    alphas = ["A0.5", "A1.0", "A3.0", "A5.0"]
    n_steps = 20_000

    print("Starting baseline runs...")

    # 1) Universal / Fixed Geometry
    print("\\n>>> Running Universal (Fixed) Baseline <<<")
    run_baseline_populations(
        alphas=alphas,
        geom_eff_func=geometric_efficiency_fixed,
        run_dir_name="universal",
        n_steps=n_steps
    )

    # 2) Log-normal Geometry
    print("\\n>>> Running Log-Normal Baseline <<<")
    geom_eff_lognorm = create_geometric_efficiency_lognormal_interpolator(
        sigma_theta_c=0.5,
        n_points=200,
        minimum_theta_c=1.0,
        maximum_theta_c=50.0
    )
    run_baseline_populations(
        alphas=alphas,
        geom_eff_func=geom_eff_lognorm,
        run_dir_name="lognormal",
        n_steps=n_steps
    )

    # 3) Flat Midpoint Geometry (new!)
    print("\\n>>> Running Flat Midpoint Baseline <<<")
    run_baseline_populations(
        alphas=alphas,
        geom_eff_func=geometric_efficiency_flat_midpoint,
        run_dir_name="flat_midpoint",
        n_steps=n_steps
    )

    print("\\nAll baseline runs completed!")

if __name__ == "__main__":
    main()
