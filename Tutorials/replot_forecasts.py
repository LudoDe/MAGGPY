"""Redraw Tutorial 6 figures from existing MAGGPY forecast CSVs."""
import argparse

from forecast_plotting import replot_saved_forecasts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir', help='Directory containing the four forecast CSVs.')
    parser.add_argument('--years', type=float, required=True,
                        help='Physical catalogue duration, exactly as used in Tutorial 6.')
    parser.add_argument('--cut', type=float, default=100., help='Stored GW localization cut [deg^2].')
    parser.add_argument('--snr-threshold', type=float, default=8.)
    parser.add_argument('--theta-v-max', type=float, default=30.)
    args = parser.parse_args()
    paths = replot_saved_forecasts(args.output_dir, args.years,
                                   theta_v_max=args.theta_v_max, cut=args.cut,
                                   snr_threshold=args.snr_threshold)
    print(f'Saved {len(paths)//2} figures as PDF and 300-dpi PNG in {args.output_dir}')


if __name__ == '__main__':
    main()
