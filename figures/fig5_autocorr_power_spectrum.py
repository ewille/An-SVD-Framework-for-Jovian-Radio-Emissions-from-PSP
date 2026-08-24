"""
Reproduces Figure 5 (2D autocorrelation and Wiener-Khinchin power
spectrum of the SVD reconstruction vs. residual matrix) for a single
representative event.

*** See src/svd_extraction/svd_core.py module docstring first ***
Rank selection (which modes get treated as "signal" going into this
figure) depends on the entropy-direction question flagged there.

Usage:
    python figures/fig5_autocorr_power_spectrum.py [--file-index 245]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.load_data import load_jupiter_data, sort_frequency_channels
from src.svd_extraction.svd_core import run_svd_extraction
from src.svd_extraction.validation import compare_autocorrelations, plot_autocorr_power_spectrum_figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file-index", type=int, default=245)
    parser.add_argument("--entropy-limit", type=float, default=3.0)
    parser.add_argument("--save-path", default="figure5.png")
    args = parser.parse_args()

    datasets = load_jupiter_data()
    datasets = sort_frequency_channels(datasets)

    svd_signal, residuals = run_svd_extraction(
        datasets=datasets,
        file_index=args.file_index,
        channel_range=(30, 120),
        num_modes_to_check=15,
        entropy_limit=args.entropy_limit,
        mode="jupiter",
        plot=False,
    )

    sub_freqs = datasets["freqs"].iloc[30:121]
    sub_phase = datasets["phase"]

    ac_svd, ac_res, ps_svd, ps_res, svd_lag1, res_lag1 = compare_autocorrelations(
        svd_signal, residuals, freqs=sub_freqs, phase=sub_phase, plot=False,
    )

    plot_autocorr_power_spectrum_figure(ac_svd, ac_res, ps_svd, ps_res, save_path=args.save_path)


if __name__ == "__main__":
    main()
