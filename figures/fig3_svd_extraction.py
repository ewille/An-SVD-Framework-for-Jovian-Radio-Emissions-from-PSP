"""
Reproduces Figure 3 (SVD extraction of Jovian decametric/hectometric
emissions: post-filtered spectrum, SVD reconstruction, residuals, and the
singular-value rank-cutoff plot) for a single representative event.

Usage:
    python figures/fig3_svd_extraction.py [--file-index 245]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.load_data import load_jupiter_data, sort_frequency_channels
from src.svd_extraction.svd_core import run_svd_extraction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file-index", type=int, default=245, help="Row index of the event to plot (default: 245, a representative quiet-HOM event).")
    parser.add_argument("--entropy-limit", type=float, default=3.0, help="Shannon entropy threshold S_lim in nats (default: 3.0, matches the paper).")
    args = parser.parse_args()

    datasets = load_jupiter_data()
    datasets = sort_frequency_channels(datasets)

    run_svd_extraction(
        datasets=datasets,
        file_index=args.file_index,
        channel_range=(30, 120),
        num_modes_to_check=15,
        entropy_limit=args.entropy_limit,
        v_limit=2.0,
        mode="jupiter",
        emission_title="Quiet HOM Type",
    )


if __name__ == "__main__":
    main()
