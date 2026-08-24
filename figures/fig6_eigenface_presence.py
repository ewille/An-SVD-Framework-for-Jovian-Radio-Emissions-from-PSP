"""
Reproduces Figure 6 (Eigenface coherence timeline and distribution,
"No Jupiter" vs "Jupiter" populations).

Requires a manually-curated `all_data_jupiter` subset with a `type`
column (1=Quiet HOM, 2=Noisy HOM, 3=Vertex-Late) -- see
src/eigenfaces/annotator.py for how that's built.

Usage:
    python figures/fig6_eigenface_presence.py [--target-type 2]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.load_data import load_jupiter_data, sort_frequency_channels
from src.eigenfaces.eigenfaces_core import (
    compute_single_eigenface,
    project_single_eigenface,
    plot_eigenface_timeline,
    plot_eigenface_histogram,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-type", type=int, default=2, help="1=Quiet HOM, 2=Noisy HOM, 3=Vertex-Late (default: 2, the broadband HOM subset used in the paper).")
    args = parser.parse_args()

    datasets = load_jupiter_data()
    datasets = sort_frequency_channels(datasets)

    eigenface = compute_single_eigenface(datasets, target_type=args.target_type)
    if eigenface is None:
        raise SystemExit("No eigenface could be computed -- check that 'all_data_jupiter' has a populated 'type' column.")

    proj_jupiter = project_single_eigenface(datasets["all_data_jupiter"], eigenface)
    proj_total = project_single_eigenface(datasets["all_data_total"], eigenface)

    plot_eigenface_timeline(proj_total, proj_jupiter)
    plot_eigenface_histogram(proj_total, proj_jupiter)


if __name__ == "__main__":
    main()
