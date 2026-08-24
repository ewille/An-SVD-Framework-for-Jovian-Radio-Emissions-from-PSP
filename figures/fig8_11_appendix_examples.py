"""
Reproduces Appendix Figures 8-11 (additional SVD extraction examples
spanning different emission morphologies/intensities). These use the
exact same machinery as Figure 3 -- run_svd_extraction -- just on
different representative file indices.

NOTE: Appendix Figures 8-11 in the manuscript are currently missing
descriptive captions (per the outstanding manuscript revisions) -- the
`--emission-title` argument here is a placeholder per-run label, not a
substitute for the actual figure captions that need to be written for
the paper itself.

Usage:
    python figures/fig8_11_appendix_examples.py --file-index 612 --emission-title "Vertex-Early Arc"
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.load_data import load_jupiter_data, sort_frequency_channels
from src.svd_extraction.svd_core import run_svd_extraction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file-index", type=int, required=True, help="Row index of the appendix example to plot. The specific indices used for Figures 8-11 in the manuscript aren't recorded in the source notebook -- fill these in once identified.")
    parser.add_argument("--entropy-limit", type=float, default=3.0)
    parser.add_argument("--num-modes-to-check", type=int, default=15)
    parser.add_argument("--v-limit", type=float, default=2.0)
    parser.add_argument("--emission-title", default="Jovian Emission")
    args = parser.parse_args()

    datasets = load_jupiter_data()
    datasets = sort_frequency_channels(datasets)

    run_svd_extraction(
        datasets=datasets,
        file_index=args.file_index,
        channel_range=(30, 120),
        num_modes_to_check=args.num_modes_to_check,
        entropy_limit=args.entropy_limit,
        v_limit=args.v_limit,
        mode="jupiter",
        emission_title=args.emission_title,
    )


if __name__ == "__main__":
    main()
