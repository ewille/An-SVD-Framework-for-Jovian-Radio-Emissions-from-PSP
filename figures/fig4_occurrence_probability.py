"""
Reproduces Figure 4 (occurrence probability distributions by Io phase and
Jovian System III longitude, separated by LHCP/RHCP polarization).

*** See src/occurrence/occurrence_plots.py module docstring first ***
The rescaling constants (Note 1 there) still need author confirmation
before this script's output should be treated as reproducing the
published figure exactly.

Loads both phase-folded archives (Phi_Io and lambda_III -- see
src/preprocessing/load_data.py's `phase_frame` parameter) since Figure 4's
two columns need genuinely different data, not two views of one archive.

Usage:
    python figures/fig4_occurrence_probability.py [--annotation-dir ../data]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.load_data import load_jupiter_data, sort_frequency_channels
from src.occurrence.occurrence_plots import plot_jovian_occurrence_matrix
from src.config import DATA_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-dir", default=None, help="Directory containing the four annotation JSON files (default: PSP_DATA_DIR).")
    args = parser.parse_args()

    datasets_io_phase = load_jupiter_data(phase_frame="phi_io")
    datasets_io_phase = sort_frequency_channels(datasets_io_phase)

    datasets_longitude = load_jupiter_data(phase_frame="lambda_iii")
    datasets_longitude = sort_frequency_channels(datasets_longitude)

    annotation_dir = args.annotation_dir or str(DATA_DIR)

    plot_jovian_occurrence_matrix(
        datasets_io_phase=datasets_io_phase,
        datasets_longitude=datasets_longitude,
        annotation_dir=annotation_dir,
        title="Jovian Radio Emission Occurrence Probability",
    )


if __name__ == "__main__":
    main()
