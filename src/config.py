"""
Central configuration for data locations.

The original analysis was developed on NERSC Perlmutter against a fixed
project path. That path is meaningless outside the author's allocation, so
every module in this package resolves data locations through this file
instead of hardcoding them.

Set PSP_DATA_DIR for the processed data products (see data/README.md).
Set PSP_HFR_DIR / PSP_LFR_DIR for the raw CDF archives consumed by
src/preprocessing/build_monthly_archive.py. Set PSP_RAW_DATA_DIR for the
monthly combined HDF5 files that script produces (which
src/preprocessing/make_folded_data.py then folds, and which
src/stokes_comparison/ also reads directly for the Stokes V comparison --
see the note on RAW_MONTHLY_PATTERN below, they're the same files). Set
PSP_SPICE_KERNEL_DIR for the SPICE kernels make_folded_data.py needs.
"""
import os
from pathlib import Path

# Resolution order: env var -> ./data (repo-relative) -> error.
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

DATA_DIR = Path(os.environ.get("PSP_DATA_DIR", DEFAULT_DATA_DIR))
RAW_DATA_DIR = Path(os.environ.get("PSP_RAW_DATA_DIR", DATA_DIR / "raw_monthly"))
SPICE_KERNEL_DIR = Path(os.environ.get("PSP_SPICE_KERNEL_DIR", DATA_DIR / "spiceypy_kernels"))
HFR_DIR = Path(os.environ.get("PSP_HFR_DIR", DATA_DIR / "raw_cdf" / "HFR"))
LFR_DIR = Path(os.environ.get("PSP_LFR_DIR", DATA_DIR / "raw_cdf" / "LFR"))

# Expected processed data products (see data/README.md for provenance and
# the upstream scripts that generate them from the public PSP FIELDS
# archive). These are NOT raw archive files -- they are the intermediate
# phase-folded products this pipeline consumes.
#
# The total archive exists in TWO reference frames (Section 3.1: "the
# temporal spectra are folded into two periodic reference frames"),
# produced by two separate runs of make_folded_data.py against different
# angle columns:
#   - data_theta_psp.h5:  folded to PSP's Jovian System III longitude
#                         (lambda_III). 3711 segments.
#   - data_phi_io.h5:     folded to Io phase (Phi_Io). 938 segments.
# Both are "total" archives (unfiltered) -- neither is the
# manually-curated Jovian-positive subset (all_jupiter_data*.h5, still
# missing -- see data/README.md).
PATHS = {
    "all_data_total": DATA_DIR / "data_theta_psp.h5",
    "all_data_total_phi_io": DATA_DIR / "data_phi_io.h5",
    "all_jupiter_data": DATA_DIR / "all_jupiter_data.h5",
    "all_jupiter_data_unrolled": DATA_DIR / "all_jupiter_data_unrolled.h5",
    "freqs": DATA_DIR / "freqs.csv",
    "annotations": DATA_DIR / "jovian_annotations.json",
}

# SPICE kernels required by src/preprocessing/make_folded_data.py. These are
# public (NAIF/PDS) but not bundled in this repo -- see data/README.md for
# where to download them. Standard DE ephemeris, leapseconds, Jupiter
# satellite ephemeris, and planetary constants kernels, plus the PSP
# trajectory kernel (SPK) covering the analysis window.
SPICE_KERNELS = [
    SPICE_KERNEL_DIR / "de440s.bsp",
    SPICE_KERNEL_DIR / "naif0012.tls",
    SPICE_KERNEL_DIR / "jup365.bsp",
    SPICE_KERNEL_DIR / "pck00010.tpc",
    SPICE_KERNEL_DIR / "spp_nom_20180812_20300101_v042_PostV7.bsp",
]

# Monthly combined HDF5 files (geometry + Stokes V + C_IM + C_RE + A12 +
# A34, one file per calendar month) produced by
# src/preprocessing/build_monthly_archive.py from the raw HFR/LFR CDF archive.
#
# NOTE: this is the SAME file family previously referred to (in the
# original notebook, via a hardcoded path) as living under a
# "monthly_stokes_v" directory as "stokes_data_<YYYY>_<MM>.h5" -- that
# was this exact data, just under a different name/location. There is
# only one raw-monthly file per month, used both as make_folded_data.py's
# input (via its C_IM columns) and as stokes_comparison's Stokes V source
# (via its STOKES V columns). See src/preprocessing/build_monthly_archive.py's
# module docstring for the full column layout.
RAW_MONTHLY_PATTERN = str(RAW_DATA_DIR / "data_{year}_{month:02d}.h5")

# Output of the folding step (src/preprocessing/make_folded_data.py) --
# this IS data_theta_psp.h5 by default (pass phase_frame="phi_io" via
# make_folded_data.fold_archive() / process_month() to build data_phi_io.h5
# instead).
MASTER_ARCHIVE_PATH = PATHS["all_data_total"]


def require(path_key: str) -> Path:
    """Return a data path, raising a clear error if it doesn't exist yet."""
    path = PATHS[path_key]
    if not path.exists():
        raise FileNotFoundError(
            f"Expected data product '{path_key}' at {path}, but it doesn't "
            f"exist. Set PSP_DATA_DIR to your data directory, or see "
            f"data/README.md for how to obtain/build this file."
        )
    return path
