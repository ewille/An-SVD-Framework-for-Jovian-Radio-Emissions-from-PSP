"""
Loads the processed PSP data products used throughout the pipeline.

These HDF5/CSV files are the OUTPUT of the upstream phase-folding step
(Section 3.1 of the paper: SPICE-based lambda_III / Phi_Io computation,
light-travel-time correction, and phase-grid binning --
src/preprocessing/make_folded_data.py). This module only loads that output.

The total archive exists in two reference frames (see src/config.py):
lambda_III (PSP's Jovian System III longitude, 3711 segments) and Phi_Io
(Io phase, 938 segments). Pass `phase_frame` to select which one to load.
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime

from ..config import PATHS, DATA_DIR
from .noise_reduction import filter_invalid_plots, format_all_data_total

EXPECTED_UNROLLED_LENGTH = 128 * 721  # 92288: fixed (128 freq, 721 phase) grid


def _read_master_table(path, preferred_key="master_list"):
    """
    Reads a master-archive HDF5 table, trying `preferred_key` first and
    falling back to whichever single key the file actually has (different
    runs/versions of the folding step may have used a different key name).
    """
    try:
        return pd.read_hdf(path, preferred_key)
    except KeyError:
        with pd.HDFStore(path, mode="r") as store:
            keys = store.keys()
        if len(keys) == 1:
            print(f"   -> Key '{preferred_key}' not found in {path}; using the only key present ('{keys[0]}').")
            return pd.read_hdf(path, keys[0])
        raise KeyError(
            f"Could not find key '{preferred_key}' in {path}, and the file has multiple keys {keys} "
            f"-- pass the correct one explicitly."
        )


def _normalize_plot_column(df, n_freq=128, n_phase=721):
    """
    Standardizes a master table's spectrogram column to a flat
    'unrolled_plot' column (length 92288), regardless of whether the
    source file called it 'plot' or 'unrolled_plot', and regardless of
    whether the stored arrays are already flat or already 2D (128, 721).

    Different runs of make_folded_data.py-like scripts have used both
    conventions (see make_folded_data.py's module docstring for why), so
    this normalizes rather than assuming one or the other.
    """
    if "unrolled_plot" in df.columns:
        return df

    if "plot" not in df.columns:
        raise ValueError(f"Expected a 'plot' or 'unrolled_plot' column; found columns: {list(df.columns)}")

    sample = np.asarray(df["plot"].iloc[0])

    if sample.ndim == 2:
        print(f"   -> 'plot' column contains already-2D arrays (shape {sample.shape}); flattening to 'unrolled_plot'.")
        # Round-trips through flat form even though it's already 2D, so
        # every archive goes through the same filter_invalid_plots() +
        # format_all_data_total() path afterward regardless of source
        # format. Reshape is row-major (C order) both ways, so this is a
        # lossless round-trip.
        df = df.rename(columns={"plot": "plot_2d"})
        df["unrolled_plot"] = df["plot_2d"].apply(lambda m: np.asarray(m).reshape(-1))
        df = df.drop(columns=["plot_2d"])
    else:
        print(f"   -> 'plot' column contains flat arrays (length {sample.size}); renaming to 'unrolled_plot'.")
        df = df.rename(columns={"plot": "unrolled_plot"})

    return df


def _load_master_archive(path, n_freq=128, n_phase=721):
    """Reads, normalizes, filters, and reshapes one master archive file into a ready-to-use DataFrame with a 2D 'plot' column."""
    df = _read_master_table(path)
    df = _normalize_plot_column(df, n_freq=n_freq, n_phase=n_phase)
    df = filter_invalid_plots(df, column_name="unrolled_plot", expected_length=n_freq * n_phase)
    df = format_all_data_total(df, source_col="unrolled_plot", target_col="plot")
    return df


def load_jupiter_data(data_dir=None, phase_frame="lambda_iii"):
    """
    Loads the processed PSP data products for Jupiter emission analysis.

    Args:
        data_dir (str or Path, optional): Directory containing the processed
            data files. Defaults to the PSP_DATA_DIR environment variable
            (see src/config.py).
        phase_frame (str): "lambda_iii" (default) loads the archive folded
            to PSP's Jovian System III longitude (3711 segments); "phi_io"
            loads the archive folded to Io phase (938 segments). In either
            case, `datasets['phase']` is the same generic 0-360 degree,
            721-point axis -- which physical quantity it represents
            depends on which `phase_frame` you loaded.

    Returns:
        dict with whatever of 'all_data_total', 'all_data_jupiter',
        'all_data_unrolled', 'freqs', 'phase', 'phase_frame' loaded
        successfully. Missing files are reported and left out (or set to
        None for 'all_data_jupiter'/'all_data_unrolled', since the
        morphology-labeled subset files are still commonly unavailable --
        see data/README.md) rather than failing the whole load.
    """
    data_dir = str(data_dir) if data_dir is not None else str(DATA_DIR)
    print(f"Loading data from {data_dir}...")

    if phase_frame not in ("lambda_iii", "phi_io"):
        raise ValueError("phase_frame must be 'lambda_iii' or 'phi_io'")

    master_key = "all_data_total" if phase_frame == "lambda_iii" else "all_data_total_phi_io"
    master_path = PATHS[master_key] if data_dir == str(DATA_DIR) else os.path.join(data_dir, os.path.basename(PATHS[master_key]))

    datasets = {"phase": np.linspace(0, 360, 721), "phase_frame": phase_frame}

    try:
        datasets["all_data_total"] = _load_master_archive(master_path)
        print(f"   -> Loaded '{master_path}': {datasets['all_data_total'].shape[0]} valid segments ({phase_frame}).")
    except Exception as e:
        print(f"   -> Could not load master archive at {master_path}: {e}")
        datasets["all_data_total"] = None

    try:
        datasets["all_data_jupiter"] = pd.read_hdf(os.path.join(data_dir, "all_jupiter_data.h5"), "all_data")
    except Exception as e:
        print(f"   -> 'all_jupiter_data.h5' not available ({e}). Jovian-positive subset features (eigenfaces, "
              f"Section 4.4 validation) won't work until this is added -- see data/README.md.")
        datasets["all_data_jupiter"] = None

    try:
        datasets["all_data_unrolled"] = pd.read_hdf(os.path.join(data_dir, "all_jupiter_data_unrolled.h5"), "all_data_unrolled")
    except Exception as e:
        print(f"   -> 'all_jupiter_data_unrolled.h5' not available ({e}).")
        datasets["all_data_unrolled"] = None

    try:
        datasets["freqs"] = pd.read_csv(os.path.join(data_dir, "freqs.csv"))
    except Exception as e:
        print(f"   -> 'freqs.csv' not available ({e}). Nothing downstream will work without this.")
        datasets["freqs"] = None

    if datasets["all_data_total"] is not None:
        start = datetime.strptime(datasets["all_data_total"]["start_time"].iloc[0], "%Y-%m-%d %H:%M:%S.%f")
        end = datetime.strptime(datasets["all_data_total"]["start_time"].iloc[-1], "%Y-%m-%d %H:%M:%S.%f")

        jupiter_status = "not loaded" if datasets["all_data_jupiter"] is None else f"{datasets['all_data_jupiter'].shape[0]} entries"
        unrolled_status = "not loaded" if datasets["all_data_unrolled"] is None else f"{datasets['all_data_unrolled'].shape[0]} entries"
        freqs_status = "not loaded" if datasets["freqs"] is None else "128 channels, LFR+HFR combined"

        print(
            "-" * 50,
            f"\n phase_frame: '{phase_frame}'"
            f"\n 'all_data_total': {start:%Y-%m-%d} to {end:%Y-%m-%d}, {datasets['all_data_total'].shape[0]} entries"
            f"\n 'all_data_jupiter': {jupiter_status}"
            f"\n 'all_data_unrolled': {unrolled_status}"
            f"\n 'phase': 721 points, 0 to 360 degrees, 0.5 degree steps ({phase_frame})"
            f"\n 'freqs': {freqs_status}",
        )

    return datasets


def sort_frequency_channels(datasets):
    """
    Finds the correct monotonic permutation for the frequency channels,
    sorts the main 'freqs' array, and reorders the corresponding frequency
    rows (axis 0) in all 2D spectrogram matrices.

    The LFR/HFR bands overlap between 1.4-1.7 MHz (Section 2), so the raw
    channel order isn't guaranteed monotonic -- this must be run right
    after reshaping the 1D data into 2D (128, 721) matrices.
    """
    if datasets is None:
        print("Error: Datasets dictionary is None.")
        return datasets

    if datasets.get("freqs") is None:
        print("Error: 'freqs' not loaded -- cannot sort channels.")
        return datasets

    raw_freqs = np.array(datasets["freqs"]).flatten()
    if len(raw_freqs) > 128:
        raw_freqs = raw_freqs[:128]

    sort_indices = np.argsort(raw_freqs)

    datasets["freqs"] = pd.Series(raw_freqs[sort_indices])
    print("Main frequency array sorted to be strictly monotonic.")

    for key in ["all_data_total", "all_data_jupiter"]:
        df = datasets.get(key)
        if df is not None and "plot" in df.columns:
            print(f"Reordering frequency channel rows for matrices in '{key}'...")
            df["plot"] = df["plot"].apply(lambda matrix: matrix[sort_indices, :])

    print("Frequency channel sorting alignment complete!")
    return datasets
