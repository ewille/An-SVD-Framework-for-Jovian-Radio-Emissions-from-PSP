"""
Upstream phase-folding step (Section 2 / 3.1): converts the monthly
combined HDF5 files (produced by
src/preprocessing/build_monthly_archive.py from the raw PSP FIELDS/RFS CDF
archive) into the phase-folded, noise-reduced master archive
(`data_theta_psp.h5` / `data_phi_io.h5`) that the rest of this pipeline
consumes.

For each calendar month with available raw data, this:
  1. Chunks the month into overlapping daily windows.
  2. Applies row-wise percentile whitening and channel>=110 mean
     subtraction -- the actual two stages behind the paper's "dual-stage
     noise filtering" (see note below).
  3. Detects and drops solar Type II/III burst-contaminated time samples
     (two independent detectors: a slow dynamic-baseline rise detector
     and a fast spike detector, merged).
  4. Corrects each timestamp for PSP-Jupiter light-travel time via SPICE,
     then computes PSP's Jovian System III longitude (the "psp theta" /
     lambda_III coordinate) for each corrected timestamp.
  5. Bins onto a uniform 0.5-degree phase grid, splits into per-rotation
     segments, and median-inpaints any phase bins left empty by the
     burst-sample removal in step 3 (using the archive-wide median across
     all segments in the month at that phase bin).
  6. Appends the resulting per-rotation segments to the master HDF5
     archive that src/preprocessing/load_data.py reads.

Ported from `phase_data_new.py` (confirmed against a near-duplicate,
`phase_all_data.py`, which had three separate broken/undefined-variable
bugs stemming from an unfinished attempt to add a parallel Io-phase
folding path -- `phase_data_new.py` is the version that actually runs).

*** CRITICAL FIX -- column selection was reading the wrong data ***
The original script selected its "spectra" and "pos" inputs by
POSITIONAL slice: `data.iloc[:, 8:136]` and `data.iloc[:, 5:8]`. Checked
against the column layout build_monthly_archive.py actually produces (see
that module's docstring), those positions land on 4 leftover geometry
columns plus Stokes V channels 1-124 -- NOT the C IM (C^i_XY) block,
which the paper states the entire extraction method relies on exclusively
(Section 2). The real C IM block is at columns 140-267 by position, or
more robustly, the columns named "C IM channel 1".."C IM channel 128".
This module now selects both "spectra" and "pos" by column NAME instead
of position, which fixes this and is robust to future column-order
changes. This needs a sanity check against how the paper's actual
reported results were produced -- if a different (already-correct)
column mapping was used for the real analysis, that's a good sign this
fix matches; if the paper's results were actually produced from the
mis-mapped columns, the reported statistics would need to be regenerated.

NOTE -- resolves an earlier open question about "dual-stage" noise
filtering: the abstract's "deterministic, dual-stage noise filtering
process" happens HERE (step 2 above: percentile scaling + channel>=110
mean subtraction), not in the analysis notebook. The notebook's
`clean_spectrogram_noise` (src/preprocessing/noise_reduction.py) reapplies
Stage 1 (percentile scaling) again on already-whitened data and correctly
leaves Stage 2 disabled, since it's already been applied here. Reapplying
percentile scaling to already-scaled data isn't strictly a no-op, but the
effect should be small; worth confirming that's intentional rather than
an accidental double-application.

NOTE -- fix applied during porting: the original script computed a
flattened 1D version of each segment's spectrogram (`unrolled = ...`) but
the line was commented out, so it stored the un-flattened 2D/DataFrame
`plot` object directly under the key `"plot"`. Downstream,
`format_all_data_total()` (noise_reduction.py) expects a column named
`"unrolled_plot"` containing flattened 1D arrays, which it then reshapes
back to (128, 721). This module restores that flattening step and stores
it under `"unrolled_plot"` so the two stages actually connect -- please
confirm this matches what you intended (i.e. that the flattening was
disabled for local debugging rather than deliberately).

NOTE -- three small functions from the original script (`SVD`,
`normalize_vector`, `shannon_entropy`, an apparent early draft of an
SVD/entropy-based burst detector) were never called anywhere in it and
are not reproduced here. The burst detection actually used is the
dynamic-baseline / spike-detector pair below.

Unused imports from the original script (sys, re, time, datetime as a
bare import, scipy, astropy.time, date, the bare `svd` import, bisect,
defaultdict, Parallel/delayed, glob, spacepy.pycdf, statistics, norm,
curve_fit, plt, mdates, gridspec, gaussian_filter1d, label, h5py) have
been dropped; none were referenced anywhere in the script.
"""
import os
import numpy as np
import pandas as pd
import spiceypy as spice
from datetime import timedelta
from scipy.signal import savgol_filter

from ..config import SPICE_KERNELS, RAW_MONTHLY_PATTERN, MASTER_ARCHIVE_PATH, PATHS

C_KM_PER_S = 299792.458

ANGLE_INDEX = np.round(np.arange(0, 360.5, 0.5), 1)


# ==========================================
# Burst detection
# ==========================================

def compute_dynamic_baseline(signal, window_size=201):
    series = pd.Series(signal)
    return series.rolling(window=window_size, center=True, min_periods=1).median().values


def detect_transients_dynamic_baseline(
    signal, rise_thresh=0.4, margin=0.1, decay_patience=10, smoothing_window=11, baseline_window=201
):
    """Slow-rise burst detector: flags a sustained departure from a rolling-median baseline."""
    signal = np.asarray(signal)
    smoothed = savgol_filter(signal, window_length=smoothing_window, polyorder=3)
    diff = np.diff(smoothed)
    baseline = compute_dynamic_baseline(smoothed, window_size=baseline_window)

    transients = []
    in_transient = False
    start = None
    low_count = 0

    for i in range(1, len(smoothed)):
        if not in_transient:
            if diff[i - 1] > rise_thresh:
                in_transient = True
                start = i - 1
                low_count = 0
        else:
            if smoothed[i] < baseline[i] + margin:
                low_count += 1
                if low_count >= decay_patience:
                    end = i
                    transients.append((start, end))
                    in_transient = False
            else:
                low_count = 0

    if in_transient:
        transients.append((start, len(smoothed) - 1))

    return transients, smoothed, baseline


def detect_spike_transients(
    signal, slope_thresh=0.09, decay_window=41, decay_margin=0.07, decay_patience=6, smoothing_window=9, consec_rise_thresh=3
):
    """Fast-spike burst detector: flags several consecutive steep-slope samples."""
    signal = np.asarray(signal)
    smoothed = savgol_filter(signal, window_length=smoothing_window, polyorder=3)
    diff = np.diff(smoothed)
    local_mean = pd.Series(smoothed).rolling(window=decay_window, center=True, min_periods=1).mean().values

    transients = []
    in_transient = False
    start = None
    low_count = 0
    consec_rise_count = 0

    for i in range(1, len(smoothed)):
        if not in_transient:
            if diff[i - 1] > slope_thresh:
                consec_rise_count += 1
                if consec_rise_count >= consec_rise_thresh:
                    in_transient = True
                    start = i - consec_rise_thresh
                    low_count = 0
                    consec_rise_count = 0
            else:
                consec_rise_count = 0
        else:
            if smoothed[i] < local_mean[i] + decay_margin:
                low_count += 1
                if low_count >= decay_patience:
                    end = i
                    transients.append((start, end))
                    in_transient = False
            else:
                low_count = 0

    if in_transient:
        transients.append((start, len(smoothed) - 1))

    return transients, smoothed, local_mean


def merge_ranges(*range_lists, buffer=1):
    """Merges overlapping/adjacent (start, end) index ranges from multiple detectors into a flat index list."""
    all_ranges = sorted([r for lst in range_lists for r in lst], key=lambda x: x[0])
    merged = []

    for start, end in all_ranges:
        if not merged:
            merged.append((start, end))
        else:
            prev_start, prev_end = merged[-1]
            if start <= prev_end + buffer:
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))

    all_indices = []
    for start, end in merged:
        all_indices.extend(range(start, end + 1))

    return all_indices


# ==========================================
# Phase-grid binning and median inpainting
# ==========================================

def build_universal_median(spectra_dict, angle_type="psp theta"):
    """Builds a per-phase-bin median spectrum across all segments in a month, for inpainting gaps."""
    all_spectra = []

    for df in spectra_dict.values():
        angle_col = df[angle_type]
        freq_data = df.iloc[:, 2:].copy()

        freq_data.index = angle_col
        freq_data = freq_data.groupby(freq_data.index).mean()
        freq_data = freq_data.reindex(ANGLE_INDEX)

        all_spectra.append(freq_data)

    stacked = np.stack([spec.values for spec in all_spectra])
    nan_angle_mask = np.all(np.isnan(stacked), axis=(0, 2))
    dropped_angles = ANGLE_INDEX[nan_angle_mask].tolist()

    if dropped_angles:
        print(f"Dropping {len(dropped_angles)} angle rows with all-NaN data...")

    valid_angles = ANGLE_INDEX[~nan_angle_mask]
    valid_idx = np.where(~nan_angle_mask)[0]

    median_spec = np.nanmedian(stacked[:, valid_idx, :], axis=0)
    median_df = pd.DataFrame(median_spec, index=valid_angles)

    return median_df, valid_angles


def inpaint_spectra_with_median(spectra_dict, median_df, valid_angles, angle_type="psp theta"):
    """Fills phase bins left empty by burst-sample removal using the month's median spectrum at that bin."""
    filled_dict = {}

    for key, df in spectra_dict.items():
        angle_vals = df[angle_type]
        time_vals = pd.to_datetime(df["times"])

        freq_data = df.iloc[:, 2:].copy()

        freq_data.index = angle_vals
        freq_data = freq_data.groupby(freq_data.index).mean()
        freq_data = freq_data.reindex(valid_angles)

        time_series = pd.Series(time_vals.values, index=angle_vals)
        time_series = time_series.groupby(time_series.index).mean()
        time_series = time_series.reindex(valid_angles)
        time_interp = pd.to_datetime(time_series.interpolate(limit_direction="both"))

        missing_rows = freq_data.isna().all(axis=1)
        freq_data[missing_rows] = median_df.loc[missing_rows]

        final_df = pd.concat(
            [
                pd.Series(valid_angles, name=angle_type),
                pd.Series(time_interp.values, name="times"),
                freq_data.reset_index(drop=True),
            ],
            axis=1,
        )

        filled_dict[key] = final_df

    return filled_dict


def detect_wrap_indices(df, angle_col):
    """Finds indices where the phase angle wraps (jumps by >300 deg), marking rotation boundaries."""
    diffs = np.diff(df[angle_col].values)
    wrap_indices = np.where(np.abs(diffs) > 300)[0]
    return wrap_indices.tolist()


def slice_df_by_wrap_and_sort(df, angle_col):
    """Splits a continuous phase-tagged DataFrame into one segment per Jovian rotation."""
    wrap_indices = detect_wrap_indices(df, angle_col)
    cut_points = [0] + [i + 1 for i in wrap_indices] + [len(df)]

    segments = {}
    for i in range(len(cut_points) - 1):
        seg = df.iloc[cut_points[i] : cut_points[i + 1]].copy()
        seg[angle_col] = pd.to_numeric(seg[angle_col], errors="coerce")
        seg = seg.dropna(subset=[angle_col])
        if seg.columns.size > 1:
            seg.columns = list(seg.columns)
            seg.columns.values[0] = "times"
        segments[f"segment_{i}"] = seg

    print(f"Sliced {len(segments)} segments from {angle_col}.")
    return segments


def extract_all_segments(psp_check, psp_data, freqs, year, month, angle_col="psp theta"):
    """
    Builds the master-archive rows for one month: one row per rotation
    segment, each holding its start/end time and its flattened spectrogram.

    `angle_col` names whichever phase column was used for binning --
    "psp theta" (lambda_III) or "io phase" (Phi_Io); see `process_month`'s
    `phase_frame` parameter.
    """
    annotations_list = []
    for i in range(len(psp_check) - 1):
        segment_data = psp_data[f"segment_{i}"]

        z_data = segment_data.iloc[:, 2:].T

        theta_col = segment_data[angle_col]
        if isinstance(theta_col, pd.DataFrame):
            theta_col = theta_col.iloc[:, 0]
        thetas = theta_col.values

        times_col = segment_data["times"]
        if isinstance(times_col, pd.DataFrame):
            times_col = times_col.iloc[:, 0]
        times = pd.to_datetime(times_col).reset_index(drop=True)

        start_idx = np.argmin(np.abs(thetas - 0))
        start_time = times.iloc[start_idx].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        end_idx = np.argmin(np.abs(thetas - 360))
        end_time = times.iloc[end_idx].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        # Flattened 1D form -- see module docstring "fix applied during
        # porting". This line was present but commented out (and its
        # result unused) in the original script; format_all_data_total()
        # downstream expects exactly this, under "unrolled_plot".
        unrolled = np.reshape(z_data.values, -1).tolist()

        annotations_list.append(
            {
                "segment": f"{year}_{month:02d}_segment_{i}",
                "start_time": start_time,
                "end_time": end_time,
                "unrolled_plot": unrolled,
            }
        )

    return pd.DataFrame(annotations_list)


# ==========================================
# Per-month processing
# ==========================================

def _whiten_and_remove_bursts(spectra_day):
    """The paper's two-stage noise filtering, applied per calendar day (see module docstring)."""
    for i in range(spectra_day.shape[0]):
        data_row = spectra_day.iloc[i]
        median = np.median(data_row)
        positive_half = data_row[data_row >= median]
        q3 = np.percentile(positive_half, 75)
        spectra_day.iloc[i] = spectra_day.iloc[i] / (abs(q3 - median) + 1e-9)  # safety epsilon

    mean_per_row = np.mean((spectra_day.iloc[110:]).to_numpy(), axis=1, keepdims=True)
    spectra_mean_subtract = pd.DataFrame(spectra_day.iloc[110:] - mean_per_row)
    spectra_day = pd.concat([spectra_day.iloc[:110], spectra_mean_subtract])

    solar = [np.median(np.abs(spectra_day.iloc[:, i])) for i in range(spectra_day.shape[1])]

    transients_b, _, _ = detect_transients_dynamic_baseline(
        solar, rise_thresh=0.12, margin=0.035, decay_patience=13, smoothing_window=9, baseline_window=85
    )
    transients_l, _, _ = detect_spike_transients(
        solar, slope_thresh=0.09, decay_window=51, decay_margin=0.07, decay_patience=6, smoothing_window=9, consec_rise_thresh=3
    )

    solar_merged = merge_ranges(transients_b, transients_l)

    return spectra_day, solar_merged


def process_month(year, month, freqs, master_filepath=None, raw_monthly_pattern=None, phase_frame="lambda_iii"):
    """
    Processes one calendar month of raw PSP data into phase-folded,
    noise-reduced rotation segments, and appends them to the master
    archive HDF5. Assumes SPICE kernels are already furnsh'd (see
    fold_archive below).

    Args:
        phase_frame (str): "lambda_iii" (default) bins/folds on PSP's
            Jovian System III longitude and writes to `data_theta_psp.h5`
            by default; "phi_io" bins/folds on Io phase instead and
            writes to `data_phi_io.h5` by default. Both are computed from
            the same raw monthly file -- see build_monthly_archive.py -- via
            two separate calls with different `phase_frame`.

    Returns the number of segments appended, or None if the month's raw
    file wasn't found.
    """
    if phase_frame not in ("lambda_iii", "phi_io"):
        raise ValueError("phase_frame must be 'lambda_iii' or 'phi_io'")

    angle_col = "psp theta" if phase_frame == "lambda_iii" else "io phase"
    default_master_path = PATHS["all_data_total"] if phase_frame == "lambda_iii" else PATHS["all_data_total_phi_io"]

    master_filepath = str(master_filepath or default_master_path)
    raw_monthly_pattern = raw_monthly_pattern or RAW_MONTHLY_PATTERN
    filename = raw_monthly_pattern.format(year=year, month=month)

    if not os.path.exists(filename):
        print(f"[{year}-{month:02d}] File not found: {filename}, skipping.")
        return None

    print(f"\n=====================================")
    print(f"Processing Year: {year}, Month: {month:02d}")
    print(f"=====================================")

    data = pd.read_hdf(filename)
    print(f"Loaded {data.shape[0]} rows from {filename}")

    times = data["time"]
    # Name-based selection -- see this module's docstring "CRITICAL FIX".
    # C IM ("C IM channel 1".."C IM channel 128") is the C^i_XY product
    # the paper's extraction method relies on exclusively (Section 2).
    cim_cols = [f"C IM channel {i+1}" for i in range(128)]
    pos_cols = ["x_pos", "y_pos", "z_pos"]
    spectra = pd.concat([data[cim_cols]]).T
    pos = pd.concat([data[pos_cols]]).T
    iau = data["IAU.PSP"]

    dropna = spectra.dropna(axis=1).columns
    spectra = spectra[dropna].reset_index(drop=True)
    iau = iau.loc[dropna].reset_index(drop=True)
    pos = pos[dropna].reset_index(drop=True)
    times = times.loc[dropna].reset_index(drop=True)

    times_series = pd.Series(times)
    if times_series.empty:
        print("No valid times after dropping NA. Skipping.")
        return None

    start_time = times_series.iloc[0]
    end_time = times_series.iloc[-1]
    delta = timedelta(days=1)

    # Chunk into daily windows
    dict_by_day = []
    current_start = start_time
    while current_start < end_time:
        current_end = current_start + delta
        mask = (times_series >= current_start) & (times_series < current_end)
        idx = mask[mask].index

        if len(idx) < 9:
            current_start = current_end
            continue

        time_slice = times_series.iloc[idx].reset_index(drop=True)
        spectra_slice = spectra.iloc[:, idx].reset_index(drop=True)
        pos_slice = pos.iloc[:, idx].reset_index(drop=True)

        dict_by_day.append(
            {
                "start_time": current_start,
                "end_time": current_end,
                "times": time_slice,
                "spectra": spectra_slice.copy(),
                "freqs": freqs,
                "position": pos_slice,
                "jupiter": iau,
            }
        )
        current_start = current_end

    # Whiten and remove solar bursts, per day
    dict_continuous = []
    for day_chunk in dict_by_day:
        spectra_day = day_chunk["spectra"].T.reset_index(drop=True).T
        times_day = day_chunk["times"]
        position_day = day_chunk["position"].T.reset_index(drop=True).T

        spectra_day, solar_merged = _whiten_and_remove_bursts(spectra_day)

        spectra_cols = spectra_day.columns[solar_merged]
        spectra_day = spectra_day.drop(columns=spectra_cols, axis=1)
        position_day = position_day.drop(columns=spectra_cols, axis=1)
        times_day = times_day.drop(labels=solar_merged)

        dict_continuous.append({"times": times_day, "position": position_day, "spectra": spectra_day})

    # Light-travel-time correction via SPICE
    for cont in dict_continuous:
        corrected_times = []
        for t in cont["times"]:
            et = spice.utc2et(t.strftime("%Y-%m-%dT%H:%M:%S.%f"))
            state_psp_sun, _ = spice.spkezr("PARKER SOLAR PROBE", et, "J2000", "NONE", "SUN")
            state_jup_sun, _ = spice.spkezr("JUPITER BARYCENTER", et, "J2000", "NONE", "SUN")
            distance_km = np.linalg.norm(np.subtract(state_psp_sun[:3], state_jup_sun[:3]))
            delay_sec = distance_km / C_KM_PER_S
            corrected_times.append(t - timedelta(seconds=delay_sec))
        cont["times"] = corrected_times

    # PSP Jovian System III longitude and Io phase via SPICE, for every
    # corrected timestamp. Both are computed regardless of `phase_frame`
    # so either can be selected for binning below.
    dict_psp_raw = []

    for cont in dict_continuous:
        times_cont = cont["times"]
        spectra_cont = cont["spectra"]

        theta_psp_list = []
        theta_io_list = []

        for t in times_cont:
            et = spice.utc2et(t.strftime("%Y-%m-%dT%H:%M:%S.%f"))

            state_psp_sun, _ = spice.spkezr("PARKER SOLAR PROBE", et, "J2000", "NONE", "SUN")
            state_jup_sun, _ = spice.spkezr("JUPITER BARYCENTER", et, "J2000", "NONE", "SUN")
            pos_rel = np.subtract(state_psp_sun[:3], state_jup_sun[:3])
            rotmat = spice.pxform("J2000", "IAU_JUPITER", et)
            pos_psp_rot = spice.mxv(rotmat, pos_rel)
            theta_psp_list.append(np.degrees(np.arctan2(pos_psp_rot[1], pos_psp_rot[0])) % 360)

            state_io, _ = spice.spkezr("IO", et, "J2000", "NONE", "JUPITER")
            pos_io_rot = spice.mxv(rotmat, state_io[:3])
            theta_io_list.append(np.degrees(np.arctan2(pos_io_rot[1], pos_io_rot[0])) % 360)

        theta_psp_array = np.array(theta_psp_list)
        theta_io_array = np.array(theta_io_list)
        phase_io = np.mod(theta_psp_array + 180 - theta_io_array, 360)

        dict_psp_raw.append({"io theta": theta_io_array, "psp theta": theta_psp_array, "io phase": phase_io, "spectra": spectra_cont})

    # Concatenate the month, split by rotation -- folds on whichever
    # angle `angle_col` selects ("psp theta" for lambda_iii, "io phase"
    # for phi_io; both were computed above for every timestamp).
    spectra_full = pd.DataFrame([])
    times_full = pd.DataFrame([])
    angle_full = pd.DataFrame([])

    for i, cont in enumerate(dict_continuous):
        angle_init = pd.DataFrame(dict_psp_raw[i][angle_col], columns=[angle_col])
        angle_full = pd.concat([angle_full, angle_init], ignore_index=True)
        spectra_full = pd.concat([spectra_full, cont["spectra"]], axis=1)
        times_full = pd.concat([times_full, pd.Series(cont["times"])]).reset_index(drop=True)

    angle_full_rounded = [round(2 * angle_full.iloc[j].values[0], 0) / 2 for j in range(angle_full.shape[0])]
    spectra_full = spectra_full.T.reset_index(drop=True)
    continuous_phase_data = pd.concat([times_full, pd.DataFrame(angle_full_rounded, columns=[angle_col]), spectra_full], axis=1)

    psp_segments = slice_df_by_wrap_and_sort(continuous_phase_data, angle_col)
    for key in psp_segments:
        psp_segments[key] = psp_segments[key].groupby(angle_col).agg("mean").reset_index()

    median_inpaint_psp, psp_valid_angles = build_universal_median(psp_segments, angle_col)
    psp_inpainted = inpaint_spectra_with_median(psp_segments, median_inpaint_psp, psp_valid_angles, angle_type=angle_col)

    print(f"Extracting annotations for {year}-{month:02d}...")
    annotations = extract_all_segments(psp_segments, psp_inpainted, freqs, year, month, angle_col=angle_col)

    try:
        add_to_master = pd.read_hdf(master_filepath, "master_list")
    except (KeyError, FileNotFoundError):
        print("Master file or key not found. Initializing a new master DataFrame.")
        add_to_master = pd.DataFrame(columns=["segment", "start_time", "end_time", "unrolled_plot"])

    add_to_master = pd.concat([add_to_master, annotations], ignore_index=True)
    add_to_master.to_hdf(master_filepath, key="master_list", mode="w")
    print(f"Successfully appended {len(annotations)} segments to {master_filepath}.")

    return len(annotations)


# ==========================================
# Archive-wide driver
# ==========================================

def fold_archive(years=range(2019, 2025), months=range(1, 13), kernels=None, freqs_path=None, master_filepath=None, raw_monthly_pattern=None, phase_frame="lambda_iii"):
    """
    Loads SPICE kernels once, then processes every (year, month) with
    available raw data, appending each month's segments to the master
    archive HDF5 for the given `phase_frame` ("lambda_iii" or "phi_io").

    To build both final archives (data_theta_psp.h5 and data_phi_io.h5)
    from the same monthly input, call this twice with each phase_frame.
    """
    for kernel_path in (kernels or SPICE_KERNELS):
        spice.furnsh(str(kernel_path))

    freqs = pd.read_csv(str(freqs_path or PATHS["freqs"]))

    for year in years:
        for month in months:
            process_month(year, month, freqs, master_filepath=master_filepath, raw_monthly_pattern=raw_monthly_pattern, phase_frame=phase_frame)

    print("\nProcessing complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2019)
    parser.add_argument("--end-year", type=int, default=2024, help="Inclusive.")
    parser.add_argument("--phase-frame", choices=["lambda_iii", "phi_io", "both"], default="both",
                         help="Which archive(s) to build. 'both' (default) runs both passes, producing data_theta_psp.h5 and data_phi_io.h5 from the same monthly input.")
    args = parser.parse_args()

    frames = ["lambda_iii", "phi_io"] if args.phase_frame == "both" else [args.phase_frame]
    for frame in frames:
        print(f"\n### Folding to phase_frame='{frame}' ###")
        fold_archive(years=range(args.start_year, args.end_year + 1), phase_frame=frame)
