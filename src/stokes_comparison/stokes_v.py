"""
Validation against the independently-processed Stokes V product
(Section 4.2): the polarimetric cross-check and the control experiment
showing the SVD extraction advantage is specific to the C^i_XY signal
chain rather than a generic property of SVD.

NOTE: `filter_by_time_range` was defined identically twice in the original
notebook (once for the raw load, once again just before the interpolation
step) -- consolidated to a single definition here.

NOTE: the Stokes V loader in the original notebook pointed at a single
hardcoded file, `monthly_stokes_v/stokes_data_2020_06.h5`. That file is
the same family produced by src/preprocessing/build_monthly_archive.py
as `data_<YYYY>_<MM>.h5` under RAW_MONTHLY_PATTERN (same monthly combined
archive; see that module's docstring for the full column layout) -- use
`extract_stokes_v_columns()` below to pull the Stokes V block out of it
by name, not `.iloc[:, -128:]` (the last 128 columns are actually the
A34 auto-average block, not Stokes V -- see build_monthly_archive.py).

The comparison shown in the original notebook built `stokes_filtered`
from one ~10-hour window on 2020-06-15. If the paper's reported
r = 0.783 (N = 65,611 pixels) and the rsvd,V = 0.32 control statistic are
meant to characterize a single representative window (consistent with
how Section 4.2 reads), this matches. If they're meant to be aggregated
across multiple high-rate telemetry windows, that aggregation loop isn't
present in this notebook and would need to be added/located before
publishing this module -- please confirm which is the case.
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter


def filter_by_time_range(df, start_time_str, end_time_str, time_col="time"):
    """Filters a DataFrame to keep only rows between two specific dates/times."""
    df[time_col] = pd.to_datetime(df[time_col])

    start_time = pd.to_datetime(start_time_str)
    end_time = pd.to_datetime(end_time_str)

    mask = (df[time_col] >= start_time) & (df[time_col] <= end_time)

    filtered_df = df.loc[mask].copy()
    filtered_df = filtered_df.reset_index(drop=True)

    return filtered_df


def extract_stokes_v_columns(df, n_channels=128):
    """
    Pulls the Stokes V block out of a monthly combined archive row set
    (see build_monthly_archive.py's column layout) by NAME, since its
    position (right after the 11 geometry columns, before C_IM/C_RE/A12/
    A34) is easy to get wrong with a positional slice -- see module
    docstring.
    """
    cols = [f"STOKES V channel {i+1}" for i in range(n_channels)]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Expected {n_channels} 'STOKES V channel N' columns; missing e.g. {missing[:3]}. "
            f"Is this a build_monthly_archive.py-produced file?"
        )
    return df[cols].to_numpy()


def resample_stokes_to_phase_grid(stokes_filtered, n_phase_target=721, n_channels=128):
    """
    Interpolates a raw Stokes V time-series slice (rows = time samples,
    columns = the `n_channels` Stokes V frequency channels -- use
    `extract_stokes_v_columns()` to get this from a raw monthly archive
    row set) onto the standard 721-point phase grid used throughout the
    rest of the pipeline, and orients it to (channels, phase) to match
    the C^i_XY matrix convention.
    """
    n_original = stokes_filtered.shape[0]
    x_original = np.linspace(0, 1, n_original)
    x_new = np.linspace(0, 1, n_phase_target)

    interpolator = interp1d(x_original, stokes_filtered, axis=0, kind="linear")

    # Sign flip and row-reversal match the frequency-channel ordering
    # convention used elsewhere in the pipeline (see sort_frequency_channels).
    resampled = -interpolator(x_new)[::-1, :].T
    return resampled


def compare_svd_to_stokes_v(svd_matrix, stokes_v_matrix, emission_mask=None):
    """
    Pearson correlation between the SVD-reconstructed C^i_XY spectrogram
    and an independently-processed Stokes V spectrogram over the same
    phase-frequency domain (the headline r = 0.783 statistic).
    """
    if emission_mask is not None:
        svd_flat = svd_matrix[emission_mask].flatten()
        sv_flat = stokes_v_matrix[emission_mask].flatten()
    else:
        svd_flat = svd_matrix.flatten()
        sv_flat = stokes_v_matrix.flatten()

    r, p = stats.pearsonr(svd_flat, sv_flat)

    print(f"Pearson r:  {r:.4f}")
    print(f"p-value:    {p:.2e}")
    print(f"N pixels:   {len(svd_flat)}")

    return r, p


def baseline_stokes_v_extraction(stokes_v_matrix, noise_percentile=75, smooth_sigma_freq=2.0, smooth_sigma_phase=2.0):
    """
    Simple (non-SVD) baseline for the Section 4.2 control experiment:
    row-wise percentile background subtraction (mirroring the C^i_XY
    pipeline's noise-percentile normalization) followed by 2D Gaussian
    smoothing.
    """
    background = np.percentile(stokes_v_matrix, noise_percentile, axis=1, keepdims=True)
    subtracted = stokes_v_matrix - background

    smoothed = gaussian_filter(subtracted, sigma=[smooth_sigma_freq, smooth_sigma_phase])

    return smoothed


def full_baseline_comparison(
    svd_matrix, stokes_v_raw, stokes_v_reference, noise_percentile=75, smooth_sigma_freq=2.0, smooth_sigma_phase=2.0, emission_mask=None
):
    """
    Computes Pearson r for both the SVD extraction and the simple
    percentile+smoothing baseline against the same independent Stokes V
    reference, for the direct performance comparison in Section 4.2
    (rsvd = 0.797 vs rsvd,V = 0.32 for the Stokes-V-input control).
    """
    baseline_output = baseline_stokes_v_extraction(
        stokes_v_raw, noise_percentile=noise_percentile, smooth_sigma_freq=smooth_sigma_freq, smooth_sigma_phase=smooth_sigma_phase
    )

    if emission_mask is not None:
        ref_flat = stokes_v_reference[emission_mask].flatten()
        svd_flat = svd_matrix[emission_mask].flatten()
        baseline_flat = baseline_output[emission_mask].flatten()
    else:
        ref_flat = stokes_v_reference.flatten()
        svd_flat = svd_matrix.flatten()
        baseline_flat = baseline_output.flatten()

    r_svd, p_svd = stats.pearsonr(svd_flat, ref_flat)
    r_baseline, p_baseline = stats.pearsonr(baseline_flat, ref_flat)

    print(f"{'Method':<25} {'Pearson r':>10} {'p-value':>12} {'N pixels':>10}")
    print("-" * 60)
    print(f"{'SVD Extraction':<25} {r_svd:>10.4f} {p_svd:>12.2e} {len(svd_flat):>10}")
    print(f"{'Baseline (V+sub+smooth)':<25} {r_baseline:>10.4f} {p_baseline:>12.2e} {len(baseline_flat):>10}")
    print(f"\nSVD improvement over baseline: {r_svd - r_baseline:+.4f}")

    return r_svd, r_baseline, baseline_output
