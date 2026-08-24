"""
Eigenfaces global-trend detection (Section 3.3 / 4.4).

NOTE on what was excluded during cleanup:
The original notebook contained an earlier, superseded implementation
(`compute_jovian_eigenfaces` / `project_onto_eigenfaces` /
`plot_eigenface_presence`) that computed presence as an un-normalized dot
product and then applied arbitrary visual multipliers (1e2-1e4x) to the
Jupiter-confirmed population before plotting, purely to make the two
populations visually separate on a shared axis. That version does not
match the paper's definition of C (Equation in Section 3.3: normalized
cosine similarity, bounded [-1, 1]) and was already superseded in the
notebook itself by the corrected functions below -- a later cell's
docstring explicitly notes the multiplier was removed "since real archive
separation is visible on log scale without an artificial multiplier."
The old version is not reproduced here since keeping both would let a
reader accidentally run the deprecated, visually-distorted version and
mistake it for the published metric.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_single_eigenface(datasets, target_type=2, freq_slice=None, phase_slice=None, cuts=None):
    """
    Slices, flattens, median-inpaints, and SVDs a confirmed-Jovian subset
    to extract the primary eigenface (dominant structural mode) for one
    emission morphology (1: Quiet HOM, 2: Noisy HOM, 3: Vertex-Late).
    """
    if datasets is None or "all_data_jupiter" not in datasets:
        print("Error: 'all_data_jupiter' not found in datasets.")
        return None

    type_names = {1: "Quiet HOM", 2: "Noisy HOM", 3: "Vertex-Late"}
    mech_name = type_names.get(target_type, f"Type {target_type}")

    print(f"Computing '{mech_name}' eigenface...")

    df_subset = datasets["all_data_jupiter"][datasets["all_data_jupiter"]["type"] == target_type].copy()
    if cuts:
        df_subset = df_subset.drop(index=[idx for idx in cuts if idx in df_subset.index])

    print(f"   -> Retained {len(df_subset)} valid {mech_name} arrays.")
    if len(df_subset) == 0:
        return None

    f_s = slice(freq_slice[0], freq_slice[1]) if freq_slice else slice(None)
    p_s = slice(phase_slice[0], phase_slice[1]) if phase_slice else slice(None)

    flattened_arrays = df_subset["plot"].apply(lambda matrix: matrix[f_s, p_s].flatten()).to_numpy()
    stacked_matrix = np.stack(flattened_arrays, axis=0).astype(float)

    col_medians = np.nanmedian(stacked_matrix, axis=0)
    nan_mask = np.isnan(stacked_matrix)
    stacked_matrix[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

    U, S, Vt = np.linalg.svd(stacked_matrix.T, full_matrices=False)
    eigenface = U[:, 0]
    eigenface = eigenface / np.linalg.norm(eigenface)  # explicit unit-norm safeguard

    print(f"   -> Extracted eigenface successfully (1D feature shape: {eigenface.shape}).")
    return eigenface


def project_single_eigenface(df, eigenface, freq_slice=None, phase_slice=None, outlier_threshold=1e11):
    """
    Slices target plots using the same bounds as the eigenface, flattens,
    inpaints, and computes the normalized projection
    C = (p . e) / sqrt((p.p)(e.e)) -- the coherence between each
    spectrogram and the dominant eigenface, bounded [-1, 1] (Section 3.3).

    The sign is preserved: a spectrogram anti-correlated with the
    eigenface template returns negative C rather than being folded into
    the same value as a correlated one. Returns both 'coherence' (signed)
    and 'presence' (=|coherence|, the detection metric used in Section 4.4).
    """
    f_s = slice(freq_slice[0], freq_slice[1]) if freq_slice else slice(None)
    p_s = slice(phase_slice[0], phase_slice[1]) if phase_slice else slice(None)

    flattened_arrays = np.stack(df["plot"].apply(lambda m: m[f_s, p_s].flatten()).to_numpy(), axis=0).astype(float)

    col_medians = np.nanmedian(flattened_arrays, axis=0)
    nan_mask = np.isnan(flattened_arrays)
    flattened_arrays[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

    e_norm = np.sqrt(eigenface @ eigenface)
    p_norms = np.sqrt(np.einsum("ij,ij->i", flattened_arrays, flattened_arrays))

    raw_dot = flattened_arrays @ eigenface
    denom = p_norms * e_norm
    with np.errstate(invalid="ignore", divide="ignore"):
        coherence = np.where(denom > 0, raw_dot / denom, np.nan)

    coherence = np.where(np.abs(raw_dot) > outlier_threshold, np.nan, coherence)

    return pd.DataFrame(
        {
            "time": pd.to_datetime(df["start_time"]),
            "coherence": coherence,
            "presence": np.abs(coherence),
        }
    )


def plot_eigenface_histogram(proj_total, proj_jupiter, bins=60, title="Eigenface Coherence Distribution", figsize=(12, 4.5)):
    """
    Histogram of |coherence| (Figure 6, bottom panel), on a log x-axis,
    shown as fraction-of-events per bin rather than matplotlib's
    density=True. With log-spaced bins, density=True normalizes by the
    (tiny, at the low end) linear bin width, inflating apparent density
    there in a way that has nothing to do with how many events actually
    fall in that bin. Fraction-of-events keeps the y-axis directly
    interpretable and comparable between populations of different N.
    """
    fig, ax = plt.subplots(figsize=figsize)

    total_vals = proj_total["presence"].to_numpy()
    jup_vals = proj_jupiter["presence"].to_numpy()

    total_vals = total_vals[np.isfinite(total_vals)]
    jup_vals = jup_vals[np.isfinite(jup_vals)]

    n_dropped_total = np.sum(total_vals <= 0)
    n_dropped_jup = np.sum(jup_vals <= 0)
    total_vals = total_vals[total_vals > 0]
    jup_vals = jup_vals[jup_vals > 0]

    if n_dropped_total or n_dropped_jup:
        print(f"Note: dropped {n_dropped_total} background and {n_dropped_jup} "
              f"Jupiter entries with presence <= 0 (cannot be shown on log axis).")

    lo = min(total_vals.min(), jup_vals.min())
    hi = max(total_vals.max(), jup_vals.max())
    log_bins = np.logspace(np.log10(lo), np.log10(hi), bins)

    w_total = np.full(len(total_vals), 1.0 / len(total_vals))
    w_jup = np.full(len(jup_vals), 1.0 / len(jup_vals))

    ax.hist(total_vals, bins=log_bins, weights=w_total, color="steelblue", alpha=0.6, label="No Jupiter")
    ax.hist(jup_vals, bins=log_bins, weights=w_jup, color="crimson", alpha=0.6, label="Jupiter")

    ax.set_xscale("log")
    ax.set_xlim(1e-7, 1)
    ax.set_xlabel("Coherence |C|", fontsize=14)
    ax.set_ylabel("Fraction of Events", fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, which="both", ls="--", alpha=0.3)

    plt.tight_layout()
    plt.show()
    return fig


def plot_eigenface_timeline(proj_total, proj_jupiter, title="Jovian Eigenface Presence Over Time",
                             ylabel="Coherence |C|", ylim=(1e-4, 1.5)):
    """Time-series scatter of |coherence| on a log y-axis (Figure 6, top panel)."""
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.scatter(proj_total["time"], proj_total["presence"], color="blue", s=3, alpha=0.5, label="No Jupiter")
    ax.scatter(proj_jupiter["time"], proj_jupiter["presence"], color="red", s=3, alpha=0.9, label="Jupiter")
    ax.set_yscale("log")
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_xlabel("Time", fontsize=16)
    ax.set_xlim(pd.to_datetime("2019-01-01"), pd.to_datetime("2025-01-01"))
    ax.set_title(title, fontsize=20, pad=10)
    ax.legend(loc="upper right", fontsize=14)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()
    plt.show()
    return fig
