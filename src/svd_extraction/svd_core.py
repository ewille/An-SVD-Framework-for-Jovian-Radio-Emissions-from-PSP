"""
SVD extraction core (Section 3.2): entropy-thresholded rank selection,
Eckart-Young reconstruction, and empirical SNR comparison.

*** CRITICAL -- NEEDS AUTHOR VERIFICATION BEFORE ANYTHING ELSE IN THIS REPO ***

The `boolean` parameter controls the direction of the entropy comparison:

    kept_modes = [i for i in range(num_modes_to_check)
                  if boolean * entropies[i] > boolean * entropy_limit]

With the default `boolean=1` (used at every call site in the original
notebook -- confirmed by an exhaustive search; `boolean` is never
overridden anywhere), this reduces algebraically to:

    entropies[i] > entropy_limit   # KEEPS HIGH-entropy modes

The paper (Section 3.2, directly below Equation 6) states the opposite
criterion: "A mode is retained in the Eckart-Young reconstruction if and
only if S_i < S_lim" -- i.e. LOW-entropy (structured, concentrated) modes
should be kept and HIGH-entropy (noise-like, spread out) modes discarded.
`boolean=-1` would implement that ( -entropies[i] > -entropy_limit is
equivalent to entropies[i] < entropy_limit ), but nothing in the notebook
ever sets it.

This matters because the real-data outputs in the original notebook are
consistent with the "keep high entropy" reading, not the paper's stated
"keep low entropy" one: calling this with entropy_limit=3 and
num_modes_to_check=15 on file_index=245 kept 14 of the 15 candidate
modes; num_modes_to_check=10 on file_index=489 kept 9 of 10. That's
consistent with "almost everything with entropy above 3 nats passes" (a
threshold that excludes almost nothing, since entropy for a 90-ish
channel sub-band clusters not far below the log(90) ~= 4.5 nat ceiling
for both structured and noise-like modes in these two examples) rather
than a genuinely selective low-rank truncation. If that pattern holds
generally, the reported k = 9.56 +/- 0.35 retained-mode statistic (Section
4.1) would mostly reflect whatever `num_modes_to_check` was manually
passed at each of the 50 validation-event call sites, rather than an
emergent, data-driven quantity -- which would undercut the "low standard
error... indicates the pipeline consistently isolates Jovian emission
structure" argument built on top of it.

This function is reproduced here EXACTLY as it behaved in the original
notebook (boolean=1 default, unchanged) -- I have not silently "fixed"
the direction, since that would change the pipeline's actual output and
is a scientific decision, not a code-cleanup one. Please verify which
direction is correct against how the paper's reported statistics were
actually generated before treating anything in this module as final.
"""
import numpy as np
import matplotlib.pyplot as plt


def _normalize_vector(v):
    p = np.abs(v) ** 2
    s = np.sum(p)
    return p / s if s > 0 else p


def _shannon_entropy(p):
    p_nz = p[p > 0]
    return -np.sum(p_nz * np.log(p_nz))


def run_svd_extraction(
    datasets=None,
    file_index=None,
    standalone_matrix=None,
    standalone_freqs=None,
    standalone_phase=None,
    channel_range=(30, 119),
    num_modes_to_check=13,
    entropy_limit=2.5,
    entropy_mode="U",
    boolean=1,
    mode="all",
    v_limit=1.0,
    emission_title="Jovian Emission",
    plot=True,
):
    """
    Performs SVD on a single spectrogram (from the datasets dict, by index,
    or a standalone matrix), retains modes with Shannon entropy below
    `entropy_limit` (default matches the paper's S_lim = 3.0 nats when
    called with entropy_limit=3), and optionally renders the four
    diagnostic panels used to build Figures 3/8-11 (singular values,
    post-processed spectrum, SVD-reconstructed spectrum, residuals).

    Originally `analyze_single_index_svd`; renamed for clarity when split
    out of the monolithic notebook. Returns (svd_reconstructed, residuals),
    where residuals = svd_reconstructed - band_matrix (note: this is the
    negative of the paper's M_res = M - M_k; harmless for downstream
    autocorrelation use since autocorrelation is sign-invariant, but keep
    the convention in mind if used elsewhere).

    Parameters
    ----------
    datasets : dict, optional
        Dict containing 'all_data_total', 'all_data_jupiter', 'freqs', 'phase'.
    file_index : int, optional
        Row index of the file to analyze (if using `datasets`).
    standalone_matrix : ndarray, optional
        A 2D array to process directly, bypassing `datasets`.
    standalone_freqs, standalone_phase : ndarray, optional
        Axes for `standalone_matrix`.
    channel_range : tuple
        (first_channel, last_channel) sub-band to isolate.
    num_modes_to_check : int
        Number of top singular modes to evaluate via Shannon entropy.
    entropy_limit : float
        Cutoff value S_lim for the Shannon entropy filter (nats).
    entropy_mode : str
        'U' for frequency-domain entropy, 'Vt' for phase-domain entropy.
    boolean : int
        1 or -1, controls the inequality direction for entropy selection.
    mode : str
        'all' -> all_data_total, 'jupiter' -> all_data_jupiter.
    v_limit : float
        Colorbar +/- limits for uniform scaling across plots.
    emission_title : str
        Label used in plot titles.
    plot : bool
        If False, skip rendering and just return the arrays.

    Returns
    -------
    (svd_reconstructed, residuals) : tuple of ndarray, or (None, None) on error.
    """
    if standalone_matrix is not None:
        full_matrix = np.asarray(standalone_matrix)
        freqs_full = np.asarray(standalone_freqs) if standalone_freqs is not None else np.arange(full_matrix.shape[0])
        phase = np.asarray(standalone_phase) if standalone_phase is not None else np.arange(full_matrix.shape[1])
        print(f"Starting SVD pipeline for standalone matrix ({emission_title})")
    else:
        if datasets is None or file_index is None:
            print("Error: provide either 'standalone_matrix' OR both 'datasets' and 'file_index'.")
            return None, None

        data_key = "all_data_total" if mode == "all" else "all_data_jupiter"
        if data_key not in datasets:
            print(f"Error: '{data_key}' not found in datasets dictionary.")
            return None, None

        df = datasets[data_key]
        if not (0 <= file_index < len(df)):
            print(f"Error: invalid file index ({file_index}). Range is 0 to {len(df)-1}.")
            return None, None

        full_matrix = df["plot"].iloc[file_index]
        freqs_full = datasets["freqs"].to_numpy().flatten()
        phase = np.asarray(datasets["phase"])
        print(f"Starting SVD pipeline for file index {file_index} ({emission_title}) [mode: {mode}]")

    start_ch, end_ch = channel_range
    if start_ch < 0 or end_ch >= full_matrix.shape[0] or start_ch > end_ch:
        print(f"Error: invalid channel range {channel_range}. Must be within (0, {full_matrix.shape[0]-1}).")
        return None, None

    band_matrix = full_matrix[start_ch : end_ch + 1, :]
    freqs = freqs_full[start_ch : end_ch + 1]

    print(f"   -> Sub-band isolated: {band_matrix.shape[0]} frequency channels x {band_matrix.shape[1]} phase angles.")

    U, S, Vt = np.linalg.svd(band_matrix, full_matrices=False)

    if entropy_mode == "U":
        entropies = [_shannon_entropy(_normalize_vector(U[:, i])) for i in range(U.shape[1])]
    elif entropy_mode == "Vt":
        entropies = [_shannon_entropy(_normalize_vector(Vt[i, :])) for i in range(Vt.shape[0])]
    else:
        raise ValueError("Invalid entropy_mode. Select either 'U' or 'Vt'.")

    kept_modes = [i for i in range(num_modes_to_check) if boolean * entropies[i] > boolean * entropy_limit]
    print(f"   -> Top {num_modes_to_check} modes scanned. Kept low-entropy modes: {kept_modes}")

    S_clean = np.zeros_like(S)
    if kept_modes:
        S_clean[kept_modes] = S[kept_modes]
    svd_reconstructed = (U @ np.diag(S_clean)) @ Vt

    residuals = svd_reconstructed - band_matrix

    if plot:
        plt.figure(figsize=(11.5, 4.2))
        modes_x = np.arange(len(S))
        plt.semilogy(modes_x, S, "o-", color="royalblue", label="Discarded Modes", markersize=5, alpha=0.7)
        if kept_modes:
            plt.semilogy(kept_modes, S[kept_modes], "o", color="crimson", label="Kept Signal Modes", markersize=7)
        plt.title("Singular Values of Spectrum", fontsize=20)
        plt.xlabel("Singular Mode Rank", fontsize=16)
        plt.ylabel("Singular Value Magnitude", fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.legend(frameon=True, fontsize=14)
        plt.tight_layout()
        plt.show()

        spectrograms_to_draw = [
            (band_matrix, "Post-Processed Spectrum"),
            (svd_reconstructed, "Spectrum with SVD Applied"),
            (residuals, "Residuals"),
        ]

        for matrix_data, title_text in spectrograms_to_draw:
            plt.figure(figsize=(13, 4.2))
            im = plt.pcolormesh(phase, freqs, matrix_data, shading="auto", vmin=-v_limit, vmax=v_limit, cmap="viridis")
            plt.yscale("log")
            plt.title(title_text, fontsize=20)
            plt.xlabel("Jovian Longitude (Phase Degrees)", fontsize=16)
            plt.ylabel("Frequency (Hz)", fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            cb = plt.colorbar(im, pad=0.015)
            cb.set_label("Intensity Scale", fontsize=14)
            cb.ax.tick_params(labelsize=14)
            plt.tight_layout()
            plt.show()

        print("SVD analysis complete.")

    return svd_reconstructed, residuals


def analyze_spectrogram_snr(datasets, file_index, mode="jupiter", noise_percentile=10, channel_range=(30, 119), vmin=0, vmax=20, plot=True):
    """
    Calculates and (optionally) plots the empirical SNR (dB) for a single
    spectrogram, using a per-channel baseline from `noise_percentile` of
    squared amplitudes across phase bins (Section 3.2).
    """
    if datasets is None:
        print("Error: datasets not loaded.")
        return None

    data_key = "all_data_total" if mode == "all" else "all_data_jupiter"
    if data_key not in datasets:
        print(f"Error: key '{data_key}' not found.")
        return None

    df = datasets[data_key]

    start_ch, end_ch = channel_range
    full_matrix = df["plot"].iloc[file_index]

    band_matrix = full_matrix[start_ch : end_ch + 1, :]
    band_freqs = datasets["freqs"].iloc[start_ch : end_ch + 1].to_numpy().flatten()
    phase = datasets["phase"]

    power_matrix = band_matrix ** 2

    noise_baseline = np.percentile(power_matrix, noise_percentile, axis=1, keepdims=True)
    noise_baseline = np.where(noise_baseline == 0, 1e-12, noise_baseline)

    snr_linear = power_matrix / noise_baseline
    snr_db = 10 * np.log10(snr_linear + 1e-12)

    peak_snr = np.max(snr_db)
    avg_snr = np.mean(snr_db[snr_db > 3])  # active-region average, matches Section 3.2's >3 dB mask

    print(f"SNR analysis for index {file_index} ({mode} mode) | channels {start_ch}-{end_ch}")
    print(f"   -> Peak SNR: {peak_snr:.2f} dB")
    print(f"   -> Average active SNR (>3 dB): {avg_snr:.2f} dB")

    if plot:
        plt.figure(figsize=(14, 5))
        im = plt.pcolormesh(phase, band_freqs, snr_db, shading="auto", cmap="magma", vmin=vmin, vmax=vmax)
        plt.yscale("log")
        plt.title(f"Jovian Emission SNR Spectrogram (Index: {file_index} | Channels {start_ch}-{end_ch})\n"
                  f"Baseline: {noise_percentile}th Percentile", fontsize=15, fontweight="bold")
        plt.xlabel("Jovian Longitude (Phase Degrees)", fontsize=12)
        plt.ylabel("Frequency (Hz)", fontsize=12)
        cb = plt.colorbar(im, pad=0.015)
        cb.set_label("Signal-to-Noise Ratio (dB)", fontsize=12, fontweight="bold")
        plt.contour(phase, band_freqs, snr_db, levels=[10], colors="blue", alpha=1, linewidths=1, linestyles="dashed")
        plt.tight_layout()
        plt.show()

    return snr_db


def compare_svd_snr(
    datasets=None,
    file_index=None,
    standalone_matrix=None,
    standalone_freqs=None,
    standalone_phase=None,
    plot=True,
    channel_range=(30, 119),
    num_modes_to_check=10,
    entropy_limit=2.5,
    entropy_mode="U",
    boolean=1,
    mode="jupiter",
    noise_percentile=10,
    vmin=0,
    vmax=25,
):
    """
    Computes SVD on a sub-band, calculates empirical SNR (dB) for both the
    raw and SVD-reconstructed matrices, and (optionally) plots them for
    direct comparison. This is the function behind the paper's headline
    SNR-improvement statistic (Section 4.1): call with entropy_limit=3 to
    match the published S_lim.
    """
    if standalone_matrix is not None:
        full_matrix = np.asarray(standalone_matrix)
        freqs_full = np.asarray(standalone_freqs) if standalone_freqs is not None else np.arange(full_matrix.shape[0])
        phase = np.asarray(standalone_phase) if standalone_phase is not None else np.arange(full_matrix.shape[1])
        print(f"SNR comparison for standalone matrix | channels {channel_range[0]}-{channel_range[1]}")
    else:
        if datasets is None or file_index is None:
            print("Error: provide either 'standalone_matrix' OR both 'datasets' and 'file_index'.")
            return None, None

        data_key = "all_data_total" if mode == "all" else "all_data_jupiter"
        if data_key not in datasets:
            print(f"Error: key '{data_key}' not found.")
            return None, None

        df = datasets[data_key]
        full_matrix = df["plot"].iloc[file_index]
        freqs_full = datasets["freqs"].to_numpy().flatten()
        phase = np.asarray(datasets["phase"])
        print(f"SNR comparison for index {file_index} (mode: {mode}) | channels {channel_range[0]}-{channel_range[1]}")

    start_ch, end_ch = channel_range
    if start_ch < 0 or end_ch >= full_matrix.shape[0] or start_ch > end_ch:
        print(f"Error: invalid channel range {channel_range}. Must be within (0, {full_matrix.shape[0]-1}).")
        return None, None

    band_matrix = full_matrix[start_ch : end_ch + 1, :]
    freqs = freqs_full[start_ch : end_ch + 1]

    U, S, Vt = np.linalg.svd(band_matrix, full_matrices=False)

    if entropy_mode == "U":
        entropies = [_shannon_entropy(_normalize_vector(U[:, i])) for i in range(U.shape[1])]
    else:
        entropies = [_shannon_entropy(_normalize_vector(Vt[i, :])) for i in range(Vt.shape[0])]

    kept_modes = [i for i in range(num_modes_to_check) if boolean * entropies[i] > boolean * entropy_limit]

    S_clean = np.zeros_like(S)
    if kept_modes:
        S_clean[kept_modes] = S[kept_modes]
    svd_reconstructed = (U @ np.diag(S_clean)) @ Vt

    def compute_snr_db(matrix, _noise_percentile=noise_percentile):
        power = matrix ** 2
        baseline = np.percentile(power, _noise_percentile, axis=1, keepdims=True)
        baseline = np.where(baseline == 0, 1e-12, baseline)
        snr_linear = power / baseline
        return 10 * np.log10(snr_linear + 1e-12)

    raw_snr_db = compute_snr_db(band_matrix)
    svd_snr_db = compute_snr_db(svd_reconstructed)

    print(f"   -> RAW Peak SNR: {np.max(raw_snr_db):.2f} dB")
    print(f"   -> SVD Peak SNR: {np.max(svd_snr_db):.2f} dB")

    raw_active = raw_snr_db[raw_snr_db > 3]
    svd_active = svd_snr_db[svd_snr_db > 3]

    if len(raw_active) > 0 and len(svd_active) > 0:
        print(f"   -> RAW Avg Active SNR (>3dB): {np.mean(raw_active):.2f} dB")
        print(f"   -> SVD Avg Active SNR (>3dB): {np.mean(svd_active):.2f} dB")
        print(f"   -> SNR improvement: +{(np.mean(svd_active) - np.mean(raw_active)):.2f} dB")

    if plot:
        fig, axs = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
        titles = ["1. Raw Spectrogram SNR (Before SVD)", "2. SVD Reconstructed SNR (Noise Filtered)"]
        matrices = [raw_snr_db, svd_snr_db]

        for i in range(2):
            im = axs[i].pcolormesh(phase, freqs, matrices[i], shading="auto", cmap="magma", vmin=vmin, vmax=vmax)
            axs[i].set_yscale("log")
            axs[i].set_title(titles[i], fontsize=14, fontweight="bold")
            axs[i].set_ylabel("Frequency (Hz)", fontsize=12)
            cb = fig.colorbar(im, ax=axs[i], pad=0.015)
            cb.set_label("SNR (dB)", fontsize=11, fontweight="bold")
            axs[i].contour(phase, freqs, matrices[i], levels=[10], colors="white", alpha=0.5, linewidths=0.8, linestyles="dashed")

        axs[1].set_xlabel("Jovian Longitude (Phase Degrees)", fontsize=12)
        plt.tight_layout()
        plt.show()

    return raw_snr_db, svd_snr_db
