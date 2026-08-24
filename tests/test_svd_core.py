"""
Sanity checks for the SVD extraction core, using synthetic data (no PSP
archive access required). These don't validate the paper's reported
statistics -- they check that the entropy-threshold mechanism behaves as
documented, and they directly demonstrate the entropy-direction finding
flagged in src/svd_extraction/svd_core.py's module docstring, so it's
verifiable by running code rather than just reading a claim.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.svd_extraction.svd_core import run_svd_extraction, _normalize_vector, _shannon_entropy


def _make_narrowband_matrix(n_freq=90, n_phase=200, band_width=8, noise_scale=2.0, seed=0):
    """
    A signal concentrated in a narrow frequency sub-band (genuinely
    low-entropy in the U-column sense, like a real narrowband Jovian
    arc) riding on white noise spread across all channels (high entropy,
    close to the log(n_freq) ceiling). Returns (matrix, clean_signal) so
    tests can measure reconstruction error against ground truth rather
    than a noisy proxy metric.
    """
    rng = np.random.default_rng(seed)

    band_center = n_freq // 2
    freq_profile = np.zeros(n_freq)
    freq_profile[band_center - band_width // 2 : band_center + band_width // 2] = 1.0

    phase_profile = 1.0 + 0.5 * np.sin(np.linspace(0, 4 * np.pi, n_phase))

    clean_signal = 20.0 * np.outer(freq_profile, phase_profile)
    noise = rng.normal(scale=noise_scale, size=(n_freq, n_phase))
    return clean_signal + noise, clean_signal


def test_entropy_is_lower_for_the_structured_mode_than_noise_modes():
    """
    Confirms the entropy metric itself behaves as the paper describes:
    the mode carrying genuine narrowband structure should have
    substantially lower Shannon entropy than the noise-floor modes.
    """
    matrix, _ = _make_narrowband_matrix()
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)

    entropies = [_shannon_entropy(_normalize_vector(U[:, i])) for i in range(10)]

    structured_mode = int(np.argmax(S[:10]))  # dominant singular value = the true signal
    noise_modes = [i for i in range(10) if i != structured_mode]

    assert entropies[structured_mode] < min(entropies[i] for i in noise_modes), (
        "Expected the dominant (structured) mode to have the lowest entropy "
        "among the top 10 modes -- if this fails, the entropy metric itself "
        "isn't behaving as Section 3.2 describes for this synthetic case."
    )


def test_default_boolean_excludes_the_structured_mode():
    """
    *** Demonstrates the CRITICAL finding flagged in svd_core.py ***

    With the default boolean=1 (used at every call site in the original
    notebook), the kept-mode condition is `entropy > entropy_limit`. On
    a matrix with one genuinely low-entropy structured mode and several
    high-entropy noise modes, this should -- and does -- EXCLUDE the
    structured mode and INCLUDE noise modes near the entropy ceiling.
    boolean=-1 (matching the paper's stated "S_i < S_lim" criterion)
    does the opposite. If this test ever starts failing because the
    default direction changed, that's a deliberate fix to the finding
    below, not a regression -- update this test alongside it.
    """
    matrix, clean_signal = _make_narrowband_matrix()
    channel_range = (0, matrix.shape[0] - 1)

    _, residuals_default = run_svd_extraction(
        standalone_matrix=matrix, channel_range=channel_range,
        entropy_limit=3.0, num_modes_to_check=10, plot=False,
    )
    reconstructed_default = matrix + residuals_default  # residuals = reconstructed - raw, see svd_core.py

    _, residuals_paper_direction = run_svd_extraction(
        standalone_matrix=matrix, channel_range=channel_range,
        entropy_limit=3.0, num_modes_to_check=10, boolean=-1, plot=False,
    )
    reconstructed_paper_direction = matrix + residuals_paper_direction

    error_default = np.sum((reconstructed_default - clean_signal) ** 2)
    error_paper_direction = np.sum((reconstructed_paper_direction - clean_signal) ** 2)

    # The paper-consistent direction (keep low entropy) should reconstruct
    # much closer to the noise-free ground truth than the default
    # (keep high entropy) direction, on a case built specifically to have
    # one clean structured mode and several noise modes.
    assert error_paper_direction < error_default, (
        "Expected boolean=-1 (keep low entropy, per the paper's stated "
        "criterion) to reconstruct closer to the true noise-free signal "
        "than the current default boolean=1 (keep high entropy). If this "
        "assertion fails, re-examine the finding in svd_core.py's module "
        "docstring -- it may mean the default direction has already been "
        "corrected, or that this synthetic case needs revisiting."
    )


if __name__ == "__main__":
    test_entropy_is_lower_for_the_structured_mode_than_noise_modes()
    print("PASS: entropy is lower for the structured mode than noise modes.")
    test_default_boolean_excludes_the_structured_mode()
    print("PASS: (documents finding) default boolean=1 reconstructs worse than boolean=-1 on this case.")
    print("All tests passed.")
