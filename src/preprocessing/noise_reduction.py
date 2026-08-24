"""
Reshaping and baseline noise-continuum flattening (Section 3.1).

NOTE (flagged during repo cleanup, needs author confirmation):
`clean_spectrogram_noise` is documented below as a two-stage filter, but
only Stage 1 (percentile scaling) is active -- Stage 2 (row-mean
subtraction on channels >= 110) is commented out in the original notebook.
The paper's abstract describes "a deterministic, dual-stage noise
filtering process." Before publishing this repo, confirm whether:
  (a) the manuscript's "dual-stage" language refers to this function
      (in which case Stage 2 needs to be restored, or the docstring/paper
      text corrected to reflect single-stage percentile flattening), or
  (b) it refers to the separate burst-detection + median-inpainting step
      described in Section 3.1 / Figure 2, which is not implemented in
      this notebook at all and likely lives in the upstream phase-folding
      script (see data/README.md) -- in which case this docstring should
      just be corrected to not claim "dual-stage" itself.
Stage 2 is left in place (commented) rather than deleted so the original
intent isn't lost.
"""
import numpy as np


def format_all_data_total(all_data_total, source_col="unrolled_plot", target_col="plot"):
    """
    Reshapes 1D unrolled plots into 2D numpy arrays of shape (128, 721),
    stored in a new column named 'plot' to match the Jupiter dataset formatting.
    """
    if source_col not in all_data_total.columns:
        print(f"Error: Column '{source_col}' not found in the DataFrame.")
        return all_data_total

    print(f"Reshaping 1D arrays from '{source_col}' into 2D arrays of shape (128, 721)...")

    all_data_total[target_col] = all_data_total[source_col].apply(
        lambda raw_array: np.reshape(np.array(raw_array), (128, 721))
    )

    print(f"Successfully reshaped! Data is now in the '{target_col}' column.")

    return all_data_total


def filter_invalid_plots(df, column_name="unrolled_plot", expected_length=92288):
    """Filters out rows where the 1D plot array doesn't match the expected length."""
    if column_name not in df.columns:
        print(f"Error: Column '{column_name}' not found in the DataFrame.")
        return df

    print(f"Filtering dataset to keep only plots of length {expected_length}...")

    initial_count = len(df)
    filtered_df = df[df[column_name].apply(len) == expected_length].copy()
    final_count = len(filtered_df)

    print(f"Ignored {initial_count - final_count} invalid plots. Remaining valid plots: {final_count}")

    filtered_df.reset_index(drop=True, inplace=True)
    return filtered_df


def clean_spectrogram_noise(matrix):
    """
    Applies baseline continuum flattening to a single 2D spectrogram matrix
    (Section 3.1: 75th-percentile frequency-dependent multiplier).

    See module-level NOTE above regarding "dual-stage" terminology.

    Args:
        matrix (np.ndarray): A 2D array of shape (128, 721)

    Returns:
        np.ndarray: The filtered and cleaned 2D matrix.
    """
    cleaned_matrix = matrix.copy()

    # --- Stage 1: Percentile Filter (Plasma Noise Removal) ---
    for i in range(cleaned_matrix.shape[0]):
        row = cleaned_matrix[i, :]
        median = np.median(row)

        positive_half = row[row >= median]

        q3 = np.percentile(positive_half, 75)

        # 1e-9 epsilon prevents division by zero
        scale_factor = abs(q3 - median) + 1e-9
        cleaned_matrix[i, :] = row / scale_factor

    # --- Stage 2 (disabled in original notebook -- see module NOTE) ---
    # high_freq_rows = cleaned_matrix[110:, :]
    # mean_per_row = np.mean(high_freq_rows, axis=1, keepdims=True)
    # cleaned_matrix[110:, :] = high_freq_rows - mean_per_row

    return cleaned_matrix


def apply_pipeline_filters(datasets, target_key="all_data_total", column_name="plot"):
    """Pipeline wrapper to apply the noise filter to the chosen dataset."""
    if datasets is None or target_key not in datasets:
        print(f"Error: Dataset '{target_key}' not found.")
        return datasets

    print(f"Applying noise filtering to '{target_key}'...")

    datasets[target_key][column_name] = datasets[target_key][column_name].apply(clean_spectrogram_noise)

    print("Noise filtering completed successfully!")
    return datasets
