"""General-purpose phase/frequency spectrogram plotting utilities."""
import numpy as np
import matplotlib.pyplot as plt


def plot_phase_spectrogram(x, y, colormesh, xlabel, ylabel, title, vmax=None, vmin=None, yscale="log"):
    """Streamlined plotting function to generate Phase/Frequency colormeshes."""
    fig, axs = plt.subplots(1, figsize=(20, 5))

    im0 = axs.pcolormesh(x, y, colormesh, vmax=vmax, vmin=vmin, shading="auto")

    axs.set_yscale(yscale)
    cb = fig.colorbar(im0, ax=axs)
    cb.ax.tick_params(labelsize=12)

    axs.set_xlabel(xlabel, fontsize=14)
    axs.set_ylabel(ylabel, fontsize=14)
    axs.set_title(title, fontsize=16)
    axs.tick_params(axis="both", which="major", labelsize=12)

    plt.tight_layout()
    plt.show()


def plot_file_index(datasets, file_index, mode="all"):
    """
    Plots the spectrogram for a specific, known file index.

    Non-interactive counterpart to the original notebook's
    `plot_user_selection`, which prompted for input() -- not suitable for
    a script/CI context. Use this for programmatic/reproducible use;
    see notebooks/ for an interactive browsing version if needed.
    """
    if datasets is None:
        print("Datasets are not loaded. Please run load_jupiter_data() first.")
        return

    data_key = "all_data_total" if mode == "all" else "all_data_jupiter"
    if data_key not in datasets:
        print(f"Error: '{data_key}' not found in the datasets dictionary.")
        return

    df = datasets[data_key]
    max_index = len(df) - 1
    if not (0 <= file_index <= max_index):
        print(f"Invalid index! Must be between 0 and {max_index}.")
        return

    freqs = datasets["freqs"]
    phase = datasets["phase"]

    raw_data = df["plot"].iloc[file_index]
    colormesh_data = np.reshape(raw_data, (128, 721))

    title = f"Jovian Emission Spectrogram ({mode.capitalize()} Mode - Index: {file_index})"

    plot_phase_spectrogram(
        x=phase,
        y=freqs.iloc[:128],
        colormesh=colormesh_data,
        xlabel="Jovian Longitude",
        ylabel="Frequency (Hz)",
        vmax=np.percentile(colormesh_data, 95),
        vmin=np.percentile(colormesh_data, 5),
        title=title,
    )
    print("start time: ", df["start_time"].iloc[file_index], "\n end time: ", df["end_time"].iloc[file_index])


def plot_spectral_smoothness(
    datasets,
    file_index,
    start_phase=0,
    end_phase=360,
    phase_step=2,
    offset_step=0.5,
    mode="all",
    max_lines=100,
    channel_range=None,
):
    """
    Generates a stacked Power Spectrum line plot for a single data file.
    Converts raw voltage data to power (squared values) and slices lines
    by exact degree intervals. Allows isolating specific frequency bands
    via index channel ranges.

    Args:
        datasets (dict): Dictionary containing the loaded datasets.
        file_index (int): The row index of the file to plot.
        start_phase (float): Minimum phase angle to include (0 to 360).
        end_phase (float): Maximum phase angle to include (0 to 360).
        phase_step (float): Grating interval in degrees between lines (typically 1 or 2).
        offset_step (float): Vertical power offset added to each subsequent line baseline.
        mode (str): 'all' for all_data_total or 'jupiter' for all_data_jupiter.
        max_lines (int): Absolute cap on the number of stacked lines to plot (default 100).
        channel_range (tuple): Optional tuple of (first_channel, last_channel) indices to isolate.
    """
    if datasets is None:
        print("Datasets are not loaded. Please load data first.")
        return

    data_key = "all_data_total" if mode == "all" else "all_data_jupiter"
    if data_key not in datasets:
        print(f"Error: Key '{data_key}' not found in datasets.")
        return

    df = datasets[data_key]
    if not (0 <= file_index < len(df)):
        print(f"Error: Invalid file index {file_index}. Must be between 0 and {len(df)-1}.")
        return

    freqs = datasets["freqs"].iloc[:128].to_numpy().flatten()
    phase = datasets["phase"]

    grating_degrees = phase_step

    if channel_range is not None:
        first_ch, last_ch = channel_range
        if not (0 <= first_ch <= last_ch < len(freqs)):
            print(f"Error: Invalid channel range {channel_range}. Must be within (0, {len(freqs)-1}).")
            return
        freqs = freqs[first_ch : last_ch + 1]

    phase_resolution = phase[1] - phase[0]

    stride = int(round(grating_degrees / phase_resolution))
    if stride < 1:
        stride = 1

    matrix = df["plot"].iloc[file_index]

    if channel_range is not None:
        matrix = matrix[first_ch : last_ch + 1, :]

    phase_indices = np.where((phase >= start_phase) & (phase <= end_phase))[0]
    if len(phase_indices) == 0:
        print("Error: No data found within the specified phase range.")
        return

    selected_indices = phase_indices[::stride]

    if len(selected_indices) > max_lines:
        print(
            f"Note: Your settings generated {len(selected_indices)} lines. "
            f"Truncating to the first {max_lines} lines to preserve clarity."
        )
        selected_indices = selected_indices[:max_lines]

    fig, ax = plt.subplots(figsize=(14, 8))

    colors = plt.cm.plasma(np.linspace(0, 0.85, len(selected_indices)))

    print(
        f"Stacking {len(selected_indices)} power lines with a {grating_degrees} deg "
        f"grating and offset step of {offset_step}..."
    )

    for i, p_idx in enumerate(selected_indices):
        voltage_intensity = matrix[:, p_idx]
        power_intensity = voltage_intensity ** 2

        current_offset = i * offset_step
        stacked_power = power_intensity + current_offset

        current_phase_val = phase[p_idx]

        ax.plot(
            freqs,
            stacked_power,
            color=colors[i],
            alpha=0.85,
            linewidth=1.2,
            label=f"{current_phase_val:.1f} deg" if len(selected_indices) <= 20 else None,
        )

        ax.axhline(y=current_offset, color="gray", linestyle=":", alpha=0.15)

    ax.set_xscale("log")
    ax.set_xlim(freqs[0], freqs[-1])

    ax.set_ylim(0, (len(selected_indices) + 10) * offset_step)
    ax.set_xlabel("Frequency (Hz)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Power (Voltage^2) + Stacked Offset", fontsize=14, fontweight="bold")

    channel_info = f" | Channels: {channel_range}" if channel_range is not None else ""
    ax.set_title(
        f"Jovian Power Spectrum Waterfall - Smoothness Tracking\n"
        f"(Index: {file_index} | Grating: {grating_degrees} deg | {mode.capitalize()} Mode{channel_info})",
        fontsize=16,
        fontweight="bold",
    )
    ax.tick_params(axis="both", which="major", labelsize=12)

    if 0 < len(selected_indices) <= 20:
        ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.0), title="Jovian Phase")

    plt.tight_layout()
    plt.show()
