"""
Occurrence probability distributions (Section 4.3 / paper Figure 4).

*** FLAG -- NOTE 1 BELOW STILL NEEDS AUTHOR REVIEW BEFORE PUBLIC RELEASE ***

These functions consume manually-produced annotation JSON files (built
with src/eigenfaces/annotator.py: JovianAnnotator), not the raw spectrogram
arrays directly. That means the annotation files themselves are a required
data product for reproducing Figure 4 and must ship with the repo (or be
regenerated) -- they are not derivable from code alone.

NOTE 1 -- unexplained rescaling constants (STILL OPEN):
`plot_jovian_occurrence_matrix`'s grid_config hardcodes two "magic"
multipliers: 1/2.7113423305672644 (top-right panel) and
1/1.8558867143806506 (bottom-right panel), and the exact same two literals
are passed independently at the `plot_jovian_occurrence` call sites in the
original notebook. Their consistency across two separate call paths
suggests they're a deliberate, real calibration -- most likely correcting
for the "jovian_annotations_*.json" and "all_annotations_*.json" files
having different total annotation counts (the function normalizes by
`total_annotated`, so if the two annotation files span different numbers
of labeled plots, a raw ratio between them wouldn't be apples-to-apples
without this correction). But as written, the numbers are opaque literals
with no derivation shown anywhere -- there's no line computing them from
the actual annotation-file sizes, just the hardcoded decimals. Before this
goes in a public repo: please confirm what these represent and, ideally,
replace them with a runtime calculation (e.g.
`len(all_annotations) / len(jovian_annotations)`) so the relationship is
visible and can't silently drift out of sync with the underlying data.

NOTE 2 -- RESOLVED: `plot_jovian_occurrence_matrix` now takes two separate
`datasets` dicts (`datasets_io_phase`, `datasets_longitude`) instead of one
shared `datasets` + a `lon_key` that was never actually read. This matches
how the data really works (confirmed once the two phase-folded archives --
938 Phi_Io segments, 3711 lambda_III segments -- were available): Phi_Io
and lambda_III are two separately-folded archives (Section 3.1), not two
columns of one archive, so the left and right columns of Figure 4 need
two different `datasets` (see src/preprocessing/load_data.py's
`phase_frame` parameter). The old `lon_key` parameter is gone.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt


def plot_jovian_occurrence(datasets, annotation_file, multiplier=1, color="midnightblue", xlabel="Io Phase (Degrees)", title=None, plot=True):
    """
    Reads manual annotations (from JovianAnnotator) and computes the
    occurrence probability of Jovian emission as a function of phase.

    `datasets` should be loaded with the `phase_frame` matching whatever
    physical quantity `xlabel` describes (e.g. load with
    `phase_frame="phi_io"` for the default "Io Phase" label, or
    `phase_frame="lambda_iii"` and `xlabel="Jovian Longitude (Degrees)"`
    for the longitude case).

    `multiplier` defaults to 1; see module docstring Note 1 for the
    specific rescaling values used for the "jovian_annotations_*" files
    in the original analysis, which need verification before reuse.
    """
    if not os.path.exists(annotation_file):
        print(f"Error: annotation file not found: {annotation_file}. Run the Annotator first.")
        return None, None

    with open(annotation_file, "r") as f:
        annotations = json.load(f)

    phase = datasets["phase"]
    occurrence_counts = np.zeros_like(phase)

    total_annotated = len(annotations)
    if total_annotated == 0:
        print("No annotations found inside the file.")
        return None, None

    for idx_str, ranges in annotations.items():
        file_mask = np.zeros_like(phase, dtype=bool)
        for (xmin, xmax) in ranges:
            file_mask |= (phase >= xmin) & (phase <= xmax)
        occurrence_counts += file_mask.astype(int)

    occurrence_rate = (occurrence_counts / total_annotated) * 100 * multiplier

    if plot:
        plt.figure(figsize=(7, 5))
        plt.plot(phase, occurrence_rate, color="black", linewidth=1)
        plt.fill_between(phase, 0, occurrence_rate, color=color)
        plt.xlim(phase[0], phase[-1])
        plt.ylim(0, np.max(occurrence_rate) + 5 * multiplier)
        plt.title(title, fontsize=30)
        plt.xlabel(xlabel, fontsize=24)
        plt.ylabel("Occurrence Probability (%)", fontsize=24)
        plt.xticks(fontsize=21)
        plt.yticks(fontsize=21)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

    return occurrence_counts, occurrence_rate


def plot_jovian_occurrence_matrix(datasets_io_phase, datasets_longitude, annotation_dir=".", title="Jovian Radio Emission Occurrence Probability", plot=True):
    """
    Generates the 2x2 occurrence-probability grid (paper Figure 4):
    Left column Io phase / right column Jovian System III longitude,
    top row LHCP / bottom row RHCP.

    Args:
        datasets_io_phase: dict from `load_jupiter_data(phase_frame="phi_io")`.
            Used for the left column (938-segment Phi_Io archive).
        datasets_longitude: dict from `load_jupiter_data(phase_frame="lambda_iii")`
            (the default `phase_frame`). Used for the right column
            (3711-segment lambda_III archive).

    See module docstring Note 1 -- the rescaling multipliers still need
    author confirmation before this is treated as reproducing Figure 4
    exactly.

    Expects four annotation files in `annotation_dir`:
    all_annotations_negative.json, jovian_annotations_negative.json,
    all_annotations_positive.json, jovian_annotations_positive.json.
    """
    grid_config = {
        (0, 0): {
            "file": os.path.join(annotation_dir, "all_annotations_negative.json"),
            "datasets": datasets_io_phase,
            "x_label": None,
            "y_label": "LHCP Occurrence (%)",
            "multiplier": 1.0,
            "color": "midnightblue",
        },
        (0, 1): {
            "file": os.path.join(annotation_dir, "jovian_annotations_negative.json"),
            "datasets": datasets_longitude,
            "x_label": None,
            "y_label": "",
            "multiplier": 1 / 2.7113423305672644,  # NOTE 1: unexplained, needs verification
            "color": "midnightblue",
        },
        (1, 0): {
            "file": os.path.join(annotation_dir, "all_annotations_positive.json"),
            "datasets": datasets_io_phase,
            "x_label": "Io Phase (Degrees)",
            "y_label": "RHCP Occurrence (%)",
            "multiplier": 1.0,
            "color": "yellow",
        },
        (1, 1): {
            "file": os.path.join(annotation_dir, "jovian_annotations_positive.json"),
            "datasets": datasets_longitude,
            "x_label": "Jovian Longitude (Degrees)",
            "y_label": "",
            "multiplier": 1 / 1.8558867143806506,  # NOTE 1: unexplained, needs verification
            "color": "yellow",
        },
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex="col", sharey="row") if plot else (None, None)
    rates_output = {}

    for (row, col), config in grid_config.items():
        annotation_file = config["file"]

        if not os.path.exists(annotation_file):
            print(f"Warning: file not found: {annotation_file}. Skipping subplot ({row}, {col}).")
            rates_output[(row, col)] = None
            continue

        with open(annotation_file, "r") as f:
            annotations = json.load(f)

        panel_datasets = config["datasets"]
        if panel_datasets is None or panel_datasets.get("phase") is None:
            print(f"Warning: no 'phase' data available for subplot ({row}, {col}). Skipping.")
            rates_output[(row, col)] = None
            continue

        x_data = panel_datasets["phase"]
        occurrence_counts = np.zeros_like(x_data)
        total_annotated = len(annotations)

        if total_annotated == 0:
            print(f"Warning: no annotations in {annotation_file}.")
            rates_output[(row, col)] = np.zeros_like(x_data)
            continue

        for idx_str, ranges in annotations.items():
            file_mask = np.zeros_like(x_data, dtype=bool)
            for (xmin, xmax) in ranges:
                file_mask |= (x_data >= xmin) & (x_data <= xmax)
            occurrence_counts += file_mask.astype(int)

        occurrence_rate = (occurrence_counts / total_annotated) * 100 * config["multiplier"]
        rates_output[(row, col)] = occurrence_rate

        if plot:
            ax = axes[row, col]
            ax.plot(x_data, occurrence_rate, color="black", linewidth=1.5)
            ax.fill_between(x_data, 0, occurrence_rate, color=config["color"], alpha=0.85)
            ax.set_xlim(x_data[0], x_data[-1])
            ax.set_ylim(0, np.max(occurrence_rate) + 6 * config["multiplier"])
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.tick_params(axis="both", labelsize=16)
            if config["x_label"]:
                ax.set_xlabel(config["x_label"], fontsize=18)
            if config["y_label"]:
                ax.set_ylabel(config["y_label"], fontsize=18)

    if plot:
        if title:
            fig.suptitle(title, fontsize=30, y=0.98)
        plt.tight_layout()
        plt.show()

    return (
        rates_output[(0, 0)],  # occurrence_all_neg
        rates_output[(0, 1)],  # occurrence_jup_neg
        rates_output[(1, 0)],  # occurrence_all_pos
        rates_output[(1, 1)],  # occurrence_jup_pos
    )
