"""
Interactive Jupyter-widget tool used to build the manually-curated
Jovian-positive training set described in Section 3.1 ("Selecting and
categorizing Jovian data was initially done manually...").

This is a labeling GUI, not part of the automated pipeline -- it's kept
separate from src/eigenfaces/eigenfaces_core.py because it requires extra
interactive-only dependencies (ipywidgets, and a Jupyter frontend that
supports the widget matplotlib backend) that the rest of the pipeline
does not need. Only use this inside a Jupyter notebook, not as an import
in a script.

Requires: ipywidgets (not listed in the paper's Software section, since
it's a development tool rather than something that shaped the analysis
output -- add it to environment.yml only if you intend to re-run/extend
the labeling step).
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
import ipywidgets as widgets
from IPython.display import display, clear_output


class JovianAnnotator:
    """
    Paginated, click-and-drag annotation tool: browse spectrograms one at a
    time and highlight (via horizontal span-select) the phase ranges
    containing Jovian emission. Annotations persist to `save_file` (JSON)
    after every action, keyed by dataframe row index.
    """

    def __init__(self, datasets, save_file=None):
        from ..config import PATHS

        self.df = datasets["all_data_total"]
        self.phase = datasets["phase"]
        self.freqs = datasets["freqs"]
        self.save_file = str(save_file) if save_file is not None else str(PATHS["annotations"])

        self.annotations = {}
        if os.path.exists(self.save_file):
            with open(self.save_file, "r") as f:
                loaded = json.load(f)
                self.annotations = {int(k): v for k, v in loaded.items()}

        self.current_idx = max([int(k) for k in self.annotations.keys()] + [0])
        if self.current_idx >= len(self.df):
            self.current_idx = len(self.df) - 1

        self.max_idx = len(self.df) - 1

        self.btn_prev = widgets.Button(description="Previous", button_style="info")
        self.btn_next = widgets.Button(description="Next", button_style="success")
        self.btn_clear = widgets.Button(description="Clear Selection", button_style="warning")

        self.out_log = widgets.Output()
        self.plot_output = widgets.Output()

        self.btn_prev.on_click(self.prev_plot)
        self.btn_next.on_click(self.next_plot)
        self.btn_clear.on_click(self.clear_selection)

        with self.plot_output:
            self.fig, self.ax = plt.subplots(figsize=(15, 5))
            self.cbar = None
            self.span = SpanSelector(
                self.ax, self.on_select, "horizontal", useblit=True, props=dict(alpha=0.3, facecolor="red")
            )
            plt.show()

        ui = widgets.VBox([self.plot_output, widgets.HBox([self.btn_prev, self.btn_next, self.btn_clear]), self.out_log])

        display(ui)
        self.draw_plot()

    def draw_plot(self):
        self.ax.clear()
        matrix = self.df["plot"].iloc[self.current_idx]
        colormesh_data = matrix

        vmax = 1
        vmin = -1

        im0 = self.ax.pcolormesh(self.phase, self.freqs, colormesh_data, shading="auto", cmap="viridis", vmax=vmax, vmin=vmin)
        self.ax.set_yscale("log")

        if self.cbar is None:
            self.cbar = self.fig.colorbar(im0, ax=self.ax)
            self.cbar.ax.tick_params(labelsize=12)
        else:
            self.cbar.update_normal(im0)

        self.ax.set_title(f"Index: {self.current_idx} / {self.max_idx} | Highlight Jovian emissions", fontsize=16, fontweight="bold")
        self.ax.set_ylabel("Frequency (Hz)", fontsize=14)
        self.ax.set_xlabel("Jovian Longitude", fontsize=14)
        self.ax.tick_params(axis="both", which="major", labelsize=12)

        if self.current_idx in self.annotations:
            for (xmin, xmax) in self.annotations[self.current_idx]:
                self.ax.axvspan(xmin, xmax, color="red", alpha=0.3)

        self.fig.canvas.draw_idle()

    def on_select(self, xmin, xmax):
        if self.current_idx not in self.annotations:
            self.annotations[self.current_idx] = []

        self.annotations[self.current_idx].append((xmin, xmax))
        self.ax.axvspan(xmin, xmax, color="red", alpha=0.3)
        self.fig.canvas.draw_idle()
        self.save_annotations()

    def clear_selection(self, b):
        if self.current_idx in self.annotations:
            self.annotations[self.current_idx] = []
        self.save_annotations()
        self.draw_plot()

    def next_plot(self, b):
        if self.current_idx not in self.annotations:
            self.annotations[self.current_idx] = []
        self.save_annotations()

        if self.current_idx < self.max_idx:
            self.current_idx += 1
            self.draw_plot()

    def prev_plot(self, b):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.draw_plot()

    def save_annotations(self):
        with open(self.save_file, "w") as f:
            json.dump(self.annotations, f)

        with self.out_log:
            clear_output(wait=True)
            print(f"Saved progress for index {self.current_idx} to {self.save_file}")
