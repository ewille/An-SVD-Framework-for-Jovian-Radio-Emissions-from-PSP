"""
Residual validation via 2D autocorrelation and Wiener-Khinchin power
spectrum (Section 4.1). Produces the arrays behind r_svd / r_res and the
paper's Figure 5.

Two bugs from the original notebook are fixed here (see inline notes):
1. `compare_autocorrelations` used `datasets['freqs']` / `datasets['phase']`
   as literal default argument values. Default argument values are
   evaluated once, at `def` time, against whatever the global `datasets`
   variable happened to be at that moment in the notebook -- not at call
   time. This only worked in the original notebook because `datasets` had
   already been loaded earlier in the session; it breaks immediately on
   import in a fresh module (no notebook global to grab) and would silently
   use a stale snapshot even in the notebook if `datasets` was reloaded or
   modified afterward. Fixed by making freqs/phase required parameters.
2. `plot_figure4_publication` (renamed `plot_autocorr_power_spectrum_figure`
   below, since it corresponds to the paper's published Figure 5, not
   Figure 4) called `LogLocator` without importing it. This is a genuine
   hidden-state bug: it only ran without error in the original notebook
   because `from matplotlib.ticker import LogLocator` had been executed
   interactively in a since-deleted cell, leaving it in the live kernel's
   namespace but not in the saved notebook source. A fresh "Restart & Run
   All" would raise NameError here. Fixed by adding the import.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator  # fix: was missing, see module docstring


def compute_2d_autocorrelation(matrix):
    """Normalized 2D autocorrelation via the FFT-based Wiener-Khinchin approach."""
    m = matrix - np.mean(matrix)

    F = np.fft.fft2(m, s=(2 * m.shape[0] - 1, 2 * m.shape[1] - 1))
    power_spectrum = np.abs(F) ** 2
    autocorr = np.fft.ifft2(power_spectrum).real

    autocorr = np.fft.fftshift(autocorr)
    autocorr /= autocorr.max()

    return autocorr


def compute_autocorr_power_spectrum(ac, taper=True):
    """
    2D power spectrum of an autocorrelation function via Wiener-Khinchin.

    The zero-lag bin is zeroed before transforming (it encodes total
    window variance, not structured correlation, and would otherwise
    dominate as a flat broadband offset). A 2D Hann taper (default on)
    suppresses the spectral-leakage cross this finite window otherwise
    produces along the axes, without measurably attenuating genuine
    off-axis periodic power (verified on synthetic signals: on-axis
    leakage drops ~24x after tapering; injected off-axis signal preserved
    at ~100% amplitude).
    """
    ac_zeroed = ac.copy()
    cy, cx = np.array(ac_zeroed.shape) // 2
    ac_zeroed[cy, cx] = 0.0

    if taper:
        wy = np.hanning(ac_zeroed.shape[0])
        wx = np.hanning(ac_zeroed.shape[1])
        ac_zeroed = ac_zeroed * np.outer(wy, wx)

    F = np.fft.fft2(np.fft.ifftshift(ac_zeroed))
    power = np.abs(F) ** 2
    power = np.fft.fftshift(power)
    power = power / power.max()

    return power


def compare_autocorrelations(svd_matrix, residual_matrix, freqs, phase, ps_crop=40, taper=True, plot=True):
    """
    Computes the 2D autocorrelations of the SVD reconstruction and residual
    matrix, their power spectra, and the scalar lag-1 autocorrelation
    summary (mean |autocorrelation| in the 3x3 neighborhood around zero
    lag, excluding the peak itself) reported as r_svd / r_res in Section 4.1.

    `freqs` and `phase` are now required (see module docstring, fix #1) --
    pass the sub-band frequency axis and phase axis you used for the SVD
    call, e.g. `datasets['freqs'].iloc[30:120]` and `datasets['phase']`.
    """
    ac_svd = compute_2d_autocorrelation(svd_matrix)
    ac_res = compute_2d_autocorrelation(residual_matrix)

    ps_svd = compute_autocorr_power_spectrum(ac_svd, taper=taper)
    ps_res = compute_autocorr_power_spectrum(ac_res, taper=taper)

    def lag1_mean(ac):
        cy, cx = np.array(ac.shape) // 2
        neighborhood = ac[cy - 1 : cy + 2, cx - 1 : cx + 2].copy()
        neighborhood[1, 1] = np.nan
        return np.nanmean(np.abs(neighborhood))

    svd_lag1 = lag1_mean(ac_svd)
    res_lag1 = lag1_mean(ac_res)

    print(f"SVD reconstruction lag-1 autocorrelation:  {svd_lag1:.4f}")
    print(f"Residual matrix lag-1 autocorrelation:     {res_lag1:.4f}")
    print(f"Ratio (SVD/Residual):                      {svd_lag1/res_lag1:.2f}x")

    if plot:
        cy, cx = np.array(ac_res.shape) // 2
        w = 20

        fig, axs = plt.subplots(2, 2, figsize=(14, 12))

        ac_titles = ["SVD Reconstruction Autocorrelation", "Residual Matrix Autocorrelation"]
        acs = [ac_svd, ac_res]

        for i, (ac, title) in enumerate(zip(acs, ac_titles)):
            ax = axs[i, 0]
            im = ax.pcolormesh(ac[cy - w : cy + w + 1, cx - w : cx + w + 1], cmap="RdBu_r", vmin=-0.3, vmax=1.0)
            ticks = np.arange(0, 2 * w + 1, 10)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xticklabels(ticks - w)
            ax.set_yticklabels(ticks - w)
            ax.set_title(title, fontsize=18)
            ax.set_xlabel("Phase Lag (bins)", fontsize=14)
            ax.set_ylabel("Frequency Lag (bins)", fontsize=14)
            plt.colorbar(im, ax=ax).set_label("Normalized Autocorrelation", fontsize=14)

        ps_titles = ["SVD Reconstruction Power Spectrum", "Residual Matrix Power Spectrum"]
        pss = [ps_svd, ps_res]
        py, px = np.array(ps_res.shape) // 2
        pw = min(py, px, ps_crop)

        for i, (ps, title) in enumerate(zip(pss, ps_titles)):
            ax = axs[i, 1]
            crop = ps[py - pw : py + pw + 1, px - pw : px + pw + 1]
            floor = crop[crop > 0].min() if np.any(crop > 0) else 1e-12
            im = ax.pcolormesh(crop, cmap="viridis", norm=LogNorm(vmin=max(floor, crop.max() * 1e-6), vmax=crop.max()))
            pticks = np.arange(0, 2 * pw + 1, 10)
            ax.set_xticks(pticks)
            ax.set_yticks(pticks)
            ax.set_xticklabels(pticks - pw)
            ax.set_yticklabels(pticks - pw)
            ax.set_title(title, fontsize=18)
            ax.set_xlabel("Phase-Lag Wavenumber (cycles/window)", fontsize=14)
            ax.set_ylabel("Freq-Lag Wavenumber (cycles/window)", fontsize=14)
            plt.colorbar(im, ax=ax).set_label("Normalized Power", fontsize=14)

        plt.tight_layout()
        plt.show()

    return ac_svd, ac_res, ps_svd, ps_res, svd_lag1, res_lag1


def plot_autocorr_power_spectrum_figure(ac_svd, ac_res, ps_svd, ps_res, fig_width_in=3.5, fig_height_in=4.9, save_path=None):
    """
    Renders the single-column, publication-ready version of the paper's
    Figure 5: autocorrelation (left) and power spectrum (right) for the
    SVD reconstruction (top) and residual (bottom), as a 2x2 panel.

    Originally `plot_figure4_publication` -- renamed to match the
    published figure number (see module docstring). All text defaults to
    >=10pt to match/exceed AASTeX7 two-column body text; verify against
    \\the\\baselineskip in the compiled PDF and adjust BODY_FS if your
    document uses a different base size. fig_width_in should match the
    journal's single-column width (AASTeX7 twocolumn is ~3.5in; check
    \\the\\columnwidth in the compiled document).
    """
    BODY_FS = 10
    LABEL_FS = BODY_FS
    TICK_FS = BODY_FS
    INPANEL_FS = BODY_FS
    CBAR_FS = BODY_FS

    fig, axs = plt.subplots(2, 2, figsize=(fig_width_in, fig_height_in), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.02, hspace=0.02, wspace=0.08)

    w = 20
    cy, cx = np.array(ac_res.shape) // 2
    ac_ticks = [-20, 0, 20]
    inpanel_labels = ["SVD", "Residual"]

    im_ac = None
    for row, (ac, lbl) in enumerate(zip([ac_svd, ac_res], inpanel_labels)):
        ax = axs[row, 0]
        im_ac = ax.pcolormesh(ac[cy - w : cy + w + 1, cx - w : cx + w + 1], cmap="RdBu_r", vmin=-0.3, vmax=1.0, rasterized=True)
        ax.set_xticks([t + w for t in ac_ticks])
        ax.set_xticklabels(ac_ticks if row == 1 else [], fontsize=TICK_FS)
        ax.set_yticks([t + w for t in ac_ticks])
        ax.set_yticklabels(ac_ticks, fontsize=TICK_FS)
        ax.set_ylabel("Freq. Lag (bins)", fontsize=LABEL_FS)
        if row == 1:
            ax.set_xlabel("Phase Lag (bins)", fontsize=LABEL_FS)
        ax.text(0.05, 0.93, lbl, transform=ax.transAxes, fontsize=INPANEL_FS, va="top", ha="left",
                fontweight="bold", color="black", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))

    pw = 40
    py, px = np.array(ps_res.shape) // 2
    ps_ticks = [-40, 0, 40]
    PS_VMIN, PS_VMAX = 1e-6, 1.0

    im_ps = None
    for row, ps in enumerate([ps_svd, ps_res]):
        ax = axs[row, 1]
        crop = ps[py - pw : py + pw + 1, px - pw : px + pw + 1]
        im_ps = ax.pcolormesh(crop, cmap="viridis", norm=LogNorm(vmin=PS_VMIN, vmax=PS_VMAX), rasterized=True)
        ax.set_xticks([t + pw for t in ps_ticks])
        ax.set_xticklabels(ps_ticks if row == 1 else [], fontsize=TICK_FS)
        ax.set_yticks([t + pw for t in ps_ticks])
        ax.set_yticklabels([])
        if row == 1:
            ax.set_xlabel("Wavenumber\n(cyc./win.)", fontsize=LABEL_FS)

    cbar_ac = fig.colorbar(im_ac, ax=axs[:, 0], location="top", shrink=0.88, aspect=22, pad=0.03)
    cbar_ac.set_label("Norm. Autocorrelation", fontsize=CBAR_FS)
    cbar_ac.ax.tick_params(labelsize=TICK_FS)
    cbar_ac.set_ticks([-0.2, 0.3, 0.8])

    cbar_ps = fig.colorbar(im_ps, ax=axs[:, 1], location="top", shrink=0.88, aspect=22, pad=0.03)
    cbar_ps.set_label("Norm. Power", fontsize=CBAR_FS)
    cbar_ps.ax.tick_params(labelsize=TICK_FS)
    cbar_ps.ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=4))
    cbar_ps.set_ticks([1e-6, 1e-4, 1e-2, 1e0])

    if save_path is not None:
        fig.savefig(save_path, dpi=300)
        fig.savefig(save_path.rsplit(".", 1)[0] + ".pdf")

    plt.show()
