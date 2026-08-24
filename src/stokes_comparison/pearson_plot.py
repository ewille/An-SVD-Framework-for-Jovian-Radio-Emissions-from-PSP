"""2D density-scatter figure for the SVD-vs-Stokes-V polarimetric comparison (Section 4.2)."""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.colors import LogNorm


def plot_stokes_v_pearson(svd_matrix, stokes_v_reference, emission_mask=None, n_bins=200, figsize=(7, 6)):
    """
    2D density scatter of SVD-reconstructed C^i_XY amplitude versus the
    independently SVD-reconstructed Stokes V amplitude at matched pixels,
    with Pearson r and a linear fit annotated. Uses a 2D histogram
    (log-scaled density) rather than individual points given the pixel
    count (N ~ 65,611 in the paper).

    Both inputs are expected to have already been through the SVD
    pipeline independently (Section 4.2: "independently processed through
    the SVD pipeline to suppress stochastic noise prior to comparison").

    Returns (r, p, fig).
    """
    if emission_mask is not None:
        x = svd_matrix[emission_mask].flatten()
        y = stokes_v_reference[emission_mask].flatten()
    else:
        x = svd_matrix.flatten()
        y = stokes_v_reference.flatten()

    finite_mask = np.isfinite(x) & np.isfinite(y)
    x = x[finite_mask]
    y = y[finite_mask]

    r, p = stats.pearsonr(x, y)

    slope, intercept, _, _, _ = stats.linregress(x, y)
    x_fit = np.linspace(x.min(), x.max(), 500)
    y_fit = slope * x_fit + intercept

    fig, ax = plt.subplots(figsize=figsize)

    h, xedges, yedges = np.histogram2d(x, y, bins=n_bins)

    h_masked = np.ma.masked_where(h == 0, h)

    im = ax.pcolormesh(xedges, yedges, h_masked.T, norm=LogNorm(vmin=1, vmax=h.max()), cmap="plasma", shading="auto")

    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("Pixel Count (log scale)", fontsize=12)
    cb.ax.tick_params(labelsize=10)

    ax.plot(x_fit, y_fit, color="black", linewidth=1.5, linestyle="--", label="Linear fit", zorder=5)

    if p == 0.0:
        p_str = r"$p \ll 10^{-10}$"
    else:
        p_str = f"$p = {p:.2e}$"

    annotation = f"$r = {r:.3f}$\n{p_str}\n$N = {len(x):,}$ pixels"

    ax.text(0.05, 0.95, annotation, transform=ax.transAxes, fontsize=16, verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="gray"))

    ax.set_xlabel(r"SVD-Reconstructed $C^i_{XY}$ Amplitude", fontsize=16)
    ax.set_ylabel(r"SVD-Reconstructed Stokes V Amplitude", fontsize=16)
    ax.set_title(r"SVD $C^i_{XY}$ vs. Independent Stokes $V$", fontsize=20)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-0.5, 0.5)
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=16, loc="lower right")

    plt.tight_layout()
    plt.show()

    print(f"Pearson r:       {r:.4f}")
    print(f"p-value:         {p:.2e}" if p > 0 else "p-value:         << 10^-300 (machine zero)")
    print(f"N pixels used:   {len(x):,}")
    print(f"Linear fit:      slope = {slope:.4f}, intercept = {intercept:.4f}")

    return r, p, fig
