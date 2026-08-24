# psp-jovian-svd

Code accompanying **"A Singular Value Decomposition Framework for Jovian
Radio Emissions from Parker Solar Probe"** (Wille et al., in review at
AGU). This repository extracts Jovian decametric/hectometric radio
emissions from six years of PSP FIELDS/RFS data using entropy-thresholded
SVD, cross-validates against an independent Stokes V product and Juno-Waves,
and reproduces the paper's figures.

> **Status:** this repository was split out of the original analysis
> notebook and is undergoing cleanup before the public release tied to
> the AGU submission. See "Known issues" below for what's resolved and
> what still needs author confirmation.

## Pipeline overview

| Stage | Paper section | Module |
|---|---|---|
| Raw CDF archive -> monthly combined HDF5 | Sec. 2 | `src/preprocessing/build_monthly_archive.py` |
| Phase-folding (SPICE, light-travel-time, binning) | Sec. 2 / 3.1 | `src/preprocessing/make_folded_data.py` |
| Data loading & channel sorting | Sec. 2 | `src/preprocessing/load_data.py` |
| Noise-continuum flattening | Sec. 3.1 | `src/preprocessing/noise_reduction.py` |
| Manual annotation (GUI tool) | Sec. 3.1 | `src/eigenfaces/annotator.py` |
| SVD extraction & SNR | Sec. 3.2 / 4.1 | `src/svd_extraction/svd_core.py` |
| Autocorrelation / power spectrum validation | Sec. 4.1 | `src/svd_extraction/validation.py` |
| Eigenfaces detection metric | Sec. 3.3 / 4.4 | `src/eigenfaces/eigenfaces_core.py` |
| Occurrence probability distributions | Sec. 4.3 | `src/occurrence/occurrence_plots.py` |
| Stokes V cross-validation | Sec. 4.2 | `src/stokes_comparison/` |
| General spectrogram plotting | — | `src/plotting/spectrogram_plots.py` |

`make_folded_data.py` is run twice (once per `phase_frame`) against the
same monthly input to produce both final archives: `data_theta_psp.h5`
(folded to PSP's Jovian System III longitude, lambda_III) and
`data_phi_io.h5` (folded to Io phase, Phi_Io).

**Still missing:** the script that turns `data_theta_psp.h5` plus the
manual annotations into the morphology-labeled `all_jupiter_data.h5` /
`all_jupiter_data_unrolled.h5` files that `load_jupiter_data()` expects.
See `data/README.md` for detail.

## Installation

```bash
conda env create -f environment.yml
conda activate psp-jovian-svd
```

or

```bash
pip install -r requirements.txt
```

## Data

See [`data/README.md`](data/README.md) for what this pipeline expects as
input, where the public raw archives live, and how the intermediate
products are derived. Set `PSP_DATA_DIR` to point at your data directory:

```bash
export PSP_DATA_DIR=/path/to/your/data
```

## Usage

```python
from src.preprocessing.load_data import load_jupiter_data, sort_frequency_channels
from src.svd_extraction.svd_core import run_svd_extraction, compare_svd_snr

datasets = load_jupiter_data()  # phase_frame="lambda_iii" by default
datasets = sort_frequency_channels(datasets)

svd_signal, residuals = run_svd_extraction(
    datasets=datasets, file_index=245, channel_range=(30, 120),
    entropy_limit=3, mode="all",
)
```

See `notebooks/` for a worked end-to-end example and `figures/` for
scripts that reproduce each numbered figure in the paper.

## Software

NumPy, SciPy, pandas, Matplotlib, Astropy, SpiceyPy, SpacePy, h5py, joblib
(see manuscript Software section for citations). `ipywidgets` is required
only for the interactive annotation tool.

## License

MIT — see `LICENSE`.

## Citation

See `CITATION.cff` (to be finalized with the Zenodo archive DOI and paper
DOI once available).
