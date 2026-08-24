# Data

This directory is where the pipeline expects its processed data products
to live (or point `PSP_DATA_DIR` elsewhere -- see `src/config.py`). Nothing
in this directory is committed to the repository; the raw PSP/Juno archive
data are already public elsewhere, and the intermediate products are large.

## Provenance chain

1. **Raw data (public, external):**
   - PSP FIELDS/RFS HFR/LFR CDF archive: <https://fields.ssl.berkeley.edu/data/>
   - Juno-Waves survey data (Section 5 comparison): NASA PDS Planetary
     Plasma Interactions Node, Juno WAVES full-resolution survey catalog
   - SPICE kernels: NAIF node of the Planetary Data System

2. **Raw CDF archive -> monthly combined HDF5
   (`src/preprocessing/build_monthly_archive.py`):** loads the raw HFR/LFR
   CDF files, computes the antenna/Sun-direction geometry diagnostics,
   bins everything to a uniform 15s cadence, and writes one combined file
   per calendar month with geometry + Stokes V + C_IM + C_RE + A12 + A34
   columns (652 columns total -- see that module's docstring for the
   exact layout). Run as:

   ```bash
   python -m src.preprocessing.build_monthly_archive --start-year 2019
   ```

   Reads CDFs from `PSP_HFR_DIR` / `PSP_LFR_DIR`, writes to
   `PSP_RAW_DATA_DIR` (default `data/raw_monthly/`) as
   `data_<YYYY>_<MM>.h5`.

3. **Phase-folding (`src/preprocessing/make_folded_data.py`):**
   SPICE-based computation of PSP's Jovian System III longitude
   (lambda_III) and Io phase (Phi_Io), light-travel-time correction,
   daily-window noise whitening and solar burst removal, phase-grid
   binning, and median inpainting (Section 2 / 3.1 of the paper). Reads
   the monthly files from step 2 and selects the C IM (C^i_XY) column
   block by name. Run twice, once per reference frame:

   ```bash
   python -m src.preprocessing.make_folded_data --start-year 2019 --end-year 2024 --phase-frame both
   ```

   (`--phase-frame both`, the default, runs both passes in one call.)
   This produces `data_theta_psp.h5` (lambda_III, 3711 segments) and
   `data_phi_io.h5` (Phi_Io, 938 segments) -- both `master_list` HDF5
   tables, read by `load_jupiter_data(phase_frame=...)`.

4. **Manual categorization into `all_jupiter_data.h5` / `all_jupiter_data_unrolled.h5`:**
   Section 3.1 describes filtering `data_theta_psp.h5` down to a
   Jovian-positive subset with morphology labels (`type`: 1=Quiet HOM,
   2=Noisy HOM, 3=Vertex-Late), used both as ground truth for validation
   (Section 4.4) and as the eigenfaces training set (Section 3.3).

   **Gap:** the script that turns `data_theta_psp.h5` + the
   `JovianAnnotator` output into `all_jupiter_data.h5` /
   `all_jupiter_data_unrolled.h5` (i.e. that assigns the `type` column
   and does the final positive-subset filtering) isn't in this repo yet.
   `src/eigenfaces/annotator.py` produces the raw span annotations; the
   step that turns those into the two `all_jupiter_data*` files is still
   missing. If you have that script too, send it over the same way.

5. **Manual annotations (this repo, via `src/eigenfaces/annotator.py`):**
   `jovian_annotations_{negative,positive}.json` and
   `all_annotations_{negative,positive}.json` are produced by hand with
   the `JovianAnnotator` GUI tool, and are themselves required inputs to
   `src/occurrence/occurrence_plots.py` (Figure 4). These should be
   included directly in this data directory (they're small JSON files)
   rather than regenerated, so Figure 4 is exactly reproducible without
   redoing the manual labeling pass.

6. **Stokes V comparison (Section 4.2 control comparison):** uses the
   SAME monthly combined archive from step 2 (via its Stokes V columns,
   selected by name with `extract_stokes_v_columns()`) -- not a separate
   file. `src/stokes_comparison/` reads `data_<YYYY>_<MM>.h5` from
   `PSP_RAW_DATA_DIR` directly.

## Raw inputs

- **Raw CDF archive** (`PSP_HFR_DIR` / `PSP_LFR_DIR`, default
  `data/raw_cdf/HFR/` and `data/raw_cdf/LFR/`): the public FIELDS/RFS
  CDF files, consumed by `build_monthly_archive.py`.
- **SPICE kernels** (`PSP_SPICE_KERNEL_DIR`, default
  `data/spiceypy_kernels/`): `de440s.bsp`, `naif0012.tls`, `jup365.bsp`,
  `pck00010.tpc` (all from the NAIF generic kernel archive), and
  `spp_nom_20180812_20300101_v042_PostV7.bsp` (PSP's trajectory SPK,
  from the PSP SPICE kernel archive on NAIF/PDS).

## Expected files

```
data/
├── data_theta_psp.h5                    # <- output of make_folded_data.py (phase_frame="lambda_iii")
├── data_phi_io.h5                       # <- output of make_folded_data.py (phase_frame="phi_io")
├── freqs.csv
└── spiceypy_kernels/
    ├── de440s.bsp
    ├── naif0012.tls
    ├── jup365.bsp
    ├── pck00010.tpc
    └── spp_nom_20180812_20300101_v042_PostV7.bsp
```

Set these to point elsewhere if you keep data outside the repo:

```bash
export PSP_DATA_DIR=/path/to/your/data
export PSP_RAW_DATA_DIR=/path/to/raw/monthly/files    # defaults under PSP_DATA_DIR
export PSP_SPICE_KERNEL_DIR=/path/to/spice/kernels     # defaults under PSP_DATA_DIR
export PSP_HFR_DIR=/path/to/raw/cdf/HFR                # defaults under PSP_DATA_DIR
export PSP_LFR_DIR=/path/to/raw/cdf/LFR                # defaults under PSP_DATA_DIR
```
