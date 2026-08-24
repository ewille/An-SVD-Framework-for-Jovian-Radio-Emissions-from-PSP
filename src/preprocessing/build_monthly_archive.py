"""
Converts the raw PSP FIELDS/RFS HFR/LFR CDF archive into one combined
monthly HDF5 file per calendar month -- the RAW_MONTHLY_PATTERN files
that src/preprocessing/make_folded_data.py folds, and that
src/stokes_comparison/ reads directly for the Stokes V comparison
(Section 4.2). This is the step upstream of everything else in this
repository.

Each monthly output file has columns, in order:
    time, HAE.PSP, IAU.PSP, x_pos, y_pos, z_pos,
    x_V1V2, y_V1V2, z_V1V2, x_V3V4, y_V3V4, z_V3V4,
    STOKES V channel 1-128, C IM channel 1-128, C RE channel 1-128,
    A 12 channel 1-128, A 34 channel 1-128
(652 columns total). "HAE.PSP"/"IAU.PSP" are the antenna/Sun-geometry
dot-product diagnostics (see build_vector_analysis below); "pos"/V1V2/V3V4
are Sun-interpolated position vectors; C_IM is the cross-imaginary
product (C^i_XY, what the paper's extraction pipeline exclusively uses --
Section 2); STOKES V is the RFS Level-3 Stokes V product used as the
independent cross-check in Section 4.2.

*** BUGS FIXED DURING PORTING (both would have crashed the original
script before it produced a single file) ***
1. `hfr_stokes_v_df = pd.DataFrame(..., index=pd.to_datetime(l=hfr_times))`
   -- `to_datetime()` has no parameter named `l`; this was a stray typo.
   Fixed to `pd.to_datetime(hfr_times)`.
2. `hfr_stokes_v = concat_from_dict(lfr_dict, "psp_fld_l3_rfs_hfr_STOKES_V")`
   -- pulled an HFR-named CDF variable from `lfr_dict` instead of
   `hfr_dict`, unlike every other HFR extraction in the script. Fixed to
   use `hfr_dict`.

*** IMPORTANT: column-selection consistency with make_folded_data.py ***
make_folded_data.py originally selected its "spectra" and "pos" inputs by
POSITIONAL slice (`data.iloc[:, 8:136]`, `data.iloc[:, 5:8]`). Checked
against the column layout actually produced here, those positions land
on the wrong columns entirely -- `iloc[:, 8:136]` selects 4 leftover
geometry columns plus STOKES V channels 1-124, not the C IM block (which
lives at columns 140-267). Since the paper's whole extraction method
relies exclusively on C^i_XY (C IM), this would have meant
make_folded_data.py folding and SVD-extracting mostly Stokes V data instead.
make_folded_data.py has been updated to select columns by NAME instead of
position to fix this and to be robust to future column-order changes --
see its module docstring. This needs a sanity check against how the
paper's actual reported results were produced, in case a different,
already-correct column mapping was used elsewhere.
"""
import os
import numpy as np
import pandas as pd
import spacepy
import spacepy.pycdf
import datetime
from datetime import timedelta
from glob import glob
from joblib import Parallel, delayed

from ..config import HFR_DIR, LFR_DIR, RAW_DATA_DIR


def _load_cdf_dict(directory):
    """Loads every non-empty CDF file in `directory`, sorted by the timestamp token in its filename."""
    paths = sorted(glob(os.path.join(str(directory), "*")), key=lambda x: os.path.basename(x).split("_")[5])

    cdf_dict = {}
    idx = 0
    for filepath in paths:
        if os.path.getsize(filepath) == 0:
            continue
        try:
            cdf_dict[idx] = spacepy.pycdf.CDF(filepath)
            idx += 1
        except Exception:
            break

    return cdf_dict


def check_orientation(v12, v34, s):
    """Projects the antenna cross-product onto the Sun direction, for the antenna/Sun geometry diagnostic."""
    return np.dot((np.cross(v12, v34) / (np.sin(np.deg2rad(85)))), s) / np.linalg.norm(s)


def sun_interp(atime, sdata):
    """Interpolates a 1-minute-cadence Sun-direction vector onto the antenna's own (finer) timestamps."""
    sun_time = [atime[0].timestamp() + 60 * i for i in range(len(sdata))]
    antenna_time = [atime[i].timestamp() for i in range(len(atime))]

    x_interp = pd.DataFrame(np.interp(antenna_time, sun_time, sdata.iloc[:][0]))
    y_interp = pd.DataFrame(np.interp(antenna_time, sun_time, sdata.iloc[:][1]))
    z_interp = pd.DataFrame(np.interp(antenna_time, sun_time, sdata.iloc[:][2]))
    return pd.concat([x_interp, y_interp, z_interp], axis=1)


def concat_from_dict(data_dict, key):
    """Concatenates one CDF variable across every loaded file in a CDF dict, in load order."""
    return np.concatenate([data_dict[i][key][...] for i in range(len(data_dict))])


def build_vector_analysis(lfr_dict):
    """
    Computes the antenna/Sun-direction geometry diagnostics (HAE and IAU
    dot products/angles) and the Sun-interpolated position vectors, for
    every LFR cross-correlation timestamp.

    Returns (vector_analysis_df, pos_df, v1v2_df, v3v4_df, lfr_times).
    """
    hae_dot, hae_angle, iau_dot, iau_angle, time_tot = [], [], [], [], []
    pos = pd.DataFrame([])
    v1v2 = pd.DataFrame([])
    v3v4 = pd.DataFrame([])

    print("begin vector analysis and vector packaging")

    for i in range(len(lfr_dict)):
        hae = pd.DataFrame(lfr_dict[i]["psp_fld_l3_rfs_lfr_position_HAE"][...])
        iau_psp = pd.DataFrame(lfr_dict[i]["psp_fld_l3_rfs_lfr_position_IAU_JUPITER"][...])
        iau_12 = pd.DataFrame(lfr_dict[i]["psp_fld_l3_rfs_lfr_ch0_V1V2_IAU_JUPITER"][...])
        iau_34 = pd.DataFrame(lfr_dict[i]["psp_fld_l3_rfs_lfr_ch1_V3V4_IAU_JUPITER"][...])
        epoch = lfr_dict[i]["epoch_lfr_cross_im_V1V2_V3V4"][...]

        if i % 100 == 0:
            print("days completed: ", i + 1)

        s_hae = sun_interp(epoch, hae)
        s_iau = sun_interp(epoch, iau_psp)

        v1v2 = pd.concat([v1v2, iau_12])
        v3v4 = pd.concat([v3v4, iau_34])
        pos = pd.concat([pos, s_iau])

        def compute_row(j):
            h_dot = check_orientation(iau_12.iloc[j], iau_34.iloc[j], s_hae.iloc[j])
            h_angle = np.arccos(h_dot)
            i_dot = check_orientation(iau_12.iloc[j], iau_34.iloc[j], s_iau.iloc[j])
            i_angle = np.rad2deg(np.arccos(i_dot))
            return h_dot, h_angle, i_dot, i_angle, epoch[j]

        results = Parallel(n_jobs=32, prefer="threads")(delayed(compute_row)(j) for j in range(len(epoch)))

        for h_dot, h_angle, i_dot, i_angle, t in results:
            hae_dot.append(h_dot)
            hae_angle.append(h_angle)
            iau_dot.append(i_dot)
            iau_angle.append(i_angle)
            time_tot.append(t)

    vector_analysis = pd.DataFrame(
        {"HAE dot PSP": hae_dot, "HAE_PSP angle": hae_angle, "IAU dot PSP": iau_dot, "IAU_PSP angle": iau_angle},
        index=pd.to_datetime(time_tot),
    )
    print("vector analysis done")

    return vector_analysis, pos, v1v2, v3v4


def build_spectral_dataframes(lfr_dict, hfr_dict):
    """
    Loads, concatenates, and 15s-bins the cross-imaginary (C_IM),
    cross-real (C_RE), auto-average (A12/A34), and Stokes V products from
    both receivers.

    Returns (spectra_cim, spectra_cre, spectra_a12, spectra_a34, stokes_v, lfr_times).
    """
    lfr_times = pd.to_datetime(concat_from_dict(lfr_dict, "epoch_lfr_cross_im_V1V2_V3V4"))
    lfr_spectra_cim = concat_from_dict(lfr_dict, "psp_fld_l3_rfs_lfr_cross_im_V1V2_V3V4")
    lfr_spectra_cre = concat_from_dict(lfr_dict, "psp_fld_l3_rfs_lfr_cross_re_V1V2_V3V4")
    lfr_spectra_a12 = concat_from_dict(lfr_dict, "psp_fld_l3_rfs_lfr_auto_averages_ch0_V1V2")
    lfr_spectra_a34 = concat_from_dict(lfr_dict, "psp_fld_l3_rfs_lfr_auto_averages_ch1_V3V4")
    lfr_stokes_v = concat_from_dict(lfr_dict, "psp_fld_l3_rfs_lfr_STOKES_V")

    lfr_cim_df = pd.DataFrame(lfr_spectra_cim, index=pd.to_datetime(lfr_times))
    lfr_cre_df = pd.DataFrame(np.asarray(lfr_spectra_cre), index=pd.to_datetime(lfr_times))
    lfr_a12_df = pd.DataFrame(np.asarray(lfr_spectra_a12), index=pd.to_datetime(lfr_times))
    lfr_a34_df = pd.DataFrame(np.asarray(lfr_spectra_a34), index=pd.to_datetime(lfr_times))
    lfr_stokes_v_df = pd.DataFrame(np.asarray(lfr_stokes_v), index=pd.to_datetime(lfr_times))

    hfr_times = pd.to_datetime(concat_from_dict(hfr_dict, "epoch_hfr_cross_im_V1V2_V3V4"))
    hfr_spectra_cim = concat_from_dict(hfr_dict, "psp_fld_l3_rfs_hfr_cross_im_V1V2_V3V4")
    hfr_spectra_cre = concat_from_dict(hfr_dict, "psp_fld_l3_rfs_hfr_cross_re_V1V2_V3V4")
    hfr_spectra_a12 = concat_from_dict(hfr_dict, "psp_fld_l3_rfs_hfr_auto_averages_ch0_V1V2")
    hfr_spectra_a34 = concat_from_dict(hfr_dict, "psp_fld_l3_rfs_hfr_auto_averages_ch1_V3V4")
    hfr_stokes_v = concat_from_dict(hfr_dict, "psp_fld_l3_rfs_hfr_STOKES_V")  # fix: was lfr_dict, see module docstring

    hfr_cim_df = pd.DataFrame(hfr_spectra_cim, index=pd.to_datetime(hfr_times))
    hfr_cre_df = pd.DataFrame(np.asarray(hfr_spectra_cre), index=pd.to_datetime(hfr_times))
    hfr_a12_df = pd.DataFrame(np.asarray(hfr_spectra_a12), index=pd.to_datetime(hfr_times))
    hfr_a34_df = pd.DataFrame(np.asarray(hfr_spectra_a34), index=pd.to_datetime(hfr_times))
    hfr_stokes_v_df = pd.DataFrame(np.asarray(hfr_stokes_v), index=pd.to_datetime(hfr_times))  # fix: was pd.to_datetime(l=hfr_times)

    lfr_cim_binned = lfr_cim_df.resample("15s").mean()
    lfr_cre_binned = lfr_cre_df.resample("15s").mean()
    lfr_a12_binned = lfr_a12_df.resample("15s").mean()
    lfr_a34_binned = lfr_a34_df.resample("15s").mean()
    lfr_stokes_v_binned = lfr_stokes_v_df.resample("15s").mean()

    hfr_cim_binned = hfr_cim_df.resample("15s").mean()
    hfr_cre_binned = hfr_cre_df.resample("15s").mean()
    hfr_a12_binned = hfr_a12_df.resample("15s").mean()
    hfr_a34_binned = hfr_a34_df.resample("15s").mean()
    hfr_stokes_v_binned = hfr_stokes_v_df.resample("15s").mean()

    spectra_cim = pd.concat([lfr_cim_binned, hfr_cim_binned], axis=1)
    spectra_cre = pd.concat([lfr_cre_binned, hfr_cre_binned], axis=1)
    spectra_a12 = pd.concat([lfr_a12_binned, hfr_a12_binned], axis=1)
    spectra_a34 = pd.concat([lfr_a34_binned, hfr_a34_binned], axis=1)
    stokes_v = pd.concat([lfr_stokes_v_binned, hfr_stokes_v_binned], axis=1)

    print("Spectral data merged and aligned.")
    return spectra_cim, spectra_cre, spectra_a12, spectra_a34, stokes_v, lfr_times


COLUMN_NAMES = (
    ["HAE.PSP", "IAU.PSP", "x_pos", "y_pos", "z_pos", "x_V1V2", "y_V1V2", "z_V1V2", "x_V3V4", "y_V3V4", "z_V3V4"]
    + [f"STOKES V channel {i+1}" for i in range(128)]
    + [f"C IM channel {i+1}" for i in range(128)]
    + [f"C RE channel {i+1}" for i in range(128)]
    + [f"A 12 channel {i+1}" for i in range(128)]
    + [f"A 34 channel {i+1}" for i in range(128)]
)


def export_month(year, month, spectra_cim, spectra_cre, spectra_a12, spectra_a34, stokes_v, vector_analysis_binned, pos_binned, v1v2_binned, v3v4_binned, output_dir=None):
    """Writes one calendar month's combined geometry+spectral data to an HDF5 file, if any data falls in that month."""
    output_dir = str(output_dir or RAW_DATA_DIR)
    os.makedirs(output_dir, exist_ok=True)

    mask = (spectra_cim.index.year == year) & (spectra_cim.index.month == month)
    if not mask.any():
        return None

    spectra_cim_month = spectra_cim.loc[mask]
    spectra_cre_month = spectra_cre.loc[mask]
    spectra_a12_month = spectra_a12.loc[mask]
    spectra_a34_month = spectra_a34.loc[mask]
    stokes_v_month = stokes_v.loc[mask]
    vector_analysis_month = vector_analysis_binned.loc[mask]
    pos_month = pos_binned.loc[mask]

    if not (len(spectra_cim_month) == len(vector_analysis_month) == len(pos_month)):
        print(f"Mismatch in lengths for {year}-{month:02d}")
        return None

    this_month = pd.concat(
        [
            vector_analysis_month[["HAE dot PSP", "IAU dot PSP"]],
            pos_month,
            v1v2_binned,
            v3v4_binned,
            stokes_v_month,
            spectra_cim_month,
            spectra_cre_month,
            spectra_a12_month,
            spectra_a34_month,
        ],
        axis=1,
    )
    this_month.columns = COLUMN_NAMES
    this_month.insert(0, "time", this_month.index)
    this_month.reset_index(drop=True, inplace=True)

    filename = os.path.join(output_dir, f"data_{year}_{month:02d}.h5")
    this_month.to_hdf(filename, key="df", index=False)
    print(f"Wrote: {filename}, shape: {this_month.shape}")
    return len(this_month)


def run(hfr_dir=None, lfr_dir=None, output_dir=None, start_year=2019, end_year=None):
    """Loads the full raw CDF archive and exports every available calendar month."""
    end_year = end_year or datetime.datetime.today().year

    hfr_dict = _load_cdf_dict(hfr_dir or HFR_DIR)
    lfr_dict = _load_cdf_dict(lfr_dir or LFR_DIR)
    print("dictionaries made!")

    vector_analysis, pos, v1v2, v3v4 = build_vector_analysis(lfr_dict)
    spectra_cim, spectra_cre, spectra_a12, spectra_a34, stokes_v, lfr_times = build_spectral_dataframes(lfr_dict, hfr_dict)

    pos_df = pd.DataFrame(pos.reset_index(drop=True).to_numpy(), index=pd.to_datetime(lfr_times))
    v1v2_df = pd.DataFrame(v1v2.reset_index(drop=True).to_numpy(), index=pd.to_datetime(lfr_times))
    v3v4_df = pd.DataFrame(v3v4.reset_index(drop=True).to_numpy(), index=pd.to_datetime(lfr_times))

    vector_analysis_binned = vector_analysis.resample("15s").mean()
    pos_binned = pos_df.resample("15s").mean()
    v1v2_binned = v1v2_df.resample("15s").mean()
    v3v4_binned = v3v4_df.resample("15s").mean()

    output_dir = str(output_dir or RAW_DATA_DIR)
    records = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            n = export_month(
                year, month, spectra_cim, spectra_cre, spectra_a12, spectra_a34, stokes_v,
                vector_analysis_binned, pos_binned, v1v2_binned, v3v4_binned, output_dir=output_dir,
            )
            if n is not None:
                records.append({"year": year, "month": month, "amount of data": n})

    usable_data = pd.DataFrame(records)
    summary_path = os.path.join(output_dir, "available_data_monthly.csv")
    usable_data.to_csv(summary_path, index=False)
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2019)
    parser.add_argument("--end-year", type=int, default=None, help="Defaults to the current year.")
    args = parser.parse_args()

    run(start_year=args.start_year, end_year=args.end_year)
