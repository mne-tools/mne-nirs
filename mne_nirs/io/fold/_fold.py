import pandas as pd
import numpy as np
import mne


def _read_fold_xls(fname, atlas="Juelich"):

    page_reference = {"AAL2": 2,
                      "AICHA": 5,
                      "Brodmann": 8,
                      "Juelich": 11,
                      "Loni": 14}

    tbl = pd.read_excel(fname,
                        sheet_name=page_reference[atlas])

    # Remove the spacing between rows
    empty_rows = np.where(np.isnan(tbl["Specificity"]))[0]
    tbl = tbl.drop(empty_rows).reset_index(drop=True)

    # Empty values in the table mean its the same as above
    for row_idx in range(1, tbl.shape[0]):
        for col in tbl.columns:
            if not isinstance(tbl[col][row_idx], str):
                if np.isnan(tbl[col][row_idx]):
                    tbl[col][row_idx] = \
                        tbl[col][row_idx - 1]

    return tbl


def _generate_all_locations():

    montage = mne.channels.make_standard_montage(
        'standard_1020', head_size=0.1)
    coords = pd.DataFrame.from_dict(
        montage.get_positions()['ch_pos']).T
    coords["label"] = coords.index
    coords = coords.rename(columns={0: "x", 1: "y", 2: "z"})

    return coords.reset_index(drop=True)


def _find_closest_standard_location(x, y, z, reference):

    dists = np.zeros(reference["x"].shape)
    p0 = np.array([x, y, z])
    for idx in range(len(dists)):
        p1 = np.array([reference["x"][idx],
                       reference["y"][idx],
                       reference["z"][idx]])
        dists[idx] = np.linalg.norm(p0 - p1)

    min_idx = np.argmin(dists)

    return reference["label"][min_idx]
