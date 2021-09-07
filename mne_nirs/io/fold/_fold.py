import pandas
import pandas as pd
import numpy as np
import eeg_positions as ep


fname = "/Users/rluke/Downloads/10-10.xls"

def _read_fold_xls(fname, atlas="Juelich"):

    page_reference = {"AAL2": 2,
                      "AICHA": 5,
                      "Brodmann": 8,
                      "Juelich": 11,
                      "Loni": 14}

    tbl = pd.read_excel(fname, sheet_name = page_reference[atlas])

    # Remove the spacing between rows
    empty_rows = np.where(np.isnan(tbl["Specificity"]))[0]
    tbl = tbl.drop(empty_rows).reset_index(drop=True)

    # Empty values in the table mean its the same as above
    for row_idx in range(1, tbl.shape[0]):
        for col in tbl.columns:
            if not isinstance(tbl[col][row_idx], str):
                if np.isnan(tbl[col][row_idx]):
                    tbl[col][row_idx] = tbl[col][row_idx-1]

    return tbl


def _generate_all_locations(system=["1020", "1010", "1005"]):
    coords = pd.DataFrame()
    for sys in system:
        coords = coords.append(ep.get_elec_coords(sys, dim="3d"))

    return coords.reset_index(drop=True)


def _find_closest_standard_location(x, y, z, reference):

    dists = np.zeros(reference["x"].shape)
    p0 = np.array([x, y, z])
    for idx in range(len(dists)):
        p1 = np.array([reference["x"][idx], reference["y"][idx], reference["z"][idx]])
        dists[idx] = np.linalg.norm(p0 -  p1)

    min_idx = np.argmin(dists)

    return reference["label"][min_idx]

tbl = _read_fold_xls(fname)
