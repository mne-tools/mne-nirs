# Author: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import pandas as pd
import numpy as np
import mne
from mne.transforms import apply_trans, _get_trans
from mne.utils import _validate_type
from mne.io import BaseRaw


def _read_fold_xls(fname, atlas="Juelich"):
    """Read fOLD toolbox xls file.

    The values are then manipulated in to a tidy dataframe.

    Note the xls files are not included as no license is provided.

    Parameters
    ----------
    fname : str
        Path to xls file.
    atlas : str
        Requested atlas.
    """
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
        for col_idx, col in enumerate(tbl.columns):
            if not isinstance(tbl[col][row_idx], str):
                if np.isnan(tbl[col][row_idx]):
                    tbl.iloc[row_idx, col_idx] = \
                        tbl.iloc[row_idx - 1, col_idx]

    tbl["Specificity"] = tbl["Specificity"] * 100
    tbl["brainSens"] = tbl["brainSens"] * 100
    return tbl


def _generate_montage_locations(montage='standard_1005'):
    """Get standard montage locations in dataframe.

    Data is returned in the same format as the eeg_positions library.

    Parameters
    ----------
    montage : str
        Standard MNE montage to use.
    """
    montage = mne.channels.make_standard_montage(montage)
    coords = pd.DataFrame.from_dict(
        montage.get_positions()['ch_pos']).T
    coords["label"] = coords.index
    coords = coords.rename(columns={0: "x", 1: "y", 2: "z"})

    return coords.reset_index(drop=True)


def _find_closest_standard_location(position, reference, trans_pos='mri'):
    """Return closest montage label to coordinates.

    Parameters
    ----------
    position : array
        Coordinates.
    reference : dataframe
        As generated by _generate_montage_locations.
    trans_pos : str
        Apply a transformation to positions to specified frame.
        Use None for no transformation.
    """
    p0 = np.array(position)
    if trans_pos is not None:
        head_mri_t, _ = _get_trans('fsaverage', 'head', 'mri')
        p0 = apply_trans(head_mri_t, p0)

    dists = np.zeros(reference["x"].shape)

    for idx in range(len(dists)):
        p1 = np.array([reference["x"][idx],
                       reference["y"][idx],
                       reference["z"][idx]])
        dists[idx] = np.linalg.norm(p0 - p1)

    min_idx = np.argmin(dists)

    return reference["label"][min_idx]


def fold_landmark_specificity(raw, landmark, fold_files=[None],
                              atlas="Juelich"):
    """Return the specificity of each channel to a specified brain landmark.

    Specificity values as stored in the fOLD toolbox
    :footcite:`morais2018fnirs`
    excel files.

    The data is not provided with MNE-NIRS. You must download the excel
    spreadsheets from the authors website and provide the paths using
    the ``fold_files`` argument.

    Parameters
    ----------
    raw : BaseRaw
        The fNIRS data.
    landmark : str
        Landmark of interest. Must be present in fOLD toolbox data file.
    fold_files : list
        Paths to fold toolbox files.
    atlas : str
        Brain atlas to use.

    Returns
    -------
    spec : array
        Specificity values for each channel to brain landmark.

    References
    ----------
    .. footbibliography::
    """
    if None in fold_files:
        raise ValueError("You must specify the path to fOLD xls files")

    if not isinstance(landmark, str):
        raise ValueError(f"Landmark must be a string. Got {type(landmark)}")

    _validate_type(raw, BaseRaw, 'raw')

    reference_locations = _generate_montage_locations()

    fold_tbl = pd.DataFrame()
    for fname in fold_files:
        fold_tbl = pd.concat([fold_tbl, _read_fold_xls(fname, atlas=atlas)],
                             ignore_index=True)

    specificity = np.zeros(len(raw.ch_names))
    for cidx in range(len(raw.ch_names)):

        tbl = _source_detector_fold_table(raw, cidx,
                                          reference_locations, fold_tbl)

        if len(tbl) > 0:
            tbl["ContainsLmk"] = [landmark in la for la in tbl["Landmark"]]
            tbl = tbl.query("ContainsLmk == True")["Specificity"]

        if len(tbl) == 0:
            continue
            # print(f"No data for {src_name}-{det_name}")
        elif len(tbl) == 1:
            specificity[cidx] = tbl.values[0]
        else:
            raise RuntimeError("Multiple specificity values returned")

    return np.array(specificity)


def fold_channel_specificity(raw, fold_files=[None], atlas="Juelich"):
    """Return the landmarks and specificity a channel is sensitive to.

    Specificity values as stored in the fOLD toolbox
    :footcite:`morais2018fnirs`
    excel files.

    The data is not provided with MNE-NIRS. You must download the excel
    spreadsheets from the authors website and provide the paths using
    the ``fold_files`` argument.

    Parameters
    ----------
    raw : BaseRaw
        The fNIRS data.
    fold_files : list
        Paths to fold toolbox files.
    atlas : str
        Brain atlas to use.

    Returns
    -------
    spec : list of dataframes
        List of dataframes, one for each channel.

    References
    ----------
    .. footbibliography::
    """
    if None in fold_files:
        raise ValueError("You must specify the path to fOLD xls files")

    _validate_type(raw, BaseRaw, 'raw')

    reference_locations = _generate_montage_locations()

    fold_tbl = pd.DataFrame()
    for fname in fold_files:
        fold_tbl = pd.concat([fold_tbl, _read_fold_xls(fname, atlas=atlas)],
                             ignore_index=True)

    chan_spec = list()
    for cidx in range(len(raw.ch_names)):

        tbl = _source_detector_fold_table(raw, cidx,
                                          reference_locations, fold_tbl)
        chan_spec.append(tbl.reset_index(drop=True))

    return chan_spec


def _source_detector_fold_table(raw, cidx, reference_locations, fold_tbl):
    src = raw.info['chs'][cidx]['loc'][3:6]
    det = raw.info['chs'][cidx]['loc'][6:9]

    src_name = _find_closest_standard_location(src,  # noqa
                                               reference_locations)
    det_name = _find_closest_standard_location(det,  # noqa
                                               reference_locations)

    tbl = fold_tbl.query("Source == @src_name"). \
        query("Detector == @det_name")
    # Try reversing source and detector
    if len(tbl) == 0:
        tbl = fold_tbl.query("Source == @det_name"). \
            query("Detector == @src_name")

    return tbl
