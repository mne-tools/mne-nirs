# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import numpy as np

from mne.utils import warn


def glm_region_of_interest(glm, group_by, cond_idx,
                           cond_name, weighted=True):
    """
    Calculate statistics for region of interest.

    Parameters
    ----------
    glm : dict
        Need to write.
    group_by : dict
        Specifies which channels are aggregated into a single ROI.
        The dict key will be used as the ROI label and the dict
        values must be lists of picks (either channel names or integer indices
        of ``epochs.ch_names``). For example::

            group_by=dict(Left_ROI=[1, 2, 3, 4], Right_ROI=[5, 6, 7, 8])

        Note that within a dict entry all channels must have the same type.
    cond_idx : int
        Index of condition of interest.
    cond_name : str
        Name to be used for condition.
    weighted : Bool
        Should channels be weighted by inverse of standard error (True).

    Returns
    -------
    stats : DataFrame
        Statistics for each ROI.
    """
    warn('"glm_region_of_interest" has been deprecated in favor of the more '
         'comprehensive GLM class and will be removed in v1.0.0. '
         'Use the RegressionResults class "region_of_interest_dataframe()" '
         'method instead.',
         DeprecationWarning)

    return _glm_region_of_interest(glm, group_by,
                                   cond_idx, cond_name, weighted=weighted)


def _glm_region_of_interest(stats, group_by, cond_idx,
                            cond_name, weighted=True):
    """
    Calculate statistics for region of interest.

    Parameters
    ----------
    stats : dict
        Need to write.
    group_by : dict
        Specifies which channels are aggregated into a single ROI.
        The dict key will be used as the ROI label and the dict
        values must be lists of picks (either channel names or integer indices
        of ``epochs.ch_names``). For example::

            group_by=dict(Left_ROI=[1, 2, 3, 4], Right_ROI=[5, 6, 7, 8])

        Note that within a dict entry all channels must have the same type.
    cond_idx : int
        Index of condition of interest.
    cond_name : str
        Name to be used for condition.
    weighted : Bool | dict
        Weighting to be applied to each channel in the ROI computation.
        If False, then all channels will be weighted equally.
        If True, channels will be weighted by the inverse of
        the standard error of the GLM fit.
        For manual specification of the channel weighting a dictionary
        can be provided.
        If a dictionary is provided, the keys and length of lists must
        match the ``group_by`` parameters.
        The weights will be scaled internally to sum to 1.

    Returns
    -------
    stats : DataFrame
        Statistics for each ROI.
    """
    from scipy import stats as ss
    import pandas as pd

    df = pd.DataFrame()

    ch_names = np.array(list(stats.keys()))
    chromas = np.array([name[-3:] for name in ch_names])

    for region in group_by:

        if isinstance(weighted, dict):
            weights_region = weighted[region]

        roi_name = region
        picks = group_by[region]

        for chroma in np.unique(chromas[picks]):

            chroma_idxs = np.where([c == chroma for c in chromas[picks]])[0]
            chroma_picks = [picks[ci] for ci in chroma_idxs]

            thetas = list()
            ses = list()
            dfe = 0

            for pick in chroma_picks:
                thetas.append(1e6 * (stats[ch_names[pick]].theta)[cond_idx][0])
                ses.append(_se(stats[ch_names[pick]])[cond_idx])
                dfe = stats[ch_names[pick]].df_model

            # Apply weighting by standard error or custom values
            if weighted is True:
                weights = 1. / np.asarray(ses)
            elif weighted is False:
                weights = np.ones((len(ses), 1))
            elif isinstance(weighted, dict):
                weights = [float(weights_region[ci]) for ci in chroma_idxs]
                weights = np.asarray(weights).reshape((len(ses), 1))
            else:
                raise ValueError("Weighted parameter is not bool or dict")
            # Ensure weights sum to one
            weights /= np.sum(weights)

            theta = np.sum(thetas * weights)
            s = np.sum(ses * weights)

            t = theta / s

            p = 2 * ss.t.cdf(-1.0 * np.abs(t), df=dfe)

            this_df = pd.DataFrame(
                {'ROI': roi_name,
                 'Condition': cond_name,
                 'Chroma': chroma,
                 'theta': theta / 1.0e6,
                 'se': s,
                 't': t,
                 'dfe': dfe,
                 'p': p, }, index=[0])
            df = pd.concat([df, this_df], ignore_index=True)

    df.reset_index(inplace=True, drop=True)

    return df


def _se(d):
    return np.sqrt(np.diag(d.vcov())) * 1e6
