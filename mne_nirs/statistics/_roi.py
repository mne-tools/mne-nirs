import numpy as np


def glm_region_of_interest(stats, group_by, cond_idx,
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
    weighted : Bool
        Should channels be weighted by inverse of standard error (True).

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

        roi_name = region
        picks = group_by[region]

        for chroma in np.unique(chromas[picks]):

            chroma_picks = np.where([c == chroma for c in chromas[picks]])[0]
            chroma_picks = [picks[cp] for cp in chroma_picks]

            thetas = list()
            ses = list()
            dfe = 0

            for pick in chroma_picks:
                thetas.append(1e6 * (stats[ch_names[pick]].theta)[cond_idx][0])
                ses.append(_se(stats[ch_names[pick]])[cond_idx])
                dfe = stats[ch_names[pick]].df_model

            # Should channels be weighted by inverse of standard error
            if weighted:
                weights = 1. / np.asarray(ses)
            else:
                weights = np.ones(len(ses))
            weights /= np.sum(weights)

            theta = np.sum(thetas * weights)
            s = np.sum(ses * weights)

            t = theta / s

            p = 2 * ss.t.cdf(-1.0 * np.abs(t), df=dfe)

            df = df.append({'ROI': roi_name,
                            'Condition': cond_name,
                            'Chroma': chroma,
                            'theta': theta / 1.0e6,
                            'se': s,
                            't': t,
                            'dfe': dfe,
                            'p': p, },
                           ignore_index=True)

    return df


def _se(d):
    return np.sqrt(np.diag(d.vcov())) * 1e6
