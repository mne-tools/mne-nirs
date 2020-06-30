import numpy as np


def glm_region_of_interest(stats, group_by, cond_idx, cond_name):
    """
    Calculate statistics for region of interest.

    Parameters
    ----------
    raw : instance of Raw
        The haemoglobin data.
    sd_pairs : list of lists
        List of source detector pairs. For example, to request the picks for
        channels comprising of source 1 detector 4 and source 13
        detector 4 you would specify [[1, 4], [13, 4]].
    on_missing : str
        What to do if one or several requested source detector are not found
        in the recording.
        Valid keys are 'error' | 'warning' | 'ignore'
        Default is 'error'. If on_missing is 'warning' it will proceed but
        warn, if 'ignore' it will proceed silently. Note.
        If none of the event ids are found in the data, an error will be
        automatically generated irrespective of this parameter.

    Returns
    -------
    picks : list of integers
        List of picks corresponding to requested source detector pairs.
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

            weights = 1. / np.asarray(ses)
            weights /= np.sum(weights)

            theta = np.sum(thetas * weights)
            s = np.sum(ses * weights)

            t = theta / s

            p = 2 * ss.t.cdf(-1.0 * np.abs(t), df=dfe)

            df = df.append({'ROI': roi_name,
                            'Condition': cond_name,
                            'Chroma': chroma,
                            'theta': theta,
                            'se': s,
                            't': t,
                            'dfe': dfe,
                            'p': p, },
                           ignore_index=True)

    return df


def _se(d):
    return np.sqrt(np.diag(d.vcov())) * 1e6

