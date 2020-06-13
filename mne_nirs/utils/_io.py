import pandas as pd
from scipy import stats
import numpy as np
import re
from mne.utils import warn


def _GLM_to_tidy_long(data, glm_est, design_matrix):

    if not (data.ch_names == list(glm_est.keys())):
        warn("MNE data structure does not match regression results")

    theta_estimates = np.zeros((len(glm_est), len(design_matrix.columns)))
    t_estimates = np.zeros((len(glm_est), len(design_matrix.columns)))
    df_estimates = np.zeros((len(glm_est), len(design_matrix.columns)))
    p_estimates = np.zeros((len(glm_est), len(design_matrix.columns)))
    mse_estimates = np.zeros((len(glm_est), len(design_matrix.columns)))

    for idx, name in enumerate(glm_est.keys()):
        theta_estimates[idx, :] = glm_est[name].theta.T
        df_estimates[idx, :] = glm_est[name].df_model
        mse_estimates[idx, :] = glm_est[name].MSE[0]
        for cond_idx, cond in enumerate(design_matrix.columns):
            t_estimates[idx, cond_idx] = glm_est[name].t(
                column=cond_idx)
            p_estimates[idx, cond_idx] = 2 * stats.t.cdf(
                -1.0 * np.abs(t_estimates[idx, cond_idx]),
                df=df_estimates[idx, cond_idx])

    df = pd.DataFrame()

    for ch_idx, ch in enumerate(data.ch_names):
        for cond_idx, cond in enumerate(design_matrix.columns):
            df = df.append({'ch_name': ch, 'condition': cond,
                            'variable': "theta",
                            'value': theta_estimates[ch_idx][cond_idx]},
                           ignore_index=True)
            df = df.append({'ch_name': ch, 'condition': cond,
                            'variable': "t",
                            'value': t_estimates[ch_idx][cond_idx]},
                           ignore_index=True)
            df = df.append({'ch_name': ch, 'condition': cond,
                            'variable': "df",
                            'value': df_estimates[ch_idx][cond_idx]},
                           ignore_index=True)
            df = df.append({'ch_name': ch, 'condition': cond,
                            'variable': "p",
                            'value': p_estimates[ch_idx][cond_idx]},
                           ignore_index=True)
            df = df.append({'ch_name': ch, 'condition': cond,
                            'variable': "mse",
                            'value': mse_estimates[ch_idx][cond_idx]},
                           ignore_index=True)

    return df


def _tidy_long_to_wide(d, expand_output=True):
    d = d.set_index(['ch_name', 'condition'])
    d = d.pivot_table(columns='variable', values='value',
                      index=['ch_name', 'condition'])
    d.reset_index(inplace=True)

    if expand_output:
        try:
            d["Source"] = [re.search(r'S(\d+)_D(\d+) (\w+)', ch).group(1)
                           for ch in d["ch_name"]]
            d["Detector"] = [re.search(r'S(\d+)_D(\d+) (\w+)', ch).group(2)
                             for ch in d["ch_name"]]
            d["Chroma"] = [re.search(r'S(\d+)_D(\d+) (\w+)', ch).group(3)
                           for ch in d["ch_name"]]
        except AttributeError:
            warn("Non standard source detector names used")
        d["Significant"] = d["p"] < 0.05

    return d
