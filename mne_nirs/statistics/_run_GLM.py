# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import numpy as np
from mne.io.pick import _picks_to_idx


def run_GLM(raw, design_matrix, noise_model='ar1', bins=100,
            n_jobs=1, verbose=0):
    """
    Run GLM on data using supplied design matrix.

    This is a wrapper function for nilearn.stats.first_level_model.run_glm.

    Parameters
    ----------
    raw : instance of Raw
        The haemoglobin data.
    design_matrix : as specified in Nilearn
        The design matrix.
    noise_model : {'ar1', 'ols'}, optional
        The temporal variance model. Defaults to 'ar1'.
    bins : : int, optional
        Maximum number of discrete bins for the AR(1) coef histogram.
    n_jobs : int, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs'.
    verbose : int, optional
        The verbosity level. Defaut is 0

    Returns
    -------
    glm_estimates : dict
        Keys correspond to the different labels values values are
        RegressionResults instances corresponding to the voxels.
    """
    from nilearn.stats.first_level_model import run_glm

    picks = _picks_to_idx(raw.info, 'fnirs', exclude=[], allow_empty=True)
    ch_names = raw.ch_names

    results = dict()
    for pick in picks:
        labels, glm_estimates = run_glm(raw.get_data(pick).T,
                                        design_matrix.values,
                                        noise_model=noise_model, bins=bins,
                                        n_jobs=n_jobs, verbose=verbose)
        results[ch_names[pick]] = glm_estimates[labels[0]]

    return results


def compute_contrast(glm_est, contrast, contrast_type=None):
    """
    Compute contrasts on regression results.

    This is a wrapper function for nilearn.stats.contrasts.

    Parameters
    ----------
    glm_estimates : dict
        Dictionary of nilearn regression results as returned by `run_glm`.
    contrast : numpy.ndarray of shape (p) or (q, p),
        Where q = number of contrast vectors and p = number of regressors.
    contrast_type : {None, ‘t’, ‘F’}, optional
        Type of the contrast. If None, then defaults to ‘t’ for 1D con_val
        and ‘F’ for 2D con_val.

    Returns
    -------
    contrast : Contrast instance,
        Yields the statistics of the contrast (effects, variance, p-values).
    """

    from nilearn.stats.contrasts import compute_contrast as _cc
    return _cc(np.array(list(glm_est.keys())), glm_est, contrast,
               contrast_type=contrast_type)
