# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)


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
    labels : array
        A map of values on voxels used to identify the corresponding model.
    glm_estimates : dict
        Keys correspond to the different labels values values are
        RegressionResults instances corresponding to the voxels.
    """
    from nilearn.stats.first_level_model import run_glm

    labels, glm_estimates = run_glm(raw._data.T, design_matrix.values,
                                    noise_model=noise_model, bins=bins,
                                    n_jobs=n_jobs, verbose=verbose)

    return labels, glm_estimates
