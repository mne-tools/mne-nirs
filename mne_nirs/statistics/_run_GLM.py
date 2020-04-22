# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)


def run_GLM(raw, design_matrix, noise_model='ar1', bins=100,
            n_jobs=1, verbose=0):
    """
    Run GLM on data using supplied design matrix.

    This is a wrapper function for nilearn.stats.first_level_model.run_glm.

      .. warning:: Work in progress: I am trying to think on the best API.

    Parameters
    ----------
    raw : instance of Raw
        Haemoglobin data.
    design_matrix : as specified in Nilearn
        Design matrix of experiment.
    noise_model : As specified in Nilearn
    bins : As specified in Nilearn
    n_jobs : As specified in Nilearn
    verbose : As specified in Nilearn

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
