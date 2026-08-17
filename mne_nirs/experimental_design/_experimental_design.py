# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import mne
import numpy as np
from mne.utils import logger, verbose

# Regressors above this VIF are reported to the user as problematic. Values
# between 1 and 5 indicate low to moderate correlation between regressors.
_VIF_THRESHOLD = 5.0


@verbose
def make_first_level_design_matrix(
    raw,
    stim_dur=1.0,
    hrf_model="glover",
    drift_model="cosine",
    high_pass=0.01,
    drift_order=1,
    fir_delays=(0,),
    add_regs=None,
    add_reg_names=None,
    min_onset=-24,
    oversampling=50,
    *,
    return_vif=False,
    verbose=None,
):
    """
    Generate a design matrix based on annotations and model HRF.

    This is a wrapper function for the nilearn :footcite:`abraham2014machine`
    function ``make_first_level_design_matrix``. For detailed description
    of the arguments see the nilearn documentation at http://nilearn.github.io

    Parameters
    ----------
    raw : instance of Raw
        Haemoglobin data.

    stim_dur : Number
        The length of your stimulus.

    hrf_model : {'glover', 'spm', 'spm + derivative', \
        'spm + derivative + dispersion',\
        'glover + derivative', 'glover + derivative + dispersion',\
        'fir', None}, optional
        Specifies the hemodynamic response function. Default='glover'.

    drift_model : {'cosine', 'polynomial', None}, optional
        Specifies the desired drift model. Default='cosine'.

    high_pass : float, optional
        High-pass frequency in case of a cosine model (in Hz).
        Default=0.01.

    drift_order : int, optional
        Order of the drift model (in case it is polynomial).
        Default=1.

    fir_delays : array of shape(n_onsets) or list, optional
        In case of FIR design, yields the array of delays used in the FIR
        model (in scans). Default=[0].

    add_regs : array of shape(n_frames, n_add_reg) or pandas DataFrame
        Additional user-supplied regressors, e.g. data driven noise regressors
        or seed based regressors.

    add_reg_names : list of (n_add_reg,) str, optional
        If None, while add_regs was provided, these will be termed
        'reg_%%i', i = 0..n_add_reg - 1
        If add_regs is a DataFrame, the corresponding column names are used
        and add_reg_names is ignored.

    min_onset : float, optional
        Minimal onset relative to frame_times[0] (in seconds)
        events that start before frame_times[0] + min_onset are not considered.
        Default=-24.

    oversampling : int, optional
        Oversampling factor used in temporal convolutions. Default=50.

    return_vif : bool, optional
        If True, also return the variance inflation factor (VIF) of each
        regressor. Default=False. Should only be passed as a keyword argument.
    %(verbose)s

    Returns
    -------
    design_matrix : DataFrame instance,
        Holding the computed design matrix, the index being the frames_times
        and each column a regressor.

    vif : instance of pandas.Series
        The VIF of each non-constant regressor, indexed by regressor name.
        Only returned if ``return_vif=True``.

    Notes
    -----
    The variance inflation factor (VIF) quantifies how much the variance of a
    regressor's coefficient is inflated by collinearity with the other
    regressors. It is defined as ``1 / (1 - R**2)``, where ``R**2`` is obtained
    by regressing one regressor on all of the others (plus an intercept), and
    is not defined for the ``"constant"`` regressor itself. Values are always
    at least 1, and a perfectly collinear regressor yields ``np.inf``.

    A VIF between 1 and 5 indicates low to moderate correlation between
    regressors. Larger values indicate high multicollinearity, which suggests
    that the offending regressors should be combined or dropped, and are
    reported in the log output regardless of ``return_vif``.

    These are the same values as
    ``statsmodels.stats.outliers_influence.variance_inflation_factor`` gives
    for a design matrix that contains an intercept column.

    References
    ----------
    .. footbibliography::
    """
    from nilearn.glm.first_level import make_first_level_design_matrix
    from pandas import DataFrame

    frame_times = raw.times

    # Create events for nilearn
    conditions = raw.annotations.description
    onsets = raw.annotations.onset - raw.first_time
    duration = stim_dur * np.ones(len(conditions))
    events = DataFrame(
        {"trial_type": conditions, "onset": onsets, "duration": duration}
    )

    dm = make_first_level_design_matrix(
        frame_times,
        events,
        drift_model=drift_model,
        drift_order=drift_order,
        hrf_model=hrf_model,
        min_onset=min_onset,
        high_pass=high_pass,
        add_regs=add_regs,
        oversampling=oversampling,
        add_reg_names=add_reg_names,
        fir_delays=fir_delays,
    )

    vif = _design_matrix_vif(dm)

    return (dm, vif) if return_vif else dm


def _design_matrix_vif(design_matrix):
    """Compute the variance inflation factor of each design matrix regressor."""
    from pandas import Series
    from scipy.linalg import cho_solve

    names = [name for name in design_matrix.columns if name != "constant"]
    # Centering the regressors is equivalent to including an intercept in each
    # of the regressions below (Frisch-Waugh-Lovell), and lets us drop the
    # "constant" column entirely. Normalizing does not change the VIF, but
    # keeps the Gram matrix below well scaled.
    data = np.asarray(design_matrix[names].values, float)
    data = data - data.mean(0)
    norms = np.linalg.norm(data, axis=0)
    n_regressors = len(names)

    # A regressor with no variance is a duplicate of the intercept, and one
    # that the others reproduce exactly has an infinite VIF by definition
    vif = np.full(n_regressors, np.inf)
    use = norms > 0
    data = data[:, use] / norms[use]

    # For centered and normalized regressors the Gram matrix is the correlation
    # matrix, and VIF is the diagonal of its inverse. This is much faster than
    # regressing each column on all of the others in turn, but is only valid
    # when the regressors are linearly independent, which the Cholesky
    # decomposition tells us.
    try:
        chol = np.linalg.cholesky(data.T @ data)
    except np.linalg.LinAlgError:  # singular, so fall back to least squares
        logger.debug("Design matrix is rank deficient, computing VIF directly")
        vif[use] = _vif_lstsq(data)
    else:
        vif[use] = np.diag(cho_solve((chol, True), np.eye(chol.shape[0])))
    # Numerically, an exactly orthogonal regressor can come out just below 1.
    # At the other end, a VIF above the resolution of the decomposition means
    # the regressor is collinear with the others to within floating point
    # error, which is as close to infinite as we can measure.
    vif = np.maximum(vif, 1.0)
    vif[vif > 1.0 / (np.finfo(float).eps * max(data.shape))] = np.inf

    order = np.argsort(vif)[::-1]  # worst first
    for ii in order:
        logger.debug(f"    VIF of {names[ii]}: {vif[ii]:0.3f}")
    bad = order[vif[order] > _VIF_THRESHOLD]
    if len(bad):
        show, extra = bad[:5], len(bad) - 5
        logger.warning(
            f"High collinearity (VIF > {_VIF_THRESHOLD:0.0f}) detected in "
            f"{len(bad)}/{n_regressors} design matrix regressors: "
            + ", ".join(f"{names[ii]} ({vif[ii]:0.3f})" for ii in show)
            + (f", and {extra} more" if extra > 0 else "")
        )
    elif n_regressors:
        worst = np.argmax(vif)
        logger.info(f"Maximum design matrix VIF was {vif[worst]:0.3f} ({names[worst]})")

    return Series(vif, index=names, name="vif")


def _vif_lstsq(data):
    """Compute VIF by regressing each column on all of the others."""
    # A residual this small relative to the total means that the regressor is
    # perfectly collinear with the others, up to floating point error
    tol = np.finfo(float).eps * max(data.shape)
    n_regressors = data.shape[1]
    vif = np.zeros(n_regressors)
    for ii in range(n_regressors):
        x_i = data[:, ii]
        x_noti = data[:, np.arange(n_regressors) != ii]
        coef = np.linalg.lstsq(x_noti, x_i, rcond=None)[0]
        rss = np.sum((x_i - x_noti @ coef) ** 2)
        tss = np.sum(x_i**2)
        # rss / tss is the unexplained fraction, and VIF is just its inverse
        vif[ii] = np.inf if rss <= tss * tol else tss / rss
    return vif


def create_boxcar(raw, event_id=None, stim_dur=1):
    """
    Generate boxcar representation of the experimental paradigm.

    Parameters
    ----------
    raw : instance of Raw
        Haemoglobin data.
    event_id : as specified in MNE
        Information about events.
    stim_dur : Number
        The length of your stimulus.

    Returns
    -------
    s : array
        Returns an array for each annotation label.
    """
    bc = np.ones(int(round(raw.info["sfreq"] * stim_dur)))
    events, ids = mne.events_from_annotations(raw, event_id=event_id)
    s = np.zeros((len(raw.times), len(ids)))
    for idx, _ in enumerate(ids):
        id_idx = [e[2] == idx + 1 for e in events]
        id_evt = events[id_idx]
        event_samples = [e[0] for e in id_evt]
        s[event_samples, idx] = 1.0
        s[:, idx] = np.convolve(s[:, idx], bc)[: len(raw.times)]
    return s


def longest_inter_annotation_interval(raw):
    """
    Compute longest ISI per annotation.

    Specifically, longest period between two trials of
    the same condition.

    Parameters
    ----------
    raw : instance of Raw
        Haemoglobin data.

    Returns
    -------
    longest : list
        Longest ISI per annotation.
    annotation_name : list
        Annotation name corresponding to reported interval.
    """
    annotation_name = np.unique(raw.annotations.description)
    longest = []
    for desc in annotation_name:
        mask = raw.annotations.description == desc
        longest.append(np.max(np.diff(raw.annotations.onset[mask])))
    return longest, annotation_name


def drift_high_pass(raw):
    """
    Compute cosine drift regressor high pass cut off.

    Value computed according to Nilearn :footcite:`abraham2014machine`
    `suggestion <http://nilearn.github.io/auto_examples/04_glm_first
    _level/plot_first_level_details.html#changing-the-drift-model>`__.

    Parameters
    ----------
    raw : instance of Raw
        Haemoglobin data.

    Returns
    -------
    cutoff : number
        Suggested high pass cut off.

    References
    ----------
    .. footbibliography::
    """
    longest, annotation_name = longest_inter_annotation_interval(raw)
    max_isi = np.max(longest)
    return 1 / (2 * max_isi)
