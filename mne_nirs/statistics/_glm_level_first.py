# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

from copy import deepcopy

import pandas as pd
import numpy as np
from numpy import array_equal, where

import nilearn.glm
from nilearn.glm.first_level import run_glm as nilearn_glm

from mne.channels.channels import ContainsMixin
from mne.utils import fill_doc, warn, verbose
from mne.io.pick import _picks_to_idx
from mne.io.constants import FIFF

from ..visualisation._plot_GLM_topo import _plot_glm_topo,\
    _plot_glm_contrast_topo
from ..visualisation import plot_glm_surface_projection
from ..statistics._roi import _glm_region_of_interest


@fill_doc
class _BaseGLM(ContainsMixin):
    """Base GLM results class."""

    @property
    def ch_names(self):
        """Return the channel names.

        Returns
        -------
        names : array
            The channel names.
        """
        return self.info['ch_names']

    def __str__(self):
        return (f"GLM Results for {len(self.ch_names)} channels")

    def __repr__(self):
        return (f"GLM Results for {len(self.ch_names)} channels")

    def __len__(self):
        return len(self.info.ch_names)

    def copy(self):
        """Return a copy of the GLM results.

        Returns
        -------
        inst : instance of ResultsGLM
            A copy of the object.
        """
        return deepcopy(self)

    def to_dataframe(self, order=None):
        """Return a tidy dataframe representing the GLM results.

        Parameters
        ----------
        order : list
            Order in which the rows should be returned by channel name.

        Returns
        -------
        tidy : pandas.DataFrame
            Dataframe containing GLM results.
        """
        from ..utils import glm_to_tidy
        return glm_to_tidy(self.info, self._data, self.design, order=order)

    def scatter(self, conditions=[], exclude_no_interest=True, axes=None,
                no_interest=None):
        """Scatter plot of the GLM results.

        Parameters
        ----------
        conditions : list
            List of condition names to plot. By default plots all regressors
            of interest.
        exclude_no_interest : bool
            Exclude regressors of no interest from the figure.
        axes : Axes
            Optional axes on which to plot the data.
        no_interest : list
            List of regressors that are of no interest. If none are specified
            then conditions starting with
            ["drift", "constant", "short", "Short"] will be excluded.

        Returns
        -------
        plt : matplotlib.Figure
            Scatter plot.
        """
        if no_interest is None:
            no_interest = ["drift", "constant", "short", "Short"]
        import matplotlib.pyplot as plt
        df = self.to_dataframe()

        x_column = "Condition"
        y_column = "theta"
        if "ContrastType" in df.columns:
            x_column = "ch_name"
            y_column = "effect"
            if len(conditions) == 0:
                conditions = ["t", "f"]
            df = df.query('ContrastType in @conditions')

        else:
            if len(conditions) == 0:
                conditions = self.design.columns
            df = df.query('Condition in @conditions')

        if exclude_no_interest:
            for no_i in no_interest:
                df = df[~df[x_column].astype(str).str.startswith(no_i)]

        df_hbo = df.query('Chroma == "hbo"')
        df_hbr = df.query('Chroma == "hbr"')

        if axes is None:
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))

        axes.scatter(df_hbo[x_column], df_hbo[y_column] * 1e6, c="r")
        axes.scatter(df_hbr[x_column], df_hbr[y_column] * 1e6, c="b")
        axes.set_xlabel(x_column)
        axes.set_ylabel(y_column.capitalize())
        axes.legend(["Oxyhaemoglobin", "Deoxyhaemoglobin"])
        axes.hlines([0.0], 0, len(np.unique(df[x_column])) - 1)
        if len(np.unique(df[x_column])) > 8:
            plt.xticks(rotation=45, ha='right')

        return axes


@fill_doc
class RegressionResults(_BaseGLM):
    """
    Class containing GLM regression results.

    Parameters
    ----------
    info : mne.Info
        Info.
    data : dict
        Dictionary.
    design : dataframe
        Design matrix.

    Returns
    -------
    glm_est : ResultsGLM,
        Result class.
    """

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        if type(data) is not dict:
            raise TypeError("Data must be a dictionary type")
        if not array_equal(list(data.keys()), self.info.ch_names):
            raise TypeError("Dictionary keys must match ch_names")
        for idx in range(len(self.info.ch_names)):
            if list(data.keys())[idx] is not self.info.ch_names[idx]:
                raise TypeError("Data names and channel names do not match")
        for d in data:
            if type(data[d]) is not nilearn.glm.regression.RegressionResults:
                raise TypeError("Dictionary items must be"
                                " nilearn RegressionResults")

        self._data = data

    def __init__(self, info, data, design):  # noqa: D102
        self.info = info
        self.data = data
        self.design = design
        self.preload = True

    @fill_doc
    def pick(self, picks, exclude=()):
        """Pick a subset of channels.

        Parameters
        ----------
        %(picks_all)s
        exclude : list | str
            Set of channels to exclude, only used when picking based on
            types (e.g., exclude="bads" when picks="meg").

        Returns
        -------
        inst : instance of ResultsGLM
            The modified instance.
        """
        picks = _picks_to_idx(self.info, picks, 'all', exclude,
                              allow_empty=False)
        picks = [self.ch_names[p] for p in picks]
        self.info.pick_channels(picks)
        self._data = {key: self._data[key] for key in self.info.ch_names}
        return self

    def theta(self):
        """Return the GLM theta results.

        Returns
        -------
        thetas : array
            Array of thetas. An array is provided per channel.
        """
        return [self._data[key].theta for key in self.info.ch_names]

    def MSE(self):
        """Return the GLM MSE.

        Returns
        -------
        thetas : array
            Array of MSEs. A value is provided per channel.
        """
        return [self._data[key].MSE[0] for key in self.info.ch_names]

    def model(self):
        """Return the GLM model.

        Returns
        -------
        models : array
            Array of models. An model is provided per channel.
        """
        return [self._data[key].model for key in self.info.ch_names]

    def compute_contrast(self, contrast, contrast_type=None):
        """
        Compute contrasts on regression results.

        This is a wrapper function for nilearn.stats.contrasts.

        Parameters
        ----------
        contrast : numpy.ndarray of shape (p) or (q, p),
            Where q = number of contrast vectors and p = number of regressors.
        contrast_type : {None, ‘t’, ‘F’}, optional
            Type of the contrast. If None, then defaults to ‘t’ for 1D con_val
            and ‘F’ for 2D con_val.

        Returns
        -------
        contrast : Contrast instance,
            Yields the statistics of the contrast
            (effects, variance, p-values).
        """
        cont = _compute_contrast(self._data, contrast,
                                 contrast_type=contrast_type)
        return ContrastResults(self.info, cont, self.design)

    def plot_topo(self, conditions=None,
                  axes=None, vmin=None, vmax=None, colorbar=True,
                  figsize=(12, 7), sphere=None):
        """Plot 2D topography of GLM data.

        Parameters
        ----------
        conditions : array
            Which conditions should be displayed.
        axes : instance of Axes | None
            The axes to plot to. If None, a new figure is used.
        vmin : float | None
            The value specifying the lower bound of the color range.
            If None, and vmax is None, -vmax is used. Else np.min(data).
            Defaults to None.
        vmax : float | None
            The value specifying the upper bound of the color range.
            If None, the maximum absolute value is used. Defaults to None.
        colorbar : Bool
            Should a colorbar be plotted.
        figsize : two values
            Figure size.
        sphere : As specified in MNE
            Sphere parameter from mne.viz.topomap.plot_topomap.

        Returns
        -------
        fig : figure
            Figure of each design matrix componenent for hbo (top row)
            and hbr (bottom row).
        """
        return _plot_glm_topo(self.info, self._data, self.design,
                              requested_conditions=conditions,
                              axes=axes, vmin=vmin, vmax=vmax,
                              colorbar=colorbar,
                              figsize=figsize, sphere=sphere)

    def to_dataframe_region_of_interest(self, group_by, condition,
                                        weighted=True, demographic_info=False):
        """Region of interest results as a dataframe.

        Parameters
        ----------
        group_by : dict
            Specifies which channels are aggregated into a single ROI.
            The dict key will be used as the ROI label and the dict
            values must be lists of picks (either channel names or
            integer indices of ``epochs.ch_names``). For example::

                group_by=dict(Left_ROI=[1, 2, 3, 4], Right_ROI=[5, 6, 7, 8])

            Note that within a dict entry all channels must have the same type.
        condition : str | list
            Name to be used for condition.
        weighted : Bool
            Should channels be weighted by inverse of standard error.
        demographic_info : Bool
            Add an extra column with demographic information from
            info["subject_info"].

        Returns
        -------
        tidy : pandas.DataFrame
            Dataframe containing GLM results.
        """
        if isinstance(condition, str):
            condition = [condition]

        tidy = pd.DataFrame()
        for cond in condition:
            cond_idx = where([c == cond for c in self.design.columns])[0]

            roi = _glm_region_of_interest(self._data, group_by,
                                          cond_idx, cond, weighted)
            tidy = tidy.append(roi)

        if demographic_info:
            tidy['Age'] = float(self.info["subject_info"]['age'])
            if self.info["subject_info"]['sex'] == FIFF.FIFFV_SUBJ_SEX_MALE:
                sex = "male"
            elif self.info["subject_info"]['sex'] == \
                    FIFF.FIFFV_SUBJ_SEX_FEMALE:
                sex = "female"
            else:
                sex = "unknown"
            tidy['Sex'] = sex
            tidy['Hand'] = self.info["subject_info"]['hand']

        return tidy

    @verbose
    def surface_projection(self, chroma="hbo", condition=None,
                           background='w', figure=None, clim='auto',
                           mode='weighted', colormap='RdBu_r',
                           surface='pial', hemi='both', size=800,
                           view=None, colorbar=True, distance=0.03,
                           subjects_dir=None, src=None, verbose=False):
        """
        Project GLM results on to the surface of the brain.

        Note: This function provides a convenient wrapper around low level
        MNE-Python functions. It is convenient if you wish to use a generic
        head model.
        If you have acquired fMRI images you may wish to use the underlying
        lower level functions.

        Note: This function does not conduct a forward model analysis with
        photon migration etc. It simply projects the values from each channel
        to the local cortical surface.
        It is useful for visualisation, but users should
        take this in to consideration when drawing conclusions from the
        visualisation.

        Parameters
        ----------
        chroma : str
            Chromophore to plot`.
        condition : str
            Condition to plot`.
        background : matplotlib color
            Color of the background of the display window.
        figure : mayavi.core.api.Scene, matplotlib.figure.Figure, list, None
            If None, a new figure will be created. If multiple views or a
            split view is requested, this must be a list of the appropriate
            length. If int is provided it will be used to identify the Mayavi
            figure by it's id or create a new figure with the given id. If an
            instance of matplotlib figure, mpl backend is used for plotting.
        %(clim)s
        mode : str
            Can be "sum" to do a linear sum of weights, "weighted" to make this
            a weighted sum, "nearest" to
            use only the weight of the nearest sensor, or "single" to
            do a distance-weight of the nearest sensor. Default is "sum".
        colormap : str
            Colormap to use.
        surface : str
            The type of surface (inflated, white etc.).
        hemi : str
            Hemisphere id (ie 'lh', 'rh', 'both', or 'split'). In the case
            of 'both', both hemispheres are shown in the same window.
            In the case of 'split' hemispheres are displayed side-by-side
            in different viewing panes.
        size : float or tuple of float
            The size of the window, in pixels. can be one number to specify
            a square window, or the (width, height) of a rectangular window.
            Has no effect with mpl backend.
        view : str
            View to set brain to.
        colorbar : bool
            If True, display colorbar on scene.
        distance : float
            Distance (m) defining the activation "ball" of the sensor.
        %(subjects_dir)s
        src : instance of SourceSpaces
            The source space.
        %(verbose)s

        Returns
        -------
        figure : instance of mne.viz.Brain | matplotlib.figure.Figure
            An instance of :class:`mne.viz.Brain` or matplotlib figure.
        """

        df = self.to_dataframe(order=self.ch_names)
        if condition is None:
            warn("You must provide a condition to plot", ValueError)
        df = df.query("Condition in @condition")
        df = df.query("Chroma in @chroma")
        df["theta"] = df["theta"] * 1e6

        info = self.copy().pick(chroma).info

        return plot_glm_surface_projection(info, df, value="theta",
                                           background=background,
                                           figure=figure, clim=clim,
                                           mode=mode, colormap=colormap,
                                           surface=surface, hemi=hemi,
                                           size=size,
                                           view=view, colorbar=colorbar,
                                           distance=distance,
                                           subjects_dir=subjects_dir, src=src,
                                           verbose=verbose
                                           )


@fill_doc
class ContrastResults(_BaseGLM):
    """
    Class containing GLM contrast results.

    Parameters
    ----------
    info : mne.Info
        Info.
    data : dict
        Dictionary.
    design : dataframe
        Design matrix.

    Returns
    -------
    glm_est : ResultsGLM,
        Result class.
    """

    def __init__(self, info, data, design):  # noqa: D102
        self.info = info
        self.data = data
        self.design = design
        self.preload = True

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        if not isinstance(data, nilearn.glm.contrasts.Contrast):
            raise TypeError("Data must be a nilearn glm contrast type")
        if data.effect.size != len(self.info.ch_names):
            raise TypeError("Data results must be the same length "
                            "as the number of channels")

        self._data = data

    def plot_topo(self, figsize=(12, 7), sphere=None):
        """
        Plot topomap GLM contrast data.

        Parameters
        ----------
        figsize : numbers
            TODO: Remove this, how does MNE usually deal with this.
        sphere : numbers
            As specified in MNE.

        Returns
        -------
        fig : figure
            Figure of each design matrix componenent for hbo (top row)
            and hbr (bottom row).
        """
        return _plot_glm_contrast_topo(self.info, self._data,
                                       figsize=figsize, sphere=sphere)


def run_GLM(raw, design_matrix, noise_model='ar1', bins=0,
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
    noise_model : {'ar1', 'ols', 'arN', 'auto'}, optional
        The temporal variance model. Defaults to first order
        auto regressive model 'ar1'.
        The AR model can be set to any integer value by modifying the value
        of N. E.g. use `ar5` for a fifth order model.
        If the string `auto` is provided a model with order 4 times the sample
        rate will be used.
    bins : int, optional
        Maximum number of discrete bins for the AR coef histogram/clustering.
        By default the value is 0, which will set the number of bins to the
        number of channels, effectively estimating the AR model for each
        channel.
    n_jobs : int, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs'.
    verbose : int, optional
        The verbosity level. Default is 0.
    Returns
    -------
    glm_estimates : dict
        Keys correspond to the different labels values values are
        RegressionResults instances corresponding to the voxels.
    """
    warn('"run_GLM" has been deprecated in favor of the more '
         'comprehensive run_glm function, and will be removed in v1.0.0. '
         'See the changelog for further details.',
         DeprecationWarning)
    res = run_glm(raw, design_matrix, noise_model=noise_model, bins=bins,
                  n_jobs=n_jobs, verbose=verbose)
    return res.data


def run_glm(raw, design_matrix, noise_model='ar1', bins=0,
            n_jobs=1, verbose=0):
    """
    GLM fit for an MNE structure containing fNIRS data.

    This is a wrapper function for nilearn.stats.first_level_model.run_glm.

    Parameters
    ----------
    raw : instance of Raw
        The haemoglobin data.
    design_matrix : as specified in Nilearn
        The design matrix.
    noise_model : {'ar1', 'ols', 'arN', 'auto'}, optional
        The temporal variance model. Defaults to first order
        auto regressive model 'ar1'.
        The AR model can be set to any integer value by modifying the value
        of N. E.g. use `ar5` for a fifth order model.
        If the string `auto` is provided a model with order 4 times the sample
        rate will be used.
    bins : int, optional
        Maximum number of discrete bins for the AR coef histogram/clustering.
        By default the value is 0, which will set the number of bins to the
        number of channels, effectively estimating the AR model for each
        channel.
    n_jobs : int, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs'.
    verbose : int, optional
        The verbosity level. Default is 0.

    Returns
    -------
    glm_estimates : RegressionResults
        RegressionResults class which stores the GLM results.
    """
    picks = _picks_to_idx(raw.info, 'fnirs', exclude=[], allow_empty=True)
    ch_names = raw.ch_names

    if noise_model == 'auto':
        noise_model = f"ar{int(np.round(raw.info['sfreq'] * 4))}"

    if bins == 0:
        bins = len(raw.ch_names)

    results = dict()
    for pick in picks:
        labels, glm_estimates = nilearn_glm(raw.get_data(pick).T,
                                            design_matrix.values,
                                            noise_model=noise_model, bins=bins,
                                            n_jobs=n_jobs, verbose=verbose)
        results[ch_names[pick]] = glm_estimates[labels[0]]

    return RegressionResults(raw.info, results, design_matrix)


def compute_contrast(glm_est, contrast, contrast_type=None):
    """
    Compute contrasts on regression results.

    This is a wrapper function for nilearn.stats.contrasts.

    Parameters
    ----------
    glm_est : dict
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
    warn('"compute_contrast" has been deprecated in favor of the more '
         'comprehensive GLM class and will be removed in v1.0.0. '
         'Use the ResultsGLM class "compute_contrast()" method instead.',
         DeprecationWarning)
    return _compute_contrast(glm_est, contrast, contrast_type=contrast_type)


def _compute_contrast(glm_est, contrast, contrast_type=None):
    from nilearn.glm.contrasts import compute_contrast as _cc
    return _cc(np.array(list(glm_est.keys())), glm_est, contrast,
               contrast_type=contrast_type)
