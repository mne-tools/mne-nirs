"""
.. _tut-fnirs-decoding:

Decoding Analysis
=================

This is an example of a decoding analysis performed on
functional near-infrared spectroscopy (fNIRS) data using
MNE-Python, SK-Learn, and MNE-NIRS.

.. note::

   This tutorial uses data in the BIDS format.
   The BIDS specification for NIRS data is still under development. See:
   `fNIRS BIDS proposal <https://github.com/bids-standard/bids-specification/pull/802>`_.
   As such, to run this tutorial you must use the fNIRS development branch of MNE-BIDS.

   To install the fNIRS development branch of MNE-BIDS run:
   `pip install -U https://codeload.github.com/rob-luke/mne-bids/zip/nirs`.

   MNE-Python. allows you to process fNIRS data that is not in BIDS format too.
   Simply modify the ``read_raw_`` function to match your data type.
   See :ref:`data importing tutorial <tut-importing-fnirs-data>` to learn how
   to use your data with MNE-Python.

.. note::

   The data in this example is agressively downsampled and filtered to allow
   the tutorial to be run on the cloud server. You may wish to use a higher
   sample rate and cut-off filter frequency in your own studies.

.. contents:: Page contents
   :local:
   :depth: 2
"""
# sphinx_gallery_thumbnail_number = 2

# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)


# Import common libraries
import numpy as np
import pandas as pd
import os

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from mne.decoding import (SlidingEstimator, Scaler,
                          cross_val_multiscore, LinearModel,
                          Vectorizer, get_coef)

import contextlib
import mne_nirs

# Import MNE processing
from mne.preprocessing.nirs import optical_density, beer_lambert_law
from mne import Epochs, events_from_annotations

# Import MNE-NIRS processing
from mne_nirs.statistics import run_glm
from mne_nirs.experimental_design import make_first_level_design_matrix
from mne_nirs.statistics import statsmodels_to_results
from mne_nirs.channels import get_short_channels, get_long_channels
from mne_nirs.channels import picks_pair_to_idx
from mne_nirs.visualisation import plot_glm_group_topo
from mne_nirs.datasets import fnirs_motor_group
from mne_nirs.visualisation import plot_glm_surface_projection
from mne_nirs.io.fold import fold_landmark_specificity, fold_channel_specificity

# Import MNE-BIDS processing
from mne_bids import BIDSPath, read_raw_bids, get_entity_vals

# Import StatsModels
import statsmodels.formula.api as smf

# Import Plotting Library
import matplotlib.pyplot as plt
import matplotlib as mpl
from lets_plot import *
LetsPlot.setup_html()


# %%
# Set up directories
# ------------------
# .. sidebar:: Requires MNE-BIDS fNIRS branch
#
#    This section of code requires the MNE-BIDS fNIRS branch.
#    See instructions at the top of the page on how to install.
#    Alternatively, if your data is not in BIDS format,
#    skip to the next section.
#
# First we will define where the raw data is stored. We will analyse a
# BIDS dataset. This ensures we have all the metadata we require
# without manually specifying the trigger names etc.
# We first define where the root directory of our dataset is.
# In this example we use the example dataset ``audio_or_visual_speech``.

root = mne_nirs.datasets.audio_or_visual_speech.data_path()

dataset = BIDSPath(root=root, task="AudioVisualBroadVsRestricted",
                   session = "01",
                   datatype="nirs", suffix="nirs", extension=".snirf")
subjects = get_entity_vals(root, 'subject')


# %%
# Define individual analysis
# --------------------------
#
# More details on the epoching analysis can be found
# at :ref:`Waveform individual analysis <tut-fnirs-processing>`


def epoch_preprocessing(bids_path):

    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        raw_intensity = read_raw_bids(bids_path=bids_path, verbose=False).load_data()

    raw_od = optical_density(raw_intensity)

    # Agressive downsampling is performed here to enable this to run on
    # the cloud servers. You may wish to use a higher value in real studies
    # and then modify the filter cut off frequencies accordingly below.
    raw_od.resample(1.5)

    raw_haemo = beer_lambert_law(raw_od, ppf=6)
    raw_haemo = raw_haemo.filter(None, 0.6,
                                 h_trans_bandwidth=0.05, l_trans_bandwidth=0.01,
                                 verbose=False)

    events, event_dict = events_from_annotations(raw_haemo, verbose=False)
    epochs = Epochs(raw_haemo, events, event_id=event_dict, tmin=-5, tmax=30,
                    reject=dict(hbo=100e-6), reject_by_annotation=True,
                    proj=True, baseline=(None, 0), detrend=1,
                    preload=True, verbose=False)

    epochs = epochs[["Control", "Audio"]]
    return raw_haemo, epochs


# %%
# Run analysis on all participants
# --------------------------------
#
# Next we loop through the five measurements and run the individual analysis
# on each. We append the individual results in to a large dataframe that
# will contain the results from all measurements. We create a group dataframe
# for the region of interest, channel level, and contrast results.

st_scores = []

for chroma in ['hbo', 'hbr']:
    for sub in subjects:

        bids_path = dataset.update(subject=sub)
        raw_haemo, epochs = epoch_preprocessing(bids_path)

        epochs.pick(chroma)

        X = epochs.get_data()
        y = epochs.events[:, 2]

        clf = make_pipeline(Scaler(epochs.info),
                            Vectorizer(),
                            LogisticRegression(solver='liblinear'))

        scores = 100 *  cross_val_multiscore(clf, X, y, cv=5, n_jobs=1, scoring='roc_auc')

        score = np.mean(scores, axis=0)
        score_std = np.std(scores, axis=0)
        st_scores.append(score)
        # print(f"Subject: {sub} mean ROC-AUC = {np.round(score, 2)} % ({np.round(score_std, 2)})")

    print(f"Average spatio-temporal ROC-AUC performance ({chroma}) = "
          f"{np.round(np.mean(st_scores))} % ({np.round(np.std(st_scores))})")
