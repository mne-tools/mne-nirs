"""
.. _tut-fnirs-decoding:

Decoding Analysis
=================

This is an example of a decoding analysis performed on
functional near-infrared spectroscopy (fNIRS) data using
MNE-Python, scikit-learn, and MNE-NIRS.

Detailed information about decoding of neural signals can be found
in the MNE-Python documentation. For example see
`Decoding (MVPA) <https://mne.tools/stable/auto_examples/decoding/decoding_csp_eeg.html>`_,
:ref:`Linear classifier on sensor data  <mne:ex-linear-patterns>`,
:ref:`Decoding source space data <mne:ex-dec-st-source>`.
This example will use the techniques covered in the MNE-Python tutorials,
but applied specifically to fNIRS data.

This script is an example of analysis performed in the manuscript
Luke et. al. (2021)
:footcite:`Luke2021.11.19.469225`.

.. note::

   This tutorial uses data in the BIDS format.
   The BIDS specification for NIRS data is still under development. See:
   `fNIRS BIDS proposal <https://github.com/bids-standard/bids-specification/pull/802>`_.
   As such, to run this tutorial you must use the MNE-BIDS 0.10 or later.

   MNE-Python allows you to process fNIRS data that is not in BIDS format too.
   Simply modify the ``read_raw_`` function to match your data type.
   See :ref:`data importing tutorial <tut-importing-fnirs-data>` to learn how
   to use your data with MNE-Python.


.. contents:: Page contents
   :local:
   :depth: 2
"""

# Authors: Robert Luke <mail@robertluke.net>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD (3-clause)


# Import common libraries
import os
import contextlib
import numpy as np

# Import sklearn processing
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

# Import MNE-Python processing
from mne.preprocessing.nirs import optical_density, beer_lambert_law
from mne import Epochs, events_from_annotations
from mne.decoding import (Scaler,
                          cross_val_multiscore,
                          Vectorizer)

# Import MNE-NIRS processing
from mne_nirs.datasets.fnirs_motor_group import data_path

# Import MNE-BIDS processing
from mne_bids import BIDSPath, read_raw_bids, get_entity_vals


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

root = data_path()
dataset = BIDSPath(root=root, suffix="nirs", extension=".snirf",
                   task="tapping", datatype="nirs")
subjects = get_entity_vals(root, 'subject')


# %%
# Define individual analysis
# --------------------------
#
# More details on the epoching analysis can be found
# at :ref:`Waveform individual analysis <tut-fnirs-processing>`.
# A minimal processing pipeline is demonstrated here, as the focus
# of this tutorial is to demonstrate the decoding pipeline.
# In this example only the epochs for the two conditions we wish to decode
# between are retained.


def epoch_preprocessing(bids_path):

    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        raw_intensity = read_raw_bids(bids_path=bids_path).load_data()

    raw_od = optical_density(raw_intensity)
    raw_od.resample(1.5)
    raw_haemo = beer_lambert_law(raw_od, ppf=6)
    raw_haemo = raw_haemo.filter(None, 0.6, h_trans_bandwidth=0.05,
                                 l_trans_bandwidth=0.01, verbose=False)

    events, event_dict = events_from_annotations(raw_haemo, verbose=False)
    epochs = Epochs(raw_haemo, events, event_id=event_dict, tmin=-5, tmax=30,
                    reject=dict(hbo=100e-6), reject_by_annotation=True,
                    proj=True, baseline=(None, 0), detrend=1,
                    preload=True, verbose=False)

    epochs = epochs[["Tapping/Right", "Tapping/Left"]]
    return raw_haemo, epochs


# %%
# Run analysis on all participants
# --------------------------------
#
# Next we loop through each measurement and decode between the control and
# audio condition.
# Here we compute a single spatio-temporal metric approach that simultaneously
# uses all channels and time points to estimate the experimental condition.
# The data is scaled for each channel by the mean and standard deviation
# from all time points and epochs, after which they were vectorized to
# comply with the scikit-learn data structure, and a logistic regression
# classifier was applied using the liblinear solver.
# This approach classifies the data within, rather than across, subjects.


for chroma in ['hbo', 'hbr']:

    st_scores = []
    for sub in subjects:

        bids_path = dataset.update(subject=sub)
        raw_haemo, epochs = epoch_preprocessing(bids_path)

        epochs.pick(chroma)

        X = epochs.get_data()
        y = epochs.events[:, 2]

        clf = make_pipeline(Scaler(epochs.info),
                            Vectorizer(),
                            LogisticRegression(solver='liblinear'))

        scores = 100 * cross_val_multiscore(clf, X, y,
                                            cv=5, n_jobs=1, scoring='roc_auc')

        st_scores.append(np.mean(scores, axis=0))

    print(f"Average spatio-temporal ROC-AUC performance ({chroma}) = "
          f"{np.round(np.mean(st_scores))} % ({np.round(np.std(st_scores))})")


# %%
# Conclusion
# ----------
#
# Data were epoched then decoding was performed on the hbo signal and the hbr
# signal. The HbO signal decodes the conditions with 6% greater accuracy
# than the HbR signal. For further discussion about the efficacy of fNIRS
# signals in decoding experimental condition see Luke et. al. (2021)
# :footcite:`Luke2021.11.19.469225`.


# %%
# Bibliography
# -----------------------------------------------
#
# .. footbibliography::
