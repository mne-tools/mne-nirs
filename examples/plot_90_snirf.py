"""
.. _tut-fnirs-snirf:

SNIRF Handling With MNE
=======================

SNIRF is a file format for storing NIRS data. The protocol is produced
by the society for functional near infrared spectroscopy.

Read the protocol over at https://github.com/fNIRS/snirf/blob/master/snirf_specification.md

.. contents:: Page contents
   :local:
   :depth: 2

"""


# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mne
import mne_nirs

from mne_nirs.experimental_design import make_first_level_design_matrix
from mne_nirs.statistics import run_GLM, glm_region_of_interest
from mne_nirs.visualisation import plot_glm_topo
from mne_nirs.channels import (get_long_channels, get_short_channels,
                               picks_pair_to_idx)

from nilearn.reporting import plot_design_matrix
from mne_nirs.utils._io import glm_to_tidy, _tidy_long_to_wide


###############################################################################
# Import raw NIRS data from vendor
# --------------------------------
#
# First we import the motor tapping data, these data are also
# described and used in the
# `MNE fNIRS tutorial <https://mne.tools/stable/auto_tutorials/preprocessing/plot_70_fnirs_processing.html>`_.
#
# After reading the data we resample down to 1Hz
# to meet github memory constraints.
#
# .. collapse:: Data description (click to expand)
#    :class: success
#
#    Optodes were placed over the motor cortex using the standard NIRX motor
#    montage, but with 8 short channels added (see their web page for details).
#    To view the sensor locations run
#    `raw_intensity.plot_sensors()`.
#    A sound was presented to indicate which hand the participant should tap.
#    Participants taped their thumb to fingers for 5s.
#    Conditions were presented in a random order with a randomised inter
#    stimulus interval.

fnirs_data_folder = mne.datasets.fnirs_motor.data_path()
fnirs_raw_dir = os.path.join(fnirs_data_folder, 'Participant-1')
raw_intensity = mne.io.read_raw_nirx(fnirs_raw_dir).load_data()
raw_intensity.resample(1.0)


###############################################################################
# Write data as SNIRF
# -------------------
#
# Next we update the annotations by assigning names to each trigger ID.
# Then we crop the recording to the section containing our
# experimental conditions.

original_annotations = raw_intensity.annotations
new_des = [des for des in raw_intensity.annotations.description]
new_des = ['Control' if x == "1.0" else x for x in new_des]
new_des = ['Tapping/Left' if x == "2.0" else x for x in new_des]
new_des = ['Tapping/Right' if x == "3.0" else x for x in new_des]
keepers = [n == 'Control' or
           n == "Tapping/Left" or
           n == "Tapping/Right" for n in new_des]
idxs = np.array(np.where(keepers)[0])
annot = mne.Annotations(original_annotations.onset[idxs],
                        original_annotations.duration[idxs] * 5., 
                        np.array([new_des[idx] for idx in np.where(keepers)[0]]))
raw_intensity.set_annotations(annot)


###############################################################################
# Read back SNIRF file
# --------------------
# Next we convert the raw data to haemoglobin concentration.

raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od)


###############################################################################
#
# .. sidebar:: Relevant literature
#
#    Tachtsidis, Ilias, and Felix Scholkmann. "False positives and false
#    negatives in functional near-infrared spectroscopy: issues, challenges,
#    and the way forward." Neurophotonics 3.3 (2016): 031405.
#
# We then split the data in to
# short channels which predominantly contain systemic responses and
# long channels which have both neural and systemic contributions.

short_chs = get_short_channels(raw_haemo)
raw_haemo = get_long_channels(raw_haemo)


###############################################################################
# Compare original and SNIRF
# --------------------------
#
# Next we examine the timing and order of events in this experiment.
# There are several options for how to view event information.
# The first option is to use MNE's plot events command.
# Here each dot represents when an event started.
# We observe that the order of conditions was randomised and the time between
# events is also randomised.

events, _ = mne.events_from_annotations(raw_haemo, verbose=False)
event_dict = {'Control': 1, 'Tapping/Left': 2, 'Tapping/Right': 3}
mne.viz.plot_events(events, event_id=event_dict,
                    sfreq=raw_haemo.info['sfreq'])

