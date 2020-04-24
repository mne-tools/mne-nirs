"""
.. _tut-fnirs-freq:

NIRS Frequency and Filter Commentary
====================================

In this example we discuss frequency and filters in the context
of NIRS analysis.
We examine the interplay between the expected brain response based
on experimental design and our model of how the brain reacts to stimuli,
the actual data measured during an experiment, and the filtering
that is applied to the data.

.. contents:: Page contents
   :local:
   :depth: 2

"""

# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import os
import mne
import numpy as np
from mne_nirs.experimental_design import create_first_level_design_matrix


###############################################################################
# Import and preprocess data
# --------------------------
#
# This code is similar to the first sections in the MNE tutorial,
# so will not be described in detail here.
# We read in the data, annotate the triggers, remove the control condition,
# convert to haemoglobin concentration. See
# https://mne.tools/dev/auto_tutorials/preprocessing/plot_70_fnirs_processing.html#

fnirs_data_folder = mne.datasets.fnirs_motor.data_path()
fnirs_raw_dir = os.path.join(fnirs_data_folder, 'Participant-1')
raw_intensity = mne.io.read_raw_nirx(fnirs_raw_dir,
                                     verbose=True).load_data()
new_des = [des for des in raw_intensity.annotations.description]
new_des = ['Control' if x == "1.0" else x for x in new_des]
new_des = ['Tapping/Left' if x == "2.0" else x for x in new_des]
new_des = ['Tapping/Right' if x == "3.0" else x for x in new_des]
annot = mne.Annotations(raw_intensity.annotations.onset,
                        raw_intensity.annotations.duration * 5., new_des)
raw_intensity.set_annotations(annot)
raw_intensity.annotations.crop(60, 2967)
raw_intensity.annotations.delete(
    np.where([d == 'Control' for d in raw_intensity.annotations.description]))

raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od)


###############################################################################
# Extract expected HRF from data
# ------------------------------
#
# First we extract the expected HRF function from
# the data. See :ref:`_tut-fnirs-hrf` for more details on this analysis.

design_matrix = create_first_level_design_matrix(raw_haemo,
                                                 hrf_model='spm', stim_dur=5.0,
                                                 drift_order=0,
                                                 drift_model='polynomial')


# This is a bit of a hack.
# Overwrite the first NIRS channel with the expected response.
# Rescale to be in expected units of uM.
hrf = raw_haemo.copy().pick(picks=[0])
hrf._data[0] = 1e-6 * (design_matrix['Tapping/Left'] +
                       design_matrix['Tapping/Right']).T

fig = hrf.pick(picks='hbo').plot_psd(average=True, fmax=2,
                                     color='r', show=False)


###############################################################################
# Plot raw measured data
# ----------------------
#
# Next we plot the PSD of the raw data.
# Here we rescale the data to fit in the figure.
#
# TODO: Find a nice way to show this data with correct scale, perhaps a left
# y axis scale.

raw_haemo._data = raw_haemo._data * 1e-2
fig = raw_haemo.pick(picks='hbo').plot_psd(average=True, fmax=2,
                                           ax=fig.axes, show=False)


###############################################################################
# Plot epoched data
# -----------------
#
# Next we plot the PSD of the epoched data.


events, _ = mne.events_from_annotations(raw_haemo)
event_dict = {'Tapping/Left': 1, 'Tapping/Right': 2}
reject_criteria = dict(hbo=120e-6)
tmin, tmax = -5, 15
epochs = mne.Epochs(raw_haemo, events, event_id=event_dict,
                    tmin=tmin, tmax=tmax,
                    reject=reject_criteria, reject_by_annotation=True,
                    proj=True, baseline=(None, 0), preload=True,
                    detrend=None, verbose=True)
fig = epochs.pick(picks='hbo').plot_psd(average=True, fmax=2, ax=fig.axes,
                                        show=False, color='g')


###############################################################################
# Plot filter response
# --------------------
#
# Next we plot the filter response.

filter_params = mne.filter.create_filter(
    raw_haemo.get_data(), raw_haemo.info['sfreq'],
    l_freq=0.01, h_freq=0.4,
    h_trans_bandwidth=0.2, l_trans_bandwidth=0.005)
fig = mne.viz.plot_filter(filter_params, raw_haemo.info['sfreq'],
                          flim=(0.005, 2), fscale='log', gain=False,
                          plot='magnitude', axes=fig.axes, show=False)


###############################################################################
# Discussion
# ----------
#
# Next we plot the filter response.

leg_lines = [line for line in fig.axes[0].lines if line.get_linestyle() == '-']
fig.legend(leg_lines, ['Theoretical HRF', 'Measured Data',
                       'Epoched Data', 'Filter Response'])
fig.axes[0].set_ylabel('Filter Magnitude (dB) [invalid for other lines]')
fig.axes[0].set_title('')

fig.show()
