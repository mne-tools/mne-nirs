"""
.. _tut-fnirs-freq:

Frequency and Filter Commentary
===============================

In this example we discuss frequency and filters in the context
of NIRS analysis.
We examine the interplay between the expected brain response based
on experimental design and our model of how the brain reacts to stimuli:,
the actual data measured during an experiment, and the filtering
that is applied to the data.

This is a work in progress. I am just getting the figures correct
before I fill in the text.

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
# https://mne.tools/dev/auto_tutorials/preprocessing/plot_70_fnirs_processing.html

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
# Model neural response
# ---------------------
#
# We know the when each stimulus was presented to the lister (see Annotations)
# and we have a model of how we expect the brain to react to each
# stimulus presentation (add wikipedia HRF link).
# From this information we can build a model of how we expect the brain
# to be active during this experiment.
# See :ref:`tut-fnirs-hrf` for more details on this analysis.
#
# Here we create the expected model neural response function using the data
# and plot the frequency spectrum.
#
# We note there is a peak at 0.03 which corresponds approximately to
# the repetition rate of the experiment.

design_matrix = create_first_level_design_matrix(raw_haemo, drift_order=0)

# This is a bit of a hack.
# Overwrite the first NIRS channel with the expected response.
# Rescale to be in expected units of uM.
hrf = raw_haemo.copy().pick(picks=[0])
hrf._data[0] = 1e-6 * (design_matrix['Tapping/Left'] +
                       design_matrix['Tapping/Right']).T

hrf.pick(picks='hbo').plot_psd(average=True, fmax=2, xscale='log',
                                     color='r', show=False)


###############################################################################
# Plot raw measured data
# ----------------------
#
# Next we plot the PSD of the raw data.
#
# Note the increased activity around 1 Hz. .... Heart rate ....
#
# Discuss Mayer waves...
#
# Discuss breathing rate...
#
# Discuss low frequency....
#
# TODO: I would like to find a nicer way to fit everything in one plot.
# Could I use a left y axis for the real data?
#
# TODO: Find a way not to have to rescale the data to fit in plot

# rescale data to fit in plot. TODO: fix this
raw_haemo._data = raw_haemo._data * 1e-2
raw_haemo.pick(picks='hbo').plot_psd(average=True, fmax=2, xscale='log')


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

epochs.pick(picks='hbo').plot_psd(average=True, fmax=2, color='g', xscale='log')


###############################################################################
# Plot filter response
# --------------------
#
# Next we plot the filter response.
#
# Basic description of filters and link to MNE tut...
#
# Discuss with specific relation to NIRS.

filter_params = mne.filter.create_filter(
    raw_haemo.get_data(), raw_haemo.info['sfreq'],
    l_freq=0.03, h_freq=0.4,
    h_trans_bandwidth=0.2, l_trans_bandwidth=0.005)

mne.viz.plot_filter(filter_params, raw_haemo.info['sfreq'],
                          flim=(0.005, 2), fscale='log', gain=False,
                          plot='magnitude')


###############################################################################
# Discussion
# ----------
#
# Finally we overlay all lines on the same figure to see how the
# different components relate.
#
# First, note that the filter cutoff above 0.4 Hz attenuates
# the unwanted heart rate component which is situated around 1 Hz.
#
# Next we observe in the measured data that the breathing rate / Mayer waves
# situated around 0.1 Hz are in the same range as
# the expected peak in the model response.
# For this reason the filter does not attenuate these frequencies.
# However, as the experiment was designed with a randomised inter
# stimulus interval the act of epoching and averaging the data
# removes these unwanted components and they are not visible in the
# epoched data.
#
# The filter attenuates low frequency components in the signal. We observe that
# the expected model response has reduced content below 0.015 Hz,
# so filtering below 0.01 should not overly affect the signal of interest.

fig = hrf.pick(picks='hbo').plot_psd(average=True, fmax=2,
                                     color='r', show=False)
raw_haemo.pick(picks='hbo').plot_psd(average=True, fmax=2,
                                           ax=fig.axes, show=False)
epochs.pick(picks='hbo').plot_psd(average=True, fmax=2, ax=fig.axes,
                                        show=False, color='g')
mne.viz.plot_filter(filter_params, raw_haemo.info['sfreq'],
                          flim=(0.003, 2), fscale='log', gain=False,
                          plot='magnitude', axes=fig.axes, show=False)

leg_lines = [line for line in fig.axes[0].lines if line.get_linestyle() == '-']
fig.legend(leg_lines, ['Model Response', 'Measured Data',
                       'Epoched Data', 'Filter Response'])
fig.axes[0].set_ylabel('Filter Magnitude (dB) [invalid for other lines]')
fig.axes[0].set_title('')
