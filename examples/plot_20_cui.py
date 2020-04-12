"""
.. _tut-fnirs-cui:

Examine the effect of signal enhancement approaches
===================================================

.. contents:: Page contents
   :local:
   :depth: 2

"""

import os

import matplotlib.pyplot as plt

import mne
import mne_nirs


###############################################################################
# Import and preprocess data
# --------------------------
#
# This code is exactly the same as the first sections in the MNE tutorial.

fnirs_data_folder = mne.datasets.fnirs_motor.data_path()
fnirs_raw_dir = os.path.join(fnirs_data_folder, 'Participant-1')
raw_intensity = mne.io.read_raw_nirx(fnirs_raw_dir, verbose=True).load_data()
picks = mne.pick_types(raw_intensity.info, meg=False, fnirs=True)
dists = mne.preprocessing.nirs.source_detector_distances(
    raw_intensity.info, picks=picks)
raw_intensity.pick(picks[dists > 0.01])
raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od)
raw_haemo = raw_haemo.filter(0.05, 0.7, h_trans_bandwidth=0.2,
                             l_trans_bandwidth=0.02)
events, _ = mne.events_from_annotations(raw_haemo, event_id={'1.0': 1,
                                                             '2.0': 2,
                                                             '3.0': 3})
event_dict = {'Control': 1, 'Tapping/Left': 2, 'Tapping/Right': 3}


###############################################################################
# Extract epochs with no additional processing
# --------------------------------------------
#
# First we extract the epochs with no additional processing
# this result should be the same as the MNE tutorial.

reject_criteria = dict(hbo=80e-6)
tmin, tmax = -5, 15

epochs = mne.Epochs(raw_haemo, events, event_id=event_dict,
                    tmin=tmin, tmax=tmax,
                    reject=reject_criteria, reject_by_annotation=True,
                    proj=True, baseline=(None, 0), preload=True,
                    detrend=None, verbose=True)

evoked_dict = {'Tapping/HbO': epochs['Tapping'].average(picks='hbo'),
               'Tapping/HbR': epochs['Tapping'].average(picks='hbr'),
               'Control/HbO': epochs['Control'].average(picks='hbo'),
               'Control/HbR': epochs['Control'].average(picks='hbr')}

# Rename channels until the encoding of frequency in ch_name is fixed
for condition in evoked_dict:
    evoked_dict[condition].rename_channels(lambda x: x[:-4])


###############################################################################
# Apply negative correlation enhancment algorithm
# -----------------------------------------------
#
# Apply Cui et. al. 2010 and extract epochs.

raw_anti = mne_nirs.signal_enhancement.enhance_negative_correlation(raw_haemo)

epochs_anti = mne.Epochs(raw_anti, events, event_id=event_dict,
                         tmin=tmin, tmax=tmax,
                         reject=reject_criteria, reject_by_annotation=True,
                         proj=True, baseline=(None, 0), preload=True,
                         detrend=None, verbose=True)

evoked_dict_anti = {'Tapping/HbO': epochs_anti['Tapping'].average(picks='hbo'),
                    'Tapping/HbR': epochs_anti['Tapping'].average(picks='hbr'),
                    'Control/HbO': epochs_anti['Control'].average(picks='hbo'),
                    'Control/HbR': epochs_anti['Control'].average(picks='hbr')}

# Rename channels until the encoding of frequency in ch_name is fixed
for condition in evoked_dict_anti:
    evoked_dict_anti[condition].rename_channels(lambda x: x[:-4])


###############################################################################
# Plot two approaches for comparison
# ----------------------------------
#
# Plot the average epochs with and without Cui 2010 applied.

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

color_dict = dict(HbO='#AA3377', HbR='b')
styles_dict = dict(Control=dict(linestyle='dashed'))

mne.viz.plot_compare_evokeds(evoked_dict, combine="mean", ci=0.95,
                             axes=axes[0], colors=color_dict,
                             styles=styles_dict,
                             ylim=dict(hbo=[-10, 13]))

mne.viz.plot_compare_evokeds(evoked_dict_anti, combine="mean", ci=0.95,
                             axes=axes[1], colors=color_dict,
                             styles=styles_dict,
                             ylim=dict(hbo=[-10, 13]))

for column, condition in enumerate(['Original Data', 'Cui Enhanced Data']):
    axes[column].set_title('{}'.format(condition))
