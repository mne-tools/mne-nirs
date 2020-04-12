"""
.. _tut-fnirs-cui:

Plot experiment expected haemodynamic response
==============================================



.. contents:: Page contents
   :local:
   :depth: 2

"""

import os

import matplotlib.pyplot as plt

import mne
import mne_nirs

fnirs_data_folder = mne.datasets.fnirs_motor.data_path()
fnirs_raw_dir = os.path.join(fnirs_data_folder, 'Participant-1')
raw_intensity = mne.io.read_raw_nirx(fnirs_raw_dir, verbose=True).load_data()


###############################################################################
# Selecting channels appropriate for detecting neural responses
# -------------------------------------------------------------
#
# First we remove channels that are too close together (short channels) to
# detect a neural response (less than 1 cm distance between optodes).
# To achieve this we pick all the channels that are not considered to be short.

picks = mne.pick_types(raw_intensity.info, meg=False, fnirs=True)
dists = mne.preprocessing.nirs.source_detector_distances(
    raw_intensity.info, picks=picks)
raw_intensity.pick(picks[dists > 0.01])
raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od)
raw_haemo = raw_haemo.filter(0.05, 0.7, h_trans_bandwidth=0.2,
                             l_trans_bandwidth=0.02)

###############################################################################
# Extract epochs
# --------------
#
# Now that the signal has been converted to relative haemoglobin concentration,
# and the unwanted heart rate component has been removed, we can extract epochs
# related to each of the experimental conditions.
#
# First we extract the events of interest and visualise them to ensure they are
# correct.

events, _ = mne.events_from_annotations(raw_haemo, event_id={'1.0': 1,
                                                             '2.0': 2,
                                                             '3.0': 3})
event_dict = {'Control': 1, 'Tapping/Left': 2, 'Tapping/Right': 3}


###############################################################################
# Next we define the range of our epochs, the rejection criteria,
# baseline correction, and extract the epochs. We visualise the log of which
# epochs were dropped.

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
# Apply anti correlation
# ----------------------------------
#
# Apply Cui et. al. 2010

raw_anti = mne_nirs.signal_enhancement.enhance_negative_correlation(raw_haemo)

epochs_anti = mne.Epochs(raw_anti, events, event_id=event_dict,
                         tmin=tmin, tmax=tmax,
                         reject=reject_criteria, reject_by_annotation=True,
                         proj=True, baseline=(None, 0), preload=True,
                         detrend=1, verbose=True)

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
# Plot figures

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