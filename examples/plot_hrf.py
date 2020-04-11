"""
.. _tut-fnirs-processing:

Plot experiment expected haemodynamic response
==============================================



.. contents:: Page contents
   :local:
   :depth: 2

"""
# sphinx_gallery_thumbnail_number = 1

import os
import matplotlib.pyplot as plt
from itertools import compress

import mne


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
raw_intensity.plot(n_channels=len(raw_intensity.ch_names),
                   duration=500, show_scrollbars=False)


###############################################################################
# Converting from raw intensity to optical density
# ------------------------------------------------
#
# The raw intensity values are then converted to optical density.

raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
raw_od.plot(n_channels=len(raw_od.ch_names),
            duration=500, show_scrollbars=False)


###############################################################################
# Evaluating the quality of the data
# ----------------------------------
#
# At this stage we can quantify the quality of the coupling
# between the scalp and the optodes using the scalp coupling index. This
# method looks for the presence of a prominent synchronous signal in the
# frequency range of cardiac signals across both photodetected signals.
#
# In this example the data is clean and the coupling is good for all
# channels, so we will not mark any channels as bad based on the scalp
# coupling index.

sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)
fig, ax = plt.subplots()
ax.hist(sci)
ax.set(xlabel='Scalp Coupling Index', ylabel='Count', xlim=[0, 1])


###############################################################################
# In this example we will mark all channels with a SCI less than 0.5 as bad
# (this dataset is quite clean, so no channels are marked as bad).

raw_od.info['bads'] = list(compress(raw_od.ch_names, sci < 0.5))



###############################################################################
# Converting from optical density to haemoglobin
# ----------------------------------------------
#
# Next we convert the optical density data to haemoglobin concentration using
# the modified Beer-Lambert law.

raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od)
raw_haemo.plot(n_channels=len(raw_haemo.ch_names),
               duration=500, show_scrollbars=False)
