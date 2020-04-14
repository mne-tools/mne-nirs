"""
.. _tut-fnirs-processing:

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
raw_intensity = raw_intensity.pick(picks=[0])


###############################################################################
# Display boxcar of experiment
# ----------------------------
#
# Add some text here.

s = mne_nirs.experimental_design.create_boxcar(raw_intensity)
plt.plot(s)
plt.xlim(0, 3000)


###############################################################################
# Examine the expected haemodynamic response
# ------------------------------------------
#
# Add some text here.

s = mne_nirs.experimental_design.create_hrf(raw_intensity)
plt.plot(s)
plt.xlim(0, 3000)
plt.show()