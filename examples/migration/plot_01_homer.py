"""
.. _tut-migration-homer:

Migrating from Homer to MNE
===========================

This tutorial will demonstrate how to do convert a basic Homer script to MNE
and highlight differences and common issues you may encounter.

Homer2 is a MATLAB based software analysis package. https://homer-fnirs.org/

Homer3 is also a MATLAB based analysis package. https://github.com/BUNPC/Homer3


Basic Homer2 script to be reproduced in MNE
===========================================

Below is a common example analysis performed in Homer.
The NIRx data is converted to .nirs format.
Then the intensity signal is convert to optical density,
motion corrected using TDDR, and converted to haemoglobin concentration.


.. code-block:: matlab

   HomerOfflineConverter('~/mne_data/MNE-fNIRS-motor-data/Participant-1');
   load('file.nirs', '-mat');
   fs = 7.8125;
   dRange = [0.07 3];
   SNRthresh = 7;
   SDrange = [0 45];
   reset = 0;
   tIncMan = ones(size(s,1),1);
   dod = hmrIntensity2OD(d);
   SD = enPruneChannels(d,SD,tIncMan,dRange,SNRthresh,SDrange,reset);
   tddr = hmrMotionCorrectTDDR(dod,SD,fs);
   ppf = [6 6];
   dc = hmrOD2Conc(tddr,SD,ppf);

.. contents:: Page contents
   :local:
   :depth: 2

"""

# %%
# MNE equivalent of Homer script
# ==============================
#
# First the necessary libraries and functions are imported.

# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import os
import mne

from mne.io import read_raw_nirx
from mne.preprocessing.nirs import (optical_density, beer_lambert_law,
                                    temporal_derivative_distribution_repair)


# %%
# Convert to optical density and motion correct
# ---------------------------------------------
#
# First we load the data which simply involves pointing the load function
# to the correct directory.

# First we obtain the path to the data
fnirs_data_folder = mne.datasets.fnirs_motor.data_path()
fnirs_raw_dir = os.path.join(fnirs_data_folder, 'Participant-1')

# Next we read the data
raw_intensity = read_raw_nirx(fnirs_raw_dir).load_data()


# %%
# Convert signal to optical density and apply TDDR
# ------------------------------------------------
#
# As with Homer we can convert the intensity data to optical density and
# apply motion correction using the TDDR method.

raw_od = optical_density(raw_intensity)
corrected_tddr = temporal_derivative_distribution_repair(raw_od)


# %%
# Convert to haemoglobin concentration
# ------------------------------------
#
# Next we convert the signal to changes in haemoglobin concentration.
# MNE uses a different default value for the partial pathlength factor (ppf),
# Homer uses a default value of ppf=6, whereas MNE uses ppf=0.1,
# To exactly match the results from Homer we can manually set the ppf value to
# 6 in MNE.

raw_h = beer_lambert_law(corrected_tddr, ppf=6.)


# %%
# Further analysis details
# ------------------------------------
#
# Commonly this preprocessing is followed by an averaging analysis as described
# in the :ref:`MNE fNIRS tutorial <mne:tut-fnirs-processing>`.
# If there is useful processing in the Homer
# that is not available in MNE
# please let us know by creating an issue at
# https://github.com/mne-tools/mne-nirs/issues
