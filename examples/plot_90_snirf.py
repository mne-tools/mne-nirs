"""
.. _tut-fnirs-snirf:

SNIRF Handling With MNE
=======================

.. sidebar:: .nirs files

   If you wish to process your .nirs files in MNE use the official snirf
   converter to create .snirf file.
   See https://github.com/fNIRS/snirf_homer3

SNIRF is a file format for storing NIRS data. The protocol is produced
by the society for functional near infrared spectroscopy. In this tutorial
we demonstrate how to convert your MNE data to SNIRF and also how to read
SNIRF files.

Read the SNIRF protocol over at https://github.com/fNIRS/snirf/blob/master/snirf_specification.md

.. contents:: Page contents
   :local:
   :depth: 2

"""


# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import os
import mne
import mne_nirs
from numpy.testing import assert_allclose



###############################################################################
# Import raw NIRS data from vendor
# --------------------------------
#
# First we import the motor tapping data, these data are also
# described and used in the
# `MNE fNIRS tutorial <https://mne.tools/stable/auto_tutorials/preprocessing/plot_70_fnirs_processing.html>`_.


fnirs_data_folder = mne.datasets.fnirs_motor.data_path()
fnirs_raw_dir = os.path.join(fnirs_data_folder, 'Participant-1')
raw_intensity = mne.io.read_raw_nirx(fnirs_raw_dir).load_data()


###############################################################################
# Write data as SNIRF
# -------------------
#
# Now we can write this data back to disk in the SNIRF format.

mne_nirs.io.snirf.write_raw_snirf(raw_intensity, 'test_raw.snirf')


###############################################################################
# Read back SNIRF file
# --------------------
# 
# Next we can read back the snirf file.

snirf_intensity = mne.io.read_raw_snirf('test_raw.snirf')


###############################################################################
# Compare files
# -------------
# 
# Finally we can compare the data of the original to the SNIRF format and
# ensure that the values are the same.

assert_allclose(raw_intensity.get_data(), snirf_intensity.get_data())
