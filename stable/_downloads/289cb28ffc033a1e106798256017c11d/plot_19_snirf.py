"""
.. _tut-fnirs-snirf:

SNIRF Support in MNE
====================

.. sidebar:: .nirs files

   MNE does not support reading ``.nirs`` files. If you have ``.nirs`` files that
   you would like to process in MNE, you should first convert them to SNIRF.
   To convert ``.nirs`` files to SNIRF you can use the Homer3 ``Nirs2Snirf``
   function. See https://github.com/fNIRS/snirf_homer3

SNIRF is a file format for storing functional near-infrared spectroscopy (fNIRS)
data. The specification is maintained by the society for functional near infrared
spectroscopy. In this tutorial we demonstrate how to convert your MNE data to
the SNIRF  and also how to read SNIRF files. We also demonstrate how to validate
that a SNIRF file conforms to the SNIRF specification.

You can read the SNIRF protocol at the official site https://github.com/fNIRS/snirf.

.. contents:: Page contents
   :local:
   :depth: 2

"""


# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import os
import mne
import snirf

from mne.io import read_raw_nirx, read_raw_snirf
from mne_nirs.io import write_raw_snirf
from numpy.testing import assert_allclose


# %%
# Import raw NIRS data from vendor
# --------------------------------
#
# First we import some example data recorded with a NIRX device.


fnirs_data_folder = mne.datasets.fnirs_motor.data_path()
fnirs_raw_dir = os.path.join(fnirs_data_folder, 'Participant-1')
raw_intensity = read_raw_nirx(fnirs_raw_dir).load_data()


# %%
# Write data as SNIRF
# -------------------
#
# Now we can write this data back to disk in the SNIRF format.

write_raw_snirf(raw_intensity, 'test_raw.snirf')


# %%
# Read back SNIRF file
# --------------------
# 
# Next we can read back the snirf file.

snirf_intensity = read_raw_snirf('test_raw.snirf')


# %%
# Compare files
# -------------
# 
# Finally we can compare the data of the original to the SNIRF format and
# ensure that the values are the same.

assert_allclose(raw_intensity.get_data(), snirf_intensity.get_data())

snirf_intensity.plot(n_channels=30, duration=300, show_scrollbars=False)


# %%
# Validate SNIRF File
# -------------------
#
# To validate that a file complies with the SNIRF standard you should use the
# official SNIRF validator from the Boston University Neurophotonics Center
# called ``snirf``. Detailed instructions for this program can be found at
# https://github.com/BUNPC/pysnirf2. Below we demonstrate that the files created
# by MNE-NIRS are compliant with the specification.

result = snirf.validateSnirf('test_raw.snirf')
assert result.is_valid()
result.display()
