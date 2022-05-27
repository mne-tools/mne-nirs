# -*- coding: utf-8 -*-
r"""
.. _tut-gowerlabs-data:

===================
Read Gowerlabs data
===================

High Density Diffuse Optical Tomography (HD-DOT) results in a greater
number of channels than traditional fNIRS devices.

`Gowerlabs. <https://www.gowerlabs.co.uk>`__
produces the `Lumo. <https://www.gowerlabs.co.uk/lumo>`__,
a HD-DOT device.
This tutorial demonstrates how to load data from Gowerlabs devices,
including how to utilise 3D digitisation information collected with
the HD-DOT measurement.

Data should be collected using the guidelines provided by Gowerlabs.
Once collected, the data should be converted to the SNIRF format using
`the lumomat software. <https://github.com/Gowerlabs/lumomat>`__.


HD-DOT data is often
collected with individual registration of the sensor positions. In this
tutorial we demonstrate how to load HD-DOT data from a Gowerlabs Lumo device,
co-register the channels to a head, and visualise the resulting channel space.

This tutorial uses the 3d graphical functionality provided by MNE-Python,
to ensure you have all the required packages installed we recommend using the
`official MNE installers. <https://mne.tools/stable/install/index.html>`__
"""

# %%

import os.path as op
import mne
from mne.datasets.testing import data_path

# sphinx_gallery_thumbnail_number = 1

# %%
# Import Gowerlabs Example File
# -----------------------------
# First, we must instruct the software where the file we wish to import
# resides. In this example we will use a small test file that is
# included in the MNE testing data set. To load your own data, replace
# the path stored in the `fname` variable by
# running `fname = /path/to/data.snirf`.
#
# .. note:: This will be updated to demonstrate how to load data with a
#           greater number of tiles and more meaningful measurement.
#           In the meantime, a small sample file is used.

testing_path = data_path(download=True)
fname = op.join(testing_path, 'SNIRF', 'GowerLabs', 'lumomat-1-1-0.snirf')

# %%
# We can view the path to the data by calling the variable `fname`.

fname


# %%
# To load the file we call the function :func:`mne:mne.io.read_raw_snirf`.

raw = mne.io.read_raw_snirf(fname, preload=True)

# %%
# We can then look at the file metadata by calling the variable `raw`.

raw

# %%
# Visualise Data
# --------------
# Next, we visualise the raw data to visually inspect the data quality.

raw.plot()
