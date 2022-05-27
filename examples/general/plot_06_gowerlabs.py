# -*- coding: utf-8 -*-
r"""
.. _tut-gowerlabs-data:

==================
Import HD-DOT data
==================

High Density Diffuse Optical Tomography (HD-DOT) results in a greater
number of channels than traditional fNIRS devices. HD-DOT data is often
collected with individual registration of the sensor positions. In this
tutorial we demonstrate how to load HD-DOT data from a Gowerlabs Lumo device,
co-register the channels to a head, and visualise the resulting channel space.

This tutorial uses the 3d graphical functionality provided by MNE-Python,
to ensure you have all the required packages installed we recommend using the
`official MNE installers. <https://mne.tools/stable/install/index.html>`__
"""

# %%

import os.path as op
import numpy as np
import pandas as pd
import mne
from mne.datasets.testing import data_path

# sphinx_gallery_thumbnail_number = 1

# %%
# Import Gowerlabs Example File
# -----------------------------
# Import Gowerlabs example file

testing_path = data_path(download=True)
fname = op.join(testing_path, 'SNIRF', 'GowerLabs', 'lumomat-1-1-0.snirf')
fname


# %%
# Next, we will load the example CSV file.

raw = mne.io.read_raw_snirf(fname, preload=True)
raw
