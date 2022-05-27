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

# %%
# By looking at the traces above we see that there are no flat channels
# and the signal includes event annotations. Next, we can view the
# annotations to ensure that they match what we expect from our experiment.
# Annotations provide a flexible tool to represent events in your
# experiment. They can also be used to annotate other useful information
# such as bad segments of data, participant movements, etc.

raw.annotations


# %%
# And we observe that there were six `A` annotations, one `Cat` annotation,
# and two `Dog` annotations. We can view the specific data for each annotation
# by converting the annotations to a dataframe.

raw.annotations.to_data_frame()


# %%
# View Optode Positions in 3D Space
# ---------------------------------
# The position of optodes in 3D space is recorded and stored in the SNIRF file.
# These positions are stored in head coordinate frame,
# for a detailed overview of coordinate frames and how they are handled in MNE
# see :ref:`mne:tut-source-alignment`.
# The position of each optode is stored, along with scalp landmarks (“fiducials”).
# These positions are in an arbitrary space, and must be aligned to a scan of
# the participants, or a generic, head.
#
# For this data, we do not have a MRI scan of the participants head.
# Instead, we will align the positions to a generic head created from
# a collection of 40 MRI scans of real brains called
# `fsaverage. <https://mne.tools/stable/auto_tutorials/forward/10_background_freesurfer.html#fsaverage>`__.
#
# First, lets just look at the sensors in arbitrary space.

subjects_dir = op.join(mne.datasets.sample.data_path(), 'subjects')
mne.datasets.fetch_fsaverage(subjects_dir=subjects_dir)

brain = mne.viz.Brain('fsaverage', subjects_dir=subjects_dir, alpha=0.0, cortex='low_contrast', background="w")
brain.add_sensors(raw.info, trans='fsaverage', fnirs=["sources", "detectors"])
brain.show_view(azimuth=130, elevation=80, distance=700)
