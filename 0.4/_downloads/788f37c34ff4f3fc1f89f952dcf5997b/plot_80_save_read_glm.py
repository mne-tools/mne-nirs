"""
.. _tut-glm-save:

Save and load GLM results
=========================

This is an example of how to save and load
functional near-infrared spectroscopy (fNIRS)
GLM results from analysis in MNE-NIRS.
As computation can be expensive and time consuming it can be useful
to store computed results to disk, so that you can query the results later.
For example, to remake a figure or answer a new scientific question.

For a description of the analysis in this tutorial see the
:ref:`MNE-NIRS individual GLM tutorial <tut-fnirs-hrf>` and
:ref:`MNE-NIRS group GLM tutorial <tut-fnirs-group>`.
This tutorial will simply focus on saving and loading data.

The data used in this example is available
`at this location <https://github.com/rob-luke/BIDS-NIRS-Tapping>`_.
It is a finger tapping example and is briefly described below.
The dataset contains 5 participants.
The example dataset is in
`BIDS <https://bids.neuroimaging.io>`_
format and therefore already contains
information about triggers, condition names, etc.

.. note::

   The BIDS specification for NIRS data is still under development. See:
   `fNIRS BIDS proposal <https://github.com/bids-standard/bids-specification/pull/802>`_.
   As such, to run this tutorial you must use the MNE-BIDS 0.10 or later.

.. contents:: Page contents
   :local:
   :depth: 2
"""
# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

# Import common libraries
from os.path import join
import pandas as pd

# Import MNE functions
from mne.preprocessing.nirs import optical_density, beer_lambert_law

# Import MNE-NIRS functions
from mne_nirs.statistics import run_glm
from mne_nirs.experimental_design import make_first_level_design_matrix
from mne_nirs.statistics import read_glm
from mne_nirs.datasets import fnirs_motor_group

# Import MNE-BIDS processing
from mne_bids import BIDSPath, read_raw_bids, get_entity_vals


# %%
# Set up directories
# ------------------
#
# First we will define where the raw data is stored. We will analyse a
# BIDS dataset, note that the BIDS specification for NIRS data is still
# under development and you will need to install the development branch
# as described above.
#
# We first define the root directory of our dataset.

root = fnirs_motor_group.data_path()
print(root)


# %%
# And as we are using MNE-BIDS we can create a BIDSPath.
# This helps to handle all the path wrangling.

dataset = BIDSPath(root=root, task="tapping")
print(dataset.directory)


# %%
# For example we can automatically query the subjects, tasks, and sessions.

subjects = get_entity_vals(root, 'subject')
print(subjects)


# %%
# But for this example we will only process the first two subjects
subjects = subjects[:2]


# %%
# Define individual analysis
# --------------------------
#
# First we define the analysis that will be applied to each file.
# This is a GLM analysis as described in the
# :ref:`individual GLM tutorial <tut-fnirs-hrf>`,
# so this example will skim over the individual level details.


def individual_analysis(bids_path):

    raw_intensity = read_raw_bids(bids_path=bids_path, verbose=False)
    # Delete annotation labeled 15, as these just signify the start and end of experiment.
    raw_intensity.annotations.delete(raw_intensity.annotations.description == '15.0')
    raw_intensity.pick(picks=range(20)).crop(200).resample(0.3)  # Reduce load
    raw_haemo = beer_lambert_law(optical_density(raw_intensity), ppf=0.1)
    design_matrix = make_first_level_design_matrix(raw_haemo)
    glm_est = run_glm(raw_haemo, design_matrix)

    return glm_est


# %%
# Run analysis and save to disk
# -----------------------------
#
# Next we loop through the five measurements and run the individual analysis
# on each. We will then save the GLM results to disk as ``.h5`` files.


for sub in subjects:

    # Create path to file based on experiment info
    data_path = dataset.update(subject=sub,
                               datatype="nirs",
                               suffix="nirs",
                               extension=".snirf")

    # Analyse data and glm results
    glm = individual_analysis(data_path)

    # Next we create a location to store the results.
    # In BIDS fashion we will store this in a subdirectory called derivatives.
    # And we can use the BIDSPath type from above to handle the path details.

    save_path = dataset.copy().update(
        root=join(root, "derivatives"),
        datatype="nirs", suffix="glm", extension=".h5",check=False)
    # Ensure the folder exists, and make it if not.
    save_path.fpath.parent.mkdir(exist_ok=True, parents=True)

    # Finally we save the results to disk as a hdf5 file
    glm.save(save_path.fpath, overwrite=True)


# %%
# Reload results and extract summary statistics
# ---------------------------------------------
#
# Next we loop through the five measurements and reload all the results.

# Create a dataframe to store results
df = pd.DataFrame()

for sub in subjects:

    # Point to the correct subject
    save_path = save_path.update(subject=sub)

    # Read the data
    results = read_glm(save_path)

    # Extract results from data as dataframe
    individual_results = results.to_dataframe()

    # Indicate the subject ID
    individual_results["ID"] = sub

    # Append individual results to larger dataframe
    df = pd.concat([df, individual_results], ignore_index=True)


# %%
# View the resulting dataframe
# ----------------------------
#
# Finally we can view the resulting dataframe which contains data from all
# subjects.

df
