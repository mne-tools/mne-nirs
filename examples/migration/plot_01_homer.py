"""
.. _tut-migration-homer:

Migrating your analysis from Homer to MNE
=========================================

Homer2 is a MATLAB based software analysis package. https://homer-fnirs.org/

Homer3 is also a MATLAB based analysis package. https://github.com/BUNPC/Homer3

This tutorial will demonstrate how to do convert a basic Homer script to MNE
and highlight differences and common issues you may face.


Basic Homer2 script to be reproduced in MNE
-------------------------------------------

Below is an example common analysis performed in


.. code-block:: matlab

   a = some_function(test)
   b = another_function(a)
   the_output = log(b)


.. contents:: Page contents
   :local:
   :depth: 2

"""


# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import os
import mne

from mne.io import read_raw_nirx


###############################################################################
# Basic Homer2 script to be reproduced in MNE
# ------------------------------------------
#
# Below is an example common analysis performed in Homer.
#


fnirs_data_folder = mne.datasets.fnirs_motor.data_path()
fnirs_raw_dir = os.path.join(fnirs_data_folder, 'Participant-1')
raw_intensity = read_raw_nirx(fnirs_raw_dir).load_data()

