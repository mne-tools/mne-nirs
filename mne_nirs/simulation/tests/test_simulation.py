# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import os
import mne
import mne_nirs
import numpy as np
import pytest


def test_simulate_block_design():
    raw = mne_nirs.simulation.simulate_block_design()
