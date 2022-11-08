# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

# This script runs each of the example scripts. It acts as a system test.

import os
import pytest
import sys

from mne.utils import check_version


def examples_path():

    if not os.path.isdir("BIDS-NIRS-Tapping"):
        os.system("git clone --depth 1 "
                  "https://github.com/rob-luke/BIDS-NIRS-Tapping.git")

    if os.path.isdir("examples"):
        path = "examples/general/"
    else:
        path = "../examples/general/"

    return path


def run_script_and_check(test_file_path):
    with open(test_file_path) as fid:
        return exec(fid.read(), locals(), locals())


requires_mne_1p2 = pytest.mark.skipif(
    not check_version('mne', '1.2'), reason='Needs MNE-Python 1.2')
# https://github.com/mne-tools/mne-bids/pull/406
try:
    from mne_bids.config import EPHY_ALLOWED_DATATYPES
except Exception:
    missing_mne_bids_fnirs = 'Could not import EPHY_ALLOWED_DATATYPES'
else:
    if 'nirs' in EPHY_ALLOWED_DATATYPES:
        missing_mne_bids_fnirs = None
    else:
        missing_mne_bids_fnirs = '"nirs" not in EPHY_ALLOWED_DATATYPES'
requires_mne_bids_nirs = pytest.mark.skipif(
    missing_mne_bids_fnirs is not None,
    reason=f'Incorrect MNE-BIDS version: {missing_mne_bids_fnirs}',
)


@pytest.mark.filterwarnings('ignore:No bad channels to interpolate.*:')
@pytest.mark.skipif(
    sys.platform.startswith('win'), reason='Unstable on Windows')
@pytest.mark.examples
@pytest.mark.parametrize('fname', ([
    "plot_01_data_io.py",
    pytest.param("plot_05_datasets.py", marks=requires_mne_bids_nirs),
    "plot_10_hrf_simulation.py",
    pytest.param("plot_11_hrf_measured.py", marks=requires_mne_1p2),
    pytest.param("plot_12_group_glm.py", marks=requires_mne_1p2),
    pytest.param("plot_13_fir_glm.py", marks=requires_mne_bids_nirs),
    pytest.param("plot_14_glm_components.py", marks=requires_mne_1p2),
    "plot_15_waveform.py",
    "plot_16_waveform_group.py",
    pytest.param("plot_19_snirf.py", marks=requires_mne_bids_nirs),
    "plot_20_enhance.py",
    "plot_21_artifacts.py",
    "plot_22_quality.py",
    "plot_30_frequency.py",
    "plot_40_mayer.py",
    pytest.param("plot_80_save_read_glm.py", marks=requires_mne_bids_nirs),
    "plot_99_bad.py"]))
def test_examples(fname, requires_pyvista):
    test_file_path = examples_path() + fname
    run_script_and_check(test_file_path)
