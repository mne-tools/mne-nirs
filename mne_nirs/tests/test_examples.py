# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

# This script runs each of the example scripts. It acts as a system test.

import os
import pytest

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


requires_lets_plot = pytest.mark.skipif(
    not check_version('lets_plot'), reason='Requires lets-plot')


@pytest.mark.examples
@pytest.mark.parametrize('fname', ([
    "plot_01_data_io.py",
    "plot_05_datasets.py",
    "plot_10_hrf_simulation.py",
    "plot_11_hrf_measured.py",
    pytest.param("plot_12_group_glm.py", marks=requires_lets_plot),
    "plot_13_fir_glm.py",
    "plot_14_glm_components.py",
    "plot_15_waveform.py",
    pytest.param("plot_16_waveform_group.py", marks=requires_lets_plot),
    "plot_19_snirf.py",
    "plot_20_enhance.py",
    "plot_21_artifacts.py",
    "plot_22_quality.py",
    "plot_30_frequency.py",
    "plot_40_mayer.py",
    "plot_80_save_read_glm.py",
    "plot_99_bad.py"]))
def test_examples(fname):
    test_file_path = examples_path() + fname
    run_script_and_check(test_file_path)
