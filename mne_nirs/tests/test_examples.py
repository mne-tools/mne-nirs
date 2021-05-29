# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

# This script runs each of the example scripts. It acts as a system test.

import os
import pytest


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
    import matplotlib as mpl
    mpl.use("Agg")  # Useful when testing locally as not to block progress
    return exec(open(test_file_path).read(), locals(), locals())


@pytest.mark.examples
@pytest.mark.parametrize('fname', (["plot_10_hrf_simulation.py",
                                    "plot_11_hrf_measured.py",
                                    "plot_12_group_glm.py",
                                    "plot_16_waveform_group.py",
                                    "plot_19_snirf.py",
                                    "plot_20_enhance.py",
                                    "plot_30_frequency.py",
                                    "plot_99_bad.py"]))
def test_examples(fname):
    test_file_path = examples_path() + fname
    run_script_and_check(test_file_path)
