# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

# This script runs each of the example scripts. It acts as a system test.

import os
import pytest


def examples_path():
    # test_file_path = os.path.dirname(os.path.abspath(__file__))
    print(os.system("pwd"))
    print(os.system("ls"))
    print(os.system("ls .."))
    print(os.system("ls ../examples"))
    # print(os.system("ls ../.."))
    # print(os.system("ls ../../..?"))
    # return test_file_path + "/../../examples/"
    return "../examples/"


def run_script_and_check(test_file_path):
    return os.popen(f"python3 {test_file_path} && echo 'success'").read()


@pytest.mark.parametrize('fname', (["plot_10_hrf_simulation.py",
                                    "plot_11_hrf_measured.py",
                                    "plot_12_group_glm.py",
                                    "plot_19_snirf.py",
                                    "plot_20_enhance.py",
                                    "plot_30_frequency.py",
                                    "plot_99_bad.py"]))
def test_hrf_simulation(fname):
    test_file_path = examples_path() + fname
    assert "success" in run_script_and_check(test_file_path)
