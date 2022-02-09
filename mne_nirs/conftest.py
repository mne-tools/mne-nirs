# -*- coding: utf-8 -*-
# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause

import warnings

import pytest
import mne
from mne.datasets import testing


# most of this adapted from MNE-Python

def pytest_configure(config):
    """Configure pytest options."""
    # Markers
    for marker in ('examples',):
        config.addinivalue_line('markers', marker)
    for fixture in ('matplotlib_config', 'close_all'):
        config.addinivalue_line('usefixtures', fixture)

    warning_lines = r"""
    error::
    ignore:.*np\.MachAr.*:DeprecationWarning
    ignore:.*sysconfig module is deprecated.*:DeprecationWarning
    ignore:.*nilearn.glm module is experimental.*:
    ignore:.*Using or importing the ABCs from.*:
    ignore:.*distutils Version classes are deprecated.*:
    ignore:.*LUTSIZE was deprecated in Matplotlib.*:
    ignore:.*pandas\.Int64Index is deprecated and will be removed.*:
    ignore:.*Setting non-standard config type.*:
    ignore:.*The MLE may be on the boundary.*:
    ignore:.*The Hessian matrix at the estimated parameter values.*:
    always:`np\..*is a deprecated alias for the builtin.*:DeprecationWarning
    always::ResourceWarning
    """  # noqa: E501
    for warning_line in warning_lines.split('\n'):
        warning_line = warning_line.strip()
        if warning_line and not warning_line.startswith('#'):
            config.addinivalue_line('filterwarnings', warning_line)


@pytest.fixture(scope='session')
def matplotlib_config():
    """Configure matplotlib for viz tests."""
    from matplotlib import cbook, use
    want = 'agg'  # don't pop up windows
    with warnings.catch_warnings(record=True):  # ignore warning
        warnings.filterwarnings('ignore')
        use(want, force=True)
    import matplotlib.pyplot as plt
    assert plt.get_backend() == want
    # overwrite some params that can horribly slow down tests that
    # users might have changed locally (but should not otherwise affect
    # functionality)
    plt.ioff()
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['figure.max_open_warning'] = 100

    # Make sure that we always reraise exceptions in handlers
    orig = cbook.CallbackRegistry

    class CallbackRegistryReraise(orig):
        def __init__(self, exception_handler=None):
            super(CallbackRegistryReraise, self).__init__(exception_handler)

    cbook.CallbackRegistry = CallbackRegistryReraise


@pytest.fixture(autouse=True)
def close_all():
    """Close all matplotlib plots, regardless of test status."""
    # This adds < 1 µS in local testing, and we have ~2500 tests, so ~2 ms max
    import matplotlib.pyplot as plt
    yield
    plt.close('all')


@pytest.fixture
@testing.requires_testing_data
def requires_pyvista():
    pyvista = pytest.importorskip('pyvista')
    pytest.importorskip('pyvistaqt')
    mne.viz.set_3d_backend('pyvista')
    yield
    pyvista.close_all()
