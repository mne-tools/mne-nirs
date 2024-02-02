# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause

import os
import warnings
from unittest import mock

import mne
import pytest
from mne.datasets import testing

# most of this adapted from MNE-Python


def pytest_configure(config):
    """Configure pytest options."""
    # Markers
    for marker in ("examples",):
        config.addinivalue_line("markers", marker)
    for fixture in ("matplotlib_config", "close_all"):
        config.addinivalue_line("usefixtures", fixture)

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
    ignore:.*data_path functions now return.*
    ignore:.*default value of `n_init`*
    ignore:.*get_cmap function will be deprecated`*
    ignore:.*The register_cmap function`*
    ignore:.*is not yet supported for.*in qdarkstyle.*:RuntimeWarning
    always::ResourceWarning
    ignore:_SixMetaPathImporter.find_spec\(\) not found.*:ImportWarning
    # Should probably fix these at some point...
    ignore:unclosed file.*:ResourceWarning
    # seaborn
    ignore:The register_cmap function.*:
    ignore:The get_cmap function.*:
    ignore:The figure layout has changed.*:UserWarning
    # H5py
    ignore:`product` is deprecated as of NumPy.*:DeprecationWarning
    # seaborn
    ignore:is_categorical_dtype is deprecated.*:FutureWarning
    ignore:use_inf_as_na option is deprecated.*:FutureWarning
    # nilearn
    ignore:The provided callable <function sum.*:FutureWarning
    ignore:The parameter "contrast_type" will be removed.*:DeprecationWarning
    # TODO: in an example (should fix eventually)
    ignore:The behavior of DataFrame concatenation.*:FutureWarning
    # mne-bids needs a release
    ignore:.*mne\.io\.pick.* is deprecated.*:FutureWarning
    # MESA
    ignore:Mesa version 10\.2\.4 is too old.*:RuntimeWarning
    # Pandas
    ignore:np\.find_common_type is deprecated.*:DeprecationWarning
    """  # noqa: E501
    for warning_line in warning_lines.split("\n"):
        warning_line = warning_line.strip()
        if warning_line and not warning_line.startswith("#"):
            config.addinivalue_line("filterwarnings", warning_line)


@pytest.fixture(scope="session")
def matplotlib_config():
    """Configure matplotlib for viz tests."""
    from matplotlib import cbook, use

    want = "agg"  # don't pop up windows
    with warnings.catch_warnings(record=True):  # ignore warning
        warnings.filterwarnings("ignore")
        use(want, force=True)
    import matplotlib.pyplot as plt

    assert plt.get_backend() == want
    # overwrite some params that can horribly slow down tests that
    # users might have changed locally (but should not otherwise affect
    # functionality)
    plt.ioff()
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["figure.max_open_warning"] = 100

    # Make sure that we always reraise exceptions in handlers
    orig = cbook.CallbackRegistry

    class CallbackRegistryReraise(orig):
        def __init__(self, exception_handler=None, signals=None):
            super().__init__(exception_handler)

    cbook.CallbackRegistry = CallbackRegistryReraise


@pytest.fixture(autouse=True)
def close_all():
    """Close all matplotlib plots, regardless of test status."""
    # This adds < 1 µS in local testing, and we have ~2500 tests, so ~2 ms max
    import matplotlib.pyplot as plt

    yield
    plt.close("all")


# We can't use monkeypatch because its scope (function-level) conflicts with
# the requests fixture (module-level), so we live with a module-scoped version
# that uses mock
@pytest.fixture(scope="module")
def options_3d():
    """Disable advanced 3d rendering."""
    with mock.patch.dict(
        os.environ,
        {
            "MNE_3D_OPTION_ANTIALIAS": "false",
            "MNE_3D_OPTION_DEPTH_PEELING": "false",
            "MNE_3D_OPTION_SMOOTH_SHADING": "false",
        },
    ):
        yield


@pytest.fixture
@testing.requires_testing_data
def requires_pyvista(options_3d):
    """Require pyvista."""
    pyvista = pytest.importorskip("pyvista")
    pytest.importorskip("pyvistaqt")
    pyvista.close_all()
    try:
        from pyvista.plotting.plotter import _ALL_PLOTTERS
    except Exception:  # PV < 0.40
        from pyvista.plotting.plotting import _ALL_PLOTTERS
    assert len(_ALL_PLOTTERS) == 0
    mne.viz.set_3d_backend("pyvista")
    yield pyvista
    pyvista.close_all()
    assert len(_ALL_PLOTTERS) == 0
