# Authors: Eric Larson <larson.eric.d@gmail.com>
# License: BSD (3-clause)
#
# Compat shims for different dependency versions.

from pathlib import Path
import mne
from mne.fixes import _compare_version

# Deal with fetch_data str->Path deprecation
# This can go away once we only support 1.1, along with the
# conftest.py ignore of "data_path functions now return"
try:
    from mne.datasets.utils import _mne_path  # noqa
except Exception:
    if _compare_version(mne.__version__, '>=', '1.0'):
        _mne_path = Path
    else:
        _mne_path = str  # old MNE
