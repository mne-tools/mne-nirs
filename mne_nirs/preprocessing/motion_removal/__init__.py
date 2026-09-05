# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from ._motion_artifact import (
    detect_motion_artifacts,
    detect_motion_artifacts_by_channel,
    motion_artifact,
    motion_artifact_by_channel,
)
from ._spline import motion_correct_spline, spline
from ._wavelet import (
    auto_wavelet,
    motion_correct_wavelet,
    motion_detect_and_correct_wavelet,
    wavelet,
)
